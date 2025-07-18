#!/usr/bin/env python
# coding: utf-8
import argparse
import os
from operator import itemgetter

import firedrake
import icepack
from firedrake.__future__ import interpolate
from firedrake.petsc import PETSc
from icepack.constants import ice_density as ρ_I, water_density as ρ_W, strain_rate_min
from icepack.calculus import sym_grad, trace

from icepackaccs.friction import get_ramp_weertman, get_regularized_coulomb
from icepackaccs.mismip import mismip_bed_topography
from icepackaccs import rate_factor
from libmismip import run_save_simulation, mismip_melt, par_opts
from initialize_mismip import u0_coulomb, h_t, A_scale


# Without these we get a mysterious error on new firedrake installs
mesh1d_dum = firedrake.IntervalMesh(100, 10)
mesh_dum = firedrake.ExtrudedMesh(mesh1d_dum, layers=1)


timesteps_per_year = 8
dt = 1.0 / timesteps_per_year
save_interval = 10


def effective_strain_rate(u, ε_min=strain_rate_min):
    ε = sym_grad(u)
    return firedrake.sqrt((firedrake.inner(ε, ε) + trace(ε) ** 2 + ε_min**2) / 2)


def tunable_depth_averaged_viscosity(**kwargs):
    r"""Return the viscous part of the action for depth-averaged models

    The viscous component of the action for depth-averaged ice flow is

    .. math::
        E(u) = \frac{n}{n+1}\int_\Omega h\cdot
        M(\dot\varepsilon, A):\dot\varepsilon\; dx

    where :math:`M(\dot\varepsilon, A)` is the membrane stress tensor

    .. math::
        M(\dot\varepsilon, A) = A^{-1/n}|\dot\varepsilon|^{1/n - 1}
        (\dot\varepsilon + \text{tr}\dot\varepsilon\cdot I).

    This form assumes that we're using the fluidity parameter instead
    the rheology parameter, the temperature, etc. To use a different
    variable, you can implement your own viscosity functional and pass it
    as an argument when initializing model objects to use your functional
    instead.

    We include regularization of Glen's law in the limit of zero strain rate
    by default. You can set the regularization to the value of your choice or
    to zero by passing it to the `strain_rate_min` argument.

    Parameters
    ----------
    velocity : firedrake.Function
    thickness : firedrake.Function
    fluidity : firedrake.Function
    strain_rate_min : firedrake.Constant

    Returns
    -------
    firedrake.Form
    """
    u, h, A, mod_A, is_floating = itemgetter("velocity", "thickness", "fluidity", "mod_A", "is_floating")(kwargs)
    ε_min = kwargs.get("strain_rate_min", firedrake.Constant(strain_rate_min))
    n = kwargs.get("flow_law_exponent", 3.0)
    ε_e = effective_strain_rate(u, ε_min=ε_min)
    return 2 * n / (n + 1) * h * (firedrake.exp(is_floating * mod_A * A_scale) * A) ** (-1 / n) * ε_e ** (1 / n + 1)


def depth_averaged_viscosity(**kwargs):
    r"""Return the viscous part of the action for depth-averaged models

    The viscous component of the action for depth-averaged ice flow is

    .. math::
        E(u) = \frac{n}{n+1}\int_\Omega h\cdot
        M(\dot\varepsilon, A):\dot\varepsilon\; dx

    where :math:`M(\dot\varepsilon, A)` is the membrane stress tensor

    .. math::
        M(\dot\varepsilon, A) = A^{-1/n}|\dot\varepsilon|^{1/n - 1}
        (\dot\varepsilon + \text{tr}\dot\varepsilon\cdot I).

    This form assumes that we're using the fluidity parameter instead
    the rheology parameter, the temperature, etc. To use a different
    variable, you can implement your own viscosity functional and pass it
    as an argument when initializing model objects to use your functional
    instead.

    We include regularization of Glen's law in the limit of zero strain rate
    by default. You can set the regularization to the value of your choice or
    to zero by passing it to the `strain_rate_min` argument.

    Parameters
    ----------
    velocity : firedrake.Function
    thickness : firedrake.Function
    fluidity : firedrake.Function
    strain_rate_min : firedrake.Constant

    Returns
    -------
    firedrake.Form
    """
    u, h, A = itemgetter("velocity", "thickness", "fluidity")(kwargs)
    ε_min = kwargs.get("strain_rate_min", firedrake.Constant(strain_rate_min))
    n = kwargs.get("flow_law_exponent", 3.0)
    ε_e = effective_strain_rate(u, ε_min=ε_min)
    return 2 * n / (n + 1) * h * A ** (-1 / n) * ε_e ** (1 / n + 1)


def get_a_unpert(*args, **kwargs):
    return firedrake.Constant(0.3)


def get_a_retreat(surface, thickness, bed, HC0=75, z0=-100, omega=0.2):
    return firedrake.Function(thickness.function_space()).interpolate(
        firedrake.Constant(0.3) - mismip_melt(surface, thickness, bed, HC0=75, z0=-100, omega=0.2)
    )


def volume(thickness):
    return firedrake.assemble(thickness * firedrake.dx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=float, nargs="+", default=[4])
    parser.add_argument("-T", type=int, nargs="+", default=None)
    parser.add_argument("-dev", type=float, nargs="+", default=None)
    parser.add_argument("-nyears", type=int, default=500)
    parser.add_argument("-sim", choices=["unperturbed", "retreat"], default="retreat")
    parser.add_argument("-standard", action="store_true")
    parser.add_argument("-true", action="store_true")
    args = parser.parse_args()

    num_years = args.nyears
    ns = args.n
    Ts = args.T

    if args.standard:
        if args.true:
            raise ValueError("Can only look at standard, or true, not multiple")
        checkpoint_fn = "inputs/ssa_standard_initialization_mismip.h5"
        init = "standard"
        if args.T is None:
            Ts = [-10]
    elif args.true:
        checkpoint_fn = "inputs/ssa_true_initialization_mismip.h5"
        init = "true"
        Ts = [-10]
        ns = [3]
    else:
        checkpoint_fn = "inputs/ssa_identical_initialization_mismip.h5"
        init = "identical"
        if args.T is None:
            Ts = [-10]
    input_dict = {T: {} for T in Ts}

    with firedrake.CheckpointFile(checkpoint_fn, "r") as chk:
        mesh = chk.load_mesh("fine_mesh2d")

    Q = firedrake.FunctionSpace(mesh, "CG", 2)
    b = firedrake.assemble(interpolate(mismip_bed_topography(mesh), Q))
    acc = args.sim

    if acc == "retreat":
        get_a = get_a_retreat
    else:
        PETSc.Sys.Print("This is not perturbed")
        get_a = get_a_unpert

    if init in ["identical", "standard", "true"]:
        output_template = "outputs/ssa_{:s}_{:s}_T{:d}_n{:2.1f}_{:s}.h5"
    else:
        output_template = "outputs/ssa_{:s}_{:s}_dev{:0.2f}_n{:2.1f}_{:s}.h5"

    if num_years > 1000:
        output_template = output_template[:-3] + "_long.h5"

    regularized_coulomb = get_regularized_coulomb(m=3, u_0=u0_coulomb, h_t=h_t)
    weertman_3 = get_ramp_weertman(m=3, h_t=h_t)
    weertman_1 = get_ramp_weertman(m=1, h_t=h_t)
    frictions = {"3": weertman_3, "1": weertman_1, "RCFi": regularized_coulomb}

    with firedrake.CheckpointFile(checkpoint_fn, "r") as chk:
        for T_np in Ts:
            for n in ns:
                input_dict[T_np][n] = {}
                if init in ["identical", "standard", "true"]:
                    inv_name = "T{:d}_n{:2.1f}".format(T_np, n)
                else:
                    inv_name = "dev{:1.2f}_n{:2.1f}".format(T_np, n)

                input_dict[T_np][n]["C3"] = firedrake.Function(Q).interpolate(chk.load_function(mesh, inv_name + "_C3"))
                if not args.true:
                    input_dict[T_np][n]["C1"] = chk.load_function(mesh, inv_name + "_C1")
                    input_dict[T_np][n]["CRCFi"] = chk.load_function(mesh, inv_name + "_CRCFi")
                input_dict[T_np][n]["A"] = chk.load_function(mesh, inv_name + "_A")
                if args.standard:
                    input_dict[T_np][n]["modA"] = chk.load_function(mesh, inv_name + "_modA")
                else:
                    input_dict[T_np][n]["modA"] = firedrake.Constant(1.0)
                input_dict[T_np][n]["u"] = chk.load_function(mesh, inv_name + "_u1")
                input_dict[T_np]["h0"] = chk.load_function(mesh, "input_thick")
                input_dict[T_np]["s0"] = chk.load_function(mesh, "input_surf")

    for fricname in ["RCFi", "3", "1"]:
        if args.true and fricname == "RCFi":
            print("True is m=3 only")
            continue
        friction = frictions[fricname]
        for T_np in Ts:
            for n in ns:
                if fricname in ["1", "3"]:
                    fricstring = "m=" + fricname
                else:
                    fricstring = fricname
                output_fn = output_template.format(acc, init, T_np, n, fricname)
                if os.path.exists(output_fn):
                    with firedrake.CheckpointFile(output_fn, "r") as chk:
                        if (
                            not chk.has_attr("metadata", "final_time")
                            or chk.get_attr("metadata", "final_time") < num_years
                        ):
                            remove = True
                        else:
                            continue
                        if remove:
                            os.remove(output_fn)

                    pass
                A = input_dict[T_np][n]["A"]
                modA = firedrake.Constant(1.0)
                if args.standard:
                    if n > 2:
                        A = rate_factor(firedrake.Constant(T_np + 273.15), n=n)
                    else:
                        A = rate_factor(firedrake.Constant(T_np + 273.15), n=n, m=1.0e-2, m_exp=1.4)
                    modA = input_dict[T_np][n]["modA"]

                if args.standard:
                    model = icepack.models.IceStream(friction=friction, viscosity=tunable_depth_averaged_viscosity)
                else:
                    model = icepack.models.IceStream(friction=friction, viscosity=depth_averaged_viscosity)
                solver = icepack.solvers.FlowSolver(model, **par_opts)

                h = input_dict[T_np]["h0"].copy(deepcopy=True)
                s = input_dict[T_np]["s0"].copy(deepcopy=True)
                u = input_dict[T_np][n]["u"].copy(deepcopy=True)

                h_af = firedrake.max_value(input_dict[T_np]["s0"] - input_dict[T_np]["h0"] * (1 - ρ_I / ρ_W), 0)
                ramp = firedrake.min_value(1, h_af / 50.0)
                is_floating0 = firedrake.Function(Q).interpolate(1.0 - ramp)

                sqrtC = input_dict[T_np][n]["C" + fricname]
                C = firedrake.Function(sqrtC.function_space()).interpolate(sqrtC**2.0)

                if init in ["identical", "standard"]:
                    PETSc.Sys.Print("T={:d}, n={:2.1f}, {:s}".format(T_np, n, fricstring))
                else:
                    PETSc.Sys.Print("dev={:0.2f}, n={:2.1f}, {:s}".format(T_np, n, fricstring))

                u = solver.diagnostic_solve(
                    velocity=u,
                    thickness=h,
                    surface=s,
                    fluidity=A,
                    mod_A=modA,
                    is_floating=is_floating0,
                    friction=C,
                    flow_law_exponent=firedrake.Constant(n),
                )

                run_save_simulation(
                    solver,
                    num_years,
                    dt,
                    output_fn,
                    save_interval=save_interval,
                    get_a=get_a,
                    thickness=h,
                    velocity=u,
                    surface=s,
                    bed=b,
                    A=A,
                    C=C,
                    mod_A=modA,
                    is_floating0=is_floating0,
                    flow_law_exponent=firedrake.Constant(n),
                    mesh=mesh,
                )


if __name__ == "__main__":
    main()

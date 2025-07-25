#!/usr/bin/env python
# coding: utf-8
import argparse
import os

import firedrake
import icepack
from firedrake.__future__ import interpolate
from firedrake.petsc import PETSc

from icepack.models import hybrid
from icepackaccs.friction import get_ramp_weertman, get_regularized_coulomb
from icepackaccs.mismip import mismip_bed_topography
from libmismip import run_save_simulation, par_opts, mismip_melt
from initialize_mismip import u0_coulomb, h_t, A_scale
from icepack.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
)
from icepackaccs import rate_factor


def tunable_viscosity(**kwargs):
    r"""Return the viscous part of the hybrid model action functional

    The viscous component of the action for the hybrid model is

    .. math::
        E(u) = \frac{n}{n + 1}\int_\Omega\int_0^1\left(
        M : \dot\varepsilon_x + \tau_z\cdot\varepsilon_z\right)h\, d\zeta\; dx

    where :math:`M(\dot\varepsilon, A)` is the membrane stress tensor and
    :math:`\tau_z` is the vertical shear stress vector.

    This form assumes that we're using the fluidity parameter instead
    the rheology parameter, the temperature, etc. To use a different
    variable, you can implement your own viscosity functional and pass it
    as an argument when initializing model objects to use your functional
    instead.

    Keyword arguments
    -----------------
    velocity : firedrake.Function
    surface : firedrake.Function
    thickness : firedrake.Function
    fluidity : firedrake.Function
        `A` in Glen's flow law
    mod_A : firedrake.Function
        Exponential modification to the fluidity
    is_floating : firedrake.Function
        Modify the fluidity here?

    Returns
    -------
    firedrake.Form
    """
    u, h, s, A, mod_A, is_floating = hybrid.itemgetter(
        "velocity", "thickness", "surface", "fluidity", "mod_A", "is_floating"
    )(kwargs)
    ε_min = kwargs.get("strain_rate_min", firedrake.Constant(hybrid.strain_rate_min))
    n = kwargs.get("flow_law_exponent", hybrid.glen_flow_law)

    ε_x = hybrid.horizontal_strain_rate(velocity=u, surface=s, thickness=h)
    ε_z = hybrid.vertical_strain_rate(velocity=u, thickness=h)
    ε_e = hybrid._effective_strain_rate(ε_x, ε_z, ε_min)
    return 2 * n / (n + 1) * h * (firedrake.exp(is_floating * mod_A * A_scale) * A) ** (-1 / n) * ε_e ** (1 / n + 1)


# Without these we get a mysterious error on new firedrake installs
mesh1d_dum = firedrake.IntervalMesh(100, 10)
mesh_dum = firedrake.ExtrudedMesh(mesh1d_dum, layers=1)


timesteps_per_year = 8
dt = 1.0 / timesteps_per_year
save_interval = 10


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
    parser.add_argument("-sim", choices=["unperturbed", "retreat"], default="retreat")
    parser.add_argument("-standard", action="store_true")
    parser.add_argument("-true", action="store_true")
    parser.add_argument("-vdegree", type=int, default=4)
    parser.add_argument("-fric", type=str, nargs="+", default=["RCFi", "3"])
    args = parser.parse_args()

    num_years = 500
    ns = args.n
    Ts = args.T

    if args.standard:
        if args.true:
            raise ValueError("Can only look at one initialization, not both")
        checkpoint_fn = "inputs/standard_initialization_mismip.h5"
        init = "standard"
        if args.T is None:
            Ts = [-10]
    elif args.true:
        checkpoint_fn = "inputs/true_initialization_mismip.h5"
        init = "true"
        Ts = [-10]
        ns = [3.0]
    else:
        checkpoint_fn = "inputs/identical_initialization_mismip.h5"
        init = "identical"
        if args.T is None:
            Ts = [-10]
    input_dict = {T: {} for T in Ts}

    with firedrake.CheckpointFile(checkpoint_fn, "r") as chk:
        mesh = chk.load_mesh("fine_mesh")

    Q = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="R", vdegree=0)
    Qdvar = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="GL", vdegree=args.vdegree)
    V = firedrake.VectorFunctionSpace(mesh, "CG", 2, dim=2, vfamily="GL", vdegree=args.vdegree)
    b = firedrake.assemble(interpolate(mismip_bed_topography(mesh), Q))
    acc = args.sim

    if acc == "retreat":
        get_a = get_a_retreat
    else:
        PETSc.Sys.Print("This is not perturbed")
        get_a = get_a_unpert

    if init in ["identical", "standard", "true"]:
        output_template = "outputs/{:s}_{:s}_T{:d}_n{:2.1f}_{:s}.h5"
    else:
        output_template = "outputs/{:s}_{:s}_dev{:0.2f}_n{:2.1f}_{:s}.h5"

    for T_np in Ts:
        if init in ["standard", "identical", "true"]:
            deg4_fn = "inputs/mismip_{:d}C_n3.h5".format(T_np)
        else:
            deg4_fn = "inputs/mismip_{:d}C_n3.h5".format(-10)
        with firedrake.CheckpointFile(deg4_fn, "r") as chk:
            fine_mesh = chk.load_mesh("fine_mesh")
            input_dict[T_np]["h0"] = firedrake.Function(Q).interpolate(chk.load_function(fine_mesh, "thickness"))
            input_dict[T_np]["s0"] = firedrake.Function(Q).interpolate(chk.load_function(fine_mesh, "surface"))

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

                input_dict[T_np][n]["C1"] = firedrake.Function(Q).interpolate(chk.load_function(mesh, inv_name + "_C1"))
                input_dict[T_np][n]["C3"] = firedrake.Function(Q).interpolate(chk.load_function(mesh, inv_name + "_C3"))
                input_dict[T_np][n]["CRCFi"] = firedrake.Function(Q).interpolate(
                    chk.load_function(mesh, inv_name + "_CRCFi")
                )
                input_dict[T_np][n]["A"] = firedrake.Function(Qdvar).interpolate(
                    chk.load_function(mesh, inv_name + "_A")
                )
                input_dict[T_np][n]["modA"] = firedrake.Function(Qdvar).interpolate(
                    chk.load_function(mesh, inv_name + "_modA")
                )
                input_dict[T_np][n]["u"] = firedrake.Function(V).interpolate(chk.load_function(mesh, inv_name + "_u1"))

    for fricname in args.fric:
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
                        A = rate_factor(firedrake.Constant(T_np + 273.15), n=n, m=5.0e-3, m_exp=-1.4)
                    modA = input_dict[T_np][n]["modA"]
                h_af = firedrake.max_value(input_dict[T_np]["s0"] - input_dict[T_np]["h0"] * (1 - ρ_I / ρ_W), 0)
                ramp = firedrake.min_value(1, h_af / 50.0)
                is_floating0 = firedrake.Function(Q).interpolate(1.0 - ramp)

                model = icepack.models.HybridModel(friction=friction, viscosity=tunable_viscosity)
                solver = icepack.solvers.FlowSolver(model, **par_opts)

                h = input_dict[T_np]["h0"].copy(deepcopy=True)
                s = input_dict[T_np]["s0"].copy(deepcopy=True)
                u = input_dict[T_np][n]["u"].copy(deepcopy=True)

                sqrtC = input_dict[T_np][n]["C" + fricname]
                C = firedrake.Function(sqrtC.function_space()).interpolate(sqrtC**2.0)

                if init in ["identical", "standard", "true"]:
                    PETSc.Sys.Print("{:s} T={:d}, n={:2.1f}, {:s}".format(acc, T_np, n, fricstring))
                else:
                    PETSc.Sys.Print("{:s} dev={:0.2f}, n={:2.1f}, {:s}".format(acc, T_np, n, fricstring))

                u = solver.diagnostic_solve(
                    velocity=u,
                    thickness=h,
                    surface=s,
                    fluidity=A,
                    friction=C,
                    flow_law_exponent=firedrake.Constant(n),
                    mod_A=modA,
                    is_floating=is_floating0,
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
                    flow_law_exponent=firedrake.Constant(n),
                    mod_A=modA,
                    is_floating=is_floating0,
                    mesh=mesh,
                )


if __name__ == "__main__":
    main()

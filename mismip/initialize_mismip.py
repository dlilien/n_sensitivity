#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from operator import itemgetter

import firedrake
from firedrake.petsc import PETSc
import icepack
import icepack.models.hybrid
from icepack.models import hybrid
from icepack.statistics import (
    StatisticsProblem,
    MaximumProbabilityEstimator,
)
from icepack.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
)

from icepackaccs import rate_factor, extract_bed
from icepackaccs.friction import get_ramp_weertman
from libmismip import par_opts

u0_coulomb = 250.0
h_t = 50.0
Area = 640e3 * 40e3

weertman_1 = get_ramp_weertman(m=1, h_t=h_t)

field_names = ["surface", "thickness", "velocity"]
min_thick = firedrake.Constant(10.0)
deg4_fn = "inputs/mismip_-10C_n3.h5"

A_scale = 1.0e2

LC_dict = {3: 1.0e6, 3.5: 1.0e6, 4: 1.0e6, 1.8: 1.0e6}
LA_dict = {3: 1.0e4, 3.5: 1.0e4, 4: 1.0e4, 1.8: 1.0e4}

# Needed to avoid assertion error bug in firedrake
mesh1d = firedrake.IntervalMesh(100, 120)
meshdum = firedrake.ExtrudedMesh(mesh1d, layers=1)


def A3_to_An(A3, u, h, s, n, Q):
    ε_e = effective_strain_rate(velocity=u, thickness=h, surface=h)
    # return firedrake.Function(u.function_space()).interpolate(A3 * ε_e ** (n - 3.0))
    An = firedrake.Function(Q).interpolate(A3 ** (n / 3.0) * ε_e ** (1.0 - n / 3.0))
    An_mean = firedrake.assemble(An * firedrake.dx) / Area
    return firedrake.Function(Q).interpolate(firedrake.conditional(h > 10.1, An, An_mean))


def effective_strain_rate(**kwargs):
    r"""Return the effective strain rate

    Keyword arguments
    -----------------
    velocity : firedrake.Function
    surface : firedrake.Function
    thickness : firedrake.Function

    Returns
    -------
    firedrake.Form
    """
    u, h, s = itemgetter("velocity", "thickness", "surface")(kwargs)
    ε_min = kwargs.get("strain_rate_min", firedrake.Constant(icepack.constants.strain_rate_min))

    ε_x = hybrid.horizontal_strain_rate(velocity=u, surface=s, thickness=h)
    ε_z = hybrid.vertical_strain_rate(velocity=u, thickness=h)
    ε_e = hybrid._effective_strain_rate(ε_x, ε_z, ε_min)
    return ε_e


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


def c3_to_c1(C3, u, minslide=0.0):
    Q2D = firedrake.FunctionSpace(u.ufl_domain()._base_mesh, "CG", 2)
    C1 = firedrake.Function(C3.function_space())
    U = firedrake.max_value(firedrake.sqrt(firedrake.dot(u, u)), minslide)
    C1_bed = firedrake.Function(Q2D).interpolate(
        firedrake.sqrt(extract_bed(C3) ** 2.0 * abs(extract_bed(U)) ** (1.0 / 3.0 - 1.0))
    )
    C1.dat.data[:] = C1_bed.dat.data[:]
    return C1


def pos_linear_weertman(**kwargs):
    C = kwargs.pop("friction")
    return weertman_1(friction=C**2.0, **kwargs)


def get_loss(u_true):
    def loss_functional(u):
        δu = u - u_true
        return 0.5 / Area * ((δu[0]) ** 2 + (δu[1]) ** 2) * firedrake.ds_t(u_true.ufl_domain())

    return loss_functional


def get_reg(LC, LA, is_floating):
    LCn = firedrake.Constant(LC)
    LAn = firedrake.Constant(LA)

    def regularization(C_and_A):
        return (
            0.5
            / Area
            * (LCn) ** 2
            * firedrake.inner(firedrake.grad(C_and_A[0]), firedrake.grad(C_and_A[0]))
            * firedrake.dx
            + 0.5
            / Area
            * (LAn) ** 2
            * firedrake.inner(
                firedrake.grad(C_and_A[1] * is_floating * A_scale), firedrake.grad(C_and_A[1] * is_floating * A_scale)
            )
            * firedrake.dx
        )

    return regularization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=float, nargs="+", default=[1.8, 3, 4])
    parser.add_argument("-T", type=int, nargs="+", default=[-10])
    parser.add_argument("-dev", type=float, nargs="+", default=[1.0])
    parser.add_argument("-n_iter", type=int, default=100)
    parser.add_argument("-n_pingpong", type=int, default=2, help="Number of times each inversion gets run")
    parser.add_argument("-vdegree", type=int, default=4)
    parser.add_argument("-float_tol", type=float, default=1.0e-3)
    parser.add_argument("-LC", type=float, default=None)
    parser.add_argument("-LA", type=float, default=None)
    parser.add_argument("-just_C", action="store_true")
    args = parser.parse_args()

    with firedrake.CheckpointFile(deg4_fn, "r") as chk:
        fine_mesh = chk.load_mesh("fine_mesh")
        u_in = chk.load_function(fine_mesh, "velocity")
        h0 = chk.load_function(fine_mesh, "thickness")
        s0 = chk.load_function(fine_mesh, "surface")
    x, y, ζ = firedrake.SpatialCoordinate(fine_mesh)

    C0 = firedrake.Function(h0.function_space()).interpolate(firedrake.Constant(1.0e-1))

    checkpoint_fn = "inputs/standard_initialization_mismip_simul.h5"

    V = firedrake.VectorFunctionSpace(fine_mesh, "CG", 2, dim=2, vfamily="GL", vdegree=args.vdegree)
    u_true = firedrake.Function(V).interpolate(u_in)

    if not os.path.exists(checkpoint_fn):
        with firedrake.CheckpointFile(checkpoint_fn, "w") as chk:
            chk.save_mesh(fine_mesh)
            chk.create_group("partial")
            chk.create_group("A0")
            chk.save_function(h0, name="input_thick")
            chk.save_function(s0, name="input_surf")
            chk.save_function(u_true, name="input_u")
            chk.save_function(C0, name="input_C")

    with firedrake.CheckpointFile(checkpoint_fn, "r") as chk:
        fine_mesh = chk.load_mesh("fine_mesh")
        h0 = chk.load_function(fine_mesh, name="input_thick")
        s0 = chk.load_function(fine_mesh, name="input_surf")
        u_true = chk.load_function(fine_mesh, name="input_u")
        C0 = chk.load_function(fine_mesh, name="input_C")

    V = firedrake.VectorFunctionSpace(fine_mesh, "CG", 2, dim=2, vfamily="GL", vdegree=args.vdegree)
    Q = firedrake.FunctionSpace(fine_mesh, "CG", 2, vfamily="R", vdegree=0)

    h_af = firedrake.max_value(s0 - h0 * (1 - ρ_I / ρ_W), 0)
    ramp = firedrake.min_value(1, h_af / 50.0)
    is_floating = firedrake.Function(Q).interpolate(1.0 - ramp)

    for dev in args.dev:
        for T_np in args.T:
            for n in args.n:
                if args.LC is not None:
                    LCap_C = args.LC
                else:
                    LCap_C = LC_dict[n]
                if args.LA is not None:
                    LCap_A = args.LA
                else:
                    LCap_A = LA_dict[n]
                PETSc.Sys.Print("LC={:e} and LA={:e}".format(LCap_C, LCap_A))

                regularization = get_reg(LCap_C, LCap_A, is_floating)
                inv_name = "T{:d}_n{:2.1f}".format(T_np, n)
                with firedrake.CheckpointFile(checkpoint_fn, "r") as chk:
                    if chk.has_attr("partial", inv_name + "_{:d}".format(args.n_pingpong - 1)):
                        continue

                PETSc.Sys.Print("Initializing T={:d}, n={:2.1f}".format(T_np, n))
                C1_preopt_var = c3_to_c1(C0, u_true, minslide=0.001)
                good_area = firedrake.conditional(C1_preopt_var < 1.0, 1.0, 0.0)
                C1_preopt = firedrake.Function(Q).interpolate(
                    firedrake.assemble(C1_preopt_var * good_area * firedrake.ds_t)
                    / firedrake.assemble(good_area * firedrake.ds_t)
                )
                T = firedrake.Constant(T_np + 273.15)

                if n > 2:
                    A0_fd = rate_factor(T, n=n)
                else:
                    A0_fd = rate_factor(T, n=n, m=5.0e-3, m_exp=-1.4)
                A0 = A0_fd.values()[0]

                PETSc.Sys.Print("A0 is {:f}".format(A0))
                A1 = firedrake.Function(Q).interpolate(firedrake.Constant(A0))

                model = icepack.models.HybridModel(friction=pos_linear_weertman, viscosity=tunable_viscosity)
                solver = icepack.solvers.FlowSolver(model, **par_opts)

                C1 = C1_preopt.copy(deepcopy=True)
                u = u_true.copy(deepcopy=True)
                A = A1.copy(deepcopy=True)
                mod_A = firedrake.Function(Q).interpolate(0.0)

                PETSc.Sys.Print("Optimizing T={:d}, n={:2.1f}".format(T_np, n))
                for inverse_iter in range(args.n_pingpong):
                    with firedrake.CheckpointFile(checkpoint_fn, "r") as chk:
                        if chk.has_attr("partial", inv_name + "_{:d}".format(inverse_iter + 1)):
                            PETSc.Sys.Print("Skipping iteration {:d}".format(inverse_iter + 1))
                            continue
                        elif chk.has_attr("partial", inv_name + "_{:d}".format(inverse_iter)):
                            PETSc.Sys.Print("Skipping iteration {:d}".format(inverse_iter + 1))
                            u = firedrake.Function(V).interpolate(
                                chk.load_function(fine_mesh, inv_name + "_u_{:d}".format(inverse_iter))
                            )
                            C1 = chk.load_function(fine_mesh, inv_name + "_C1_{:d}".format(inverse_iter))
                            A = chk.load_function(fine_mesh, inv_name + "_A_{:d}".format(inverse_iter))
                            mod_A = chk.load_function(fine_mesh, inv_name + "_modA_{:d}".format(inverse_iter))
                            continue

                        PETSc.Sys.Print("Round {:d}:".format(inverse_iter + 1))
                        C1, mod_A, A, u = inversion(
                            solver,
                            regularization,
                            C1,
                            u,
                            h0,
                            s0,
                            A1,
                            mod_A,
                            n,
                            is_floating,
                            args.n_iter,
                            loss_functional=get_loss(u_true),
                        )

                        with firedrake.CheckpointFile(checkpoint_fn, "a") as chk:
                            chk.set_attr("partial", inv_name + "_{:d}".format(inverse_iter), True)
                            chk.set_attr("A0", inv_name + "_{:d}".format(inverse_iter), A0)
                            chk.save_function(C1, name=inv_name + "_C1_{:d}".format(inverse_iter))
                            chk.save_function(u, name=inv_name + "_u_{:d}".format(inverse_iter))
                            chk.save_function(A, name=inv_name + "_A_{:d}".format(inverse_iter))
                            chk.save_function(mod_A, name=inv_name + "_modA_{:d}".format(inverse_iter))


def inversion(solver, regularization, C, u0, h0, s0, A0, mod_A, n, is_floating, n_iter, loss_functional):
    PETSc.Sys.Print("Initializing")

    u_w1 = solver.diagnostic_solve(
        velocity=u0,
        thickness=h0,
        surface=s0,
        fluidity=A0,
        mod_A=mod_A,
        is_floating=is_floating,
        friction=C,
        flow_law_exponent=firedrake.Constant(n),
    )

    PETSc.Sys.Print("Pre-opt regularization is {:f}".format(firedrake.assemble(regularization([C, mod_A]))))
    PETSc.Sys.Print("Pre-opt loss is {:f}".format(firedrake.assemble(loss_functional(u_w1))))

    def simulation(input_args):
        return solver.diagnostic_solve(
            velocity=u_w1,
            fluidity=A0,
            mod_A=input_args[1],
            is_floating=is_floating,
            friction=input_args[0],
            surface=s0,
            thickness=h0,
            flow_law_exponent=firedrake.Constant(n),
        )

    problem = StatisticsProblem(
        simulation=simulation,
        loss_functional=loss_functional,
        regularization=regularization,
        controls=[C, mod_A],
    )

    estimator = MaximumProbabilityEstimator(
        problem,
        gradient_tolerance=1e-5,
        step_tolerance=1e-5,
        max_iterations=n_iter,
    )

    PETSc.Sys.Print("Optimizing")
    C1, mod_A = estimator.solve()

    u = simulation([C1, mod_A])
    PETSc.Sys.Print("Post-opt regularization is {:f}".format(firedrake.assemble(regularization([C1, mod_A]))))
    PETSc.Sys.Print("Post-opt loss is {:f}".format(firedrake.assemble(loss_functional(u))))
    del estimator
    return (
        C1,
        mod_A,
        firedrake.Function(mod_A.function_space()).interpolate(A0 * firedrake.exp(mod_A * is_floating * A_scale)),
        u,
    )


if __name__ == "__main__":
    main()

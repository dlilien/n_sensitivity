#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from operator import itemgetter

import firedrake
import icepack
from firedrake.petsc import PETSc
from icepackaccs import rate_factor, extract_surface
from icepackaccs.friction import get_ramp_weertman
from icepack.calculus import sym_grad, trace
from icepack.statistics import (
    StatisticsProblem,
    MaximumProbabilityEstimator,
)
from icepack.constants import ice_density as ρ_I, water_density as ρ_W, strain_rate_min
from libmismip import par_opts

u0_coulomb = 250.0
h_t = 50.0
Area = 640e3 * 40e3

weertman_1 = get_ramp_weertman(m=1, h_t=h_t)
weertman_3 = get_ramp_weertman(m=3, h_t=h_t)

field_names = ["surface", "thickness", "velocity"]
min_thick = firedrake.Constant(10.0)
ssa_fn = "inputs/mismip-fine-ssa-15.0kyr.h5"
true_fn = "inputs/mismip_-10C_n3.h5"
std_checkpoint_fn = "inputs/ssa_standard_initialization_mismip_simul.h5"

A_scale = 100.0


# Needed to avoid assertion error bug in firedrake
mesh1d = firedrake.IntervalMesh(100, 120)
mesh_dum = firedrake.ExtrudedMesh(mesh1d, layers=1)

with firedrake.CheckpointFile(ssa_fn, "r") as chk:
    fine_mesh = chk.load_mesh("fine_mesh2d")

with firedrake.CheckpointFile(true_fn, "r") as chk:
    threed_mesh = chk.load_mesh("fine_mesh")
    u_in_3d = chk.load_function(threed_mesh, "velocity")
    h0_3d = chk.load_function(threed_mesh, "thickness")
    s0_3d = chk.load_function(threed_mesh, "surface")

x, ζ = firedrake.SpatialCoordinate(fine_mesh)

Q = firedrake.FunctionSpace(fine_mesh, "CG", 2)
V = firedrake.VectorFunctionSpace(fine_mesh, "CG", 2, dim=2)

h0 = firedrake.Function(Q).interpolate(extract_surface(h0_3d))
s0 = firedrake.Function(Q).interpolate(extract_surface(s0_3d))
u0 = firedrake.Function(V).interpolate(extract_surface(u_in_3d))

A0 = firedrake.Constant(rate_factor(263.15))
C0 = firedrake.Function(h0.function_space()).interpolate(firedrake.Constant(1.0e-1))
a = firedrake.Constant(0.3)


def A3_to_An(A3, u, h, s, n, Q):
    ε_e = effective_strain_rate(u)
    # return firedrake.Function(u.function_space()).interpolate(A3 * ε_e ** (n - 3.0))
    An = firedrake.Function(Q).interpolate(A3 ** (n / 3.0) * ε_e ** (1.0 - n / 3.0))
    An_mean = firedrake.assemble(An * firedrake.dx) / Area
    return firedrake.Function(Q).interpolate(firedrake.conditional(h > 10.1, An, An_mean))


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


def c1_to_c3(C1, u):
    U = firedrake.sqrt(firedrake.dot(u, u))
    C3 = firedrake.Function(C1.function_space()).interpolate(firedrake.sqrt(C1**2.0 / abs(U) ** (1.0 / 3.0 - 1.0)))
    return C3


def c3_to_beta(C3, u, u0):
    U = firedrake.sqrt(firedrake.dot(u, u))
    beta = firedrake.Function(firedrake.FunctionSpace(C3.ufl_domain(), "CG", 2)).interpolate(
        firedrake.sqrt(C3**2.0 * (U ** (1.0 / 3.0 + 1) + u0 ** (1.0 / 3.0 + 1)) ** (1.0 / (3.0 + 1.0)))
    )
    return beta


def c3_to_c1(C3, u, minslide=0.0):
    C1 = firedrake.Function(C3.function_space())
    U = firedrake.max_value(firedrake.sqrt(firedrake.dot(u, u)), minslide)
    C1_bed = firedrake.Function(Q).interpolate(firedrake.sqrt(C3**2.0 * abs(U) ** (1.0 / 3.0 - 1.0)))
    C1.dat.data[:] = C1_bed.dat.data[:]
    return C1


def pos_linear_weertman(**kwargs):
    C = kwargs.pop("friction")
    return weertman_1(friction=C**2.0, **kwargs)


def get_loss(u0):
    def loss_functional(u):
        δu = u - u0
        return 0.5 / Area * ((δu[0]) ** 2 + (δu[1]) ** 2) * firedrake.dx(fine_mesh)

    return loss_functional


def get_reg(LC, LA, is_floating, opt_A=True):
    LCn = firedrake.Constant(LC)
    LAn = firedrake.Constant(LA)
    if opt_A:

        def regularization(C_and_A):
            return 0.5 / Area * (LCn) ** 2 * firedrake.inner(
                firedrake.grad(C_and_A[0]), firedrake.grad(C_and_A[0])
            ) * firedrake.dx(fine_mesh) + 0.5 / Area * (LAn) ** 2 * firedrake.inner(
                firedrake.grad(C_and_A[1] * is_floating * A_scale), firedrake.grad(C_and_A[1] * is_floating * A_scale)
            ) * firedrake.dx(
                fine_mesh
            )

    else:

        def regularization(C):
            return (
                0.5
                / Area
                * (LCn) ** 2
                * firedrake.inner(firedrake.grad(C), firedrake.grad(C))
                * firedrake.dx(fine_mesh)
            )

    return regularization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=float, nargs="+", default=[3])
    parser.add_argument("-devs", type=float, nargs="+", default=[1.0])
    parser.add_argument("-T", type=int, nargs="+", default=[-10])
    parser.add_argument("-n_iter", type=int, default=100)
    parser.add_argument("-n_pingpong", type=int, default=2, help="Number of times each inversion gets run")
    parser.add_argument("-float_tol", type=float, default=1.0e-3)
    parser.add_argument("-LC", type=float, default=1.0e5)
    parser.add_argument("-LA", type=float, default=1.0e5)
    parser.add_argument("-identical", action="store_true")
    parser.add_argument("-true", action="store_true")
    parser.add_argument("-just_C", action="store_true")
    args = parser.parse_args()

    if args.identical:
        checkpoint_fn = "inputs/ssa_identical_initialization_mismip.h5"
        args.T = [-10]
        args.devs = [1.0]
    elif args.true:
        checkpoint_fn = "inputs/ssa_true_initialization_mismip.h5"
    else:
        checkpoint_fn = std_checkpoint_fn
        args.devs = [1.0]

    ns = args.n

    # p_W = ρ_W * g * firedrake.max_value(0, h0 - s0)
    # p_I = ρ_I * g * h0
    # is_floating = firedrake.Function(Q).interpolate(firedrake.conditional(p_I - p_W < args.float_tol, firedrake.Constant(1.0), firedrake.Constant(0.0)))

    h_af = firedrake.max_value(s0 - h0 * (1 - ρ_I / ρ_W), 0)
    ramp = firedrake.min_value(1, h_af / 50.0)
    is_floating = firedrake.Function(Q).interpolate(1.0 - ramp)

    if not os.path.exists(checkpoint_fn):
        model = icepack.models.IceStream(friction=weertman_3)
        solver = icepack.solvers.FlowSolver(model, **par_opts)
        u_truish = solver.diagnostic_solve(
            velocity=u0,
            thickness=h0,
            surface=s0,
            fluidity=A0,
            friction=firedrake.Function(C0.function_space()).interpolate(C0**2.0),
            flow_law_exponent=firedrake.Constant(3.0),
        )
        with firedrake.CheckpointFile(checkpoint_fn, "w") as chk:
            chk.save_mesh(fine_mesh)
            chk.create_group("partial")
            chk.create_group("A0")
            chk.save_function(h0, name="input_thick")
            chk.save_function(s0, name="input_surf")
            chk.save_function(u0, name="input_u")
            chk.save_function(u_truish, name="input_u_ssa")
            chk.save_function(C0, name="input_C")

            if args.true:
                inv_name = "T{:d}_n{:2.1f}".format(-10, 3)
                chk.save_function(u_truish, name=inv_name + "_u1")
                chk.save_function(firedrake.Function(h0.function_space()).interpolate(A0), name=inv_name + "_A")
                chk.save_function(C0, name=inv_name + "_C3")

    if (not args.identical) and (not args.true):
        regularization = get_reg(args.LC, args.LA, is_floating)
        for T_np in args.T:
            for dev in args.devs:
                for n in ns:
                    inv_name = "T{:d}_n{:2.1f}_LC{:1.0e}_LA{:1.0e}".format(T_np, n, args.LC, args.LA)
                    with firedrake.CheckpointFile(checkpoint_fn, "r") as chk:
                        if chk.has_attr("partial", inv_name + "_{:d}".format(args.n_pingpong - 1)):
                            continue

                    T = firedrake.Constant(T_np + 273.15)

                    PETSc.Sys.Print("Initializing " + inv_name)
                    C1_preopt_var = c3_to_c1(C0, u0, minslide=0.1)
                    good_area = firedrake.conditional(C1_preopt_var < 1.0, 1.0, 0.0)
                    C1_preopt = firedrake.Function(Q).interpolate(
                        firedrake.assemble(C1_preopt_var * good_area * firedrake.dx)
                        / firedrake.assemble(good_area * firedrake.dx)
                    )

                    if n > 2:
                        A1_fd = rate_factor(T, n=n)
                    else:
                        A1_fd = rate_factor(T, n=n, m=5.0e-3, m_exp=-1.4)
                    A1_np = A1_fd.values()[0]
                    A1 = firedrake.Function(h0.function_space()).interpolate(A1_np)

                    model = icepack.models.IceStream(
                        friction=pos_linear_weertman, viscosity=tunable_depth_averaged_viscosity
                    )
                    solver = icepack.solvers.FlowSolver(model, **par_opts)

                    C1 = C1_preopt.copy(deepcopy=True)
                    u = u0.copy(deepcopy=True)
                    A = A1.copy(deepcopy=True)
                    mod_A = firedrake.Function(Q).interpolate(0.0)

                    PETSc.Sys.Print("Optimizing " + inv_name)
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
                                A = chk.load_function(fine_mesh, inv_name + "_A_{:d}".format(inverse_iter))
                                mod_A = chk.load_function(fine_mesh, inv_name + "_modA_{:d}".format(inverse_iter))
                                C1 = chk.load_function(fine_mesh, inv_name + "_C1_{:d}".format(inverse_iter))
                                continue
                        PETSc.Sys.Print("Round {:d}".format(inverse_iter + 1))
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
                            loss_functional=get_loss(u0),
                        )

                        with firedrake.CheckpointFile(checkpoint_fn, "a") as chk:
                            chk.set_attr("partial", inv_name + "_{:d}".format(inverse_iter), True)
                            chk.set_attr("A0", inv_name + "_{:d}".format(inverse_iter), A1_np)
                            chk.save_function(u, name=inv_name + "_u_{:d}".format(inverse_iter))
                            chk.save_function(A, name=inv_name + "_A_{:d}".format(inverse_iter))
                            chk.save_function(mod_A, name=inv_name + "_modA_{:d}".format(inverse_iter))
                            chk.save_function(C1, name=inv_name + "_C1_{:d}".format(inverse_iter))
    elif args.identical:
        regularization = get_reg(args.LC, args.LA, is_floating, opt_A=False)
        print("Identical: ignoring all options except LC!")
        inv_name = "T{:d}_n{:2.1f}".format(-10, 3)
        inv_name4 = "T{:d}_n{:2.1f}".format(-10, 4)
        inv_name1_8 = "T{:d}_n{:2.1f}".format(-10, 1.8)
        inv_name3_5 = "T{:d}_n{:2.1f}".format(-10, 3.5)
        with firedrake.CheckpointFile(checkpoint_fn, "r") as chk:
            if not chk.has_attr("partial", inv_name):
                doit = True
            else:
                doit = False
        if doit:
            A3 = firedrake.Function(Q).interpolate(rate_factor(263.15, 3))

            model = icepack.models.IceStream(friction=pos_linear_weertman, viscosity=tunable_depth_averaged_viscosity)
            solver = icepack.solvers.FlowSolver(model, **par_opts)

            C1_preopt_var = c3_to_c1(C0, u0, minslide=0.1)
            good_area = firedrake.conditional(C1_preopt_var < 1.0, 1.0, 0.0)
            C1_preopt = firedrake.Function(Q).interpolate(
                firedrake.assemble(C1_preopt_var * good_area * firedrake.dx)
                / firedrake.assemble(good_area * firedrake.dx)
            )

            C1 = C1_preopt.copy(deepcopy=True)
            u = u0.copy(deepcopy=True)
            mod_A = firedrake.Function(Q).interpolate(0.0)
            C1, _, _, u = inversion(
                solver,
                regularization,
                C1,
                u,
                h0,
                s0,
                A3,
                mod_A,
                3.0,
                is_floating,
                args.n_iter,
                loss_functional=get_loss(u0),
                opt_A=False,
            )

            A4 = A3_to_An(A3, u, h0, s0, 4, Q)
            A3_5 = A3_to_An(A3, u, h0, s0, 3.5, Q)
            A1_8 = A3_to_An(A3, u, h0, s0, 1.8, Q)
            C3 = c1_to_c3(C1, u)
            beta = c3_to_beta(C3, u, u0_coulomb)
            with firedrake.CheckpointFile(checkpoint_fn, "a") as chk:
                chk.set_attr("partial", inv_name, True)
                chk.save_function(u, name=inv_name + "_u1")
                chk.save_function(C1, name=inv_name + "_C1")
                chk.save_function(A3, name=inv_name + "_A")
                chk.save_function(C3, name=inv_name + "_C3")
                chk.save_function(beta, name=inv_name + "_CRCFi")

                chk.save_function(u, name=inv_name4 + "_u1")
                chk.save_function(C1, name=inv_name4 + "_C1")
                chk.save_function(A4, name=inv_name4 + "_A")
                chk.save_function(C3, name=inv_name4 + "_C3")
                chk.save_function(beta, name=inv_name4 + "_CRCFi")

                chk.save_function(u, name=inv_name1_8 + "_u1")
                chk.save_function(C1, name=inv_name1_8 + "_C1")
                chk.save_function(A1_8, name=inv_name1_8 + "_A")
                chk.save_function(C3, name=inv_name1_8 + "_C3")
                chk.save_function(beta, name=inv_name1_8 + "_CRCFi")

                chk.save_function(u, name=inv_name3_5 + "_u1")
                chk.save_function(C1, name=inv_name3_5 + "_C1")
                chk.save_function(A3_5, name=inv_name3_5 + "_A")
                chk.save_function(C3, name=inv_name3_5 + "_C3")
                chk.save_function(beta, name=inv_name3_5 + "_CRCFi")
        else:
            print("Doing nothing!")


def inversion(solver, regularization, C, u0, h0, s0, A0, mod_A, n, is_floating, n_iter, loss_functional, opt_A=True):
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

    PETSc.Sys.Print("Pre-opt loss is {:f}".format(firedrake.assemble(loss_functional(u_w1))))
    if opt_A:
        PETSc.Sys.Print("Pre-opt regularization is {:f}".format(firedrake.assemble(regularization([C, mod_A]))))

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
    else:
        PETSc.Sys.Print("Pre-opt regularization is {:f}".format(firedrake.assemble(regularization(C))))

        def simulation(input_args):
            return solver.diagnostic_solve(
                velocity=u_w1,
                fluidity=A0,
                mod_A=mod_A,
                is_floating=is_floating,
                friction=input_args,
                surface=s0,
                thickness=h0,
                flow_law_exponent=firedrake.Constant(n),
            )

        problem = StatisticsProblem(
            simulation=simulation,
            loss_functional=loss_functional,
            regularization=regularization,
            controls=C,
        )

    estimator = MaximumProbabilityEstimator(
        problem,
        gradient_tolerance=1e-5,
        step_tolerance=1e-5,
        max_iterations=n_iter,
    )

    PETSc.Sys.Print("Optimizing")
    if opt_A:
        C1, mod_A = estimator.solve()
        PETSc.Sys.Print("Post-opt regularization is {:f}".format(firedrake.assemble(regularization([C1, mod_A]))))
    else:
        C1 = estimator.solve()
        PETSc.Sys.Print("Post-opt regularization is {:f}".format(firedrake.assemble(regularization(C1))))
    u = simulation([C1, mod_A])

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

#!/usr/bin/env python
# coding: utf-8
import os
import argparse
from operator import itemgetter
import firedrake
import icepack
import icepack.models.hybrid
from icepackaccs import rate_factor, extract_bed
from icepackaccs.friction import get_weertman, get_regularized_coulomb_simp
from icepack.statistics import (
    StatisticsProblem,
    MaximumProbabilityEstimator,
)
from true_flowline import u0_coulomb

parser = argparse.ArgumentParser
parser = argparse.ArgumentParser()
parser.add_argument("-n", type=float, nargs="+", default=[1.8, 3, 3.5, 4])
parser.add_argument("-n_iter", type=int, default=500)
parser.add_argument("-nbumps", type=int, default=2)
parser.add_argument("-Lexp", type=int, nargs="+", default=[7, 6, 5, 4, 3])
args = parser.parse_args()

ns = args.n
T_np = -10
init = "standard"


def volume(thickness):
    return firedrake.assemble(thickness * firedrake.dx)


def c3_to_c1(C3, u):
    Q2D = firedrake.FunctionSpace(u.ufl_domain()._base_mesh, "CG", 2)
    C1 = firedrake.Function(C3.function_space())
    C1_bed = firedrake.Function(Q2D).interpolate(
        firedrake.sqrt(extract_bed(C3) ** 2.0 * abs(extract_bed(u)) ** (1.0 / 3.0 - 1.0))
    )
    C1.dat.data[:] = C1_bed.dat.data[:]
    return C1


def c1_to_c3(C1, u):
    Q2D = firedrake.FunctionSpace(u.ufl_domain()._base_mesh, "CG", 2)
    C3 = firedrake.Function(C1.function_space())
    C3_bed = firedrake.Function(Q2D).interpolate(
        firedrake.sqrt(extract_bed(C1) ** 2.0 / abs(extract_bed(u)) ** (1.0 / 3.0 - 1.0))
    )
    C3.dat.data[:] = C3_bed.dat.data[:]
    return C3


def c3_to_beta(C3, u, u0):
    Q2D = firedrake.FunctionSpace(u.ufl_domain()._base_mesh, "CG", 2)
    beta = firedrake.Function(firedrake.FunctionSpace(u.ufl_domain(), "CG", 2, vfamily="R", vdegree=0))
    beta_bed = firedrake.Function(Q2D).interpolate(
        firedrake.sqrt(
            extract_bed(C3) ** 2.0
            * (abs(extract_bed(u)) ** (1.0 / 3.0 + 1) + u0 ** (1.0 / 3.0 + 1)) ** (1.0 / (3.0 + 1.0))
        )
    )
    beta.dat.data[:] = beta_bed.dat.data[:]
    return beta


def taub_rcfi(beta, u, u0):
    return firedrake.Function(u.function_space()).interpolate(
        beta * u ** (1 / 3) / ((abs(u)) ** (1.0 / 3.0 + 1) + u0 ** (1.0 / 3.0 + 1)) ** (1.0 / (3.0 + 1))
    )


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

    ε_x = icepack.models.hybrid.horizontal_strain_rate(velocity=u, surface=s, thickness=h)
    ε_z = icepack.models.hybrid.vertical_strain_rate(velocity=u, thickness=h)
    ε_e = icepack.models.hybrid._effective_strain_rate(ε_x, ε_z, ε_min)
    return ε_e


def A3_to_An(A3, u, h, s, n):
    ε_e = effective_strain_rate(velocity=u, thickness=h, surface=h)
    # return firedrake.Function(u.function_space()).interpolate(A3 * ε_e ** (n - 3.0))
    An = firedrake.Function(u.function_space()).interpolate(A3 ** (n / 3.0) * ε_e ** (1.0 - n / 3.0))
    An_mean = firedrake.assemble(
        An * firedrake.conditional(h > 10.1, 1.0, 0.0) * firedrake.conditional(h < 50.1, 1.0, 0.0) * firedrake.dx
    ) / firedrake.assemble(
        firedrake.conditional(h > 10.1, 1.0, 0.0) * firedrake.conditional(h < 50.1, 1.0, 0.0) * firedrake.dx
    )
    return firedrake.Function(u.function_space()).interpolate(firedrake.conditional(h > 10.1, An, An_mean))


min_thick = firedrake.Constant(10.0)
Lx = 500e3
nx = 2001

# Without these we get a mysterious error on new firedrake installs
mesh1d = firedrake.IntervalMesh(nx, Lx)
mesh = firedrake.ExtrudedMesh(mesh1d, layers=1)

nbumps = args.nbumps
checkpoint_fn = "inputs/lcurve_inversion_results_{:s}_bumps{:d}.h5".format(init, nbumps)
if not os.path.exists(checkpoint_fn):
    write_inputs = True
    mesh1d = firedrake.IntervalMesh(nx, Lx)
    mesh = firedrake.ExtrudedMesh(mesh1d, layers=1)
    mesh.name = "flowline"
    with firedrake.CheckpointFile(checkpoint_fn, "w") as chk:
        chk.save_mesh(mesh)
        chk.create_group("already_run")
else:
    write_inputs = False
    with firedrake.CheckpointFile(checkpoint_fn, "r") as chk:
        mesh = chk.load_mesh("flowline")

Q = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="R", vdegree=0)
V = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="GL", vdegree=2)
V_2 = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="GL", vdegree=2)

V_8 = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="GL", vdegree=8)

x, ζ = firedrake.SpatialCoordinate(mesh)

cache_fn = "inputs/flowline_n3_{:03d}C_weertman3_bumps{:1d}.h5".format(-10, nbumps)
with firedrake.CheckpointFile(cache_fn, "r") as chk:
    field_names = ["surf", "bed", "thick", "u2", "u4", "C"]
    mesh_cache = chk.load_mesh("flowline")
    # start_time = chk.get_attr("metadata", "total_time")
    fields = {name: chk.load_function(mesh_cache, name) for name in field_names}
    h0 = firedrake.Function(Q).interpolate(fields["thick"])
    s0 = firedrake.Function(Q).interpolate(fields["surf"])
    u0 = firedrake.Function(V_8).interpolate(fields["u4"])
    C0 = firedrake.Function(Q).interpolate(fields["C"])
    b = firedrake.Function(Q).interpolate(fields["bed"])
    u_dummy = firedrake.Function(V_8).interpolate(fields["u4"] * 0.5)

if write_inputs:
    with firedrake.CheckpointFile(checkpoint_fn, "a") as chk:
        chk.save_function(h0, name="input_thick")
        chk.save_function(s0, name="input_surf")
        chk.save_function(u0, name="input_u")
        chk.save_function(C0, name="input_C")
        chk.save_function(b, name="input_bed")


regularized_coulomb = get_regularized_coulomb_simp(m=3, u_0=u0_coulomb)
weertman_3 = get_weertman(m=3)
weertman_1 = get_weertman(m=1)


def pos_linear_weertman(**kwargs):
    C = kwargs.pop("friction")
    return weertman_1(friction=C**2.0, **kwargs)


def loss_functional(u):
    δu = u - u0
    return (
        0.5
        / Lx
        * firedrake.conditional(h0 > firedrake.Constant(10.1), firedrake.Constant(1.0), firedrake.Constant(0.0))
        * δu**2
        * firedrake.ds_t(mesh)
    )


def smoothness(C):
    return (
        0.5
        / Lx
        * firedrake.conditional(x > firedrake.Constant(1.0), firedrake.Constant(1.0), firedrake.Constant(0.0))
        * firedrake.conditional(h0 > firedrake.Constant(10.1), firedrake.Constant(1.0), firedrake.Constant(0.0))
        * firedrake.inner(firedrake.grad(C), firedrake.grad(C))
        * firedrake.ds_b(mesh)
    )


opts = {"dirichlet_ids": [1], "diagnostic_solver_type": "icepack"}
opts = {
    "dirichlet_ids": [1],
    "diagnostic_solver_type": "petsc",
    "diagnostic_solver_parameters": {
        "snes_type": "newtonls",
        "snes_max_it": 10000,
        "snes_stol": 1.0e-4,
        "snes_rtol": 1.0e-8,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "snes_linesearch_type": "bt",
        "snes_linesearch_order": 2,
        "pc_factor_mat_solver_type": "mumps",
        "max_iterations": 2,
        # "snes_monitor": None,
        # "snes_linesearch_monitor": None,
    },
}

for n in ns:
    C1_opt = c3_to_c1(C0, u0).interpolate(0.1)
    # good = firedrake.conditional(h0 > 10.1, 1.0, 0.0) * firedrake.conditional(x > 1000.0, 1.0, 0.0)
    # C1_opt.dat.data[:] = firedrake.assemble(C1_opt * good * firedrake.dx) / firedrake.assemble(good * firedrake.dx)
    for Lexp in args.Lexp:
        LCap = 10.0**Lexp
        inv_name = "T{:d}_n{:2.1f}_L{:2.1e}".format(T_np, n, LCap)
        smoother_name = "T{:d}_n{:2.1f}_L{:2.1e}".format(T_np, n, 10.0 ** (Lexp + 1))
        with firedrake.CheckpointFile(checkpoint_fn, "r") as chk:
            if chk.has_attr("already_run", inv_name) and chk.get_attr("already_run", inv_name):
                C1_opt = chk.load_function(mesh, inv_name + "_C1")
                continue
            # if chk.has_attr("already_run", smoother_name) and chk.get_attr("already_run", smoother_name):
            #    print("Reloading smoother")
            #    C1_opt = chk.load_function(mesh, smoother_name + "_C1")

        C1_preopt = C1_opt.copy(deepcopy=True)

        T = firedrake.Constant(273.15 + T_np)
        if n > 2:
            A = rate_factor(T, n=n)
        else:
            A = rate_factor(T, n=n, m=1.0e-2, m_exp=1.4)

        model = icepack.models.HybridModel(friction=pos_linear_weertman)
        solver = icepack.solvers.FlowSolver(model, **opts)
        u_w1 = solver.diagnostic_solve(
            velocity=u0,
            thickness=h0,
            surface=s0,
            fluidity=A,
            friction=C1_preopt,
            flow_law_exponent=firedrake.Constant(n),
        )
        C1 = C1_preopt.copy(deepcopy=True)

        def regularization(C):
            L = firedrake.Constant(LCap)
            return 0.5 / Lx * (L) ** 2 * firedrake.inner(firedrake.grad(C), firedrake.grad(C)) * firedrake.ds_b(mesh)

        def simulation(C):
            return solver.diagnostic_solve(
                velocity=u_w1, fluidity=A, friction=C, surface=s0, thickness=h0, flow_law_exponent=firedrake.Constant(n)
            )

        problem = StatisticsProblem(
            simulation=simulation,
            loss_functional=loss_functional,
            regularization=regularization,
            controls=C1,
        )

        n_iter = 50
        estimator = MaximumProbabilityEstimator(
            problem,
            gradient_tolerance=1e-5,
            step_tolerance=1e-5,
            max_iterations=args.n_iter,
        )
        print("Optimizing T={:d}, n={:2.1f}, L={:2.1e}".format(T_np, n, LCap))
        C1_opt = estimator.solve()
        u_w1_opt = simulation(C1_opt)

        state = estimator._solver.getAlgorithmState()
        with firedrake.CheckpointFile(checkpoint_fn, "a") as chk:
            chk.create_group(inv_name)
            chk.set_attr(inv_name, "lambda", LCap)
            chk.set_attr(inv_name, "max_iterations", n_iter)
            chk.set_attr(inv_name, "gnorm", state.gnorm)
            chk.set_attr(inv_name, "cnorm", state.cnorm)
            chk.set_attr(inv_name, "snorm", state.snorm)
            chk.set_attr(inv_name, "loss", firedrake.assemble(loss_functional(u_w1_opt)))
            chk.set_attr(inv_name, "regularization", firedrake.assemble(regularization(C1_opt)))
            chk.set_attr(inv_name, "smoothness", firedrake.assemble(smoothness(C1_opt)))
            chk.set_attr(
                inv_name,
                "cost",
                firedrake.assemble(loss_functional(u_w1_opt)) + firedrake.assemble(regularization(C1_opt)),
            )

            chk.set_attr("already_run", inv_name, True)
            chk.save_function(C1_opt, name=inv_name + "_C1")
            chk.save_function(u_w1_opt, name=inv_name + "_u1")

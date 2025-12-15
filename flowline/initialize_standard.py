#!/usr/bin/env python
# coding: utf-8

import sys
import os
import argparse
import time
import firedrake
import icepack
import icepack.models.hybrid
import matplotlib.pyplot as plt
from icepackaccs import rate_factor, extract_surface, extract_bed
from icepackaccs.friction import get_weertman, get_regularized_coulomb_simp
from icepack.statistics import (
    StatisticsProblem,
    MaximumProbabilityEstimator,
)
from true_flowline import u0_coulomb

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=float, nargs="+", default=[1.8, 3, 3.5, 4])
parser.add_argument("-T", type=int, nargs="+", default=[-12, -10, -8])
parser.add_argument("-nbumps", type=int, default=2)
parser.add_argument("-n_iter", type=int, default=2500)
args = parser.parse_args()

ns = args.n
Ts = args.T


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


def fix_end(C, x, cutoff=4.25e5, lcut=4.0e5, rcut=4.1e5):
    mean_C = firedrake.assemble(
        firedrake.conditional(x < rcut, 1.0, 0.0) * firedrake.conditional(x > lcut, 1.0, 0) * C * firedrake.dx
    ) / (rcut - lcut)
    return firedrake.Function(C.function_space()).interpolate(firedrake.conditional(x < cutoff, C, mean_C))


min_thick = firedrake.Constant(10.0)
Lx = 500e3
nx = 2001

# Without these we get a mysterious error on new firedrake installs
mesh1d_dum = firedrake.IntervalMesh(nx, Lx)
mesh_dum = firedrake.ExtrudedMesh(mesh1d_dum, layers=1)


nbumps = args.nbumps
checkpoint_fn = "inputs/standard_init_bumps{:d}.h5".format(nbumps)
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
    u_dummy = firedrake.Function(V_8).interpolate(0.1)

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


for T_np in Ts:
    for n in ns:
        if n == 3.5:
            LCap = 1.0e6
        else:
            LCap = 1.0e5


        def regularization(C):
            L = firedrake.Constant(LCap)
            return 0.5 / Lx * (L) ** 2 * firedrake.inner(firedrake.grad(C), firedrake.grad(C)) * firedrake.ds_b(mesh)



        inv_name = "T{:d}_n{:2.1f}".format(T_np, n)
        with firedrake.CheckpointFile(checkpoint_fn, "r") as chk:
            if chk.has_attr("already_run", inv_name) and chk.get_attr("already_run", inv_name):
                continue

        C1_preopt = c3_to_c1(C0, u0)
        if n == 3.5:
            C1_preopt.dat.data[:] = 1.0
        else:
            C1_preopt.dat.data[:] = 0.1
        T = firedrake.Constant(T_np + 273.15)
        if n > 2:
            A = rate_factor(T, n=n)
        else:
            A = rate_factor(T, n=n, m=5.0e-3, m_exp=-1.4)

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

        if False:  # T_np == -10 and n == 3:
            C1 = c3_to_c1(C0, u0)
            u_w1 = solver.diagnostic_solve(
                velocity=u0,
                thickness=h0,
                surface=s0,
                fluidity=A,
                friction=C1_preopt,
                flow_law_exponent=firedrake.Constant(n),
            )
            u_w1_opt = u_w1.copy(deepcopy=True)
        else:

            def simulation(C):
                return solver.diagnostic_solve(
                    velocity=u_w1,
                    fluidity=A,
                    friction=C,
                    surface=s0,
                    thickness=h0,
                    flow_law_exponent=firedrake.Constant(n),
                )

            problem = StatisticsProblem(
                simulation=simulation,
                loss_functional=loss_functional,
                regularization=regularization,
                controls=C1,
            )

            estimator = MaximumProbabilityEstimator(
                problem,
                gradient_tolerance=1e-5,
                step_tolerance=1e-5,
                max_iterations=args.n_iter,
            )
            print("Optimizing T={:d}, n={:2.1f}".format(T_np, n))
            fn_log = os.path.join(
                "logs", "inversion_T{:d}_n{:2.1f}_L{:2.1e}_nbumps{:d}.log".format(T_np, n, LCap, nbumps)
            )
            if os.path.exists(fn_log):
                os.remove(fn_log)
            sys.stdout.flush()
            time.sleep(2)
            newstdout = os.dup(1)
            new_fout = os.open(fn_log, os.O_WRONLY | os.O_CREAT, 0o644)
            os.dup2(new_fout, 1)
            C1 = estimator.solve()
            os.close(new_fout)
            sys.stdout = os.fdopen(newstdout, "w")
            sys.stdout.flush()
            time.sleep(2)

            C1 = fix_end(C1, x)
            u_w1_opt = simulation(C1)

        C3 = c1_to_c3(C1, u_w1_opt)
        C3 = fix_end(C3, x)
        model = icepack.models.HybridModel(friction=weertman_3)
        solver = icepack.solvers.FlowSolver(model, **opts)
        u_w3 = solver.diagnostic_solve(
            velocity=u0,
            thickness=h0,
            surface=s0,
            fluidity=A,
            friction=firedrake.Function(Q).interpolate(C3**2.0),
            flow_law_exponent=firedrake.Constant(n),
        )

        Beta = c3_to_beta(C3, u_w1_opt, u0_coulomb)
        Beta = fix_end(Beta, x)
        model = icepack.models.HybridModel(friction=regularized_coulomb)
        solver = icepack.solvers.FlowSolver(model, **opts)
        u_rcf = solver.diagnostic_solve(
            velocity=u0,
            thickness=h0,
            surface=s0,
            fluidity=A,
            friction=firedrake.Function(Q).interpolate(Beta**2.0),
            flow_law_exponent=firedrake.Constant(n),
        )

        with firedrake.CheckpointFile(checkpoint_fn, "a") as chk:
            chk.set_attr("already_run", inv_name, True)
            chk.save_function(C1, name=inv_name + "_C1")
            chk.save_function(C3, name=inv_name + "_C3")
            chk.save_function(Beta, name=inv_name + "_CRCFi")
            chk.save_function(u_w1_opt, name=inv_name + "_u1")
            chk.save_function(u_w3, name=inv_name + "_u3")
            chk.save_function(u_rcf, name=inv_name + "_uRCFi")

        fig, ax = plt.subplots()
        firedrake.plot(
            extract_surface(firedrake.Function(V_8).interpolate(u0 * (2 * (h0 > 10) - 1))),
            axes=ax,
            edgecolor="k",
            label="Target",
        )
        firedrake.plot(
            extract_surface(firedrake.Function(V_8).interpolate(u_w1 * (2 * (h0 > 10) - 1))),
            axes=ax,
            edgecolor="C1",
            label="$m$=1 Initial",
        )
        firedrake.plot(
            extract_surface(firedrake.Function(V_8).interpolate(u_w1_opt * (2 * (h0 > 10) - 1))),
            axes=ax,
            edgecolor="C1",
            label="$m$=1",
        )
        firedrake.plot(
            extract_surface(firedrake.Function(V_8).interpolate(u_w3 * (2 * (h0 > 10) - 1))),
            axes=ax,
            edgecolor="C0",
            label="$m$=3",
        )
        firedrake.plot(
            extract_surface(firedrake.Function(V_8).interpolate(u_rcf * (2 * (h0 > 10) - 1))),
            axes=ax,
            edgecolor="C2",
            label="RCFi",
        )
        ax.legend(loc="best")
        fig.savefig("figs/tests/test_that_vels_match_n{:2.1f}_T{:d}.pdf".format(n, T_np))

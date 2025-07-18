#!/usr/bin/env python
# coding: utf-8
import sys
from operator import itemgetter
import firedrake
import icepack
import icepack.models.hybrid
import matplotlib.pyplot as plt
from icepackaccs import rate_factor, extract_surface, extract_bed
from icepackaccs.friction import get_weertman, get_regularized_coulomb_simp
from true_flowline import a_0, δa, u0_coulomb


Ts = [-20, -10, -5]


def kmfmt(x, pos):
    return "{:d}".format(int(x / 1000.0))


def volume(thickness):
    return firedrake.assemble(thickness * firedrake.dx)


def c3_to_c1(C3, u, h):
    Q2D = firedrake.FunctionSpace(u.ufl_domain()._base_mesh, "CG", 2)
    C1 = firedrake.Function(C3.function_space())
    C1_bed = firedrake.Function(Q2D).interpolate(
        firedrake.sqrt(extract_bed(C3) ** 2.0 * abs(extract_bed(u)) ** (1.0 / 3.0 - 1.0))
    )
    C1.dat.data[:] = C1_bed.dat.data[:]

    x, _ = firedrake.SpatialCoordinate(C1.ufl_domain())
    C1_mean = firedrake.assemble(
        C1 * firedrake.conditional(h > 100.1, 1.0, 0.0) * firedrake.conditional(x > 3e5, 1.0, 0.0) * firedrake.dx
    ) / firedrake.assemble(
        firedrake.conditional(h > 100.1, 1.0, 0.0) * firedrake.conditional(x > 3e5, 1.0, 0.0) * firedrake.dx
    )
    return firedrake.Function(C3.function_space()).interpolate(firedrake.conditional(h > 10.1, C1, C1_mean))


def c1_to_c3(C1, u, h):
    Q2D = firedrake.FunctionSpace(u.ufl_domain()._base_mesh, "CG", 2)
    C3 = firedrake.Function(C1.function_space())
    C3_bed = firedrake.Function(Q2D).interpolate(
        firedrake.sqrt(extract_bed(C1) ** 2.0 / abs(extract_bed(u)) ** (1.0 / 3.0 - 1.0))
    )
    C3.dat.data[:] = C3_bed.dat.data[:]

    x, _ = firedrake.SpatialCoordinate(C1.ufl_domain())
    C3_mean = firedrake.assemble(
        C3 * firedrake.conditional(h > 100.1, 1.0, 0.0) * firedrake.conditional(x > 3e5, 1.0, 0.0) * firedrake.dx
    ) / firedrake.assemble(
        firedrake.conditional(h > 100.1, 1.0, 0.0) * firedrake.conditional(x > 3e5, 1.0, 0.0) * firedrake.dx
    )
    return firedrake.Function(C1.function_space()).interpolate(firedrake.conditional(h > 10.1, C3, C3_mean))


def c3_to_beta(C3, u, u0, h):
    Q2D = firedrake.FunctionSpace(u.ufl_domain()._base_mesh, "CG", 2)
    beta = firedrake.Function(firedrake.FunctionSpace(u.ufl_domain(), "CG", 2, vfamily="R", vdegree=0))
    beta_bed = firedrake.Function(Q2D).interpolate(
        firedrake.sqrt(
            extract_bed(C3) ** 2.0
            * (abs(extract_bed(u)) ** (1.0 / 3.0 + 1) + u0 ** (1.0 / 3.0 + 1)) ** (1.0 / (3.0 + 1.0))
        )
    )
    beta.dat.data[:] = beta_bed.dat.data[:]

    x, _ = firedrake.SpatialCoordinate(C1.ufl_domain())
    beta_mean = firedrake.assemble(
        beta * firedrake.conditional(h > 100.1, 1.0, 0.0) * firedrake.conditional(x > 3e5, 1.0, 0.0) * firedrake.dx
    ) / firedrake.assemble(
        firedrake.conditional(h > 100.1, 1.0, 0.0) * firedrake.conditional(x > 3e5, 1.0, 0.0) * firedrake.dx
    )
    return firedrake.Function(C3.function_space()).interpolate(firedrake.conditional(h > 10.1, beta, beta_mean))


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
Lx_mesh = 500e3
nx = 2001


nbumps = 2
if len(sys.argv) == 2:
    nbumps = int(sys.argv[1])

checkpoint_fn = "inputs/identical_init_bumps{:d}.h5".format(nbumps)

mesh1d = firedrake.IntervalMesh(nx, Lx_mesh)
mesh = firedrake.ExtrudedMesh(mesh1d, layers=1)
mesh.name = "flowline"
with firedrake.CheckpointFile(checkpoint_fn, "w") as chk:
    chk.save_mesh(mesh)


Q = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="R", vdegree=0)
V = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="GL", vdegree=2)
V_2 = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="GL", vdegree=2)

V_8 = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="GL", vdegree=8)

x, ζ = firedrake.SpatialCoordinate(mesh)
a = firedrake.Function(Q).interpolate(a_0 - δa * x / Lx)

ns = [1.8, 3, 3.5, 4]

regularized_coulomb = get_regularized_coulomb_simp(m=3, u_0=u0_coulomb)
weertman_3 = get_weertman(m=3)
weertman_1 = get_weertman(m=1)


for T_np in Ts:
    cache_fn = "inputs/flowline_n3_{:03d}C_weertman3_bumps{:1d}.h5".format(T_np, nbumps)
    with firedrake.CheckpointFile(cache_fn, "r") as chk:
        field_names = ["surf", "bed", "thick", "u2", "u4", "C"]
        mesh_cache = chk.load_mesh("flowline")
        # start_time = chk.get_attr("metadata", "total_time")
        fields = {name: chk.load_function(mesh_cache, name) for name in field_names}
        h0 = firedrake.Function(Q).interpolate(fields["thick"])
        s0 = firedrake.Function(Q).interpolate(fields["surf"])
        u0 = firedrake.Function(V_8).interpolate(fields["u4"])
        C3 = firedrake.Function(Q).interpolate(fields["C"])
        b = firedrake.Function(Q).interpolate(fields["bed"])
        u_dummy = firedrake.Function(V_8).interpolate(0.1)

    T = firedrake.Constant(T_np + 273.15)
    A3 = rate_factor(T, n=3)

    fig, ax = plt.subplots()
    ax.semilogy()
    A3var = A3_to_An(A3, u0, h0, s0, 3.0)
    A4 = A3_to_An(A3, u0, h0, s0, 4.0)
    A1_8 = A3_to_An(A3, u0, h0, s0, 1.8)
    firedrake.plot(extract_surface(A3var), axes=ax, label="$n$=3")
    firedrake.plot(extract_surface(A4), axes=ax, label="$n$=4")
    firedrake.plot(extract_surface(A1_8), axes=ax, label="$n$=1.8")
    ax.legend(loc="best")
    ax.set_ylim(0.01, 1e3)
    fig.savefig("figs/tests/comp_A_{:d}C.pdf".format(T_np))

    opts = {
        "dirichlet_ids": [1],
        "diagnostic_solver_type": "petsc",
        "diagnostic_solver_parameters": {
            "snes_type": "newtonls",
            "snes_max_it": 10000,
            "snes_stol": 1.0e-8,
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
    # opts = {"dirichlet_ids": [1], "diagnostic_solver_type": "icepack"}

    fig, ax = plt.subplots()
    firedrake.plot(
        extract_surface(firedrake.Function(V_8).interpolate(u0 * (2 * (h0 > 10) - 1))),
        axes=ax,
        edgecolor="k",
        label="Target",
    )
    for n in ns:
        inv_name = "T{:d}_n{:2.1f}".format(T_np, n)
        print(inv_name.replace("_", " "))
        C1 = c3_to_c1(C3, u0, h0)
        Beta = c3_to_beta(C3, u0, u0_coulomb, h0)
        A = A3_to_An(A3, u0, h0, s0, n)

        model = icepack.models.HybridModel(friction=weertman_1)
        solver = icepack.solvers.FlowSolver(model, **opts)
        u_w1 = solver.diagnostic_solve(
            velocity=u_dummy,
            thickness=h0,
            surface=s0,
            fluidity=A,
            friction=firedrake.Function(Q).interpolate(C1**2.0),
            flow_law_exponent=firedrake.Constant(n),
        )

        model = icepack.models.HybridModel(friction=weertman_3)
        solver = icepack.solvers.FlowSolver(model, **opts)
        u_w3 = solver.diagnostic_solve(
            velocity=u_dummy,
            thickness=h0,
            surface=s0,
            fluidity=A,
            friction=firedrake.Function(Q).interpolate(C3**2.0),
            flow_law_exponent=firedrake.Constant(n),
        )

        model = icepack.models.HybridModel(friction=regularized_coulomb)
        solver = icepack.solvers.FlowSolver(model, **opts)
        u_rcf = solver.diagnostic_solve(
            velocity=u_dummy,
            thickness=h0,
            surface=s0,
            fluidity=A,
            friction=firedrake.Function(Q).interpolate(Beta**2.0),
            flow_law_exponent=firedrake.Constant(n),
        )

        with firedrake.CheckpointFile(checkpoint_fn, "a") as chk:
            chk.save_function(C1, name=inv_name + "_C1")
            chk.save_function(C3, name=inv_name + "_C3")
            chk.save_function(Beta, name=inv_name + "_CRCFi")
            chk.save_function(u_w1, name=inv_name + "_u1")
            chk.save_function(u_w3, name=inv_name + "_u3")
            chk.save_function(u_rcf, name=inv_name + "_uRCFi")
            chk.save_function(A, name=inv_name + "_A")

        firedrake.plot(
            extract_surface(firedrake.Function(V_8).interpolate(u_w1 * (2 * (h0 > 10) - 1))),
            axes=ax,
            label="$m$=1, n={:2.1f}".format(n),
        )
        firedrake.plot(
            extract_surface(firedrake.Function(V_8).interpolate(u_w3 * (2 * (h0 > 10) - 1))),
            axes=ax,
            label="$m$=3, n={:2.1f}".format(n),
        )
        firedrake.plot(
            extract_surface(firedrake.Function(V_8).interpolate(u_rcf * (2 * (h0 > 10) - 1))),
            axes=ax,
            label="RCFi, n={:2.1f}".format(n),
        )
    ax.legend(loc="best")
    fig.savefig("figs/tests/test_that_vels_match_T{:d}_nbumps{:d}.pdf".format(T_np, nbumps))

    fig, ax = plt.subplots()
    firedrake.plot(extract_surface(C1), axes=ax, label="$m$=1")
    firedrake.plot(extract_surface(C3), axes=ax, label="$m$=3")
    firedrake.plot(extract_surface(Beta), axes=ax, label="RCFi")
    ax.legend(loc="best")
    ax.set_ylim(0.01, 1e3)
    fig.savefig("figs/tests/comp_C_{:d}C.pdf".format(T_np))

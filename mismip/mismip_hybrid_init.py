#!/usr/bin/env python3
# coding: utf-8

import os
import firedrake

from firedrake.__future__ import interpolate
import icepack
import icepack.plot
from icepack.utilities import lift3d

from icepackaccs import extract_surface
from icepackaccs.mismip import Lx, Ly, mismip_bed_topography
from icepackaccs.friction import get_ramp_weertman
from icepackaccs.viscosity import rate_factor

from libmismip import (
    run_simulation,
    fast_opts,
    par_opts,
    mirrored_tripcolor as tripcolor,
    mirrored_tricontour as tricontour,
)
from firedrake.petsc import PETSc
from icepack.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
)

smooth_weertman_m3 = get_ramp_weertman(m=3.0, h_t=50.0)

field_names = ["surface", "thickness", "velocity"]

fields_ssa = None

T0 = 263.15

# Needed to avoid assertion error bug in firedrake
mesh1d = firedrake.IntervalMesh(100, 120)
mesh_dum = firedrake.ExtrudedMesh(mesh1d, layers=1)


def subplots(**kwargs):
    fig, axes = icepack.plot.subplots(figsize=(10, 4))
    axes.set_aspect(2)
    axes.set_xlim((0, Lx))
    axes.set_ylim((0, Ly))
    return fig, axes


def colorbar(fig, colors, **kwargs):
    return fig.colorbar(colors, fraction=0.012, pad=0.025, **kwargs)


A = firedrake.Constant(rate_factor(T0))
# C = firedrake.Constant(1.3e-4)  # For the garbage version (no flotation)
C0 = 1.0e-2  # for ramped weertman (cutoff at 50 m above flotation)
C = firedrake.Constant(C0)
a = firedrake.Constant(0.3)

model = icepack.models.HybridModel(friction=smooth_weertman_m3)

dt = 2.0
final_time = 3600.0

opts = {
    "dirichlet_ids": [4],
    "side_wall_ids": [1, 3],
    "diagnostic_solver_type": "icepack",
    "diagnostic_solver_parameters": {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "tolerance": 1e-8,
    },
}
solver = icepack.solvers.FlowSolver(model, **opts)

time_now = 20000.0

if not os.path.exists("inputs/mismip-fine-degree4.h5"):
    if os.path.exists("inputs/too_rough/mismip-fine-degree4.h5"):
        if os.path.exists("inputs/mismip-fine-ssa-15.0kyr.h5"):
            PETSc.Sys.Print("Loading SSA")
            with firedrake.CheckpointFile("inputs/mismip-fine-ssa-15.0kyr.h5", "r") as chk:
                fine_mesh2d = chk.load_mesh("fine_mesh2d")
        else:
            raise FileNotFoundError("Need SSA output at 15 kyr to continue")
        fine_mesh = firedrake.ExtrudedMesh(fine_mesh2d, layers=1, name="fine_mesh")

        target_time = 20000.0

        Q2 = firedrake.FunctionSpace(fine_mesh, "CG", 2, vfamily="R", vdegree=0)
        V2 = firedrake.VectorFunctionSpace(fine_mesh, "CG", 2, dim=2, vfamily="GL", vdegree=2)
        V4 = firedrake.VectorFunctionSpace(fine_mesh, "CG", 2, dim=2, vfamily="GL", vdegree=4)

        z_b = firedrake.assemble(interpolate(mismip_bed_topography(fine_mesh), Q2))

        PETSc.Sys.Print("Reloading old degree 4 results...")
        fields_4 = {}
        with firedrake.CheckpointFile("inputs/too_rough/mismip-fine-degree4.h5", "r") as chk:
            fine_mesh4 = chk.load_mesh("fine_mesh")
            time_now = chk.get_attr("metadata", "total_time")
            for key in field_names:
                fields_4[key] = chk.load_function(fine_mesh4, key)

        solver = icepack.solvers.FlowSolver(model, **par_opts)
        fields_complete = {
            "surface": firedrake.Function(Q2).interpolate(fields_4["surface"]),
            "thickness": firedrake.Function(Q2).interpolate(fields_4["thickness"]),
        }
        x = firedrake.SpatialCoordinate(fine_mesh)[0]
        u_init = firedrake.assemble(interpolate(firedrake.as_vector((90 * x / Lx, 0)), V4))
        # u_init0 = firedrake.Function(V2).interpolate(0.9 * fields_4["velocity"])
        # u_init = firedrake.Function(V4).interpolate(u_init0)
        PETSc.Sys.Print("Re-initializing velocity")
        u_0 = solver.diagnostic_solve(
            velocity=u_init,
            thickness=fields_complete["thickness"],
            surface=fields_complete["surface"],
            fluidity=A,
            friction=C,
        )
        fields_complete["velocity"] = u_0
    else:
        if not os.path.exists("inputs/mismip-fine-degree2.h5"):
            if os.path.exists("inputs/mismip-fine-ssa-15.0kyr.h5"):
                PETSc.Sys.Print("Loading SSA")
                with firedrake.CheckpointFile("inputs/mismip-fine-ssa-15.0kyr.h5", "r") as chk:
                    fields_ssa = {}
                    fine_mesh2d = chk.load_mesh("fine_mesh2d")
                    Q2_2d = firedrake.FunctionSpace(fine_mesh2d, "CG", 2)
                    V2_2d = firedrake.VectorFunctionSpace(fine_mesh2d, "CG", 2, dim=2)
                    for key in field_names:
                        if key == "velocity":
                            fields_ssa[key] = firedrake.project(chk.load_function(fine_mesh2d, key), V2_2d)
                        else:
                            fields_ssa[key] = firedrake.project(chk.load_function(fine_mesh2d, key), Q2_2d)
            else:
                raise FileNotFoundError("Need SSA output at 15 kyr to continue")

            fine_mesh = firedrake.ExtrudedMesh(fine_mesh2d, layers=1, name="fine_mesh")

            target_time = 20000.0

            Q2 = firedrake.FunctionSpace(fine_mesh, "CG", 2, vfamily="R", vdegree=0)
            V2 = firedrake.VectorFunctionSpace(fine_mesh, "CG", 2, dim=2, vfamily="GL", vdegree=2)

            z_b = firedrake.assemble(interpolate(mismip_bed_topography(fine_mesh), Q2))
            if fields_ssa is not None:
                PETSc.Sys.Print("Interpolating thickness")
                h_0 = lift3d(fields_ssa["thickness"], Q2)
                time_so_far = 15000.0
                dt = 0.5
            else:
                h_0 = firedrake.assemble(interpolate(firedrake.Constant(100), Q2))
                time_so_far = 0.0
                dt = 1.0

            s_0 = icepack.compute_surface(thickness=h_0, bed=z_b)
            x = firedrake.SpatialCoordinate(fine_mesh)[0]
            u_init = firedrake.assemble(interpolate(firedrake.as_vector((90 * x / Lx, 0)), V2))
            solver = icepack.solvers.FlowSolver(model, **opts)

            PETSc.Sys.Print("Getting initial velocity")
            u_0 = solver.diagnostic_solve(
                velocity=u_init,
                thickness=h_0,
                surface=s_0,
                fluidity=A,
                friction=C,
            )

            fields = {"surface": s_0, "thickness": h_0, "velocity": u_0}

            PETSc.Sys.Print("Advancing 500 yr with fine steps to start")
            exception, fields_2 = run_simulation(solver, 500, dt, bed=z_b, a=a, C=C, A=A, **fields)
            time_so_far += 500

            PETSc.Sys.Print(f"Initializing to {target_time} kyr with a {(target_time - time_so_far) / 1000} kyr run")
            dt = 1.0
            exception, fields_2 = run_simulation(
                solver, target_time - time_so_far, dt, bed=z_b, a=a, C=C, A=A, **fields_2
            )

            with firedrake.CheckpointFile("inputs/mismip-fine-degree2.h5", "w") as chk:
                chk.save_mesh(fine_mesh)
                for key in fields_2:
                    chk.save_function(fields_2[key], name=key)
        else:
            PETSc.Sys.Print("Found initialized h5 checkpoint")

        first_hybrid_fn = "inputs/mismip-fine-degree2.h5"
        output_fn = "inputs/mismip-fine-degree2_comp.h5"

        fields_deg2 = {}
        deg2_fn = "inputs/mismip-fine-degree2.h5"
        PETSc.Sys.Print("Reloading looped results...")
        with firedrake.CheckpointFile(deg2_fn, "r") as chk:
            fine_mesh = chk.load_mesh("fine_mesh")
            try:
                time_now = chk.get_attr("metadata", "total_time")
            except KeyError:
                time_now = 20000
            for key in field_names:
                fields_deg2[key] = chk.load_function(fine_mesh, key)

        fig, axes = subplots(figsize=(14, 6))
        colors = tripcolor(extract_surface(fields_deg2["thickness"]), axes=axes)
        s2d = extract_surface(fields_deg2["surface"])
        height_above_flotation = firedrake.assemble(
            interpolate(s2d - (1 - ρ_I / ρ_W) * extract_surface(fields_deg2["thickness"]), s2d.function_space())
        )
        levels = [5]
        tricontour(height_above_flotation, levels=levels, axes=axes, colors=["k"])
        colorbar(fig, colors, label="H [m]")
        axes.set_xlim(4e5, 5e5)
        fig.savefig("plots/init/hybrid/initialized_mismip_thickness.png", dpi=300)

deg4_fn = "inputs/mismip-fine-degree4.h5"
fields_4 = {}
if os.path.exists(deg4_fn):
    PETSc.Sys.Print("Reloading degree 4 results...")
    with firedrake.CheckpointFile(deg4_fn, "r") as chk:
        fine_mesh = chk.load_mesh("fine_mesh")
        time_now = chk.get_attr("metadata", "total_time")
        for key in field_names:
            fields_4[key] = chk.load_function(fine_mesh, key)

    Q2 = firedrake.FunctionSpace(fine_mesh, "CG", 2, vfamily="R", vdegree=0)
    V2 = firedrake.VectorFunctionSpace(fine_mesh, "CG", 2, dim=2, vfamily="GL", vdegree=2)
    V4 = firedrake.VectorFunctionSpace(fine_mesh, "CG", 2, dim=2, vfamily="GL", vdegree=4)
    x = firedrake.SpatialCoordinate(fine_mesh)[0]
    u_init = firedrake.assemble(interpolate(0.9 * fields_4["velocity"], V4))

    solver = icepack.solvers.FlowSolver(model, **fast_opts)

    z_b = firedrake.assemble(interpolate(mismip_bed_topography(fine_mesh), Q2))

    solver = icepack.solvers.FlowSolver(model, **par_opts)
    PETSc.Sys.Print("Re-initializing velocity")

    fields_complete = {
        "surface": firedrake.Function(Q2).interpolate(fields_4["surface"]),
        "thickness": firedrake.Function(Q2).interpolate(fields_4["thickness"]),
    }
    u_0 = solver.diagnostic_solve(
        velocity=u_init,
        thickness=fields_complete["thickness"],
        surface=fields_complete["surface"],
        fluidity=A,
        friction=C,
    )
    fields_complete["velocity"] = u_0
else:
    PETSc.Sys.Print("Re-initializing velocity")
    u_0 = solver.diagnostic_solve(
        velocity=u_init,
        thickness=fields_deg2["thickness"],
        surface=fields_deg2["surface"],
        fluidity=A,
        friction=C,
    )

    solver = icepack.solvers.FlowSolver(model, **fast_opts)

    fields_complete = {
        "surface": fields_deg2["surface"],
        "thickness": fields_deg2["thickness"],
        "velocity": u_0,
    }

if False:
    final_time = 500.0
    increment = 250
    dt = 1.0
    for i in range(int(final_time / increment)):
        PETSc.Sys.Print(
            "Running and saving years {:d} to {:d}".format(
                int(i * increment + time_now), int(i + 1) * increment + int(time_now)
            )
        )
        exception, fields_complete = run_simulation(
            solver, increment, dt, bed=z_b, a=a, C=C, A=A, plot=False, **fields_complete
        )

        if exception:
            with firedrake.CheckpointFile("inputs/mismip-fine-degree4_err.h5", "w") as chk:
                chk.create_group("metadata")
                chk.set_attr("metadata", "total_time", increment * (i + 1) + time_now)
                chk.save_mesh(fine_mesh)
                for key in fields_complete:
                    chk.save_function(fields_complete[key], name=key)
            raise exception

        with firedrake.CheckpointFile("inputs/mismip-fine-degree4.h5", "w") as chk:
            chk.create_group("metadata")
            chk.set_attr("metadata", "total_time", increment * (i + 1) + time_now)
            chk.save_mesh(fine_mesh)
            for key in fields_complete:
                chk.save_function(fields_complete[key], name=key)

        fig, axes = subplots(figsize=(14, 6))
        colors = tripcolor(extract_surface(fields_complete["thickness"]), axes=axes)
        s2d = extract_surface(fields_complete["surface"])
        height_above_flotation = firedrake.assemble(
            interpolate(s2d - (1 - ρ_I / ρ_W) * extract_surface(fields_complete["thickness"]), s2d.function_space())
        )
        levels = [5]
        tricontour(height_above_flotation, levels=levels, axes=axes, colors=["k"])
        colorbar(fig, colors, label="H [m]")
        axes.set_xlim(4e5, 5e5)
        fig.savefig(
            "plots/init/hybrid/initialized_d4_mismip_thickness_{:d}.png".format(int(increment * (i + 1) + time_now)),
            dpi=300,
        )

    time_now += final_time


final_time = 500.0
increment = 100
dt = 1.0 / 4.0
for i in range(int(final_time / increment)):
    PETSc.Sys.Print(
        "Running and saving years {:d} to {:d}".format(
            int(i * increment + time_now), int(i + 1) * increment + int(time_now)
        )
    )
    exception, fields_complete = run_simulation(
        solver,
        increment,
        dt,
        bed=z_b,
        a=a,
        C=C,
        A=A,
        plot=False,
        cutoff_dV=0.01,
        cutoff_dh=0.001,
        min_step=100,
        **fields_complete,
    )

    if exception:
        with firedrake.CheckpointFile("inputs/mismip-fine-degree4_err.h5", "w") as chk:
            chk.create_group("metadata")
            chk.set_attr("metadata", "total_time", increment * (i + 1) + time_now)
            chk.save_mesh(fine_mesh)
            for key in fields_complete:
                chk.save_function(fields_complete[key], name=key)
        raise exception

    with firedrake.CheckpointFile("inputs/mismip-fine-degree4.h5", "w") as chk:
        chk.create_group("metadata")
        chk.set_attr("metadata", "total_time", increment * (i + 1) + time_now)
        chk.save_mesh(fine_mesh)
        for key in fields_complete:
            chk.save_function(fields_complete[key], name=key)

    fig, axes = subplots(figsize=(14, 6))
    colors = tripcolor(extract_surface(fields_complete["thickness"]), axes=axes)
    s2d = extract_surface(fields_complete["surface"])
    height_above_flotation = firedrake.assemble(
        interpolate(s2d - (1 - ρ_I / ρ_W) * extract_surface(fields_complete["thickness"]), s2d.function_space())
    )
    levels = [5]
    tricontour(height_above_flotation, levels=levels, axes=axes, colors=["k"])
    colorbar(fig, colors, label="H [m]")
    axes.set_xlim(4e5, 5e5)
    fig.savefig(
        "plots/init/hybrid/initialized_d4_mismip_thickness_{:d}.png".format(int(increment * (i + 1) + time_now)),
        dpi=300,
    )

#!/usr/bin/env python
# coding: utf-8

import sys
import firedrake
import icepack
import icepack.models.hybrid
import numpy as np
import matplotlib.pyplot as plt
from firedrake.__future__ import interpolate
from firedrake import inner, sqrt
import tqdm
from icepack.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
    gravity as g,
)
from icepackaccs import rate_factor, extract_surface, extract_bed
import pandas as pd
import os
import xarray as xr


RdPu = ["#fbb4b9", "#f768a1", "#c51b8a", "#7a0177"]
Blues = ["#6baed6", "#3182bd", "#08519c"]
Greens = ["#74c476", "#31a354", "#006d2c"]
Reds = ["#fb6a4a", "#de2d26", "#a50f15"]
Purples = ["#9e9ac8", "#756bb1", "#54278f"]
Oranges = ["#fd8d3c", "#e6550d", "#a63603"]
PuBuGn = ["#67a9cf", "#1c9099", "#016c59"]
BuPr = ["#8c96c6", "#8856a7", "#810f7c"]
YlGn = ["#c2e699", "#78c679", "#238443"]
Yellows = ["#FFF394", "#FFE761", "#FFDC2E"]

CW1 = ["#ffff2d", "#bac317", "#7b8900"]  # Yellow
CW2 = ["#f6c137", "#d09c1f", "#aa7900"]  # Light Orange
CW3 = ["#f69537", "#e67a1e", "#d55e00"]  # Dark orange
CW4 = ["#ef3c42", "#a51e23", "#610004"]  # Cranberry
CW5 = ["#b528c5", "#7f168a", "#4c0554"]  # Purple
CW6 = ["#3438bd", "#1a1c81", "#00014a"]  # Royal blues
# CW7 = ["#4592ca", "#006bc0", "#0040ac"]  # Lighter blues
CW7 = ["#52c67f", "#2c803d", "#004100"]  # BlueGreens
CW8 = ["#79c725", "#387f14", "#003d00"]  # True Greens

CW1 = ["#FFA07A", "#FF8C00", "#FF2400"]
CW2 = ["#FF6347", "#FF0000", "#6B0000"]
CW3 = ["#ADD8E6", "#4682B4", "#00008B"]
CW4 = ["#00BFFF", "#1E90FF", "#0E4C92"]
CW5 = ["#FF77FF", "#FF00FF", "#990099"]
CW6 = ["#DDA0DD", "#BA55D3", "#4B0082"]
CW7 = ["#98FB98", "#32CD32", "#004D00"]
CW8 = ["#9ACD32", "#6B8E23", "#405D20"]


tripcolors1 = [Oranges[0], Greens[0], Blues[0], Oranges[1], Greens[1], Blues[1], Oranges[2], Greens[2], Blues[2]]

tripcolors2 = [Reds[0], PuBuGn[0], Purples[0], Reds[1], PuBuGn[1], Purples[1], Reds[2], PuBuGn[2], Purples[2]]

tripcolors3 = [RdPu[0], YlGn[0], BuPr[0], RdPu[1], YlGn[1], BuPr[1], RdPu[2], YlGn[2], BuPr[2]]

color_dict = {
    "identical": {
        1.8: {
            -20: CW1[0],
            -10: CW1[1],
            -5: CW1[2],
        },
        3.0: {-20: CW3[0], -10: CW3[1], -5: CW3[2]},
        3.5: {
            -20: CW5[0],
            -10: CW5[1],
            -5: CW5[2],
        },
        4.0: {-20: CW7[0], -10: CW7[1], -5: CW7[2]},
    },
    "parinferred": {
        1.8: {
            0.75: CW1[0],
            1.0: CW1[1],
            1.25: CW1[2],
        },
        3.0: {0.75: CW3[0], 1.0: CW3[1], 1.25: CW3[2]},
        3.5: {
            0.75: CW5[0],
            1.0: CW5[1],
            1.25: CW5[2],
        },
        4.0: {0.75: CW7[0], 1.0: CW7[1], 1.25: CW7[2]},
    },
    "inferred": {
        1.8: {
            -12: CW2[0],
            -10: CW2[1],
            -8: CW2[2],
        },
        3.0: {-12: CW4[0], -10: CW4[1], -8: CW4[2]},
        3.5: {
            -12: CW6[0],
            -10: CW6[1],
            -8: CW6[2],
        },
        4.0: {-12: CW8[0], -10: CW8[1], -8: CW8[2]},
    },
    "true": {3.0: {-20: "k", -10: "k", -5: "k"}},
}


a_0 = firedrake.Constant(1.0)
δa = firedrake.Constant(2.3529411764705883)
u0_coulomb = 50


def main(nbumps=1):
    T_C = -10
    if len(sys.argv) > 1:
        T_C = int(sys.argv[1])

    # Are we starting from scratch?
    do_2 = False
    cache_fn = "inputs/flowline_n3_{:03d}C_weertman3_bumps{:1d}.h5".format(T_C, nbumps)
    if not os.path.exists(cache_fn):
        cache_fn = "inputs/flowline_n3_{:03d}C_weertman3_smooth.h5".format(T_C)
        do_2 = True

    fac = 100  # Set this based on characteristic timescale
    num_years = 200 * fac
    num_years2 = 250 * fac

    if T_C > -10:
        timesteps_per_year_2 = 2
    else:
        timesteps_per_year_2 = 2

    # For round 1 we will need shorter timesteps just for the warm case, to get steady state be consistently fine
    if do_2:
        if T_C > -10:
            timesteps_per_year_4 = 4
        else:
            timesteps_per_year_4 = 2
    else:
        num_years2 = 100 * fac
        timesteps_per_year_4 = 8

    timesteps_per_year_final = 8

    def volume(thickness):
        return firedrake.assemble(thickness * firedrake.dx)

    min_thick = firedrake.Constant(10.0)
    Lx = 500e3
    nx = 2001
    mesh1d = firedrake.IntervalMesh(nx, Lx)
    mesh = firedrake.ExtrudedMesh(mesh1d, layers=1)

    Q = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="R", vdegree=0)
    V = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="GL", vdegree=2)

    V_8 = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="GL", vdegree=8)

    x, ζ = firedrake.SpatialCoordinate(mesh)

    a = firedrake.Function(Q).interpolate(a_0 - δa * x / Lx)

    field_names = ["surf", "bed", "thick", "u2", "u4"]

    T_K = 273.15 + T_C

    b_in, b_out = 3500, 500
    b = firedrake.Function(Q).interpolate(b_in - (b_in - b_out) * x / Lx)
    s_in, s_out = 5500, 700

    profile_fn = "rough_profile.csv"

    if os.path.exists(cache_fn):
        with firedrake.CheckpointFile(cache_fn, "r") as chk:
            mesh_cache = chk.load_mesh("flowline")
            # start_time = chk.get_attr("metadata", "total_time")
            fields = {name: chk.load_function(mesh_cache, name) for name in field_names}
            h0 = firedrake.Function(Q).interpolate(fields["thick"], allow_missing_dofs=True)
            s0 = firedrake.Function(Q).interpolate(fields["surf"], allow_missing_dofs=True)
            h0 = firedrake.assemble(interpolate(firedrake.max_value(h0, min_thick), Q))
            s0 = icepack.compute_surface(thickness=h0, bed=b)
    elif os.path.exists(profile_fn):
        df = pd.read_csv(profile_fn)
        hin = np.vstack((df["H"], df["H"]))
        harr = xr.DataArray(hin, coords={"y": np.array([-1e3, 1e4]), "x": df["x"]})
        h0 = icepack.interpolate(harr, Q)
        sin = np.vstack((df["s"], df["s"]))
        sarr = xr.DataArray(sin, coords={"y": np.array([-1e3, 1e4]), "x": df["x"]})
        s0 = icepack.interpolate(sarr, Q)
    else:
        s0 = firedrake.Function(Q).interpolate(
            firedrake.max_value(s_in - (s_in - s_out) * ((x + 1.0e4) / Lx) ** 2.0, b + min_thick)
        )
        h0 = firedrake.Function(Q).interpolate(s0 - b)
        s0 = icepack.compute_surface(thickness=h0, bed=b)

    h_in = s_in - b_in
    δs_δx = (s_out - s_in) / Lx
    τ_D = -ρ_I * g * h_in * δs_δx

    u_in, u_out = 0, 100
    velocity_x = u_in + (u_out - u_in) * (x / Lx) ** 2
    u0 = firedrake.Function(V).interpolate(velocity_x)

    T = firedrake.Constant(T_K)
    A3 = rate_factor(T)

    # C3 = firedrake.Function(Q).interpolate(firedrake.sqrt(25.0 * τ_D / (u_in + 1)**(1 / m) * (firedrake.cos(x * 2 * np.pi * nbumps / Lx) + 1) / 3))
    if nbumps == 2:
        C3 = firedrake.Function(Q).interpolate(
            firedrake.sqrt(
                0.2
                * τ_D
                * (
                    1
                    + 3.0
                    * (
                        firedrake.exp(-((x - 1e5) ** 2.0) / 10e3**2.0)
                        + firedrake.exp(-((x - 2.5e5) ** 2.0) / 15e3**2.0)
                    )
                )
            )
        )
    elif nbumps == 1:
        C3 = firedrake.Function(Q).interpolate(
            firedrake.sqrt(0.2 * τ_D * (1 + 4 / (1 + firedrake.exp((x - 200e3) / 1e4))))
        )

    fig, axes = plt.subplots()
    axes.set_xlabel("distance along centerline (m)")
    firedrake.plot(icepack.depth_average(C3), edgecolor="tab:brown", axes=axes)

    def friction(**kwargs):
        u = kwargs["velocity"]
        h = kwargs["thickness"]
        s = kwargs["surface"]
        C = kwargs["friction"]

        p_W = ρ_W * g * firedrake.max_value(0, h - s)
        p_I = ρ_I * g * h
        ϕ = 1 - p_W / p_I
        return icepack.models.hybrid.bed_friction(
            velocity=u,
            friction=C**2.0 * ϕ,
        )

    model = icepack.models.HybridModel(friction=friction)
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
    solver = icepack.solvers.FlowSolver(model, **opts)
    solver4 = icepack.solvers.FlowSolver(model, **opts)

    fig, axes = plt.subplots()
    axes.set_xlabel("distance along centerline (m)")
    axes.set_ylabel("elevation (m)")
    firedrake.plot(icepack.depth_average(b), edgecolor="tab:brown", axes=axes)
    z_b = firedrake.Function(Q).interpolate(s0 - h0)
    firedrake.plot(icepack.depth_average(z_b), edgecolor="tab:blue", axes=axes)
    firedrake.plot(icepack.depth_average(s0), edgecolor="tab:blue", axes=axes)

    u0 = solver.diagnostic_solve(
        velocity=u0, thickness=h0, surface=s0, fluidity=A3, friction=C3, flow_law_exponent=firedrake.Constant(3)
    )

    fig, axes = plt.subplots()
    axes.set_xlabel("distance along centerline (m)")
    firedrake.plot(icepack.depth_average(u0), edgecolor="tab:brown", axes=axes)

    h = h0.copy(deepcopy=True)
    s = s0.copy(deepcopy=True)
    u = u0.copy(deepcopy=True)

    if do_2:

        δt = 1.0 / timesteps_per_year_2
        num_timesteps = num_years * timesteps_per_year_2

        progress_bar = tqdm.trange(num_timesteps)
        description = f"dV,max(abs(dH)): {(0 - 0) / δt:4.2f} [m2/yr] {(0) / δt:4.3f} [m/yr]"
        progress_bar.set_description(description)
        for step in progress_bar:
            h_prev = h.copy(deepcopy=True)
            h = solver.prognostic_solve(
                δt,
                thickness=h,
                velocity=u,
                accumulation=a,
                thickness_inflow=h0,
            )
            h = firedrake.assemble(interpolate(firedrake.max_value(h, min_thick), Q))
            s = icepack.compute_surface(thickness=h, bed=b)
            u = solver.diagnostic_solve(
                velocity=u, thickness=h, surface=s, fluidity=A3, friction=C3, flow_law_exponent=firedrake.Constant(3)
            )
            dh = firedrake.assemble(interpolate(h - h_prev, h.function_space()))
            H_v_t = volume(h)
            H_v_prev = volume(h_prev)
            description = f"dV,max(abs(dH)): {(H_v_t - H_v_prev) / δt:4.2f} [m3/yr] {(np.abs(dh.dat.data_ro).max()) / δt:4.3f} [m/yr]"
            progress_bar.set_description(description)

    fig, axes = plt.subplots()
    axes.set_ylabel("ice speed (m / yr)")
    firedrake.plot(icepack.depth_average(u), axes=axes)

    fig, axes = plt.subplots()
    axes.set_ylabel("acc (m / yr)")
    firedrake.plot(icepack.depth_average(a), axes=axes)

    f = icepack.depth_average(firedrake.Function(Q).project(a - (h * u).dx(0)))
    fig, axes = plt.subplots()
    axes.set_ylim(-0.1, +0.1)
    axes.set_ylabel("meters / year")
    firedrake.plot(f, axes=axes)

    fig, axes = plt.subplots()
    axes.set_xlabel("distance along centerline (m)")
    axes.set_ylabel("elevation (m)")
    firedrake.plot(icepack.depth_average(b), edgecolor="tab:brown", axes=axes)
    z_b = firedrake.Function(Q).interpolate(s - h)
    firedrake.plot(icepack.depth_average(z_b), edgecolor="tab:blue", axes=axes)
    firedrake.plot(icepack.depth_average(s), edgecolor="tab:blue", axes=axes)

    u_shear = icepack.depth_average(u, weight=np.sqrt(3) * (2 * ζ - 1))

    fig, axes = plt.subplots()
    axes.set_title("Shear component of ice velocity")
    axes.set_ylabel("velocity (m / yr)")
    firedrake.plot(u_shear, axes=axes)

    u_avg = icepack.depth_average(u)

    U_shear = sqrt(inner(u_shear, u_shear))
    U_avg = sqrt(inner(u_avg, u_avg))
    V2D = u_avg.function_space()
    ratio = firedrake.Function(V2D).interpolate(U_shear / U_avg)

    fig, axes = plt.subplots()
    axes.set_title("Ratio of shear / plug flow")
    firedrake.plot(ratio, axes=axes)

    u_4 = firedrake.Function(V_8).interpolate(u)
    u_4 = solver4.diagnostic_solve(
        velocity=u_4, thickness=h, surface=s, fluidity=A3, friction=C3, flow_law_exponent=firedrake.Constant(3)
    )
    fields_n3 = {"surf": s, "bed": b, "thick": h, "u2": u, "u4": u_4, "C": C3}
    with firedrake.CheckpointFile(
        "inputs/flowline_n3_{:03d}C_weertman3_bumps{:1d}.h5".format(int(T.dat.data_ro[0] - 273.15), nbumps), "w"
    ) as chk:
        chk.create_group("metadata")
        # chk.set_attr("metadata", "total_time", num_years + num_years2 + start_time)
        mesh.name = "flowline"
        chk.save_mesh(mesh)
        for key in fields_n3:
            chk.save_function(fields_n3[key], name=key)

    if do_2:
        δt = 1.0 / timesteps_per_year_4
        num_timesteps = num_years2 * timesteps_per_year_4
        progress_bar = tqdm.trange(num_timesteps)
        description = f"dV,max(abs(dH)): {(0 - 0) / δt:4.2f} [m2/yr] {(0) / δt:4.3f} [m/yr]"
        progress_bar.set_description(description)
        for step in progress_bar:
            h_prev = h.copy(deepcopy=True)
            h = solver4.prognostic_solve(
                δt,
                thickness=h,
                velocity=u_4,
                accumulation=a,
                thickness_inflow=h0,
            )
            h = firedrake.assemble(interpolate(firedrake.max_value(h, min_thick), Q))

            s = icepack.compute_surface(thickness=h, bed=b)
            u_4 = solver4.diagnostic_solve(
                velocity=u_4,
                thickness=h,
                surface=s,
                fluidity=A3,
                friction=C3,
                flow_law_exponent=firedrake.Constant(3),
            )
            dh = firedrake.assemble(interpolate(h - h_prev, h.function_space()))
            H_v_t = volume(h)
            H_v_prev = volume(h_prev)
            description = f"dV,max(abs(dH)): {(H_v_t - H_v_prev) / δt:4.2f} [m2/yr] {(np.abs(dh.dat.data_ro).max()) / δt:4.3f} [m/yr]"
            progress_bar.set_description(description)

    u = solver.diagnostic_solve(
        velocity=u, thickness=h, surface=s, fluidity=A3, friction=C3, flow_law_exponent=firedrake.Constant(3)
    )

    f = icepack.depth_average(firedrake.Function(Q).project(a - (h * u_4).dx(0)))
    fig, axes = plt.subplots()
    axes.set_ylim(-0.1, +0.1)
    axes.set_ylabel("meters / year")
    firedrake.plot(f, axes=axes)
    fig.savefig(
        "figs/dhdt_flowline_n3_{:03d}C_weertman3_bumps{:1d}.pdf".format(int(T.dat.data_ro[0] - 273.15), nbumps), dpi=300
    )

    u_shear4 = icepack.depth_average(u_4, weight=np.sqrt(3) * (2 * ζ - 1))

    fig, axes = plt.subplots()
    axes.set_ylabel("velocity (m / yr)")
    firedrake.plot(icepack.depth_average(u_4), axes=axes, label="Depth averaged")
    firedrake.plot(u_shear4, axes=axes, label="Shear")
    firedrake.plot(extract_surface(u_4), axes=axes, label="Surface")
    firedrake.plot(extract_bed(u_4), axes=axes, label="Sliding")
    axes.legend(loc="best")
    fig.savefig(
        "figs/vel_flowline_n3_{:03d}C_weertman3_bumps{:1d}.pdf".format(int(T.dat.data_ro[0] - 273.15), nbumps), dpi=300
    )

    u_avg4 = icepack.depth_average(u_4)

    U_shear4 = sqrt(inner(u_shear4, u_shear4))
    U_avg4 = sqrt(inner(u_avg4, u_avg4))
    V2D = u_avg4.function_space()
    ratio4 = firedrake.Function(V2D).interpolate(
        U_shear4 / U_avg4 * icepack.depth_average(firedrake.Function(Q).interpolate((h > min_thick)))
    )

    fig, axes = plt.subplots()
    axes.set_title("Ratio of shear / plug flow")
    firedrake.plot(ratio4, axes=axes)
    axes.set_ylim(0, 1)
    fig.savefig(
        "figs/flowratio_flowline_n3_{:03d}C_weertman3_bumps{:1d}.pdf".format(int(T.dat.data_ro[0] - 273.15), nbumps),
        dpi=300,
    )

    fig, axes = plt.subplots()
    axes.set_xlabel("distance along centerline (m)")
    axes.set_ylabel("elevation (m)")
    firedrake.plot(icepack.depth_average(b), edgecolor="tab:brown", axes=axes)
    z_b = firedrake.Function(Q).interpolate(s - h)
    firedrake.plot(icepack.depth_average(z_b), edgecolor="tab:blue", axes=axes)
    firedrake.plot(icepack.depth_average(s), edgecolor="tab:blue", axes=axes)
    fig.savefig(
        "figs/profile_flowline_n3_{:03d}C_weertman3_bumps{:1d}.pdf".format(int(T.dat.data_ro[0] - 273.15), nbumps),
        dpi=300,
    )

    fields_n3 = {"surf": s, "bed": b, "thick": h, "u2": u, "u4": u_4, "C": C3}
    with firedrake.CheckpointFile(
        "inputs/flowline_n3_{:03d}C_weertman3_bumps{:1d}.h5".format(int(T.dat.data_ro[0] - 273.15), nbumps), "w"
    ) as chk:
        chk.create_group("metadata")
        # chk.set_attr("metadata", "total_time", num_years + num_years2 + start_time)
        mesh.name = "flowline"
        chk.save_mesh(mesh)
        for key in fields_n3:
            chk.save_function(fields_n3[key], name=key)

    δt = 1.0 / timesteps_per_year_final
    num_timesteps = num_years2 * timesteps_per_year_final

    dhm = 1.0
    dvm = 1.0
    i = 0
    miniter = 100  # prevents stopping early since change has not started
    while abs(dhm) > 0.001 or abs(dvm) > 0.001 or i < miniter:
        h_prev = h.copy(deepcopy=True)
        h = solver4.prognostic_solve(
            δt,
            thickness=h,
            velocity=u_4,
            accumulation=a,
            thickness_inflow=h0,
        )
        h = firedrake.assemble(interpolate(firedrake.max_value(h, min_thick), Q))

        s = icepack.compute_surface(thickness=h, bed=b)
        u_4 = solver4.diagnostic_solve(
            velocity=u_4,
            thickness=h,
            surface=s,
            fluidity=A3,
            friction=C3,
            flow_law_exponent=firedrake.Constant(3),
        )
        dh = firedrake.assemble(interpolate(h - h_prev, h.function_space()))
        dhm = dh.dat.data_ro[np.argmax(np.abs(dh.dat.data_ro))]
        H_v_t = volume(h)
        H_v_prev = volume(h_prev)
        dvm = (H_v_t - H_v_prev) / δt
        print("Iteration {:8d}... dV: {:8.3f}, dH: {:5.3f}".format(i, dvm, dhm), end="\r")
        i += 1

    u = solver.diagnostic_solve(
        velocity=u, thickness=h, surface=s, fluidity=A3, friction=C3, flow_law_exponent=firedrake.Constant(3)
    )

    f = icepack.depth_average(firedrake.Function(Q).project(a - (h * u_4).dx(0)))
    fig, axes = plt.subplots()
    axes.set_ylim(-0.1, +0.1)
    axes.set_ylabel("meters / year")
    firedrake.plot(f, axes=axes)
    fig.savefig(
        "figs/dhdt_flowline_n3_{:03d}C_weertman3_bumps{:1d}.pdf".format(int(T.dat.data_ro[0] - 273.15), nbumps), dpi=300
    )

    u_shear4 = icepack.depth_average(u_4, weight=np.sqrt(3) * (2 * ζ - 1))

    fig, axes = plt.subplots()
    axes.set_ylabel("velocity (m / yr)")
    firedrake.plot(icepack.depth_average(u_4), axes=axes, label="Depth averaged")
    firedrake.plot(u_shear4, axes=axes, label="Shear")
    firedrake.plot(extract_surface(u_4), axes=axes, label="Surface")
    firedrake.plot(extract_bed(u_4), axes=axes, label="Sliding")
    axes.legend(loc="best")
    fig.savefig(
        "figs/vel_flowline_n3_{:03d}C_weertman3_bumps{:1d}.pdf".format(int(T.dat.data_ro[0] - 273.15), nbumps), dpi=300
    )

    u_avg4 = icepack.depth_average(u_4)

    U_shear4 = sqrt(inner(u_shear4, u_shear4))
    U_avg4 = sqrt(inner(u_avg4, u_avg4))
    V2D = u_avg4.function_space()
    ratio4 = firedrake.Function(V2D).interpolate(
        U_shear4 / U_avg4 * icepack.depth_average(firedrake.Function(Q).interpolate((h > min_thick)))
    )

    fig, axes = plt.subplots()
    axes.set_title("Ratio of shear / plug flow")
    firedrake.plot(ratio4, axes=axes)
    axes.set_ylim(0, 1)
    fig.savefig(
        "figs/flowratio_flowline_n3_{:03d}C_weertman3_bumps{:1d}.pdf".format(int(T.dat.data_ro[0] - 273.15), nbumps),
        dpi=300,
    )

    fig, axes = plt.subplots()
    axes.set_xlabel("distance along centerline (m)")
    axes.set_ylabel("elevation (m)")
    firedrake.plot(icepack.depth_average(b), edgecolor="tab:brown", axes=axes)
    z_b = firedrake.Function(Q).interpolate(s - h)
    firedrake.plot(icepack.depth_average(z_b), edgecolor="tab:blue", axes=axes)
    firedrake.plot(icepack.depth_average(s), edgecolor="tab:blue", axes=axes)
    fig.savefig(
        "figs/profile_flowline_n3_{:03d}C_weertman3_bumps{:1d}.pdf".format(int(T.dat.data_ro[0] - 273.15), nbumps),
        dpi=300,
    )

    fields_n3 = {"surf": s, "bed": b, "thick": h, "u2": u, "u4": u_4, "C": C3}
    with firedrake.CheckpointFile(
        "inputs/flowline_n3_{:03d}C_weertman3_bumps{:1d}.h5".format(int(T.dat.data_ro[0] - 273.15), nbumps), "w"
    ) as chk:
        chk.create_group("metadata")
        # chk.set_attr("metadata", "total_time", num_years + num_years2 + start_time)
        mesh.name = "flowline"
        chk.save_mesh(mesh)
        for key in fields_n3:
            chk.save_function(fields_n3[key], name=key)


if __name__ == "__main__":
    main()

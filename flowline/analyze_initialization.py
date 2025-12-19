#!/usr/bin/env python
# coding: utf-8

import numpy as np
import firedrake
import icepack
import icepack.models.hybrid
import matplotlib.pyplot as plt
from icepackaccs import extract_surface, extract_bed
from icepackaccs.friction import get_weertman, get_regularized_coulomb_simp, friction_stress
from transient_simulations import a_unpert, a_pert, a_retreat

# Need to muck around to use color consistently outside a package
import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from common_colors import color_dict_T

ls_dict = {-12: "dotted", -10: "solid", -8: "dashed"}

def kmfmt(x, pos):
    return "{:d}".format(int(x / 1000.0))


def volume(thickness):
    return firedrake.assemble(thickness * firedrake.dx)


def taub_rcfi(beta, u, u0):
    return firedrake.Function(u.function_space()).interpolate(
        beta * u ** (1 / 3) / ((abs(u)) ** (1.0 / 3.0 + 1) + u0 ** (1.0 / 3.0 + 1)) ** (1.0 / (3.0 + 1))
    )


min_thick = firedrake.Constant(10.0)
Lx = 500e3
nx = 2001

# Without these we get a mysterious error on new firedrake installs
mesh1d = firedrake.IntervalMesh(nx, Lx)
mesh_dum = firedrake.ExtrudedMesh(mesh1d, layers=1)

plot_x = np.linspace(0, 425e3, 500)


for nbumps in [2, 1]:
    checkpoint_fn = "inputs/standard_init_bumps{:d}.h5".format(nbumps)
    with firedrake.CheckpointFile(checkpoint_fn, "r") as chk:
        mesh = chk.load_mesh("flowline")

    Q = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="R", vdegree=0)
    V = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="GL", vdegree=2)
    V_2 = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="GL", vdegree=2)

    V_8 = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="GL", vdegree=8)

    x, ζ = firedrake.SpatialCoordinate(mesh)

    a = firedrake.Function(Q).interpolate(a_unpert)
    a_pert = firedrake.Function(Q).interpolate(a_pert)
    a_retreat = firedrake.Function(Q).interpolate(a_retreat)

    cache_fn = "inputs/flowline_n3_{:03d}C_weertman3_bumps{:1d}.h5".format(-10, nbumps)
    with firedrake.CheckpointFile(cache_fn, "r") as chk:
        field_names = ["surf", "bed", "thick", "u2", "u4", "C"]
        mesh_cache = chk.load_mesh("flowline")
        # start_time = chk.get_attr("metadata", "total_time")
        fields = {name: chk.load_function(mesh_cache, name) for name in field_names}
        h0_10 = firedrake.Function(Q).interpolate(fields["thick"])
        s0_10 = firedrake.Function(Q).interpolate(fields["surf"])
        u0_10 = firedrake.Function(V_8).interpolate(fields["u4"])
        C0 = firedrake.Function(Q).interpolate(fields["C"])
        b = firedrake.Function(Q).interpolate(fields["bed"])

    us_10 = extract_surface(u0_10)

    cache_fn = "inputs/flowline_n3_{:03d}C_weertman3_bumps{:1d}.h5".format(-20, nbumps)
    with firedrake.CheckpointFile(cache_fn, "r") as chk:
        field_names = ["surf", "bed", "thick", "u2", "u4", "C"]
        mesh_cache = chk.load_mesh("flowline")
        # start_time = chk.get_attr("metadata", "total_time")
        fields = {name: chk.load_function(mesh_cache, name) for name in field_names}
        h0_20 = firedrake.Function(Q).interpolate(fields["thick"])
        s0_20 = firedrake.Function(Q).interpolate(fields["surf"])
        u0_20 = firedrake.Function(V_8).interpolate(fields["u4"])

    cache_fn = "inputs/flowline_n3_{:03d}C_weertman3_bumps{:1d}.h5".format(-5, nbumps)
    with firedrake.CheckpointFile(cache_fn, "r") as chk:
        field_names = ["surf", "bed", "thick", "u2", "u4", "C"]
        mesh_cache = chk.load_mesh("flowline")
        # start_time = chk.get_attr("metadata", "total_time")
        fields = {name: chk.load_function(mesh_cache, name) for name in field_names}
        h0_5 = firedrake.Function(Q).interpolate(fields["thick"])
        s0_5 = firedrake.Function(Q).interpolate(fields["surf"])
        u0_5 = firedrake.Function(V_8).interpolate(fields["u4"])

    da_interper = firedrake.PointEvaluator(mesh, np.vstack((plot_x, np.ones_like(plot_x) * 0.5)).T)
    bed_interper = firedrake.PointEvaluator(mesh, np.vstack((plot_x, np.zeros_like(plot_x))).T)
    surf_interper = firedrake.PointEvaluator(mesh, np.vstack((plot_x, np.ones_like(plot_x))).T)
    mesh_x = extract_bed(h0_10).ufl_domain()
    interper = firedrake.PointEvaluator(mesh_x, plot_x)
    Q1d = extract_bed(h0_10).function_space()

    def da(obj):
        return da_interper.evaluate(firedrake.Function(V_8).interpolate(obj))

    def get_at_surf(obj):
        return surf_interper.evaluate(firedrake.Function(V_8).interpolate(obj))

    def get_at_bed(obj):
        return interper.evaluate(extract_bed(obj))

    def get_taub(u, C):
        return interper.evaluate(firedrake.Function(Q1d).interpolate(-extract_bed(friction_stress(u, C**2.0, m=3)) * 1000))

    def get_sliding(u):
        sliding = extract_bed(firedrake.Function(V_8).interpolate(u * 100)) / extract_surface(firedrake.Function(V_8).interpolate(u))
        return interper.evaluate(firedrake.Function(Q1d).interpolate(sliding))

    Ts = [-12, -10, -8]
    ns = [1.8, 3.0, 3.5, 4.0]
    devs = [0.75, 1.0, 1.25]

    u0_coulomb = 300
    regularized_coulomb = get_regularized_coulomb_simp(m=3, u_0=u0_coulomb)
    weertman_3 = get_weertman(m=3)
    weertman_1 = get_weertman(m=1)

    done = []
    output_dict = {T: {} for T in Ts}
    for dev in devs:
        output_dict[dev] = {}

    with firedrake.CheckpointFile(checkpoint_fn, "r") as chk:
        for T_np in Ts:
            for n in ns:
                output_dict[T_np][n] = {}

                inv_name = "T{:d}_n{:2.1f}".format(T_np, n)

                if chk.has_attr("already_run", inv_name) and chk.get_attr("already_run", inv_name):
                    done.append(inv_name)

                    output_dict[T_np][n]["C1"] = chk.load_function(mesh, inv_name + "_C1")
                    output_dict[T_np][n]["C3"] = chk.load_function(mesh, inv_name + "_C3")
                    output_dict[T_np][n]["CRCFi"] = chk.load_function(mesh, inv_name + "_CRCFi")
                    output_dict[T_np][n]["u1"] = chk.load_function(mesh, inv_name + "_u1")
                    output_dict[T_np][n]["u3"] = chk.load_function(mesh, inv_name + "_u3")
                    output_dict[T_np][n]["uRCFi"] = chk.load_function(mesh, inv_name + "_uRCFi")

                    du = us_10 - firedrake.Function(us_10.function_space()).interpolate(extract_surface(output_dict[T_np][n]["u1"]))
                    print("{:s}: RMS: {:f} m/yr, max v: {:f} m/yr".format(inv_name, (firedrake.assemble(du **2 * firedrake.dx) / firedrake.assemble(firedrake.conditional(extract_surface(h0_10) > 10.0, 1.0, 0.0) * firedrake.dx)) ** 0.5, np.max(output_dict[T_np][n]["u1"].dat.data[:])))

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(7.0, 5.0), gridspec_kw={"height_ratios": (2, 1, 1, 1)})
    axes[0].set_xlim(0, 500e3)
    axes[-1].xaxis.set_major_formatter(kmfmt)
    axes[-1].set_xlabel("Distance (km)")
    axes[1].set_ylabel("Speed (m yr$^{-1}$)")
    axes[1].set_ylim(0, 125)

    axes[0].plot(
        plot_x,
        da(b + h0_20 * (h0_20 > 10)),
        color=color_dict_T["identical"][3.0][-20],
        label=r"T=-20 $^\circ$C",
    )
    axes[0].plot(
        plot_x,
        da(b + h0_10 * (h0_10 > 10)),
        color=color_dict_T["identical"][3.0][-10],
        label=r"T=-10 $^\circ$C",
    )
    axes[0].plot(
        plot_x,
        da(b + h0_5 * (h0_5 > 10)),
        color=color_dict_T["identical"][3.0][-5],
        label=r"T=-5 $^\circ$C",
    )


    firedrake.plot(icepack.depth_average(b), axes=axes[0], edgecolor="brown", label="_nolegend_")
    axes[0].legend(loc="lower left")
    axes[0].set_ylabel("Elevation (m)")
    axes[0].set_ylim(0, 6000)

    taub_20 = get_taub(u0_20, C0)
    taub_10 = get_taub(u0_10, C0)
    taub_5 = get_taub(u0_5, C0)

    axes[2].plot(plot_x, taub_20, color=color_dict_T["identical"][3.0][-20])
    axes[2].plot(plot_x, taub_10, color=color_dict_T["identical"][3.0][-10])
    axes[2].plot(plot_x, taub_5, color=color_dict_T["identical"][3.0][-5])
    axes[2].set_ylabel(r"$\tau_b$ (kPa)")
    axes[2].set_ylim(0, 300)

    for u, T in zip([u0_20, u0_10, u0_5], [-20, -10, -5]):
        axes[1].plot(
            plot_x,
            get_at_surf(u),
            color=color_dict_T["identical"][3.0][T],
            label="_nolegend_",
            linestyle="solid"
        )
        axes[1].plot(
            plot_x,
            get_at_bed(u),
            color=color_dict_T["identical"][3.0][T],
            label="_nolegend_",
            linestyle="dashed"
        )


    axes[1].plot([], [], color="k", linestyle="solid", label="Surface")
    axes[1].plot([], [], color="k", linestyle="dashed", label="Basal")
    axes[1].legend(loc="best")

    axes[3].set_ylabel("SMB (m yr$^{-1}$)")
    firedrake.plot(icepack.depth_average(a), axes=axes[3], edgecolor="k", label="Initial")
    firedrake.plot(icepack.depth_average(a_retreat), axes=axes[3], edgecolor="0.6", label="Retreat")
    # firedrake.plot(icepack.depth_average(a_pert), axes=axes[3], edgecolor='0.6', label="Perturbed")
    axes[3].set_ylim(-1.5, 1.5)
    # axes[3].axhline(0, color="k", lw="0.5")
    axes[3].legend(loc="best")

    for ax, letter in zip(axes, "abcde"):
        ax.text(0.01, 0.98, letter, fontsize=14, ha="left", va="top", transform=ax.transAxes)

    fig.tight_layout(pad=0.4)
    fig.savefig("figs/initialized_flowline_nbumps{:1d}.pdf".format(nbumps))

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7.0, 4.5))
    fig.subplots_adjust(right=0.78, top=0.98, bottom=0.12, left=0.11)
    axes[0].set_xlim(0, 500e3)
    axes[-1].xaxis.set_major_formatter(kmfmt)
    axes[-1].set_xlabel("Distance (km)")

    for i in range(1):
        axes[0 + i * 3].set_ylabel("Speed\n (m yr$^{-1}$)")
        axes[1 + i * 3].set_ylabel("Sliding (%)")
        axes[2 + i * 3].set_ylabel(r"$\tau_b$ (kPa)")

    axes[0].set_ylim(0, 200)
    axes[1].set_ylim(-2, 102)
    axes[2].set_ylim(-4, 200)

    firedrake.plot(
        extract_surface(firedrake.Function(V_8).interpolate(u0_10 * (2 * (h0_10 > 10) - 1))),
        axes=axes[0],
        edgecolor="k",
        lw=2,
        label="True\n" + r"({:d}$^\circ$C $n$={:2.1f})".format(-10, 3),
        zorder=1000
    )

    firedrake.plot(
        firedrake.Function(extract_bed(C0).function_space()).interpolate(
            extract_bed(firedrake.Function(V_8).interpolate(u0_10 * 100 * (2 * (h0_10 > 10) - 1)))
            / extract_surface(firedrake.Function(V_8).interpolate(u0_10))
        ),
        axes=axes[1],
        edgecolor="k",
        label=[None],
        zorder=1000
    )

    taub = get_taub(u0_10, C0)
    axes[2].plot(plot_x, taub, color="k", zorder=1000)

    ind = 0
    for T, lw in zip([Ts[1], Ts[0], Ts[2]], [1.5, 1.5, 1.5]):
        for n in ns:
            color = color_dict_T["standard"][n][T]
            ind += 1
            inv_name = "T{:d}_n{:2.1f}".format(T, n)
            if inv_name not in done:
                print("No", inv_name)
                continue

            if nbumps == 2:
                misfit = extract_surface(
                    firedrake.Function(V_8).interpolate((output_dict[T][n]["u3"] - u0_10) * (2 * (h0_10 > 10) - 1))
                )
                rms = (
                    firedrake.assemble(misfit**2.0 * firedrake.dx)
                    / firedrake.assemble(extract_surface(2 * (h0_10 > 10) - 1) * firedrake.dx)
                ) ** 0.5

            y = get_at_surf(output_dict[T][n]["u3"] * (2 * (h0_10 > 10) - 1))
            if T == -10:
                label = r"$n$={:2.1f}".format(n)
                # r"{:d}$^\circ$C $n$={:2.1f}".format(T, n)
            else:
                label = "_nolegend_"

            axes[0].plot(
                plot_x, y,
                color=color,
                lw=lw,
                label=label,
                ls=ls_dict[T]
            )

            axes[1].plot(
                    plot_x,
                    get_sliding(output_dict[T][n]["u3"]),
                    color=color,
                    label=[None],
                    lw=lw,
                    ls=ls_dict[T]
                    )

            taub = get_taub(output_dict[T][n]["u3"], output_dict[T][n]["C3"])
            axes[2].plot(plot_x, taub, color=color, lw=lw, ls=ls_dict[T])
        if T == -10:
            leg = axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 0.99), fontsize=9)
            fig.savefig("figs/initialized_variableC_standard_nbumps{:1d}_{:d}C.pdf".format(nbumps, T))
            leg.remove()

    axes[0].plot([], [], linestyle="solid", color="0.4", label="$T$=-10$^\circ$C")
    axes[0].plot([], [], linestyle="dotted", color="0.4", label="$T$=-12$^\circ$C")
    axes[0].plot([], [], linestyle="dashed", color="0.4", label="$T$=-8$^\circ$C")


    if False:
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend([handles[0]] + handles[5:9]  + handles[1:5] + handles[9:],
                       [labels[0]] + labels[5:9]  + labels[1:5] + labels[9:],
                       loc="upper left", bbox_to_anchor=(1.01, 0.99), fontsize=9)

    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 0.99), fontsize=9)

    for ax, letter in zip(axes, "abcdefgh"):
        ax.text(0.01, 0.98, letter, fontsize=14, ha="left", va="top", transform=ax.transAxes)

    # fig.tight_layout(pad=0.1)
    fig.savefig("figs/initialized_variableC_standard_nbumps{:1d}.pdf".format(nbumps))

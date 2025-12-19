#!/usr/bin/env python
# coding: utf-8
import os
import firedrake
import numpy as np
from icepackaccs import extract_surface
from icepackaccs.friction import get_weertman, get_regularized_coulomb_simp
import matplotlib.pyplot as plt
from true_flowline import u0_coulomb
from matplotlib.patches import Rectangle

# Need to muck around to use color consistently outside a package
import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from common_colors import color_dict_T


plotvel = False
plotsurf = False


def volume(thickness):
    return firedrake.assemble(thickness * firedrake.dx)


Lx = 500e3
nx = 2001
mesh1d = firedrake.IntervalMesh(nx, Lx)
mesh = firedrake.ExtrudedMesh(mesh1d, layers=1, name="flowline")


num_years = 10
min_thick = firedrake.Constant(10.0)
Ts_standard = [-12, -10, -8]
Ts_ident = [-20, -10, -5]
devs = [0.75, 1.0, 1.25]
ns = [1.8, 3, 3.5, 4]

regularized_coulomb = get_regularized_coulomb_simp(m=3, u_0=u0_coulomb)
weertman_3 = get_weertman(m=3)
weertman_1 = get_weertman(m=1)
frictions = {"3": weertman_3, "1": weertman_1, "RCFi": regularized_coulomb}
samba_mnt = "/Volumes/slate/Rheology/n4/flowline/outputs"
if os.path.exists(samba_mnt):
    basename = samba_mnt
else:
    basename = "outputs"


def main():
    for nbumps in [2, 1]:
        vol_dict_pert = {T: {n: {fricname: {} for fricname in frictions} for n in ns} for T in Ts_standard}
        vel_dict_pert = {T: {n: {fricname: {} for fricname in frictions} for n in ns} for T in Ts_standard}
        term_dict_pert = {T: {n: {fricname: {} for fricname in frictions} for n in ns} for T in Ts_standard}

        vol_dict_pert_ident = {T: {n: {fricname: {} for fricname in frictions} for n in ns} for T in Ts_ident}
        vel_dict_pert_ident = {T: {n: {fricname: {} for fricname in frictions} for n in ns} for T in Ts_ident}
        term_dict_pert_ident = {T: {n: {fricname: {} for fricname in frictions} for n in ns} for T in Ts_ident}

        do_these = [
            ("retreat", "standard", vol_dict_pert, vel_dict_pert, term_dict_pert, Ts_standard),
            ("retreat", "identical", vol_dict_pert_ident, vel_dict_pert_ident, term_dict_pert_ident, Ts_ident),
        ]

        targ_times = [100, 1000, 10000]

        term_dict_pert["True"] = {}
        vol_dict_pert["True"] = {}
        vel_dict_pert["True"] = {}
        output_fn = basename + "/{:s}_{:s}_T{:d}_n{:2.1f}_{:s}_nbumps{:d}.h5".format(
            "retreat", "identical", -10, 3, "3", 2
        )
        with firedrake.CheckpointFile(output_fn, "r") as chk:
            mesh = chk.load_mesh("flowline")
            x, ζ = firedrake.SpatialCoordinate(mesh)
            hist = chk.get_timestepping_history(mesh, "h")
            if "time" in hist:
                times = hist["time"]
            else:
                times = np.arange(chk.get_attr("metadata", "final_idx") + 1) * chk.get_attr("metadata", "dt")
            for i, time in enumerate(times):
                if int(time) in targ_times + [0]:
                    h = chk.load_function(mesh, name="h", idx=i)
                    vol = volume(h)
                    u = chk.load_function(mesh, name="u", idx=i)
                    vel = np.max(extract_surface(u * firedrake.conditional(h > 15.0, 1.0, 0.0)).dat.data_ro[:])
                    has_glacier = firedrake.Function(h.function_space()).interpolate(
                        x * ((h > 10.1) - 0.5) * 2.0 * (x < 500e3)
                    )
                    terminus = np.max(has_glacier.dat.data_ro[:])
                    term_dict_pert["True"][int(time)] = terminus
                    vol_dict_pert["True"][int(time)] = vol
                    vel_dict_pert["True"][int(time)] = vel

        for simname, initname, vol_dict, vel_dict, term_dict, Ts in do_these:
            for T_np in Ts:
                for n in vel_dict[T_np]:
                    for fricname, friction in frictions.items():

                        if initname in ["standard", "identical"]:
                            output_fn = basename + "/{:s}_{:s}_T{:d}_n{:2.1f}_{:s}_nbumps{:d}.h5".format(
                                simname, initname, T_np, n, fricname, nbumps
                            )
                        else:
                            output_fn = basename + "/{:s}_{:s}_dev{:1.2f}_n{:2.1f}_{:s}_nbumps{:d}.h5".format(
                                simname, initname, T_np, n, fricname, nbumps
                            )
                        print(output_fn)
                        if not os.path.exists(output_fn):
                            print("no " + output_fn)
                            term_dict[T_np][n][fricname] = np.array([])
                            vol_dict[T_np][n][fricname] = np.array([])
                            vel_dict[T_np][n][fricname] = np.array([])
                            continue

                        with firedrake.CheckpointFile(output_fn, "r") as chk:
                            mesh = chk.load_mesh("flowline")
                            x, ζ = firedrake.SpatialCoordinate(mesh)
                            hist = chk.get_timestepping_history(mesh, "h")
                            if "time" in hist:
                                times = hist["time"]
                            else:
                                times = np.arange(chk.get_attr("metadata", "final_idx") + 1) * chk.get_attr(
                                    "metadata", "dt"
                                )
                            for i, time in enumerate(times):
                                if int(time) in targ_times:
                                    h = chk.load_function(mesh, name="h", idx=i)
                                    vol = volume(h)
                                    u = chk.load_function(mesh, name="u", idx=i)
                                    vel = np.max(
                                        extract_surface(u * firedrake.conditional(h > 15.0, 1.0, 0.0)).dat.data_ro[:]
                                    )
                                    has_glacier = firedrake.Function(h.function_space()).interpolate(
                                        x * ((h > 10.1) - 0.5) * 2.0 * (x < 500e3)
                                    )
                                    terminus = np.max(has_glacier.dat.data_ro[:])
                                    term_dict[T_np][n][fricname][int(time)] = terminus
                                    vol_dict[T_np][n][fricname][int(time)] = vol
                                    vel_dict[T_np][n][fricname][int(time)] = vel

            if initname == "standard":
                # fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7.9, 6))
                # fig.subplots_adjust(right=0.79, top=0.98, bottom=0.1)
                fig, axes = plt.subplots(3, 1, sharex=True, figsize=(3.5, 5))
                fig.subplots_adjust(bottom=0.33, top=0.98, right=0.99, left=0.20)
            else:
                fig, axes = plt.subplots(3, 1, sharex=True, figsize=(3.0, 5))
                fig.subplots_adjust(bottom=0.33, top=0.98, right=0.98, left=0.23)

            plot_pts(axes, Ts, vol_dict, vol_dict_pert, targ_times, initname)
            fig.savefig("figs/transient_{:s}_bytime_nbumps{:d}.pdf".format(initname, nbumps))
        fig, axes = plt.subplots(3, 2, sharey="row", sharex="col", figsize=(6.5, 5))
        fig.subplots_adjust(bottom=0.33, top=0.98, right=0.98, left=0.11)
        plot_pts(axes[:, 0], Ts_standard, vol_dict_pert, vol_dict_pert, targ_times, "standard", labelstuff=False)
        axes[0, 0].set_ylim(-1, 3)
        axes[1, 0].set_ylim(-50, 150)
        axes[2, 0].set_ylim(-50, 150)
        for ax, time, letter in zip([ax for ax in axes[:, 0]] + [ax for ax in axes[:, 1]], targ_times, "abcdefgh"):
            ax.text(
                0.01, 0.99, letter + " {:d} yrs".format(time), transform=ax.transAxes, ha="left", va="top", fontsize=12
            )
        axes[1, 0].set_ylabel("Relative change in volume (%)\n")
        fig.savefig("figs/transient_sixp_bytime_nbumps{:d}.pdf".format(nbumps))


def plot_pts(
    axes, Ts, vol_dict, vol_dict_pert, targ_times, initname, labelstuff=True, legend=True, xlabelall=False, offsize=0.25, shade=None, markersize=8,
):
    for ax in axes:
        ax.axhline(0, color="k", zorder=0.5, lw=0.5)
    for T_np in Ts:
        for n in vol_dict[T_np]:
            color = color_dict_T[initname][n][T_np]
            for fricname, friction in frictions.items():
                if fricname == "1":
                    marker = "o"
                    off = -offsize
                elif fricname == "3":
                    marker = "s"
                    off = 0.0
                else:
                    marker = "d"
                    off = offsize
                suboff = (ns.index(n) - 1) / 8 * offsize


                for ax, time in zip(axes, targ_times):
                    if n in [1.8, 3.5, 4]:
                        if fricname == "1":
                            if initname in ["standard", "identical"]:
                                label = r"{:d}$^\circ$C $n$={:2.1f}".format(T_np, n)
                            else:
                                label = r"{:d}%$A_0$ $n$={:2.1f}".format(int(T_np * 100), n)
                        else:
                            label = None
                        ax.plot(
                            2 + off + suboff,
                            (vol_dict[T_np][n][fricname][time] - vol_dict[T_np][3][fricname][time])
                            / (vol_dict_pert["True"][time] - vol_dict_pert["True"][0])
                            * 1.0e2,
                            color=color,
                            marker=marker,
                            linestyle="None",
                            label=label,
                            markersize=markersize
                        )

                    if fricname in ["1", "RCFi"]:
                        if n == 3 and fricname == "1":
                            if initname in ["standard", "identical"]:
                                label = r"{:d}$^\circ$C $n$={:2.1f}".format(T_np, n)
                            else:
                                label = r"{:d}%$A_0$ $n$={:2.1f}".format(int(T_np * 100), n)
                        else:
                            label = None
                        ax.plot(
                            3 + off + suboff,
                            (vol_dict[T_np][n][fricname][time] - vol_dict[T_np][n]["3"][time])
                            / (vol_dict_pert["True"][time] - vol_dict_pert["True"][0])
                            * 1.0e2,
                            color=color,
                            marker=marker,
                            linestyle="None",
                            label=label,
                            markersize=markersize
                        )

                    if initname == "standard":
                        if T_np in [T for T in Ts if T != -10]:
                            ax.plot(
                                1 + off + suboff,
                                (vol_dict[T_np][n][fricname][time] - vol_dict[-10][n][fricname][time])
                                / (vol_dict_pert["True"][time] - vol_dict_pert["True"][0])
                                * 1.0e2,
                                color=color,
                                marker=marker,
                                linestyle="None",
                                markersize=markersize
                            )

    if shade is not None:
        for ax, time in zip(axes, targ_times):

            vs = []
            for fricname in frictions:
                for T_np in shade:
                    for n in [1.8, 3, 4]:
                        vs.append((vol_dict[T_np][n][fricname][time] - vol_dict[T_np][3][fricname][time]) / (vol_dict_pert["True"][time] - vol_dict_pert["True"][0]) * 1e2)
            print(initname, "1.8, 3, 4, diff n", time, np.max(vs), np.min(vs))
            rect = Rectangle((2 - offsize, np.min(vs)),
                             offsize * 2,
                             np.max(vs) - np.min(vs),
                             zorder=0.5,
                             fill=False,
                             hatch="///")
            ax.add_patch(rect)

            vs = []
            for fricname in ["1", "RCFi"]:
                for T_np in shade:
                    for n in [1.8, 3, 4]:
                        vs.append((vol_dict[T_np][n][fricname][time] - vol_dict[T_np][n]["3"][time]) / (vol_dict_pert["True"][time] - vol_dict_pert["True"][0]) * 1.0e2)
            rect = Rectangle((3 - offsize, np.min(vs)),
                             offsize * 2,
                             np.max(vs) - np.min(vs),
                             zorder=0.5,
                             fill=False,
                             hatch="///")
            print(initname, "1.8, 3, 4, diff slide", time, np.max(vs), np.min(vs))
            ax.add_patch(rect)

            if initname == "standard":
                vs = []
                for fricname in frictions:
                    for T_np in shade:
                        for n in [1.8, 3, 4]:
                            vs.append((vol_dict[T_np][n][fricname][time] - vol_dict[-10][n][fricname][time]) / (vol_dict_pert["True"][time] - vol_dict_pert["True"][0]) * 1.0e2)
                rect = Rectangle((1 - offsize, np.min(vs)),
                                 offsize * 2,
                                 np.max(vs) - np.min(vs),
                                 zorder=0.5,
                                 fill=False,
                                 hatch="///")
                print(initname, "1.8, 3, 4, diff T", time, np.max(vs), np.min(vs))
                ax.add_patch(rect)

            vs = []
            for fricname in frictions:
                for T_np in shade:
                    for n in [1.8, 3, 3.5, 4]:
                        vs.append((vol_dict[T_np][n][fricname][time] - vol_dict[T_np][3][fricname][time]) / (vol_dict_pert["True"][time] - vol_dict_pert["True"][0]) * 1e2)
            rect = Rectangle((2 - offsize, np.min(vs)),
                             offsize * 2,
                             np.max(vs) - np.min(vs),
                             zorder=0.5,
                             fill=False,
                             hatch='\\\\\\')
            print(initname, "all n diff n", time, np.max(vs), np.min(vs))
            ax.add_patch(rect)

            vs = []
            for fricname in ["1", "RCFi"]:
                for T_np in shade:
                    for n in [1.8, 3, 3.5, 4]:
                        vs.append((vol_dict[T_np][n][fricname][time] - vol_dict[T_np][n]["3"][time]) / (vol_dict_pert["True"][time] - vol_dict_pert["True"][0]) * 1.0e2)
            rect = Rectangle((3 - offsize, np.min(vs)),
                             offsize * 2,
                             np.max(vs) - np.min(vs),
                             zorder=0.5,
                             fill=False,
                             hatch="\\\\\\")
            print(initname, "all n diff slide", time, np.max(vs), np.min(vs))
            ax.add_patch(rect)

            if initname == "standard":
                vs = []
                for fricname in frictions:
                    for T_np in shade:
                        for n in [1.8, 3, 3.5, 4]:
                            vs.append((vol_dict[T_np][n][fricname][time] - vol_dict[-10][n][fricname][time]) / (vol_dict_pert["True"][time] - vol_dict_pert["True"][0]) * 1.0e2)
                rect = Rectangle((1 - offsize, np.min(vs)),
                                 offsize * 2,
                                 np.max(vs) - np.min(vs),
                                 zorder=0.5,
                                 fill=False,
                                 hatch="\\\\\\")
                print(initname, "all n diff T", time, np.max(vs), np.min(vs))
                ax.add_patch(rect)


    if legend:
        axes[0].plot([], [], marker="o", color="k", linestyle="None", label="$m$=1")
        axes[0].plot([], [], marker="s", color="k", linestyle="None", label="$m$=3")
        axes[0].plot([], [], marker="d", color="k", linestyle="None", label="RCFi")
        if initname in ["identical"]:
            kwargs = {"bbox_to_anchor": (-0.33, -2.7), "ncol": 2, "frameon": False}
        else:
            kwargs = {"bbox_to_anchor": (-0.2, -2.7), "ncol": 2, "frameon": False}
        axes[0].legend(loc="upper left", **kwargs)

    if xlabelall:
        labelthese = axes
    else:
        labelthese = [axes[2]]
    if initname == "identical":
        for ax in labelthese:
            ax.set_xticks([2, 3])
            ax.set_xticklabels(["Diff.\n$n$", "Diff.\nsliding"])
    elif initname == "standard":
        for ax in labelthese:
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(["Diff.\n$T$", "Diff.\n$n$", "Diff.\nsliding"])
    if labelstuff:
        for ax, time, letter in zip(axes, targ_times, "abcde"):
            ax.text(
                0.01, 0.99, letter + " {:d} yrs".format(time), transform=ax.transAxes, ha="left", va="top", fontsize=12
            )

        if initname in ["identical"]:
            axes[1].set_ylabel("Relative change in volume (%)")
        else:
            axes[1].set_ylabel("Relative change in volume (%)\n")


if __name__ == "__main__":
    main()

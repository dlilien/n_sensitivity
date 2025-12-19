#!/usr/bin/env python
# coding: utf-8
import os
import firedrake
import numpy as np
import xarray as xr
from icepackaccs import extract_surface
from icepackaccs.friction import get_weertman, get_regularized_coulomb_simp
import matplotlib.pyplot as plt
from matplotlib import colors
from true_flowline import u0_coulomb
from analyze_discrete import plot_pts
from matplotlib import gridspec
from matplotlib.ticker import NullFormatter

# Need to muck around to use color consistently outside a package
import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from common_colors import color_dict_T




plotvel = True
plotsurf = False


def volume(thickness):
    return firedrake.assemble(thickness * firedrake.dx)


def kmfmt(x, pos):
    return "{:d}".format(int(x / 1000.0))


def kmfmt1(x, pos):
    return "{:2.1f}".format(x / 1000.0)


Lx = 500e3
nx = 2001
mesh1d = firedrake.IntervalMesh(nx, Lx)
mesh = firedrake.ExtrudedMesh(mesh1d, layers=1, name="flowline")


num_years = 10
min_thick = firedrake.Constant(10.0)
Ts_standard = [-12, -10, -8]
Ts_ident = [-20, -10, -5]
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

# Load the "True" file to get the time vector
first_fn = samba_mnt + "/{:s}_{:s}_T{:d}_n{:2.1f}_{:s}_nbumps{:d}.h5".format("retreat", "identical", -10, 3.0, "3", 2)
print(first_fn)
with firedrake.CheckpointFile(first_fn, "r") as chk:
    mesh = chk.load_mesh("flowline")
    hist = chk.get_timestepping_history(mesh, "h")
    times = hist["time"]


coords = [
    ("T", [-20, -12, -10, -8, -5]),
    ("dev", [0.75, 1.0, 1.25]),
    ("n", [1.8, 3, 3.5, 4]),
    ("fric", ["1", "3", "RCFi"]),
    ("time", times),
    ("attr", ["vel", "vol", "term"]),
]

devs = [0.75, 1.0, 1.25]
out_dict = {"retreat": {}, "unperturbed": {}}


def loc0(indict):
    d = indict.copy()
    d["time"] = 0
    return d


for nbumps in [2, 1]:
    do_these = [
        ("retreat", "standard", True, Ts_standard),
        ("unperturbed", "standard", True, Ts_standard),
        ("retreat", "identical", True, Ts_ident),
    ]

    for (
        simname,
        initname,
        by_T,
        Ts_or_dev,
    ) in do_these:
        nc_fn = "outputs/transient_results_{:s}_{:s}_nbumps{:d}.nc".format(simname, initname, nbumps)
        if not os.path.exists(nc_fn):
            da = xr.DataArray(
                np.empty([len(dim[1]) for dim in coords]), dims=("T", "dev", "n", "fric", "time", "attr"), coords=coords
            )

            for T_np in Ts_or_dev:
                for n in ns:
                    for fricname, friction in frictions.items():
                        if fricname in ["1", "3"]:
                            fricstring = "m=" + fricname
                        else:
                            fricstring = fricname

                        if by_T:
                            output_fn = basename + "/{:s}_{:s}_T{:d}_n{:2.1f}_{:s}_nbumps{:d}.h5".format(
                                simname, initname, T_np, n, fricname, nbumps
                            )
                        else:
                            output_fn = basename + "/{:s}_{:s}_dev{:1.2f}_n{:2.1f}_{:s}_nbumps{:d}.h5".format(
                                simname, initname, T_np, n, fricname, nbumps
                            )
                        print(output_fn)

                        with firedrake.CheckpointFile(output_fn, "r") as chk:
                            mesh = chk.load_mesh("flowline")
                            x, ζ = firedrake.SpatialCoordinate(mesh)
                            b = chk.load_function(mesh, name="b")
                            a = chk.load_function(mesh, name="a")
                            C_sqrd = chk.load_function(mesh, name="C_sqrd")
                            hist = chk.get_timestepping_history(mesh, "h")
                            this_times = hist["time"]
                            volumes = np.zeros_like(times.astype(float))
                            vels = np.zeros_like(times.astype(float))
                            terminus = np.zeros_like(times.astype(float))
                            if plotsurf:
                                fig, ax = plt.subplots()
                                norm = colors.Normalize(vmin=0, vmax=times[-1])
                                sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.jet)
                            for i, time in enumerate(this_times):
                                h = chk.load_function(mesh, name="h", idx=i)
                                if plotsurf:
                                    if i % 10 == 0:
                                        firedrake.plot(
                                            extract_surface(h), axes=ax, edgecolor=plt.cm.jet(time / times[-1])
                                        )
                                volumes[i] = volume(h)
                                if plotvel:
                                    u = chk.load_function(mesh, name="u", idx=i)
                                    vels[i] = np.max(
                                        extract_surface(u * firedrake.conditional(h > 15.0, 1.0, 0.0)).dat.data_ro[:]
                                    )

                                has_glacier = firedrake.Function(h.function_space()).interpolate(
                                    x * ((h > 10.1) - 0.5) * 2.0 * (x < 500e3)
                                )
                                terminus[i] = np.max(has_glacier.dat.data_ro[:])
                            if plotsurf:
                                plt.colorbar(sm, label="Time (yrs)", ax=ax)
                                fig.savefig(
                                    "figs/surfs/surfaces_{:s}_{:s}_{:d}_{:2.1f}_{:s}_nbumps{:d}.pdf".format(
                                        simname, initname, T_np, n, fricname, nbumps
                                    )
                                )
                                plt.close(fig)

                        if by_T:
                            Tind = T_np
                            dev = 1.0
                        else:
                            Tind = -10
                            dev = T_np

                        da.loc[{"T": Tind, "dev": dev, "n": n, "fric": fricname, "attr": "vel"}] = vels
                        da.loc[{"T": Tind, "dev": dev, "n": n, "fric": fricname, "attr": "vol"}] = volumes
                        da.loc[{"T": Tind, "dev": dev, "n": n, "fric": fricname, "attr": "term"}] = terminus
            da.to_netcdf(nc_fn)
        else:
            da = xr.load_dataarray(nc_fn)
        out_dict[simname][initname] = da

    for (
        simname,
        initname,
        by_T,
        Ts_or_dev,
    ) in do_these:
        da = out_dict[simname][initname]
        ind = 0

    for initname, by_T, Ts_or_dev, col, norm in [
        ("standard", True, Ts_standard, 0, False),
        ("identical", True, Ts_ident, 2, False),
    ]:
        da = out_dict["retreat"][initname]
        da_ident = out_dict["retreat"]["identical"]

        base_dict = {"T": Ts_standard, "dev": 1.0}
        contract_dims = ["n", "fric", "T"]

        if initname != "identical":
            da_unpert = out_dict["unperturbed"][initname]
            unpert_vels_nonrel = da_unpert.loc[{"attr": "vel", **base_dict}]
            da_unpert_rel = da_unpert[:]
            da_unpert_rel.loc[{"attr": "vel"}] = (
                da_unpert_rel.loc[{"attr": "vel"}] / da_unpert_rel.loc[{"attr": "vel", "time": 0}]
            )
            da_unpert_rel.loc[{"attr": "vol"}] = (
                da_unpert_rel.loc[{"attr": "vol"}] - da_unpert_rel.loc[{"attr": "vol", "time": 0}]
            )
            da_unpert_rel.loc[{"attr": "term"}] = (
                da_unpert_rel.loc[{"attr": "term"}] - da_unpert_rel.loc[{"attr": "term", "time": 0}]
            )
            unpert_terms = da_unpert_rel.loc[{"attr": "term", **base_dict}]
            unpert_vels = da_unpert_rel.loc[{"attr": "vel", **base_dict}]
            unpert_vols = da_unpert_rel.loc[{"attr": "vol", **base_dict}]

        if initname == "standard":
            figcomb = plt.figure(figsize=(7, 6.0))
            bottom = 0.16
        else:
            figcomb = plt.figure(figsize=(7, 6.5))
            bottom = 0.06
        gs = gridspec.GridSpec(5, 6, width_ratios=(1, 0.35, 1, 0.35, 0.1, 0.9), height_ratios=(1, 1, 1, 0.1, 1), top=0.99, right=0.99, wspace=0.0, bottom=bottom)
        axes_byattr = [
            figcomb.add_subplot(gs[0, :-1]),
            figcomb.add_subplot(gs[1, :-1]),
            figcomb.add_subplot(gs[2, :-1]),
        ]
        axes_bytime = [figcomb.add_subplot(gs[4, 0]), figcomb.add_subplot(gs[4, 2]), figcomb.add_subplot(gs[4, 4:])]

        figdum, axdum = plt.subplots()

        if not norm and initname != "standard":
            if initname == "identical":
                axes_zoom = [
                    axdum,
                    axes_byattr[1].inset_axes([0.6, 0.15, 0.3, 0.9], xlim=(1.8e3, 2e3), ylim=(-16, -9)),
                    axes_byattr[2].inset_axes([1.05, 0.3, 0.27, 1.25], xlim=(3e3, 4e3), ylim=(-60, -41)),
                ]
            elif initname == "standard":
                axes_zoom = [
                    axdum,
                    axes_byattr[1].inset_axes([0.6, 0.15, 0.3, 0.9], xlim=(2.8e3, 3e3), ylim=(-20, -5)),
                    axes_byattr[2].inset_axes([1.05, -0.33, 0.27, 1.05], xlim=(8.5e3, 9e3), ylim=(-150, -50)),
                ]
            else:
                axes_zoom = [
                    axdum,
                    axes_byattr[1].inset_axes([0.6, 0.15, 0.3, 0.9], xlim=(1.8e3, 2e3), ylim=(-15, -8)),
                    axes_byattr[2].inset_axes([1.05, -0.36, 0.27, 1.18], xlim=(3e3, 4e3), ylim=(-60, -42)),
                ]

            axes_zoom[1].xaxis.set_major_formatter(kmfmt1)
            axes_zoom[2].xaxis.set_major_formatter(kmfmt)
            axes_zoom[1].set_xticks([])
            axes_zoom[1].set_yticks([])
            axes_zoom[2].set_xticks([])
            axes_zoom[2].set_yticks([])

        if not norm and initname != "standard":
            axlist = [axes_byattr, axes_zoom]
        else:
            axlist = [axes_byattr]

        for axes in axlist:
            if not norm:
                if initname == "standard":
                    axes[0].fill_between(
                        da.time,
                        unpert_vels_nonrel.min(dim=contract_dims),
                        unpert_vels_nonrel.max(dim=contract_dims),
                        color="0.6",
                        alpha=0.5,
                        # label="Unperturbed",
                        ec="none",
                    )
                    axes[1].fill_between(
                        da.time,
                        unpert_terms.min(dim=contract_dims) * 1.0e-3,
                        unpert_terms.max(dim=contract_dims) * 1.0e-3,
                        color="0.6",
                        alpha=0.5,
                        # label="Unperturbed",
                        ec="none",
                    )
                    axes[2].fill_between(
                        da.time,
                        unpert_vols.min(dim=contract_dims) * 1.0e-6,
                        unpert_vols.max(dim=contract_dims) * 1.0e-6,
                        color="0.6",
                        alpha=0.5,
                        # label="Unperturbed",
                        ec="none",
                    )

            ind = 0
            for T_np in [-10]:  # Ts_or_dev:
                if by_T:
                    Tind = T_np
                    dev = 1.0
                else:
                    Tind = -10
                    dev = T_np
                for n in ns:
                    color = color_dict_T[initname][n][T_np]
                    lw = 1
                    for fricname, friction in frictions.items():
                        if fricname == "1":
                            ls = "dashed"
                        elif fricname == "3":
                            ls = "solid"
                        else:
                            ls = "dotted"
                        if fricname == "3":
                            if by_T:
                                # label = r"{:d}$^\circ$C $n$={:2.1f}".format(T_np, n)
                                label = "$n$={:2.1f}".format(n)
                            else:
                                label = r"{:d}%$A_0$ $n$={:2.1f}".format(int(dev * 100), n)
                        else:
                            label = None
                        loc = {"T": Tind, "dev": dev, "n": n, "fric": fricname, "attr": "vel"}
                        loc_up = {"T": Tind, "dev": 1.0, "n": 3.0, "fric": "3", "attr": "vel"}
                        if norm:
                            axes[0].plot(da.time, da.loc[loc] - da_ident.loc[loc_up], color=color, ls=ls, label=label)
                            loc["attr"] = "term"
                            loc_up["attr"] = "term"
                            axes[1].plot(
                                da.time[::16], (da.loc[loc] - da_ident.loc[loc_up])[::16] / 1.0e3, color=color, ls=ls
                            )
                            loc["attr"] = "vol"
                            loc_up["attr"] = "vol"
                            axes[2].plot(
                                da.time, (da.loc[loc] - da_ident.loc[loc_up]) * 1.0e-6, color=color, ls=ls, label=label
                            )
                        else:
                            axes[0].plot(da.time, da.loc[loc], color=color, ls=ls, label=label)
                            loc["attr"] = "term"
                            axes[1].plot(
                                da.time, (da.loc[loc].values - da.loc[loc].values[0]) / 1.0e3, color=color, ls=ls
                            )
                            loc["attr"] = "vol"
                            axes[2].plot(
                                da.time,
                                (da.loc[loc].values - da.loc[loc].values[0]) * 1.0e-6,
                                color=color,
                                ls=ls,
                                label=label,
                            )
                        axes[1].plot([], [], color=color, ls=ls, label=label, marker="o")

            if not norm:
                loc = {"T": -10, "dev": 1.0, "n": 3, "fric": "3", "attr": "vel"}
                axes[0].plot(
                    da_ident.time,
                    da_ident.loc[loc],
                    color="k",
                    ls="solid",
                    label="True"
                )
                loc["attr"] = "term"
                label = "True"
                axes[1].plot(
                    da_ident.time,
                    (da_ident.loc[loc] - da_ident.loc[loc0(loc)]) / 1.0e3,
                    color="k",
                    ls="solid",
                    label=label,
                )
                loc["attr"] = "vol"
                axes[2].plot(
                    da_ident.time,
                    (da_ident.loc[loc] - da_ident.loc[loc0(loc)]) * 1.0e-6,
                    color="k",
                    ls="solid",
                    label="True"
                )

        targ_times = [100, 1000, 10000]

        if by_T:
            vol_dict = {
                T: {
                    n: {
                        fricname: {
                            time: da.loc[{"T": T, "dev": 1.0, "n": n, "fric": fricname, "attr": "vol", "time": time}]
                            for time in targ_times + [0]
                        }
                        for fricname in frictions
                    }
                    for n in ns
                }
                for T in Ts_or_dev
            }
        else:
            vol_dict = {
                T: {
                    n: {
                        fricname: {
                            time: da.loc[{"T": -10, "dev": T, "n": n, "fric": fricname, "attr": "vol", "time": time}]
                            for time in targ_times + [0]
                        }
                        for fricname in frictions
                    }
                    for n in ns
                }
                for T in Ts_or_dev
            }
        da_true = out_dict["retreat"]["identical"]
        vol_dict_pert = {
            "True": {
                time: da_true.loc[{"T": -10, "dev": 1.0, "n": 3.0, "fric": "3", "attr": "vol", "time": time}]
                for time in targ_times + [0]
            }
        }

        if initname == "identical":
            offsize = 0.25
        else:
            offsize = 0.2

        plot_pts(
            axes_bytime,
            [-10],
            vol_dict,
            vol_dict_pert,
            targ_times,
            initname,
            shade=[T for T in Ts_or_dev if T != -10],
            labelstuff=False,
            legend=False,
            xlabelall=True,
            offsize=offsize,
        )

        axes_byattr[1].plot([], [], ls="solid", color="0.6", label="$m$=3")
        axes_byattr[1].plot([], [], ls="dashed", color="0.6", label="$m$=1")
        axes_byattr[1].plot([], [], ls="dotted", color="0.6", label="RCFi")
        axes_byattr[1].plot([], [], marker="s", color="0.6", linestyle="None", label="$m$=3")
        axes_byattr[1].plot([], [], marker="o", color="0.6", linestyle="None", label="$m$=1")
        axes_byattr[1].plot([], [], marker="d", color="0.6", linestyle="None", label="RCFi")

        if not norm and initname != "standard":
            axes_byattr[1].indicate_inset_zoom(axes_zoom[1], edgecolor="black")
            axes_byattr[2].indicate_inset_zoom(axes_zoom[2], edgecolor="black")

        if initname == "identical":
            axes_byattr[1].legend(loc="upper left", bbox_to_anchor=(1.01, 2.24), fontsize=8)
        else:
            axes_byattr[1].legend(loc="upper left", bbox_to_anchor=(1.01, 2.00), fontsize=8)

        if norm:
            axes_byattr[0].set_ylabel("Rel. change in \nspeed (m yr$^{-1}$)")
            axes_byattr[1].set_ylabel("Rel. change in\nterm. position (km)")
            axes_byattr[2].set_ylabel("Rel. change in\nvolume (km$^2$)")
        else:
            axes_byattr[0].set_ylabel("Max. speed\n(m yr$^{-1}$)")
            axes_byattr[1].set_ylabel("Change in terminus\nposition (km)")
            axes_byattr[2].set_ylabel("Change in\nvolume (km$^2$)")
        if initname in ["standard"]:
            if nbumps == 2:
                axes_byattr[0].set_ylim(60, 100)
                axes_byattr[1].set_ylim(-30, 20)
                axes_byattr[2].set_ylim(-110, 10)

                axes_bytime[0].set_ylim(-2, 0.5)

                axes_bytime[1].set_ylim(-50, 60)
                axes_bytime[1].set_yticks([-50, -25, 0, 25, 50])

                axes_bytime[2].set_ylim(-60, 110)
                axes_bytime[2].set_yticks([-50, 0, 50, 100])

            else:
                axes_byattr[0].set_ylim(60, 100)
                axes_byattr[1].set_ylim(-30, 30)
                axes_byattr[2].set_ylim(-110, 10)
                axes_bytime[0].set_ylim(-2, 1)
                axes_bytime[1].set_ylim(-50, 75)
                axes_bytime[2].set_ylim(-75, 125)

        else:
            axes_bytime[0].set_ylim(-0.05, 0.15)
            axes_bytime[1].set_ylim(-3, 5)
            axes_bytime[2].set_ylim(-15, 25)
            if norm:
                axes_byattr[0].set_ylim(-1, 2)
                axes_byattr[1].set_ylim(-2, 4)
                axes_byattr[2].set_ylim(-13, 7)
            else:
                if nbumps == 2:
                    axes_byattr[0].set_ylim(72, 80)
                else:
                    axes_byattr[0].set_ylim(70, 75)
                axes_byattr[1].set_ylim(-26, 1)
                axes_byattr[2].set_ylim(-70, 0)
        axes_byattr[2].set_xlabel("Time (kyr)")

        axes_bytime[0].set_ylabel("Relative change\nin volume (%)")

        for ax, letter in zip(axes_byattr, "abcdefg"):
            if norm or initname == "standard":
                ax.text(0.01, 0.99, letter, transform=ax.transAxes, fontsize=12, ha="left", va="top")
            else:
                ax.text(0.07, 0.99, letter, transform=ax.transAxes, fontsize=12, ha="left", va="top")
            ax.xaxis.set_major_formatter(kmfmt)

        for ax in axes_byattr[:2]:
            ax.set_xlim(0, 10000)
            ax.xaxis.set_major_formatter(NullFormatter())
        axes_byattr[2].set_xlim(0, 10000)

        for ax, letter, time in zip(axes_bytime, "defg", targ_times):
            if time < 1000:
                ax.text(
                    0.01,
                    0.99,
                    "{:s} {:d} yr".format(letter, time),
                    transform=ax.transAxes,
                    fontsize=12,
                    ha="left",
                    va="top",
                )
            else:
                ax.text(
                    0.01,
                    0.99,
                    "{:s} {:d} kyr".format(letter, time // 1000),
                    transform=ax.transAxes,
                    fontsize=12,
                    ha="left",
                    va="top",
                )


        if initname == "identical":
            axes_bytime[2].text(0.9, 0.28, "Spread with\n$T$=-20 or -5" + r"$^\circ$C", transform=figcomb.transFigure, ha="center", va="bottom")
            axes_bytime[2].annotate("", (2.1, 2.5), (0.9, 0.28), textcoords="figure fraction", arrowprops={"arrowstyle": '->'})
            axes_bytime[2].annotate("", (3, 21), (0.9, 0.28), textcoords="figure fraction", arrowprops={"arrowstyle": '->'})
            axes_bytime[1].annotate("", (3, 4.2), (0.9, 0.28), textcoords="figure fraction", arrowprops={"arrowstyle": '->'})
        else:
            axes_bytime[2].text(0.9, 0.38, "Spread with\n$T$=-12, -10, or -8" + r"$^\circ$C" + "\n$n$=1.8, 3, 3.5, or 4", transform=figcomb.transFigure, ha="center", va="bottom")
            if nbumps == 2:
                axes_bytime[1].annotate("", (2, 51), (0.9, 0.38), textcoords="figure fraction", arrowprops={"arrowstyle": '->'})
                axes_bytime[2].annotate("", (2, 95), (0.9, 0.38), textcoords="figure fraction", arrowprops={"arrowstyle": '->'})
            else:
                axes_bytime[1].annotate("", (2.2, 51), (0.9, 0.38), textcoords="figure fraction", arrowprops={"arrowstyle": '->'})
                axes_bytime[2].annotate("", (2, 105), (0.9, 0.38), textcoords="figure fraction", arrowprops={"arrowstyle": '->'})

            axes_bytime[2].annotate("", (0.85, -13), (0.685, 0.06), textcoords="figure fraction", arrowprops={"arrowstyle": '->'})
            axes_bytime[1].annotate("", (1.15, -1), (0.55, 0.06), textcoords="figure fraction", arrowprops={"arrowstyle": '->'})
            axes_bytime[2].text(0.62, 0.00, "Spread with\n$T$=-12, -10, or -8" + r"$^\circ$C" + "\nexcluding $n$=3.5", transform=figcomb.transFigure, ha="center", va="bottom")

            axes_byattr[1].text(0.55, 0.7, "Spread of\nunperturbed\nsimulations", transform=figcomb.transFigure, ha="left", va="center")
            axes_byattr[1].annotate("", (0.4, 0.855), (0.55, 0.7), xycoords="figure fraction", textcoords="figure fraction", arrowprops={"arrowstyle": '->'}, zorder=99999, annotation_clip=False)
            axes_byattr[1].annotate("", (3500, 2), (0.55, 0.7), textcoords="figure fraction", arrowprops={"arrowstyle": '->'})
            axes_byattr[2].annotate("", (5000, -3), (0.55, 0.7), textcoords="figure fraction", arrowprops={"arrowstyle": '->'})

        figcomb.savefig("figs/transient_{:s}_comb_bumps{:d}.pdf".format(initname, nbumps))

    da_pert = out_dict["retreat"]["standard"]
    pert_terms = da_pert.loc[{"dev": 1.0, "attr": "term", "T": Ts_standard}]
    tdiff = (pert_terms.max(dim=["n", "fric", "T"]) - pert_terms.min(dim=["n", "fric", "T"])).values
    print(
        "Maximum spread in standard terminus: {:5.1f} km at time {:d}".format(
            np.max(tdiff), int(da.time.values[np.argmax(tdiff)])
        )
    )

    da_pert = out_dict["retreat"]["identical"]
    pert_terms = da_pert.loc[{"dev": 1.0, "attr": "term", "T": Ts_ident}]
    tdiff = (pert_terms.max(dim=["n", "fric", "T"]) - pert_terms.min(dim=["n", "fric", "T"])).values
    print(
        "Maximum spread in identical terminus: {:5.1f} km at time {:d}".format(
            np.max(tdiff), int(da.time.values[np.argmax(tdiff)])
        )
    )

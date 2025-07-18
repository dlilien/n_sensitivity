#!/usr/bin/env python
# coding: utf-8
import sys
import os
import firedrake
import numpy as np
import xarray as xr
from icepackaccs import extract_surface
from icepackaccs.friction import get_weertman, get_regularized_coulomb_simp
import matplotlib.pyplot as plt
from matplotlib import colors
from true_flowline import u0_coulomb, color_dict
from analyze_discrete import plot_pts
from matplotlib import gridspec
from matplotlib.ticker import NullFormatter

sys.path.append("../mismip/")
from discrete_plots import plot_init


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

samba_mnt = "/Volumes/slate/Rheology/n4/flowline/outputs/"
if os.path.exists(samba_mnt):
    basename = samba_mnt
else:
    basename = "outputs"

# Load the "True" file to get the time vector
first_fn = "outputs/{:s}_{:s}_T{:d}_n{:2.1f}_{:s}_nbumps{:d}.h5".format("retreat", "identical", -10, 3.0, "3", 2)
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
                            volumes = np.zeros_like(times.astype(float))
                            vels = np.zeros_like(times.astype(float))
                            terminus = np.zeros_like(times.astype(float))
                            if plotsurf:
                                fig, ax = plt.subplots()
                                norm = colors.Normalize(vmin=0, vmax=times[-1])
                                sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.jet)
                            for i, time in enumerate(times):
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

        figcomb = plt.figure(figsize=(7, 6.5))
        gs = gridspec.GridSpec(
            5,
            6,
            width_ratios=(1, 0.35, 1, 0.35, 0.1, 0.9),
            height_ratios=(1, 1, 1, 0.1, 0.8),
            top=0.99,
            right=0.99,
            wspace=0.0,
            bottom=0.06,
        )
        axes_byattr = [
            figcomb.add_subplot(gs[0, :-1]),
            figcomb.add_subplot(gs[1, :-1]),
            figcomb.add_subplot(gs[2, :-1]),
        ]
        axes_bytime = [figcomb.add_subplot(gs[4, 0]), figcomb.add_subplot(gs[4, 2]), figcomb.add_subplot(gs[4, 4:])]

        figdum, axdum = plt.subplots()

        if not norm:
            if initname == "identical":
                axes_zoom = [
                    axdum,
                    axes_byattr[1].inset_axes([0.6, 0.15, 0.3, 0.9], xlim=(1.8e3, 2e3), ylim=(-16, -9)),
                    axes_byattr[2].inset_axes([1.05, -0.33, 0.27, 1.25], xlim=(3e3, 4e3), ylim=(-60, -41)),
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

        if not norm:
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
                        label="Unperturbed",
                        ec="none",
                    )
                    axes[1].fill_between(
                        da.time,
                        unpert_terms.min(dim=contract_dims) * 1.0e-3,
                        unpert_terms.max(dim=contract_dims) * 1.0e-3,
                        color="0.6",
                        alpha=0.5,
                        label="Unperturbed",
                        ec="none",
                    )
                    axes[2].fill_between(
                        da.time,
                        unpert_vols.min(dim=contract_dims) * 1.0e-6,
                        unpert_vols.max(dim=contract_dims) * 1.0e-6,
                        color="0.6",
                        alpha=0.5,
                        label="Unperturbed",
                        ec="none",
                    )

                    axes[0].fill_between(
                        da.time,
                        unpert_vels_nonrel.loc[{"n": [3.0, 3.5, 4.0]}].min(dim=contract_dims),
                        unpert_vels_nonrel.loc[{"n": [3.0, 3.5, 4.0]}].max(dim=contract_dims),
                        color="0.4",
                        alpha=0.5,
                        label="Unpert. w/o $n$=1.8",
                        ec="none",
                    )
                    axes[1].fill_between(
                        da.time,
                        unpert_terms.loc[{"n": [3.0, 3.5, 4.0]}].min(dim=contract_dims) * 1.0e-3,
                        unpert_terms.loc[{"n": [3.0, 3.5, 4.0]}].max(dim=contract_dims) * 1.0e-3,
                        color="0.4",
                        alpha=0.5,
                        label="Unpert. w/o $n$=1.8",
                        ec="none",
                    )
                    axes[2].fill_between(
                        da.time,
                        unpert_vols.loc[{"n": [3.0, 3.5, 4.0]}].min(dim=contract_dims) * 1.0e-6,
                        unpert_vols.loc[{"n": [3.0, 3.5, 4.0]}].max(dim=contract_dims) * 1.0e-6,
                        color="0.4",
                        alpha=0.5,
                        label="Unpert. w/o $n$=1.8",
                        ec="none",
                    )

            ind = 0
            for T_np in Ts_or_dev:
                if by_T:
                    Tind = T_np
                    dev = 1.0
                else:
                    Tind = -10
                    dev = T_np
                for n in ns:
                    color = color_dict[initname][n][T_np]
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
                                label = r"{:d}$^\circ$C $n$={:2.1f}".format(T_np, n)
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
                    label='"True"\n' + r"({:d}$^\circ$C $n$={:2.1f})".format(-10, 3),
                )
                loc["attr"] = "term"
                label = r"True ({:d}$^\circ$C $n$=3)".format(-10)
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
                    label='"True"\n' + r"({:d}$^\circ$C $n$={:2.1f})".format(-10, 3),
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
            offsize = 0.1
        else:
            offsize = 0.2

        plot_pts(
            axes_bytime,
            Ts_or_dev,
            vol_dict,
            vol_dict_pert,
            targ_times,
            initname,
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

        if not norm:
            axes_byattr[1].indicate_inset_zoom(axes_zoom[1], edgecolor="black")
            axes_byattr[2].indicate_inset_zoom(axes_zoom[2], edgecolor="black")

        axes_byattr[1].legend(loc="upper left", bbox_to_anchor=(1.01, 2.24), fontsize=8)
        if norm:
            axes_byattr[0].set_ylabel("Rel. change in \nspeed (m yr$^{-1}$)")
            axes_byattr[1].set_ylabel("Rel. change in\nterm. position (km)")
            axes_byattr[2].set_ylabel("Rel. change in\nvolume (km$^2$)")
        else:
            axes_byattr[0].set_ylabel("Max. speed\n(m yr$^{-1}$)")
            axes_byattr[1].set_ylabel("Change in terminus\nposition (km)")
            axes_byattr[2].set_ylabel("Change in\nvolume (km$^2$)")
        if initname in ["standard"]:
            axes_byattr[0].set_ylim(0, 100)
            axes_byattr[1].set_ylim(-200, 75)
            axes_byattr[2].set_ylim(-250, 750)
            axes_bytime[0].set_ylim(-3, 2)
            axes_bytime[1].set_ylim(-100, 150)
            axes_bytime[2].set_ylim(-1500, 500)
        else:
            axes_bytime[0].set_ylim(-0.05, 0.15)
            axes_bytime[1].set_ylim(-3, 5)
            axes_bytime[2].set_ylim(-15, 25)
            if norm:
                axes_byattr[0].set_ylim(-1, 2)
                axes_byattr[1].set_ylim(-2, 4)
                axes_byattr[2].set_ylim(-13, 7)
            else:
                axes_byattr[0].set_yticks(range(55, 90, 10))
                axes_byattr[0].set_ylim(65, 90)
                axes_byattr[1].set_ylim(-26, 1)
                axes_byattr[2].set_ylim(-75, 0)
        axes_byattr[2].set_xlabel("Time (kyr)")

        axes_bytime[0].set_ylabel("Relative change\nin volume (%)")

        for ax, letter in zip(axes_byattr, "abcdefg"):
            if norm:
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

    fig, axes = plt.subplots(3, 1, figsize=(3, 4.25), sharex=True)
    fig.subplots_adjust(right=0.98, top=0.99, bottom=0.320, left=0.18, wspace=0.1)
    for init in ["standard"]:
        Tind = -10
        dev = 1.0
        for n in ns:
            if init == "standard":
                color = color_dict[init][n][Tind]
                off = -offsize
            else:
                color = color_dict[init][n][dev]
                off = offsize

            if n == 1.8:
                off += -offsize / 5
            elif n == 3:
                off += 0
            else:
                off += offsize / 5
            for fricname, friction in frictions.items():
                if fricname == "1":
                    marker = "o"
                elif fricname == "3":
                    marker = "s"
                else:
                    marker = "d"
                loc = {"T": Tind, "dev": dev, "n": n, "fric": fricname, "attr": "vol"}
                ref_vols = out_dict["retreat"]["identical"].loc[loc]
                vols = out_dict["retreat"][init].loc[loc]
                for time, ax in zip([100, 1000, 10000], axes):
                    index = np.where(ref_vols.time == time)[0][0]
                    if fricname == "1":
                        label = "Standard $n$={:2.1f}".format(n)
                        off = -offsize
                    else:
                        label = None
                    ax.plot(
                        2 + off,
                        (vols - ref_vols)[index] / (ref_vols[index] - ref_vols[0]) * 1.0e2,
                        color=color,
                        marker=marker,
                        linestyle="None",
                        label=label,
                    )

    axes[0].plot([], [], marker="o", color="k", linestyle="None", label="$m$=1")
    axes[0].plot([], [], marker="s", color="k", linestyle="None", label="$m$=3")
    axes[0].plot([], [], marker="d", color="k", linestyle="None", label="RCFi")
    axes[0].legend(loc="upper left", bbox_to_anchor=(-0.25, -2.7), ncol=2, frameon=False)

    axes[2].set_xlim(1.8, 2.2)
    # axes[2].set_xticks([1.9, 2.0, 2.1])
    # axes[2].set_xticklabels(["$n$=1.8", "$n$=3", "$n$=4"])
    axes[2].set_xticks([1.9, 2.1])
    axes[2].set_xticklabels(["Standard"])

    axes[1].set_ylabel(r"$\Delta V$ compared to identical (%)")

    for ax, time, letter in zip(axes, targ_times, "abcde"):
        if time < 1000:
            ax.text(0.01, 0.99, letter, transform=ax.transAxes, ha="left", va="top", fontsize=12)
            ax.text(0.99, 0.99, "{:d} yrs".format(time), transform=ax.transAxes, ha="right", va="top", fontsize=12)
        else:
            ax.text(0.01, 0.99, letter, transform=ax.transAxes, ha="left", va="top", fontsize=12)
            ax.text(
                0.99, 0.99, "{:d} kyr".format(time // 1000), transform=ax.transAxes, ha="right", va="top", fontsize=12
            )

    fig.savefig("figs/initialization_sensitivity_nbumps{:d}.pdf".format(nbumps))

    targ_times = [100, 500, 10000]
    fig, axes = plt.subplots(3, 1, figsize=(3, 4.25), sharex=True)
    fig.subplots_adjust(right=0.98, top=0.99, bottom=0.320, left=0.18, wspace=0.1)
    for init in ["standard"]:
        Tind = -10
        dev = 1.0
        for n in ns:
            if init == "standard":
                color = color_dict[init][n][Tind]
            else:
                color = color_dict[init][n][dev]

            if n == 1.8:
                off = -offsize
            elif n == 3:
                off = 0
            else:
                off = offsize
            for fricname, friction in frictions.items():
                if fricname == "1":
                    marker = "o"
                elif fricname == "3":
                    marker = "s"
                else:
                    marker = "d"
                loc = {"T": Tind, "dev": dev, "n": n, "fric": fricname, "attr": "vol"}
                ref_vols = out_dict["retreat"]["identical"].loc[loc]
                vols = out_dict["retreat"][init].loc[loc]
                for time, ax in zip(targ_times, axes):
                    index = np.where(ref_vols.time == time)[0][0]
                    if fricname == "1":
                        label = "Standard $n$={:2.1f}".format(n)
                    else:
                        label = None
                    ax.plot(
                        2 + off,
                        (vols - ref_vols)[index] / (ref_vols[index] - ref_vols[0]) * 1.0e2,
                        color=color,
                        marker=marker,
                        linestyle="None",
                        label=label,
                    )

    plot_init(axes, pos=3)

    axes[0].plot([], [], marker="o", color="k", linestyle="None", label="$m$=1")
    axes[0].plot([], [], marker="s", color="k", linestyle="None", label="$m$=3")
    axes[0].plot([], [], marker="d", color="k", linestyle="None", label="RCFi")
    axes[0].legend(loc="upper left", bbox_to_anchor=(-0.25, -2.7), ncol=2, frameon=False)

    axes[2].set_xlim(1.7, 3.3)
    # axes[2].set_xticks([1.9, 2.0, 2.1])
    # axes[2].set_xticklabels(["$n$=1.8", "$n$=3", "$n$=4"])
    axes[2].set_xticks([2, 3])
    axes[2].set_xticklabels(["Flowline", "MISMIP+"])

    axes[1].set_ylabel(r"$\Delta V$ compared to identical (%)")

    for ax, time, letter in zip(axes, targ_times, "abcde"):
        ax.text(0.01, 0.99, letter, transform=ax.transAxes, ha="left", va="top", fontsize=12)
        if time < 1000:
            ax.text(0.99, 0.99, "{:d} yrs".format(time), transform=ax.transAxes, ha="right", va="top", fontsize=12)
        else:
            ax.text(
                0.99, 0.99, "{:d} kyr".format(time // 1000), transform=ax.transAxes, ha="right", va="top", fontsize=12
            )

    fig.savefig("figs/initialization_sensitivity_overall_nbumps{:d}.pdf".format(nbumps))

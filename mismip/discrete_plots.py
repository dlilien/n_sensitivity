# coding: utf-8
import os
import copy
import firedrake
import numpy as np
import matplotlib.pyplot as plt
from icepackaccs import extract_surface
from icepack.constants import ice_density as ρ_I, water_density as ρ_W, gravity as g

# Need to muck around to use color consistently outside a package
import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from common_colors import color_dict_T


phys_names = {"": "hybrid", "ssa_": "ssa"}
dirs = {"": "/Volumes/slate/Rheology/n4/toy_model/outputs", "ssa_": "../toy_model/outputs"}
physicses = ["", "ssa_"]
sims = ["unperturbed", "retreat"]
lss = {"unperturbed": "solid", "retreat": "dashed"}
lss_fric = {"3": "solid", "RCFi": "dotted", "1": "dashed"}
fricdict = {"3": "$m$=3", "RCFi": "RCFi", "1": "$m$=1"}
inits = ["identical", "standard", "true"]
frics = ["RCFi", "3", "1"]
ns = [1.8, 3, 3.5, 4]

times = {
    physics: {sim: {init: {fric: {} for fric in frics} for init in inits} for sim in sims} for physics in physicses
}
gls = {physics: {sim: {init: {fric: {} for fric in frics} for init in inits} for sim in sims} for physics in physicses}
vafs = {physics: {sim: {init: {fric: {} for fric in frics} for init in inits} for sim in sims} for physics in physicses}
vels = {physics: {sim: {init: {fric: {} for fric in frics} for init in inits} for sim in sims} for physics in physicses}
grounded_areas = {
    physics: {sim: {init: {fric: {} for fric in frics} for init in inits} for sim in sims} for physics in physicses
}

targ_times = [10, 50, 500]

mesh1d = firedrake.IntervalMesh(10, 10)
meshdum = firedrake.ExtrudedMesh(mesh1d, layers=1)


def get_gl(h, s, z_b, tol=10.0):
    d = len(firedrake.SpatialCoordinate(h.ufl_domain()))
    if d == 3:  # hybrid
        x, y = firedrake.SpatialCoordinate(extract_surface(h).ufl_domain())
        floating = firedrake.conditional(extract_surface(s - h) > extract_surface(z_b) + tol, 0.0, 1.0)
    else:
        x, y = firedrake.SpatialCoordinate(h.ufl_domain())
        floating = firedrake.conditional(s - h > z_b + tol, 0.0, 1.0)
    return floating


def get_floating(h0, s0):
    d = len(firedrake.SpatialCoordinate(h0.ufl_domain()))
    p_W = ρ_W * g * firedrake.max_value(0, h0 - s0)
    p_I = ρ_I * g * h0
    if d == 3:
        return extract_surface(
            firedrake.conditional(p_I - p_W < 1.0e-3, firedrake.Constant(1.0), firedrake.Constant(0.0))
        )
    else:
        return firedrake.conditional(p_I - p_W < 1.0e-3, firedrake.Constant(1.0), firedrake.Constant(0.0))


def main():
    for physics in physicses:
        for sim in sims:
            for init in inits:
                for fric in frics:
                    for n in ns:
                        if init in ["identical", "standard", "true"]:
                            fn = dirs[physics] + "/{:s}{:s}_{:s}_T-10_n{:2.1f}_{:s}.h5".format(
                                physics, sim, init, n, fric
                            )
                        else:
                            fn = dirs[physics] + "/{:s}{:s}_{:s}_dev1.00_n{:2.1f}_{:s}.h5".format(
                                physics, sim, init, n, fric
                            )
                        if not os.path.exists(fn):
                            print("no", fn)
                            times[physics][sim][init][fric][n] = None
                            continue
                        print(fn)
                        with firedrake.CheckpointFile(fn, "r") as fin:
                            times[physics][sim][init][fric][n] = np.arange(
                                len(fin.get_attr("metadata", "gl"))
                            ) * fin.get_attr("metadata", "dt")
                            gls[physics][sim][init][fric][n] = fin.get_attr("metadata", "gl") / 1000
                            grounded_areas[physics][sim][init][fric][n] = fin.get_attr("metadata", "grounded_area")
                            vafs[physics][sim][init][fric][n] = fin.get_attr("metadata", "vaf")
                            vels[physics][sim][init][fric][n] = fin.get_attr("metadata", "max_vel")

                        mask = gls[physics][sim][init][fric][n] < 1.0
                        for d in [times, gls, vafs, vels]:
                            d[physics][sim][init][fric][n][mask] = np.nan

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(3.5, 3.75))
    fig.subplots_adjust(bottom=0.47, top=0.98, right=0.98, left=0.20)
    plot_pts(axes, times, vafs, targ_times[1:])
    axes[0].set_ylabel("Relative change in V.A.F. (%)                           ")
    fig.savefig("plots/transient_discrete.pdf")

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(3.5, 4.75))
    fig.subplots_adjust(bottom=0.39, top=0.985, right=0.98, left=0.20)
    plot_pts(axes, times, vafs, targ_times, plot_init=True)
    axes[1].set_ylabel("Relative change in V.A.F. (%)")
    fig.savefig("plots/transient_discrete_3p.pdf")


def plot_pts(
    axes,
    times_in,
    vafs_in,
    targ_times,
    labelstuff=True,
    legend=True,
    xlabelall=False,
    plot_init=True,
    phys="",
    plot_inits=["identical", "standard"],
    color_dict=color_dict_T,
):
    times = copy.deepcopy(times_in)
    vafs = copy.deepcopy(vafs_in)

    for ax in axes:
        ax.axhline(0, color="k", zorder=0.5, lw=0.5)
        # ax.set_yscale('symlog', linthresh=0.1)
    for initname in plot_inits:
        for n in ns:
            color = color_dict[initname][n][-10]
            for fricname in frics:
                if fricname == "1":
                    marker = "o"
                    off = -0.1
                elif fricname == "3":
                    marker = "s"
                    off = 0.0
                else:
                    marker = "d"
                    off = 0.1

                suboff = (ns.index(n) - 1) / 8 * 0.1

                for ax, time in zip(axes, targ_times):
                    if (
                        n not in times[phys]["retreat"][initname][fricname]
                        or times[phys]["retreat"][initname][fricname][n] is None
                        or times[phys]["retreat"][initname][fricname][n].ndim == 0
                    ):
                        continue
                    try:
                        index = np.where(abs(times[phys]["retreat"][initname][fricname][n] - time) < 0.1)[0][0]
                    except IndexError:
                        continue

                    if n != 3:
                        if fricname == "RCFi":
                            label = r"{:s} $n$={:2.1f}".format(initname.title(), n)
                        else:
                            label = None
                        ax.plot(
                            2 + off + suboff,
                            (
                                vafs[phys]["retreat"][initname][fricname][n][index]
                                - vafs[phys]["retreat"][initname][fricname][3][index]
                            )
                            / (
                                vafs[phys]["retreat"][initname][fricname][3][index]
                                - vafs[phys]["retreat"][initname][fricname][3][0]
                            )
                            * 1.0e2,
                            color=color,
                            marker=marker,
                            linestyle="None",
                            label=label,
                        )

                    if fricname in ["1", "RCFi"]:
                        if n == 3 and fricname == "RCFi":
                            label = r"{:s} $n$={:2.1f}".format(initname.title(), n)
                        else:
                            label = None
                        if n in vafs[phys]["retreat"][initname]["3"]:
                            ax.plot(
                                3 + off + suboff,
                                (
                                    vafs[phys]["retreat"][initname][fricname][n][index]
                                    - vafs[phys]["retreat"][initname]["3"][n][index]
                                )
                                / (
                                    vafs[phys]["retreat"][initname]["3"][n][index]
                                    - vafs[phys]["retreat"][initname]["3"][n][0]
                                )
                                * 1.0e2,
                                color=color,
                                marker=marker,
                                linestyle="None",
                                label=label,
                            )

                    if (
                        n in times["ssa_"]["retreat"][initname][fricname]
                        and times["ssa_"]["retreat"][initname][fricname][n] is not None
                        and times["ssa_"]["retreat"][initname][fricname][n].ndim != 0
                        and n in times[""]["retreat"][initname][fricname]
                        and times[""]["retreat"][initname][fricname][n] is not None
                        and times[""]["retreat"][initname][fricname][n].ndim != 0
                    ):
                        ax.plot(
                            1 + off + suboff,
                            (
                                vafs["ssa_"]["retreat"][initname][fricname][n][index]
                                - vafs[""]["retreat"][initname][fricname][n][index]
                            )
                            / (
                                vafs[""]["retreat"][initname][fricname][n][index]
                                - vafs[""]["retreat"][initname][fricname][n][0]
                            )
                            * 1.0e2,
                            color=color,
                            marker=marker,
                            linestyle="None",
                        )
                        print(initname, fricname, n,
                            (
                                vafs["ssa_"]["retreat"][initname][fricname][n][index]
                                - vafs[""]["retreat"][initname][fricname][n][index]
                            )
                            / (
                                vafs[""]["retreat"][initname][fricname][n][index]
                                - vafs[""]["retreat"][initname][fricname][n][0]
                            )
                            * 1.0e2)
                    if plot_init:
                        if initname in ["standard"]:
                            if n in vafs[phys]["retreat"]["identical"][fricname]:
                                ax.plot(
                                    4 + off + suboff,
                                    (
                                        vafs[phys]["retreat"][initname][fricname][n][index]
                                        - vafs[phys]["retreat"]["identical"][fricname][n][index]
                                    )
                                    / (
                                        vafs[phys]["retreat"]["identical"][fricname][n][index]
                                        - vafs[phys]["retreat"]["identical"][fricname][n][0]
                                    )
                                    * 1.0e2,
                                    color=color,
                                    marker=marker,
                                    linestyle="None",
                                )

    if legend:
        axes[-1].plot([], [], marker="o", color="k", linestyle="None", label="$m$=1")
        axes[-1].plot([], [], marker="s", color="k", linestyle="None", label="$m$=3")
        axes[-1].plot([], [], marker="d", color="k", linestyle="None", label="RCFi")
        kwargs = {"bbox_to_anchor": (-0.26, -0.5), "ncol": 2, "frameon": False}
        axes[-1].legend(loc="upper left", **kwargs)

    if xlabelall:
        labelthese = axes
    else:
        labelthese = [axes[1]]

    for ax in labelthese:
        if plot_init:
            ax.set_xlim(0.75, 4.25)
            ax.set_xticks([1, 2, 3, 4])
            ax.set_xticklabels(["Diff.\nphysics", "Diff.\n$n$", "Diff.\nsliding", "Diff.\ninit."])
        else:
            ax.set_xlim(0.75, 3.25)
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(["Diff.\nphysics", "Diff.\n$n$", "Diff.\nsliding"])
    for ax in axes:
        ax.set_ylim(-125, 100)

    if labelstuff:
        for ax, time, letter in zip(axes, targ_times, "abcde"):
            ax.text(
                0.01, 0.99, letter + " {:d} yrs".format(time), transform=ax.transAxes, ha="left", va="top", fontsize=12
            )


def plot_init(
    axes, pos=3, vafs=vafs, targ_times=[100, 500], labelstuff=True, legend=True, xlabelall=False, color_dict=color_dict_T
):
    for ax in axes:
        ax.axhline(0, color="k", zorder=0.5, lw=0.5)
        # ax.set_yscale('symlog', linthresh=0.1)
    for initname in inits[:3]:
        for n in ns:
            color = color_dict[initname][n]
            for fricname in frics:
                if fricname == "1":
                    marker = "o"
                    off = -0.1
                elif fricname == "3":
                    marker = "s"
                    off = 0.0
                else:
                    marker = "d"
                    off = 0.1

                suboff = (ns.index(n) - 1) / 8 * 0.1

                for ax, time in zip(axes, targ_times):
                    if (
                        n not in times[""]["retreat"][initname][fricname]
                        or times[""]["retreat"][initname][fricname][n] is None
                        or times[""]["retreat"][initname][fricname][n].ndim == 0
                    ):
                        continue
                    try:
                        index = np.where(abs(times[""]["retreat"][initname][fricname][n] - time) < 0.2)[0][0]
                    except IndexError:
                        continue

                    if initname in ["standard"]:
                        print(
                            initname,
                            fricname,
                            n,
                            vafs[""]["retreat"][initname][fricname][n][index],
                            vafs[""]["retreat"]["identical"][fricname][n][index],
                            vafs[""]["retreat"]["identical"][fricname][n][0],
                        )
                        ax.plot(
                            pos + off + suboff,
                            (
                                vafs[""]["retreat"][initname][fricname][n][index]
                                - vafs[""]["retreat"]["identical"][fricname][n][index]
                            )
                            / (
                                vafs[""]["retreat"]["identical"][fricname][n][index]
                                - vafs[""]["retreat"]["identical"][fricname][n][0]
                            )
                            * 1.0e2,
                            color=color,
                            marker=marker,
                            linestyle="None",
                        )
                        ax.plot(
                            pos + off + suboff,
                            (
                                vafs[""]["retreat"][initname][fricname][n][index]
                                - vafs[""]["retreat"]["identical"][fricname][n][index]
                            )
                            / (
                                vafs[""]["retreat"]["identical"][fricname][n][index]
                                - vafs[""]["retreat"]["identical"][fricname][n][0]
                            )
                            * 1.0e2,
                            color=color,
                            marker=marker,
                            linestyle="None",
                        )


if __name__ == "__main__":
    main()

# coding: utf-8
import os
import firedrake
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from icepackaccs import extract_surface
from discrete_plots import color_dict, plot_pts
from matplotlib import gridspec

from icepack.constants import ice_density as ρ_I, water_density as ρ_W, gravity as g

phys_names = {"": "hybrid", "ssa_": "ssa"}
dirs = {"": "/Volumes/ice_rheology/n4/MISMIP", "ssa_": "/Volumes/ice_rheology/n4/MISMIP"}
physicses = ["", "ssa_"]
sims = ["retreat", "unperturbed"]
lss = {"unperturbed": "solid", "retreat": "dashed"}
lss_fric = {"3": "solid", "RCFi": "dotted", "1": "dashed"}
fricdict = {"3": "$m$=3", "RCFi": "RCFi", "1": "$m$=1"}
inits = ["identical", "standard", "true"]
frics = ["RCFi", "3", "1"]
ns = [1.8, 3, 3.5, 4]

disc_years = [10, 50, 500]

times = {
    physics: {sim: {init: {fric: {} for fric in frics} for init in inits} for sim in sims} for physics in physicses
}
gls = {physics: {sim: {init: {fric: {} for fric in frics} for init in inits} for sim in sims} for physics in physicses}
vafs = {physics: {sim: {init: {fric: {} for fric in frics} for init in inits} for sim in sims} for physics in physicses}
vels = {physics: {sim: {init: {fric: {} for fric in frics} for init in inits} for sim in sims} for physics in physicses}
grounded_areas = {
    physics: {sim: {init: {fric: {} for fric in frics} for init in inits} for sim in sims} for physics in physicses
}

full_gls = {
    physics: {sim: {init: {fric: {n: {} for n in ns} for fric in frics} for init in inits} for sim in sims}
    for physics in physicses
}
full_thicks = {
    physics: {sim: {init: {fric: {n: {} for n in ns} for fric in frics} for init in inits} for sim in sims}
    for physics in physicses
}
full_vels = {
    physics: {sim: {init: {fric: {n: {} for n in ns} for fric in frics} for init in inits} for sim in sims}
    for physics in physicses
}
full_surfs = {
    physics: {sim: {init: {fric: {n: {} for n in ns} for fric in frics} for init in inits} for sim in sims}
    for physics in physicses
}

full_times = [0, 100, 500]

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
                            times[physics][sim][init][fric][n] = None
                            continue
                        print(fn)
                        with firedrake.CheckpointFile(fn, "r") as fin:
                            if physics == "":
                                mesh_name = "fine_mesh"
                            else:
                                mesh_name = "fine_mesh2d"
                            mesh = fin.load_mesh(mesh_name)

                            times[physics][sim][init][fric][n] = np.arange(
                                len(fin.get_attr("metadata", "gl"))
                            ) * fin.get_attr("metadata", "dt")
                            gls[physics][sim][init][fric][n] = fin.get_attr("metadata", "gl") / 1000
                            grounded_areas[physics][sim][init][fric][n] = fin.get_attr("metadata", "grounded_area")
                            vafs[physics][sim][init][fric][n] = fin.get_attr("metadata", "vab")
                            vels[physics][sim][init][fric][n] = fin.get_attr("metadata", "max_vel")
                            print(
                                "Max vel: {:4.2f} m/yr at {:f} yrs, GL ret 100: {:4.1f} km, GL ret 500: {:4.1f} km, vol init {:5.0f} km^3, vol loss 100: {:5.0f} km^3, vol loss final {:5.0f} km^3".format(
                                    np.max(vels[physics][sim][init][fric][n]),
                                    times[physics][sim][init][fric][n][np.argmax(vels[physics][sim][init][fric][n])],
                                    gls[physics][sim][init][fric][n][0]
                                    - gls[physics][sim][init][fric][n][
                                        np.argmin(abs(times[physics][sim][init][fric][n] - 100))
                                    ],
                                    gls[physics][sim][init][fric][n][0] - gls[physics][sim][init][fric][n][-1],
                                    vafs[physics][sim][init][fric][n][0],
                                    vafs[physics][sim][init][fric][n][0]
                                    - vafs[physics][sim][init][fric][n][
                                        np.argmin(abs(times[physics][sim][init][fric][n] - 100))
                                    ],
                                    vafs[physics][sim][init][fric][n][0] - vafs[physics][sim][init][fric][n][-1],
                                )
                            )

                            for time in full_times:
                                if fin.get_attr("metadata", "final_time") >= time:
                                    hist = fin.get_timestepping_history(mesh, "h")
                                    ind = np.where(hist["time"] == time)[0][0]
                                    if physics == "":
                                        fun = extract_surface
                                    else:

                                        def fun(x):
                                            return x

                                    full_vels[physics][sim][init][fric][n][time] = fun(
                                        fin.load_function(mesh, name="u", idx=ind)
                                    )
                                    s = fin.load_function(mesh, name="s", idx=ind)
                                    h = fin.load_function(mesh, name="h", idx=ind)
                                    # full_gls[physics][sim][init][fric][n][time] = firedrake.Function(fun(h).function_space()).interpolate(get_gl(h, s, z_b))
                                    full_gls[physics][sim][init][fric][n][time] = firedrake.Function(
                                        fun(h).function_space()
                                    ).interpolate(get_floating(h, s))
                                    full_thicks[physics][sim][init][fric][n][time] = h
                                    full_surfs[physics][sim][init][fric][n][time] = s

                        mask = gls[physics][sim][init][fric][n] < 1.0
                        for d in [times, gls, vafs, vels]:
                            d[physics][sim][init][fric][n][mask] = np.nan

    {physics: {sim: {init: {fric: {} for fric in frics} for init in inits} for sim in sims} for physics in physicses}

    coords = [
        ("physics", physicses),
        ("n", [1.8, 3, 3.5, 4]),
        ("sim", sims),
        ("fric", ["1", "3", "RCFi"]),
        ("init", inits),
        ("time", times["ssa_"]["unperturbed"]["true"]["3"][3]),
        ("attr", ["vel", "vaf", "gl", "ga"]),
    ]

    da = xr.DataArray(
        np.empty([len(dim[1]) for dim in coords]),
        dims=("physics", "n", "sim", "fric", "init", "time", "attr"),
        coords=coords,
    )
    da[:] = np.nan

    for physics in physicses:
        for sim in sims:
            for init in inits:
                for fric in frics:
                    for n in ns:
                        if n in vels[physics][sim][init][fric]:
                            wheregood = np.array(
                                [
                                    np.argwhere(times[physics][sim][init][fric][n] == t)[0][0]
                                    for t in times["ssa_"]["unperturbed"]["true"]["3"][3]
                                    if t in times[physics][sim][init][fric][n]
                                ]
                            )
                            da.loc[{"n": n, "fric": fric, "attr": "vel", "physics": physics, "sim": sim, "init": init}][
                                : len(wheregood)
                            ] = vels[physics][sim][init][fric][n][wheregood]
                            da.loc[{"n": n, "fric": fric, "attr": "vaf", "physics": physics, "sim": sim, "init": init}][
                                : len(wheregood)
                            ] = vafs[physics][sim][init][fric][n][wheregood]
                            da.loc[{"n": n, "fric": fric, "attr": "gl", "physics": physics, "sim": sim, "init": init}][
                                : len(wheregood)
                            ] = gls[physics][sim][init][fric][n][wheregood]
                            da.loc[{"n": n, "fric": fric, "attr": "ga", "physics": physics, "sim": sim, "init": init}][
                                : len(wheregood)
                            ] = grounded_areas[physics][sim][init][fric][n][wheregood]

    for physics in physicses:
        fig_gl, ax_gl = plt.subplots()
        fig_area, ax_area = plt.subplots()
        fig_vaf, ax_vaf = plt.subplots()
        fig_vel, ax_vel = plt.subplots()
        for sim in sims:
            for init in inits:
                for fric in frics:
                    for n in ns:
                        if times[physics][sim][init][fric][n] is not None:
                            if sim == "unperturbed":
                                label = "{:s} n={:1.1f}".format(init.title(), n)
                            else:
                                label = "_nolegend_"
                            ax_area.plot(
                                times[physics][sim][init][fric][n],
                                grounded_areas[physics][sim][init][fric][n],
                                linestyle=lss[sim],
                                color=color_dict[init][n][-10],
                                label=label,
                            )
                            ax_gl.plot(
                                times[physics][sim][init][fric][n],
                                gls[physics][sim][init][fric][n],
                                linestyle=lss[sim],
                                color=color_dict[init][n][-10],
                                label=label,
                            )
                            ax_vaf.plot(
                                times[physics][sim][init][fric][n],
                                vafs[physics][sim][init][fric][n],
                                linestyle=lss[sim],
                                color=color_dict[init][n][-10],
                                label=label,
                            )
                            ax_vel.plot(
                                times[physics][sim][init][fric][n],
                                vels[physics][sim][init][fric][n],
                                linestyle=lss[sim],
                                color=color_dict[init][n][-10],
                                label=label,
                            )
        ax_area.set_ylabel("Grounded area [1000 km$^3$]")
        ax_gl.set_ylabel("Grounding line [km]")
        ax_vaf.set_ylabel("V.A.F. [km$^3$]")
        ax_vel.set_ylabel("Max. speed [m yr$^{-1}$]")
        for ax in [ax_gl, ax_vaf, ax_vel, ax_area]:
            ax.legend(loc="best")
            ax.set_xlabel("Time [yrs]")

        for sim in ["retreat", "unperturbed"]:

            if sim == "retreat":
                base_dict = {"sim": "unperturbed", "physics": physics}
                contract_dims = ["n", "fric", "init"]
                unpert_terms = da.loc[{"attr": "gl", **base_dict}]
                unpert_vels = da.loc[{"attr": "vel", **base_dict}]
                unpert_vols = da.loc[{"attr": "vaf", **base_dict}]

        for sim in ["retreat"]:
            fig, (ax_gl, ax_vel, ax_vaf) = plt.subplots(3, 1, sharex=True, figsize=(7, 4.5))
            fig = plt.figure(figsize=(7, 6))
            gs = gridspec.GridSpec(
                5,
                6,
                width_ratios=(1, 0.3, 1, 0.3, 0.1, 0.9),
                height_ratios=(1, 1, 1, 0.2, 1),
                top=0.99,
                right=0.99,
                wspace=0.0,
                bottom=0.07,
            )
            ax_vel, ax_gl, ax_vaf = (
                fig.add_subplot(gs[0, :-1]),
                fig.add_subplot(gs[1, :-1]),
                fig.add_subplot(gs[2, :-1]),
            )
            axes_bytime = [fig.add_subplot(gs[4, 0]), fig.add_subplot(gs[4, 2]), fig.add_subplot(gs[4, 4:])]
            plot_pts(
                axes_bytime,
                times,
                vafs,
                disc_years,
                labelstuff=False,
                legend=False,
                xlabelall=True,
                plot_init=False,
                phys=physics,
                plot_inits=["identical"],
            )
            axes_bytime[0].set_ylabel(r"Relative $\Delta$V.A.F. (%)")

            if sim == "retreat" and physics != "":
                base_dict = {"sim": "unperturbed", "physics": physics, "init": "identical"}
                contract_dims = ["n", "fric"]
                unpert_terms = da.loc[{"attr": "gl", **base_dict}]
                unpert_vels = da.loc[{"attr": "vel", **base_dict}]
                unpert_vols = da.loc[{"attr": "vaf", **base_dict}]

                ax_vel.fill_between(
                    da.time,
                    unpert_vels.min(dim=contract_dims),
                    unpert_vels.max(dim=contract_dims),
                    color="0.6",
                    alpha=0.5,
                    label="Unperturbed",
                    ec="none",
                )
                ax_gl.fill_between(
                    da.time,
                    unpert_terms.min(dim=contract_dims),
                    unpert_terms.max(dim=contract_dims),
                    color="0.6",
                    alpha=0.5,
                    label="Unperturbed",
                    ec="none",
                )
                ax_vaf.fill_between(
                    da.time,
                    unpert_vols.min(dim=contract_dims),
                    unpert_vols.max(dim=contract_dims),
                    color="0.6",
                    alpha=0.5,
                    label="Unperturbed",
                    ec="none",
                )

            for init in ["identical", "true"]:
                for fric in frics:
                    for n in ns:
                        if times[physics][sim][init][fric][n] is not None:
                            if fric == "3":
                                label = "{:s} $n$={:1.1f}".format(init.title(), n)
                                ax_vel.plot(
                                    [],
                                    [],
                                    linestyle=lss_fric[fric],
                                    color=color_dict[init][n][-10],
                                    marker="o",
                                    label=label,
                                )
                            else:
                                label = "_nolegend_"
                            ax_gl.plot(
                                times[physics][sim][init][fric][n],
                                gls[physics][sim][init][fric][n],
                                linestyle=lss_fric[fric],
                                color=color_dict[init][n][-10],
                            )
                            ax_vaf.plot(
                                times[physics][sim][init][fric][n],
                                vafs[physics][sim][init][fric][n],
                                linestyle=lss_fric[fric],
                                color=color_dict[init][n][-10],
                            )
                            ax_vel.plot(
                                times[physics][sim][init][fric][n],
                                vels[physics][sim][init][fric][n],
                                linestyle=lss_fric[fric],
                                color=color_dict[init][n][-10],
                            )
            ax_vel.plot([], [], ls="solid", color="0.6", label="$m$=3")
            ax_vel.plot([], [], ls="dashed", color="0.6", label="$m$=1")
            ax_vel.plot([], [], ls="dotted", color="0.6", label="RCFi")

            ax_vel.plot([], [], marker="s", color="k", linestyle="None", label="$m$=3")
            ax_vel.plot([], [], marker="o", color="k", linestyle="None", label="$m$=1")
            ax_vel.plot([], [], marker="d", color="k", linestyle="None", label="RCFi")

            fig.subplots_adjust(right=0.75, top=0.98, left=0.13, bottom=0.11)
            ax_gl.set_ylabel("Grounding\nline (km)")
            ax_vaf.set_ylabel("V.A.F. (km$^3$)")
            ax_vel.set_ylabel("Max. speed\n(m yr$^{-1}$)")
            ax_vel.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
            ax_vaf.set_xlabel("Time (yrs)")
            ax_vel.set_xlim(0, 500)
            ax_vaf.set_xlim(0, 500)
            ax_gl.set_xlim(0, 500)

            ax_gl.xaxis.set_major_formatter(lambda x, y: "")
            ax_vel.xaxis.set_major_formatter(lambda x, y: "")

            for ax in axes_bytime:
                ax.set_ylim(-75, 75)

            for ax, letter, tstr in zip(
                [ax_vel, ax_gl, ax_vaf] + axes_bytime,
                "abcdefgh",
                ["", "", ""] + [" {:d} yrs".format(t) for t in disc_years],
            ):
                ax.text(0.01, 0.99, letter + tstr, ha="left", va="top", transform=ax.transAxes, fontsize=12)
            fig.savefig("plots/transient_ident_{:s}_{:s}_comb.pdf".format(phys_names[physics], sim))
            plt.close(fig)

            for init in ["standard"]:
                fig = plt.figure(figsize=(7, 6))
                gs = gridspec.GridSpec(
                    5,
                    6,
                    width_ratios=(1, 0.3, 1, 0.3, 0.1, 0.9),
                    height_ratios=(1, 1, 1, 0.2, 1),
                    top=0.99,
                    right=0.99,
                    wspace=0.0,
                    bottom=0.07,
                )
                ax_vel, ax_gl, ax_vaf = (
                    fig.add_subplot(gs[0, :-1]),
                    fig.add_subplot(gs[1, :-1]),
                    fig.add_subplot(gs[2, :-1]),
                )
                axes_bytime = [fig.add_subplot(gs[4, 0]), fig.add_subplot(gs[4, 2]), fig.add_subplot(gs[4, 4:])]
                plot_pts(
                    axes_bytime,
                    times,
                    vafs,
                    disc_years,
                    labelstuff=False,
                    legend=False,
                    xlabelall=True,
                    plot_init=True,
                    phys=physics,
                    plot_inits=[init],
                )
                axes_bytime[0].set_ylabel(r"Relative $\Delta$V.A.F. (%)")

                base_dict = {"sim": "unperturbed", "physics": physics, "init": init}
                contract_dims = ["n", "fric"]
                unpert_terms = da.loc[{"attr": "gl", **base_dict}]
                unpert_vels = da.loc[{"attr": "vel", **base_dict}]
                unpert_vols = da.loc[{"attr": "vaf", **base_dict}]

                ax_vel.fill_between(
                    da.time,
                    unpert_vels.min(dim=contract_dims),
                    unpert_vels.max(dim=contract_dims),
                    color="0.6",
                    alpha=0.5,
                    label="Unperturbed",
                    ec="none",
                )
                ax_gl.fill_between(
                    da.time,
                    unpert_terms.min(dim=contract_dims),
                    unpert_terms.max(dim=contract_dims),
                    color="0.6",
                    alpha=0.5,
                    label="Unperturbed",
                    ec="none",
                )
                ax_vaf.fill_between(
                    da.time,
                    unpert_vols.min(dim=contract_dims),
                    unpert_vols.max(dim=contract_dims),
                    color="0.6",
                    alpha=0.5,
                    label="Unperturbed",
                    ec="none",
                )

                for fric in frics:
                    for n in ns:
                        if times[physics][sim][init][fric][n] is not None:
                            if fric == "3":
                                label = "{:s} $n$={:1.1f}".format(init.title(), n)
                                ax_vel.plot(
                                    [],
                                    [],
                                    linestyle=lss_fric[fric],
                                    color=color_dict[init][n][-10],
                                    marker="o",
                                    label=label,
                                )
                            else:
                                label = "_nolegend_"
                            ax_gl.plot(
                                times[physics][sim][init][fric][n],
                                gls[physics][sim][init][fric][n],
                                linestyle=lss_fric[fric],
                                color=color_dict[init][n][-10],
                            )
                            ax_vaf.plot(
                                times[physics][sim][init][fric][n],
                                vafs[physics][sim][init][fric][n],
                                linestyle=lss_fric[fric],
                                color=color_dict[init][n][-10],
                            )
                            ax_vel.plot(
                                times[physics][sim][init][fric][n],
                                vels[physics][sim][init][fric][n],
                                linestyle=lss_fric[fric],
                                color=color_dict[init][n][-10],
                            )
                ax_gl.plot(
                    times[physics]["retreat"]["true"]["3"][3.0],
                    gls[physics]["retreat"]["true"]["3"][3.0],
                    linestyle=lss_fric["3"],
                    color=color_dict["true"][3.0][-10],
                )
                ax_vaf.plot(
                    times[physics]["retreat"]["true"]["3"][3.0],
                    vafs[physics]["retreat"]["true"]["3"][3.0],
                    linestyle=lss_fric["3"],
                    color=color_dict["true"][3.0][-10],
                )
                ax_vel.plot(
                    times[physics]["retreat"]["true"]["3"][3.0],
                    vels[physics]["retreat"]["true"]["3"][3.0],
                    linestyle=lss_fric["3"],
                    color=color_dict["true"][3.0][-10],
                )
                ax_vel.plot(
                    [],
                    [],
                    marker="o",
                    label='"True" ($n$=3)',
                    color=color_dict["true"][3.0][-10],
                )
                ax_vel.plot([], [], ls="solid", color="0.6", label="$m$=3")
                ax_vel.plot([], [], ls="dashed", color="0.6", label="$m$=1")
                ax_vel.plot([], [], ls="dotted", color="0.6", label="RCFi")

                ax_vel.plot([], [], marker="s", color="0.6", linestyle="None", label="$m$=3")
                ax_vel.plot([], [], marker="o", color="0.6", linestyle="None", label="$m$=1")
                ax_vel.plot([], [], marker="d", color="0.6", linestyle="None", label="RCFi")

                fig.subplots_adjust(right=0.75, top=0.98, left=0.13, bottom=0.11)
                ax_gl.set_ylabel("Grounding\nline (km)")
                ax_vaf.set_ylabel("V.A.F. (km$^3$)")
                ax_vel.set_ylabel("Max. speed\n(m yr$^{-1}$)")
                ax_vel.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
                ax_vaf.set_xlabel("Time (yrs)")
                ax_vel.set_xlim(0, 500)
                ax_vaf.set_xlim(0, 500)
                ax_gl.set_xlim(0, 500)

                ax_gl.xaxis.set_major_formatter(lambda x, y: "")
                ax_vel.xaxis.set_major_formatter(lambda x, y: "")

                for ax, letter, tstr in zip(
                    [ax_vel, ax_gl, ax_vaf] + axes_bytime,
                    "abcdefgh",
                    ["", "", ""] + [" {:d} yrs".format(t) for t in disc_years],
                ):
                    ax.text(0.01, 0.99, letter + tstr, ha="left", va="top", transform=ax.transAxes, fontsize=12)
                if init == "standard":
                    axes_bytime[0].set_ylim(-100, 200)
                    axes_bytime[1].set_ylim(-100, 200)
                    axes_bytime[2].set_ylim(-100, 100)
                fig.savefig("plots/transient_{:s}_{:s}_{:s}_comb.pdf".format(init, phys_names[physics], sim))
                plt.close(fig)


if __name__ == "__main__":
    main()

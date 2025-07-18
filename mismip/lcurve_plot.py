#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 dlilien <dlilien@noatak>
#
# Distributed under terms of the MIT license.

"""

"""

import numpy as np

import firedrake

import matplotlib.pyplot as plt
import matplotlib.colors

from initialize_mismip import LC_dict as LC_dict_val, A_scale

u0_coulomb = 250.0
h_t = 50.0
Area = 640e3 * 40e3

field_names = ["surface", "thickness", "velocity"]
min_thick = firedrake.Constant(10.0)


# Needed to avoid assertion error bug in firedrake
# mesh1d = firedrake.IntervalMesh(100, 120)
# mesh_dum = firedrake.ExtrudedMesh(mesh1d, layers=1)

LC_dict = {name: int(np.log10(val)) for name, val in LC_dict_val.items()}


def get_loss(u0):
    def loss_functional(u):
        δu = u - u0
        return 0.5 / Area * ((δu[0]) ** 2 + (δu[1]) ** 2) * firedrake.dx(u0.ufl_domain())

    return loss_functional


def smoothness(C_or_A, scale=1.0):
    return (
        0.5
        / Area
        * firedrake.inner(firedrake.grad(C_or_A) * scale, firedrake.grad(C_or_A) * scale)
        * firedrake.dx(C_or_A.ufl_domain())
    )


def main():

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

    for name, ls, marker in zip(["standard"], ["solid", "dashed"], ["o", "s"]):
        checkpoint_template = "inputs/ssa_{:s}_initialization_mismip_simul.h5"
        checkpoint_fn = checkpoint_template.format(name)
        T_np = -10

        ns = [1.8, 3, 3.5, 4]
        dev = 1.0

        with firedrake.CheckpointFile(checkpoint_fn, "r") as chk:
            fine_mesh = chk.load_mesh("fine_mesh2d")
            u0 = chk.load_function(fine_mesh, name="input_u")
            # C0 = chk.load_function(fine_mesh, name="input_C")

        loss = get_loss(u0)
        exps = np.arange(3, 9)
        smoothnessesC = np.empty((len(exps), len(exps), len(ns)))
        losses = np.empty((len(exps), len(exps), len(ns)))
        smoothnessesA = np.empty((len(exps), len(exps), len(ns)))
        for mat in [smoothnessesC, smoothnessesA, losses]:
            mat[:] = np.nan

        with firedrake.CheckpointFile(checkpoint_fn, "r") as chk:
            for i, LCexp in enumerate(exps):
                for j, LAexp in enumerate(exps):
                    for k, n in enumerate(ns):
                        LC = 10.0**LCexp
                        LA = 10.0**LAexp
                        if name == "standard":
                            inv_name = "T{:d}_n{:2.1f}_LC{:1.0e}_LA{:1.0e}".format(T_np, n, LC, LA)
                        else:
                            inv_name = "dev{:1.2f}_n{:2.1f}_LC{:1.0e}_LA{:1.0e}".format(dev, n, LC, LA)

                        for iternum in range(10):
                            if chk.has_attr("partial", inv_name + "_{:d}".format(iternum)):
                                C1 = chk.load_function(fine_mesh, inv_name + "_C1_{:d}".format(iternum))
                                smoothnessesC[i, j, k] = firedrake.assemble(smoothness(C1))
                                u = chk.load_function(fine_mesh, inv_name + "_u_{:d}".format(iternum))
                                losses[i, j, k] = firedrake.assemble(loss(u))
                                mod_A = chk.load_function(fine_mesh, inv_name + "_modA_{:d}".format(iternum))
                                smoothnessesA[i, j, k] = firedrake.assemble(smoothness(mod_A, scale=A_scale))

        smoothnessesC = smoothnessesC**0.5
        smoothnessesA = smoothnessesA**0.5
        losses = losses**0.5

        if name == "standard":
            these_ns = ns
        else:
            these_ns = [1.8, 3.5, 4]
        if len(these_ns) == 3:
            fig2d, ax2d = plt.subplots(1, len(these_ns), figsize=(7, 3), sharex=True, sharey=True)
            ax2d[0].set_ylabel("$C$ Roughness (MPa yr m$^{-3}$)")
            ax2d[1].set_xlabel("$A$ Roughness (MPa$^{-n}$ yr$^{-1}$ m$^{-2}$)")
            minexp, maxexp = -1, 1
        else:
            fig2d, axd = plt.subplots(2, 2, figsize=(7, 5), sharex=True, sharey=True)
            ax2d = axd.flatten()
            ax2d[0].set_ylabel("$C$ Roughness (MPa yr m$^{-3}$)")
            ax2d[2].set_ylabel("$C$ Roughness (MPa yr m$^{-3}$)")
            ax2d[2].set_xlabel("$A$ Roughness (MPa$^{-n}$ yr$^{-1}$ m$^{-2}$)")
            ax2d[3].set_xlabel("$A$ Roughness (MPa$^{-n}$ yr$^{-1}$ m$^{-2}$)")
            minexp, maxexp = -1, 3
        for kk, n in enumerate(these_ns):
            k = ns.index(n)
            mask = np.isfinite(losses[:, :, k].flatten())
            ax2d[kk].loglog()
            if np.sum(mask) < 3:
                continue
            pcm = ax2d[kk].tricontourf(
                smoothnessesA[:, :, k].flatten()[mask],
                smoothnessesC[:, :, k].flatten()[mask],
                losses[:, :, k].flatten()[mask],
                norm=matplotlib.colors.LogNorm(vmin=10**minexp, vmax=10**maxexp),
                levels=[10**a for a in np.linspace(minexp, maxexp, 256)],
                cmap="nipy_spectral",
            )
            ax2d[kk].plot(
                smoothnessesA[:, :, k].flatten(),
                smoothnessesC[:, :, k].flatten(),
                color="k",
                marker=".",
                linestyle="none",
            )
            for i, LCexp in enumerate(exps):
                for j, LAexp in enumerate(exps):
                    if not np.isnan(losses[i, j, k]):
                        ax2d[kk].text(
                            smoothnessesA[i, j, k],
                            smoothnessesC[i, j, k],
                            r"$10^{:d}, 10^{:d}$".format(LAexp, LCexp),
                            fontsize=4,
                        )
        fig2d.tight_layout(pad=0.5)
        cbr = fig2d.colorbar(pcm, ax=ax2d, location="right", label="Misfit (m yr$^{-1}$)")
        cbr.set_ticks(np.hstack([np.arange(1, 11, 1) * a for a in [10**b for b in range(minexp, maxexp)]]))
        fig2d.savefig("plots/lsurface_{:s}.png".format(name), dpi=300)

        for k, n in enumerate(ns):
            plt.figure()
            plt.loglog()
            for ind_C in range(len(exps)):
                plt.plot(
                    losses[:, ind_C, k], smoothnessesC[:, ind_C, k], marker="o", label="$10^{:d}$".format(exps[ind_C])
                )
                for L, mis, smooth in zip(exps, losses[:, ind_C, k], smoothnessesC[:, ind_C, k]):
                    plt.text(mis, smooth, "$10^{:d}$".format(L), ha="left", va="bottom")
            plt.xlabel("Misfit [m yr$^{-1}$]")
            plt.ylabel("Roughness [MPa yr m$^{-3}$]")
            plt.legend(loc="best")
            plt.savefig("plots/lcurves_{:2.1f}_{:s}_C.pdf".format(n, name))

        for k, n in enumerate(ns):
            plt.figure()
            plt.loglog()
            for ind_A in range(len(exps)):
                plt.plot(
                    losses[ind_A, :, k], smoothnessesA[ind_A, :, k], marker="o", label="$10^{:d}$".format(exps[ind_A])
                )
                for L, mis, smooth in zip(exps, losses[ind_A, :, k], smoothnessesA[ind_A, :, k]):
                    plt.text(mis, smooth, "$10^{:d}$".format(L), ha="left", va="bottom")
            plt.xlabel("Misfit [m yr$^{-1}$]")
            plt.ylabel("Roughness [MPa yr$^{-1}$ m$^{-2}$]")
            plt.legend(loc="best")
            plt.savefig("plots/lcurves_{:2.1f}_{:s}_A.pdf".format(n, name))

        ind_C = np.where(exps == 5)[0][0]

        plt.figure()
        plt.loglog()
        for k, n in enumerate(ns):
            plt.plot(losses[:, ind_C, k], smoothnessesC[:, ind_C, k], marker="o", label="n={:2.1f}".format(n))
            for L, mis, smooth in zip(exps, losses[:, ind_C, k], smoothnessesC[:, ind_C, k]):
                plt.text(mis, smooth, "$10^{:d}$".format(L), ha="left", va="bottom")
            plt.xlabel("Misfit [m yr$^{-1}$]")
            plt.ylabel("Roughness [MPa yr m$^{-3}$]")
            plt.legend(loc="best")
        plt.savefig("plots/lcurves_{:s}_C.pdf".format(name))

        plt.figure()
        plt.loglog()
        for k, n in enumerate(ns):
            ind_A = np.where(exps == LC_dict[n])[0][0]
            print(f"{n} {name}: {losses[ind_A, :, k]}")
            plt.plot(losses[ind_A, :, k], smoothnessesA[ind_A, :, k], marker="o", label="n={:2.1f}".format(n))
            for L, mis, smooth in zip(exps, losses[ind_A, :, k], smoothnessesA[ind_A, :, k]):
                plt.text(mis, smooth, "$10^{:d}$".format(L), ha="left", va="bottom")
            plt.xlabel("Misfit [m yr$^{-1}$]")
            plt.ylabel("Roughness [MPa yr$^{-1}$ m$^{-2}$]")
            plt.legend(loc="best")
        plt.savefig("plots/lcurves_{:s}_A.pdf".format(name))

        plt.figure()
        plt.loglog()
        for k, n in enumerate(ns):
            ind_A = np.where(exps == LC_dict[n])[0][0]
            plt.plot(
                losses[:, ind_C, k],
                smoothnessesC[:, ind_C, k],
                marker="o",
                label="n={:2.1f}".format(n),
                color="C{:d}".format(k),
            )
            plt.plot(
                losses[ind_A, :, k], smoothnessesA[ind_A, :, k], marker="s", linestyle="dashed", color="C{:d}".format(k)
            )
            for L, mis, smooth in zip(exps, losses[:, ind_C, k], smoothnessesC[:, ind_C, k]):
                plt.text(mis, smooth, "$10^{:d}$".format(L), ha="left", va="bottom")
            for L, mis, smooth in zip(exps, losses[ind_A, :, k], smoothnessesA[ind_A, :, k]):
                plt.text(mis, smooth, "$10^{:d}$".format(L), ha="left", va="bottom")
        plt.plot([], [], color="0.6", marker="o", label="$C_1$")
        plt.plot([], [], color="0.6", linestyle="dashed", marker="s", label="$A$")
        plt.xlabel("Misfit [m yr$^{-1}$]")
        plt.ylabel("Roughness [MPa yr m$^{-3}$ ($C_1$) or MPa$^{-n}$ yr$^{-1}$ m$^{-2}$ ($A$)]")
        plt.legend(loc="best")
        plt.savefig("plots/lcurves_{:s}.pdf".format(name))

        minreg = [1.0e-7, 1.0e-6]
        for k, n in enumerate(ns):
            ind_A = np.where(exps == LC_dict[n])[0][0]
            label = "n={:2.1f}".format(n)
            ax1.plot(
                losses[:, ind_C, k],
                smoothnessesC[:, ind_C, k],
                marker=marker,
                label=label,
                color="C{:d}".format(k),
                linestyle=ls,
            )
            ax2.plot(
                losses[ind_A, :, k], smoothnessesA[ind_A, :, k], marker=marker, linestyle=ls, color="C{:d}".format(k)
            )
            for L, mis, smooth in zip(exps, losses[:, ind_C, k], smoothnessesC[:, ind_C, k]):
                if n == 3.0 or n == 3.5:
                    ha = "right"
                    va = "top"
                else:
                    ha = "left"
                    va = "bottom"
                if smooth > minreg[0]:
                    ax1.text(mis, smooth, "$10^{:d}$".format(L), ha=ha, va=va)
            for L, mis, smooth in zip(exps, losses[ind_A, :, k], smoothnessesA[ind_A, :, k]):
                if smooth > minreg[1]:
                    ax2.text(mis, smooth, "$10^{:d}$".format(L), ha="left", va="bottom")

    for ax, letter in zip((ax1, ax2), "abcd"):
        ax.loglog()
        ax.text(0.01, 0.99, letter, transform=ax.transAxes, ha="left", va="top", fontsize=14)
    ax2.set_xlabel("Misfit (m yr$^{-1}$)")
    ax1.set_ylim(minreg[0], 1e-5)
    ax2.set_ylim(minreg[1], 3e-3)
    ax2.set_xlim(0.1, 300)

    ax1.legend(loc="upper right", ncol=2)
    ax1.set_ylabel("Roughness (MPa yr m$^{-3}$)")
    ax2.set_ylabel("Roughness (MPa$^{-n}$ yr$^{-1}$ m$^{-2}$)")
    fig.tight_layout(pad=0.1)
    fig.savefig("plots/all_lcurves.pdf")


if __name__ == "__main__":
    main()

#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 dlilien <dlilien@noatak>
#
# Distributed under terms of the MIT license.

"""

"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import firedrake
from icepackaccs import extract_surface


ns = [1.8, 3, 3.5, 4]
T = -10
LCaps = [10.0**i for i in range(3, 8)]
full_all_outs = {init: {n: {"smoothness": [], "misfit": [], "L": []} for n in ns} for init in ["standard"]}

for init in full_all_outs:
    # fn = "inputs/lcurve_inversion_results_{:s}_bumps2.h5".format(init)
    all_names = []
    for n in ns:
        fn = "inputs/lcurve_inversion_results_{:s}_bumps2.h5".format("standard")

        with h5py.File(fn) as fin:
            for L in LCaps:
                name = "T{:d}_n{:2.1f}_L{:2.1e}".format(T, n, L)
                if name in fin:
                    all_names.append(name)
                    group = fin[name]
                    full_all_outs[init][n]["smoothness"].append(fin[name].attrs["smoothness"])
                    full_all_outs[init][n]["misfit"].append(fin[name].attrs["loss"])
                    full_all_outs[init][n]["L"].append(L)
                else:
                    continue

    for n in ns:
        plt.figure()
        plt.loglog()
        plt.title("{:s} $n$={:2.1f}".format(init, n))
        misfit = np.array(full_all_outs[init][n]["misfit"]) ** 0.5
        smoothness = np.array(full_all_outs[init][n]["smoothness"]) ** 0.5
        Ls = np.array(full_all_outs[init][n]["L"])
        plt.plot(misfit, smoothness, marker="o", label="n={:2.1f}".format(n))
        for L, smooth, mis in zip(Ls, smoothness, misfit):
            plt.text(mis, smooth, "{:2.1e}".format(L), ha="left", va="bottom")
        plt.xlabel("Misfit [m yr$^{-1}$]")
        plt.ylabel("Smoothness [MPa yr m$^{-2}$]")

    vels = []
    cs = []
    if False:
        with firedrake.CheckpointFile(fn, "r") as chk:
            mesh = chk.load_mesh("flowline")
            u0 = chk.load_function(mesh, "input_u")
            for name in all_names:
                print(name)
                vels.append(chk.load_function(mesh, name + "_u1"))
                cs.append(chk.load_function(mesh, name + "_C1"))

        for n in ns:
            fig, (ax1, ax2) = plt.subplots(2, sharex=True)
            plt.title("{:s} $n$={:2.1f}".format(init, n))
            firedrake.plot(extract_surface(u0), axes=ax1, label="True", edgecolor="k")
            i = 0
            for vel, name, C in zip(vels, all_names, cs):
                if name.split("_")[1][1:] == "{:2.1f}".format(n):
                    firedrake.plot(extract_surface(vel), axes=ax1, label=name, edgecolor="C{:d}".format(i))
                    firedrake.plot(
                        extract_surface(firedrake.Function(C.function_space()).interpolate(C**2.0)),
                        axes=ax2,
                        label=name,
                        edgecolor="C{:d}".format(i),
                    )
                    i += 1

            ax1.legend(loc="best")
            ax2.set_ylim(0, 0.5)

        for n in ns:
            fig, (ax1, ax2) = plt.subplots(2, sharex=True)
            plt.title("{:s} $n$={:2.1f}".format(init, n))
            firedrake.plot(extract_surface(u0), axes=ax1, label="True", edgecolor="k")
            for vel, name, C in zip(vels, all_names, cs):
                if name.split("_")[1][1:] == "{:2.1f}".format(n) and name[-1] == "1":
                    firedrake.plot(extract_surface(vel), axes=ax1, label=name)
                    firedrake.plot(
                        extract_surface(firedrake.Function(C.function_space()).interpolate(C**2.0)),
                        axes=ax2,
                        label=name,
                    )

            ax1.legend(loc="best")
            ax2.set_ylim(0, 0.5)

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog()
for init, ls, marker in [("standard", "solid", "o")]:
    all_outs = full_all_outs[init]
    for i, n in enumerate(ns):
        if init == "standard":
            label = "n={:2.1f}".format(n)
        else:
            label = None
        misfit = np.array(all_outs[n]["misfit"]) ** 0.5
        smoothness = np.array(all_outs[n]["smoothness"]) ** 0.5
        Ls = np.array(all_outs[n]["L"])
        ax.plot(misfit, smoothness, marker=marker, linestyle=ls, label=label, color="C{:d}".format(i))
        for L, smooth, mis in zip(Ls, smoothness, misfit):
            ax.text(mis, smooth, "$10^{:d}$".format(int(np.log10(L))), ha="left", va="bottom")

ax.plot([], [], color="0.6", ls="solid", label="Standard")
ax.set_xlabel("Misfit (m yr$^{-1}$)")
ax.set_ylabel("Smoothness (MPa yr m$^{-2}$)")
ax.legend(loc="best")
fig.savefig("figs/lcurve_standard.pdf")
plt.show()

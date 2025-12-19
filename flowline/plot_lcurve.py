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
import matplotlib.ticker

# Need to muck around to use color consistently outside a package
import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from common_colors import ncolors


ns = [1.8, 3, 3.5, 4]
T = -10
LCaps = [10.0**i for i in range(3, 8)]
full_all_outs = {init: {n: {"smoothness": [], "misfit": [], "L": []} for n in ns} for init in ["standard"]}

for init in full_all_outs:
    fn = "inputs/lcurve_inversion_results_{:s}_bumps2.h5".format(init)
    all_names = []
    for n in ns:
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

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
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
        ax.plot(misfit, smoothness, marker=marker, linestyle=ls, label=label, color=ncolors[i])
        for L, smooth, mis in zip(Ls, smoothness, misfit):
            if n == 4 or (n == 1.8 and L == 1.0e3):
                ha = "right"
            else:
                ha = "left"
            ax.text(mis, smooth, "$10^{:d}$".format(int(np.log10(L))), ha=ha, va="bottom")

ax.set_xlabel("Misfit (m yr$^{-1}$)")
ax.set_ylabel(r"Roughness (MPa$^{-\frac{1}{2}}$ yr$^{-\frac{1}{2}}$ m$^{-\frac{5}{2}}$)")
ax.legend(loc="best")

fig.tight_layout(pad=0.1)

axz = fig.add_axes([0.4, 0.6, 0.35, 0.35])

axz.loglog()
for init, ls, marker in [("standard", "solid", "o")]:
    all_outs = full_all_outs[init]
    for i, n in enumerate(ns):
        if n != 3.5:
            continue
        if init == "standard":
            label = "n={:2.1f}".format(n)
        else:
            label = None
        misfit = np.array(all_outs[n]["misfit"]) ** 0.5
        smoothness = np.array(all_outs[n]["smoothness"]) ** 0.5
        Ls = np.array(all_outs[n]["L"])
        axz.plot(misfit, smoothness, marker=marker, linestyle=ls, label=label, color=ncolors[i])
        for L, smooth, mis in zip(Ls, smoothness, misfit):
            axz.text(mis, smooth, "$10^{:d}$".format(int(np.log10(L))), ha="left", va="bottom")

axz.set_xlim(53, 58)
axz.get_xaxis().set_ticklabels([], minor=True)

ax.indicate_inset_zoom(axz)

axz.set_xticks([53, 60])
axz.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())



fig.savefig("figs/lcurve_standard.pdf")

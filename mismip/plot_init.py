# coding: utf-8
import firedrake
import numpy as np
import matplotlib.pyplot as plt
from icepackaccs import extract_surface, extract_bed
from icepack.constants import ice_density as ρ_I, water_density as ρ_W, gravity as g
from libmismip import mismip_melt, mirrored_tripcolor, mirrored_tricontour
from icepackaccs.mismip import mismip_bed_topography
from matplotlib import gridspec
import matplotlib.colors as mcolors


# Needed to avoid assertion error bug in firedrake
mesh1d = firedrake.IntervalMesh(100, 120)
firedrake.ExtrudedMesh(mesh1d, layers=1)


class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        v_ext = np.max( [ np.abs(self.vmin), np.abs(self.vmax) ] )
        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def kmfmt(x, pos):
    return "{:d}".format(int(x / 1000.0))


def steady_state():
    fig, axes = plt.subplots(
        4,
        2,
        gridspec_kw={
            "width_ratios": (1, 0.02),
            "left": 0.08,
            "top": 0.98,
            "bottom": 0.11,
            "wspace": 0.08,
            "hspace": 0.41,
        },
        figsize=(7.0, 4.05),
    )
    deg4_fn = "inputs/mismip_-10C_n3.h5"

    with firedrake.CheckpointFile(deg4_fn, "r") as chk:
        fine_mesh = chk.load_mesh("fine_mesh")
        u0 = chk.load_function(fine_mesh, "velocity")
        h0 = chk.load_function(fine_mesh, "thickness")
        s0 = chk.load_function(fine_mesh, "surface")

    ssa_fn = "inputs/ssa_standard_initialization_mismip.h5"
    with firedrake.CheckpointFile(ssa_fn, "r") as chk:
        ssa_mesh = chk.load_mesh("fine_mesh2d")
        u0_ssa = chk.load_function(ssa_mesh, "input_u_ssa")

    p_W = ρ_W * g * firedrake.max_value(0, h0 - s0)
    p_I = ρ_I * g * h0
    Q = h0.function_space()
    is_floating = extract_surface(
        firedrake.Function(Q).interpolate(
            firedrake.conditional(p_I - p_W < 1.0e-3, firedrake.Constant(1.0), firedrake.Constant(0.0))
        )
    )
    b = firedrake.Function(Q).interpolate(mismip_bed_topography(fine_mesh))

    cm_thick = mirrored_tripcolor(extract_surface(h0), axes=axes[0][0], vmin=0, vmax=2000, cmap="viridis")
    mirrored_tricontour(is_floating, levels=[0.5], axes=axes[0][0], colors="k")
    plt.colorbar(cm_thick, cax=axes[0][1], label="H (m)")

    cm_vel = mirrored_tripcolor(extract_surface(u0[0]), axes=axes[1][0], vmin=0, vmax=1000, cmap="Reds")
    mirrored_tricontour(is_floating, levels=[0.5], axes=axes[1][0], colors="k")
    plt.colorbar(cm_vel, cax=axes[1][1], label="$u_x$ (m yr$^{-1}$)", extend="max")

    u0_s = extract_surface(u0[0])
    du = firedrake.Function(u0_s.function_space()).interpolate(
        (u0_s - firedrake.Function(u0_s.function_space()).interpolate(u0_ssa[0])) / u0_s * 100
    )
    cm_perc = mirrored_tripcolor(du, axes=axes[2][0], vmin=-10, vmax=10, cmap="PiYG")
    mirrored_tricontour(is_floating, levels=[0.5], axes=axes[2][0], colors="k")
    plt.colorbar(cm_perc, cax=axes[2][1], label=r"$\Delta u$ (%)", extend="both")

    melt = mismip_melt(s0, h0, b)
    cm_melt = mirrored_tripcolor(extract_surface(melt), axes=axes[3][0], vmin=0, vmax=100, cmap="Oranges")
    mirrored_tricontour(is_floating, levels=[0.5], axes=axes[3][0], colors="k")
    plt.colorbar(cm_melt, cax=axes[3][1], label="Melt (m yr$^{-1}$)")

    for ax, letter in zip(axes.flatten()[::2], "abcdef"):
        ax.axis("equal")
        ax.set_ylim(0, 8e4)
        ax.set_xlim(0, 64e4)

        ax.set_yticks([0e3, 40e3, 80e3])
        ax.set_yticklabels(["-40", "0", "40"])
        ax.xaxis.set_major_formatter(kmfmt)
        ax.text(0.01, 0.99, letter, ha="left", va="top", transform=ax.transAxes)
    for ax, letter in zip(axes[:-1].flatten()[::2], "abcdef"):
        ax.xaxis.set_major_formatter(lambda x, y: "")

    axes[2, 0].set_ylabel("                 Distance (km)")
    axes[-1, 0].set_xlabel("Distance (km)")

    fig.savefig("plots/mismip_init.png", dpi=300)


def inversion_results_alln():
    for phys in [
        "",
        "ssa_",
    ]:
        gs = gridspec.GridSpec(
            7,
            9,
            width_ratios=(0.5, 0.2, 0.5, 0.2, 0.035, 0.215, 0.1, 0.5, 0.35),
            height_ratios=[1, 1, 1, 1, 1, 0.3, 0.12],
            wspace=0.0,
            left=0.075,
            right=0.995,
            top=0.990,
            bottom=0.085,
        )
        fig = plt.figure(figsize=(7.0, 5.58))
        ax_true = fig.add_subplot(gs[0, :3])

        cax_slide = fig.add_subplot(gs[0, 4])
        cax_dslide = fig.add_subplot(gs[6, 0])
        cax_A = fig.add_subplot(gs[6, 2])
        cax_vel = fig.add_subplot(gs[6, 7])

        ax_slide_standard3 = fig.add_subplot(gs[2, :3])
        ax_vel_standard3 = fig.add_subplot(gs[2, 4:])

        ax_slide_standard35 = fig.add_subplot(gs[3, :3])
        ax_vel_standard35 = fig.add_subplot(gs[3, 4:])

        ax_slide_standard4 = fig.add_subplot(gs[4, :3])
        ax_vel_standard4 = fig.add_subplot(gs[4, 4:])

        ax_slide_standard1 = fig.add_subplot(gs[1, :3])
        ax_vel_standard1 = fig.add_subplot(gs[1, 4:])

        deg4_fn = "inputs/mismip_-10C_n3.h5"

        with firedrake.CheckpointFile(deg4_fn, "r") as chk:
            fine_mesh = chk.load_mesh("fine_mesh")
            u0 = chk.load_function(fine_mesh, "velocity")
            h0 = chk.load_function(fine_mesh, "thickness")
            s0 = chk.load_function(fine_mesh, "surface")

        standard_fn = "inputs/{:s}standard_initialization_mismip.h5".format(phys)

        if phys == "":
            meshname = "fine_mesh"
        else:
            meshname = "fine_mesh2d"

        with firedrake.CheckpointFile(standard_fn, "r") as chk:
            mesh = chk.load_mesh(meshname)
            standard_u_4 = chk.load_function(mesh, "T-10_n4.0_u1")
            standard_A_4 = chk.load_function(mesh, "T-10_n4.0_modA")
            standard_u_35 = chk.load_function(mesh, "T-10_n3.5_u1")
            standard_A_35 = chk.load_function(mesh, "T-10_n3.5_modA")
            standard_u_3 = chk.load_function(mesh, "T-10_n3.0_u1")
            standard_A_3 = chk.load_function(mesh, "T-10_n3.0_modA")
            standard_u_1 = chk.load_function(mesh, "T-10_n1.8_u1")
            standard_A_1 = chk.load_function(mesh, "T-10_n1.8_modA")

        p_W = ρ_W * g * firedrake.max_value(0, h0 - s0)
        p_I = ρ_I * g * h0
        Q = h0.function_space()
        Qv = firedrake.FunctionSpace(fine_mesh, "CG", 2, vfamily="GL", vdegree=2)
        if phys == "":
            Qvi = firedrake.FunctionSpace(fine_mesh, "CG", 2, vfamily="GL", vdegree=4)
            Q4 = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="GL", vdegree=4)
            Q2d = extract_surface(firedrake.Function(Q4).interpolate(firedrake.Constant(0.0))).function_space()

            def interp(a):
                return firedrake.Function(Q2d).interpolate(a)

            getbed = extract_bed
            getsurf = extract_surface

            def getothersurf(a):
                return extract_surface(firedrake.Function(Q4).interpolate(a))

            def getothersurf2(a):
                return extract_surface(firedrake.Function(Q4).interpolate(a))

        else:
            Qvi = firedrake.FunctionSpace(mesh, "CG", 2)

            def interp(a):
                return firedrake.Function(Qvi).interpolate(a)

            def getsurf(a):
                return interp(a)

            def getbed(a):
                return firedrake.Function(Qvi).interpolate(a)

            def getothersurf(a):
                return interp(extract_surface(a))

            def getothersurf2(a):
                return interp(a)

        is_floating = extract_bed(
            firedrake.Function(Q).interpolate(
                firedrake.conditional(p_I - p_W < 1.0e-3, firedrake.Constant(1.0), firedrake.Constant(0.0))
            )
        )

        def mask_A(A):
            A2d = firedrake.Function(is_floating.function_space()).interpolate(
                (firedrake.exp(getsurf(A) * 100) - 1) * 100
            )
            A2d.dat.data[is_floating.dat.data < 0.5] = np.nan
            return A2d

        plim = 20
        vlim = 5

        umag0 = firedrake.Function(Qv).interpolate(firedrake.sqrt(firedrake.dot(u0, u0)))
        p0 = interp((extract_bed(umag0) / extract_surface(umag0)))
        p0 = interp(p0)
        cm_perc = mirrored_tripcolor(
            firedrake.Function(extract_surface(umag0).function_space()).interpolate(p0 * 100),
            axes=ax_true,
            vmin=80,
            vmax=100,
            cmap="Blues",
        )
        cbr_perc = plt.colorbar(cm_perc, cax=cax_slide, extend="min")

        for n, init, u, A, axv, axs in zip(
            [1.8, 3, 3.5, 4, 1.8, 3.5, 4],
            ["std", "std", "std", "std"],
            [standard_u_1, standard_u_3, standard_u_35, standard_u_4],
            [standard_A_1, standard_A_3, standard_A_35, standard_A_4],
            [ax_vel_standard1, ax_vel_standard3, ax_vel_standard35, ax_vel_standard4],
            [ax_slide_standard1, ax_slide_standard3, ax_slide_standard35, ax_slide_standard4],
        ):
            umag = firedrake.Function(Qvi).interpolate(firedrake.sqrt(firedrake.dot(u, u)))
            p = interp(getbed(umag) / getsurf(umag))
            dp = interp((p - p0) / p0 * 100)
            cm_dp = mirrored_tripcolor(dp, axes=axs, vmin=-plim, vmax=plim, cmap="seismic")
            # cm_A = mirrored_tripcolor(mask_A(A), axes=axs, cmap="BrBG", norm=mcolors.SymLogNorm(linthresh=1.0, linscale=1.0, vmin=-Alim, vmax=Alim, base=10))
            normA = MidpointNormalize(vmin=-100, vmax=100, midpoint=0)
            cm_A = mirrored_tripcolor(mask_A(A), axes=axs, cmap="BrBG",  norm=normA)
            du = interp(getothersurf2(u[0]) - getothersurf(u0[0]))
            cm_vel = mirrored_tripcolor(du, axes=axv, vmin=-vlim, vmax=vlim, cmap="PiYG")

            rms = (
                firedrake.assemble(du**2.0 * firedrake.dx)
                / firedrake.assemble(firedrake.Function(Qvi).interpolate(1) * firedrake.dx)
            ) ** 0.5
            print(
                "Phys {:s}, n={:2.1f}, {:s}: max misfit: {:f} m/yr, rms: {:f} m/yr".format(
                    phys, n, init, abs(du.dat.data[:]).max(), rms
                )
            )

        cbr_dp = plt.colorbar(cm_dp, cax=cax_dslide, extend="both", orientation="horizontal")
        cbr_A = plt.colorbar(cm_A, cax=cax_A, extend="max", orientation="horizontal")
        cbr_vel = plt.colorbar(cm_vel, cax=cax_vel, extend="both", orientation="horizontal")

        cbr_perc.set_label(label="Sliding (%)", fontsize=9)
        cbr_dp.set_label(label=r"$\Delta$Sliding (%)", fontsize=9)
        cbr_A.set_label(label=r"$\Delta A$ (%)", fontsize=9)
        cbr_vel.set_label(label=r"$\Delta u_x$ (m yr$^{-1}$)", fontsize=9)
        for cbr in [cbr_perc, cbr_dp, cbr_A, cbr_vel]:
            cbr.ax.tick_params(labelsize=9)

        for ax, letter, name in zip(
            [
                ax_true,
                ax_slide_standard1,
                ax_slide_standard3,
                ax_slide_standard35,
                ax_slide_standard4,
                ax_vel_standard1,
                ax_vel_standard3,
                ax_vel_standard35,
                ax_vel_standard4,
            ],
            "abcdefghijklmnopq",
            [
                "",
                "Standard $n$=1.8",
                "Standard $n$=3",
                "Standard $n$=3.5",
                "Standard $n$=4",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ],
        ):
            mirrored_tricontour(is_floating, levels=[0.5], axes=ax, colors="k")
            ax.axis("equal")
            ax.set_ylim(0, 8e4)
            ax.set_xlim(34e4, 64e4)

            ax.set_yticks([0e3, 40e3, 80e3])
            ax.set_yticklabels(["-40", "0", "40"])
            ax.text(0.01, 0.99, letter + " " + name, ha="left", va="top", transform=ax.transAxes)
            ax.xaxis.set_major_formatter(lambda x, y: "")

        ax_slide_standard4.xaxis.set_major_formatter(kmfmt)
        ax_vel_standard4.xaxis.set_major_formatter(kmfmt)

        ax_slide_standard3.set_ylabel("Distance (km)")

        ax_slide_standard4.set_xlabel("Distance (km)")
        ax_vel_standard4.set_xlabel("Distance (km)")
        fig.savefig("plots/{:s}mismip_inversion.png".format(phys), dpi=300)


def inversion_results_trueident():
    for phys in ["ssa_"]:
        gs = gridspec.GridSpec(
            5,
            8,
            width_ratios=(1, 0.2, 0.033333, 0.3, 0.033333, 0.3, 0.033333, 0.3),
            height_ratios=(1, 1, 1, 0.35, 0.1),
            wspace=0.0,
            left=0.075,
            right=0.995,
            top=0.98,
            bottom=0.130,
        )
        fig = plt.figure(figsize=(7.0, 3.75))
        ax_true = fig.add_subplot(gs[0, 0])

        cax_slide = fig.add_subplot(gs[0, 2])
        cax_dslide = fig.add_subplot(gs[4, 0])
        cax_vel = fig.add_subplot(gs[4, 2:])

        ax_slide_true3 = fig.add_subplot(gs[1, 0])
        ax_vel_true3 = fig.add_subplot(gs[1, 2:])

        ax_slide_ident = fig.add_subplot(gs[2, 0])
        ax_vel_ident = fig.add_subplot(gs[2, 2:])

        deg4_fn = "inputs/mismip_-10C_n3.h5"

        with firedrake.CheckpointFile(deg4_fn, "r") as chk:
            fine_mesh = chk.load_mesh("fine_mesh")
            u0 = chk.load_function(fine_mesh, "velocity")
            h0 = chk.load_function(fine_mesh, "thickness")
            s0 = chk.load_function(fine_mesh, "surface")

        true_fn = "inputs/{:s}true_initialization_mismip.h5".format(phys)
        ident_fn = "inputs/{:s}identical_initialization_mismip.h5".format(phys)

        if phys == "":
            meshname = "fine_mesh"
        else:
            meshname = "fine_mesh2d"

        with firedrake.CheckpointFile(true_fn, "r") as chk:
            mesh = chk.load_mesh(meshname)
            true_u_3 = chk.load_function(mesh, "T-10_n3.0_u1")

        with firedrake.CheckpointFile(ident_fn, "r") as chk:
            mesh = chk.load_mesh(meshname)
            ident_u = chk.load_function(mesh, "T-10_n3.0_u1")

        p_W = ρ_W * g * firedrake.max_value(0, h0 - s0)
        p_I = ρ_I * g * h0
        Q = h0.function_space()
        Qv = firedrake.FunctionSpace(fine_mesh, "CG", 2, vfamily="GL", vdegree=2)
        if phys == "":
            Qvi = firedrake.FunctionSpace(fine_mesh, "CG", 2, vfamily="GL", vdegree=4)
            Q4 = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="GL", vdegree=4)

            def func(a):
                return extract_surface(firedrake.Function(Q4).interpolate(a))

            def bfunc(a):
                return extract_bed(firedrake.Function(Q4).interpolate(a))

            def dfunc(a):
                return extract_surface(firedrake.Function(Q4).interpolate(a))

        else:
            Qvi = firedrake.FunctionSpace(mesh, "CG", 2)

            def func(a):
                return firedrake.Function(Qvi).interpolate(a)

            def bfunc(a):
                return firedrake.Function(Qvi).interpolate(a)

            def dfunc(a):
                return func(extract_surface(a))

        is_floating = extract_bed(
            firedrake.Function(Q).interpolate(
                firedrake.conditional(p_I - p_W < 1.0e-3, firedrake.Constant(1.0), firedrake.Constant(0.0))
            )
        )

        def mask_A(A):
            A2d = firedrake.Function(is_floating.function_space()).interpolate((firedrake.exp(func(A) * 100) - 1) * 100)
            A2d.dat.data[is_floating.dat.data < 0.5] = np.nan
            return A2d

        plim = 20
        vlim = 50

        umag0 = firedrake.Function(Qv).interpolate(firedrake.sqrt(firedrake.dot(u0, u0)))
        p0 = firedrake.Function(extract_surface(umag0).function_space()).interpolate(
            (extract_bed(umag0) / extract_surface(umag0))
        )
        if phys == "ssa_":
            p0 = func(p0)
        cm_perc = mirrored_tripcolor(
            firedrake.Function(extract_surface(umag0).function_space()).interpolate(p0 * 100),
            axes=ax_true,
            vmin=80,
            vmax=100,
            cmap="Blues",
        )
        cbr_perc = plt.colorbar(cm_perc, cax=cax_slide, extend="min")

        umag = firedrake.Function(Qvi).interpolate(firedrake.sqrt(firedrake.dot(true_u_3, true_u_3)))
        p = firedrake.Function(func(umag).function_space()).interpolate((bfunc(umag) / func(umag)))
        cm_dp = mirrored_tripcolor(
            firedrake.Function(func(umag).function_space()).interpolate((p - p0) / p0 * 100),
            axes=ax_slide_true3,
            vmin=-plim,
            vmax=plim,
            cmap="seismic",
        )
        cbr_dp = plt.colorbar(cm_dp, cax=cax_dslide, extend="both", orientation="horizontal")
        cm_vel = mirrored_tripcolor(
            func(func(true_u_3[0]) - dfunc(u0[0])), axes=ax_vel_true3, vmin=-vlim, vmax=vlim, cmap="PiYG"
        )
        cbr_vel = plt.colorbar(cm_vel, cax=cax_vel, extend="both", orientation="horizontal")
        du = func(func(true_u_3[0]) - dfunc(u0[0]))
        rms = (
            firedrake.assemble(du**2.0 * firedrake.dx)
            / firedrake.assemble(firedrake.Function(Qvi).interpolate(1) * firedrake.dx)
        ) ** 0.5
        print(
            "With true initialization, misfits are {:f} m/yr max, {:f} m/yr rms".format(abs(du.dat.data[:]).max(), rms)
        )

        du = func(func(ident_u[0]) - dfunc(u0[0]))
        rms = (
            firedrake.assemble(du**2.0 * firedrake.dx)
            / firedrake.assemble(firedrake.Function(Qvi).interpolate(1) * firedrake.dx)
        ) ** 0.5
        print(
            "With identical initialization, misfits are {:f} m/yr max, {:f} m/yr rms".format(
                abs(du.dat.data[:]).max(), rms
            )
        )

        umag = firedrake.Function(Qvi).interpolate(firedrake.sqrt(firedrake.dot(ident_u, ident_u)))
        p = firedrake.Function(func(umag).function_space()).interpolate((bfunc(umag) / func(umag)))
        cm_dp = mirrored_tripcolor(
            firedrake.Function(func(umag).function_space()).interpolate((p - p0) / p0 * 100),
            axes=ax_slide_ident,
            vmin=-plim,
            vmax=plim,
            cmap="seismic",
        )
        mirrored_tripcolor(func(func(ident_u[0]) - dfunc(u0[0])), axes=ax_vel_ident, vmin=-vlim, vmax=vlim, cmap="PiYG")

        cbr_perc.set_label(label="Sliding (%)", fontsize=9)
        cbr_dp.set_label(label=r"$\Delta$Sliding (%)", fontsize=9)
        cbr_vel.set_label(label=r"$\Delta u_x$ (m yr$^{-1}$)", fontsize=9)
        for cbr in [cbr_perc, cbr_dp, cbr_vel]:
            cbr.ax.tick_params(labelsize=9)

        for ax, letter, name in zip(
            [ax_true, ax_slide_true3, ax_slide_ident, ax_vel_true3, ax_vel_ident],
            "abcdefghijklmnopq",
            [
                "",
                "True $n$=3",
                "Identical",
                "",
                "",
                "",
            ],
        ):
            mirrored_tricontour(is_floating, levels=[0.5], axes=ax, colors="k")
            ax.axis("equal")
            ax.set_ylim(0, 8e4)
            ax.set_xlim(34e4, 64e4)

            ax.set_yticks([0e3, 40e3, 80e3])
            ax.set_yticklabels(["-40", "0", "40"])
            ax.text(0.01, 0.99, letter + " " + name, ha="left", va="top", transform=ax.transAxes)
            ax.xaxis.set_major_formatter(lambda x, y: "")

        ax_slide_ident.xaxis.set_major_formatter(kmfmt)
        ax_vel_ident.xaxis.set_major_formatter(kmfmt)

        ax_slide_true3.set_ylabel("Distance (km)")

        ax_slide_ident.set_xlabel("Distance (km)")
        ax_vel_ident.set_xlabel("Distance (km)")
        fig.savefig("plots/{:s}mismip_inversion_identtrue.png".format(phys), dpi=300)


def main():
    inversion_results_alln()
    inversion_results_trueident()
    steady_state()


if __name__ == "__main__":
    main()

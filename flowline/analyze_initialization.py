#!/usr/bin/env python
# coding: utf-8

import firedrake
import icepack
import icepack.models.hybrid
import matplotlib.pyplot as plt
from icepackaccs import extract_surface, extract_bed
from icepackaccs.friction import get_weertman, get_regularized_coulomb_simp, friction_stress
from transient_simulations import a_unpert, a_pert, a_retreat
from true_flowline import color_dict


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

for nbumps in [2, 1]:
    checkpoint_fn = "inputs/standard_init_bumps{:d}.h5".format(nbumps)
    with firedrake.CheckpointFile(checkpoint_fn, "r") as chk:
        mesh = chk.load_mesh("flowline")

    parcheckpoint_fn = "inputs/mixed_init_bumps{:d}.h5".format(nbumps)

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

    with firedrake.CheckpointFile(parcheckpoint_fn, "r") as chk:
        for dev in devs:
            for n in ns:
                output_dict[dev][n] = {}

                inv_name = "dev{:0.2f}_n{:2.1f}".format(dev, n)

                if chk.has_attr("already_run", inv_name) and chk.get_attr("already_run", inv_name):
                    done.append(inv_name)

                    output_dict[dev][n]["C1"] = chk.load_function(mesh, inv_name + "_C1")
                    output_dict[dev][n]["C3"] = chk.load_function(mesh, inv_name + "_C3")
                    output_dict[dev][n]["CRCFi"] = chk.load_function(mesh, inv_name + "_CRCFi")
                    output_dict[dev][n]["u1"] = chk.load_function(mesh, inv_name + "_u1")
                    output_dict[dev][n]["u3"] = chk.load_function(mesh, inv_name + "_u3")
                    output_dict[dev][n]["uRCFi"] = chk.load_function(mesh, inv_name + "_uRCFi")

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(7.0, 5.0), gridspec_kw={"height_ratios": (2, 1, 1, 1)})
    axes[0].set_xlim(0, 500e3)
    axes[-1].xaxis.set_major_formatter(kmfmt)
    axes[-1].set_xlabel("Distance (km)")
    axes[1].set_ylabel("Speed (m yr$^{-1}$)")
    axes[1].set_ylim(0, 125)

    firedrake.plot(
        icepack.depth_average(firedrake.Function(Q).interpolate(b + h0_20 * (h0_20 > 10))),
        axes=axes[0],
        edgecolor=color_dict["identical"][3.0][-20],
        label=r"T=-20 $^\circ$C",
    )
    firedrake.plot(
        icepack.depth_average(firedrake.Function(Q).interpolate(b + h0_10 * (h0_10 > 10))),
        axes=axes[0],
        edgecolor=color_dict["identical"][3.0][-10],
        label=r"T=-10 $^\circ$C",
    )
    firedrake.plot(
        icepack.depth_average(firedrake.Function(Q).interpolate(b + h0_5 * (h0_5 > 10))),
        axes=axes[0],
        edgecolor=color_dict["identical"][3.0][-5],
        label=r"T=-5 $^\circ$C",
    )
    firedrake.plot(icepack.depth_average(b), axes=axes[0], edgecolor="brown", label="_nolegend_")
    axes[0].legend(loc="lower left")
    axes[0].set_ylabel("Elevation (m)")
    axes[0].set_ylim(0, 6000)

    taub_10 = firedrake.Function(extract_bed(C0).function_space()).interpolate(
        extract_bed(C0) ** 2.0 * abs(extract_bed(u0_10)) ** (1.0 / 3.0) * 1000 * (2 * (extract_bed(h0_10) > 10) - 1)
    )
    taub_20 = firedrake.Function(extract_bed(C0).function_space()).interpolate(
        extract_bed(C0) ** 2.0 * abs(extract_bed(u0_20)) ** (1.0 / 3.0) * 1000 * (2 * (extract_bed(h0_20) > 10) - 1)
    )
    taub_5 = firedrake.Function(extract_bed(C0).function_space()).interpolate(
        extract_bed(C0) ** 2.0 * abs(extract_bed(u0_5)) ** (1.0 / 3.0) * 1000 * (2 * (extract_bed(h0_5) > 10) - 1)
    )
    firedrake.plot(taub_20, axes=axes[2], edgecolor=color_dict["identical"][3.0][-20])
    firedrake.plot(taub_10, axes=axes[2], edgecolor=color_dict["identical"][3.0][-10])
    firedrake.plot(taub_5, axes=axes[2], edgecolor=color_dict["identical"][3.0][-5])
    axes[2].set_ylabel(r"$\tau_b$ (kPa)")
    axes[2].set_ylim(0, 300)

    # firedrake.plot(icepack.depth_average(u0), axes=axes[1], label="Depth averaged")
    # firedrake.plot(firedrake.Function(extract_surface(u0).function_space()).interpolate(extract_surface(u0) - extract_bed(u0)), axes=axes[1], label="Shear")
    firedrake.plot(
        extract_surface(firedrake.Function(V_8).interpolate(u0_10 * (2 * (h0_10 > 10) - 1))),
        axes=axes[1],
        edgecolor=color_dict["identical"][3.0][-10],
        label="_nolegend_",
        linewidth=3,
    )
    firedrake.plot(
        extract_bed(firedrake.Function(V_8).interpolate(u0_10 * (2 * (h0_10 > 10) - 1))),
        axes=axes[1],
        edgecolor=color_dict["identical"][3.0][-10],
        label="_nolegend_",
        lw=1,
    )
    firedrake.plot(
        extract_surface(firedrake.Function(V_8).interpolate(u0_20 * (2 * (h0_20 > 10) - 1))),
        axes=axes[1],
        edgecolor=color_dict["identical"][3.0][-20],
        label="_nolegend_",
        linewidth=3,
    )
    firedrake.plot(
        extract_bed(firedrake.Function(V_8).interpolate(u0_20 * (2 * (h0_20 > 10) - 1))),
        axes=axes[1],
        edgecolor=color_dict["identical"][3.0][-20],
        label="_nolegend_",
    )
    firedrake.plot(
        extract_surface(firedrake.Function(V_8).interpolate(u0_20 * (2 * (h0_5 > 10) - 1))),
        axes=axes[1],
        edgecolor=color_dict["identical"][3.0][-5],
        label="_nolegend_",
        linewidth=3,
    )
    firedrake.plot(
        extract_bed(firedrake.Function(V_8).interpolate(u0_5 * (2 * (h0_5 > 10) - 1))),
        axes=axes[1],
        edgecolor=color_dict["identical"][3.0][-5],
        label="_nolegend_",
    )
    axes[1].plot([], [], color="k", lw=3, label="Surface")
    axes[1].plot([], [], color="k", lw=1, label="Basal")
    axes[1].legend(loc="best")

    axes[3].set_ylabel("SMB (m yr$^{-1}$)")
    firedrake.plot(icepack.depth_average(a), axes=axes[3], edgecolor="k", label="Initial")
    firedrake.plot(icepack.depth_average(a_retreat), axes=axes[3], edgecolor="0.6", label="Retreat")
    # firedrake.plot(icepack.depth_average(a_pert), axes=axes[3], edgecolor='0.6', label="Perturbed")
    axes[3].set_ylim(-1.5, 1.5)
    axes[3].axhline(0, color="k", lw="0.5")
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

    ind = 0
    for T in Ts:
        for n in ns:
            if False:  # T == -10 and n == 3:
                color = "k"
                lw = 2
            else:
                color = color_dict["standard"][n][T]
                ind += 1
                lw = 1
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
            firedrake.plot(
                extract_surface(firedrake.Function(V_8).interpolate(output_dict[T][n]["u3"] * (2 * (h0_10 > 10) - 1))),
                axes=axes[0],
                edgecolor=color,
                lw=lw,
                label=r"{:d}$^\circ$C $n$={:2.1f}".format(T, n),
            )

            firedrake.plot(
                firedrake.Function(extract_bed(C0).function_space()).interpolate(
                    extract_bed(
                        firedrake.Function(V_8).interpolate(output_dict[T][n]["u3"] * 100 * (2 * (h0_10 > 10) - 1))
                    )
                    / extract_surface(firedrake.Function(V_8).interpolate(output_dict[T][n]["u3"]))
                ),
                axes=axes[1],
                edgecolor=color,
                label=[None],
            )

            taub = firedrake.Function(extract_bed(C0).function_space()).interpolate(
                -extract_bed(friction_stress(output_dict[T][n]["u3"], output_dict[T][n]["C1"] ** 2.0, m=1))
                * 1000
                * (2 * (extract_bed(h0_10) > 10) - 1)
            )
            firedrake.plot(taub, axes=axes[2], edgecolor=color)

    for i in range(1):
        firedrake.plot(
            extract_surface(firedrake.Function(V_8).interpolate(u0_10 * (2 * (h0_10 > 10) - 1))),
            axes=axes[0 + i * 3],
            edgecolor="k",
            lw=2,
            label="True\n" + r"({:d}$^\circ$C $n$={:2.1f})".format(-10, 3),
        )

        firedrake.plot(
            firedrake.Function(extract_bed(C0).function_space()).interpolate(
                extract_bed(firedrake.Function(V_8).interpolate(u0_10 * 100 * (2 * (h0_10 > 10) - 1)))
                / extract_surface(firedrake.Function(V_8).interpolate(u0_10))
            ),
            axes=axes[1 + i * 3],
            edgecolor="k",
            label=[None],
        )

        taub = firedrake.Function(extract_bed(C0).function_space()).interpolate(
            -extract_bed(friction_stress(u0_10, C0**2.0, m=3)) * 1000 * (2 * (extract_bed(h0_10) > 10) - 1)
        )
        firedrake.plot(taub, axes=axes[2 + i * 3], edgecolor="k")

    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 0.99), fontsize=9)

    for ax, letter in zip(axes, "abcdefgh"):
        ax.text(0.01, 0.98, letter, fontsize=14, ha="left", va="top", transform=ax.transAxes)

    # fig.tight_layout(pad=0.1)
    fig.savefig("figs/initialized_variableC_standard_nbumps{:1d}.pdf".format(nbumps))

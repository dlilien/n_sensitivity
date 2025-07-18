#!/usr/bin/env python
# coding: utf-8
import argparse
import os
import firedrake
import icepack
import icepack.models.hybrid
import numpy as np
from firedrake.__future__ import interpolate
import tqdm
from icepackaccs import rate_factor, extract_surface
from icepackaccs.friction import get_weertman, get_regularized_coulomb_simp
from true_flowline import a_0, δa, u0_coulomb
import matplotlib.pyplot as plt

check_vel = False

a2 = 0.95
δa2 = 2.375

initial_fn = "inputs/identical_init_bumps2.h5"


with firedrake.CheckpointFile(initial_fn, "r") as chk:
    mesh = chk.load_mesh("flowline")
Q_init = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="R", vdegree=0)

x, ζ = firedrake.SpatialCoordinate(mesh)
Lx = 500e3
nx = 2001

# Without these we get a mysterious error on new firedrake installs
mesh1d_dum = firedrake.IntervalMesh(nx, Lx)
mesh_dum = firedrake.ExtrudedMesh(mesh1d_dum, layers=1)

a_unpert = firedrake.Function(Q_init).interpolate(a_0 - δa * x / Lx)
a_pert = firedrake.Function(Q_init).interpolate(
    a_0 - δa * x / Lx - δa * (((x - 150000) / Lx) ** 2.0 - (150000 / Lx) ** 2.0) / 5
)
a_retreat = firedrake.Function(Q_init).interpolate(a2 - δa2 * x / Lx)


def volume(thickness):
    return firedrake.assemble(thickness * firedrake.dx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=float, nargs="+")
    parser.add_argument("-T", type=int, nargs="+", default=None)
    parser.add_argument("-dev", type=float, nargs="+", default=None)
    parser.add_argument("-frics", type=str, nargs="+", default=["RCFi", "3", "1"], choices=["RCFi", "3", "1"])
    parser.add_argument("-sim", choices=["perturbed", "unperturbed", "retreat"], default="retreat")
    parser.add_argument("-std", action="store_true")
    parser.add_argument("-nbumps", type=int, default=2)
    parser.add_argument("-reload_ind", type=int, default=-1)
    parser.add_argument("-nt", type=int, default=8, help="Timesteps per year (1/dt: default 8)")
    args = parser.parse_args()

    nbumps = args.nbumps

    num_years = 10000
    min_thick = firedrake.Constant(10.0)
    ns = args.n
    Ts = args.T

    if args.std:
        checkpoint_fn = "inputs/standard_init_bumps{:d}.h5".format(nbumps)
        init = "standard"
        if args.T is None:
            Ts = [-12, -10, -8]
    else:
        checkpoint_fn = "inputs/identical_init_bumps{:d}.h5".format(nbumps)
        init = "identical"
        if args.T is None:
            Ts = [-20, -10, -5]
    input_dict = {T: {} for T in Ts}

    with firedrake.CheckpointFile(checkpoint_fn, "r") as chk:
        mesh = chk.load_mesh("flowline")

    Q = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="R", vdegree=0)
    V_8 = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="GL", vdegree=8)

    acc = args.sim

    if acc == "perturbed":
        a = a_pert
    elif acc == "retreat":
        a = a_retreat
    else:
        print("This is not perturbed")
        a = a_unpert
    a = firedrake.Function(Q).interpolate(a)

    if init in ["identical", "inferred"]:
        output_template = "outputs/{:s}_{:s}_T{:d}_n{:2.1f}_{:s}_nbumps{:d}.h5"
    else:
        output_template = "outputs/{:s}_{:s}_dev{:0.2f}_n{:2.1f}_{:s}_nbumps{:d}.h5"

    for T_np in Ts:
        if init == "identical":
            cache_fn = "inputs/flowline_n3_{:03d}C_weertman3_bumps{:1d}.h5".format(T_np, nbumps)
        else:
            cache_fn = "inputs/flowline_n3_{:03d}C_weertman3_bumps{:1d}.h5".format(-10, nbumps)
        with firedrake.CheckpointFile(cache_fn, "r") as chk:
            field_names = ["surf", "bed", "thick", "u2", "u4", "C"]
            mesh_cache = chk.load_mesh("flowline")
            # start_time = chk.get_attr("metadata", "total_time")
            fields = {name: chk.load_function(mesh_cache, name) for name in field_names}
            input_dict[T_np]["h0"] = firedrake.Function(Q).interpolate(fields["thick"])
            input_dict[T_np]["s0"] = firedrake.Function(Q).interpolate(fields["surf"])
            input_dict[T_np]["u0"] = firedrake.Function(V_8).interpolate(fields["u4"])
            b = firedrake.Function(Q).interpolate(fields["bed"])

    regularized_coulomb = get_regularized_coulomb_simp(m=3, u_0=u0_coulomb)
    weertman_3 = get_weertman(m=3)
    weertman_1 = get_weertman(m=1)
    frictions = {"3": weertman_3, "1": weertman_1, "RCFi": regularized_coulomb}

    opts = {"dirichlet_ids": [1], "diagnostic_solver_type": "icepack"}
    opts = {
        "dirichlet_ids": [1],
        "diagnostic_solver_type": "petsc",
        "diagnostic_solver_parameters": {
            "snes_type": "newtonls",
            "snes_max_it": 10000,
            "snes_stol": 1.0e-4,
            "snes_rtol": 1.0e-8,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "snes_linesearch_type": "bt",
            "snes_linesearch_order": 2,
            "pc_factor_mat_solver_type": "mumps",
            "max_iterations": 2,
            # "snes_monitor": None,
            # "snes_linesearch_monitor": None,
        },
    }

    with firedrake.CheckpointFile(checkpoint_fn, "r") as chk:
        for T_np in Ts:
            for n in ns:
                input_dict[T_np][n] = {}
                if init in ["identical", "inferred"]:
                    inv_name = "T{:d}_n{:2.1f}".format(T_np, n)
                else:
                    inv_name = "dev{:0.2f}_n{:2.1f}".format(T_np, n)

                input_dict[T_np][n]["C1"] = chk.load_function(mesh, inv_name + "_C1")
                input_dict[T_np][n]["C3"] = chk.load_function(mesh, inv_name + "_C3")
                input_dict[T_np][n]["CRCFi"] = chk.load_function(mesh, inv_name + "_CRCFi")
                input_dict[T_np][n]["u1"] = chk.load_function(mesh, inv_name + "_u1")
                input_dict[T_np][n]["u3"] = chk.load_function(mesh, inv_name + "_u3")
                input_dict[T_np][n]["uRCFi"] = chk.load_function(mesh, inv_name + "_uRCFi")
                if init == "identical":
                    input_dict[T_np][n]["A"] = chk.load_function(mesh, inv_name + "_A")
                elif init == "parinferred":
                    input_dict[T_np][n]["A"] = firedrake.Constant(chk.get_attr("A", inv_name))

    for fricname in args.frics:
        friction = frictions[fricname]
        for T_np in Ts:
            Q = input_dict[T_np]["h0"].function_space()
            V = input_dict[T_np]["u0"].function_space()
            for n in ns:
                if fricname in ["1", "3"]:
                    fricstring = "m=" + fricname
                else:
                    fricstring = fricname
                output_fn = output_template.format(acc, init, T_np, n, fricname, nbumps)
                already_run = 0
                if os.path.exists(output_fn):
                    with firedrake.CheckpointFile(output_fn, "r") as chk:
                        if (
                            not chk.has_attr("metadata", "final_time")
                            or chk.get_attr("metadata", "final_time") < num_years
                        ):
                            mesh1 = chk.load_mesh("flowline")
                            hist = chk.get_timestepping_history(mesh1, "h")
                            already_run = int(hist["time"][args.reload_ind])
                            index_prev = args.reload_ind
                            if index_prev < 0:
                                index_prev += len(hist["time"])
                            print(
                                "Reloading previous run at {:d} of {:d}, time {:d}".format(
                                    index_prev, len(hist["time"]), already_run
                                )
                            )
                            h = firedrake.Function(Q).interpolate(chk.load_function(mesh1, "h", idx=index_prev))
                            s = firedrake.Function(Q).interpolate(chk.load_function(mesh1, "s", idx=index_prev))
                            u = firedrake.Function(V).interpolate(chk.load_function(mesh1, "u", idx=index_prev))
                        else:
                            continue
                if init in ["identical", "parinferred"]:
                    A = input_dict[T_np][n]["A"]
                else:
                    T = firedrake.Constant(T_np + 273.15)
                    if n > 2:
                        A = rate_factor(T, n=n)
                    else:
                        A = rate_factor(T, n=n, m=1.0e-2, m_exp=1.4)

                model = icepack.models.HybridModel(friction=friction)
                solver = icepack.solvers.FlowSolver(model, **opts)

                if already_run == 0:
                    h = input_dict[T_np]["h0"].copy(deepcopy=True)
                    s = input_dict[T_np]["s0"].copy(deepcopy=True)
                    u = input_dict[T_np]["u0"].copy(deepcopy=True)
                    index_prev = 0

                sqrtC = input_dict[T_np][n]["C" + fricname]
                C = firedrake.Function(sqrtC.function_space()).interpolate(sqrtC**2.0)

                if init in ["identical", "inferred"]:
                    print("T={:d}, n={:2.1f}, {:s}".format(T_np, n, fricstring))
                else:
                    print("dev={:0.2f}, n={:2.1f}, {:s}".format(T_np, n, fricstring))

                u = solver.diagnostic_solve(
                    velocity=u, thickness=h, surface=s, fluidity=A, friction=C, flow_law_exponent=firedrake.Constant(n)
                )
                if check_vel:
                    fig, ax = plt.subplots()
                    firedrake.plot(extract_surface(input_dict[T_np]["u0"]), axes=ax, edgecolor="k")
                    firedrake.plot(extract_surface(input_dict[T_np][n]["u" + fricname]), axes=ax, edgecolor="0.6")
                    firedrake.plot(extract_surface(u), axes=ax, edgecolor="C0")
                    plt.show()
                    raise

                timesteps_per_year = args.nt
                δt = 1.0 / timesteps_per_year
                num_timesteps = (num_years - already_run) * timesteps_per_year

                if already_run == 0:
                    with firedrake.CheckpointFile(output_fn, "w") as chk:
                        chk.save_mesh(mesh)
                        chk.create_group("metadata")
                        chk.set_attr("metadata", "dt", timesteps_per_year * δt)
                        chk.save_function(u, name="u", idx=0, timestepping_info={"time": 0.0})
                        chk.save_function(s, name="s", idx=0, timestepping_info={"time": 0.0})
                        chk.save_function(h, name="h", idx=0, timestepping_info={"time": 0.0})
                        chk.save_function(b, name="b")
                        chk.save_function(a, name="a")
                        chk.save_function(C, name="C_sqrd")

                progress_bar = tqdm.trange(num_timesteps)
                description = f"dV,max(abs(dH)): {(0 - 0) / δt:4.2f} [m2/yr] {(0) / δt:4.3f} [m/yr]"
                progress_bar.set_description(description)
                for j, step in enumerate(progress_bar):
                    h_prev = h.copy(deepcopy=True)
                    h = solver.prognostic_solve(
                        δt,
                        thickness=h,
                        velocity=u,
                        accumulation=a,
                        thickness_inflow=input_dict[T_np]["h0"],
                    )
                    h = firedrake.assemble(interpolate(firedrake.max_value(h, min_thick), Q))

                    s = icepack.compute_surface(thickness=h, bed=b)
                    u = solver.diagnostic_solve(
                        velocity=u,
                        thickness=h,
                        surface=s,
                        fluidity=A,
                        friction=C,
                        flow_law_exponent=firedrake.Constant(n),
                    )
                    dh = firedrake.assemble(interpolate(h - h_prev, h.function_space()))
                    H_v_t = volume(h)
                    H_v_prev = volume(h_prev)
                    description = f"dV,max(abs(dH)): {(H_v_t - H_v_prev) / δt:4.2f} [m2/yr] {(np.abs(dh.dat.data_ro).max()) / δt:4.3f} [m/yr]"
                    progress_bar.set_description(description)

                    save_interval = 10
                    if (j + 1) % int(timesteps_per_year * save_interval) == 0:
                        with firedrake.CheckpointFile(output_fn, "a") as chk:
                            idx = (j + 1) // (timesteps_per_year * save_interval) + index_prev
                            time = (j + 1) / timesteps_per_year + already_run
                            chk.save_function(u, name="u", idx=idx, timestepping_info={"time": time})
                            chk.save_function(s, name="s", idx=idx, timestepping_info={"time": time})
                            chk.save_function(h, name="h", idx=idx, timestepping_info={"time": time})
                            chk.set_attr("metadata", "final_time", time)


if __name__ == "__main__":
    main()

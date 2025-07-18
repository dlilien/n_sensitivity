#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 David Lilien <dlilien@iu.edu>
#
# Distributed under terms of the GNU GPL3.0 license.

"""

"""
from firedrake.petsc import PETSc
import socket
import tqdm
import datetime
import numpy as np
import firedrake
import icepack
from firedrake.__future__ import interpolate
from firedrake.pyplot import tripcolor
from firedrake import dx, dS_v, dS
from icepack.constants import ice_density as ρ_I, water_density as ρ_W
import matplotlib.pyplot as plt
from icepackaccs import extract_surface
from icepackaccs.mismip import Lx, Ly


ts_name = "partial_stream"


fast_opts = {
    "dirichlet_ids": [4],
    "side_wall_ids": [1, 3],
    "diagnostic_solver_type": "petsc",
    "diagnostic_solver_parameters": {
        "snes_type": "newtonls",
        "snes_max_it": 1000,
        "snes_stol": 1.0e-6,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "snes_linesearch_type": "bt",
        "snes_linesearch_order": 2,
        "pc_factor_mat_solver_type": "mumps",
        "max_iterations": 2500,
    },
}

par_opts = {
    "dirichlet_ids": [4],
    "side_wall_ids": [1, 3],
    "diagnostic_solver_type": "petsc",
    "diagnostic_solver_parameters": {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_linesearch_order": 2,
        "snes_max_it": 1000,
        "snes_stol": 1.0e-6,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "max_iterations": 2500,
    },
    "prognositc_solver_parameters": {"ksp_type": "gmres", "pc_type": "asm", "sub_pc_type": "ilu"},
}


reliable_opts = {
    "dirichlet_ids": [4],
    "side_wall_ids": [1, 3],
    "diagnostic_solver_type": "icepack",
    "diagnostic_solver_parameters": {
        "max_iterations": 5000,
    },
}

other_opts = {
    "dirichlet_ids": [4],
    "side_wall_ids": [1, 3],
    "diagnostic_solver_type": "petsc",
    "diagnostic_solver_parameters": {
        # "snes_type": "newtonls",
        "snes_line_search_type": "cp",
        "snes_linesearch_max_it": 2500,
        "snes_max_it": 5000,
        "snes_stol": 1.0e-8,
        "snes_rtol": 1.0e-6,
        "ksp_type": "cg",
        "pc_type": "mg",
        # "pc_factor_mat_solver_type": "mumps",
        "pc_factor_shift_amount": 1.0e-10,
        "max_iterations": 5000,
        "snes_linesearch_damping": 0.1,
        "snes_monitor": None,
        "snes_linesearch_monitor": None,
    },
}
damping_and_iters = [(0.5, 50), (0.1, 1000), (0.05, 2000)]
# damping_and_iters = [(0.5, 50)]  # , (0.1, 1000)]


def equivalent_thinning(func, surf, thick, mesh, depth, x0, y_c, **kwargs):
    vol_chan = vol_thinning(func, mesh, depth, x0, y_c, **kwargs)
    floating = gradual_floating_area(mesh, surf, thick)
    vol_float = firedrake.assemble(floating * dx)
    scale = vol_chan / vol_float
    return floating * scale


def vol_thinning(func, mesh, depth, x0, y_c, **kwargs):
    tc = func(mesh, depth, x0, y_c, **kwargs)
    return firedrake.assemble(tc * dx)


def gradual_floating_area(mesh, s, h0):
    """Find the floating area, easing in gradually

    Parameters
    ----------
    mesh: firedrake.Mesh
        Needed since we construct function space in addition to that on which s and h0 live
    s: firedrake.Function
        The ice surface elevation.
    h0: firedrake.Function
        The ice thickness.

    Returns
    -------
    floating_mask: firedrake.Function
        A gradual mask of the floating area living in the same function space as s.
    """
    height_above_flotation = firedrake.assemble(interpolate(s - (1 - ρ_I / ρ_W) * h0, s.function_space()))
    hob1 = firedrake.assemble(interpolate(height_above_flotation > 0.01, s.function_space()))

    if mesh.geometric_dimension() == 3:
        DG0 = firedrake.FunctionSpace(mesh, "DG", 0, vdegree=0)
        use_dS = dS_v
    else:
        DG0 = firedrake.FunctionSpace(mesh, "DG", 0)
        use_dS = dS

    ϵ = firedrake.Function(DG0)
    J = 0.5 * ((ϵ - hob1) ** 2 * dx + (5e3) * (ϵ("+") - ϵ("-")) ** 2 * use_dS)
    F = firedrake.derivative(J, ϵ)
    firedrake.solve(F == 0, ϵ)
    definitely_ungrounded = firedrake.assemble(interpolate(ϵ < 0.02, DG0))
    ϵ2 = firedrake.Function(DG0)
    J2 = 0.5 * ((ϵ2 - definitely_ungrounded) ** 2 * dx + (5e4) * (ϵ2("+") - ϵ2("-")) ** 2 * use_dS)
    F2 = firedrake.derivative(J2, ϵ2)
    firedrake.solve(F2 == 0, ϵ2)
    return firedrake.assemble(interpolate(firedrake.max_value(0.0, (ϵ2 - 0.5) * 2.0), s.function_space()))


def smooth_floating(res, s, h0, cutoff=0.05):
    """Find the floating area

    Parameters
    ----------
    mesh: firedrake.Mesh
        Needed since we construct function space in addition to that on which s and h0 live
    s: firedrake.Function
        The ice surface elevation.
    h0: firedrake.Function
        The ice thickness.

    Returns
    -------
    floating_mask: firedrake.Function
        A gradual mask of the floating area living in the same function space as s.
    """
    mesh2 = firedrake.RectangleMesh(int(Lx / res), int(Ly / res), Lx, Ly)
    Q2 = firedrake.FunctionSpace(mesh2, "CG", degree=1)
    height_above_flotation = firedrake.assemble(interpolate(s - (1 - ρ_I / ρ_W) * h0, Q2))
    hob1 = firedrake.assemble(interpolate((firedrake.min_value(height_above_flotation, 10) - 5.0) / 5.0, Q2))
    DG0 = firedrake.FunctionSpace(mesh2, "DG", 0)
    ϵ = firedrake.Function(DG0)
    J = 0.5 * ((ϵ - hob1) ** 2 * dx + (1e3) * (ϵ("+") - ϵ("-")) ** 2 * dS)
    F = firedrake.derivative(J, ϵ)
    firedrake.solve(F == 0, ϵ)
    return firedrake.project(ϵ, Q2)


def volume(thickness):
    return firedrake.assemble(thickness * firedrake.dx) / 1.0e9


def vab(thickness, surface):
    bottom = surface - thickness
    eq_thick = firedrake.conditional(bottom < 0.0, -bottom * ρ_W / ρ_I, 0.0)
    return firedrake.assemble((thickness - eq_thick) * firedrake.dx) / 1.0e9


def run_simulation(
    solver,
    time,
    dt,
    return_all=False,
    recomp_u=False,
    plot=False,
    cutoff_dV=None,
    cutoff_dh=None,
    min_step=0,
    mod_A=None,
    is_floating0=None,
    **fields,
):
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 6))
        axes[1][1].axis("off")
        cbrs = None
    h, s, u, z_b, a, A, C = map(fields.get, ("thickness", "surface", "velocity", "bed", "a", "A", "C"))
    h_0 = h.copy(deepcopy=True)
    num_steps = int(time / dt)
    H_v_t = np.zeros((num_steps,))
    if return_all:
        thicks = []
        vels = []
        surfs = []
        dH = []

    if recomp_u:
        print("Recomputing velocity because of divergence errors")
        Q = h.function_space()
        mesh = Q.mesh()
        x, _ = firedrake.SpatialCoordinate(mesh)
        # u_init = firedrake.assemble(interpolate(firedrake.as_vector((x / Lx, 0)), u.function_space()))
        u_init = firedrake.assemble(0.5 * u)
        if mod_A is None:
            u = solver.diagnostic_solve(
                velocity=u_init,
                thickness=h,
                surface=s,
                fluidity=A,
                friction=C,
            )
        else:
            u = solver.diagnostic_solve(
                velocity=u_init, thickness=h, surface=s, fluidity=A, friction=C, mod_A=mod_A, is_floating=is_floating0
            )

    try:
        if "noatak" in socket.gethostname():
            progress_bar = tqdm.trange(num_steps)
        else:
            progress_bar = np.arange(num_steps)
            start = datetime.datetime.now()
        for step in progress_bar:
            h_prev = h.copy(deepcopy=True)
            h = solver.prognostic_solve(
                dt,
                thickness=h,
                velocity=u,
                accumulation=a,
                thickness_inflow=h_0,
            )
            firedrake.assemble(h.interpolate(firedrake.max_value(h, 10.0)))
            s = icepack.compute_surface(thickness=h, bed=z_b)

            if mod_A is None:
                u = solver.diagnostic_solve(
                    velocity=u,
                    thickness=h,
                    surface=s,
                    fluidity=A,
                    friction=C,
                )
            else:
                u = solver.diagnostic_solve(
                    velocity=u, thickness=h, surface=s, fluidity=A, friction=C, mod_A=mod_A, is_floating=is_floating0
                )

            dh = firedrake.assemble(interpolate(h - h_prev, h.function_space()))
            H_v_t[step] = volume(h)
            if return_all:
                thicks.append(h.copy(deepcopy=True))
                vels.append(u.copy(deepcopy=True))
                surfs.append(s.copy(deepcopy=True))
                dH.append(dh.copy(deepcopy=True))
            max_h = firedrake.Vector(dh).max()
            min_h = -firedrake.Vector(
                firedrake.Function(dh.function_space()).interpolate(-dh)
            ).max()  # funky because there is no .min()
            if -min_h > max_h:
                max_h = min_h
            if cutoff_dV is not None:
                if cutoff_dh is not None:
                    if (
                        abs((H_v_t[step] - H_v_t[step - 1]) / dt) < cutoff_dV
                        and step > min_step
                        and abs(max_h) < cutoff_dh
                    ):
                        PETSc.Sys.Print("Thresholds met. Return at step {:d}".format(step))
                        break
                elif abs((H_v_t[step] - H_v_t[step - 1]) / dt) < cutoff_dV and step > min_step:
                    PETSc.Sys.Print("Volume change threshold met. Return at step {:d}".format(step))
                    break
            description = f"dV: {(H_v_t[step] - H_v_t[step - 1]) / dt:4.2f} [km3/yr], max(dH): {max_h / dt:4.3f} [m/yr]"
            if "noatak" in socket.gethostname():
                progress_bar.set_description(description)
            else:
                if step % 10 == 0:
                    elapsed = (datetime.datetime.now() - start).total_seconds()
                    tot = (datetime.datetime.now() - start).total_seconds() / (step + 1) * num_steps
                    PETSc.Sys.Print(
                        "At step {:d}, ".format(step)
                        + description
                        + " time is {:d}:{:02d}:{:02d} of {:d}:{:02d}:{:02d}".format(
                            int(elapsed // 3600),
                            int((elapsed % 3600) / 60),
                            int((elapsed % 3600) % 60),
                            int(tot // 3600),
                            int((tot % 3600) / 60),
                            int((tot % 3600) % 60),
                        )
                    )
            if plot:
                axes[0][0].clear()
                axes[0][1].clear()
                axes[1][0].clear()
                colorsv = tripcolor(
                    extract_surface(firedrake.assemble(interpolate(u[0], h.function_space()))),
                    axes=axes[0][0],
                    cmap="Reds",
                    vmin=0,
                    vmax=500,
                )
                colorss = tripcolor(extract_surface(s), axes=axes[0][1], vmin=0, vmax=500)
                colorsdh = tripcolor(
                    extract_surface(firedrake.assemble(interpolate(dh / dt, dh.function_space()))),
                    axes=axes[1][0],
                    vmin=-0.25,
                    vmax=0.25,
                    cmap="bwr",
                )
                if cbrs is None:
                    cbrs = [
                        plt.colorbar(
                            colorsv,
                            ax=axes[1][1],
                            label=r"u$_x$ (m yr$^{-1}$",
                            extend="max",
                            location="left",
                            pad=0.1,
                        ),
                        plt.colorbar(
                            colorss,
                            ax=axes[1][1],
                            label=r"s (m)",
                            extend="max",
                            pad=0.4,
                        ),
                        plt.colorbar(
                            colorsdh,
                            ax=axes[1][1],
                            label=r"$\Delta$H (m yr$^{-1}$",
                            extend="both",
                            pad=0.0,
                        ),
                    ]
                fig.savefig("progress.png", dpi=300)
        if return_all:
            return None, {
                "thickness": thicks,
                "surface": surfs,
                "velocity": vels,
                "dH": dH,
            }
        else:
            return None, {"thickness": h, "surface": s, "velocity": u, "dH": dh}
    except firedrake.exceptions.ConvergenceError as e:
        if return_all:
            return e, {
                "thickness": thicks,
                "surface": surfs,
                "velocity": vels,
                "dH": dH,
            }
        else:
            return e, {"thickness": h, "surface": s, "velocity": u, "dH": dh}


def try_alot(model, u_prev, h_0, s_0, A, C):
    converged = False
    try:
        solver = icepack.solvers.FlowSolver(model, **fast_opts)
        u_0 = solver.diagnostic_solve(velocity=u_prev, thickness=h_0, surface=s_0, fluidity=A, friction=C)
        converged = True
    except firedrake.exceptions.ConvergenceError:
        for damping, iters in damping_and_iters:
            try:
                other_opts["diagnostic_solver_parameters"]["snes_max_it"] = iters
                other_opts["diagnostic_solver_parameters"]["snes_linesearch_damping"] = damping
                print("Retrying with {:1.2f} damping".format(damping))
                solver = icepack.solvers.FlowSolver(model, **other_opts)
                u_0 = solver.diagnostic_solve(
                    velocity=u_prev,
                    thickness=h_0,
                    surface=s_0,
                    fluidity=A,
                    friction=C,
                )
                converged = True
                break
            except firedrake.exceptions.ConvergenceError:
                continue
    if not converged:
        print("Retrying fancily")
        opts = other_opts.copy()
        opts["diagnostic_solver_parameters"]["snes_max_it"] = 2000
        opts["diagnostic_solver_parameters"]["snes_linesearch_damping"] = 0.05
        opts["diagnostic_solver_parameters"]["snes_stol"] = 1.0e-6
        opts["diagnostic_solver_parameters"]["snes_rtol"] = 1.5e-4
        solver = icepack.solvers.FlowSolver(model, **opts)
        u_0 = solver.diagnostic_solve(
            velocity=u_prev,
            thickness=h_0,
            surface=s_0,
            fluidity=A,
            friction=C,
        )
        opts["diagnostic_solver_parameters"]["snes_stol"] = 1.0e-8
        opts["diagnostic_solver_parameters"]["snes_rtol"] = 1.0e-6
        if False:
            opts["diagnostic_solver_parameters"]["snes_stol"] = 1.0e-8
            opts["diagnostic_solver_parameters"]["snes_rtol"] = 1.0e-1
            opts["diagnostic_solver_parameters"]["snes_max_it"] = 10000
            opts["diagnostic_solver_parameters"]["snes_linesearch_damping"] = 0.001
            solver = icepack.solvers.FlowSolver(model, **opts)
            u_0 = solver.diagnostic_solve(
                velocity=u_0,
                thickness=h_0,
                surface=s_0,
                fluidity=A,
                friction=C,
            )
    return {"surface": s_0, "thickness": h_0, "velocity": u_0}


def mismip_melt(surface, thickness, bed, HC0=75, z0=-100, omega=0.2):
    """Return the melt according to Eq. 17 of Asay-Davis, using default values of parameters"""
    HC = (surface - thickness) - bed
    return omega * firedrake.tanh(HC / HC0) * firedrake.max_value(z0 - (surface - thickness), firedrake.Constant(0.0))


def get_gl(h, s, z_b, tol=0.1, n_pts=640e4 + 1):
    d = len(firedrake.SpatialCoordinate(h.ufl_domain()))
    if d == 3:  # hybrid
        x, y = firedrake.SpatialCoordinate(extract_surface(h).ufl_domain())
        not_floating = firedrake.conditional(extract_surface(s - h) > extract_surface(z_b) + tol, 0.0, 1.0)
    else:
        x, y = firedrake.SpatialCoordinate(h.ufl_domain())
        not_floating = firedrake.conditional(s - h > z_b + tol, 0.0, 1.0)
    return (
        firedrake.assemble(not_floating * firedrake.dx) * 2 / 1e9,
        firedrake.assemble(not_floating * firedrake.conditional(y > 35e3, 1.0, 0.0) * firedrake.dx) / 5e3,
    )


def run_save_simulation(solver, time, dt, fn, save_interval=10, **fields):
    h, s, u, z_b, get_a, A, C, flow_law_exponent, mesh = map(
        fields.get, ("thickness", "surface", "velocity", "bed", "get_a", "A", "C", "flow_law_exponent", "mesh")
    )
    h_0 = h.copy(deepcopy=True)
    num_steps = int(time / dt)
    H_v_t = np.zeros((num_steps + 1,))
    v_t = np.zeros((num_steps + 1,))
    vab_t = np.zeros((num_steps + 1,))
    gl_t = np.zeros((num_steps + 1,))
    grounded_area = np.zeros((num_steps + 1,))

    with firedrake.CheckpointFile(fn, "w") as chk:
        chk.save_mesh(mesh)
        chk.create_group("metadata")
        chk.set_attr("metadata", "dt", dt)
        chk.set_attr("metadata", "times", dt * np.arange(num_steps + 1))
        chk.save_function(u, name="u", idx=0, timestepping_info={"time": 0.0})
        chk.save_function(s, name="s", idx=0, timestepping_info={"time": 0.0})
        chk.save_function(h, name="h", idx=0, timestepping_info={"time": 0.0})
        chk.save_function(C, name="C_sqrd")

    start = datetime.datetime.now()

    dh = firedrake.assemble(interpolate(firedrake.Constant(0.0), h.function_space()))
    H_v_t[0] = volume(h)
    v_t[0] = firedrake.Vector(
        firedrake.Function(h.function_space()).interpolate(firedrake.sqrt(firedrake.dot(u, u)))
    ).max()
    vab_t[0] = vab(h, s)
    grounded_area[0], gl_t[0] = get_gl(h, s, z_b)

    for step in range(num_steps):
        h_prev = h.copy(deepcopy=True)
        a = get_a(s, h, z_b)
        h = solver.prognostic_solve(
            dt,
            thickness=h,
            velocity=u,
            accumulation=a,
            thickness_inflow=h_0,
        )
        firedrake.assemble(h.interpolate(firedrake.max_value(h, 10.0)))
        s = icepack.compute_surface(thickness=h, bed=z_b)

        u = solver.diagnostic_solve(
            velocity=u,
            thickness=h,
            surface=s,
            fluidity=A,
            friction=C,
            flow_law_exponent=flow_law_exponent,
        )

        dh = firedrake.assemble(interpolate(h - h_prev, h.function_space()))
        H_v_t[step + 1] = volume(h)
        vab_t[step + 1] = vab(h, s)
        v_t[step + 1] = firedrake.Vector(
            firedrake.Function(h.function_space()).interpolate(firedrake.sqrt(firedrake.dot(u, u)))
        ).max()
        grounded_area[step + 1], gl_t[step + 1] = get_gl(h, s, z_b)

        if (step + 1) % int(1 / dt * save_interval) == 0:
            with firedrake.CheckpointFile(fn, "a") as chk:
                idx = (step + 1) // (1 / dt * save_interval)
                chk.save_function(u, name="u", idx=idx, timestepping_info={"time": (step + 1) * dt})
                chk.save_function(s, name="s", idx=idx, timestepping_info={"time": (step + 1) * dt})
                chk.save_function(h, name="h", idx=idx, timestepping_info={"time": (step + 1) * dt})
                chk.set_attr("metadata", "final_time", (step + 1) * dt)
                chk.set_attr("metadata", "vol", H_v_t)
                chk.set_attr("metadata", "vab", vab_t)
                chk.set_attr("metadata", "max_vel", v_t)
                chk.set_attr("metadata", "gl", gl_t)
                chk.set_attr("metadata", "grounded_area", grounded_area)

        if step > 0:
            description = f"dV: {(H_v_t[step] - H_v_t[step - 1]) / dt:4.2f} [km3/yr], max(dH): {(dh.dat.data_ro[np.abs(dh.dat.data_ro).argmax()]) / dt:4.3f} [m/yr], GL: {gl_t[step] / 1000:4.1f} [km]"
        else:
            description = f"V: {H_v_t[step]:4.2f} [km3], GL: {gl_t[step] / 1000:4.1f} [km]"
        if step % 10 == 0:
            elapsed = (datetime.datetime.now() - start).total_seconds()
            tot = (datetime.datetime.now() - start).total_seconds() / (step + 1) * num_steps
            PETSc.Sys.Print(
                "At step {:d}, ".format(step)
                + description
                + " time is {:d}:{:02d}:{:02d} of {:d}:{:02d}:{:02d}".format(
                    int(elapsed // 3600),
                    int((elapsed % 3600) / 60),
                    int((elapsed % 3600) % 60),
                    int(tot // 3600),
                    int((tot % 3600) / 60),
                    int((tot % 3600) % 60),
                )
            )
    return None, {"thickness": h, "surface": s, "velocity": u, "dH": dh}


def toreal(array, component):
    if array.dtype.kind == "c":
        assert component in {"real", "imag"}
        return getattr(array, component)
    else:
        assert component == "real"
        return array


def _plot_2d_field(method_name, function, *args, complex_component="real", rev=False, **kwargs):
    axes = kwargs.pop("axes", None)
    if axes is None:
        figure = plt.figure()
        axes = figure.add_subplot(111)

    Q = function.function_space()
    mesh = Q.mesh()
    if rev:
        mesh.coordinates.dat.data[:, 1] = 80e3 - mesh.coordinates.dat.data[:, 1]

    if len(function.ufl_shape) == 1:
        element = function.ufl_element().sub_elements[0]
        Q = firedrake.FunctionSpace(mesh, element)
        function = firedrake.assemble(firedrake.Interpolate(firedrake.sqrt(firedrake.inner(function, function)), Q))

    num_sample_points = kwargs.pop("num_sample_points", 10)
    function_plotter = firedrake.FunctionPlotter(mesh, num_sample_points)
    triangulation = function_plotter.triangulation
    values = function_plotter(function)

    if rev:
        mesh.coordinates.dat.data[:, 1] = 80e3 - mesh.coordinates.dat.data[:, 1]
    method = getattr(axes, method_name)
    return method(triangulation, toreal(values, complex_component), *args, **kwargs)


@PETSc.Log.EventDecorator()
def mirrored_tripcolor(function, *args, complex_component="real", **kwargs):
    r"""Create a pseudo-color plot of a 2D Firedrake :class:`~.Function`

    If the input function is a vector field, the magnitude will be plotted.

    :arg function: the function to plot
    :arg args: same as for matplotlib :func:`tripcolor <matplotlib.pyplot.tripcolor>`
    :kwarg complex_component: If plotting complex data, which
        component? (``'real'`` or ``'imag'``). Default is ``'real'``.
    :arg kwargs: same as for matplotlib
    :return: matplotlib :class:`PolyCollection <matplotlib.collections.PolyCollection>` object
    """
    element = function.ufl_element()
    dg0 = (element.family() == "Discontinuous Lagrange") and (element.degree() == 0)
    kwargs["shading"] = kwargs.get("shading", "flat" if dg0 else "gouraud")
    _plot_2d_field("tripcolor", function, *args, complex_component=complex_component, rev=True, **kwargs)
    return _plot_2d_field("tripcolor", function, *args, complex_component=complex_component, rev=False, **kwargs)


@PETSc.Log.EventDecorator()
def mirrored_tricontour(function, *args, complex_component="real", **kwargs):
    r"""Create a contour plot of a 2D Firedrake :class:`~.Function`

    If the input function is a vector field, the magnitude will be plotted.

    :arg function: the Firedrake :class:`~.Function` to plot
    :arg args: same as for matplotlib :func:`tricontour <matplotlib.pyplot.tricontour>`
    :kwarg complex_component: If plotting complex data, which
        component? (``'real'`` or ``'imag'``). Default is ``'real'``.
    :arg kwargs: same as for matplotlib
    :return: matplotlib :class:`ContourSet <matplotlib.contour.ContourSet>` object
    """
    _plot_2d_field("tricontour", function, *args, complex_component=complex_component, rev=True, **kwargs)
    return _plot_2d_field("tricontour", function, *args, complex_component=complex_component, rev=False, **kwargs)

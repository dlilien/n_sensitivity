#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 dlilien <dlilien@noatak>
#
# Distributed under terms of the MIT license.

"""

"""
import os
from operator import itemgetter
import firedrake
from firedrake.petsc import PETSc
import argparse
from icepackaccs import rate_factor, extract_bed
import icepack
from libmismip import par_opts
from icepackaccs.friction import get_ramp_weertman, get_regularized_coulomb
from initialize_mismip import h_t, u0_coulomb
from icepack.models import hybrid

regularized_coulomb = get_regularized_coulomb(m=3, u_0=u0_coulomb, h_t=50.0)
weertman_3 = get_ramp_weertman(m=3, h_t=h_t)
weertman_1 = get_ramp_weertman(m=1, h_t=h_t)

Area = 640e3 * 40e3


def A3_to_An(A3, u, h, s, n, Q):
    ε_e = effective_strain_rate(velocity=u, thickness=h, surface=h)
    An = firedrake.Function(Q).interpolate(A3 ** (n / 3.0) * ε_e ** (1.0 - n / 3.0))
    An_mean = firedrake.assemble(An * firedrake.dx) / Area
    return firedrake.Function(Q).interpolate(firedrake.conditional(h > 10.1, An, An_mean))


def effective_strain_rate(**kwargs):
    r"""Return the effective strain rate

    Keyword arguments
    -----------------
    velocity : firedrake.Function
    surface : firedrake.Function
    thickness : firedrake.Function

    Returns
    -------
    firedrake.Form
    """
    u, h, s = itemgetter("velocity", "thickness", "surface")(kwargs)
    ε_min = kwargs.get("strain_rate_min", firedrake.Constant(icepack.constants.strain_rate_min))

    ε_x = hybrid.horizontal_strain_rate(velocity=u, surface=s, thickness=h)
    ε_z = hybrid.vertical_strain_rate(velocity=u, thickness=h)
    ε_e = hybrid._effective_strain_rate(ε_x, ε_z, ε_min)
    return ε_e


def c3_to_c1(C3, u, minslide=0.0):
    Q2D = firedrake.FunctionSpace(u.ufl_domain()._base_mesh, "CG", 2)
    C1 = firedrake.Function(C3.function_space())
    U = firedrake.max_value(firedrake.sqrt(firedrake.dot(u, u)), minslide)
    C1_bed = firedrake.Function(Q2D).interpolate(
        firedrake.sqrt(extract_bed(C3) ** 2.0 * abs(extract_bed(U)) ** (1.0 / 3.0 - 1.0))
    )
    C1.dat.data[:] = C1_bed.dat.data[:]
    return C1


def c1_to_c3(C1, u):
    Q2D = firedrake.FunctionSpace(u.ufl_domain()._base_mesh, "CG", 2)
    C3 = firedrake.Function(C1.function_space())
    U = firedrake.sqrt(firedrake.dot(u, u))
    C3_bed = firedrake.Function(Q2D).interpolate(
        firedrake.sqrt(extract_bed(C1) ** 2.0 / abs(extract_bed(U)) ** (1.0 / 3.0 - 1.0))
    )
    C3.dat.data[:] = C3_bed.dat.data[:]
    return C3


def c3_to_beta(C3, u, u0):
    Q2D = firedrake.FunctionSpace(u.ufl_domain()._base_mesh, "CG", 2)
    beta = firedrake.Function(firedrake.FunctionSpace(u.ufl_domain(), "CG", 2, vfamily="R", vdegree=0))
    U = firedrake.sqrt(firedrake.dot(u, u))
    beta_bed = firedrake.Function(Q2D).interpolate(
        firedrake.sqrt(
            extract_bed(C3) ** 2.0 * (extract_bed(U) ** (1.0 / 3.0 + 1) + u0 ** (1.0 / 3.0 + 1)) ** (1.0 / (3.0 + 1.0))
        )
    )
    beta.dat.data[:] = beta_bed.dat.data[:]
    return beta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=float, nargs="+", default=[1.8, 3, 3.5, 4])
    parser.add_argument("-T", type=int, nargs="+", default=[-12, -10, -8])
    parser.add_argument("-dev", type=float, nargs="+", default=[0.75, 1.0, 1.25])
    args = parser.parse_args()

    input_fn = "inputs/standard_initialization_mismip_simul.h5"
    with firedrake.CheckpointFile(input_fn, "r") as chk:
        fine_mesh = chk.load_mesh("fine_mesh")
        h0 = chk.load_function(fine_mesh, name="input_thick")
        s0 = chk.load_function(fine_mesh, name="input_surf")
        u0 = chk.load_function(fine_mesh, name="input_u")
        C3 = chk.load_function(fine_mesh, name="input_C")
    A3 = rate_factor(firedrake.Constant(263.15), n=3.0)
    Q = firedrake.FunctionSpace(fine_mesh, "CG", 2, vfamily="R", vdegree=0)
    QA = firedrake.FunctionSpace(fine_mesh, "CG", 2, vfamily="GL", vdegree=4)

    print("Doing 'true' first")
    An = A3_to_An(A3, u0, h0, s0, 3.0, QA)
    name_fmt = "T{:d}_n{:2.1f}"
    inv_name = name_fmt.format(-10, 3)
    output_fn = "inputs/true_initialization_mismip.h5"
    with firedrake.CheckpointFile(output_fn, "w") as chk:
        chk.save_mesh(fine_mesh)
        chk.create_group("already_run")
        chk.save_function(h0, name="input_thick")
        chk.save_function(s0, name="input_surf")
        chk.save_function(u0, name="input_u")
        chk.save_function(C3, name="input_C")

        chk.set_attr("already_run", inv_name, True)
        chk.save_function(C3, name=inv_name + "_C3")
        chk.save_function(C3, name=inv_name + "_CRCFi")
        chk.save_function(C3, name=inv_name + "_C1")
        chk.save_function(u0, name=inv_name + "_u1")
        chk.save_function(An, name=inv_name + "_A")

    print("Doing analytical conversions next")
    output_fn = "inputs/identical_initialization_mismip.h5"
    if not os.path.exists(output_fn):
        with firedrake.CheckpointFile(output_fn, "w") as chk:
            chk.save_mesh(fine_mesh)
            chk.create_group("already_run")
            chk.save_function(h0, name="input_thick")
            chk.save_function(s0, name="input_surf")
            chk.save_function(u0, name="input_u")
            chk.save_function(C3, name="input_C")

    C1 = c3_to_c1(C3, u0)
    Beta = c3_to_beta(C3, u0, u0_coulomb)
    for n in [1.8, 3, 3.5, 4]:
        An = A3_to_An(A3, u0, h0, s0, n, QA)
        name_fmt = "T{:d}_n{:2.1f}"
        inv_name = name_fmt.format(-10, n)
        with firedrake.CheckpointFile(output_fn, "a") as chk:
            if not chk.has_attr("already_run", inv_name):
                chk.set_attr("already_run", inv_name, True)
                chk.save_function(C1, name=inv_name + "_C1")
                chk.save_function(C3, name=inv_name + "_C3")
                chk.save_function(Beta, name=inv_name + "_CRCFi")
                chk.save_function(u0, name=inv_name + "_u1")
                chk.save_function(An, name=inv_name + "_A")

    print("Aggregating inversion results")
    for input_fn, output_fn, T_not_dev in [
        ("inputs/standard_initialization_mismip_simul.h5", "inputs/standard_initialization_mismip.h5", True)
    ]:
        with firedrake.CheckpointFile(input_fn, "r") as chk:
            fine_mesh = chk.load_mesh("fine_mesh")
            h0 = chk.load_function(fine_mesh, name="input_thick")
            s0 = chk.load_function(fine_mesh, name="input_surf")
            u0 = chk.load_function(fine_mesh, name="input_u")
            C0 = chk.load_function(fine_mesh, name="input_C")

        Q = h0.function_space()

        if not os.path.exists(output_fn):
            with firedrake.CheckpointFile(output_fn, "w") as chk:
                chk.save_mesh(fine_mesh)
                chk.create_group("already_run")
                chk.save_function(h0, name="input_thick")
                chk.save_function(s0, name="input_surf")
                chk.save_function(u0, name="input_u")
                chk.save_function(C0, name="input_C")

        if T_not_dev:
            outeriter = args.T
            name_fmt = "T{:d}_n{:2.1f}"
        else:
            outeriter = args.dev
            name_fmt = "dev{:1.2f}_n{:2.1f}"

        for T_or_dev in outeriter:
            for n in args.n:
                if T_not_dev:
                    inv_name = name_fmt.format(T_or_dev, n)
                else:
                    inv_name = name_fmt.format(T_or_dev, n)
                with firedrake.CheckpointFile(output_fn, "r") as chkout:
                    if chkout.has_attr("already_run", inv_name):
                        continue

                with firedrake.CheckpointFile(input_fn, "r") as chk:
                    i = 0
                    if not chk.has_attr("partial", inv_name + "_{:d}".format(i)):
                        continue
                    while chk.has_attr("partial", inv_name + "_{:d}".format(i + 1)):
                        i += 1

                    PETSc.Sys.Print("Loading iteration {:d} for ".format(i + 1) + name_fmt.format(T_or_dev, n))
                    A = chk.load_function(fine_mesh, inv_name + "_A_{:d}".format(i))
                    mod_A = chk.load_function(fine_mesh, inv_name + "_modA_{:d}".format(i))
                    C1 = chk.load_function(fine_mesh, inv_name + "_C1_{:d}".format(i))
                    u = chk.load_function(fine_mesh, inv_name + "_u_{:d}".format(i))

                PETSc.Sys.Print("Converting C for " + name_fmt.format(T_or_dev, n))
                C3 = c1_to_c3(C1, u)
                Beta = c3_to_beta(C3, u, u0_coulomb)

                if False:
                    model = icepack.models.HybridModel(friction=regularized_coulomb)
                    solver = icepack.solvers.FlowSolver(model, **par_opts)
                    u_rcf = solver.diagnostic_solve(
                        velocity=u0,
                        thickness=h0,
                        surface=s0,
                        fluidity=A,
                        friction=firedrake.Function(Q).interpolate(Beta**2.0),
                        flow_law_exponent=firedrake.Constant(n),
                    )

                    model = icepack.models.HybridModel(friction=weertman_3)
                    solver = icepack.solvers.FlowSolver(model, **par_opts)
                    u_w3 = solver.diagnostic_solve(
                        velocity=u0,
                        thickness=h0,
                        surface=s0,
                        fluidity=A,
                        friction=firedrake.Function(Q).interpolate(C3**2.0),
                        flow_law_exponent=firedrake.Constant(n),
                    )

                with firedrake.CheckpointFile(output_fn, "a") as chk:
                    chk.set_attr("already_run", inv_name, True)
                    chk.save_function(C1, name=inv_name + "_C1")
                    chk.save_function(C3, name=inv_name + "_C3")
                    chk.save_function(Beta, name=inv_name + "_CRCFi")
                    chk.save_function(u, name=inv_name + "_u1")
                    chk.save_function(A, name=inv_name + "_A")
                    chk.save_function(mod_A, name=inv_name + "_modA")
                    if False:
                        chk.save_function(u_w3, name=inv_name + "_u3")
                        chk.save_function(u_rcf, name=inv_name + "_uRCFi")


if __name__ == "__main__":
    main()

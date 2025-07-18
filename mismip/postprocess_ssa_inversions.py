#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 dlilien <dlilien@noatak>
#
# Distributed under terms of the MIT license.

"""

"""
import os
import firedrake
from firedrake.petsc import PETSc
import argparse
from icepackaccs.friction import get_ramp_weertman, get_regularized_coulomb
from initialize_mismip import h_t, u0_coulomb, LC_dict, LA_dict

regularized_coulomb = get_regularized_coulomb(m=3, u_0=u0_coulomb, h_t=50.0)
weertman_3 = get_ramp_weertman(m=3, h_t=h_t)
weertman_1 = get_ramp_weertman(m=1, h_t=h_t)


def c1_to_c3(C1, u, Q):
    U = firedrake.sqrt(firedrake.dot(u, u))
    C3 = firedrake.Function(C1.function_space()).interpolate(firedrake.sqrt(C1**2.0 / abs(U) ** (1.0 / 3.0 - 1.0)))
    return C3


def c3_to_beta(C3, u, u0, Q):
    U = firedrake.sqrt(firedrake.dot(u, u))
    beta = firedrake.Function(firedrake.FunctionSpace(C3.ufl_domain(), "CG", 2)).interpolate(
        firedrake.sqrt(C3**2.0 * (U ** (1.0 / 3.0 + 1) + u0 ** (1.0 / 3.0 + 1)) ** (1.0 / (3.0 + 1.0)))
    )
    return beta


def c3_to_c1(C3, u, Q, minslide=0.0):
    C1 = firedrake.Function(C3.function_space())
    U = firedrake.max_value(firedrake.sqrt(firedrake.dot(u, u)), minslide)
    C1_bed = firedrake.Function(Q).interpolate(firedrake.sqrt(C3**2.0 * abs(U) ** (1.0 / 3.0 - 1.0)))
    C1.dat.data[:] = C1_bed.dat.data[:]
    return C1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=float, nargs="+", default=[1.8, 3, 3.5, 4])
    parser.add_argument("-T", type=int, nargs="+", default=[-12, -10, -8])
    parser.add_argument("-dev", type=float, nargs="+", default=[0.75, 1.0, 1.25])
    args = parser.parse_args()

    for input_fn, output_fn, T_not_dev in [
        ("inputs/ssa_standard_initialization_mismip_simul.h5", "inputs/ssa_standard_initialization_mismip.h5", True)
    ]:
        with firedrake.CheckpointFile(input_fn, "r") as chk:
            fine_mesh = chk.load_mesh("fine_mesh2d")
            h0 = chk.load_function(fine_mesh, name="input_thick")
            s0 = chk.load_function(fine_mesh, name="input_surf")
            u0 = chk.load_function(fine_mesh, name="input_u")
            u0_ssa = chk.load_function(fine_mesh, name="input_u_ssa")
            C0 = chk.load_function(fine_mesh, name="input_C")

        Q = h0.function_space()

        if not os.path.exists(output_fn):
            with firedrake.CheckpointFile(output_fn, "w") as chk:
                chk.save_mesh(fine_mesh)
                chk.create_group("already_run")
                chk.save_function(h0, name="input_thick")
                chk.save_function(s0, name="input_surf")
                chk.save_function(u0, name="input_u")
                chk.save_function(u0_ssa, name="input_u_ssa")
                chk.save_function(C0, name="input_C")

        if T_not_dev:
            outeriter = args.T
            name_fmt = "T{:d}_n{:2.1f}_LC{:1.0e}_LA{:1.0e}"
            out_fmt = "T{:d}_n{:2.1f}"
        else:
            outeriter = args.dev
            name_fmt = "dev{:1.2f}_n{:2.1f}_LC{:1.0e}_LA{:1.0e}"
            out_fmt = "dev{:1.2f}_n{:2.1f}"

        for T_or_dev in outeriter:
            for n in args.n:
                inv_name_in = name_fmt.format(T_or_dev, n, LC_dict[T_not_dev][n], LA_dict[T_not_dev][n])
                inv_name_out = out_fmt.format(T_or_dev, n)

                with firedrake.CheckpointFile(output_fn, "r") as chkout:
                    if chkout.has_attr("already_run", inv_name_out):
                        continue

                with firedrake.CheckpointFile(input_fn, "r") as chk:
                    i = 0
                    if not chk.has_attr("partial", inv_name_in + "_{:d}".format(i)):
                        continue
                    while chk.has_attr("partial", inv_name_in + "_{:d}".format(i + 1)):
                        i += 1

                    PETSc.Sys.Print("Loading iteration {:d} for ".format(i + 1) + inv_name_in)
                    A = chk.load_function(fine_mesh, inv_name_in + "_A_{:d}".format(i))
                    mod_A = chk.load_function(fine_mesh, inv_name_in + "_modA_{:d}".format(i))
                    C1 = chk.load_function(fine_mesh, inv_name_in + "_C1_{:d}".format(i))
                    u = chk.load_function(fine_mesh, inv_name_in + "_u_{:d}".format(i))

                PETSc.Sys.Print("Converting C for " + inv_name_out)
                C3 = c1_to_c3(C1, u, Q)
                Beta = c3_to_beta(C3, u, u0_coulomb, Q)

                with firedrake.CheckpointFile(output_fn, "a") as chk:
                    chk.set_attr("already_run", inv_name_out, True)
                    chk.save_function(C1, name=inv_name_out + "_C1")
                    chk.save_function(C3, name=inv_name_out + "_C3")
                    chk.save_function(Beta, name=inv_name_out + "_CRCFi")
                    chk.save_function(u, name=inv_name_out + "_u1")
                    chk.save_function(A, name=inv_name_out + "_A")
                    chk.save_function(mod_A, name=inv_name_out + "_modA")


if __name__ == "__main__":
    main()

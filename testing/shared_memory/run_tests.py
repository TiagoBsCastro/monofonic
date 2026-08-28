#!/usr/bin/env python3
"""Regression tests for the per-node shared-memory FastDF window.

Usage:
  run_tests.py /path/to/branch/monofonIC /path/to/master/monofonIC \
      [--mpirun mpirun] [--class-ini-out /path/to/input_class_parameters.ini]
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import h5py
import numpy as np

M3 = 8 ** 3  # NeutrinoCubeRootNum^3
REL_TOL = 1e-10
MASTER_REL_TOL = 1e-12


def make_config(path, grid_res, nu_res, invert, threads, filename):
    with open(path, "w") as f:
        f.write(f"""[setup]
GridRes         = {grid_res}
BoxLength       = 300
zstart          = 24.0
LPTorder        = 1
DoBaryons       = no
DoBaryonVrel    = no
DoFixing        = no
DoInversion     = {('yes' if invert else 'no')}
ParticleLoad    = sc
DoDensityVelocityCorr = no

WithNeutrinos   = yes
DoNeutrinoParticles = yes
NeutrinoCubeRootNum = 8
NeutrinoGridRes = {nu_res}
NeutrinoStepSize = 0.05
NeutrinoInterpOrder = 1

[cosmology]
ParameterSet    = Planck2018EE+BAO+SN
transfer        = zwindstroom
ztarget         = 0.0

[random]
generator       = NGENIC
seed            = 12345

[execution]
NumThreads      = {threads}

[output]
format          = gadget_hdf5
filename        = {filename}
""")


def load_realization(pattern):
    """Concatenate all rank-local PartType2 arrays and sort by ParticleID."""
    files = sorted(glob.glob(pattern))
    files = [f for f in files if ".fastdf-tmp." not in f]
    assert files, f"no output files match {pattern}"

    ids_parts, coord_parts, vel_parts, psd_parts = [], [], [], []
    for filename in files:
        with h5py.File(filename, "r") as h:
            group = h["PartType2"]
            ids_parts.append(np.asarray(group["ParticleIDs"][:]))
            coord_parts.append(np.asarray(group["Coordinates"][:]))
            vel_parts.append(np.asarray(group["Velocities"][:]))
            psd_parts.append(np.asarray(group["PhaseSpaceDensities"][:]))

    ids = np.concatenate(ids_parts)
    coords = np.concatenate(coord_parts)
    velocities = np.concatenate(vel_parts)
    psd = np.concatenate(psd_parts)
    order = np.argsort(ids)
    return ids[order], coords[order], velocities[order], psd[order]


def relative_error(a, b):
    abs_diff = np.max(np.abs(a - b))
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    return abs_diff / scale if scale > 0.0 else abs_diff


def float_report(name, a, b, ok):
    if np.array_equal(a, b):
        return f"{name}: bit-identical"
    abs_diff = np.max(np.abs(a - b))
    rel_diff = relative_error(a, b)
    status = "OK" if ok else "FAIL"
    return f"{name}: max|d|={abs_diff:.3e} max rel={rel_diff:.3e} [{status}]"


def compare(real_a, real_b, rel_tol):
    ids_a, coords_a, velocities_a, psd_a = real_a
    ids_b, coords_b, velocities_b, psd_b = real_b

    assert ids_a.shape == ids_b.shape, "particle count mismatch"
    assert len(ids_a) == M3, f"expected {M3} particles, got {len(ids_a)}"
    assert np.array_equal(ids_a, ids_b), "ParticleIDs differ"

    ok_coords = np.array_equal(coords_a, coords_b) or relative_error(coords_a, coords_b) < rel_tol
    ok_velocities = np.array_equal(velocities_a, velocities_b) or relative_error(velocities_a, velocities_b) < rel_tol
    ok_psd = np.array_equal(psd_a, psd_b) or relative_error(psd_a, psd_b) < rel_tol

    report = "\n".join([
        f"  particle count: {len(ids_a)} (OK)",
        "  IDs: bit-identical",
        f"  {float_report('Coordinates', coords_a, coords_b, ok_coords)}",
        f"  {float_report('Velocities', velocities_a, velocities_b, ok_velocities)}",
        f"  {float_report('PhaseSpaceDensities', psd_a, psd_b, ok_psd)}",
    ])
    assert ok_coords and ok_velocities and ok_psd, "floating-point mismatch exceeds tolerance"
    return report


def run_case(binary, mpirun, workdir, grid_res, nu_res, invert, threads, np_rank, filename):
    os.makedirs(workdir, exist_ok=True)
    config = os.path.join(workdir, "run.conf")
    make_config(config, grid_res, nu_res, invert, threads, filename)

    env = dict(os.environ)
    if threads > 1:
        env["OMP_NUM_THREADS"] = str(threads)

    command = [binary, "run.conf"] if np_rank == 1 else [mpirun, "-np", str(np_rank), binary, "run.conf"]
    result = subprocess.run(command, cwd=workdir, env=env)
    return result.returncode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("branch_binary")
    parser.add_argument("master_binary")
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--class-ini-out")
    return parser.parse_args()


def main():
    args = parse_args()
    branch_binary = os.path.abspath(args.branch_binary)
    master_binary = os.path.abspath(args.master_binary)
    work = tempfile.mkdtemp(prefix="fastdf_shared_")

    layouts = [
        ("serial", 1, 1),
        ("mpi4", 4, 1),
        ("mpi9", 9, 1),
        ("hybrid", 2, 2),
    ]
    cases = [
        ("lt", 16, 8, False),
        ("eq", 8, 8, False),
        ("lt_inv", 16, 8, True),
    ]

    failures = 0
    branch_serial_baseline = None
    class_ini_copied = False

    try:
        for case, grid_res, nu_res, invert in cases:
            realizations = {}
            for name, np_rank, threads in layouts:
                if name == "hybrid" and case != "lt":
                    continue

                subdir = os.path.join(work, f"{case}_{name}")
                print(f"=== {case}/{name} (np={np_rank}, threads={threads}) ===", flush=True)
                rc = run_case(
                    branch_binary, args.mpirun, subdir, grid_res, nu_res,
                    invert, threads, np_rank, f"ics_{name}.hdf5"
                )
                if rc != 0:
                    print(f"  FAILED (exit {rc})")
                    failures += 1
                    continue

                try:
                    realizations[name] = load_realization(os.path.join(subdir, f"ics_{name}*.hdf5"))
                except Exception as exc:
                    print(f"  FAILED while reading output: {exc}")
                    failures += 1
                    continue
                print("  OK")

                if case == "lt" and name == "serial":
                    branch_serial_baseline = realizations[name]
                    if args.class_ini_out:
                        generated_ini = os.path.join(subdir, "input_class_parameters.ini")
                        if not os.path.isfile(generated_ini):
                            print(f"  FAILED: expected generated CLASS input {generated_ini}")
                            failures += 1
                        else:
                            output_dir = os.path.dirname(os.path.abspath(args.class_ini_out))
                            os.makedirs(output_dir, exist_ok=True)
                            shutil.copy2(generated_ini, args.class_ini_out)
                            class_ini_copied = True

            if "serial" not in realizations:
                continue
            baseline = realizations["serial"]
            for name in ("mpi4", "mpi9", "hybrid"):
                if name not in realizations:
                    continue
                print(f"  serial vs {name}:")
                try:
                    print(compare(baseline, realizations[name], REL_TOL))
                except AssertionError as exc:
                    print(f"  FAIL: {exc}")
                    failures += 1

        print("\n=== master vs branch serial baseline ===", flush=True)
        master_dir = os.path.join(work, "master_serial")
        rc = run_case(
            master_binary, args.mpirun, master_dir,
            16, 8, False, 1, 1, "ics_master.hdf5"
        )
        if rc != 0:
            print(f"  FAILED master run (exit {rc})")
            failures += 1
        elif branch_serial_baseline is None:
            print("  FAILED: branch serial baseline unavailable")
            failures += 1
        else:
            try:
                master_realization = load_realization(os.path.join(master_dir, "ics_master*.hdf5"))
                print(compare(master_realization, branch_serial_baseline, MASTER_REL_TOL))
            except (AssertionError, Exception) as exc:
                print(f"  FAIL: {exc}")
                failures += 1

        if args.class_ini_out and not class_ini_copied:
            print("FAILED: CLASS input was not produced for the end-to-end failure test")
            failures += 1

        print(f"\n{'ALL LAYOUT AND MASTER COMPARISONS PASSED' if failures == 0 else 'FAILURES DETECTED'}")
        return 0 if failures == 0 else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

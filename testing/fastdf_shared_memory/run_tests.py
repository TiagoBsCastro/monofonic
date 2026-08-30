#!/usr/bin/env python3
# This file is part of monofonIC (MUSIC2), a software package for generating
# initial conditions for cosmological simulations.
# Copyright (C) 2026 monofonIC contributors
#
# monofonIC is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# monofonIC is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along
# with monofonIC. If not, see <http://www.gnu.org/licenses/>.

"""Regression tests for the per-node shared-memory FastDF window.

Usage:
  run_tests.py /path/to/branch/monofonIC /path/to/master/monofonIC \
      [--mpirun mpirun] [--suite smoke|full] [--require-nodes N]
      [--class-ini-out /path/to/input_class_parameters.ini] [--work-root DIR]
"""

import argparse
import glob
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

import h5py
import numpy as np

M3 = 8 ** 3  # NeutrinoCubeRootNum^3
REL_TOL = 1e-10
MASTER_REL_TOL = 1e-12
FFT_REPORT_RE = re.compile(
    r"FastDF node FFT .*?requested=(\d+), "
    r"affinity_available=(\d+), using=(\d+) thread"
)
CLASS_REPORT_RE = re.compile(
    r"FastDF node CLASS .*?requested=(\d+), "
    r"affinity_available=(\d+), using=(\d+) thread"
)
ZWINDSTROOM_CLASS_REPORT_RE = re.compile(
    r"Zwindstroom node CLASS .*?requested=(\d+), "
    r"affinity_available=(\d+), using=(\d+) thread"
)


def make_config(path, grid_res, nu_res, invert, threads, filename,
                invalid_class=False):
    # A coarse integration cadence keeps this regression focused on shared-grid
    # ownership and layout equivalence rather than production convergence.
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
NeutrinoStepSize = 0.5
NeutrinoInterpOrder = 1

[cosmology]
ParameterSet    = {('none' if invalid_class else 'Planck2018EE+BAO+SN')}
{('YHe            = 2.0' if invalid_class else '')}
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


def load_white_noise(workdir):
    """Load the real-space grid handed from monofonIC to FastDF."""
    filename = os.path.join(workdir, "white_noise.hdf5")
    with h5py.File(filename, "r") as h:
        return np.asarray(h["white_noise"][:])


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
    if not (ok_coords and ok_velocities and ok_psd):
        raise AssertionError(f"floating-point mismatch exceeds tolerance\n{report}")
    return report


def log_tail(path, line_count=40):
    with open(path, "r", errors="replace") as stream:
        lines = stream.readlines()
    return "".join(lines[-line_count:])


def validate_thread_reports(path, report_re, label, expected_total,
                            expected_nodes):
    """Check and summarize one-per-node leader diagnostics."""
    with open(path, "r", errors="replace") as stream:
        reports = [tuple(map(int, match))
                   for match in report_re.findall(stream.read())]
    if len(reports) != expected_nodes:
        raise AssertionError(
            f"expected exactly one {label} report per detected node: "
            f"reports={reports}, expected nodes={expected_nodes}"
        )
    if sum(requested for requested, _, _ in reports) != expected_total:
        raise AssertionError(
            f"node {label} requests do not cover all filling threads: "
            f"reports={reports}, expected total={expected_total}"
        )
    for requested, available, used in reports:
        expected_used = max(1, min(requested, available))
        if used != expected_used:
            raise AssertionError(
                f"unexpected {label} leader thread selection: "
                f"requested={requested}, available={available}, used={used}"
            )
    return (f"{label} leaders: nodes={len(reports)}, "
            f"requested={[item[0] for item in reports]}, "
            f"available={[item[1] for item in reports]}, "
            f"using={[item[2] for item in reports]}")


def validate_class_control_file(path):
    """Reject truncated, corrupted, or multiply-written CLASS controls."""
    with open(path, "rb") as stream:
        contents = stream.read()
    if not contents or not contents.endswith(b"\n") or b"\0" in contents:
        raise AssertionError(f"incomplete or corrupted CLASS control file: {path}")

    lines = contents.decode("utf-8").splitlines()
    if any(" = " not in line for line in lines):
        raise AssertionError(f"malformed CLASS control line in {path}")
    keys = [line.split(" = ", 1)[0].strip() for line in lines]
    required = {
        "z_max_pk", "P_k_max_h/Mpc", "output", "gauge", "h",
        "Omega_b", "Omega_cdm", "N_ur", "N_ncdm", "n_s", "T_cmb",
        "YHe", "z_pk", "nbody_gauge_transfer_functions",
    }
    missing = sorted(required.difference(keys))
    if missing:
        raise AssertionError(f"CLASS control file is missing keys: {missing}")
    if keys.count("nbody_gauge_transfer_functions") != 1:
        raise AssertionError(
            "CLASS control must contain nbody_gauge_transfer_functions exactly once"
        )
    if len(keys) != len(set(keys)):
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        raise AssertionError(f"duplicate CLASS control keys: {duplicates}")


def run_case(binary, mpirun, workdir, grid_res, nu_res, invert, threads,
             np_rank, filename, verbose_runs, invalid_class=False,
             timeout=None):
    os.makedirs(workdir, exist_ok=True)
    config = os.path.join(workdir, "run.conf")
    make_config(config, grid_res, nu_res, invert, threads, filename,
                invalid_class=invalid_class)

    env = dict(os.environ)
    # Set this explicitly even for pure-MPI cases.  Otherwise a caller's
    # ambient OpenMP setting changes grid-filling parallelism and the node
    # leader's requested CLASS/FFTW thread-pool sizes.
    env["OMP_NUM_THREADS"] = str(threads)

    command = ([binary, "run.conf"] if np_rank == 1 else
               mpirun + ["-np", str(np_rank), binary, "run.conf"])
    log_path = os.path.join(workdir, "run.log")
    with open(log_path, "w") as log:
        try:
            result = subprocess.run(command, cwd=workdir, env=env,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, log_path
    if verbose_runs:
        with open(log_path, "r", errors="replace") as log:
            print(log.read(), end="")
    elif result.returncode != 0:
        print(f"  last lines from {log_path}:\n{log_tail(log_path)}")
    return result.returncode, log_path


def probe_node_count(mpirun, ranks):
    """Return the number of distinct hosts used by an MPI launcher."""
    command = mpirun + [
        "-np", str(ranks), sys.executable, "-c",
        "import socket; print(socket.gethostname())",
    ]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            f"MPI host probe failed (exit {result.returncode}):\n{result.stderr}")
    hosts = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    return len(hosts), sorted(hosts)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("branch_binary")
    parser.add_argument("master_binary")
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--require-nodes", type=int, default=1)
    parser.add_argument("--verbose-runs", action="store_true")
    parser.add_argument("--class-ini-out")
    parser.add_argument("--work-root")
    return parser.parse_args()


def main():
    args = parse_args()
    branch_binary = os.path.abspath(args.branch_binary)
    master_binary = os.path.abspath(args.master_binary)
    mpirun = shlex.split(args.mpirun)
    if not mpirun:
        raise ValueError("--mpirun must not be empty")
    if args.require_nodes < 1:
        raise ValueError("--require-nodes must be positive")
    work_root = None
    if args.work_root:
        work_root = os.path.abspath(args.work_root)
        if not os.path.isdir(work_root) or not os.access(work_root, os.W_OK | os.X_OK):
            raise ValueError(f"--work-root must be a writable directory: {work_root}")
    work = tempfile.mkdtemp(prefix="fastdf_shared_", dir=work_root)

    if args.suite == "smoke":
        layouts = [("serial", 1, 1), ("mpi4", 4, 1)]
        cases = [
            ("lt", 16, 8, False),
            ("eq", 8, 8, False),
        ]
        case_layouts = {
            "lt": {"serial", "mpi4"},
            "eq": {"serial"},
        }
    else:
        layouts = [
            ("serial", 1, 1),
            ("mpi4", 4, 1),
            ("mpi9", 9, 1),
            ("hybrid", 2, 2),
        ]
        cases = [
            ("lt", 16, 8, False),
            ("eq", 8, 8, False),
            ("odd", 9, 8, False),
            ("lt_inv", 16, 8, True),
        ]
        case_layouts = {
            "lt": {"serial", "mpi4", "hybrid"},
            "eq": {"serial", "mpi9"},
            "odd": {"serial", "mpi4"},
            "lt_inv": {"serial", "mpi4"},
        }

    failures = 0
    master_comparison_baseline = None
    master_comparison_white_noise = None
    class_ini_copied = False
    layout_node_counts = {1: 1}

    try:
        if args.require_nodes > 1:
            probe_ranks = max(layout[1] for layout in layouts)
            if args.require_nodes > probe_ranks:
                raise ValueError("--require-nodes cannot exceed the largest MPI layout")
            node_count, hosts = probe_node_count(mpirun, probe_ranks)
            print(f"MPI host probe: {node_count} node(s): {', '.join(hosts)}")
            if node_count < args.require_nodes:
                print(f"FAILED: requested at least {args.require_nodes} physical nodes")
                return 1

        for case, grid_res, nu_res, invert in cases:
            realizations = {}
            white_noise_grids = {}
            for name, np_rank, threads in layouts:
                if name not in case_layouts[case]:
                    continue

                subdir = os.path.join(work, f"{case}_{name}")
                print(f"=== {case}/{name} (np={np_rank}, threads={threads}) ===", flush=True)
                if np_rank not in layout_node_counts:
                    layout_node_counts[np_rank], _ = probe_node_count(mpirun, np_rank)
                expected_nodes = layout_node_counts[np_rank]
                rc, log_path = run_case(
                    branch_binary, mpirun, subdir, grid_res, nu_res,
                    invert, threads, np_rank, f"ics_{name}.hdf5",
                    args.verbose_runs,
                )
                if rc != 0:
                    print(f"  FAILED (exit {rc})")
                    failures += 1
                    continue

                try:
                    print("  " + validate_thread_reports(
                        log_path, FFT_REPORT_RE, "FastDF FFT",
                        np_rank * threads, expected_nodes))
                    print("  " + validate_thread_reports(
                        log_path, CLASS_REPORT_RE, "FastDF CLASS",
                        np_rank * threads, expected_nodes))
                    print("  " + validate_thread_reports(
                        log_path, ZWINDSTROOM_CLASS_REPORT_RE,
                        "Zwindstroom CLASS", np_rank * threads,
                        expected_nodes))
                    validate_class_control_file(os.path.join(
                        subdir, "input_class_parameters.ini"))
                except AssertionError as exc:
                    print(f"  FAILED leader thread report: {exc}")
                    failures += 1
                    continue

                try:
                    realization = load_realization(os.path.join(subdir, f"ics_{name}*.hdf5"))
                    white_noise_grid = load_white_noise(subdir)
                except Exception as exc:
                    print(f"  FAILED while reading output: {exc}")
                    failures += 1
                    continue
                realizations[name] = realization
                white_noise_grids[name] = white_noise_grid
                print("  OK")

                if case == "eq" and name == "serial":
                    # Equal parent/neutrino resolutions bypass monofonIC's
                    # downsampling reduction. This keeps the master comparison
                    # focused on pristine versus shared-memory FastDF; master
                    # otherwise receives a truncated padded FFT grid.
                    master_comparison_baseline = realizations[name]
                    master_comparison_white_noise = white_noise_grids[name]
                if case == "lt" and name == "serial":
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
            baseline_white_noise = white_noise_grids["serial"]
            for name in ("mpi4", "mpi9", "hybrid"):
                if name not in realizations:
                    continue
                white_noise_ok = (np.array_equal(baseline_white_noise, white_noise_grids[name]) or
                                  relative_error(baseline_white_noise, white_noise_grids[name]) < REL_TOL)
                print(f"  serial vs {name} white noise:")
                print("  " + float_report("WhiteNoise", baseline_white_noise,
                                          white_noise_grids[name], white_noise_ok))
                if not white_noise_ok:
                    failures += 1

                print(f"  serial vs {name}:")
                try:
                    print(compare(baseline, realizations[name], REL_TOL))
                except AssertionError as exc:
                    print(f"  FAIL: {exc}")
                    failures += 1

        print("\n=== coherent zwindstroom CLASS failure ===", flush=True)
        invalid_np = max(layout[1] for layout in layouts)
        if invalid_np not in layout_node_counts:
            layout_node_counts[invalid_np], _ = probe_node_count(mpirun, invalid_np)
        invalid_dir = os.path.join(work, "invalid_class_mpi")
        rc, invalid_log = run_case(
            branch_binary, mpirun, invalid_dir,
            8, 8, False, 1, invalid_np, "ics_invalid.hdf5",
            args.verbose_runs, invalid_class=True, timeout=60,
        )
        if rc is None:
            print("  FAILED: invalid CLASS input hung for more than 60 seconds")
            failures += 1
        elif rc == 0:
            print("  FAILED: invalid CLASS input unexpectedly succeeded")
            failures += 1
        else:
            try:
                print("  " + validate_thread_reports(
                    invalid_log, ZWINDSTROOM_CLASS_REPORT_RE,
                    "Zwindstroom CLASS", invalid_np,
                    layout_node_counts[invalid_np]))
                with open(invalid_log, "r", errors="replace") as stream:
                    failure_output = stream.read()
                coherent_reports = failure_output.count("Zwindstroom CLASS failed:")
                if coherent_reports < invalid_np:
                    raise AssertionError(
                        "not every MPI rank reported the synchronized CLASS failure: "
                        f"reports={coherent_reports}, ranks={invalid_np}"
                    )
                print(f"  synchronized failure reached all {invalid_np} ranks")
            except AssertionError as exc:
                print(f"  FAILED coherent CLASS failure check: {exc}")
                failures += 1

        print("\n=== master vs branch equal-grid serial baseline ===", flush=True)
        master_dir = os.path.join(work, "master_serial")
        rc, _ = run_case(
            master_binary, mpirun, master_dir,
            8, 8, False, 1, 1, "ics_master.hdf5", args.verbose_runs,
        )
        if rc != 0:
            print(f"  FAILED master run (exit {rc})")
            failures += 1
        elif (master_comparison_baseline is None or
              master_comparison_white_noise is None):
            print("  FAILED: equal-resolution branch baseline unavailable")
            failures += 1
        else:
            try:
                master_white_noise = load_white_noise(master_dir)
                white_noise_ok = (
                    np.array_equal(master_white_noise, master_comparison_white_noise) or
                    relative_error(master_white_noise,
                                   master_comparison_white_noise) < MASTER_REL_TOL
                )
                print(float_report("  WhiteNoise", master_white_noise,
                                   master_comparison_white_noise, white_noise_ok))
                if not white_noise_ok:
                    raise AssertionError("master and branch FastDF inputs differ")
                master_realization = load_realization(
                    os.path.join(master_dir, "ics_master*.hdf5"))
                print(compare(master_realization, master_comparison_baseline, MASTER_REL_TOL))
            except Exception as exc:
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

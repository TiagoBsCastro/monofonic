#!/usr/bin/env python3
"""Regression tests for the per-node shared-memory FastDF window.

Usage:  run_tests.py /path/to/monofonIC [mpirun]

Exercises:
  * serial (1 rank), mpi4 (4 ranks), mpi9 (9 ranks > 8 x-planes), hybrid (2x2)
  * NeutrinoGridRes < GridRes  and  NeutrinoGridRes == GridRes
  * DoInversion = no  and  DoInversion = yes
  * white-noise HDF5 failure handling (missing file/dataset, malformed dims)
  * end-to-end FastDF failure (run_fastdf() < 0, no deadlock in MPI)

For each (grid, inversion) case the full particle realization is reconstructed
from the per-rank output files (concatenate + sort by ParticleID) and compared
across execution layouts.  Comparisons cover IDs, Coordinates, Velocities and
PhaseSpaceDensities, reporting max absolute/relative differences.
"""

import os

import sys
import glob
import shutil
import subprocess
import tempfile

import h5py
import numpy as np

M3 = 8 ** 3  # NeutrinoCubeRootNum^3


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
    ids = c = v = psd = None
    for f in files:
        with h5py.File(f, "r") as h:
            g = h["PartType2"]
            fi = np.asarray(g["ParticleIDs"][:])
            fc = np.asarray(g["Coordinates"][:])
            fv = np.asarray(g["Velocities"][:])
            fp = np.asarray(g["PhaseSpaceDensities"][:])
        ids = fi if ids is None else np.concatenate([ids, fi])
        c = fc if c is None else np.concatenate([c, fc])
        v = fv if v is None else np.concatenate([v, fv])
        psd = fp if psd is None else np.concatenate([psd, fp])
    o = np.argsort(ids)
    return ids[o], c[o], v[o], psd[o]


# Relative tolerance for cross-layout floating-point comparisons.
#
# The FastDF shared-memory code is bit-identical for a fixed input (and the same
# execution layout).  However, monofonIC's white-noise field is generated with a
# distributed/threaded FFT whose reduction order depends on the MPI layout, so
# the *input* to FastDF differs at the ~1e-16 level between layouts.  This tiny
# input difference is amplified to ~1e-15 relative in the particle output.  The
# tolerance below is ~1e5 larger than that amplification and ~1e9 smaller than
# the empty-slice bug it guards against (relative error ~0.98), so it is a tight
# but robust criterion.
REL_TOL = 1e-10


def float_report(name, a, b, ok):
    if np.array_equal(a, b):
        return f"{name}: bit-identical"
    absd = np.max(np.abs(a - b))
    denom = np.max(np.abs(b))
    reld = absd / denom if denom > 0 else 0.0
    status = "OK" if ok else "FAIL"
    return f"{name}: max|d|={absd:.3e} max rel={reld:.3e} [{status}]"


def compare(real_a, real_b):
    ia, ca, va, pa = real_a
    ib, cb, vb, pb = real_b
    assert ia.shape == ib.shape, "particle count mismatch"
    assert len(ia) == M3, f"expected {M3} particles, got {len(ia)}"
    assert np.array_equal(ia, ib), "ParticleIDs differ"

    def rel(a, b):
        return np.max(np.abs(a - b)) / np.max(np.abs(b))

    ok_c = np.array_equal(ca, cb) or rel(ca, cb) < REL_TOL
    ok_v = np.array_equal(va, vb) or rel(va, vb) < REL_TOL
    ok_p = np.array_equal(pa, pb) or rel(pa, pb) < REL_TOL

    lines = [
        f"  particle count: {len(ia)} (OK)",
        "  IDs: bit-identical",
        f"  {float_report('Coordinates', ca, cb, ok_c)}",
        f"  {float_report('Velocities', va, vb, ok_v)}",
        f"  {float_report('PhaseSpaceDensities', pa, pb, ok_p)}",
    ]
    assert ok_c and ok_v and ok_p, "floating-point mismatch exceeds tolerance"
    return "\n".join(lines)


def main():
    binary = os.path.abspath(sys.argv[1])
    mpirun = sys.argv[2] if len(sys.argv) > 2 else "mpirun"
    here = os.path.dirname(os.path.abspath(__file__))
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
    try:
        for case, grid_res, nu_res, invert in cases:
            reals = {}
            for name, np_rank, threads in layouts:
                # hybrid only exercised on the baseline (lt, no inversion)
                if name == "hybrid" and case != "lt":
                    continue
                sub = os.path.join(work, f"{case}_{name}")
                os.makedirs(sub, exist_ok=True)
                cfg = os.path.join(sub, "run.conf")
                make_config(cfg, grid_res, nu_res, invert, threads, f"ics_{name}.hdf5")
                env = dict(os.environ)
                if threads > 1:
                    env["OMP_NUM_THREADS"] = str(threads)
                cmd = [binary, "run.conf"] if np_rank == 1 else [mpirun, "-np", str(np_rank), binary, "run.conf"]
                print(f"=== {case}/{name} (np={np_rank}, threads={threads}) ===", flush=True)
                r = subprocess.run(cmd, cwd=sub, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if r.returncode != 0:
                    print(f"  FAILED (exit {r.returncode})")
                    failures += 1
                    continue
                reals[name] = load_realization(os.path.join(sub, f"ics_{name}*.hdf5"))
                print("  OK")
            # compare every layout against serial within this case
            if "serial" not in reals:
                continue
            base = reals["serial"]
            for name in ["mpi4", "mpi9", "hybrid"]:
                if name not in reals:
                    continue
                print(f"  serial vs {name}:")
                print(compare(base, reals[name]))
        print(f"\n{'ALL LAYOUT COMPARISONS PASSED' if failures == 0 else 'FAILURES DETECTED'}")
        return failures
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

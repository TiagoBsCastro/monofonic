#!/usr/bin/env bash
# Regression tests for the per-node shared-memory FastDF window.
#
# Usage:  run_tests.sh /path/to/monofonIC [mpirun]
#
# Exercises:
#   * serial run (1 MPI rank)
#   * several MPI ranks on one node
#   * more node-local ranks than neutrino-grid x-planes (empty slices)
#
# For a fixed random seed the neutrino positions must be bit-identical across all
# layouts, and the total particle count must equal NeutrinoCubeRootNum^3.

set -euo pipefail

BIN="$(cd "$(dirname "${1:-monofonIC}")" && pwd)/$(basename "${1:-monofonIC}")"
MPIRUN="${2:-mpirun}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# The CLASS transfer function (zwindstroom) is computed on every run, so each
# layout takes ~30 s.  Keep the grid small so the empty-slice layout (np 9) is
# cheap to run.
for layout in "serial:1" "mpi4:4" "mpi9:9"; do
    name="${layout%%:*}"; np="${layout##*:}"
    cfg="$WORK/smoke_${name}.conf"
    out="ics_${name}.hdf5"
    sed "s|FILENAME|${out}|" "$HERE/neutrino_smoke.conf" > "$cfg"
    echo "=== $name (np $np) ==="
    if [[ "$np" == "1" ]]; then
        ( cd "$WORK" && "$BIN" "smoke_${name}.conf" >/dev/null 2>&1 )
    else
        ( cd "$WORK" && "$MPIRUN" -np "$np" "$BIN" "smoke_${name}.conf" >/dev/null 2>&1 )
    fi
    echo "  -> OK"
done

# Verify bit-identity of positions and the total particle count across layouts.
python3 - "$WORK" <<'PY'
import sys, glob, h5py, numpy as np
work = sys.argv[1]
files = sorted(glob.glob(work + "/ics_*.hdf5"))
assert files, "no output files produced"
ref_ids = ref_c = None
total = 0
for f in files:
    with h5py.File(f, "r") as h:
        c = h["PartType2/Coordinates"][:]
        ids = h["PartType2/ParticleIDs"][:]
        total += len(c)
        o = np.argsort(ids)
        if ref_ids is None:
            ref_ids, ref_c = ids[o], c[o]
        else:
            assert np.array_equal(ref_ids, ids[o]), f"ID mismatch in {f}"
            assert np.array_equal(ref_c, c[o]), f"position mismatch in {f}"
print(f"total particles = {total}; positions bit-identical across layouts")
assert total == 8**3, f"expected {8**3} particles, got {total}"
PY

echo "ALL TESTS PASSED"

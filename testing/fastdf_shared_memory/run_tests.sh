#!/usr/bin/env bash
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

# Regression tests for the per-node shared-memory FastDF window.
#
# Usage: run_tests.sh BRANCH_BINARY MASTER_BINARY [MPIRUN] [smoke|full] [REQUIRED_NODES]

set -uo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: run_tests.sh BRANCH_BINARY MASTER_BINARY [MPIRUN] [smoke|full] [REQUIRED_NODES]" >&2
    exit 2
fi

BIN="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
MASTER_BIN="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
MPIRUN_SPEC="${3:-mpirun}"
read -r -a MPIRUN_CMD <<< "$MPIRUN_SPEC"
SUITE="${4:-smoke}"
REQUIRED_NODES="${5:-1}"
FASTDF_TEST_PYTHON="${FASTDF_TEST_PYTHON:-python3}"
FASTDF_TEST_MPICC_CMD="${FASTDF_TEST_MPICC:-mpicc}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$(dirname "$BIN")"
FASTDF_SRC="$(sed -n 's/^fastdf_SOURCE_DIR:[^=]*=//p' "$BUILD/CMakeCache.txt" | head -1)"
CLASS_SRC="$(sed -n 's/^class_SOURCE_DIR:[^=]*=//p' "$BUILD/CMakeCache.txt" | head -1)"
FASTDF_SRC="${FASTDF_SRC:-$BUILD/_deps/fastdf-src}"
CLASS_SRC="${CLASS_SRC:-$BUILD/_deps/class-src}"
FASTDF_TEST_TMPDIR="${FASTDF_TEST_TMPDIR:-$BUILD}"
if [[ ! -d "$FASTDF_TEST_TMPDIR" || ! -w "$FASTDF_TEST_TMPDIR" ]]; then
    echo "ERROR: FASTDF_TEST_TMPDIR must be a writable directory: $FASTDF_TEST_TMPDIR" >&2
    exit 2
fi
WORK="$(mktemp -d -p "$FASTDF_TEST_TMPDIR" fastdf_shared_tests.XXXXXX)" || {
    echo "ERROR: could not create the shared test work directory" >&2
    exit 2
}
# Python's case inputs must be visible from every requested node. Pass an
# explicit shared root without changing TMPDIR: MPI launchers commonly require
# their own temporary/session files to stay on node-local storage.
CLASS_INI="$WORK/input_class_parameters.ini"
trap 'rm -rf "$WORK"' EXIT

fail=0

if [[ "$SUITE" != "smoke" && "$SUITE" != "full" ]]; then
    echo "ERROR: suite must be 'smoke' or 'full'" >&2
    exit 2
fi
if [[ ${#MPIRUN_CMD[@]} -eq 0 ]]; then
    echo "ERROR: MPIRUN must contain a launcher command" >&2
    exit 2
fi
if [[ ! "$REQUIRED_NODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: REQUIRED_NODES must be a positive integer" >&2
    exit 2
fi
if ! "$FASTDF_TEST_PYTHON" -c 'import h5py, numpy' >/dev/null 2>&1; then
    echo "ERROR: $FASTDF_TEST_PYTHON must provide NumPy and h5py" >&2
    exit 2
fi

HDF5_LIBDIR="$(pkg-config --variable=libdir hdf5 2>/dev/null || true)"
HDF5_RPATH=()
if [[ -n "$HDF5_LIBDIR" ]]; then
    HDF5_RPATH=(-Wl,-rpath,"$HDF5_LIBDIR")
fi

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: branch binary is not executable: $BIN" >&2
    exit 2
fi
if [[ ! -x "$MASTER_BIN" ]]; then
    echo "ERROR: master binary is not executable: $MASTER_BIN" >&2
    exit 2
fi

echo "==== layout + master comparison tests ===="
"$FASTDF_TEST_PYTHON" "$HERE/run_tests.py" "$BIN" "$MASTER_BIN" \
    --mpirun "$MPIRUN_SPEC" --suite "$SUITE" --require-nodes "$REQUIRED_NODES" \
    --class-ini-out "$CLASS_INI" --work-root "$WORK" || fail=1

if [[ ! -s "$CLASS_INI" ]]; then
    echo "ERROR: comparison run did not produce a usable input_class_parameters.ini" >&2
    fail=1
fi

echo "==== HDF5 failure handling ===="
FDF_OBJ="$(find "$BUILD" -path '*fastdf_static.dir/src/input.c.o' 2>/dev/null | head -1)"
MINI_OBJ="$(find "$BUILD" -path '*fastdf_static.dir/parser/minIni.c.o' 2>/dev/null | head -1)"
if [[ -n "$FDF_OBJ" && -n "$MINI_OBJ" && -d "$FASTDF_SRC/include" ]]; then
    if "$FASTDF_TEST_MPICC_CMD" "$HERE/test_readfield.c" "$FDF_OBJ" "$MINI_OBJ" \
        -I"$FASTDF_SRC/include" -I"$FASTDF_SRC/parser" \
        $(pkg-config --cflags hdf5 2>/dev/null || true) \
        $(pkg-config --libs hdf5 2>/dev/null || echo -lhdf5) \
        "${HDF5_RPATH[@]}" \
        -lgsl -lgslcblas -lm -o "$WORK/test_readfield"; then
        ( cd "$WORK" && ./test_readfield ) || { echo "test_readfield FAILED"; fail=1; }
    else
        echo "ERROR: could not build test_readfield" >&2
        fail=1
    fi
else
    echo "ERROR: required FastDF objects/source not found; cannot run test_readfield" >&2
    fail=1
fi

echo "==== end-to-end FastDF failure ===="
FASTDF_LIB="$BUILD/_deps/fastdf-build/libfastdf.a"
CLASS_CPP="$BUILD/_deps/class-build/libclass_cpp.a"
CLASS="$BUILD/_deps/class-build/libclass.a"
if [[ -f "$FASTDF_LIB" && -f "$CLASS_CPP" && -f "$CLASS" && -s "$CLASS_INI" ]]; then
    if "$FASTDF_TEST_MPICC_CMD" -fopenmp "$HERE/test_fastdf_failure.c" \
        -I"$FASTDF_SRC/include" \
        -I"$CLASS_SRC/include" \
        -I"$CLASS_SRC/external/heating" \
        -I"$CLASS_SRC/external/HyRec2020" \
        -I"$CLASS_SRC/external/RecfastCLASS" \
        "$FASTDF_LIB" "$CLASS_CPP" "$CLASS" \
        -lfftw3_mpi -lfftw3_threads -lfftw3 -lgsl -lgslcblas -lhdf5 -lm -lstdc++ \
        "${HDF5_RPATH[@]}" \
        -o "$WORK/test_fastdf_failure"; then
        ( cd "$WORK" && OMP_NUM_THREADS=1 ./test_fastdf_failure "$CLASS_INI" ) \
            || { echo "serial failure test FAILED"; fail=1; }
        ( cd "$WORK" && OMP_NUM_THREADS=1 "${MPIRUN_CMD[@]}" -np 2 \
            ./test_fastdf_failure "$CLASS_INI" ) \
            || { echo "MPI failure test FAILED"; fail=1; }
        ( cd "$WORK" && OMP_NUM_THREADS=1 ./test_fastdf_failure \
            "$CLASS_INI" missing-class ) \
            || { echo "serial CLASS failure test FAILED"; fail=1; }
        ( cd "$WORK" && OMP_NUM_THREADS=1 "${MPIRUN_CMD[@]}" -np 2 \
            ./test_fastdf_failure "$CLASS_INI" missing-class ) \
            || { echo "MPI CLASS failure test FAILED"; fail=1; }
    else
        echo "ERROR: could not build test_fastdf_failure" >&2
        fail=1
    fi
else
    echo "ERROR: required FastDF/CLASS libraries or generated CLASS input not found" >&2
    fail=1
fi

if [[ "$fail" == "0" ]]; then
    echo "ALL REQUIRED TESTS PASSED"
else
    echo "SOME REQUIRED TESTS FAILED OR COULD NOT RUN"
    exit 1
fi

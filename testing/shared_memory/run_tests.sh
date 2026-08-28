#!/usr/bin/env bash
# Regression tests for the per-node shared-memory FastDF window.
#
# Usage:  run_tests.sh /path/to/monofonIC [mpirun]
#
# Runs:
#   * run_tests.py  -- serial/mpi4/mpi9/hybrid layouts, both grid cases and
#                      inversion, reconstructing the full realization and
#                      comparing IDs/positions/velocities/phase-space-densities;
#   * test_readfield.c      -- HDF5 failure handling (missing file/dataset,
#                              malformed dims, valid header/body);
#   * test_fastdf_failure.c -- end-to-end run_fastdf()<0 with a missing
#                              white-noise file (serial and 2 MPI ranks).
#
# The failure drivers are compiled against the FastDF object files / static
# library in the build tree next to the monofonIC binary, and are skipped with a
# warning if those artifacts are not found.

set -uo pipefail

BIN="$(cd "$(dirname "${1:?usage: run_tests.sh /path/to/monofonIC [mpirun]}")" && pwd)/$(basename "$1")"
MPIRUN="${2:-mpirun}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$(dirname "$BIN")"

fail=0

echo "==== comparison tests ===="
python3 "$HERE/run_tests.py" "$BIN" "$MPIRUN" || fail=1

echo "==== HDF5 failure handling ===="
FDF_OBJ="$(find "$BUILD" -path '*fastdf_static.dir/src/input.c.o' 2>/dev/null | head -1)"
MINI_OBJ="$(find "$BUILD" -path '*fastdf_static.dir/parser/minIni.c.o' 2>/dev/null | head -1)"
FDF_INC="$(dirname "$(dirname "$FDF_OBJ")")"
if [[ -n "$FDF_OBJ" && -n "$MINI_OBJ" ]]; then
    gcc "$HERE/test_readfield.c" "$FDF_OBJ" "$MINI_OBJ" \
        -I"$FDF_INC/include" -I"$FDF_INC/parser" \
        $(pkg-config --cflags hdf5 2>/dev/null || true) \
        $(pkg-config --libs hdf5 2>/dev/null || echo -lhdf5) \
        -lgsl -lgslcblas -lm -o "$BUILD/test_readfield" 2>/dev/null \
        && ( cd "$BUILD" && ./test_readfield ) || { echo "test_readfield FAILED"; fail=1; }
else
    echo "WARNING: FastDF object files not found; skipping test_readfield"
fi

echo "==== end-to-end FastDF failure ===="
FASTDF_LIB="$BUILD/_deps/fastdf-build/libfastdf.a"
CLASS_CPP="$BUILD/_deps/class-build/libclass_cpp.a"
CLASS="$BUILD/_deps/class-build/libclass.a"
if [[ -f "$FASTDF_LIB" && -f "$CLASS_CPP" && -f "$CLASS" ]]; then
    mpicc -fopenmp "$HERE/test_fastdf_failure.c" \
        -I"$BUILD/_deps/fastdf-src/include" \
        -I"$BUILD/_deps/class-src/include" \
        -I"$BUILD/_deps/class-src/external/heating" \
        -I"$BUILD/_deps/class-src/external/HyRec2020" \
        -I"$BUILD/_deps/class-src/external/RecfastCLASS" \
        "$FASTDF_LIB" "$CLASS_CPP" "$CLASS" \
        -lfftw3_mpi -lfftw3_threads -lfftw3 -lgsl -lgslcblas -lhdf5 -lm -lstdc++ \
        -o "$BUILD/test_fastdf_failure" 2>/dev/null
    if [[ -x "$BUILD/test_fastdf_failure" ]]; then
        ( cd "$BUILD" && ./test_fastdf_failure ) || { echo "serial failure test FAILED"; fail=1; }
        ( cd "$BUILD" && "$MPIRUN" -np 2 ./test_fastdf_failure ) || { echo "MPI failure test FAILED"; fail=1; }
    else
        echo "WARNING: could not build test_fastdf_failure; skipping"; fail=1
    fi
else
    echo "WARNING: FastDF/CLASS static libraries not found; skipping test_fastdf_failure"
fi

if [[ "$fail" == "0" ]]; then
    echo "ALL TESTS PASSED"
else
    echo "SOME TESTS FAILED"
    exit 1
fi

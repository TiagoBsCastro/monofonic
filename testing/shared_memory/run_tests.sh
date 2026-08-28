#!/usr/bin/env bash
# Regression tests for the per-node shared-memory FastDF window.
#
# Usage: run_tests.sh /path/to/branch/monofonIC /path/to/master/monofonIC [mpirun]

set -uo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: run_tests.sh /path/to/branch/monofonIC /path/to/master/monofonIC [mpirun]" >&2
    exit 2
fi

BIN="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
MASTER_BIN="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
MPIRUN="${3:-mpirun}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$(dirname "$BIN")"
FASTDF_SRC="$BUILD/_deps/fastdf-src"
WORK="$(mktemp -d -t fastdf_shared_tests.XXXXXX)"
CLASS_INI="$WORK/input_class_parameters.ini"
trap 'rm -rf "$WORK"' EXIT

fail=0

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: branch binary is not executable: $BIN" >&2
    exit 2
fi
if [[ ! -x "$MASTER_BIN" ]]; then
    echo "ERROR: master binary is not executable: $MASTER_BIN" >&2
    exit 2
fi

echo "==== layout + master comparison tests ===="
python3 "$HERE/run_tests.py" "$BIN" "$MASTER_BIN" \
    --mpirun "$MPIRUN" --class-ini-out "$CLASS_INI" || fail=1

if [[ ! -s "$CLASS_INI" ]]; then
    echo "ERROR: comparison run did not produce a usable input_class_parameters.ini" >&2
    fail=1
fi

echo "==== HDF5 failure handling ===="
FDF_OBJ="$(find "$BUILD" -path '*fastdf_static.dir/src/input.c.o' 2>/dev/null | head -1)"
MINI_OBJ="$(find "$BUILD" -path '*fastdf_static.dir/parser/minIni.c.o' 2>/dev/null | head -1)"
if [[ -n "$FDF_OBJ" && -n "$MINI_OBJ" && -d "$FASTDF_SRC/include" ]]; then
    if gcc "$HERE/test_readfield.c" "$FDF_OBJ" "$MINI_OBJ" \
        -I"$FASTDF_SRC/include" -I"$FASTDF_SRC/parser" \
        $(pkg-config --cflags hdf5 2>/dev/null || true) \
        $(pkg-config --libs hdf5 2>/dev/null || echo -lhdf5) \
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
    if mpicc -fopenmp "$HERE/test_fastdf_failure.c" \
        -I"$FASTDF_SRC/include" \
        -I"$BUILD/_deps/class-src/include" \
        -I"$BUILD/_deps/class-src/external/heating" \
        -I"$BUILD/_deps/class-src/external/HyRec2020" \
        -I"$BUILD/_deps/class-src/external/RecfastCLASS" \
        "$FASTDF_LIB" "$CLASS_CPP" "$CLASS" \
        -lfftw3_mpi -lfftw3_threads -lfftw3 -lgsl -lgslcblas -lhdf5 -lm -lstdc++ \
        -o "$WORK/test_fastdf_failure"; then
        ( cd "$WORK" && ./test_fastdf_failure "$CLASS_INI" ) \
            || { echo "serial failure test FAILED"; fail=1; }
        ( cd "$WORK" && "$MPIRUN" -np 2 ./test_fastdf_failure "$CLASS_INI" ) \
            || { echo "MPI failure test FAILED"; fail=1; }
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

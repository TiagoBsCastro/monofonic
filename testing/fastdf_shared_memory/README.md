# FastDF shared-memory regression tests

The smoke suite compares a serial branch run with four MPI ranks and with the
master binary, then exercises the HDF5 and collective failure paths:

```sh
./testing/fastdf_shared_memory/run_tests.sh \
    /path/to/branch/monofonIC /path/to/master/monofonIC
```

The full suite additionally covers nine ranks, hybrid MPI/OpenMP, equal-sized
grids, odd-sized FFT grids, and white-noise inversion:

```sh
./testing/fastdf_shared_memory/run_tests.sh \
    /path/to/branch/monofonIC /path/to/master/monofonIC mpirun full
```

Both the zwindstroom transfer plugin and FastDF execute their CLASS work on a
node-local leader while the other ranks wait. FastDF also runs each transform
on its leader. To let the CLASS OpenMP teams and FFTW thread pool borrow the
CPUs assigned to the waiting ranks, the leader's affinity mask must include
them. With Open MPI, pass a launcher command such as `"mpirun --bind-to none"`;
use the corresponding no-binding option for other launchers. The two workflows
report their requested, visible, and selected CLASS thread counts once per node;
FastDF does the same for FFT. The suite checks the exact number of reports,
summed thread requests, and affinity caps against the detected node layout.

For the merge-gate run, request at least two physical nodes from the scheduler
and pass the required node count as the final argument. The harness probes the
MPI placement and fails if the launcher used fewer nodes:

```sh
./testing/fastdf_shared_memory/run_tests.sh \
    /path/to/branch/monofonIC /path/to/master/monofonIC \
    mpirun full 2
```

The scripts require Python with NumPy and h5py, an MPI C compiler, FFTW, HDF5,
GSL, and access to the dependency objects in the branch build directory. They
also validate the generated CLASS control file, require its N-body-gauge option
exactly once, inject invalid zwindstroom CLASS parameters to test synchronized
MPI failure, and check FastDF's missing white-noise and CLASS input paths.
Successful runs delete their temporary output; failed monofonIC runs print the
tail of their per-case log. Set `FASTDF_TEST_PYTHON` if the required modules are
not installed for the default `python3` executable.
The harness creates its work tree under the branch build directory so every MPI
rank can see the same input files on a multi-node run. Set `FASTDF_TEST_TMPDIR`
to another writable shared filesystem directory when needed; do not use
node-local `/tmp` for this case.

The master comparison deliberately uses equal parent and neutrino grid sizes.
This isolates pristine versus shared-memory FastDF while giving both versions
bit-identical input. The branch also fixes monofonIC's reduction of a
downsampled, padded real FFT grid; comparing an unequal-resolution branch run
to unmodified master would therefore compare different white-noise fields.
The unequal-resolution serial/MPI comparisons directly check that reduction.

On the multi-node run, record per-node proportional-set size or scheduler
high-water memory alongside the test result. Increasing ranks per node should
not add another copy of either the FastDF mesh window or the extracted CLASS
tables. Particle storage remains rank-local. The grid-only peak changes from
roughly 16 real-grid equivalents per MPI rank to one per-node window of eight
real grids plus four half-complex grids (roughly 12 real-grid equivalents for
large grids). FastDF's internal CLASS execution also runs once per node instead
of once per rank; only its leader temporarily owns that private CLASS working
set.

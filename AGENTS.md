# Repository Guidelines

monofonIC (MUSIC2) is a modular, high-precision C++14 initial-conditions generator for cosmological simulations. It builds with CMake and runs with MPI + OpenMP hybrid parallelism.

## Project Structure & Module Organization

- `src/` — core implementation (`.cc` files), with `src/plugins/` holding plugin implementations.
- `include/` — public headers (`.hh` files), with `include/math/` for math utilities.
- `testing/` — integration-test configuration files (e.g., `testing/RAMSES/`).
- `external/` — third-party code and submodules (CLASS, Panphasia, zwindstroom, FastDF, GenericIO).
- `CMakeLists.txt` — build system; `example.conf` — annotated parameter reference; `Doxyfile.in` — API docs.

Every source file begins with the standard GPLv3 header block.

## Build, Test, and Development Commands

Configure and build out-of-source with CMake:

```bash
mkdir build && cd build
ccmake ..   # toggle options (MPI, CLASS, PLT, ...)
make        # produces the monofonIC executable
```

If FFTW3/HDF5 are not found automatically, pass paths explicitly:
`FFTW3_ROOT=<path> HDF5_ROOT=<path> ccmake ..`

Run with a parameter file, or under MPI:

```bash
./monofonIC ../example.conf
mpirun -np 16 ./monofonIC <config>
```

Generate API documentation by setting `BUILD_DOCUMENTATION=ON`, then run `make doc_doxygen`.

## Coding Style & Naming Conventions

- C++14, 4-space indentation, braces on their own line.
- Files use `snake_case` with `.cc`/`.hh` extensions (e.g., `grid_fft.hh`, `output_gadget_hdf5.cc`).
- Types are `snake_case`; private members carry a trailing underscore (`header_`, `lunit_`).
- Namespaces are lowercase (`testing`, `cosmology`); use `#pragma once` and Doxygen `//!` comments.
- No linter is enforced — match the style of surrounding code.

## Testing Guidelines

Unit testing is a work in progress and has no formal framework yet. Regression configs live in `testing/`; `src/testing.cc` provides analysis helpers under the `testing` namespace. When adding a feature, include a runnable config or analysis output.

## Commit & Pull Request Guidelines

- Commit messages are short, imperative, and capitalized ("Add ...", "Fix ..."), matching existing history.
- Branch from `master` and submit changes via pull request.
- Discuss larger changes in an issue first; bug reports need reproduction steps plus the config and build options used.
- Document any new config-file option in `example.conf`.

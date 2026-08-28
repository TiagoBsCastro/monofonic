/* End-to-end check that a FastDF white-noise input failure is reported as a
 * negative return value (which monofonIC's ic_generator.cc turns into a thrown
 * std::runtime_error).  run_fastdf() must return < 0, and with several MPI
 * ranks every rank must reach the same failure without deadlocking.
 *
 * Build against the FastDF + CLASS + FFTW + GSL + HDF5 libraries (see
 * run_tests.sh for the exact link line).  Requires input_class_parameters.ini
 * (produced by a prior monofonIC run) in the working directory.
 */
#include <stdio.h>
#include <string.h>
#include <mpi.h>

#include "input.h"
#include "runner.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    struct params pars;
    struct units us;

    /* Same unit system monofonIC uses. */
    us.UnitLengthMetres = 3.085677581491e22;     /* Mpc */
    us.UnitTimeSeconds = 3.08567758148957e19;    /* for km/s velocities */
    us.UnitMassKilogram = 1.988435e40;           /* 1e10 M_sol */
    us.UnitTemperatureKelvin = 1.0;
    us.UnitCurrentAmpere = 1.0;
    setPhysicalConstants(&us);

    initParams(&pars);
    pars.FirstID = 1;
    pars.CubeRootNumber = 8;
    pars.NumPartGenerate = 512;
    pars.ScaleFactorBegin = 1e-9;
    pars.ScaleFactorEnd = 1.0 / 25.0;
    pars.ScaleFactorStep = 0.05;
    pars.RecomputeTrigger = 0.01;
    pars.RecomputeScaleRef = 0.0;
    pars.InterpolationOrder = 1;
    pars.InvertField = 0;
    pars.BoxLen = 300.0;
    pars.AlternativeEquations = 0;
    pars.OutputFields = 0;
    pars.NormalizeGaussianField = 0;
    pars.AssumeMonofonicNormalization = 1;
    pars.IncludeHubbleFactors = 1;
    pars.DistributedFiles = 0;
    pars.ParticleSink = NULL;
    pars.rank = rank;

    strcpy(pars.OutputDirectory, ".");
    strcpy(pars.ExportName, "PartType2");
    strcpy(pars.OutputFilename, "ics_failtest.hdf5");
    /* Deliberately point FastDF at a white-noise file that does not exist. */
    strcpy(pars.GaussianRandomFieldFile, "missing_white_noise.hdf5");
    strcpy(pars.GaussianRandomFieldDataset, "white_noise");
    strcpy(pars.TransferFunctionDensity, "d_ncdm[0]");
    strcpy(pars.Gauge, "Newtonian");
    strcpy(pars.ClassIniFile, "input_class_parameters.ini");
    strcpy(pars.VelocityType, "Gadget");

    long long result = run_fastdf(&pars, &us);
    if (rank == 0)
        printf("run_fastdf returned %lld\n", result);

    int ok = (result < 0) ? 1 : 0;
    int global_ok = 0;
    MPI_Allreduce(&ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    MPI_Finalize();

    if (global_ok) {
        if (rank == 0) printf("PASS: failure reported as negative on all ranks\n");
        return 0;
    }
    if (rank == 0) printf("FAIL: expected negative return (got %lld)\n", result);
    return 1;
}

// This file is part of monofonIC (MUSIC2).

#include <algorithm>
#include <array>
#include <cfenv>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <config_file.hh>
#include <embedded.hh>
#include <general.hh>
#include <ic_generator.hh>

namespace
{
std::vector<double> background_scale_factors;
std::vector<double> background_hubble_rates;
const monofonic_particle_sink *particle_sink = nullptr;
bool particle_transfer_started = false;
std::array<std::uint64_t, 6> expected_particle_counts{{0, 0, 0, 0, 0, 0}};
std::array<std::uint64_t, 6> local_particle_counts{{0, 0, 0, 0, 0, 0}};

void initialise_embedded_runtime(config_file &config, int num_threads)
{
#if defined(NDEBUG)
    music::logger::set_level(music::log_level::info);
#else
    music::logger::set_level(music::log_level::debug);
#endif

#if defined(USE_MPI)
    int mpi_initialised = 0;
    MPI_Initialized(&mpi_initialised);
    if (!mpi_initialised)
        throw std::runtime_error("embedded monofonIC requires MPI to be initialised by the host");

    MPI_Query_thread(&CONFIG::MPI_thread_support);
    CONFIG::MPI_threads_ok = CONFIG::MPI_thread_support >= MPI_THREAD_MULTIPLE;
    MPI_Comm_rank(MPI_COMM_WORLD, &CONFIG::MPI_task_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &CONFIG::MPI_task_size);
    CONFIG::MPI_ok = true;
    if (CONFIG::MPI_task_rank != 0)
        music::logger::set_level(music::log_level::error);
#endif

    CONFIG::num_threads = num_threads > 0
                              ? num_threads
                              : config.get_value_safe<unsigned>("execution", "NumThreads",
                                                                std::thread::hardware_concurrency());

#if defined(USE_FFTW_THREADS)
    CONFIG::FFTW_threads_ok = FFTW_API(init_threads)();
    if (CONFIG::FFTW_threads_ok)
        FFTW_API(plan_with_nthreads)(CONFIG::num_threads);
#endif
#if defined(USE_MPI)
    FFTW_API(mpi_init)();
#endif
#if defined(_OPENMP)
    omp_set_num_threads(CONFIG::num_threads);
#endif
}

void clear_background()
{
    background_scale_factors.clear();
    background_hubble_rates.clear();
}

void require_background()
{
    if (background_scale_factors.size() < 2 || background_scale_factors.size() != background_hubble_rates.size())
        throw std::runtime_error("the selected transfer plugin did not provide a valid internal background table");
}

void fill_background_metadata(const cosmology::parameters &parameters, monofonic_background_metadata &metadata)
{
    constexpr double class_default_neutrino_temperature_ratio = 0.71611;
    constexpr double boltzmann_joule_per_kelvin = 1.380649e-23;
    constexpr double electronvolt_joule = 1.602176634e-19;

    metadata = {};
    metadata.abi_version = MONOFONIC_PARTICLE_SINK_ABI;
    metadata.struct_size = sizeof(metadata);
    metadata.omega_m = parameters.get("Omega_m");
    metadata.omega_b = parameters.get("Omega_b");
    metadata.omega_lambda = parameters.get("Omega_DE");
    metadata.omega_radiation = parameters.get("Omega_r");
    metadata.omega_curvature = parameters.get("Omega_k");
    metadata.hubble_parameter = parameters.get("h");
    metadata.neutrino_mass_ev = parameters.get("m_nu1");
    metadata.neutrino_temperature_ev = parameters.get("Tcmb") * class_default_neutrino_temperature_ratio *
                                       boltzmann_joule_per_kelvin / electronvolt_joule;
}

void check_callback_status(int local_status, const char *callback_name)
{
    int global_status = local_status;
#if defined(USE_MPI)
    MPI_Allreduce(&local_status, &global_status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
    if (global_status != 0)
        throw std::runtime_error(std::string("OpenGadget particle sink ") + callback_name + " callback failed");
}
}

namespace monofonic
{
void set_background_table(const double *scale_factors, const double *hubble_rates, int size, double hubble_unit)
{
    if (size <= 0)
    {
        background_scale_factors.clear();
        background_hubble_rates.clear();
        return;
    }

    background_scale_factors.assign(scale_factors, scale_factors + size);
    background_hubble_rates.resize(static_cast<std::size_t>(size));
    std::transform(hubble_rates, hubble_rates + size, background_hubble_rates.begin(),
                   [hubble_unit](double value) { return value * hubble_unit; });
}

const monofonic_particle_sink *active_particle_sink()
{
    return particle_sink;
}

void emit_ic_metadata(const monofonic_ic_metadata &metadata)
{
    if (particle_sink == nullptr || particle_transfer_started)
        throw std::runtime_error("invalid in-memory particle transfer state at metadata emission");

    std::array<std::uint64_t, 6> global_hinted_counts{};
#if defined(USE_MPI)
    MPI_Allreduce(metadata.local_num_particles, global_hinted_counts.data(),
                  static_cast<int>(global_hinted_counts.size()), MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#else
    std::copy(std::begin(metadata.local_num_particles), std::end(metadata.local_num_particles),
              global_hinted_counts.begin());
#endif
    for (std::size_t type = 0; type < expected_particle_counts.size(); ++type)
    {
        if (global_hinted_counts[type] != metadata.num_particles[type])
            throw std::runtime_error("rank-local particle count hint mismatch for Gadget type " +
                                     std::to_string(type));
        expected_particle_counts[type] = metadata.num_particles[type];
    }
    local_particle_counts.fill(0);

    const int status = particle_sink->begin(particle_sink->context, &metadata);
    check_callback_status(status, "begin");
    particle_transfer_started = true;
}

void emit_particle_batch(const monofonic_particle_batch &batch)
{
    if (particle_sink == nullptr || !particle_transfer_started || batch.particle_type >= local_particle_counts.size())
        throw std::runtime_error("invalid in-memory particle batch");

    const int status = particle_sink->particles(particle_sink->context, &batch);
    check_callback_status(status, "particles");
    local_particle_counts[batch.particle_type] += batch.count;
}

void finish_particle_transfer()
{
    if (particle_sink == nullptr || !particle_transfer_started)
        throw std::runtime_error("in-memory particle transfer did not start");

    std::array<std::uint64_t, 6> global_particle_counts = local_particle_counts;
#if defined(USE_MPI)
    MPI_Allreduce(local_particle_counts.data(), global_particle_counts.data(),
                  static_cast<int>(global_particle_counts.size()), MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
    for (std::size_t type = 0; type < global_particle_counts.size(); ++type)
    {
        if (global_particle_counts[type] != expected_particle_counts[type])
            throw std::runtime_error("in-memory particle count mismatch for Gadget type " + std::to_string(type));
    }

    const int status = particle_sink->end(particle_sink->context);
    check_callback_status(status, "end");
    particle_transfer_started = false;
}
}

extern "C" int monofonic_generate_to_sink(const char *config_path, int num_threads,
                                            const monofonic_particle_sink *sink)
{
    if (config_path == nullptr || sink == nullptr || sink->abi_version != MONOFONIC_PARTICLE_SINK_ABI ||
        sink->struct_size < sizeof(monofonic_particle_sink) || sink->begin == nullptr || sink->particles == nullptr ||
        sink->end == nullptr)
        return 1;

    try
    {
        config_file config(config_path);
        config.insert_value("output", "format", "opengadget_memory");
        initialise_embedded_runtime(config, num_threads);

        clear_background();
        particle_sink = sink;
        particle_transfer_started = false;
        expected_particle_counts.fill(0);
        local_particle_counts.fill(0);
        std::feclearexcept(FE_ALL_EXCEPT);

        ic_generator::initialise(config);
        ic_generator::run(config);
        monofonic::finish_particle_transfer();
        ic_generator::reset();

        require_background();

        particle_sink = nullptr;
        return 0;
    }
    catch (const std::exception &error)
    {
        music::elog << "Embedded monofonIC failed: " << error.what() << std::endl;
        ic_generator::reset();
        particle_sink = nullptr;
        particle_transfer_started = false;
        return 1;
    }
    catch (...)
    {
        music::elog << "Embedded monofonIC failed with an unknown exception." << std::endl;
        ic_generator::reset();
        particle_sink = nullptr;
        particle_transfer_started = false;
        return 1;
    }
}

extern "C" int monofonic_prepare_background(const char *config_path, int num_threads,
                                               monofonic_background_metadata *metadata)
{
    if (config_path == nullptr || metadata == nullptr || metadata->abi_version != MONOFONIC_PARTICLE_SINK_ABI ||
        metadata->struct_size < sizeof(monofonic_background_metadata))
        return 1;

    try
    {
        config_file config(config_path);
        initialise_embedded_runtime(config, num_threads);
        clear_background();
        particle_sink = nullptr;
        particle_transfer_started = false;
        std::feclearexcept(FE_ALL_EXCEPT);

        cosmology::calculator calculator(config);
        fill_background_metadata(calculator.cosmo_param_, *metadata);
        require_background();
        return 0;
    }
    catch (const std::exception &error)
    {
        music::elog << "Embedded monofonIC background failed: " << error.what() << std::endl;
        clear_background();
        return 1;
    }
    catch (...)
    {
        music::elog << "Embedded monofonIC background failed with an unknown exception." << std::endl;
        clear_background();
        return 1;
    }
}

extern "C" std::size_t monofonic_background_size()
{
    return background_scale_factors.size();
}

extern "C" int monofonic_background_point(std::size_t index, double *scale_factor, double *hubble_km_s_mpc)
{
    if (index >= background_scale_factors.size() || scale_factor == nullptr || hubble_km_s_mpc == nullptr)
        return 1;

    *scale_factor = background_scale_factors[index];
    *hubble_km_s_mpc = background_hubble_rates[index];
    return 0;
}

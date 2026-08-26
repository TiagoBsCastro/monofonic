// This file is part of monofonIC (MUSIC2).
#pragma once

#include <cstddef>
#include <cstdint>

/** ABI version of the in-memory particle transfer interface. */
#define MONOFONIC_PARTICLE_SINK_ABI 5u

enum monofonic_data_type : std::uint32_t
{
    MONOFONIC_DATA_NONE = 0,
    MONOFONIC_DATA_FLOAT32 = 1,
    MONOFONIC_DATA_FLOAT64 = 2,
    MONOFONIC_DATA_UINT32 = 3,
    MONOFONIC_DATA_UINT64 = 4
};

/** Gadget-compatible metadata supplied before the first particle batch. */
struct monofonic_ic_metadata
{
    std::uint32_t abi_version;
    std::uint32_t struct_size;
    std::uint64_t num_particles[6];
    /** Exact number of particles this MPI rank will emit, by Gadget type. */
    std::uint64_t local_num_particles[6];
    double mass_table[6];
    double time;
    double redshift;
    double box_size;
    double omega_m;
    double omega_b;
    double omega_lambda;
    double omega_radiation;
    double omega_curvature;
    double hubble_parameter;
};

/** Cosmological metadata returned by background-only initialisation. */
struct monofonic_background_metadata
{
    std::uint32_t abi_version;
    std::uint32_t struct_size;
    double omega_m;
    double omega_b;
    double omega_lambda;
    double omega_radiation;
    double omega_curvature;
    double hubble_parameter;
    double neutrino_mass_ev;
    double neutrino_temperature_ev;
};

/**
 * A rank-local particle batch. Vector components are contiguous; strides move
 * from one particle to the next. A null IDs view denotes the affine sequence
 * first_id + i * id_increment. A null mass view denotes uniform_mass.
 * All views are borrowed and remain valid only for the duration of the call.
 */
struct monofonic_particle_batch
{
    std::uint32_t abi_version;
    std::uint32_t struct_size;
    std::uint32_t particle_type;
    std::uint32_t real_type;
    std::uint32_t id_type;
    std::uint32_t reserved;
    std::uint64_t count;
    const void *positions;
    std::size_t position_stride;
    const void *velocities;
    std::size_t velocity_stride;
    const void *phase_space_densities;
    std::size_t phase_space_density_stride;
    const void *ids;
    std::size_t id_stride;
    std::uint64_t first_id;
    std::uint64_t id_increment;
    const void *masses;
    std::size_t mass_stride;
    double uniform_mass;
    double neutrino_mass_ev;
    double neutrino_temperature_ev;
};

typedef int (*monofonic_ic_begin_callback)(void *context, const monofonic_ic_metadata *metadata);
typedef int (*monofonic_particle_batch_callback)(void *context, const monofonic_particle_batch *batch);
typedef int (*monofonic_ic_end_callback)(void *context);

struct monofonic_particle_sink
{
    std::uint32_t abi_version;
    std::uint32_t struct_size;
    void *context;
    monofonic_ic_begin_callback begin;
    monofonic_particle_batch_callback particles;
    monofonic_ic_end_callback end;
};

namespace monofonic
{
void set_background_table(const double *scale_factors, const double *hubble_rates, int size, double hubble_unit);

/** Internal producer helpers used by the memory output plugin and FastDF. */
const monofonic_particle_sink *active_particle_sink();
void emit_ic_metadata(const monofonic_ic_metadata &metadata);
void emit_particle_batch(const monofonic_particle_batch &batch);
void finish_particle_transfer();
}

extern "C"
{
/** Run monofonIC inside a process which has already initialised MPI. */
int monofonic_generate_to_sink(const char *config_path, int num_threads, const monofonic_particle_sink *sink);

/** Build only the cosmological background, without generating particles. */
int monofonic_prepare_background(const char *config_path, int num_threads, monofonic_background_metadata *metadata);

/** Access the CLASS/zwindstroom background retained by the last run. */
std::size_t monofonic_background_size();
int monofonic_background_point(std::size_t index, double *scale_factor, double *hubble_km_s_mpc);
}

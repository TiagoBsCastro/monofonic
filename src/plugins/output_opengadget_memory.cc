// In-memory Gadget particle output used by the embedded OpenGadget3 bridge.

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include <embedded.hh>
#include <general.hh>
#include <output_plugin.hh>

namespace
{
std::uint64_t cube(std::uint64_t value)
{
    if (value != 0 && value > std::numeric_limits<std::uint64_t>::max() / value / value)
        throw std::overflow_error("particle-grid size exceeds the in-memory interface range");
    return value * value * value;
}

std::uint64_t multiply_checked(std::uint64_t left, std::uint64_t right)
{
    if (right != 0 && left > std::numeric_limits<std::uint64_t>::max() / right)
        throw std::overflow_error("particle count exceeds the in-memory interface range");
    return left * right;
}

std::uint64_t local_lattice_cells(std::uint64_t side)
{
#if defined(USE_MPI)
    if (side > static_cast<std::uint64_t>(std::numeric_limits<ptrdiff_t>::max()))
        throw std::overflow_error("particle-grid side exceeds FFTW's index range");
    ptrdiff_t local_0_size = 0;
    ptrdiff_t local_0_start = 0;
    ptrdiff_t local_1_size = 0;
    ptrdiff_t local_1_start = 0;
    FFTW_API(mpi_local_size_3d_transposed)(static_cast<ptrdiff_t>(side), static_cast<ptrdiff_t>(side),
                                           static_cast<ptrdiff_t>(side), MPI_COMM_WORLD, &local_0_size,
                                           &local_0_start, &local_1_size, &local_1_start);
    return multiply_checked(multiply_checked(static_cast<std::uint64_t>(local_0_size), side), side);
#else
    return cube(side);
#endif
}

std::uint64_t ceil_partition_boundary(std::uint64_t side, std::uint64_t partition,
                                      std::uint64_t num_partitions)
{
    const std::uint64_t quotient = side / num_partitions;
    const std::uint64_t remainder = side % num_partitions;
    return quotient * partition + (remainder * partition + num_partitions - 1) / num_partitions;
}

std::uint64_t local_fastdf_cells(std::uint64_t side)
{
#if defined(USE_MPI)
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const std::uint64_t begin =
        ceil_partition_boundary(side, static_cast<std::uint64_t>(rank), static_cast<std::uint64_t>(size));
    const std::uint64_t end =
        ceil_partition_boundary(side, static_cast<std::uint64_t>(rank + 1), static_cast<std::uint64_t>(size));
    return multiply_checked(multiply_checked(end - begin, side), side);
#else
    return cube(side);
#endif
}

std::uint64_t lattice_multiplier(const std::string &load)
{
    if (load == "sc")
        return 1;
    if (load == "bcc")
        return 2;
    if (load == "fcc")
        return 4;
    if (load == "rsc")
        return 8;
    throw std::runtime_error("embedded OpenGadget output does not support particle load '" + load + "'");
}

class opengadget_memory_output_plugin : public output_plugin
{
    monofonic_ic_metadata metadata_{};
    real_t lunit_;
    real_t vunit_;
    real_t munit_;
    bool long_ids_;

  public:
    explicit opengadget_memory_output_plugin(config_file &cf, std::unique_ptr<cosmology::calculator> &pcc)
        : output_plugin(cf, pcc, "OpenGadget in-memory", false),
          long_ids_(cf.get_value_safe<bool>("output", "UseLongids", false))
    {
        if (monofonic::active_particle_sink() == nullptr)
            throw std::runtime_error("OpenGadget memory output selected without an active particle sink");

        const double zstart = cf_.get_value<double>("setup", "zstart");
        const std::uint64_t ngrid = cf_.get_value<std::uint64_t>("setup", "GridRes");
        const bool baryons = cf_.get_value_safe<bool>("setup", "DoBaryons", false);
        const bool neutrinos = cf_.get_value_safe<bool>("setup", "DoNeutrinoParticles", false);
        const std::string particle_load = cf_.get_value_safe<std::string>("setup", "ParticleLoad", "sc");
        const std::uint64_t multiplier = lattice_multiplier(particle_load);
        const std::uint64_t load_count = multiply_checked(cube(ngrid), multiplier);
        const std::uint64_t local_load_count = multiply_checked(local_lattice_cells(ngrid), multiplier);
        const std::uint64_t neutrino_side =
            cf_.get_value_safe<std::uint64_t>("setup", "NeutrinoCubeRootNum", 0);

        lunit_ = cf_.get_value<double>("setup", "BoxLength");
        vunit_ = lunit_ / std::sqrt(1.0 / (1.0 + zstart));
        const double rho_critical = 27.7519737;
        munit_ = rho_critical * lunit_ * lunit_ * lunit_;

        metadata_.abi_version = MONOFONIC_PARTICLE_SINK_ABI;
        metadata_.struct_size = sizeof(metadata_);
        metadata_.num_particles[1] = load_count;
        metadata_.num_particles[0] = baryons ? load_count : 0;
        metadata_.num_particles[2] = neutrinos ? cube(neutrino_side) : 0;
        metadata_.local_num_particles[1] = local_load_count;
        metadata_.local_num_particles[0] = baryons ? local_load_count : 0;
        metadata_.local_num_particles[2] = neutrinos ? local_fastdf_cells(neutrino_side) : 0;
        metadata_.time = 1.0 / (1.0 + zstart);
        metadata_.redshift = zstart;
        metadata_.box_size = lunit_;
        metadata_.omega_m = pcc_->cosmo_param_["Omega_m"];
        metadata_.omega_b = pcc_->cosmo_param_["Omega_b"];
        metadata_.omega_lambda = pcc_->cosmo_param_["Omega_DE"];
        metadata_.omega_radiation = pcc_->cosmo_param_["Omega_r"];
        metadata_.omega_curvature = pcc_->cosmo_param_["Omega_k"];
        metadata_.hubble_parameter = pcc_->cosmo_param_["h"];

        double omega_dm = pcc_->cosmo_param_["Omega_m"];
        if (baryons)
            omega_dm -= pcc_->cosmo_param_["Omega_b"];
        const bool with_neutrinos = cf_.get_value_safe<bool>("setup", "WithNeutrinos", false);
        const bool exclude_neutrinos =
            cf_.get_value_safe<bool>("setup", "ExcludeNeutrinos", with_neutrinos);
        if (exclude_neutrinos)
            omega_dm -= pcc_->cosmo_param_["Omega_nu_massive"];

        if (!baryons && load_count != 0)
            metadata_.mass_table[1] = omega_dm * munit_ / static_cast<double>(load_count);
        if (metadata_.num_particles[2] != 0)
            metadata_.mass_table[2] = pcc_->cosmo_param_["Omega_nu_1"] * munit_ /
                                      static_cast<double>(metadata_.num_particles[2]);

        monofonic::emit_ic_metadata(metadata_);
    }

    int get_species_idx(const cosmo_species &species) const override
    {
        switch (species)
        {
        case cosmo_species::baryon:
            return 0;
        case cosmo_species::dm:
            return 1;
        case cosmo_species::neutrino:
            return 2;
        }
        return -1;
    }

    output_type write_species_as(const cosmo_species &) const override { return output_type::particles; }
    bool has_64bit_reals() const override { return sizeof(real_t) > sizeof(float); }
    bool has_64bit_ids() const override { return long_ids_; }
    real_t position_unit() const override { return lunit_; }
    real_t velocity_unit() const override { return vunit_; }
    real_t mass_unit() const override { return munit_; }

    void write_particle_data(const particle::container &particles, const cosmo_species &species,
                             double) override
    {
        const int type = get_species_idx(species);
        if (type < 0)
            throw std::runtime_error("cannot map monofonIC species to a Gadget particle type");
        if (particles.get_global_num_particles() != metadata_.num_particles[type])
            throw std::runtime_error("generated particle count disagrees with OpenGadget metadata");

        monofonic_particle_batch batch{};
        batch.abi_version = MONOFONIC_PARTICLE_SINK_ABI;
        batch.struct_size = sizeof(batch);
        batch.particle_type = static_cast<std::uint32_t>(type);
        batch.count = particles.get_local_num_particles();
        batch.id_increment = 1;
        batch.uniform_mass = metadata_.mass_table[type];

        if (has_64bit_reals())
        {
            batch.real_type = MONOFONIC_DATA_FLOAT64;
            batch.positions = particles.positions64_.data();
            batch.velocities = particles.velocities64_.data();
            batch.position_stride = batch.velocity_stride = 3 * sizeof(double);
            if (particles.bhas_individual_masses_)
            {
                batch.masses = particles.mass64_.data();
                batch.mass_stride = sizeof(double);
            }
        }
        else
        {
            batch.real_type = MONOFONIC_DATA_FLOAT32;
            batch.positions = particles.positions32_.data();
            batch.velocities = particles.velocities32_.data();
            batch.position_stride = batch.velocity_stride = 3 * sizeof(float);
            if (particles.bhas_individual_masses_)
            {
                batch.masses = particles.mass32_.data();
                batch.mass_stride = sizeof(float);
            }
        }

        if (has_64bit_ids())
        {
            batch.id_type = MONOFONIC_DATA_UINT64;
            batch.ids = particles.ids64_.data();
            batch.id_stride = sizeof(std::uint64_t);
        }
        else
        {
            batch.id_type = MONOFONIC_DATA_UINT32;
            batch.ids = particles.ids32_.data();
            batch.id_stride = sizeof(std::uint32_t);
        }

        monofonic::emit_particle_batch(batch);
    }

    void write_strided_particle_data(const void *positions, std::size_t position_stride,
                                     const void *velocities, std::size_t velocity_stride,
                                     const void *phase_space_densities,
                                     std::size_t phase_space_density_stride,
                                     std::uint64_t numpart_local, std::uint64_t first_id,
                                     double uniform_mass, double neutrino_mass_ev,
                                     double neutrino_temperature_ev, const cosmo_species &species) override
    {
        monofonic_particle_batch batch{};
        batch.abi_version = MONOFONIC_PARTICLE_SINK_ABI;
        batch.struct_size = sizeof(batch);
        batch.particle_type = static_cast<std::uint32_t>(get_species_idx(species));
        batch.real_type = MONOFONIC_DATA_FLOAT64;
        batch.id_type = MONOFONIC_DATA_NONE;
        batch.count = numpart_local;
        batch.positions = positions;
        batch.position_stride = position_stride;
        batch.velocities = velocities;
        batch.velocity_stride = velocity_stride;
        batch.phase_space_densities = phase_space_densities;
        batch.phase_space_density_stride = phase_space_density_stride;
        batch.first_id = first_id;
        batch.id_increment = 1;
        batch.uniform_mass = uniform_mass;
        batch.neutrino_mass_ev = neutrino_mass_ev;
        batch.neutrino_temperature_ev = neutrino_temperature_ev;
        monofonic::emit_particle_batch(batch);
    }
};

output_plugin_creator_concrete<opengadget_memory_output_plugin> creator("opengadget_memory");
} // namespace

// This file is part of monofonIC (MUSIC2)
// A software package to generate ICs for cosmological simulations
// Copyright (C) 2020 by Oliver Hahn
// Copyright (C) 2021 by Willem Elbers
//
// monofonIC is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// monofonIC is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifdef USE_ZWINDSTROOM
#ifdef USE_CLASS

#include <cmath>
#include <cfloat>
#include <array>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <stdexcept>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__linux__)
#include <sched.h>
#endif

#include <zwindstroom.h>
#include <ClassEngine.hh>

#include <general.hh>
#include <config_file.hh>
#include <transfer_function_plugin.hh>
#include <embedded.hh>
#include <ic_generator.hh>

#include <math/interpolate.hh>


class transfer_zwindstroom_plugin : public TransferFunction_plugin
{
private:

  using TransferFunction_plugin::cosmo_params_;

  interpolated_function_1d<true, true, false> delta_c_, delta_b_, delta_n_, delta_m_, theta_c_, theta_b_, theta_n_, theta_m_;
  interpolated_function_1d<true, true, false> delta_c0_, delta_b0_, delta_n0_, delta_m0_, theta_c0_, theta_b0_, theta_n0_, theta_m0_;

  double zstart_, ztarget_, astart_, atarget_, kmax_, kmin_, h_, tnorm_;

  // asymptotic growth factor and growth rates at large k
  double Dm_asymptotic_, fm_asymptotic_, fcb_asymptotic_, vfac_asymptotic_;

  ClassParams pars_;
  std::ostream *class_input_output_ = nullptr;

  enum class_sample { sample_z0 = 0, sample_ztarget = 1, sample_zstart = 2, sample_zminus = 3, sample_zplus = 4 };
  enum transfer_column { column_dc = 0, column_tc = 1, column_db = 2, column_tb = 3,
                         column_dn = 4, column_tn = 5, column_dm = 6, column_tm = 7 };

  struct class_tables
  {
    double A_s = 0.0;
    std::vector<double> k;
    std::array<std::array<std::vector<double>, 8>, 5> transfer;
  };

  class omp_team_guard
  {
  public:
    explicit omp_team_guard(int threads)
    {
#if defined(_OPENMP)
      dynamic_ = omp_get_dynamic();
      threads_ = omp_get_max_threads();
      omp_set_dynamic(0);
      omp_set_num_threads(threads);
#else
      _unused(threads);
#endif
    }
    ~omp_team_guard()
    {
#if defined(_OPENMP)
      omp_set_num_threads(threads_);
      omp_set_dynamic(dynamic_);
#endif
    }
  private:
#if defined(_OPENMP)
    int dynamic_ = 0;
    int threads_ = 1;
#endif
  };

  template <typename T>
  void add_class_parameter(std::string parameter_name, const T parameter_value)
  {
    pars_.add(parameter_name, parameter_value);
    if (class_input_output_ != nullptr)
      *class_input_output_ << parameter_name << " = " << parameter_value << std::endl;
  }

  //! Set up class parameters from MUSIC cosmological parameters
  void prepare_ClassParameters(bool write_control_file)
  {

    pars_ = ClassParams();
    std::ofstream class_input;
    if (write_control_file) {
      class_input.open("input_class_parameters.ini", std::ios::trunc);
      if (!class_input)
        throw std::runtime_error("Could not write input_class_parameters.ini");
      class_input_output_ = &class_input;
    }

    //--- general parameters ------------------------------------------
    add_class_parameter("z_max_pk", 1e10); // start very early to allow integrating neutrinos
    add_class_parameter("P_k_max_h/Mpc", std::max(2.0,kmax_));
    add_class_parameter("output", "dTk,vTk");
    add_class_parameter("extra_metric_transfer_functions","yes");
    // add_class_parameter("lensing", "no");

    //--- choose gauge ------------------------------------------------
    // add_class_parameter("extra metric transfer functions", "yes");
    add_class_parameter("gauge", "synchronous");

    //--- cosmological parameters, densities --------------------------
    add_class_parameter("h", cosmo_params_.get("h"));

    add_class_parameter("Omega_b", cosmo_params_.get("Omega_b"));
    add_class_parameter("Omega_cdm", cosmo_params_.get("Omega_c"));
    add_class_parameter("Omega_k", cosmo_params_.get("Omega_k"));
    add_class_parameter("Omega_scf", 0.0);

    //--- dark energy -------------------------------------------------
    real_t w_0 = cosmo_params_.get("w_0");
    real_t w_a = cosmo_params_.get("w_a");
    if (w_0 == -1.0 && w_a == 0.0) {
        // Disable fluid. CLASS will close the Universe with Lambda.
        add_class_parameter("Omega_fld", 0.0);
    } else {
        // Disable Lambda. CLASS will close the Universe with a fluid.
        add_class_parameter("Omega_Lambda", 0.0);
        add_class_parameter("fluid_equation_of_state", "CLP");
        add_class_parameter("w0_fld", cosmo_params_.get("w_0"));
        add_class_parameter("wa_fld", cosmo_params_.get("w_a"));
        add_class_parameter("cs2_fld", 1.0);
    }

    //--- massive neutrinos -------------------------------------------
    add_class_parameter("N_ur", cosmo_params_.get("N_ur"));
    add_class_parameter("N_ncdm", cosmo_params_.get("N_nu_massive"));
    if( cosmo_params_.get("N_nu_massive") > 0 ){
      std::stringstream sstr;
      if( cosmo_params_.get("m_nu1") > 1e-9 ) sstr << cosmo_params_.get("m_nu1");
      if( cosmo_params_.get("m_nu2") > 1e-9 ) sstr << ", " << cosmo_params_.get("m_nu2");
      if( cosmo_params_.get("m_nu3") > 1e-9 ) sstr << ", " << cosmo_params_.get("m_nu3");
      add_class_parameter("m_ncdm", sstr.str().c_str());

      std::stringstream sstr2;
      if( cosmo_params_.get("m_nu1") > 1e-9 ) sstr2 << cosmo_params_.get("deg_nu1");
      if( cosmo_params_.get("m_nu2") > 1e-9 ) sstr2 << ", " << cosmo_params_.get("deg_nu2");
      if( cosmo_params_.get("m_nu3") > 1e-9 ) sstr2 << ", " << cosmo_params_.get("deg_nu3");
      add_class_parameter("deg_ncdm", sstr2.str().c_str());
    }

    //--- cosmological parameters, primordial -------------------------
    add_class_parameter("P_k_ini type", "analytic_Pk");

    if( cosmo_params_.get("A_s") > 0.0 ){
      add_class_parameter("A_s", cosmo_params_.get("A_s"));
    }else{
      add_class_parameter("sigma8", cosmo_params_.get("sigma_8"));
    }
    add_class_parameter("n_s", cosmo_params_.get("n_s"));
    add_class_parameter("alpha_s", cosmo_params_.get("alpha_s"));
    add_class_parameter("T_cmb", cosmo_params_.get("Tcmb"));
    add_class_parameter("YHe", cosmo_params_.get("YHe"));

    // additional parameters
    add_class_parameter("reio_parametrization", "reio_none");

    // precision parameters
    add_class_parameter("k_per_decade_for_pk", 100);
    add_class_parameter("k_per_decade_for_bao", 100);
    add_class_parameter("compute_damping_scale", "yes");
    add_class_parameter("tol_perturbations_integration", 1.e-8);
    add_class_parameter("tol_background_integration", 1e-9);

    // high precision options from cl_permille.pre:
    // precision file to be passed as input in order to achieve at least percent precision on scalar Cls
    add_class_parameter("hyper_flat_approximation_nu", 7000.);
    add_class_parameter("transfer_neglect_delta_k_S_t0", 0.17);
    add_class_parameter("transfer_neglect_delta_k_S_t1", 0.05);
    add_class_parameter("transfer_neglect_delta_k_S_t2", 0.17);
    add_class_parameter("transfer_neglect_delta_k_S_e", 0.13);
    add_class_parameter("delta_l_max", 1000);

    int class_verbosity = 0;

    add_class_parameter("background_verbose", class_verbosity);
    add_class_parameter("thermodynamics_verbose", class_verbosity);
    add_class_parameter("perturbations_verbose", class_verbosity);
    add_class_parameter("transfer_verbose", class_verbosity);
    add_class_parameter("primordial_verbose", class_verbosity);
    add_class_parameter("harmonic_verbose", class_verbosity);
    add_class_parameter("fourier_verbose", class_verbosity);
    add_class_parameter("lensing_verbose", class_verbosity);
    add_class_parameter("output_verbose", class_verbosity);

    // output parameters, only needed for the control CLASS .ini file that we output
    std::stringstream zlist;
    if (ztarget_ == zstart_)
      zlist << ztarget_ << ((ztarget_!=0.0)? ", 0.0" : "");
    else
      zlist << std::max(ztarget_, zstart_) << ", " << std::min(ztarget_, zstart_) << ", 0.0";
    add_class_parameter("z_pk", zlist.str());

    // This is an informational control file, so include the option used by
    // the second, N-body-gauge solve without modifying the synchronous input.
    if (class_input_output_ != nullptr)
      *class_input_output_ << "nbody_gauge_transfer_functions = yes" << std::endl;
    class_input_output_ = nullptr;
    class_input.close();
  }

  static void check_class_grid(const std::vector<double>& expected, const std::vector<double>& actual)
  {
    if (expected != actual)
      throw std::runtime_error("CLASS returned inconsistent wavenumber grids");
  }

  static void validate_class_tables(const class_tables& tables)
  {
    if (tables.k.empty())
      throw std::runtime_error("CLASS returned an empty transfer table");
    for (const auto& redshift : tables.transfer)
      for (const auto& values : redshift)
        if (values.size() != tables.k.size())
          throw std::runtime_error("CLASS returned transfer vectors with inconsistent sizes");
  }

  static void validate_raw_transfer(const std::vector<double>& k,
                                    const std::array<std::vector<double>, 8>& transfer)
  {
    if (k.empty())
      throw std::runtime_error("CLASS returned an empty wavenumber grid");
    for (const auto& values : transfer)
      if (values.size() != k.size())
        throw std::runtime_error("CLASS returned raw transfer vectors with inconsistent sizes");
  }

  //! Execute both CLASS solves and retain only the numerical tables needed below.
  class_tables calculate_ClassTables(const std::array<double, 5>& redshifts, int threads, bool write_control_file)
  {
    omp_team_guard omp_guard(threads);
    prepare_ClassParameters(write_control_file);
    class_tables tables;
    std::array<std::array<std::vector<double>, 8>, 5> synchronous;
    double wtime = get_wtime();

    music::ilog << "Computing transfer function via ClassEngine... (synchronous gauge)" << std::endl;
    {
      ClassEngine synchronous_engine(pars_, false);
      for (size_t iz = 0; iz < redshifts.size(); ++iz) {
        std::vector<double> k;
        synchronous_engine.getTk(redshifts[iz], k, synchronous[iz][column_dc], synchronous[iz][column_db],
                                 synchronous[iz][column_dn], synchronous[iz][column_dm], synchronous[iz][column_tc],
                                 synchronous[iz][column_tb], synchronous[iz][column_tn], synchronous[iz][column_tm]);
        validate_raw_transfer(k, synchronous[iz]);
        if (iz == 0)
          tables.k = k;
        else
          check_class_grid(tables.k, k);
      }
      // Query this only after all synchronous-gauge samples succeeded.
      tables.A_s = synchronous_engine.get_A_s();
    }

    music::ilog << "Computing transfer function via ClassEngine... (N-body gauge)" << std::endl;
    ClassParams nbody_pars = pars_;
    nbody_pars.add("nbody_gauge_transfer_functions", "yes");
    {
      ClassEngine nbody_engine(nbody_pars, false);
      for (size_t iz = 0; iz < redshifts.size(); ++iz) {
        std::vector<double> k;
        std::array<std::vector<double>, 8> nbody;
        nbody_engine.getTk(redshifts[iz], k, nbody[column_dc], nbody[column_db],
                           nbody[column_dn], nbody[column_dm], nbody[column_tc],
                           nbody[column_tb], nbody[column_tn], nbody[column_tm]);
        validate_raw_transfer(k, nbody);
        check_class_grid(tables.k, k);
        for (size_t i = 0; i < k.size(); ++i) {
          real_t theta_shift = nbody[column_tb][i] - synchronous[iz][column_tb][i];
          real_t theta_shift_approx = synchronous[iz][column_tc][i];
          nbody[column_tb][i] = -(nbody[column_tb][i] - theta_shift_approx);
          nbody[column_tn][i] = -(nbody[column_tn][i] - theta_shift_approx);
          nbody[column_tm][i] = -(nbody[column_tm][i] - theta_shift_approx);
          nbody[column_tc][i] = -theta_shift;
          nbody[column_dn][i] = fmin(-FLT_MIN, nbody[column_dn][i]);
          nbody[column_tn][i] = fmin(-FLT_MIN, nbody[column_tn][i]);
          const auto ik2 = 1.0 / (k[i] * k[i]) * cosmo_params_.get("h") * cosmo_params_.get("h");
          for (size_t column = 0; column < nbody.size(); ++column)
            tables.transfer[iz][column].push_back(-nbody[column][i] * ik2);
        }
      }
    }
    validate_class_tables(tables);
    music::ilog << "CLASS took " << get_wtime() - wtime << " s." << std::endl;
    return tables;
  }

  static void report_class_threads(int requested, int available, int using_threads)
  {
    std::ostringstream report;
    report << "Zwindstroom node CLASS execution: requested=" << requested
           << ", affinity_available=" << available << ", using=" << using_threads
           << " threads.\n";
    std::cout << report.str() << std::flush;
  }

  static int affinity_threads()
  {
#if defined(__linux__)
    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    if (sched_getaffinity(0, sizeof(cpus), &cpus) == 0) {
      int available = 0;
      for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
        if (CPU_ISSET(cpu, &cpus)) ++available;
      return std::max(1, available);
    }
#endif
    return std::numeric_limits<int>::max();
  }

#if defined(USE_MPI)
  static void broadcast_doubles(std::vector<double>& values, int root, MPI_Comm communicator)
  {
    size_t offset = 0;
    while (offset < values.size()) {
      const int count = static_cast<int>(std::min(values.size() - offset,
          static_cast<size_t>(std::numeric_limits<int>::max())));
      MPI_Bcast(values.data() + offset, count, MPI_DOUBLE, root, communicator);
      offset += static_cast<size_t>(count);
    }
  }
#endif

  class_tables distribute_ClassTables(const std::array<double, 5>& redshifts)
  {
    int local_threads = 1;
#if defined(_OPENMP)
    local_threads = omp_get_max_threads();
#endif
#if defined(USE_MPI)
    MPI_Comm node_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, CONFIG::MPI_task_rank,
                        MPI_INFO_NULL, &node_comm);
    int node_rank = 0;
    MPI_Comm_rank(node_comm, &node_rank);
    int requested = 0;
    MPI_Allreduce(&local_threads, &requested, 1, MPI_INT, MPI_SUM, node_comm);
    const int available = affinity_threads();
    const int using_threads = std::min(requested, available);
    class_tables tables;
    unsigned long long n = 0;
    std::vector<double> flat;
    std::string error;
    if (node_rank == 0) {
      report_class_threads(requested, available, using_threads);
      try {
        tables = calculate_ClassTables(redshifts, using_threads, CONFIG::MPI_task_rank == 0);
        n = static_cast<unsigned long long>(tables.k.size());
        flat.reserve(1 + 41 * tables.k.size());
        flat.push_back(tables.A_s);
        flat.insert(flat.end(), tables.k.begin(), tables.k.end());
        for (const auto& redshift : tables.transfer)
          for (const auto& values : redshift)
            flat.insert(flat.end(), values.begin(), values.end());
        if (flat.size() != 1 + 41 * tables.k.size())
          throw std::runtime_error("internal error while flattening CLASS transfer tables");
      } catch (const std::exception& exception) {
        error = exception.what();
      } catch (...) {
        error = "unknown exception while executing CLASS";
      }
    }
    int failed = error.empty() ? 0 : 1;
    MPI_Bcast(&failed, 1, MPI_INT, 0, node_comm);
    int any_failed = 0;
    MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (any_failed) {
      int failing_rank = failed ? CONFIG::MPI_task_rank : CONFIG::MPI_task_size;
      int first_failing_rank = CONFIG::MPI_task_size;
      MPI_Allreduce(&failing_rank, &first_failing_rank, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      unsigned long long length = error.size();
      MPI_Bcast(&length, 1, MPI_UNSIGNED_LONG_LONG, first_failing_rank, MPI_COMM_WORLD);
      if (CONFIG::MPI_task_rank != first_failing_rank)
        error.resize(static_cast<size_t>(length));
      if (length != 0)
        MPI_Bcast(&error[0], static_cast<int>(length), MPI_CHAR, first_failing_rank, MPI_COMM_WORLD);
      MPI_Comm_free(&node_comm);
      throw std::runtime_error("Zwindstroom CLASS failed: " + error);
    }
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG_LONG, 0, node_comm);
    if (node_rank != 0)
      flat.resize(1 + 41 * static_cast<size_t>(n));
    broadcast_doubles(flat, 0, node_comm);
    if (node_rank != 0) {
      size_t offset = 0;
      tables.A_s = flat[offset++];
      tables.k.assign(flat.begin() + offset, flat.begin() + offset + n);
      offset += static_cast<size_t>(n);
      for (auto& redshift : tables.transfer)
        for (auto& values : redshift) {
          values.assign(flat.begin() + offset, flat.begin() + offset + n);
          offset += static_cast<size_t>(n);
        }
      validate_class_tables(tables);
    }
    MPI_Comm_free(&node_comm);
    return tables;
#else
    const int requested = local_threads;
    const int available = affinity_threads();
    const int using_threads = std::min(requested, available);
    report_class_threads(requested, available, using_threads);
    return calculate_ClassTables(redshifts, using_threads, true);
#endif
  }

public:
  explicit transfer_zwindstroom_plugin(config_file &cf, const cosmology::parameters& cosmo_params)
      : TransferFunction_plugin(cf,cosmo_params)
  {
    // Before starting, throw an error if ZeroRadiation is used, because
    // that choice implies a simplified manner of backscaling
    if (pcf_->get_value_safe<bool>("cosmology", "ZeroRadiation", false))
    {
        throw std::runtime_error("Using ZeroRadiation=true for simplified backscaling, in which case zwindstroom is not needed.");
    }

    this->tf_isnormalised_ = true;

    // all cosmological parameters need to be passed through the_cosmo_calc

    ztarget_ = pcf_->get_value_safe<double>("cosmology", "ztarget", 0.0);
    atarget_ = 1.0 / (1.0 + ztarget_);
    zstart_ = pcf_->get_value<double>("setup", "zstart");
    astart_ = 1.0 / (1.0 + zstart_);

    h_ = cosmo_params_["h"];

    if (cosmo_params_["A_s"] > 0.0) {
      music::ilog << "CLASS: Using A_s=" << cosmo_params_["A_s"] << " to normalise the transfer function." << std::endl;
    }else{
      double sigma8 = cosmo_params_["sigma_8"];
      if( sigma8 < 0 ){
        throw std::runtime_error("Need to specify either A_s or sigma_8 for CLASS plugin...");
      }
      music::ilog << "CLASS: Using sigma8_ =" << sigma8<< " to normalise the transfer function." << std::endl;
    }

    //! master switch for neutrino options
    const bool bWithNeutrinos = pcf_->get_value_safe<bool>("setup", "WithNeutrinos", false );
    //! option to exclude massive neutrinos from delta_matter
    const bool bCDMBaryonMatterOnly = pcf_->get_value_safe<bool>("setup", "CDMBaryonMatterOnly", bWithNeutrinos );
    if (bCDMBaryonMatterOnly){
        music::ilog << "Using delta_matter = delta_cb." << std::endl;
    }

    // determine highest k we will need for the resolution selected
    double lbox = pcf_->get_value<double>("setup", "BoxLength");
    int nres = pcf_->get_value<double>("setup", "GridRes");
    kmax_ = std::max(20.0, 2.0 * M_PI / lbox * nres / 2 * sqrt(3) * 2.0); // 120% of spatial diagonal, or k=10h Mpc-1

    // CLASS is expensive and is therefore executed once by each node leader.
    // Every node-local rank receives plain sampled tables and builds its own
    // interpolation objects below.
    const double delta_log_a = 0.002;
    const double log_astart = log(astart_);
    const double z_min = 1.0 / exp(log_astart - delta_log_a) - 1.0;
    const double z_pls = 1.0 / exp(log_astart + delta_log_a) - 1.0;
    const std::array<double, 5> class_redshifts = {{0.0, ztarget_, zstart_, z_min, z_pls}};
    class_tables class_data = distribute_ClassTables(class_redshifts);
    double A_s_ = class_data.A_s; // either input A_s or CLASS's sigma8 normalisation

    // compute the normalisation to interface with MUSIC
    double k_p = cosmo_params["k_p"] / cosmo_params["h"];
    tnorm_ = std::sqrt(2.0 * M_PI * M_PI * A_s_ * std::pow(1.0 / k_p, cosmo_params["n_s"] - 1) / std::pow(2.0 * M_PI, 3.0));

    // Unpack the five CLASS samples.  The order is kept explicit so that the
    // fluid integration below remains identical to the former rank-local path.
    std::vector<double> k = class_data.k;
    std::vector<double> dc = class_data.transfer[sample_zstart][column_dc], tc = class_data.transfer[sample_zstart][column_tc],
                        db = class_data.transfer[sample_zstart][column_db], tb = class_data.transfer[sample_zstart][column_tb],
                        dn = class_data.transfer[sample_zstart][column_dn], tn = class_data.transfer[sample_zstart][column_tn],
                        dm = class_data.transfer[sample_zstart][column_dm], tm = class_data.transfer[sample_zstart][column_tm];
    const std::vector<double>& dc0 = class_data.transfer[sample_z0][column_dc];
    const std::vector<double>& tc0 = class_data.transfer[sample_z0][column_tc];
    const std::vector<double>& db0 = class_data.transfer[sample_z0][column_db];
    const std::vector<double>& tb0 = class_data.transfer[sample_z0][column_tb];
    const std::vector<double>& dn0 = class_data.transfer[sample_z0][column_dn];
    const std::vector<double>& tn0 = class_data.transfer[sample_z0][column_tn];
    const std::vector<double>& dm0 = class_data.transfer[sample_z0][column_dm];
    const std::vector<double>& tm0 = class_data.transfer[sample_z0][column_tm];

    delta_c0_.set_data(k, dc0);
    theta_c0_.set_data(k, tc0);
    delta_b0_.set_data(k, db0);
    theta_b0_.set_data(k, tb0);
    delta_n0_.set_data(k, dn0);
    theta_n0_.set_data(k, tn0);
    delta_m0_.set_data(k, dm0);
    theta_m0_.set_data(k, tm0);

    std::vector<double> dc_target = class_data.transfer[sample_ztarget][column_dc], tc_target = class_data.transfer[sample_ztarget][column_tc],
                        db_target = class_data.transfer[sample_ztarget][column_db], tb_target = class_data.transfer[sample_ztarget][column_tb],
                        dn_target = class_data.transfer[sample_ztarget][column_dn], tn_target = class_data.transfer[sample_ztarget][column_tn],
                        dm_target = class_data.transfer[sample_ztarget][column_dm], tm_target = class_data.transfer[sample_ztarget][column_tm];
    std::vector<double> dc_min = class_data.transfer[sample_zminus][column_dc], tc_min = class_data.transfer[sample_zminus][column_tc],
                        db_min = class_data.transfer[sample_zminus][column_db], tb_min = class_data.transfer[sample_zminus][column_tb],
                        dn_min = class_data.transfer[sample_zminus][column_dn], tn_min = class_data.transfer[sample_zminus][column_tn],
                        dm_min = class_data.transfer[sample_zminus][column_dm], tm_min = class_data.transfer[sample_zminus][column_tm];
    std::vector<double> dc_pls = class_data.transfer[sample_zplus][column_dc], tc_pls = class_data.transfer[sample_zplus][column_tc],
                        db_pls = class_data.transfer[sample_zplus][column_db], tb_pls = class_data.transfer[sample_zplus][column_tb],
                        dn_pls = class_data.transfer[sample_zplus][column_dn], tn_pls = class_data.transfer[sample_zplus][column_tn],
                        dm_pls = class_data.transfer[sample_zplus][column_dm], tm_pls = class_data.transfer[sample_zplus][column_tm];

    // wavenumbers in 1/Mpc
    kmin_ = k[0];
    kmax_ = k.back();

    music::ilog << "CLASS table contains k = " << this->get_kmin() << " to " << this->get_kmax() << " h Mpc-1." << std::endl;

    // array of neutrino masses in eV needed by zwindstroom
    const int N_nu = cosmo_params_.get("N_nu_massive");
    std::vector<double> M_nu;
    std::vector<double> deg_nu; //degeneracies
    std::vector<double> c_s_nu; //sounds speeds

    if( cosmo_params_.get("N_nu_massive") > 0 ){
        if( cosmo_params_.get("m_nu1") > 1e-9 ) {
            M_nu.push_back(cosmo_params_.get("m_nu1"));
            deg_nu.push_back(cosmo_params_.get("deg_nu1"));
            c_s_nu.push_back(0.0);
        }
        if( cosmo_params_.get("m_nu2") > 1e-9 ) {
            M_nu.push_back(cosmo_params_.get("m_nu2"));
            deg_nu.push_back(cosmo_params_.get("deg_nu2"));
            c_s_nu.push_back(0.0);
        }
        if( cosmo_params_.get("m_nu3") > 1e-9 ) {
            M_nu.push_back(cosmo_params_.get("m_nu3"));
            deg_nu.push_back(cosmo_params_.get("deg_nu3"));
            c_s_nu.push_back(0.0);
        }
    }

    // Zwindstroom structures
    struct model m;
    struct units us;
    struct physical_consts pcs;
    struct cosmology_tables tab;

    // Set up zwindstroom cosmological parameters
    m.h = cosmo_params_.get("h");
    m.Omega_b = cosmo_params_.get("Omega_b");
    m.Omega_c = cosmo_params_.get("Omega_c");
    m.Omega_k = cosmo_params_.get("Omega_k");
    m.N_ur = cosmo_params_.get("N_ur");
    m.N_nu = N_nu;
    m.M_nu = M_nu.data();
    m.deg_nu = deg_nu.data();
    m.c_s_nu = c_s_nu.data();
    m.T_nu_0 = cosmo_params_.get("Tcmb") * 0.71611; //default CLASS value
    m.T_CMB_0 = cosmo_params_.get("Tcmb");
    m.w0 = cosmo_params_.get("w_0");
    m.wa = cosmo_params_.get("w_a");
    // Does the cosmological sim use constant mass energy for the neutrinos?
    m.sim_neutrino_nonrel_masses = 1; //TODO: make parameter
    m.sim_neutrino_nonrel_Hubble = 0;

    // Set up zwindstroom unit system
    us.UnitLengthMetres = MPC_METRES; // match CLASS
    us.UnitTimeSeconds = 1e15; // can be anything
    us.UnitMassKilogram = 1.0;
    us.UnitTemperatureKelvin = 1.0;
    us.UnitCurrentAmpere = 1.0;
    set_physical_constants(&us, &pcs);

    double wtime = get_wtime();
    music::ilog << "-------------------------------------------------------------------------------" << std::endl;
    music::ilog << "Integrating cosmological tables with zwindstroom." << std::endl;

    // Integrate the cosmological tables with zwindstroom (accounting for neutrinos)
    const double tab_a_start = astart_ * 0.99;
    const double tab_a_final = atarget_ * 1.01;
    integrate_cosmology_tables(&m, &us, &pcs, &tab, tab_a_start, tab_a_final, 1000);

    // extract the present-day neutrino fraction and the baryon fraction
    const double atoday_ = 1.0;
    const double f_nu_nr_0 = get_f_nu_nr_tot_of_a(&tab, atoday_);
    const double f_b = m.Omega_b / (m.Omega_b + m.Omega_c);
    // extract the Hubble rate at a_start and normalize by H0
    const double H_start = get_H_of_a(&tab, astart_); // in zwindstroom units
    const double H_0 = get_H_of_a(&tab, atoday_); // in zwindstroom units
    const double H_units = cosmo_params_.get("H0") / H_0;

    music::ilog << "Integrating fluid equations with zwindstroom." << std::endl;

    // prepare fluid equation integration
    const double tol = 1e-12;
    const double hstart = 1e-12;
    prepare_fluid_integrator(&m, &us, &pcs, &tab, tol, hstart);

    // if we have no neutrinos, zero out the neutrino transfer functions
    if (N_nu == 0) {
        for (size_t i = 0; i < k.size(); ++i) {
            dn[i] = 0.;
            tn[i] = 0.;
            dn_min[i] = 0.;
            tn_min[i] = 0.;
            dn_pls[i] = 0.;
            tn_pls[i] = 0.;
            dn_target[i] = 0.;
            tn_target[i] = 0.;
        }
    }

    // compute the scale-dependent logarithmic growth rates at z=z_start
    std::vector<double> gc, gb, gn, gcb, gm;
    for (size_t i = 0; i < k.size(); ++i)
    {
      // compute weighted averages
      double dcb_pls = f_b * db_pls[i] + (1.0 - f_b) * dc_pls[i];
      double dcb_min = f_b * db_min[i] + (1.0 - f_b) * dc_min[i];
      double dcb = f_b * db[i] + (1.0 - f_b) * dc[i];
      double dm_pls = f_nu_nr_0 * dn_pls[i] + (1.0 - f_nu_nr_0) * dcb_pls;
      double dm_min = f_nu_nr_0 * dn_min[i] + (1.0 - f_nu_nr_0) * dcb_min;
      double dm = f_nu_nr_0 * dn[i] + (1.0 - f_nu_nr_0) * dcb;

      // store the values for this row
      gc.push_back((dc_pls[i] - dc_min[i]) / (2.0 * delta_log_a) / dc[i]);
      gb.push_back((db_pls[i] - db_min[i]) / (2.0 * delta_log_a) / db[i]);
      gcb.push_back((dcb_pls - dcb_min) / (2.0 * delta_log_a) / dcb);
      gm.push_back((dm_pls - dm_min) / (2.0 * delta_log_a) / dm);
      if (N_nu > 0) {
          gn.push_back((dn_pls[i] - dn_min[i]) / (2.0 * delta_log_a) / dn[i]);
      } else {
          gn.push_back(0.);
      }
    }

    // compute the scale-dependent growth factors in the 3-fluid approximation
    std::vector<double> Dc, Db, Dn;
    for (size_t i = 0; i < k.size(); ++i)
    {
        // CLASSengine only gives us the density perturbations for one
        // neutrino species. If we have more than one species, the best
        // we can do is assign the same initial growth rate to all species

        // create arrays for the neutrino densities
        std::vector<double> gfac_delta_n;
        std::vector<double> gfac_gn;
        std::vector<double> gfac_Dn;
        for (int j = 0; j < N_nu; j++) {
            gfac_delta_n.push_back(dn[i]);
            gfac_gn.push_back(gn[i]);
            gfac_Dn.push_back(0.); // output goes here
        }

        // initialise the input data for the fluid equations
        struct growth_factors gfac;
        gfac.k = k[i]; // in 1/Mpc -- like CLASS, zwindstroom does not use h-units
        gfac.delta_c = dc[i];
        gfac.delta_b = db[i];
        gfac.delta_n = gfac_delta_n.data();
        gfac.gc = gc[i];
        gfac.gb = gb[i];
        gfac.gn = gfac_gn.data();
        gfac.Dc = 0.;
        gfac.Db = 0.;
        gfac.Dn = gfac_Dn.data();

        integrate_fluid_equations(&m, &us, &pcs, &tab, &gfac, astart_, atarget_);

        // store the relative growth factors between the target and starting redshifts
        Dc.push_back(gfac.Dc);
        Db.push_back(gfac.Db);
        if (N_nu > 0) {
            Dn.push_back(gfac.Dn[0]); // read off the first species
        } else {
            Dn.push_back(0.0);
        }
    }

    // done with fluid integration
    free_fluid_integrator();

    wtime = get_wtime() - wtime;
    music::ilog << "Zwindstroom took " << wtime << " s." << std::endl;

    // determine the asymptotic growth rate by averaging over small scales
    // modes (k > 1/Mpc)
    double gm_sum = 0.;
    double gcb_sum = 0.;
    int count = 0;
    for (size_t i = 0; i < k.size(); ++i)
    {
        if (k[i] < 1.0) continue; //ignore large scales

        gm_sum += gm[i];
        gcb_sum += gcb[i];
        count++;
    }

    fm_asymptotic_ = gm_sum / count;
    fcb_asymptotic_ = gcb_sum / count;

    vfac_asymptotic_ = astart_ * H_start * H_units / cosmo_params_.get("h");
    if (bCDMBaryonMatterOnly){
        vfac_asymptotic_ *= fcb_asymptotic_;
    } else {
        vfac_asymptotic_ *= fm_asymptotic_;
    }

    // The growth factor ratio as computed by monofonIC. This isn't correct
    // for massive neutrino cosmologies, but we scale forward by this factor
    // here and then scale back by the same factor in ic_generator.cc. Hence,
    // it drops out of the equation.
    double D_scale_forward = cosmo_params_.get("dplus_start") / cosmo_params_.get("dplus_target");

    // now scale forward with the asymptotic growth factor, as assumed in the ic generator
    for (size_t i = 0; i < k.size(); ++i)
    {

        // we want to use the linear theory growth rate at z = z_start,
        // so we need to keep the ratio of theta_i / delta_i.
        real_t tc_over_dc = tc[i] / dc[i];
        real_t tb_over_db = tb[i] / db[i];
        real_t tn_over_dn;
        if (N_nu > 0) {
            tn_over_dn = tn[i] / dn[i];
        } else {
            tn_over_dn = 0.;
        }

        // scale back the density transfer functions from the target redshift
        // to the starting redshift using the scale-dependent growth factors
        dc[i] = dc_target[i] * Dc[i];
        db[i] = db_target[i] * Db[i];
        dn[i] = dn_target[i] * Dn[i];

        // scale the density transfer functions forward by the monofonIC factor
        dc[i] /= D_scale_forward;
        db[i] /= D_scale_forward;
        dn[i] /= D_scale_forward;

        // set the velocity transfer functions accordingly
        tc[i] = dc[i] * tc_over_dc;
        tb[i] = db[i] * tb_over_db;
        tn[i] = dn[i] * tn_over_dn;

        // mass-weighted cdm+baryon density and velocity transfer functions
        double dcb, tcb;

        // compute the mass-weighted average
        dcb = f_b * db[i] + (1.0 - f_b) * dc[i];
        tcb = f_b * tb[i] + (1.0 - f_b) * tc[i];

        if (bCDMBaryonMatterOnly) {
            dm[i] = dcb;
            tm[i] = tcb;
        } else {
            dm[i] = f_nu_nr_0 * dn[i] + (1.0 - f_nu_nr_0) * dcb;
            tm[i] = f_nu_nr_0 * tn[i] + (1.0 - f_nu_nr_0) * tcb;
        }

        // the (baryon - cdm) difference evaluated at the target redshift
        double dbc_target = db_target[i] - dc_target[i];
        double tbc_target = -(tb_target[i] - tc_target[i]); //opposite sign velocity transfers

        // add back the (baryon - cdm) difference evaluated at the target redshift
        dc[i] = dcb - f_b * dbc_target;
        tc[i] = tcb - f_b * tbc_target;
        db[i] = dcb + (1.0 - f_b) * dbc_target;
        tb[i] = tcb + (1.0 - f_b) * tbc_target;

        // monofonic requires positive transfer functions here, so we need
        // to truncate the neutrino transfer functions when errors at large
        // k send delta_ncdm or theta_ncdm negative
        dn[i] = fmax(FLT_MIN, dn[i]);
        tn[i] = fmax(FLT_MIN, tn[i]);
    }

    // Store the rescaled transfer function data
    delta_c_.set_data(k, dc);
    delta_b_.set_data(k, db);
    delta_n_.set_data(k, dn);
    delta_m_.set_data(k, dm);
    theta_c_.set_data(k, tc);
    theta_b_.set_data(k, tb);
    theta_n_.set_data(k, tn);
    theta_m_.set_data(k, tm);

    music::ilog << "Asymptotic fm_start = " << fm_asymptotic_ << std::endl;
    music::ilog << "Asymptotic fcb_start = " << fcb_asymptotic_ << std::endl;
    music::ilog << "Asymptotic vfac = " << vfac_asymptotic_ << " km/s/Mpc at a_start" << std::endl;

    // Retain the internally calculated background for an embedding simulation
    // code.  This avoids accepting a separately supplied H(z) or w(z) table.
    monofonic::set_background_table(tab.avec, tab.Hvec, tab.size, H_units);

    // export a table with Hubble rates for cosmological sims that require this
    std::string fname_hubble = "input_hubble.txt";
    if (CONFIG::MPI_task_rank == 0)
    {
        std::ofstream ofs(fname_hubble.c_str());
        std::stringstream ss;
        ofs << "# " << std::setw(18) << "z"
                    << std::setw(20) << "H(z) [km/s/Mpc]"
                    << std::endl;
        for (int i = 0; i < tab.size; i++) {
            double z = 1.0 / tab.avec[i] - 1.0;
            double Hz = tab.Hvec[i] * H_units;

            // Output the final line at z = 0
            if (z < 0.0) {
                z = 0.0;
                Hz = cosmo_params_.get("H0");
            }

            ofs << std::setw(20) << std::setprecision(10) << z
                << std::setw(20) << std::setprecision(10) << Hz
                << std::endl;

            if (z <= 0) break;
        }
    }
    if (N_nu > 0) {
        music::wlog << " Make sure that your sim code can handle massive neutrinos in its background FLRW model." << std::endl;
    }
    music::ilog << "Wrote Hubble rate table to file \'" << fname_hubble << "\'" << std::endl;

    // clean up zwindstroom
    free_cosmology_tables(&tab);

    tf_with_asymptotic_growth_factors_ = true;
    tf_distinct_ = true;
    tf_withvel_ = true;
    tf_withtotal0_ = true;
  }

  ~transfer_zwindstroom_plugin()
  {
  }

  inline double compute(double k, tf_type type) const
  {
    k *= h_;

    if (k < kmin_ || k > kmax_)
    {
      return 0.0;
    }

    real_t val(0.0);
    switch (type)
    {
      // values at ztarget:
    case delta_matter:
      val = delta_m_(k); break;
    case delta_cdm:
      val = delta_c_(k); break;
    case delta_baryon:
      val = delta_b_(k); break;
    case theta_matter:
      val = theta_m_(k); break;
    case theta_cdm:
      val = theta_c_(k); break;
    case theta_baryon:
      val = theta_b_(k); break;
    case delta_bc:
      val = delta_b_(k)-delta_c_(k); break;
    case theta_bc:
      val = theta_b_(k)-theta_c_(k); break;
    case delta_nu:
      val = delta_n_(k); break;
    case theta_nu:
      val = theta_n_(k); break;

      // values at zstart:
    case delta_matter0:
      val = delta_m0_(k); break;
    case delta_cdm0:
      val = delta_c0_(k); break;
    case delta_baryon0:
      val = delta_b0_(k); break;
    case theta_matter0:
      val = theta_m0_(k); break;
    case theta_cdm0:
      val = theta_c0_(k); break;
    case theta_baryon0:
      val = theta_b0_(k); break;
    case delta_nu0:
      val = delta_n0_(k); break;
    case theta_nu0:
      val = theta_n0_(k); break;

    default:
      throw std::runtime_error("Invalid type requested in transfer function evaluation");
    }
    return val * tnorm_ * cosmology::compute_running_factor(&cosmo_params_, k);
  }

  inline double get_kmin(void) const { return kmin_ / h_; }
  inline double get_kmax(void) const { return kmax_ / h_; }
  inline double get_vfac_asymptotic(void) const { return vfac_asymptotic_; }
};

namespace
{
TransferFunction_plugin_creator_concrete<transfer_zwindstroom_plugin> creator("zwindstroom");
}

#endif // USE_CLASS
#endif // USE_ZWINDSTROOM

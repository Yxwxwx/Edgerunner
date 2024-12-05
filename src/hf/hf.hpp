#pragma once
#ifndef __HF_HPP__
#define __HF_HPP__

#define EIGEN_USE_THREADS
#include "gto/gto.hpp"
#include "integral/integral.hpp"
#include "linalg/einsum.hpp"
#include <Eigen/Dense>
#include <chrono>

namespace HF {
class rhf {
private:
    Integral::Integral int_eng;
    double _energy_tot { 0.0 };
    double _elec_energy { 0.0 };
    double _nuc_rep_energy;
    int _max_iter;
    double _conv_tol;
    int nao;
    int nocc;
    bool _direct;
    bool _DIIS;
    std::vector<std::tuple<int, int, int, int>> _ijkl;
    int _ijkl_size;

    Eigen::MatrixXd _S;
    Eigen::MatrixXd _A;
    Eigen::MatrixXd _H;
    Eigen::MatrixXd _F;
    Eigen::MatrixXd _D;
    Eigen::Tensor<double, 4> _I;
    Eigen::MatrixXd _C;
    Eigen::VectorXd _orb_energy;

    void compute_fock_matrix();
    void compute_fock_matrix_direct();
    void compute_density_matrix();
    void compute_init_guess();
    double compute_energy_elec();
    double compute_energy_tot();
    Eigen::MatrixXd compute_diis_error();
    Eigen::MatrixXd apply_diis();

    double degeneracy(const int s1, const int s2, const int s3, const int s4);
    Eigen::MatrixXd matrix_sqrt_inverse(const Eigen::MatrixXd& mat);

    // DIIS
    std::vector<Eigen::MatrixXd> diis_fock_list;
    std::vector<Eigen::MatrixXd> diis_error_list;
    int diis_max_space = 6;

public:
    rhf(GTO::Mol& mol, int max_iter = 100, double conv_tol = 1e-7, bool direct = true, bool DIIS = true);
    ~rhf() = default;

    const Eigen::MatrixXd& get_fock_matrix() const;
    const Eigen::MatrixXd& get_density_matrix() const;
    const double get_energy_tot() const;
    const Eigen::MatrixXd& get_coeff() const;
    const Eigen::VectorXd& get_orb_energy() const;
    bool kernel();
};

} // namespace HF

#endif
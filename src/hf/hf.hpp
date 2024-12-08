#pragma once
#ifndef __HF_HPP__
#define __HF_HPP__

#ifdef __USE_MKL__
#define EIGEN_USE_MKL_ALL
#endif

#include "gto/gto.hpp"
#include "integral/integral.hpp"
#include "linalg/einsum.hpp"
#include <Eigen/Dense>
#include <chrono>

namespace HF {
class RHF {

public:
    RHF(GTO::Mol& mol, int max_iter = 100, double conv_tol = 1e-7);
    ~RHF() = default;

    const Eigen::MatrixXd& get_fock_matrix() const; // An interface to get fock matrix
    const Eigen::MatrixXd& get_density_matrix() const; // An interface to get density matrix
    const Eigen::MatrixXd& get_int1e() const; // An interface to get 1e integral
    const Eigen::Tensor<double, 4>& get_int2e() const; // An interface to get 2e integral
    const double get_energy_tot() const; // An interface to get total energy
    const Eigen::MatrixXd& get_coeff() const; // An interface to get atomic orbital coefficient
    const Eigen::VectorXd& get_orb_energy() const; // An interface to get orbital energy
    const int get_nao() const; // An interface to get number of atomic orbitals
    const int get_nocc() const; // An interface to get number of occupied orbitals
    bool kernel(bool direct = true, bool DIIS = true, int diis_max_space = 6, int diis_start = 2); // The main function to do SCF

private:
    Integral::Integral int_eng;
    double _energy_tot { 0.0 };
    double _elec_energy { 0.0 };
    double _nuc_rep_energy;
    int _max_iter;
    double _conv_tol;
    int nao;
    int nocc;
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

    void compute_fock_matrix(); // calculate fock matrix with full integral
    void compute_fock_matrix_direct(); // calculate fock matrix without integral (directly)
    void compute_density_matrix(); // calculate density matrix
    void compute_init_guess(); // calculate initial guess(use 1e guess)
    double compute_energy_elec(); // calculate electronic energy
    double compute_energy_tot(); // calculate total energy(elec + nuc)
    Eigen::MatrixXd compute_diis_error(); // calculate diis error
    Eigen::MatrixXd apply_diis(); // apply diis

    double degeneracy(const int s1, const int s2, const int s3, const int s4); // compute degeneracy used in S8 symmetry
    Eigen::MatrixXd matrix_sqrt_inverse(const Eigen::MatrixXd& mat); // A = S^(-1/2)

    // DIIS
    std::vector<Eigen::MatrixXd> diis_fock_list;
    std::vector<Eigen::MatrixXd> diis_error_list;
};

} // namespace HF

#endif
#pragma once
#ifndef __HF_HPP__
#define __HF_HPP__

#include "gto/gto.hpp"
#include "integral/integral.hpp"
#include "linalg/einsum.hpp"
#define EIGEN_USE_THREADS
#include <Eigen/Dense>

namespace HF {
class rhf {
private:
    Integral::Integral int_eng;
    double energy_tot { 0.0 };
    double elec_energy { 0.0 };
    double nuc_rep_energy;
    int _max_iter;
    double _conv_tol;
    int nao;
    int nocc;

    Eigen::MatrixXd _S;
    Eigen::MatrixXd _H;
    Eigen::MatrixXd _F;
    Eigen::MatrixXd _D;

    void compute_fock_matrix();
    void compute_density_matrix();
    void compute_init_guess();
    void compute_energy_elec();
    void compute_energy_tot();

public:
    rhf(GTO::Mol& mol, int max_iter = 100, double conv_tol = 1e-7);
    ~rhf() = default;

    const Eigen::MatrixXd& get_fock_matrix() const;
    const Eigen::MatrixXd& get_density_matrix() const;
    const double get_energy_tot() const;

    void kernel();
};

} // namespace HF

#endif
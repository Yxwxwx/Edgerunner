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
    std::vector<std::tuple<int, int, int, int>> _ijkl;
    int _ijkl_size;

    Eigen::MatrixXd _S;
    Eigen::MatrixXd _H;
    Eigen::MatrixXd _F;
    Eigen::MatrixXd _D;
    Eigen::Tensor<double, 4> _I;

    void compute_fock_matrix();
    void compute_fock_matrix_direct();
    void compute_density_matrix();
    void compute_init_guess();
    double compute_energy_elec();
    double compute_energy_tot();

    double degeneracy(const int i, const int j, const int k, const int l)
    {
        auto s1 = i, s2 = j, s3 = k, s4 = l;
        auto s12_deg = (s1 == s2) ? 1.0 : 2.0;
        auto s34_deg = (s3 == s4) ? 1.0 : 2.0;
        auto s12_34_deg = (s1 == s3) ? (s2 == s4 ? 1.0 : 2.0) : 2.0;
        return s12_deg * s34_deg * s12_34_deg;
    }

public:
    rhf(GTO::Mol& mol, int max_iter = 100, double conv_tol = 1e-7, bool direct = true);
    ~rhf() = default;

    const Eigen::MatrixXd& get_fock_matrix() const;
    const Eigen::MatrixXd& get_density_matrix() const;
    const double get_energy_tot() const;

    bool kernel();
};

} // namespace HF

#endif
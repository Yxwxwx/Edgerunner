#pragma once
#ifndef __HF_HPP__
#define __HF_HPP__

#include "gto/gto.hpp"
#include "integral/integral.hpp"
#include "linalg/einsum.hpp"
#include <Eigen/Dense>

namespace HF {
class rhf {
private:
    Integral::Integral int_eng;
    double euc_rep_energy;
    int _max_iter;
    double _conv_tol;

    Eigen::MatrixXd _F;
    Eigen::MatrixXd _D;

    void compute_fock_matrix();
    void compute_density_matrix();

public:
    rhf(GTO::Mol& mol, int max_iter = 100, double conv_tol = 1e-7);
    ~rhf() = default;

    const Eigen::MatrixXd& get_fock_matrix() const;
    const Eigen::MatrixXd& get_density_matrix() const;

    void kernel();
};

} // namespace HF

#endif
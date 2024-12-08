#pragma once
#ifndef __MP2_HPP__
#define __MP2_HPP__

#ifdef USE_MK
#define EIGEN_USE_MKL_ALL
#endif

#include "gto/gto.hpp"
#include "hf/hf.hpp"
#include "integral/integral.hpp"
#include <Eigen/Dense>

namespace MP2 {
class MP2 {
public:
    MP2(GTO::Mol& mol, int frozen = 0);
    void kernel();

private:
    HF::RHF hf_eng;
    int nfrozen;
    int nao;
    int nocc;
    int nvir;
    double energy_mp2 { 0.0 };
    double total_energy { 0.0 };
    Eigen::MatrixXd _C;
    Eigen::VectorXd _orb_energy;
    Eigen::MatrixXd _H_mo;
    Eigen::Tensor<double, 4> _I_mo;

    void ao_to_mo();
};

} // namespace MP2

#endif
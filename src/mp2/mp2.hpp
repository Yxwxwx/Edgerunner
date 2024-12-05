#pragma once
#ifndef __MP2_HPP__
#define __MP2_HPP__

#define EIGEN_USE_THREADS
#include "gto/gto.hpp"
#include "integral/integral.hpp"
#include "hf/hf.hpp"
#include <Eigen/Dense>

namespace MP2 {
class MP2 {
public:
    MP2(GTO::Mol& mol);
    


private:
    HF::rhf hf_eng;
    Eigen::MatrixXd _C;
    Eigen::VectorXd _orb_energy;
    Eigen::Tensor<double, 4> _I_ao;
    Eigen::Tensor<double, 4> _I_mo;

    void ao_to_mo();

    void calc_mp2();

};

} // namespace MP2

#endif
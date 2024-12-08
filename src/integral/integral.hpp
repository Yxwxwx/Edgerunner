#pragma once
#ifndef INTEGRAL_HPP
#define INTEGRAL_HPP

#ifdef USE_MK
#define EIGEN_USE_MKL_ALL
#endif

#include "gto/gto.hpp"
#include <Eigen/Dense>
#include <chrono>
#include <omp.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

extern "C" {
#include <cint.h>
int cint1e_ovlp_cart(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_nuc_cart(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_kin_cart(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_grids_cart(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env); // by SunXinyu for QMMM
int cint1e_ovlp_sph(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_nuc_sph(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_kin_sph(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_grids_sph(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env); // by SunXinyu for QMMM

// first-order gradient
int cint1e_ipovlp_cart(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_ovlpip_cart(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_ipkin_cart(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_ipnuc_cart(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_iprinv_cart(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_drinv_cart(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint2e_ip1_cart(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env, CINTOpt* opt);
int cint2e_ip2_cart(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env, CINTOpt* opt);
int cint2e_ip1_optimizer(CINTOpt** opt, int* atm,
    int natm, int* bas, int nbas, double* env);
int cint2e_ip2_optimizer(CINTOpt** opt, int* atm,
    int natm, int* bas, int nbas, double* env);
int cint1e_ipovlp_sph(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_ovlpip_sph(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_ipkin_sph(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_ipnuc_sph(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_iprinv_sph(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_drinv_sph(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint2e_ip1_sph(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env, CINTOpt* opt);
int cint2e_ip2_sph(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env, CINTOpt* opt);
// second-order gradient
int cint1e_ipipovlp_cart(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_ipovlpip_cart(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_ipipnuc_cart(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_ipnucip_cart(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_ipiprinv_cart(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_iprinvip_cart(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_ipipkin_cart(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_ipkinip_cart(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint2e_ipip1_cart(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env, CINTOpt* opt);
int cint2e_ip1ip2_cart(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env, CINTOpt* opt);
int cint2e_ipvip1_cart(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env, CINTOpt* opt);
int cint2e_ipip1_optimizer(CINTOpt** opt, int* atm,
    int natm, int* bas, int nbas, double* env);
int cint2e_ip1ip2_optimizer(CINTOpt** opt, int* atm,
    int natm, int* bas, int nbas, double* env);
int cint2e_ipvip1_optimizer(CINTOpt** opt, int* atm,
    int natm, int* bas, int nbas, double* env);

int cint1e_ipipovlp_sph(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_ipovlpip_sph(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_ipipnuc_sph(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_ipnucip_sph(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_ipiprinv_sph(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_iprinvip_sph(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_ipipkin_sph(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint1e_ipkinip_sph(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env);
int cint2e_ipip1_sph(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env, CINTOpt* opt);
int cint2e_ip1ip2_sph(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env, CINTOpt* opt);
int cint2e_ipvip1_sph(double* buf, int* shls,
    int* atm, int natm, int* bas, int nbas, double* env, CINTOpt* opt);
}

#endif

namespace Integral {
class Integral {
public:
    Integral(GTO::Mol& mol);
    ~Integral() { CINTdel_optimizer(&opt); }

    const Eigen::MatrixXd& get_overlap(); // An interface for overlap matrix
    const Eigen::MatrixXd& get_kinetic(); // An interface for kinetic energy matrix
    const Eigen::MatrixXd& get_nuc(); // An interface for nuclear attraction matrix
    const Eigen::MatrixXd& get_H(); // An interface for H = T + V
    const Eigen::Tensor<double, 4>& get_int2e(); // An interface for 2-electron integral

    int get_nao() const; // An interface for number of atomic orbitals
    auto calc_int() -> void; // A interface for calc_int1e and calc_int2e
    auto calc_int1e() -> void; // A interface for calc_int1e_sph
    auto calc_int2e() -> void; // A interface for calc_int2e_sph

    auto get_ijkl() -> std::vector<std::tuple<int, int, int, int>>; // A interface for S8 symmetry
    auto get_dim(int i, int j, int k, int l) -> std::tuple<int, int, int, int>; // A interface for the int2e dimension of (ij|kl)
    auto get_offset(int i, int j, int k, int l) -> std::tuple<int, int, int, int>; // A interface for the int2e offset of (ij|kl)
    auto calc_int2e_shell(std::tuple<int, int, int, int> ijkl, std::tuple<int, int, int, int> dim) -> Eigen::Tensor<double, 4>; // A interface for calc (ij|kl)

private:
    std::vector<int> _atm; // used in Libcint
    std::vector<int> _bas; // used in Libcint
    std::vector<double> _env; // used in Libcint
    std::vector<int> _shls; // used in Libcint
    int _natm; // used in Libcint
    int _nbas; // used in Libcint
    int nao { 0 }; // number of atomic orbitals
    CINTOpt* opt = NULL; // used in Libcint

    Eigen::MatrixXd _S; // overlap matrix
    Eigen::MatrixXd _T; // kinetic energy matrix
    Eigen::MatrixXd _V; // nuclear attraction matrix
    Eigen::MatrixXd _H; // H = T + V
    Eigen::Tensor<double, 4> _I; // 2-electron integral

    std::vector<std::tuple<int, int>> _ij; // A wrapper for int2e symmetry
    std::vector<std::tuple<int, int, int, int>> _ijkl; // A wrapper for S8 symmetry
    int _ij_size; // the number of ij
    int _ijkl_size; // the number of ijkl

    void gen_nao(); // generate number of atomic orbitals
    void gen_s8(); // generate S8 symmetry
    void gen_hermit(); // generate hermit symmetry
};
} // namaspace Integral

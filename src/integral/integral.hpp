#pragma once
#ifndef INTEGRAL_HPP
#define INTEGRAL_HPP

#include "gto/gto.hpp"

#include <Eigen/Dense>
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
    Integral(GTO::Mol mol);
    ~Integral() = default;

    const Eigen::MatrixXd& get_overlap();
    const Eigen::MatrixXd& get_kinetic();
    const Eigen::MatrixXd& get_nuc();
    const Eigen::MatrixXd& get_H();
    const Eigen::Tensor<double, 4>& get_int2e();

    int get_nao() const;

private:
    std::vector<int> _atm;
    std::vector<int> _bas;
    std::vector<double> _env;
    std::vector<int> _shls;
    int _natm;
    int _nbas;
    int nao { 0 };

    Eigen::MatrixXd _S;
    Eigen::MatrixXd _T;
    Eigen::MatrixXd _V;
    Eigen::MatrixXd _H;
    Eigen::Tensor<double, 4> _I;

    void gen_nao();
};
} // namaspace Integral
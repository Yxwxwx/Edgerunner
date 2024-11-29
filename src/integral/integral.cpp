#include "integral.hpp"

namespace Integral {
Integral::Integral(GTO::Mol mol)
{
    auto tmp = mol.get_cint_info();
    _atm = tmp.atm;
    _bas = tmp.bas;
    _env = tmp.env;
    _natm = tmp.natm;
    _nbas = tmp.nbas;

    gen_nao();
}

void Integral::gen_nao()
{
    for (auto i = 0; i < _nbas; i++) {
        nao += (_bas[i * BAS_SLOTS + ANG_OF] * 2 + 1) * _bas[i * BAS_SLOTS + NCTR_OF];
    }
}

const Eigen::MatrixXd& Integral::get_overlap()
{
    _S.resize(nao, nao);
    _S.setZero();
    return _S;
}
const Eigen::MatrixXd& Integral::get_kinetic()
{
    return _T;
}
const Eigen::MatrixXd& Integral::get_nuc()
{
    return _V;
}
const Eigen::MatrixXd& Integral::get_H()
{
    return _H;
}
const Eigen::Tensor<double, 4>& Integral::get_int2e()
{
    return _I;
}

int Integral::get_nao() const { return nao; }
} // namespace Integral

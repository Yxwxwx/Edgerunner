#include "hf.hpp"

namespace HF {
rhf::rhf(GTO::Mol& mol, int max_iter, double conv_tol)
    : int_eng(mol), _max_iter(max_iter), _conv_tol(conv_tol)
{
    int_eng.calc_int();
}

void rhf::compute_fock_matrix()
{
    auto eri = int_eng.get_int2e();
    auto den = YXTensor::matrix_to_tensor(_D);

    auto J = YXTensor::einsum<2, double, 4, 2, 2>("ijkl, kl->ij", eri, den);
    auto K = YXTensor::einsum<2, double, 4, 2, 2>("ikjl, kl->ij", eri, den);
}
void rhf::compute_density_matrix()
{
}

const Eigen::MatrixXd& rhf::get_fock_matrix() const
{
    return _F;
}

const Eigen::MatrixXd& rhf::get_density_matrix() const
{
    return _D;
}

} // namespace HF
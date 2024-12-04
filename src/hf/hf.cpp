#include "hf.hpp"

namespace HF {
rhf::rhf(GTO::Mol& mol, int max_iter, double conv_tol)
    : int_eng(mol), _max_iter(max_iter), _conv_tol(conv_tol)
{
    int_eng.calc_int();
    nao = int_eng.get_nao();
    nocc = mol.get_nelec()[2] / 2;
    _nuc_rep_energy = mol.get_nuc_rep();
    _S = int_eng.get_overlap();
    _H = int_eng.get_H();
}

void rhf::compute_fock_matrix()
{
    const auto eri = int_eng.get_int2e();

    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(nao, nao);
#pragma omp parallel
    {
        Eigen::MatrixXd local_J = Eigen::MatrixXd::Zero(nao, nao);
#pragma omp for collapse(2)
        for (int i = 0; i < nao; ++i) {
            for (int j = 0; j < nao; ++j) {
                for (int k = 0; k < nao; ++k) {
                    for (int l = 0; l < nao; ++l) {
                        local_J(i, j) += eri(i, j, k, l) * _D(k, l);
                    }
                }
            }
        }

#pragma omp critical
        J += local_J;
    }

    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(nao, nao);
#pragma omp parallel
    {
        Eigen::MatrixXd local_K = Eigen::MatrixXd::Zero(nao, nao);
#pragma omp for collapse(2)
        for (int i = 0; i < nao; ++i) {
            for (int j = 0; j < nao; ++j) {
                for (int k = 0; k < nao; ++k) {
                    for (int l = 0; l < nao; ++l) {
                        local_K(i, j) += eri(i, k, j, l) * _D(k, l);
                    }
                }
            }
        }

#pragma omp critical
        K += local_K;
    }

    _F = _H + 2 * J - K;
    // const auto den = YXTensor::matrix_to_tensor(_D);
    //  auto J_ = YXTensor::einsum<2, double, 4, 2, 2>("ijkl,kl->ij", eri, den);
    //  auto K_ = YXTensor::einsum<2, double, 4, 2, 2>("ikjl,kl->ij", eri, den);
    //  _F = _H + 2 * YXTensor::tensor_to_matrix(J_) - YXTensor::tensor_to_matrix(K_);
}
void rhf::compute_density_matrix()
{
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(_F, _S);
    Eigen::MatrixXd C = solver.eigenvectors();
    Eigen::MatrixXd C_occ = C.leftCols(nocc);
    _D = C_occ * C_occ.transpose();
}

void rhf::compute_init_guess()
{
    _F = _H;
    compute_density_matrix();
}

double rhf::compute_energy_elec()
{
    auto elec_e { 0.0 };
    for (int i = 0; i < nao; ++i) {
        for (int j = 0; j < nao; ++j) {
            elec_e += (_H(i, j) + _F(i, j)) * _D(i, j);
        }
    }
    return elec_e;
}

double rhf::compute_energy_tot()
{
    return compute_energy_elec() + _nuc_rep_energy;
}

void rhf::kernel()
{
    double old_energy = 0.0;
    double delta_energy = 0.0;
    compute_init_guess();

    for (int i = 1; i <= _max_iter; ++i) {
        compute_fock_matrix();
        auto hf_energy = compute_energy_tot();
        compute_density_matrix();
        delta_energy = std::abs(hf_energy - old_energy);
        old_energy = hf_energy;
        std::cout << std::format("Iteration: {:>3} | Energy: {:>12.6f} | Difference: {:>12.6e}\n",
            i, hf_energy, delta_energy);

        if (delta_energy < _conv_tol) {
            std::cout << "Convergence achieved in " << i << " iterations" << std::endl;
            _energy_tot = hf_energy;
            break;
        }
    }
}

const Eigen::MatrixXd& rhf::get_fock_matrix() const
{
    return _F;
}

const Eigen::MatrixXd& rhf::get_density_matrix() const
{
    return _D;
}
const double rhf::get_energy_tot() const
{
    return _energy_tot;
}
} // namespace HF
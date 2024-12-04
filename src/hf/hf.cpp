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
    _I = int_eng.get_int2e();
}

void rhf::compute_fock_matrix()
{
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(nao, nao);
#pragma omp parallel
    {
        Eigen::MatrixXd local_G = Eigen::MatrixXd::Zero(nao, nao);
#pragma omp for collapse(2)
        for (int l = 0; l < nao; ++l) {
            for (int k = 0; k < nao; ++k) {
                for (int j = 0; j < nao; ++j) {
                    for (int i = 0; i < nao; ++i) {
                        local_G(i, j) += (2 * _I(i, j, k, l) - _I(i, k, j, l)) * _D(k, l);
                    }
                }
            }
        }

#pragma omp critical
        G += local_G;
    }

    _F = _H + G;
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
    elec_e = _D.cwiseProduct(_H + _F).sum();
    return elec_e;
}

double rhf::compute_energy_tot()
{
    return compute_energy_elec() + _nuc_rep_energy;
}

bool rhf::kernel()
{
    auto start = std::chrono::steady_clock::now();
    double old_energy = 0.0;
    double delta_energy = 0.0;
    double delta_D = 0.0;
    Eigen::MatrixXd old_D = Eigen::MatrixXd::Zero(nao, nao);
    compute_init_guess();

    for (int i = 1; i <= _max_iter; ++i) {
        compute_fock_matrix();
        auto hf_energy = compute_energy_tot();
        compute_density_matrix();
        delta_energy = std::abs(hf_energy - old_energy);
        Eigen::MatrixXd squared_diff = (_D - old_D).array().square();
        delta_D = std::sqrt(squared_diff.sum() / squared_diff.size());
        old_energy = hf_energy;
        old_D = _D;
        std::cout << std::format("Iteration: {:>3} | Energy: {:>12.10f} | dE: {:>12.6e} | dD: {:>12.6e}\n",
            i, hf_energy, delta_energy, delta_D);

        if (delta_energy < _conv_tol && delta_D < _conv_tol) {
            std::cout << "Convergence achieved in " << i << " iterations" << std::endl;
            _energy_tot = hf_energy;
            std::cout << "HF calculation finished in " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count() / 1000.0 
            << " s" << std::endl;
            return true;
        }
    }
    throw std::runtime_error("SCF did not converge within maximum iterations");
    return false;
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
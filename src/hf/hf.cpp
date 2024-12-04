#include "hf.hpp"

namespace HF {
rhf::rhf(GTO::Mol& mol, int max_iter, double conv_tol)
    : int_eng(mol), _max_iter(max_iter), _conv_tol(conv_tol)
{
    int_eng.calc_int();
    nao = int_eng.get_nao();
    nocc = mol.get_nelec()[2] / 2;
    nuc_rep_energy = mol.get_nuc_rep();
}

void rhf::compute_fock_matrix()
{
    auto eri = int_eng.get_int2e();
    auto den = YXTensor::matrix_to_tensor(_D);

// #ifdef _OPENMP                    AI_generated code for parallelization
//     GET_OMP_NUM_THREADS(n_thread)
//     std::vector<Eigen::MatrixXd> J_thread(n_thread);
//     std::vector<Eigen::MatrixXd> K_thread(n_thread);
//     #pragma omp parallel for num_threads(n_thread)
//     for (int i =0; i < n_thread; ++i) {}

    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(nao, nao);
    for (int i = 0; i < nao; ++i) {
        for (int j = 0; j < nao; ++j) {
            for (int k = 0; k < nao; ++k) {
                for (int l = 0; l < nao; ++l) {
                    J(i, j) += eri(i, j, k, l) * den(k, l);
                }
            }
        }
    }

    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(nao, nao);
    for (int i = 0; i < nao; ++i) {
        for (int j = 0; j < nao; ++j) {
            for (int k = 0; k < nao; ++k) {
                for (int l = 0; l < nao; ++l) {
                    K(i, j) += eri(i, k, j, l) * den(k, l);
                }
            }
        }
    }
    // auto J = YXTensor::einsum<2, double, 4, 2, 2>("ijkl, kl->ij", eri, den);
    // auto K = YXTensor::einsum<2, double, 4, 2, 2>("ikjl, kl->ij", eri, den);

    // Eigen::MatrixXd J_mat = YXTensor::tensor_to_matrix(J);
    // Eigen::MatrixXd K_mat = YXTensor::tensor_to_matrix(K);
    _F = 2 * J - K;
}
void rhf::compute_density_matrix()
{
    auto S = int_eng.get_overlap();
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(_F, S);
    Eigen::MatrixXd C = solver.eigenvectors();
    Eigen::MatrixXd C_occ = C.block(0, 0, nao, nocc);
    _D = C_occ * C_occ.transpose();
}

void rhf::compute_init_guess()
{
    _F = int_eng.get_H();
}

void rhf::compute_energy_elec()
{
    auto H = int_eng.get_H();
    for (int i = 0; i < nao; ++i) {
        for (int j = 0; j < nao; ++j) {
            elec_energy += (H(i, j) + _F(i, j)) * _D(i, j);
        }
    }
}

void rhf::compute_energy_tot()
{
    compute_energy_elec();
    energy_tot = nuc_rep_energy + elec_energy;
}


void rhf::kernel()
{
    double new_energy = 0.0;
    double d_energy = 0.0;
    compute_init_guess();
    compute_density_matrix();
    for (int i = 1; i < _max_iter; ++i) {
        compute_fock_matrix();
        compute_energy_tot();
        compute_density_matrix();
        d_energy = std::abs(new_energy - energy_tot);
        new_energy = energy_tot;
        std::cout << "Iteration: " << i;
        std::cout << " Energy: " << energy_tot;
        std::cout << " Difference: " << d_energy << std::endl;
        if (d_energy < _conv_tol) 
        {
            std::cout << "Convergence achieved in " << i << " iterations" << std::endl;
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
    return energy_tot;
}
} // namespace HF
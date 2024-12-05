#include "hf.hpp"

namespace HF {
rhf::rhf(GTO::Mol& mol, int max_iter, double conv_tol, bool direct, bool DIIS)
    : int_eng(mol), _max_iter(max_iter), _conv_tol(conv_tol), _direct(direct), _DIIS(DIIS)
{
    if (!_direct) {
        int_eng.calc_int();
        _I = int_eng.get_int2e();
    }
    else {
        int_eng.calc_int1e();
        _ijkl = int_eng.get_ijkl();
        _ijkl_size = _ijkl.size();
    }

    nao = int_eng.get_nao();
    nocc = mol.get_nelec()[2] / 2;
    _nuc_rep_energy = mol.get_nuc_rep();
    _S = int_eng.get_overlap();
    _H = int_eng.get_H();
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
                        local_G(i, j) += (_I(i, j, k, l) * 2 - _I(i, k, j, l)) * _D(k, l);
                    }
                }
            }
        }

#pragma omp critical
        G += local_G;
    }
    _F = _H + G;

    //     const auto den = YXTensor::matrix_to_tensor(_D);
    //     auto result = _I.shuffle(Eigen::array<int, 4> { 0, 2, 1, 3 }) * -1.0 + _I * 2.0;
    //     auto G_ = YXTensor::einsum<2, double, 4, 2, 2>("ijkl,kl->ij", result, den);
    //     _F = _H + YXTensor::tensor_to_matrix(G_);
}

void rhf::compute_fock_matrix_direct()
{
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(nao, nao);

#pragma omp parallel
    {
        Eigen::MatrixXd local_G = Eigen::MatrixXd::Zero(nao, nao);

#pragma omp for schedule(dynamic)
        for (int t = 0; t < _ijkl_size; t++) {
            auto [i, j, k, l] = _ijkl[t];
            auto dim = int_eng.get_dim(i, j, k, l);
            auto [di, dj, dk, dl] = dim;
            auto [x, y, z, w] = int_eng.get_offset(i, j, k, l);
            double s1234_deg = degeneracy(i, j, k, l);

            auto buf = int_eng.calc_int2e_shell(_ijkl[t], dim);

            for (int fi = 0; fi < di; fi++) {
                for (int fj = 0; fj < dj; fj++) {
                    for (int fk = 0; fk < dk; fk++) {
                        for (int fl = 0; fl < dl; fl++) {
                            local_G(x + fi, y + fj) += _D(z + fk, w + fl) * buf(fi, fj, fk, fl) * s1234_deg;
                            local_G(z + fk, w + fl) += _D(x + fi, y + fj) * buf(fi, fj, fk, fl) * s1234_deg;
                            local_G(x + fi, z + fk) -= 0.25 * _D(y + fj, w + fl) * buf(fi, fj, fk, fl) * s1234_deg;
                            local_G(y + fj, w + fl) -= 0.25 * _D(x + fi, z + fk) * buf(fi, fj, fk, fl) * s1234_deg;
                            local_G(x + fi, w + fl) -= 0.25 * _D(y + fj, z + fk) * buf(fi, fj, fk, fl) * s1234_deg;
                            local_G(y + fj, z + fk) -= 0.25 * _D(x + fi, w + fl) * buf(fi, fj, fk, fl) * s1234_deg;
                        }
                    }
                }
            }
        }

#pragma omp critical
        G += local_G; // 合并线程局部贡献
    }

    _F = _H + 0.5 * (G + G.transpose()); // 构造最终 Fock 矩阵
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

Eigen::MatrixXd rhf::compute_diis_error()
{
    Eigen::MatrixXd DFS = _F * _D * _S;
    return (DFS - DFS.transpose());
}

Eigen::MatrixXd rhf::apply_diis() 
{
    int n = diis_error_list.size();

    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(n + 1, n + 1);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            B(i, j) = (diis_error_list[i].cwiseProduct(diis_error_list[j])).sum();
        }
        B(i, n) = -1.0;
        B(n, i) = -1.0;
    }
    B(n, n) = 0.0;

    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n + 1);
    rhs(n) = -1.0;

    Eigen::VectorXd coeff = B.colPivHouseholderQr().solve(rhs);

    Eigen::MatrixXd F_new = Eigen::MatrixXd::Zero(nao, nao);
    for (int i = 0; i < n; ++i) {
        F_new += coeff(i) * diis_fock_list[i];
    }

    return F_new;
}

auto rhf::kernel() -> bool
{
    auto start = std::chrono::steady_clock::now();
    double old_energy = 0.0;
    double delta_energy = 0.0;
    double delta_D = 0.0;
    Eigen::MatrixXd old_D = Eigen::MatrixXd::Zero(nao, nao);
    compute_init_guess();

    for (int i = 1; i <= _max_iter; ++i) {
        if (!_direct) {

            compute_fock_matrix();
        }
        else {
            compute_fock_matrix_direct();
        }

        auto hf_energy = compute_energy_tot();

        if (_DIIS) {
            Eigen::MatrixXd diis_error = compute_diis_error();
            diis_fock_list.push_back(_F);
            diis_error_list.push_back(diis_error);

            if (diis_fock_list.size() > diis_max_space) {
                diis_fock_list.erase(diis_fock_list.begin());
                diis_error_list.erase(diis_error_list.begin());
            }

            if (diis_fock_list.size() > 3) {
                _F = apply_diis();
            }
        }

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

            std::cout << "Self consist files takes:" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count() / 1000.0
                      << " s" << std::endl;
            return true;
        }
    }

    throw std::runtime_error("Convergence not achieved");
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
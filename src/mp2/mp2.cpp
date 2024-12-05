#include "mp2.hpp"

namespace MP2 {
MP2::MP2(GTO::Mol& mol, int frozen)
    : hf_eng(mol), nfrozen(frozen)
{
    auto converge = hf_eng.kernel(false);
    if (!converge) {
        throw std::runtime_error("MP2 cannot be run because HF has not converged.");
    }
    _H_mo = hf_eng.get_int1e();
    _I_mo = hf_eng.get_int2e();
    _C = hf_eng.get_coeff();
    _orb_energy = hf_eng.get_orb_energy();
    nao = hf_eng.get_nao();
    nocc = hf_eng.get_nocc();
    nvir = nao - nocc;
}
void MP2::ao_to_mo()
{
    _H_mo = _C.transpose() * _H_mo * _C;

    Eigen::Tensor<double, 4> tmp(nao, nao, nao, nao);

    tmp.setZero();
#pragma omp parallel
#pragma omp for nowait schedule(dynamic)
    for (int s = 0; s < nao; ++s) // STEP 1  (iq | rs)<-(pq | rs)
    {
        for (int r = 0; r < nao; ++r)
            for (int q = 0; q < nao; ++q)
                for (int i = 0; i < nao; ++i)
                    for (int p = 0; p < nao; ++p)
                        tmp(i, q, r, s) += _I_mo(p, q, r, s) * _C(p, i);
    }

    _I_mo.setZero();
#pragma omp parallel
#pragma omp for nowait schedule(dynamic)
    for (int s = 0; s < nao; ++s) // STEP 2  (ij | rs)<-(iq | rs)
    {
        for (int r = 0; r < nao; ++r)
            for (int j = 0; j < nao; ++j)
                for (int q = 0; q < nao; ++q)
                    for (int i = 0; i < nao; ++i)
                        _I_mo(i, j, r, s) += tmp(i, q, r, s) * _C(q, j);
    }
    tmp.setZero();
#pragma omp parallel
#pragma omp for nowait schedule(dynamic)
    for (int s = 0; s < nao; ++s) // STEP 3  (ij | ks)<-(ij | rs)
    {
        for (int k = 0; k < nao; ++k)
            for (int r = 0; r < nao; ++r)
                for (int j = 0; j < nao; ++j)
                    for (int i = 0; i < nao; ++i)
                        tmp(i, j, k, s) += _I_mo(i, j, r, s) * _C(r, k);
    }
    _I_mo.setZero();

#pragma omp parallel
#pragma omp for nowait schedule(dynamic)
    for (int l = 0; l < nao; ++l) // STEP 4  (ij | kl)<-(ij | ks)
    {
        for (int s = 0; s < nao; ++s)
            for (int k = 0; k < nao; ++k)
                for (int j = 0; j < nao; ++j)
                    for (int i = 0; i < nao; ++i)
                        _I_mo(i, j, k, l) += tmp(i, j, k, s) * _C(s, l);
    }
    tmp.resize(0, 0, 0, 0);
}

void MP2::kernel()
{
    auto start = std::chrono::high_resolution_clock::now();
    ao_to_mo();

#pragma omp parallel for schedule(dynamic) reduction(+ : energy_mp2)
    for (int b = nocc; b < nao; b++) {
        for (int j = nfrozen; j < nocc; j++) {
            for (int a = nocc; a < nao; a++) {
                for (int i = nfrozen; i < nocc; i++) {
                    double numerator = _I_mo(i, a, j, b) * (2.0 * _I_mo(i, a, j, b) - _I_mo(i, b, j, a));
                    double denominator = _orb_energy(i) + _orb_energy(j) - _orb_energy(a) - _orb_energy(b);
                    energy_mp2 += numerator / denominator;
                }
            }
        }
    }
    total_energy = hf_eng.get_energy_tot() + energy_mp2;
    std::cout << std::format("MP2 energy: {:>12.10f} | Total energy: {:>12.10f}", energy_mp2, total_energy) << std::endl;
    std::cout << std::format("MP2 taken: {:>8.3f} s \n", std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count()) << std::endl;
}

} // namespace MP2

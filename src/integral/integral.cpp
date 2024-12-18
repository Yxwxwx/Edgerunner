#include "integral.hpp"

namespace Integral {
Integral::Integral(GTO::Mol& mol)
{
    // Get CINT information from the molecular object
    auto tmp = mol.get_cint_info();
    // Initialize atomic information
    _atm = tmp.atm;
    // Initialize basis set information
    _bas = tmp.bas;
    // Initialize environment information
    _env = tmp.env;
    // Initialize the number of atoms
    _natm = tmp.natm;
    // Initialize the number of basis functions
    _nbas = tmp.nbas;

    // Generate the number of atomic orbitals
    gen_nao();
    // Generate s8 integral index
    gen_s8();
    // Generate Hermitian integral index
    gen_hermit();
    // Calculate integrals
    // calc_int();
    // Optimize CINT integral calculation
    cint2e_sph_optimizer(&opt, _atm.data(), _natm, _bas.data(), _nbas, _env.data());
}

void Integral::gen_nao()
{
    // Calculate the total number of atomic orbitals
    for (auto i = 0; i < _nbas; i++) {
        // Number of orbitals for each basis function
        nao += (_bas[i * BAS_SLOTS + ANG_OF] * 2 + 1) * _bas[i * BAS_SLOTS + NCTR_OF];
    }
}

void Integral::gen_s8()
{
    // Generate s8 integral index
    // since (ij|kl) = (ij|lk) = (ji|kl) = (ji|lk) = (kl|ij) = (kl|ji) = (lk|ij) = (lk|ji)
    for (int l = 0; l < _nbas; l++) {
        for (int k = l; k < _nbas; k++) {
            for (int j = 0; j < _nbas; j++) {
                for (int i = j; i < _nbas; i++) {
                    if (std::tuple(i, j) <= std::tuple(k, l)) {
                        _ijkl.emplace_back(i, j, k, l);
                    }
                }
            }
        }
    }
    _ijkl_size = _ijkl.size();
}

void Integral::gen_hermit()
{
    // Generate Hermitian integral index
    // since (i|j) = (j|i)
    for (auto j = 0; j < _nbas; j++) {
        for (auto i = j; i < _nbas; i++) {
            _ij.emplace_back(i, j);
        }
    }
    _ij_size = _ij.size();
}

auto Integral::calc_int1e() -> void
{
    _S.resize(nao, nao);
    _S.setZero();
    _T.resize(nao, nao);
    _T.setZero();
    _V.resize(nao, nao);
    _V.setZero();

#pragma omp parallel for schedule(dynamic)
    for (int t = 0; t < _ij_size; t++) {

        auto [i, j] = _ij[t];

        int shls[2];
        shls[0] = i;
        int di = CINTcgto_spheric(i, _bas.data());
        int x = CINTtot_cgto_spheric(_bas.data(), i);

        shls[1] = j;
        int dj = CINTcgto_spheric(j, _bas.data());
        int y = CINTtot_cgto_spheric(_bas.data(), j);

        Eigen::MatrixXd buf_s(di, dj);
        Eigen::MatrixXd buf_t(di, dj);
        Eigen::MatrixXd buf_v(di, dj);

        cint1e_ovlp_sph(buf_s.data(), shls, _atm.data(), _natm, _bas.data(), _nbas, _env.data());
        cint1e_kin_sph(buf_t.data(), shls, _atm.data(), _natm, _bas.data(), _nbas, _env.data());
        cint1e_nuc_sph(buf_v.data(), shls, _atm.data(), _natm, _bas.data(), _nbas, _env.data());

#pragma omp critical
        {
            _S.block(x, y, di, dj) = buf_s;
            _S.block(y, x, dj, di) = buf_s.transpose();
            _T.block(x, y, di, dj) = buf_t;
            _T.block(y, x, dj, di) = buf_t.transpose();
            _V.block(x, y, di, dj) = buf_v;
            _V.block(y, x, dj, di) = buf_v.transpose();
        }
    }
}

auto Integral::calc_int2e() -> void
{
    auto start = std::chrono::steady_clock::now();

    _I.resize(nao, nao, nao, nao);
    _I.setZero();

#pragma omp parallel for shared(opt)
    for (auto t = 0; t < _ijkl_size; t++) {

        auto [i, j, k, l] = _ijkl[t];
        int shls[] = { i, j, k, l };

        int di = CINTcgto_spheric(i, _bas.data());
        int x = CINTtot_cgto_spheric(_bas.data(), i);

        int dj = CINTcgto_spheric(j, _bas.data());
        int y = CINTtot_cgto_spheric(_bas.data(), j);

        int dk = CINTcgto_spheric(k, _bas.data());
        int z = CINTtot_cgto_spheric(_bas.data(), k);

        int dl = CINTcgto_spheric(l, _bas.data());
        int h = CINTtot_cgto_spheric(_bas.data(), l);

        Eigen::Tensor<double, 4> buf_i(di, dj, dk, dl);
        cint2e_sph(buf_i.data(), shls, _atm.data(), _natm, _bas.data(), _nbas, _env.data(), opt);

#pragma omp critical
        {
            _I.slice(Eigen::array<int, 4> { x, y, z, h }, Eigen::array<int, 4> { di, dj, dk, dl }) = buf_i;
            //     _I.slice(Eigen::array<int, 4> { y, x, z, h }, Eigen::array<int, 4> { dj, di, dk, dl }) = buf_i.shuffle(Eigen::array<int, 4> { 1, 0, 2, 3 });
            //     _I.slice(Eigen::array<int, 4> { x, y, h, z }, Eigen::array<int, 4> { di, dj, dl, dk }) = buf_i.shuffle(Eigen::array<int, 4> { 0, 1, 3, 2 });
            //     _I.slice(Eigen::array<int, 4> { y, x, h, z }, Eigen::array<int, 4> { dj, di, dl, dk }) = buf_i.shuffle(Eigen::array<int, 4> { 1, 0, 3, 2 });
            //     _I.slice(Eigen::array<int, 4> { z, h, x, y }, Eigen::array<int, 4> { dk, dl, di, dj }) = buf_i.shuffle(Eigen::array<int, 4> { 2, 3, 0, 1 });
            //     _I.slice(Eigen::array<int, 4> { h, z, x, y }, Eigen::array<int, 4> { dl, dk, di, dj }) = buf_i.shuffle(Eigen::array<int, 4> { 3, 2, 0, 1 });
            //     _I.slice(Eigen::array<int, 4> { z, h, y, x }, Eigen::array<int, 4> { dk, dl, dj, di }) = buf_i.shuffle(Eigen::array<int, 4> { 2, 3, 1, 0 });
            //     _I.slice(Eigen::array<int, 4> { h, z, y, x }, Eigen::array<int, 4> { dl, dk, dj, di }) = buf_i.shuffle(Eigen::array<int, 4> { 3, 2, 1, 0 });
        }
    }

    std::cout << "Integral calculation time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count() / 1000.0
              << " s" << std::endl;
}

auto Integral::calc_int2e_shell(std::tuple<int, int, int, int> ijkl, std::tuple<int, int, int, int> dim) -> Eigen::Tensor<double, 4>
{
    auto [i, j, k, l] = ijkl;
    auto [di, dj, dk, dl] = dim;
    int shls[] = { i, j, k, l };

    Eigen::Tensor<double, 4> buf_i(di, dj, dk, dl);
    cint2e_sph(buf_i.data(), shls, _atm.data(), _natm, _bas.data(), _nbas, _env.data(), nullptr);

    return buf_i;
}

auto Integral::calc_int() -> void
{
    calc_int1e();
    calc_int2e();
}

const Eigen::MatrixXd& Integral::get_overlap()
{
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
    _H.resize(nao, nao);
    _H = get_kinetic() + get_nuc();
    return _H;
}

const Eigen::Tensor<double, 4>& Integral::get_int2e()
{
    return _I;
}

auto Integral::get_ijkl() -> std::vector<std::tuple<int, int, int, int>>
{
    return _ijkl;
}
auto Integral::get_offset(int i, int j, int k, int l) -> std::tuple<int, int, int, int>
{

    int x = CINTtot_cgto_spheric(_bas.data(), i);
    int y = CINTtot_cgto_spheric(_bas.data(), j);
    int z = CINTtot_cgto_spheric(_bas.data(), k);
    int h = CINTtot_cgto_spheric(_bas.data(), l);
    return { x, y, z, h };
}

auto Integral::get_dim(int i, int j, int k, int l) -> std::tuple<int, int, int, int>
{
    int di = CINTcgto_spheric(i, _bas.data());
    int dj = CINTcgto_spheric(j, _bas.data());
    int dk = CINTcgto_spheric(k, _bas.data());
    int dl = CINTcgto_spheric(l, _bas.data());
    return { di, dj, dk, dl };
}
int Integral::get_nao() const { return nao; }

} // namespace Integral

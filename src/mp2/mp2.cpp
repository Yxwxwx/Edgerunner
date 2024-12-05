#include "mp2.hpp"

namespace MP2 {
MP2::MP2(GTO::Mol& mol)
    : hf_eng(mol)
{
    auto converge = hf_eng.kernel(false);
    if (!converge) {
        throw std::runtime_error("MP2 cannot be run because HF has not converged.");
    }
    _H_ao = hf_eng.get_int1e();
    _I_ao = hf_eng.get_int2e();
    _C = hf_eng.get_coeff();
    _orb_energy = hf_eng.get_orb_energy();
    nao = hf_eng.get_nao();
    nocc = hf_eng.get_nocc();
    nvir = nao - nocc;
}
void MP2::ao_to_mo()
{
    auto C_T = _C.transpose();

    for (int j = 0; j < nao; j++) {
        for (int i = 0; i < nao; i++) {
            _H_mo(i, j) = C_T(i, j) * _H_ao(i, j) * _C(i, j);
        }
    }

    for (int S = 0; S < nao; S++) {
        for (int s = 0; s < nao; s++) {
            for (int r = 0; r < nao; r++) {
                for (int q = 0; q < nao; q++) {
                    for (int p = 0; p < nao; p++) {
                        _I_mo(p, q, r, S) = _I_ao(p, q, r, s) * _C(s, S);
                    }
                }
            }
        }
    }
    for (int R = 0; R < nao; R++) {
        for (int S = 0; S < nao; S++) {
            for (int r = 0; r < nao; r++) {
                for (int q = 0; q < nao; q++) {
                    for (int p = 0; p < nao; p++) {
                        _I_mo(p, q, R, S) += _I_mo(p, q, r, S) * _C(r, R);
                    }
                }
            }
        }
    }
    for (int Q = 0; Q < nao; Q++) {
        for (int S = 0; S < nao; S++) {
            for (int R = 0; R < nao; R++)
                for (int q = 0; q < nao; q++) {
                    for (int p = 0; p < nao; p++) {
                        _I_mo(p, Q, R, S) += _I_mo(p, q, R, S) * _C(q, Q);
                    }
                }
        }
    }
    for (int P = 0; P < nao; P++) {
        for (int S = 0; S < nao; S++) {
            for (int R = 0; R < nao; R++) {
                for (int Q = 0; Q < nao; Q++) {
                    for (int p = 0; P < nao; p++) {
                        _I_mo(P, Q, R, S) += _I_mo(p, Q, R, S) * _C(p, P);
                    }
                }
            }
        }
    }
}

void MP2::kernel()
{
    ao_to_mo();
    for (int b = nocc; b < nao; b++) {
        for (int a = nocc; a < nao; a++) {
            for (int j = 0; j < nocc; j++) {
                for (int i = 0; i < nocc; i++) {
                    energy_mp2 -= (_I_mo(a, i, b, j) * (2.0 * _I_mo(a, i, b, j) - _I_mo(a, j, b, i))) / (_orb_energy(a) + _orb_energy(b) - _orb_energy(i) - _orb_energy(j));
                }
            }
        }
    }
    total_energy = hf_eng.get_energy_tot() + energy_mp2;
    std::cout << std::format("MP2 energy: {:>12.10f} | Total energy: {:>12.10f}", energy_mp2, total_energy) << std::endl;
}

} // namespace MP2

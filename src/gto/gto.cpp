#include "gto.hpp"
#include "Eigen/Core"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" {
#include <cint.h>
}

namespace GTO {
Mol::Mol(const std::string& xyz, const std::string& basis, const int spin,
    const int charge)
    : _xyz_info(xyz), _basis_info(basis), _spin(spin), _charge(charge)
{
    parseXYZ(xyz);
    parseBasis(basis);
    setupCintInfo();
}

void Mol::_nuclear_repulsion()
{
    for (size_t i = 0; i < atoms.size(); i++)
        for (size_t j = i + 1; j < atoms.size(); j++) {
            auto xij = atoms[i].x * angstrom_to_bohr - atoms[j].x * angstrom_to_bohr;
            auto yij = atoms[i].y * angstrom_to_bohr - atoms[j].y * angstrom_to_bohr;
            auto zij = atoms[i].z * angstrom_to_bohr - atoms[j].z * angstrom_to_bohr;
            auto r2 = xij * xij + yij * yij + zij * zij;
            auto r = sqrt(r2);
            _nuc_rep += atoms[i].Z * atoms[j].Z / r;
        }
}
void Mol::parseXYZ(const std::string& xyz)
{
    std::istringstream iss(xyz);
    std::string atomInfo;

    while (std::getline(iss, atomInfo, ';')) {

        atomInfo = atomInfo.substr(atomInfo.find_first_not_of(" "),
            atomInfo.find_last_not_of(" ") - atomInfo.find_first_not_of(" ") + 1);

        std::istringstream atomStream(atomInfo);
        std::string atomSymbol;
        double x, y, z;

        atomStream >> atomSymbol >> x >> y >> z;
        if (atomStream.fail()) {
            throw std::invalid_argument("Failed to parse atom line: " + atomInfo);
        }

        if (!atomSymbol.empty()) {
            if (atomSymbol.size() == 1) {
                std::transform(atomSymbol.begin(), atomSymbol.end(), atomSymbol.begin(),
                    ::toupper);
            }
            else if (atomSymbol.size() == 2) {

                atomSymbol[0] = std::toupper(atomSymbol[0]);
                atomSymbol[1] = std::tolower(atomSymbol[1]);
            }
        }

        auto it = element_table.find(atomSymbol);
        if (it == element_table.end()) {
            throw std::invalid_argument("Unknown element symbol: " + atomSymbol);
        }
        int Z = it->second;

        atoms.push_back({ atomSymbol, Z, x, y, z });
    }
    _nuclear_repulsion();
}

void Mol::parseBasis(const std::string& basis)
{
    _basis_name = basis;

    // 转为小写
    std::transform(_basis_name.begin(), _basis_name.end(), _basis_name.begin(),
        [](unsigned char c) { return std::tolower(c); });

    // 先处理连续的两个 '*'
    _basis_name = std::regex_replace(_basis_name, std::regex("\\*\\*"), "_d_p_");
    // 再处理单个 '*'
    _basis_name = std::regex_replace(_basis_name, std::regex("\\*"), "_d_");
}

void Mol::setupCintInfo()
{
    // 初始化 atm 和 env 的大小
    info.natm = static_cast<int>(atoms.size());
    info.atm.resize(ATM_SLOTS * info.natm); // atm 是二维数组，展平为一维

    info.env.resize(20, 0.0);

    int env_index = PTR_ENV_START;
    for (int i = 0; i < info.natm; ++i) {
        const auto& atom = atoms[i];

        info.atm[i * ATM_SLOTS + 0] = atom.Z;

        info.atm[i * ATM_SLOTS + 1] = env_index;
        info.env.push_back(atom.x * angstrom_to_bohr);
        info.env.push_back(atom.y * angstrom_to_bohr);
        info.env.push_back(atom.z * angstrom_to_bohr);
        info.env.push_back(0.0);
        env_index += 4;

        info.atm[i * ATM_SLOTS + 2] = 1;

        info.atm[i * ATM_SLOTS + 3] = env_index - 1;

        info.atm[i * ATM_SLOTS + 4] = 0;
        info.atm[i * ATM_SLOTS + 5] = 0;
    }

    std::set<int> unique_Z;
    for (const auto& a : atoms) {
        unique_Z.insert(a.Z);
    }

    std::vector<int> sorted_Z(unique_Z.begin(), unique_Z.end());
    std::sort(sorted_Z.begin(), sorted_Z.end());

    std::vector<std::pair<std::string, int>> result;
    for (int Z : sorted_Z) {
        auto it = element_table_reversed.find(Z);
        if (it != element_table_reversed.end()) {
            result.emplace_back(it->second, Z);
        }
        else {
            result.emplace_back("Unknown", Z);
        }
    }

    std::string sep;
#ifdef _WIN32
    sep = "\\";
#else
    sep = "/";
#endif
    std::string basisfile = "share" + sep + "basis" + sep + _basis_name + ".g94";
    std::string line, am;
    std::fstream fin(basisfile, std::ios::in);
    int idxe, idxc;
    if (fin.good()) {
        std::vector<std::string> wl;
        int l, ncf1, cf1, npf, pf;

        for (auto e : result) {
            do // Skip the beginning for basisfile
            {
                getline(fin, line);
                line = line.erase(line.find_last_not_of("\r\n") + 1);
            } while (line != "****");
            fin.clear();
            fin.seekg(0, std::ios::beg);
            std::vector<Shell> shl;

            do // Each line begins with symb    0
            {
                getline(fin, line);
                line = line.erase(line.find_last_not_of("\r\n") + 1);
                if (fin.eof()) {
                    std::string err = basisfile + " is broken!!!!";
                    throw std::runtime_error(err);
                }

            } while (line != e.first + "     0 ");

            getline(fin, line); // S/SP 3/6 1.0
            line = line.erase(line.find_last_not_of("\r\n") + 1);
            do {
                wl = split(line);
                am = wl[0];
                npf = stoi(wl[1]);

                if (am == "SP") {
                    Eigen::MatrixXd ec(npf, 2), ec1(npf, 2);
                    for (pf = 0; pf < npf; ++pf) {
                        getline(fin, line);
                        line = line.erase(line.find_last_not_of("\r\n") + 1);
                        wl = split(line);
                        std::replace(wl[1].begin() + 8, wl[1].end(), 'D', 'E');
                        std::replace(wl[2].begin() + 8, wl[2].end(), 'D', 'E');
                        ec(pf, 0) = static_cast<double>(stod(wl[0]));
                        ec(pf, 1) = static_cast<double>(stod(wl[1]));
                        ec1(pf, 0) = static_cast<double>(stod(wl[0]));
                        ec1(pf, 1) = static_cast<double>(stod(wl[2]));
                    }

                    shlNormalize(0, ec);
                    idxe = info.env.size();
                    std::vector<double> exp;
                    for (int i = 0; i < ec.rows(); ++i) {
                        exp.push_back(ec(i, 0)); // 获取第 i 行，第 0 列的值
                    }
                    info.env.insert(info.env.end(), exp.begin(), exp.end());
                    idxc = info.env.size();
                    std::vector<double> con;
                    for (auto i = 0; i < ec.rows(); ++i) {
                        con.push_back(ec(i, 1));
                    }
                    info.env.insert(info.env.end(), con.begin(), con.end());

                    shl.emplace_back(
                        Shell { am, { 0, npf, 1, 0, idxe, idxc } });

                    shlNormalize(1, ec1);
                    idxe = info.env.size();
                    std::vector<double> exp1;
                    for (auto i = 0; i < ec1.rows(); ++i) {
                        exp1.push_back(ec1(i, 0));
                    }
                    info.env.insert(info.env.end(), exp1.begin(), exp1.end());
                    int idxc = info.env.size();
                    std::vector<double> con1;
                    for (auto i = 0; i < ec1.rows(); ++i) {
                        con1.push_back(ec1(i, 1));
                    }
                    info.env.insert(info.env.end(), con1.begin(), con1.end());

                    shl.emplace_back(
                        Shell({ am, { 1, npf, 1, 0, idxe, idxc } }));
                }
                else {
                    l = aml.at(am);
                    getline(fin, line);
                    line = line.erase(line.find_last_not_of("\r\n") + 1);
                    wl = split(line);

                    ncf1 = wl.size();
                    Eigen::MatrixXd ec(npf, ncf1);
                    for (cf1 = 0; cf1 < ncf1; ++cf1) {
                        std::replace(wl[cf1].begin() + 8, wl[cf1].end(), 'D', 'E');
                        ec(0, cf1) = stod(wl[cf1]);
                    }
                    for (pf = 1; pf < npf; ++pf) {
                        getline(fin, line);
                        wl = split(line);
                        line = line.erase(line.find_last_not_of("\r\n") + 1);
                        for (cf1 = 0; cf1 < ncf1; ++cf1) {
                            std::replace(wl[cf1].begin() + 8, wl[cf1].end(), 'D', 'E');
                            ec(pf, cf1) = stod(wl[cf1]);
                        }
                    }

                    shlNormalize(l, ec);
                    idxe = info.env.size();

                    std::vector<double> exp;
                    for (auto i = 0; i < ec.rows(); ++i) {
                        exp.push_back(ec(i, 0));
                    }
                    info.env.insert(info.env.end(), exp.begin(), exp.end());
                    idxc = info.env.size();
                    std::vector<double> con;
                    for (auto i = 0; i < ec.rows(); ++i) {
                        con.push_back(ec(i, 1));
                    }
                    info.env.insert(info.env.end(), con.begin(), con.end());

                    shl.emplace_back(Shell {
                        am, { l, npf, ncf1 - 1, 0, idxe, idxc } });
                }

                getline(fin, line); // S/SP 3/6 1.0
                line = line.erase(line.find_last_not_of("\r\n") + 1);
            } while (line != "****");

            dshl[e.first] = shl;
        }

        fin.close();
    }
    else {
        std::string err = basisfile + " not exist!";
        throw std::runtime_error(err);
    }

    for (auto i = 0; i < info.natm; i++) {
        auto sym = atoms[i].symbol;
        auto shls = dshl[sym];
        for (auto& shl : shls) {
            info.bas.push_back(i);
            info.bas.insert(info.bas.end(), shl.bas_info.begin(), shl.bas_info.end());
            info.bas.push_back(0);
        }
    }
    info.nbas = info.bas.size() / BAS_SLOTS;
}

std::vector<std::string> Mol::split(const std::string& _str,
    const std::string& _flag)
{
    std::vector<std::string> result;
    std::string str = _str + _flag;
    auto size = _flag.size();
    std::string sub;

    for (auto i = 0; i < str.size();) {
        auto p = str.find(_flag, i);
        sub = str.substr(i, p - i);
        if (sub != "") {
            result.emplace_back(sub);
        }
        i = p + size;
    }
    return result;
}
void Mol::shlNormalize(int l, Eigen::MatrixXd& shl)
{

    for (int i = 0; i < shl.rows(); i++) {
        shl(i, 1) *= CINTgto_norm(l, shl(i, 0));
    }
}

void Mol::printAtoms() const
{
    std::cout << std::format("Atoms:{}:\n", atoms.size());
    std::cout << std::format("Spin: {}, Charge: {}\n", _spin, _charge);
    for (const auto& atom : atoms) {
        std::cout << std::format("Atom: Z={}, x={}, y={}, z={}\n", atom.Z, atom.x,
            atom.y, atom.z);
    }
}

void Mol::printCintInfo() const
{
    std::cout << "natm = " << info.natm << "\n";
    std::cout << "nbas = " << info.nbas << "\n";
    // atm
    std::cout << "\n atm: \n"
              << std::endl;
    for (auto i = 0; i < info.natm; i++) {
        for (auto j = 0; j < ATM_SLOTS; j++) {
            std::cout << std::format("{:>4}", info.atm[i * ATM_SLOTS + j]);
        }
        std::cout << std::endl;
    }
    std::cout << "\n bas: \n"
              << std::endl;
    for (auto i = 0; i < info.nbas; i++) {
        for (auto j = 0; j < BAS_SLOTS; j++) {
            std::cout << std::format("{:>4}", info.bas[i * BAS_SLOTS + j]);
        }
        std::cout << std::endl;
    }
    std::cout << "\n env: \n"
              << std::endl;
    for (auto i = 0; i < info.env.size(); i++) {
        std::cout << std::format("{:10.4f}", info.env[i]);
        if ((i + 1) % 5 == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

cint_info Mol::get_cint_info() const
{
    return info;
}

double Mol::get_nuc_rep() const
{
    return _nuc_rep;
}

} // namespace GTO

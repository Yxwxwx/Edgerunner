#include "gto.hpp"
extern "C" {
#include <cint.h>
}

namespace GTO {
Mol::Mol(const std::string& xyz, const std::string& basis, const int spin, const int charge)
    : _xyz_info(xyz), _basis_info(basis), _spin(spin), _charge(charge)
{
    parseXYZ(xyz);
    parseBasis(basis);
    setupCintInfo();
}

void Mol::parseXYZ(const std::string& xyz)
{
    std::istringstream iss(xyz);
    std::string atomInfo;

    while (std::getline(iss, atomInfo, ';')) {

        atomInfo = atomInfo.substr(atomInfo.find_first_not_of(" "), atomInfo.find_last_not_of(" ") - atomInfo.find_first_not_of(" ") + 1);

        std::istringstream atomStream(atomInfo);
        std::string atomSymbol;
        double x, y, z;

        atomStream >> atomSymbol >> x >> y >> z;
        if (atomStream.fail()) {
            throw std::invalid_argument("Failed to parse atom line: " + atomInfo);
        }

        if (!atomSymbol.empty()) {
            if (atomSymbol.size() == 1) {
                std::transform(atomSymbol.begin(), atomSymbol.end(), atomSymbol.begin(), ::toupper);
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

        atoms.push_back({ Z, x, y, z });
    }
}

void Mol::parseBasis(const std::string& basis)
{
    std::string processed_basis = basis;
    BasisType type = BasisType::UNKNOWN;

    std::string basis_lower = basis;
    std::transform(basis_lower.begin(), basis_lower.end(),
        basis_lower.begin(), ::tolower);

    if (std::isdigit(basis_lower[0]) || basis_lower.substr(0, 3) == "sto") {
        type = BasisType::POPLE;
        processed_basis = basis_lower;
        std::replace(processed_basis.begin(), processed_basis.end(), '*', '_');
        std::transform(processed_basis.begin(), processed_basis.end(),
            processed_basis.begin(), ::toupper);
    }

    else if (basis_lower.substr(0, 4) == "def2") {
        type = BasisType::DEF2;
        processed_basis = "def2-";
        size_t pos = basis_lower.find("def2-");
        if (pos != std::string::npos && pos + 5 < basis_lower.length()) {
            std::string suffix = basis_lower.substr(pos + 5);
            std::transform(suffix.begin(), suffix.end(),
                suffix.begin(), ::toupper);
            processed_basis += suffix;
        }
    }

    else if (basis_lower.substr(0, 2) == "cc" || basis_lower.substr(0, 6) == "aug-cc") {
        type = BasisType::CC;

        bool is_aug = basis_lower.substr(0, 6) == "aug-cc";

        processed_basis = is_aug ? "aug-cc-p" : "cc-p";

        // 获取需要大写的后缀部分
        size_t suffix_start = is_aug ? 7 : 3;
        if (basis_lower.length() > suffix_start) {
            std::string suffix = basis_lower.substr(suffix_start);
            std::transform(suffix.begin(), suffix.end(),
                suffix.begin(), ::toupper);
            processed_basis += suffix;
        }
    }
    else {
        throw std::invalid_argument(
            std::format("Unsupported basis set: {}. Must start with a number, 'sto', 'def2', or 'cc'/'aug-cc'", basis));
    }

    _basis_type = type;
    _basis_name = processed_basis;
}

void Mol::setupCintInfo()
{
    std::filesystem::path basis_dir = "share/basis";
    basis_dir /= _basis_name;

    info.env.resize(20, 0.0);

    info.natm = atoms.size();
    for (const auto& atom : atoms) {
        auto coord_pos = info.env.size();
        info.env.emplace_back(atom.x * angstrom_to_bohr);
        info.env.emplace_back(atom.y * angstrom_to_bohr);
        info.env.emplace_back(atom.z * angstrom_to_bohr);
        info.env.emplace_back(0.0);

        info.atm.emplace_back(atom.Z);
        info.atm.emplace_back(coord_pos);
        info.atm.emplace_back(1);
        info.atm.emplace_back(coord_pos + 3);
        info.atm.emplace_back(0);
        info.atm.emplace_back(0);
    }

    info.nbas = 0;

    std::vector<std::vector<std::tuple<int, std::vector<double>, std::vector<std::vector<double>>>>> atom_shells(atoms.size());

    for (size_t atom_idx = 0; atom_idx < atoms.size(); ++atom_idx) {
        std::string element;
        for (const auto& [symbol, number] : element_table) {
            if (number == atoms[atom_idx].Z) {
                element = symbol;
                break;
            }
        }

        std::filesystem::path json_path = basis_dir / (element + ".json");
        std::ifstream json_file(json_path);
        if (!json_file) {
            throw std::runtime_error(
                std::format("Cannot open basis file for element {} at {}",
                    element, json_path.string()));
        }
        nlohmann::json basis_data;
        json_file >> basis_data;

        for (const auto& shell : basis_data["electron_shells"]) {
            int angular_momentum = shell["angular_momentum"][0];

            // 处理 POPLE 类型的 basis
            if (_basis_type == BasisType::POPLE) {
                std::vector<double> exponents;
                for (const auto& exp : shell["exponents"]) {
                    exponents.push_back(std::stod(exp.get<std::string>()));
                }

                // 判断是否存在多个 angular_momentum
                if (shell["angular_momentum"].size() > 1) {
                    // 如果 angular_momentum 有多个元素，拆分为多个 shell
                    size_t idx = 0;
                    for (const auto& coeff_set : shell["coefficients"]) {
                        // 每个系数集对应一个 angular_momentum
                        std::vector<double> coeff_vec;
                        for (const auto& coeff : coeff_set) {
                            coeff_vec.push_back(std::stod(coeff.get<std::string>()));
                        }

                        // 存储每个 shell，按照 angular_momentum 拆分
                        atom_shells[atom_idx].push_back({ shell["angular_momentum"][idx], exponents, { coeff_vec } });
                        idx++;
                    }
                }
                else {

                    std::vector<std::vector<double>> coefficients;
                    for (const auto& coeff_set : shell["coefficients"]) {
                        std::vector<double> coeff_vec;
                        for (const auto& coeff : coeff_set) {
                            coeff_vec.push_back(std::stod(coeff.get<std::string>()));
                        }
                        coefficients.push_back(coeff_vec);
                    }

                    // 存储为一个 shell
                    atom_shells[atom_idx].push_back({ angular_momentum, exponents, coefficients });
                }
            }

            else if (_basis_type == BasisType::DEF2) {
                std::vector<double> exponents;
                for (const auto& exp : shell["exponents"]) {
                    exponents.push_back(std::stod(exp.get<std::string>()));
                }

                std::vector<std::vector<double>> coefficients;
                for (const auto& coeff_set : shell["coefficients"]) {
                    std::vector<double> coeff_vec;
                    for (const auto& coeff : coeff_set) {
                        coeff_vec.push_back(std::stod(coeff.get<std::string>()));
                    }
                    coefficients.push_back(coeff_vec);
                }

                size_t exp_pos = info.env.size();
                info.env.insert(info.env.end(), exponents.begin(), exponents.end());

                size_t coeff_pos = info.env.size();
                for (auto& coeff_vec : coefficients) {
                    for (auto i = 0; i < coeff_vec.size(); i++) {
                        coeff_vec[i] *= CINTgto_norm(angular_momentum, exponents[i]);
                    }

                    info.env.insert(info.env.end(), coeff_vec.begin(), coeff_vec.end());
                }

                info.bas.push_back(atom_idx);
                info.bas.push_back(angular_momentum);
                info.bas.push_back(exponents.size());
                info.bas.push_back(coefficients.size());
                info.bas.push_back(0);
                info.bas.push_back(exp_pos);
                info.bas.push_back(coeff_pos);
                info.bas.push_back(0);

                info.nbas++;
            }
        }
    }

    for (size_t atom_idx = 0; atom_idx < atoms.size(); ++atom_idx) {
        std::sort(atom_shells[atom_idx].begin(), atom_shells[atom_idx].end(), [](const auto& lhs, const auto& rhs) {
            return std::get<0>(lhs) < std::get<0>(rhs);
        });

        for (auto& shell_data : atom_shells[atom_idx]) {
            auto& [angular_momentum, exponents, coefficients] = shell_data;

            size_t exp_pos = info.env.size();
            info.env.insert(info.env.end(), exponents.begin(), exponents.end());

            size_t coeff_pos = info.env.size();
            for (auto& coeff_vec : coefficients) {
                for (auto i = 0; i < coeff_vec.size(); i++) {
                    coeff_vec[i] *= CINTgto_norm(angular_momentum, exponents[i]);
                }

                info.env.insert(info.env.end(), coeff_vec.begin(), coeff_vec.end());
            }

            // 将 shell 数据写入 bas
            info.bas.push_back(atom_idx);
            info.bas.push_back(angular_momentum);
            info.bas.push_back(exponents.size());
            info.bas.push_back(1);
            info.bas.push_back(0);
            info.bas.push_back(exp_pos);
            info.bas.push_back(coeff_pos);
            info.bas.push_back(0);

            info.nbas++;
        }
    }
}

void Mol::printAtoms() const
{
    std::cout
        << std::format("Atoms:{}:\n", atoms.size());
    std::cout
        << std::format("Spin: {}, Charge: {}\n", _spin, _charge);
    for (const auto& atom : atoms) {
        std::cout
            << std::format("Atom: Z={}, x={}, y={}, z={}\n", atom.Z, atom.x, atom.y, atom.z);
    }
}

void Mol::printCintInfo() const
{
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
        // 使用固定宽度格式化每个元素
        std::cout << std::format("{:10.4f}", info.env[i]); // 保证每个元素宽度为10，保留4位小数
        if ((i + 1) % 5 == 0) {
            std::cout << std::endl; // 每5个元素换一行
        }
    }
}
} // namespace gto

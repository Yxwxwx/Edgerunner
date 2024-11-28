#pragma once
#ifndef GTO_HPP
#define GTO_HPP

#include <algorithm>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace GTO {

static const std::unordered_map<std::string, int> element_table = {
    {"H", 1},    {"He", 2},   {"Li", 3},   {"Be", 4},   {"B", 5},
    {"C", 6},    {"N", 7},    {"O", 8},    {"F", 9},    {"Ne", 10},
    {"Na", 11},  {"Mg", 12},  {"Al", 13},  {"Si", 14},  {"P", 15},
    {"S", 16},   {"Cl", 17},  {"Ar", 18},  {"K", 19},   {"Ca", 20},
    {"Sc", 21},  {"Ti", 22},  {"V", 23},   {"Cr", 24},  {"Mn", 25},
    {"Fe", 26},  {"Co", 27},  {"Ni", 28},  {"Cu", 29},  {"Zn", 30},
    {"Ga", 31},  {"Ge", 32},  {"As", 33},  {"Se", 34},  {"Br", 35},
    {"Kr", 36},  {"Rb", 37},  {"Sr", 38},  {"Y", 39},   {"Zr", 40},
    {"Nb", 41},  {"Mo", 42},  {"Tc", 43},  {"Ru", 44},  {"Rh", 45},
    {"Pd", 46},  {"Ag", 47},  {"Cd", 48},  {"In", 49},  {"Sn", 50},
    {"Sb", 51},  {"I", 52},   {"Te", 53},  {"Xe", 54},  {"Cs", 55},
    {"Ba", 56},  {"La", 57},  {"Ce", 58},  {"Pr", 59},  {"Nd", 60},
    {"Pm", 61},  {"Sm", 62},  {"Eu", 63},  {"Gd", 64},  {"Tb", 65},
    {"Dy", 66},  {"Ho", 67},  {"Er", 68},  {"Tm", 69},  {"Yb", 70},
    {"Lu", 71},  {"Hf", 72},  {"Ta", 73},  {"W", 74},   {"Re", 75},
    {"Os", 76},  {"Ir", 77},  {"Pt", 78},  {"Au", 79},  {"Hg", 80},
    {"Tl", 81},  {"Pb", 82},  {"Bi", 83},  {"Po", 84},  {"At", 85},
    {"Rn", 86},  {"Fr", 87},  {"Ra", 88},  {"Ac", 89},  {"Th", 90},
    {"Pa", 91},  {"U", 92},   {"Np", 93},  {"Pu", 94},  {"Am", 95},
    {"Cm", 96},  {"Bk", 97},  {"Cf", 98},  {"Es", 99},  {"Fm", 100},
    {"Md", 101}, {"No", 102}, {"Lr", 103}, {"Rf", 104}, {"Db", 105},
    {"Sg", 106}, {"Bh", 107}, {"Hs", 108}, {"Mt", 109}, {"Ds", 110},
    {"Rg", 111}, {"Cn", 112}, {"Nh", 113}, {"Fl", 114}, {"Mc", 115},
    {"Lv", 116}, {"Ts", 117}, {"Og", 118}};
const auto angstrom_to_bohr = 1 / 0.52917721092;

class Mol {
public:
  Mol(const std::string &xyz, const std::string &basis = "sto-3g",
      const int spin = 0, const int charge = 0);
  ~Mol() = default;

  void printAtoms() const;
  void printCintInfo() const;

private:
  std::string _xyz_info;
  std::string _basis_info;
  int _spin;
  int _charge;

  struct atom {
    int Z;
    double x, y, z;
  };
  std::vector<atom> atoms;

  enum class BasisType { POPLE, DEF2, CC, UNKNOWN };
  BasisType _basis_type;
  std::string _basis_name;

  struct cint_info {
    std::vector<int> atm;
    std::vector<int> bas;
    std::vector<double> env;
    int natm;
    int nbas;
  };
  cint_info info;

  void parseXYZ(const std::string &xyz);
  void setupCintInfo();
  void parseBasis(const std::string &basis);
};
} // namespace GTO
#endif

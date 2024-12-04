#include "gto/gto.hpp"
#include "hf/hf.hpp"
#include "integral/integral.hpp"

auto main() -> int
{
    GTO::Mol H2O("O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587", "sto-3g");
    HF::rhf rhf_engine(H2O);
    rhf_engine.kernel();
    // Integral::Integral int_engine(He);
    // int_engine.calc_int();
    // std::cout << "Overlap Matrix: \n";
    // std::cout << int_engine.get_overlap() << "\n"
    //           << std::endl;
    // std::cout << "Kinetic Matrix: \n";
    // std::cout << int_engine.get_kinetic() << "\n"
    //           << std::endl;
    // std::cout << "Nuclear Electron Matrix: \n";
    // std::cout << int_engine.get_nuc() << "\n"
    //           << std::endl;
    // std::cout << "Int2e: \n";
    // YXTensor::print_tensor(int_engine.get_int2e());
    return 0;
}
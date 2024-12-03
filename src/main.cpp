#include "gto/gto.hpp"
#include "integral/integral.hpp"
#include "linalg/einsum.hpp"

auto main() -> int
{
    GTO::Mol He("He 0 0 -0.64; He 0 0 0.64", "sto-3g");
    Integral::Integral int_engine(He);
    int_engine.calc_int();
    std::cout << "Overlap Matrix: \n";
    std::cout << int_engine.get_overlap() << "\n"
              << std::endl;
    std::cout << "Kinetic Matrix: \n";
    std::cout << int_engine.get_kinetic() << "\n"
              << std::endl;
    std::cout << "Nuclear Electron Matrix: \n";
    std::cout << int_engine.get_nuc() << "\n"
              << std::endl;
    std::cout << "Int2e: \n";
    YXTensor::print_tensor(int_engine.get_int2e());
    return 0;
}
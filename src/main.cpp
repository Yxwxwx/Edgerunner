#include "gto/gto.hpp"
#include "integral/integral.hpp"
#include "linalg/einsum.hpp"

auto main() -> int
{
    GTO::Mol h2o("O 0 0 0; h 0 0 0.578;h 0 0 -0.578", "sto-3g");
    Integral::Integral int_engine(h2o);
    int_engine.calc_int();
    YXTensor::print_tensor(int_engine.get_int2e());
    return 0;
}
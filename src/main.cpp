#include "gto/gto.hpp"
#include "integral/integral.hpp"

extern "C" {
#include <cint.h>
}
auto main() -> int
{
    GTO::Mol h2o("O 0 0 0; h 0 0 0.578;h 0 0 -0.578", "cc-pvdz");
    Integral::Integral int_engine(h2o);
    h2o.printCintInfo();
    return 0;
}
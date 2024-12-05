#include "gto/gto.hpp"
#include "hf/hf.hpp"
#include "mp2/mp2.hpp"

auto main() -> int
{
    GTO::Mol H2O("O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587", "cc-pvqz");
    MP2::MP2 mp2(H2O);
    mp2.kernel();

    return 0;
}
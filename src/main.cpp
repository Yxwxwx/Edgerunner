#include "gto/gto.hpp"
#include "hf/hf.hpp"

auto main() -> int
{
    GTO::Mol H2O("O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587", "cc-pvqz");
    HF::rhf rhf_engine(H2O, 100, 1e-7, true);
    rhf_engine.kernel();
    return 0;
}
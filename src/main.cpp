#include "gto/gto.hpp"

extern "C" {
#include <cint.h>
}
auto main() -> int {
  GTO::Mol h2o("O 0 0 0", "cc-pvdz");
  h2o.printCintInfo();
  return 0;
}
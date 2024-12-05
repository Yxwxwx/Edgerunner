#include"mp2.hpp"

namespace MP2{
    MP2::MP2(GTO::Mol& mol): hf_eng(mol){
        hf_eng.kernel();
        
        
    }

}
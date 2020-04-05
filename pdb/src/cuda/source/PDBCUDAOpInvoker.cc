#include <iostream>
#include "PDBCUDAOpInvoker.h"
namespace pdb{

void PDBCUDAOpInvoker::setInput(float* input, std::vector<size_t>& inputDim){
    std::cerr << "There is no setInput() method for base class\n";
    exit(-1);
};

void PDBCUDAOpInvoker::setOutput(float* output, std::vector<size_t>& outputDim){
    std::cerr << "There is no setOutput() method for base class\n";
    exit(-1);
};

bool PDBCUDAOpInvoker::invoke(){
    std::cerr << "There is no invoke() method for base class\n";
    exit(-1);
};

}
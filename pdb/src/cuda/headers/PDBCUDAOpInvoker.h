#ifndef PDB_CUDA_INVOKER
#define PDB_CUDA_INVOKER

#include <iostream>
#include <Handle.h>
#include <vector>
#include "PDBVector.h"
#include "PDBCUDAUtility.h"
#include "PDBCUDAOpType.h"
#include "PDBBufferManagerImpl.h"
#include "PDBCUDAMemoryManager.h"
// simply support two kind of operations
namespace pdb{

class PDBCUDAOpInvoker{
public:

    PDBCUDAOpInvoker(){};

    void setInput(float* input, std::vector<size_t>& inputDim){
        std::cout << "PDBCUDAOpInvoker setionput \n";
        return;
    };

    void setOutput(float* output, std::vector<size_t>& outputDim){
        std::cout << "PDBCUDAOpInvoker setoutput \n";
        return;
    };

    bool invoke(){
        std::cout << "PDBCUDAOpInvoker invoke \n";
        return true;
    };
public:
    PDBCUDAOpType op;
};
}
#endif
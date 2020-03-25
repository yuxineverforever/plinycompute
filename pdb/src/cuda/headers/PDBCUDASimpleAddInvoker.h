#ifndef PDB_CUDA_SA_INVOKER
#define PDB_CUDA_SA_INVOKER

#include <iostream>
#include <Handle.h>
#include <functional>
#include <numeric>
#include "PDBVector.h"
#include "PDBCUDAUtility.h"
#include "PDBCUDAOpType.h"
#include "PDBCUDAOpInvoker.h"

// simply support two kind of operations
namespace pdb{

class PDBCUDASimpleAddInvoker: public PDBCUDAOpInvoker{
    using T = float;
public:

    PDBCUDASimpleAddInvoker() = default;

    bool invoke(){
        //TODO: implement it
        return true;
    }

    void cublasRouting(T* in1data, T* in2data, T* outdata){
        //TODO: implement it
        // wait to add simple add
        return;
    }
/*
    void setStartAddress(void* allocationBlock){
        blockAddress = allocationBlock;
    }
 */

    void setInput(T* input, std::vector<size_t>& inputDim){
        //TODO: implement it
        return;
    }

    void setOutput(T* output, std::vector<size_t>& outputDim){
        //TODO: implement it
        return;
    }

public:

    std::vector<std::pair<T*, std::vector<size_t> >> InputParas;
    std::pair<T *, std::vector<size_t> > OutputPara;

    T * copyBackPara;

    PDBCUDAOpType op = PDBCUDAOpType::SimpleAdd;
};
}
#endif
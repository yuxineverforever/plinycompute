#ifndef PDB_CUDA_SA_INVOKER
#define PDB_CUDA_SA_INVOKER

#include <iostream>
#include <Handle.h>
#include "PDBVector.h"
#include "PDBCUDAUtility.h"
#include "PDBCUDAOpType.h"
#include "PDBCUDAOpInvoker.h"

// simply support two kind of operations
template <typename T>
class PDBCUDASimpleAddInvoker: public PDBCUDAOpInvoker{
public:

    bool invoke(){
        cublasRouting(GPUInputParas[0], GPUInputParas[1], GPUOutputPara);
        return true;
    }

    void cublasRouting(T* in1data, T* in2data, T* outdata){

    }

    void setStartAddress(void* allocationBlock){
        blockAddress = allocationBlock;
    }

    void setInput(T* input){
        InputParas.push_back(input);
        T* cudaPointer;
        copyFromHostToDevice(&cudapointer, input, sizeof(T));
        GPUInputParas.push_back(cudapointer);
    }

    void setOutput(T* output){
        OutputPara = output;
        T * cudaPointer;
        copyFromHostToDevice(&cudaPointer, output, sizeof(T));
        GPUOutputPara = cudaPointer;
    }

public:
    std::vector<T*> GPUInputParas;
    T * GPUOutputPara;

    std::vector<T*> InputParas;
    T* OutputPara;

    void * blockAddress;

    PDBCUDAOpType op = PDBCUDAOpType::MatrixMultiple;
};
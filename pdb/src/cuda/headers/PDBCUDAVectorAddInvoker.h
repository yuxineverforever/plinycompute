#ifndef PDB_CUDA_VA_INVOKER
#define PDB_CUDA_VA_INVOKER

#include <iostream>
#include <Handle.h>
#include "PDBVector.h"
#include "PDBCUDAUtility.h"
#include "PDBCUDAOpType.h"
#include "PDBCUDAOpInvoker.h"

// simply support two kind of operations
template <typename T>
class PDBCUDAVectorAddInvoker: public PDBCUDAOpInvoker{
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

    void setInput(T* input, size_t inputSize){
        InputParas.push_back(input);
        T* cudaPointer;
        copyFromHostToDevice(&cudapointer, input, sizeof(T)*inputSize);
        GPUInputParas.push_back(cudapointer);
    }

    void setOutput(T* output, size_t outputSize){
        OutputPara = output;
        T * cudaPointer;
        copyFromHostToDevice(&cudaPointer, output, sizeof(T)*outputSize);
        GPUOutputPara = cudaPointer;
    }

public:
    std::vector< std::pair<T*, size_t> > GPUInputParas;
    T * GPUOutputPara;

    std::vector< std::pair<T*, size_t> > InputParas;
    T* OutputPara;

    void * blockAddress;

    PDBCUDAOpType op = PDBCUDAOpType::MatrixMultiple;
};
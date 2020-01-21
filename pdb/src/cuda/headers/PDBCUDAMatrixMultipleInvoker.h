#ifndef PDB_CUDA_MM_INVOKER
#define PDB_CUDA_MM_INVOKER

#include <iostream>
#include <Handle.h>
#include "PDBVector.h"
#include "PDBCUDAUtility.h"
#include "PDBCUDAOpType.h"
#include "PDBCUDAOpInvoker.h"

// simply support two kind of operations
template <typename T>
class PDBCUDAMatrixMultipleInvoker: public PDBCUDAOpInvoker{
public:

    bool invoke(){
        cublasRouting(GPUInputParas[0],GPUInputParas[1], GPUOutputPara);
        return true;
    }

    void cublasRouting(T* in1data, T* in2data, T* outdata){
        cublasHandle_t handle;
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        checkCudaErrors(cublasCreate(&handle));
        checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in2NumCol, in1NumRow, in1NumCol, &alpha, in2data, in2NumCol, in1data, in1NumCol, &beta, outdata, in2NumCol));
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

    size_t in1NumRow;
    size_t in1NumCol;
    size_t in2NumRow;

    PDBCUDAOpType op = PDBCUDAOpType::MatrixMultiple;
};
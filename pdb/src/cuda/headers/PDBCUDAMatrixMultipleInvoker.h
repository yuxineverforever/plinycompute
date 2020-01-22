#ifndef PDB_CUDA_MM_INVOKER
#define PDB_CUDA_MM_INVOKER

#include <iostream>
#include <Handle.h>
#include <functional>
#include <numeric>
#include "PDBVector.h"
#include "PDBCUDAUtility.h"
#include "PDBCUDAOpType.h"
#include "PDBCUDAOpInvoker.h"

// simply support two kind of operations
template <typename T = float>
        class PDBCUDAMatrixMultipleInvoker: public PDBCUDAOpInvoker<T>{
public:

    bool invoke(){
        cublasRouting(InputParas[0].first, InputParas[1].first, OutputPara.first, InputParas[0].second[0],InputParas[0].second[1],InputParas[1].second[0]);
        cleanup();
        return true;
    }

    void cublasRouting(T* in1data, T* in2data, T* outdata, size_t in1NumRow, size_t in1NumCol, size_t in2NumCol){
        cublasHandle_t handle;
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        checkCudaErrors(cublasCreate(&handle));
        checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1NumRow, in2NumCol, in1NumCol, &alpha, in1data, in1NumRow, in2data, in1NumCol, &beta, outdata, in1NumRow));
        copyFromDeviceToHost((void*)copyBackPara, (void*)OutputPara.first, OutputPara.second[0]*OutputPara.second[1]);
    }

    void setInput(T* input, std::vector<size_t>& inputDim){
        T* cudaPointer;
        size_t length = std::accumulate(inputDim.begin(),inputDim.end(),1, std::multiplies<size_t>());
        copyFromHostToDevice((void**)&cudaPointer, input, sizeof(T) * length);
        InputParas.push_back(std::make_pair(cudaPointer, inputDim));
    }

    void setOutput(T* output, std::vector<size_t>& outputDim){
        T * cudaPointer;
        size_t length = std::accumulate(outputDim.begin(),outputDim.end(),1, std::multiplies<size_t>());
        copyFromHostToDevice((void**)&cudaPointer, output, sizeof(T) * length);
        OutputPara = std::make_pair(cudaPointer, outputDim);
        copyBackPara = output;
    }

    void cleanup(){
        for (auto& p : InputParas){
            freeGPUMemory((void**)&(p.first));
        }
        freeGPUMemory((void**)&OutputPara.first);
    }

public:

    std::vector<std::pair<T*, std::vector<size_t> >> InputParas;
    std::pair<T *, std::vector<size_t> > OutputPara;

    T* copyBackPara;

    PDBCUDAOpType op = PDBCUDAOpType::MatrixMultiple;
};
#endif
#ifndef PDB_CUDA_VA_INVOKER
#define PDB_CUDA_VA_INVOKER

#include <iostream>
#include <Handle.h>
#include "PDBVector.h"
#include <functional>
#include <numeric>
#include "PDBCUDAUtility.h"
#include "PDBCUDAOpType.h"
#include "PDBCUDAOpInvoker.h"

// simply support two kind of operations
template <typename T = float>
class PDBCUDAVectorAddInvoker: public PDBCUDAOpInvoker<T>{
public:

    bool invoke(){
        cublasRouting(InputParas[0].first, InputParas[1].first, OutputPara.first, InputParas[0].second[0]);
        return true;
    }

    void cublasRouting(T* in1data, T* in2data, T* outdata, size_t N){
        // wait to add vector add
        return;
    }

    void setInput(T* input, std::vector<size_t>& inputDim){
        T* cudaPointer;
        size_t length = std::accumulate(inputDim.begin(),inputDim.end(),1, std::multiplies<size_t>());
        copyFromHostToDevice(&cudaPointer, input, sizeof(T) * length);
        InputParas.push_back(std::make_pair(cudaPointer, inputDim));
    }

    void setOutput(T* output, std::vector<size_t>& outputDim){
        T* cudaPointer;
        size_t length = std::accumulate(outputDim.begin(),outputDim.end(),1, std::multiplies<size_t>());
        copyFromHostToDevice(&cudaPointer, output, sizeof(T) * length);
        OutputPara = std::make_pair(cudaPointer, outputDim);
        copyBackPara = output;
    }


public:

    // raw pointer and the dimension for the vector
    std::vector<std::pair<T*, std::vector<size_t> >> InputParas;
    std::pair<T *, std::vector<size_t> > OutputPara;

    T* copyBackPara;

    PDBCUDAOpType op = PDBCUDAOpType::VectorAdd;
};
#endif
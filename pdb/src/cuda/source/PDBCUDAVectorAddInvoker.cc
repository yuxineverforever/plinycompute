#include "PDBCUDAVectorAddInvoker.h"

namespace pdb{

    bool PDBCUDAVectorAddInvoker::invoke(){
        cublasRouting(InputParas[0].first, InputParas[1].first, OutputPara.first, InputParas[0].second[0]);
        return true;
    }

    void PDBCUDAVectorAddInvoker::cublasRouting(T* in1data, T* in2data, T* outdata, size_t N){
        // wait to add vector add
        return;
    }

    void PDBCUDAVectorAddInvoker::setInput(T* input, std::vector<size_t>& inputDim){
        T* cudaPointer;
        size_t length = std::accumulate(inputDim.begin(),inputDim.end(),1, std::multiplies<size_t>());
        copyFromHostToDevice(reinterpret_cast<void **>(&cudaPointer), input, sizeof(T) * length);
        InputParas.push_back(std::make_pair(cudaPointer, inputDim));
    }

    void PDBCUDAVectorAddInvoker::setOutput(T* output, std::vector<size_t>& outputDim){
        T* cudaPointer;
        size_t length = std::accumulate(outputDim.begin(),outputDim.end(),1, std::multiplies<size_t>());
        copyFromHostToDevice(reinterpret_cast<void **>(&cudaPointer), output, sizeof(T) * length);
        OutputPara = std::make_pair(cudaPointer, outputDim);
        copyBackPara = output;
    }
};
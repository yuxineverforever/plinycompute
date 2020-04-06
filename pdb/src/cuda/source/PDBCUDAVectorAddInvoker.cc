#include <assert.h>
#include "PDBCUDAVectorAddInvoker.h"

extern void* gpuMemoryManager;
extern cublasHandle_t cudaHandle;
namespace pdb{
    bool PDBCUDAVectorAddInvoker::invoke(){
        //std::cout << "PDBCUDAVectorAddInvoker invoke() \n";
        cublasRouting(OutputPara.first, InputParas[0].first, InputParas[0].second[0]);
        return true;
    }
    /**
     * Perform SAXPY on vector elements: outdata[] = outdata[] + in1data[];
     * @param in1data
     * @param in2data
     * @param outdata
     * @param N
     */
    void PDBCUDAVectorAddInvoker::cublasRouting(T* outdata, T* in1data, size_t N){
        const float alpha = 1.0;
        cublasSaxpy(cudaHandle, N, &alpha, in1data, 1, outdata, 1);
        copyFromDeviceToHost((void*)copyBackPara, (void*)OutputPara.first, OutputPara.second[0]* sizeof(float));
    }

    void PDBCUDAVectorAddInvoker::setInput(T* input, std::vector<size_t>& inputDim){
        //std::cout << "PDBCUDAVectorAddInvoker setInput() \n";
        assert(inputDim.size()==1);
        T* cudaPointer = (T*)((PDBCUDAMemoryManager*)gpuMemoryManager)->handleOneObject((void*)input);
        InputParas.push_back(std::make_pair(cudaPointer, inputDim));
    }

    void PDBCUDAVectorAddInvoker::setOutput(T* output, std::vector<size_t>& outputDim){
        //std::cout << "PDBCUDAVectorAddInvoker setOutput() \n";
        assert(outputDim.size()==1);
        T* cudaPointer = (T*)((PDBCUDAMemoryManager*)gpuMemoryManager)->handleOneObject((void*)output);
        OutputPara = std::make_pair(cudaPointer, outputDim);
        copyBackPara = output;
    }

    void PDBCUDAVectorAddInvoker::cleanup(){
        for (auto& p : InputParas){
            freeGPUMemory((void**)&(p.first));
        }
        freeGPUMemory((void**)&OutputPara.first);
    }
};
#include <assert.h>
#include "PDBCUDAVectorAddInvoker.h"
#include "PDBCUDATaskManager.h"

extern void* gpuMemoryManager;
extern void* gpuTaskManager;

namespace pdb{

    PDBCUDAVectorAddInvoker::PDBCUDAVectorAddInvoker(){
        auto threadInfo = ((PDBCUDATaskManager*)gpuTaskManager)->getThreadInfoFromPool();
        cudaStream = threadInfo.first;
        cudaHandle = threadInfo.second;
    }

    bool PDBCUDAVectorAddInvoker::invoke(){
        std::cout << (long) pthread_self() << " : PDBCUDAVectorAddInvoker invoke() \n";
        cublasRouting(outputPara.first, inputParas[0].first, inputParas[0].second[0]);
        cleanup();
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
        copyFromDeviceToHostAsync((void*)copyBackPara, (void*)outputPara.first, outputPara.second[0] * sizeof(float), cudaStream);
    }

    void PDBCUDAVectorAddInvoker::setInput(T* input, std::vector<size_t>& inputDim){
        std::cout << (long) pthread_self() << " : PDBCUDAVectorAddInvoker setInput() \n";
        assert(inputDim.size() == 1);
        auto PageInfo = ((PDBCUDAMemoryManager*)gpuMemoryManager)->getObjectPage((void*)input);
        auto cudaObjectPointer =((PDBCUDAMemoryManager*)gpuMemoryManager)->handleObject(PageInfo, (void*)input, cudaStream);
        inputParas.push_back(std::make_pair((T*)cudaObjectPointer, inputDim));
    }

    void PDBCUDAVectorAddInvoker::setOutput(T* output, std::vector<size_t>& outputDim){
        std::cout << (long) pthread_self() << " : PDBCUDAVectorAddInvoker setOutput() \n";
        assert(outputDim.size()==1);
        auto PageInfo = ((PDBCUDAMemoryManager*)gpuMemoryManager)->getObjectPage((void*)output);
        auto cudaObjectPointer =((PDBCUDAMemoryManager*)gpuMemoryManager)->handleObject(PageInfo, (void*)output, cudaStream);
        outputPara = std::make_pair((T*)cudaObjectPointer, outputDim);
        copyBackPara = output;
    }

    void PDBCUDAVectorAddInvoker::cleanup(){
        inputParas.clear();
    }
};
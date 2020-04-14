#include "PDBCUDAMatrixMultipleInvoker.h"
#include "PDBCUDATaskManager.h"

extern void* gpuMemoryManager;
extern void* gpuTaskManager;

namespace pdb{

    PDBCUDAMatrixMultipleInvoker::PDBCUDAMatrixMultipleInvoker(){
        auto threadInfo = ((PDBCUDATaskManager*)gpuTaskManager)->getThreadInfoFromPool();
        cudaStream = threadInfo.first;
        cudaHandle = threadInfo.second;
    }

    void PDBCUDAMatrixMultipleInvoker::setInput(T* input, std::vector<size_t>& inputDim){
        std::cout<< (long) pthread_self() << ": PDBCUDAMatrixMultipleInvoker setInput() \n";
        auto PageInfo = ((PDBCUDAMemoryManager*)gpuMemoryManager)->getObjectPage((void*)input);
        auto cudaObjectPointer =((PDBCUDAMemoryManager*)gpuMemoryManager)->handleInputObject(PageInfo, (void*)input, cudaStream);
        inputParas.push_back(std::make_pair((T*)cudaObjectPointer, inputDim));
    }

    void PDBCUDAMatrixMultipleInvoker::setOutput(T* output, std::vector<size_t>& outputDim){
        std::cout << (long) pthread_self()<< ": PDBCUDAMatrixMultipleInvoker setOutput() \n";
        auto PageInfo = ((PDBCUDAMemoryManager*)gpuMemoryManager)->getObjectPage((void*)output);

        auto cudaObjectPointer =((PDBCUDAMemoryManager*)gpuMemoryManager)->handleOutputObject(PageInfo, (void*)output, cudaStream);

        outputPara = std::make_pair((T*)cudaObjectPointer, outputDim);
        copyBackPara = output;
    }

    bool PDBCUDAMatrixMultipleInvoker::invoke(){
        std::cout << (long) pthread_self() << " :PDBCUDAMatrixMultipleInvoker invoke() \n";
        cublasRouting(inputParas[0].first, inputParas[1].first, outputPara.first, inputParas[0].second[0], inputParas[0].second[1], inputParas[1].second[0]);
        //cleanup();
        return true;
    }

    void PDBCUDAMatrixMultipleInvoker::cublasRouting(T* in1data, T* in2data, T* outdata, size_t in1NumRow, size_t in1NumCol, size_t in2NumCol){
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        cublasSgemm(cudaHandle, CUBLAS_OP_N, CUBLAS_OP_N, in1NumRow, in2NumCol, in1NumCol, &alpha, in1data, in1NumRow, in2data, in1NumCol, &beta, outdata, in1NumRow);
        copyFromDeviceToHostAsync((void*)copyBackPara, (void*)outputPara.first, outputPara.second[0] * outputPara.second[1] * sizeof(float), cudaStream);
    }

    void PDBCUDAMatrixMultipleInvoker::cleanup(){
        inputParas.clear();
    }

    /*
    void setInput(T* input, std::vector<size_t>& inputDim){
        T* cudaPointer;
        size_t length = std::accumulate(inputDim.begin(),inputDim.end(),1, std::multiplies<size_t>());
        copyFromHostToDevice((void**)&cudaPointer, input, sizeof(T) * length);
        inputParas.push_back(std::make_pair(cudaPointer, inputDim));
    }
     */

    /*
    void setOutput(T* output, std::vector<size_t>& outputDim){
        T * cudaPointer;
        size_t length = std::accumulate(outputDim.begin(),outputDim.end(),1, std::multiplies<size_t>());
        copyFromHostToDevice((void**)&cudaPointer, output, sizeof(T) * length);
        outputPara = std::make_pair(cudaPointer, outputDim);
        copyBackPara = output;
    }
     */
}



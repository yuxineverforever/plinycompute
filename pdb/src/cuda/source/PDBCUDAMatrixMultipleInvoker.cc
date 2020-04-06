#include "PDBCUDAMatrixMultipleInvoker.h"

extern void* gpuMemoryManager;
namespace pdb{

    PDBCUDAMatrixMultipleInvoker::PDBCUDAMatrixMultipleInvoker(){
        cublasCreate(&cudaHandle);
    }

    void PDBCUDAMatrixMultipleInvoker::setInput(T* input, std::vector<size_t>& inputDim){
        //std::cout << "PDBCUDAMatrixMultipleInvoker setInput() \n";
        T* cudaPointer = (T*)((PDBCUDAMemoryManager*)gpuMemoryManager)->handleOneObject((void*)input);
        inputParas.push_back(std::make_pair(cudaPointer, inputDim));
    }

    void PDBCUDAMatrixMultipleInvoker::setOutput(T* output, std::vector<size_t>& outputDim){
        //std::cout << "PDBCUDAMatrixMultipleInvoker setOutput() \n";
        T* cudaPointer = (T*)((PDBCUDAMemoryManager*)gpuMemoryManager)->handleOneObject((void*)output);
        outputPara = std::make_pair(cudaPointer, outputDim);
        copyBackPara = output;
    }

    bool PDBCUDAMatrixMultipleInvoker::invoke(){
        //std::cout << "PDBCUDAMatrixMultipleInvoker invoke() \n";
        cublasRouting(inputParas[0].first, inputParas[1].first, outputPara.first, inputParas[0].second[0], inputParas[0].second[1], inputParas[1].second[0]);
        //cleanup();
        return true;
    }

    void PDBCUDAMatrixMultipleInvoker::cublasRouting(T* in1data, T* in2data, T* outdata, size_t in1NumRow, size_t in1NumCol, size_t in2NumCol){
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        cublasSgemm(cudaHandle, CUBLAS_OP_N, CUBLAS_OP_N, in1NumRow, in2NumCol, in1NumCol, &alpha, in1data, in1NumRow, in2data, in1NumCol, &beta, outdata, in1NumRow);
        copyFromDeviceToHost((void*)copyBackPara, (void*)outputPara.first, outputPara.second[0] * outputPara.second[1] * sizeof(float));
        cleanup();
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



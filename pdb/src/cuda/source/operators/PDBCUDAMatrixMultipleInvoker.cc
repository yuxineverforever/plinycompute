#include "operators/PDBCUDAMatrixMultipleInvoker.h"
#include "stream/PDBCUDAStreamManager.h"


namespace pdb {

    PDBCUDAMatrixMultipleInvoker::PDBCUDAMatrixMultipleInvoker() {

        sstore_instance = PDBCUDAStaticStorage::get();
        memmgr_instance = PDBCUDAMemoryManager::get();
        stream_instance = PDBCUDAStreamManager::get();

        PDBCUDAStreamUtils util = stream_instance->bindCPUThreadToStream();
        cudaStream = util.first;
        cudaHandle = util.second;
    }

    void PDBCUDAMatrixMultipleInvoker::setInput(float* input, std::vector<size_t> &inputDim) {

        int isDevice = isDevicePointer((void *) input);
        if (isDevice) {
            inputArguments.push_back(std::make_pair(static_cast<float*>(input), inputDim));
        } else {
            auto cpuPageInfo = sstore_instance->getCPUPageFromObjectAddress(static_cast<void*>(input));
            auto gpuPageInfo = sstore_instance->getGPUPageFromCPUPage(cpuPageInfo);
            PDBCUDAPage* cudaPage = memmgr_instance->FetchPageImpl(gpuPageInfo.first);
            if (gpuPageInfo.second == GPUPageCreateStatus::NOT_CREATED_PAGE){
                checkCudaErrors(cudaMemcpyAsync(static_cast<void*>(cudaPage->getBytes()), cpuPageInfo.first, cpuPageInfo.second, cudaMemcpyKind::cudaMemcpyHostToDevice, cudaStream));
            }
            void* cudaObjectPointer = cudaPage->getBytes() + sstore_instance->getObjectOffsetWithCPUPage(cpuPageInfo.first, input);
            inputArguments.push_back(std::make_pair(static_cast<float*> (cudaObjectPointer), inputDim));
            inputPages.push_back(gpuPageInfo.first);
        }
    }

    void PDBCUDAMatrixMultipleInvoker::setOutput(float* output, std::vector<size_t> &outputDim) {
        // The output pointer should point to an address on GPU
        outputArguments = std::make_pair(static_cast<float *>(output), outputDim);
    }

    bool PDBCUDAMatrixMultipleInvoker::invoke() {
        //std::cout << (long) pthread_self() << " :PDBCUDAMatrixMultipleInvoker invoke() \n";
        kernel(inputArguments[0].first, inputArguments[1].first, outputArguments.first, inputArguments[0].second[0],
                      inputArguments[0].second[1], inputArguments[1].second[0]);
        cleanup();
        return true;
    }

    void
    PDBCUDAMatrixMultipleInvoker::kernel(float *in1data, float *in2data, float *outdata, size_t in1NumRow, size_t in1NumCol,
                                                size_t in2NumCol) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasErrCheck( cublasSgemm(cudaHandle, CUBLAS_OP_N, CUBLAS_OP_N, in1NumRow, in2NumCol, in1NumCol, &alpha, in1data, in1NumRow,
                                    in2data, in1NumCol, &beta, outdata, in1NumRow);
        );
    }

    void PDBCUDAMatrixMultipleInvoker::cleanup() {
        inputArguments.clear();
        for (auto pageID : inputPages){
            memmgr_instance->UnpinPageImpl(pageID,false);
        }
    }

}



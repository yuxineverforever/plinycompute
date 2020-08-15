#include <assert.h>
#include "operators/PDBCUDAVectorAddInvoker.h"
#include "stream/PDBCUDAStreamManager.h"

namespace pdb {
    PDBCUDAVectorAddInvoker::PDBCUDAVectorAddInvoker() {

        auto threadInfo = (static_cast<PDBCUDAStreamManager *>(gpuThreadManager))->getThreadInfoFromPool();

        cudaStream = threadInfo.first;
        cudaHandle = threadInfo.second;

        sstore_instance = PDBCUDAStaticStorage::get();
        memmgr_instance = PDBCUDAMemoryManager::get();
        stream_instance = PDBCUDAStreamManager::get();
    }

    bool PDBCUDAVectorAddInvoker::invoke() {
        //std::cout << (long) pthread_self() << " : PDBCUDAVectorAddInvoker invoke() \n";
        cublasRouting(outputArguments.first, inputArguments[0].first, inputArguments[0].second[0]);
        return true;
    }

    /**
     * Perform SAXPY on vector elements: outdata[] = outdata[] + in1data[];
     * @param in1data
     * @param in2data
     * @param outdata
     * @param N
     */
    void PDBCUDAVectorAddInvoker::cublasRouting(T *outdata, T *in1data, size_t N) {
        const float alpha = 1.0;
        cublasErrCheck(cublasSaxpy(cudaHandle, N, &alpha, in1data, 1, outdata, 1));
        //copyFromDeviceToHostAsync((void *) copyBackPara, (void *) outputArgument.first, outputArgument.second[0] * sizeof(float), cudaStream);
    }

    void PDBCUDAVectorAddInvoker::setInput(T *input, std::vector<size_t> &inputDim) {
        // std::cout << (long) pthread_self() << " : PDBCUDAVectorAddInvoker setInput() \n";
        // assert(inputDim.size() == 1);
        int isDevice = isDevicePointer((void *) input);
        if (isDevice) {
            inputArguments.push_back(std::make_pair(static_cast<T *>(input), inputDim));
        } else {
            auto cpuPageInfo = sstore_instance->getCPUPageFromObjectAddress(static_cast<void*>(input));
            auto gpuPageInfo = sstore_instance->getGPUPageFromCPUPage(cpuPageInfo);
            PDBCUDAPage* cudaPage = memmgr_instance->FetchPageImpl(gpuPageInfo.first);

            if (gpuPageInfo.second == GPUPageCreateStatus::NOT_CREATED_PAGE){
                auto stream_instance = PDBCUDAStreamManager::get();
                auto streamToUse = stream_instance->bindCPUThreadToStream();
                checkCudaErrors(cudaMemcpyAsync(static_cast<void*>(cudaPage->getBytes()), cpuPageInfo.first, cpuPageInfo.second, cudaMemcpyKind::cudaMemcpyHostToDevice, streamToUse.first));
            }
            void* cudaObjectPointer = cudaPage->getBytes() + sstore_instance->getObjectOffsetWithCPUPage(cpuPageInfo.first, input);
            inputArguments.push_back(std::make_pair(static_cast<T *> (cudaObjectPointer), inputDim));
            inputPages.push_back(gpuPageInfo.first);
        }
    }

    std::shared_ptr<pdb::RamPointerBase> PDBCUDAVectorAddInvoker::LazyAllocationHandler(void* pointer, size_t size){
        pair<void *, size_t> PageInfo = (static_cast<PDBCUDAMemoryManager *>(gpuMemoryManager))->getObjectCPUPage(
                (void *) pointer);
        return (static_cast<PDBCUDAMemoryManager *>(gpuMemoryManager))->handleInputObjectWithRamPointer(PageInfo, (void*)pointer, size, cudaStream);
    }

    void PDBCUDAVectorAddInvoker::setOutput(T *output, std::vector<size_t> &outputDim) {
        // The output pointer should point to an address on GPU
        outputArguments = std::make_pair(static_cast<T *>(output), outputDim);
        copyBackPara = output;
    }

    void PDBCUDAVectorAddInvoker::cleanup() {
        inputArguments.clear();
        for (auto pageID : inputPages){
            memmgr_instance->UnpinPageImpl(pageID,false);
        }
    }
};
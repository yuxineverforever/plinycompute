#include <assert.h>
#include "operators/PDBCUDAVectorAddInvoker.h"
#include "stream/PDBCUDAStreamManager.h"

namespace pdb {
    PDBCUDAVectorAddInvoker::PDBCUDAVectorAddInvoker() {
        sstore_instance = PDBCUDAStaticStorage::get();
        memmgr_instance = PDBCUDAMemoryManager::get();
        stream_instance = PDBCUDAStreamManager::get();
        PDBCUDAStreamUtils util = stream_instance->bindCPUThreadToStream();
        cudaStream = util.first;
        cudaHandle = util.second;
    }

    bool PDBCUDAVectorAddInvoker::invoke() {
        kernel(outputArguments.first, inputArguments[0].first, inputArguments[0].second[0]);
        cleanup();
        return true;
    }

    /**
     * Perform SAXPY on vector elements: outdata[] = outdata[] + in1data[];
     * @param in2data
     * @param in2data
     * @param in1data
     * @param N
     */
    void PDBCUDAVectorAddInvoker::kernel(float *in1data, float *in2data, size_t N) {
        const float alpha = 1.0;
        cublasErrCheck(cublasSaxpy(cudaHandle, N, &alpha, in2data, 1, in1data, 1));
    }

    void PDBCUDAVectorAddInvoker::setInput(float *input, std::vector<size_t> &inputDim) {
        // std::cout << (long) pthread_self() << " : PDBCUDAVectorAddInvoker setInput() \n";
        // assert(inputDim.size() == 1);
        int isDevice = isDevicePointer((void *) input);
        if (isDevice) {
            inputArguments.push_back(std::make_pair(static_cast<float *>(input), inputDim));
        } else {
            auto cpuPageInfo = sstore_instance->getCPUPageFromObjectAddress(static_cast<void*>(input));
            auto gpuPageInfo = sstore_instance->getGPUPageFromCPUPage(cpuPageInfo);

            PDBCUDAPage* cudaPage = memmgr_instance->FetchPageImpl(gpuPageInfo.first);

            if (gpuPageInfo.second == GPUPageCreateStatus::NOT_CREATED_PAGE){
                checkCudaErrors(cudaMemcpyAsync(static_cast<void*>(cudaPage->getBytes()), cpuPageInfo.first, cpuPageInfo.second, cudaMemcpyKind::cudaMemcpyHostToDevice, cudaStream));
            }
            void* cudaObjectPointer = cudaPage->getBytes() + sstore_instance->getObjectOffsetWithCPUPage(cpuPageInfo.first, input);
            inputArguments.push_back(std::make_pair(static_cast<float *> (cudaObjectPointer), inputDim));
            inputPages.push_back(gpuPageInfo.first);
        }
    }

    std::shared_ptr<pdb::RamPointerBase> PDBCUDAVectorAddInvoker::LazyAllocationHandler(void* pointer, size_t size){
    //    pair<void *, size_t> PageInfo = (static_cast<PDBCUDAMemoryManager *>(gpuMemoryManager))->getObjectCPUPage(
    //            (void *) pointer);
    //    return (static_cast<PDBCUDAMemoryManager *>(gpuMemoryManager))->handleInputObjectWithRamPointer(PageInfo, (void*)pointer, size, cudaStream);
    }

    void PDBCUDAVectorAddInvoker::setOutput(float *output, std::vector<size_t> &outputDim) {
        // The output pointer should point to an address on GPU
        outputArguments = std::make_pair(static_cast<float *>(output), outputDim);
    }

    void PDBCUDAVectorAddInvoker::cleanup() {
        inputArguments.clear();
        for (auto pageID : inputPages){
            memmgr_instance->UnpinPageImpl(pageID,false);
        }
    }
};
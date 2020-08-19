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

    PDBCUDAVectorAddInvoker::~PDBCUDAVectorAddInvoker() {

        for (auto pageID : inputPages){
            memmgr_instance->UnpinPageImpl(pageID.first,false);
        }
        for (auto pageID: outputPages){
            memmgr_instance->UnpinPageImpl(pageID.first, false);
        }
    }

    bool PDBCUDAVectorAddInvoker::invoke() {
        kernel(outputArguments.first, inputArguments[0].first, inputArguments[0].second[0]);
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
        //TODO:
        checkCudaErrors(cudaMemcpyAsync(static_cast<void*>(copyBackArgument), static_cast<void*>(outputArguments.first), outputArguments.second[0] * sizeof(float), cudaMemcpyDeviceToHost, cudaStream));
    }

    void PDBCUDAVectorAddInvoker::setInput(float *input, const std::vector<size_t> &inputDim) {

        int isDevice = isDevicePointer((void *) input);
        if (isDevice) {
            inputArguments.push_back(std::make_pair(input, inputDim));
            return;
        }

        // get CPU page for this object
        auto cpuPageInfo = sstore_instance->getCPUPageFromObjectAddress(static_cast<void*>(input));
        // get GPU page based on CPU page information
        pair<page_id_t, MemAllocateStatus> gpuPageInfo = sstore_instance->checkGPUPageTable(cpuPageInfo);

        // fetch GPU page
        PDBCUDAPage* cudaPage = memmgr_instance->FetchPageImpl(gpuPageInfo.first);

        // if page is never written, move the content from CPU page to GPU page.
        // Notice, here, the size of GPU page may be larger than CPU page. Some smart way for De-fragmentation is needed.
        if (gpuPageInfo.second == MemAllocateStatus::NEW){
            checkCudaErrors(cudaMemcpyAsync(cudaPage->getBytes(), cpuPageInfo.first, cpuPageInfo.second, cudaMemcpyKind::cudaMemcpyHostToDevice, cudaStream));
        }

        void* cudaObjectPointer = static_cast<char*>(cudaPage->getBytes()) + sstore_instance->getObjectOffsetWithCPUPage(cpuPageInfo.first, input);
        inputArguments.push_back(std::make_pair(static_cast<float*> (cudaObjectPointer), inputDim));

        // book keep the page id and the real number of bytes used
        inputPages.push_back(std::make_pair(gpuPageInfo.first, cpuPageInfo.second));
    }

    // std::shared_ptr<pdb::RamPointerBase> PDBCUDAVectorAddInvoker::LazyAllocationHandler(void* pointer, size_t size){
    //    pair<void *, size_t> PageInfo = (static_cast<PDBCUDAMemoryManager *>(gpuMemoryManager))->getObjectCPUPage(
    //            (void *) pointer);
    //    return (static_cast<PDBCUDAMemoryManager *>(gpuMemoryManager))->handleInputObjectWithRamPointer(PageInfo, (void*)pointer, size, cudaStream);
    // }

    void PDBCUDAVectorAddInvoker::setOutput(float *output, const std::vector<size_t> & outputDim) {

        int isDevice = isDevicePointer((void *) output);
        if (isDevice) {
            outputArguments.first = output;
            outputArguments.second = outputDim;
            return;
        }

        auto cpuPageInfo = sstore_instance->getCPUPageFromObjectAddress(static_cast<void*>(output));

        pair<page_id_t, MemAllocateStatus> gpuPageInfo = sstore_instance->checkGPUPageTable(cpuPageInfo);

        PDBCUDAPage* cudaPage = memmgr_instance->FetchPageImpl(gpuPageInfo.first);

        void* cudaObjectPointer = static_cast<char*>(cudaPage->getBytes()) + sstore_instance->getObjectOffsetWithCPUPage(cpuPageInfo.first, output);

        outputArguments = std::make_pair(static_cast<float *>(cudaObjectPointer), outputDim);

        outputPages.push_back(std::make_pair(gpuPageInfo.first, cpuPageInfo.second));

        // TODO: this should be removed in the future.
        copyBackArgument = output;
    }

};
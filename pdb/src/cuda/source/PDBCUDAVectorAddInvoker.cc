#include <assert.h>
#include "PDBCUDAVectorAddInvoker.h"
#include "PDBCUDATaskManager.h"

extern void *gpuMemoryManager;
extern void *gpuTaskManager;

namespace pdb {



    PDBCUDAVectorAddInvoker::PDBCUDAVectorAddInvoker() {
        auto threadInfo = (static_cast<PDBCUDATaskManager *>(gpuTaskManager))->getThreadInfoFromPool();
        cudaStream = threadInfo.first;
        cudaHandle = threadInfo.second;
    }
    bool PDBCUDAVectorAddInvoker::invoke() {
        //std::cout << (long) pthread_self() << " : PDBCUDAVectorAddInvoker invoke() \n";
        cublasRouting(outputPara.first, inputParas[0].first, inputParas[0].second[0]);
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
        cublasSaxpy(cudaHandle, N, &alpha, in1data, 1, outdata, 1);
        //copyFromDeviceToHostAsync((void *) copyBackPara, (void *) outputPara.first, outputPara.second[0] * sizeof(float), cudaStream);
    }

    void PDBCUDAVectorAddInvoker::setInput(T *input, std::vector<size_t> &inputDim) {
        //std::cout << (long) pthread_self() << " : PDBCUDAVectorAddInvoker setInput() \n";
        assert(inputDim.size() == 1);
        int isDevice = isDevicePointer((void *) input);
        if (isDevice) {
            inputParas.push_back(std::make_pair((T *) input, inputDim));
        } else {
            auto PageInfo = (static_cast<PDBCUDAMemoryManager *>(gpuMemoryManager))->getObjectPage((void *) input);
            auto cudaObjectPointer = (static_cast<PDBCUDAMemoryManager *>(gpuMemoryManager))->handleInputObject(PageInfo,
                                                                                                        (void *) input,
                                                                                                        cudaStream);
            inputParas.push_back(std::make_pair((T *) cudaObjectPointer, inputDim));
        }
    }

    std::shared_ptr<pdb::RamPointerBase> PDBCUDAVectorAddInvoker::LazyAllocationHandler(void* pointer, size_t size){
        pair<void *, size_t> PageInfo = (static_cast<PDBCUDAMemoryManager *>(gpuMemoryManager))->getObjectPage((void *)pointer);
        return (static_cast<PDBCUDAMemoryManager *>(gpuMemoryManager))->handleInputObjectWithRamPointer(PageInfo, (void*)pointer, cudaStream);
    }

    void PDBCUDAVectorAddInvoker::setOutput(T *output, std::vector<size_t> &outputDim) {
        assert(outputDim.size() == 1);
        // NOTE: the output pointer should point to an address on GPU
        outputPara = std::make_pair((T *) output, outputDim);
        copyBackPara = output;
    }


    void PDBCUDAVectorAddInvoker::cleanup() {
        inputParas.clear();
    }
};
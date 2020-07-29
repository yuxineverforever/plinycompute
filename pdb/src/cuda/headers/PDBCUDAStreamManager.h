#pragma once

#include <iostream>
#include <stdint.h>
#include <list>
#include "ReaderWriterLatch.h"
#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "cublas_v2.h"
#include "PDBCUDAConfig.h"

namespace pdb {

    //TODO: add some way to make thread decoupled from specific task
    // I will first try to manually manage it.

    using PDBCUDAStreamUtils = std::pair<cudaStream_t*, cublasHandle_t*>;

    class PDBCUDAStreamManager {
    public:

        PDBCUDAStreamManager(uint32_t streamNumInPool = CUDA_STREAM_NUM, bool isManager = false);

        ~PDBCUDAStreamManager();

        const PDBCUDAStreamUtils getUnUsedStream();
        void releaseUsedStream(const PDBCUDAStreamUtils& toRelease);

        static void create();
        static PDBCUDAStreamManager* get();
        static inline bool check();

    private:

        static PDBCUDAStreamManager* streamMgr;
        cudaStream_t *streams;
        cublasHandle_t *handles;
        uint32_t streamNum;
        std::mutex m;
        std::list<PDBCUDAStreamUtils> unUsedStreams;
        //std::map<long, uint64_t> threadStreamMap;
    };
}

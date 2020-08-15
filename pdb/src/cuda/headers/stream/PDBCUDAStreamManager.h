#pragma once

#include <iostream>
#include <stdint.h>
#include <list>
#include "thread"
#include "ReaderWriterLatch.h"
#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "cublas_v2.h"
#include "PDBCUDAConfig.h"

namespace pdb {

    //TODO: I will first try to manually manage it.
    //TODO: I will try to make it thread safe.
    /**
     * The design principle for StreamUtil is mainly serving for PDB worker thread.
     * Based on the fact that, we dont have ability to explicit add cuda barrier/sync methanism to PDB.
     * It is better to keep the each PDB worker thread only mapped to one cuda stream. So that a thread can only manipulate one stream and push all the operations to that stream.
     * This can make sure that, there is no need to explicit sync between stream. (e.g. one stream for a different UDF)
     */
    using PDBCUDAStreamUtils = std::pair<cudaStream_t*, cublasHandle_t*>;

    class PDBCUDAStreamManager {

    public:

        PDBCUDAStreamManager(uint32_t streamNumInPool = CUDA_STREAM_NUM, bool isManager = false);

        ~PDBCUDAStreamManager();

        const PDBCUDAStreamUtils getUnUsedStream();

        void releaseUsedStream(const PDBCUDAStreamUtils& toRelease);

        static const PDBCUDAStreamUtils bindCPUThreadToStream(std::thread::id& tID);

        static void create();

        static PDBCUDAStreamManager* get();

        static inline bool check();

    private:

        static PDBCUDAStreamManager* streamMgr;

        static std::once_flag initFlag;

        static std::map<std::thread::id, PDBCUDAStreamUtils> bindMap;

        cudaStream_t* streams;

        cublasHandle_t* handles;

        uint32_t streamNum;
        std::mutex m;

        std::list<PDBCUDAStreamUtils> unUsedStreams;

    };
}

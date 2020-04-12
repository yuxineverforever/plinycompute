#pragma once

#include <iostream>
#include <stdint.h>
#include <pthread.h>
#include "PDBCUDAUtility.h"
#include <map>
#include "threadSafeMap.h"

namespace pdb{

    using PDBCUDAThreadInfo = std::pair < cudaStream_t, cublasHandle_t >;

    class PDBCUDATaskManager{
    public:

        PDBCUDATaskManager();

        ~PDBCUDATaskManager();

        PDBCUDATaskManager(int32_t streamNum);

        PDBCUDAThreadInfo getThreadInfoFromPool();

    private:

        /**
         *  streams and handles
         */
        cudaStream_t *streams;
        cublasHandle_t * handles;

        /**
         * number of threads
         */
        int32_t  threadNum;

        /**
         * mutex for protection
         */
         std::mutex m;

        /**
         * mapping the cpu thread id to gpu stream id / handle id
         */
         threadSafeMap<long, uint64_t> threadStreamMap;
    };
}

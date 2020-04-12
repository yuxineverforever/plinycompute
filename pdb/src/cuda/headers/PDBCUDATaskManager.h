#include <iostream>
#include <stdint.h>
#include <pthread.h>
#include "PDBCUDAUtility.h"
#include <map>

namespace pdb{

    using PDBCUDAThreadInfo = std::pair < cudaStream_t, cublasHandle_t >;

    class PDBCUDATaskManager{
    public:

        PDBCUDATaskManager();

        ~PDBCUDATaskManager();

        PDBCUDATaskManager(int32_t streamNum);

        PDBCUDAThreadInfo getThreadInfoFromPool();

    private:
        cudaStream_t *streams;
        cublasHandle_t * handles;
        int32_t  threadNum;

        /**
         * mapping the cpu thread id to gpu stream id / handle id
         */
        std::map<long, int32_t > threadStreamMap;

    };
}

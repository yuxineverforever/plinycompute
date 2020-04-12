#include "PDBCUDATaskManager.h"

namespace pdb{

        PDBCUDATaskManager::PDBCUDATaskManager(int32_t NumOfthread): threadNum(NumOfthread){

            streams = new cudaStream_t[threadNum+1];
            handles = new cublasHandle_t[threadNum+1];
            for (int32_t i=0; i<threadNum+1; i++){
                    checkCudaErrors(cudaStreamCreate(&streams[i]));
                    checkCudaErrors(cublasCreate(&handles[i]));
            }
        }

        PDBCUDATaskManager::~PDBCUDATaskManager(){
            for (int i=0; i<threadNum+1; i++)
            {
                cudaStreamDestroy(streams[i]);
                cublasDestroy(handles[i]);
            }
        }

        PDBCUDAThreadInfo PDBCUDATaskManager::getThreadInfoFromPool(){
             uint64_t  counter = threadStreamMap.size();
             long threadID = (long) pthread_self();
             threadStreamMap[threadID] = counter;
             return std::make_pair(streams[counter], handles[counter]);
        }
}

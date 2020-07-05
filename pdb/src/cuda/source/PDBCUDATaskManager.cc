#include "PDBCUDATaskManager.h"

namespace pdb {

    PDBCUDATaskManager::PDBCUDATaskManager(uint32_t threadNum, bool isManager) : streamNum(threadNum) {
        if (isManager) {
            return;
        }
        streams = new cudaStream_t[threadNum];
        handles = new cublasHandle_t[threadNum];
        for (int32_t i = 0; i < streamNum; i++) {
            checkCudaErrors(cudaStreamCreate(&streams[i]));
            checkCudaErrors(cublasCreate(&handles[i]));
            checkCudaErrors(cublasSetStream(handles[i], streams[i]));
        }
    }

    PDBCUDATaskManager::~PDBCUDATaskManager() {
        for (int i = 0; i <streamNum; i++) {
            cudaStreamDestroy(streams[i]);
            cublasDestroy(handles[i]);
        }
    }

    PDBCUDAThreadInfo PDBCUDATaskManager::getThreadInfoFromPool() {

        long threadID = (long) pthread_self();
        //std::cout << "thread ID: " << threadID << std::endl;

        std::unique_lock<std::mutex> lock(m);
        if (threadStreamMap.count(threadID) != 0) {
            //std::cout << "thread ID: " << threadID << " find in map! stream: " << streams[threadStreamMap[threadID]] << std::endl;
            return std::make_pair(streams[threadStreamMap[threadID]], handles[threadStreamMap[threadID]]);
        } else {
            uint64_t counter = threadStreamMap.size();
            threadStreamMap.insert(std::make_pair(threadID, counter));
            //std::cout << "thread ID: " << threadID << " not find in map stream: " << streams[counter] << std::endl;
            return std::make_pair(streams[counter], handles[counter]);
        }
    }
}

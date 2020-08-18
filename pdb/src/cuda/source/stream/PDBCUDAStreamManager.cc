#include <assert.h>
#include "stream/PDBCUDAStreamManager.h"

namespace pdb {

    PDBCUDAStreamManager::PDBCUDAStreamManager(uint32_t streamNumInPool, bool isManager) : streamNum(streamNumInPool) {
        if (isManager) {
            return;
        }
        streams = new cudaStream_t[streamNumInPool];
        handles = new cublasHandle_t[streamNumInPool];
        for (uint32_t i = 0; i < streamNum; i++) {
            checkCudaErrors(cudaStreamCreate(&streams[i]));
            checkCudaErrors(cublasCreate(&handles[i]));
            checkCudaErrors(cublasSetStream(handles[i], streams[i]));
        }
    }

    PDBCUDAStreamManager::~PDBCUDAStreamManager() {
        for (int i = 0; i < streamNum; i++) {
            cudaStreamDestroy(streams[i]);
            cublasDestroy(handles[i]);
        }
    }

    PDBCUDAStreamUtils PDBCUDAStreamManager::bindCPUThreadToStream() {
        long threadID = (long) pthread_self();
        if (bindMap.count(threadID) != 0) {
            //std::cout << "thread ID: " << threadID << " find in map! stream: " << streams[threadStreamMap[threadID]] << std::endl;
            return std::make_pair(streams[bindMap[threadID]], handles[bindMap[threadID]]);
        } else {
            uint64_t counter = bindMap.size();
            bindMap.insert(std::make_pair(threadID, counter));
            //std::cout << "thread ID: " << threadID << " not find in map stream: " << streams[counter] << std::endl;
            return std::make_pair(streams[counter], handles[counter]);
        }
    }

    void PDBCUDAStreamManager::create(){
        streamMgr = new PDBCUDAStreamManager;
    }

    PDBCUDAStreamManager* PDBCUDAStreamManager::get(){
        // use std::call_once to make sure the singleton initialization is thread-safe
        std::call_once(initFlag, PDBCUDAStreamManager::create);
        assert(check()==true);
        return streamMgr;
    }

    inline bool PDBCUDAStreamManager::check(){
        return streamMgr != nullptr;
    }
}

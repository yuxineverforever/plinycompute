#include <assert.h>
#include "PDBCUDAStreamManager.h"

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
            unUsedStreams.push_back(std::make_pair(&streams[i], &handles[i]));
        }
    }

    PDBCUDAStreamManager::~PDBCUDAStreamManager() {
        for (int i = 0; i < streamNum; i++) {
            cudaStreamDestroy(streams[i]);
            cublasDestroy(handles[i]);
        }
    }

    const PDBCUDAStreamUtils PDBCUDAStreamManager::getUnUsedStream() {
        if (unUsedStreams.size() == 0){
            //TODO: someway to wait for available resouces
            std::cerr << "ERROR: No stream available in stream manager!\n";
        }
        PDBCUDAStreamUtils stream = unUsedStreams.front();
        unUsedStreams.pop_front();
        return stream;
    }

    void PDBCUDAStreamManager::releaseUsedStream(const PDBCUDAStreamUtils& toRelease){
        unUsedStreams.push_back(toRelease);
    }

    void PDBCUDAStreamManager::create(){
        streamMgr = new PDBCUDAStreamManager;
    }

    PDBCUDAStreamManager* PDBCUDAStreamManager::get(){
        assert(check()==true);
        return streamMgr;
    }

    inline bool PDBCUDAStreamManager::check(){
        return streamMgr != nullptr
    }



}

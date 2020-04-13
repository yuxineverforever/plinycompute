#include "PDBCUDATaskManager.h"

namespace pdb{

        PDBCUDATaskManager::PDBCUDATaskManager(int32_t NumOfthread): threadNum(NumOfthread){
            streams = new cudaStream_t[2*threadNum+1];
            handles = new cublasHandle_t[2*threadNum+1];
            for (int32_t i = 0; i < 2*threadNum+1; i++){
                checkCudaErrors(cudaStreamCreate(&streams[i]));
                checkCudaErrors(cublasCreate(&handles[i]));
                checkCudaErrors(cublasSetStream(handles[i], streams[i]));
            }
        }

        PDBCUDATaskManager::~PDBCUDATaskManager(){
            for (int i=0; i < 2*threadNum + 1; i++){
                cudaStreamDestroy(streams[i]);
                cublasDestroy(handles[i]);
            }
        }

        PDBCUDAThreadInfo PDBCUDATaskManager::getThreadInfoFromPool(){

             long threadID = (long) pthread_self();
             //std::cout << "thread ID: " << threadID << std::endl;

             std::unique_lock<std::mutex> lock(m);
             if (threadStreamMap.count(threadID) != 0){
                 std::cout << "thread ID: " << threadID << " find in map! stream: " << streams[threadStreamMap[threadID]] << std::endl;
                 return std::make_pair(streams[threadStreamMap[threadID]], handles[threadStreamMap[threadID]]);
             } else {
                 uint64_t counter = threadStreamMap.length();
                 threadStreamMap.insert(threadID,counter);
                 std::cout << "thread ID: " << threadID << " not find in map stream: " << streams[counter] << std::endl;
                 return std::make_pair(streams[counter], handles[counter]);
             }
        }
}

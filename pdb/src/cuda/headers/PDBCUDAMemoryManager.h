#ifndef PDB_CUDA_MEM_MANAGER
#define PDB_CUDA_MEM_MANAGER

#include <map>
#include <list>
#include <PDBBufferManagerInterface.h>
#include "PDBCUDAUtility.h"
#include "PDBCUDALatch.h"
#include "threadSafeMap.h"
#include <assert.h>

namespace pdb {

class PDBCUDAMemoryManager;
using PDBCUDAMemoryManagerPtr = std::shared_ptr<PDBCUDAMemoryManager>;
class PDBCUDAMemoryManager{

    using frame_id_t = int32_t;

    public:

        /**
         *
         * @param buffer
         */
        PDBCUDAMemoryManager(PDBBufferManagerInterfacePtr buffer, int32_t NumOfthread, bool isManager) {
            if (isManager){
                return;
            }

            bufferManager = buffer;
            poolSize = 2 * NumOfthread;
            pageSize = buffer->getMaxPageSize();
            for (size_t i = 0; i < poolSize; i++){
                void* cudaPointer;
                cudaMalloc((void**)&cudaPointer, pageSize);
                availablePosition.push_back(cudaPointer);
                freeList.push_back(static_cast<int32_t>(i));
            }
        }

        /**
         *
         */
        ~PDBCUDAMemoryManager(){

            for (auto & iter: gpuPageTable.tsMap){
                freeGPUMemory(&iter.second);
            }
            for (auto & frame: freeList){
                cudaFree(availablePosition[frame]);
            }
        }

        /**
         *
         * @param pageAddress
         * @param objectAddress
         * @return
         */
        size_t getObjectOffset(void* pageAddress, void* objectAddress){
            return (char*)objectAddress - (char*)pageAddress;
        }

        /**
         *
         * @param objectAddress
         * @return
         */
        pair<void*, size_t> getObjectPage(void *objectAddress){
            pdb::PDBPagePtr whichPage = bufferManager->getPageForObject(objectAddress);
            if (whichPage == nullptr){
                std::cout << "getObjectOffset: cannot get page for this object!\n";
                exit(-1);
            }
            void* pageAddress = whichPage->getBytes();
            size_t pageBytes = whichPage->getSize();
            auto pageInfo = std::make_pair(pageAddress, pageBytes);
            return pageInfo;
        }

        /**
         *
         * @param pageInfo
         * @param objectAddress
         * @return
         */
        void* handleInputObject(pair<void*, size_t> pageInfo, void *objectAddress, cudaStream_t cs) {
            size_t cudaObjectOffset = getObjectOffset(pageInfo.first, objectAddress);

            std::cout << (long) pthread_self() << " : pageInfo: " << pageInfo.first << "bytes: "<< pageInfo.second << std::endl;

            std::unique_lock<std::mutex> lock(pageTableMutex);

            if (gpuPageTable.count(pageInfo) != 0) {

                return (void*)((char *)(gpuPageTable[pageInfo]) + cudaObjectOffset);

            } else {

                void* cudaPointer;

                copyFromHostToDeviceAsync((void **) &cudaPointer, pageInfo.first, pageInfo.second, cs);

                gpuPageTable.insert(pageInfo, cudaPointer);

                return (void*)((char*)cudaPointer + cudaObjectOffset);
            }

        }


        void* handleOutputObject(pair<void*, size_t> pageInfo, void *objectAddress, cudaStream_t cs){

            size_t cudaObjectOffset = getObjectOffset(pageInfo.first, objectAddress);

            std::unique_lock<std::mutex> lock(pageTableMutex);
            if (gpuPageTable.count(pageInfo) != 0) {
                return (void*)((char *)(gpuPageTable[pageInfo]) + cudaObjectOffset);
            } else {
                assert(pageInfo.second == pageSize);
                frame_id_t frame;
                if (!freeList.empty()){
                    frame = freeList.front();
                    freeList.pop_front();
                } else {
                    std::cerr << "freeList is empty!\n";
                }
                gpuPageTable.insert(pageInfo, availablePosition[frame]);
                return (void*)((char*)availablePosition[frame] + cudaObjectOffset);
            }
        }

        /**
         *
         * @param pageInfo
         * @return
         */
        void* getCUDAPage(pair<void*, size_t> pageInfo){
            if (gpuPageTable.count(pageInfo) == 0){
                std::cout << "getCUDAPage: cannot get CUDA page for this CPU Page!\n";
                exit(-1);
            }
            return gpuPageTable[pageInfo];
        }

    public:
         /**
          * the buffer manager to help maintain gpu page table
          */
         PDBBufferManagerInterfacePtr bufferManager;

        /**
         * gpu_page_table for mapping CPU bufferManager page address to GPU bufferManager page address
         */
         //std::map<pair<void*,size_t>, void*> gpuPageTable;
         threadSafeMap < pair<void*, size_t>, void*> gpuPageTable;

        /**
         * one mutex to protect the gpuPageTable access
         */
         std::mutex pageTableMutex;

         /**
          * the size of the pool
          */
         size_t poolSize;

         /**
          * the size of the page in gpu Buffer Manager
          */
          size_t pageSize;

         /**
          * positions for holding the memory
          */
         std::vector<void*> availablePosition;

         /**
          * Frames for holding the free memory
          */
          std::list<frame_id_t> freeList;
    };

}

#endif
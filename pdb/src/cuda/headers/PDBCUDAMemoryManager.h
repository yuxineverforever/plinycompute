#ifndef PDB_CUDA_MEM_MANAGER
#define PDB_CUDA_MEM_MANAGER

#include <map>
#include <list>
#include <set>
#include <PDBBufferManagerInterface.h>
#include "PDBCUDAUtility.h"
#include "threadSafeMap.h"
#include <assert.h>
#include "PDBRamPointer.h"

namespace pdb {

class PDBCUDAMemoryManager;

using PDBCUDAMemoryManagerPtr = std::shared_ptr<PDBCUDAMemoryManager>;
using frame_id_t  = int32_t;

class PDBCUDAMemoryManager{

    public:
        /**
         *
         * @param buffer
         */
        PDBCUDAMemoryManager(PDBBufferManagerInterfacePtr buffer, int32_t NumOfthread, bool isManager) {
            if (isManager){
                return;
            }
            clock_hand = 0;
            bufferManager = buffer;
            poolSize = NumOfthread + 2;
            pageSize = buffer->getMaxPageSize();
            for (size_t i = 0; i < poolSize; i++){
                void* cudaPointer;
                cudaMalloc((void**)&cudaPointer, pageSize);
                availablePosition.push_back(cudaPointer);
                freeList.push_back(static_cast<int32_t>(i));
                recentlyUsed.push_back(false);
            }
        }

        /**
         *
         */
        ~PDBCUDAMemoryManager(){
            for (auto & iter: gpuPageTable){
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
            //std::cout << (long) pthread_self() << " : pageInfo: " << pageInfo.first << "bytes: "<< pageInfo.second << std::endl;
            std::unique_lock<std::mutex> lock(pageTableMutex);
            if (gpuPageTable.count(pageInfo) != 0) {
                return (void*)((char *)(gpuPageTable[pageInfo]) + cudaObjectOffset);
            } else {
                //std::cout << (long) pthread_self() << " handleInputObject cannot find the input page ! copy it! \n";
                void* cudaPointer;
                copyFromHostToDeviceAsync((void **) &cudaPointer, pageInfo.first, pageInfo.second, cs);
                gpuPageTable.insert(std::make_pair(pageInfo, cudaPointer));
                return (void*)((char*)cudaPointer + cudaObjectOffset);
            }
        }


        void* handleOutputObject(pair<void*, size_t> pageInfo, void *objectAddress, cudaStream_t cs){
            size_t cudaObjectOffset = getObjectOffset(pageInfo.first, objectAddress);
            long threadID = (long) pthread_self();
            std::unique_lock<std::mutex> lock(pageTableMutex);
            if (gpuPageTable.count(pageInfo) != 0) {
                recentlyUsed[framePageTable[pageInfo]] = true;
                //std::cout << "thread ID :" << threadID <<" frame : " << framePageTable[pageInfo] << " has been used recently! \n";
                return (void*)((char *)(gpuPageTable[pageInfo]) + cudaObjectOffset);
            } else {
                assert(pageInfo.second == pageSize);
                frame_id_t frame = getAvailableFrame();
                gpuPageTable.insert(std::make_pair(pageInfo, availablePosition[frame]));
                framePageTable.insert(std::make_pair(pageInfo, frame));
                return (void*)((char*)availablePosition[frame] + cudaObjectOffset);
            }
        }

        void* memMalloc(size_t memSize){
            if (allocatorPage == -1){
                frame_id_t oneframe = getAvailableFrame();
                bytesUsed = 0;
                allocatorPage = oneframe;
            }
            if (memSize > (pageSize - bytesUsed)){
                std::cerr << "Unable to allocator space : the space on page is not enough!\n";
            }
            size_t start = bytesUsed;
            bytesUsed += memSize;
            return (void*)((char*)availablePosition[allocatorPage] + start);
        }

        RamPointerReference addRamPointerCollection(void* gpuaddress, void* cpuaddress, size_t numbytes, size_t headerbytes){

            RamPointer pt(gpuaddress, numbytes, headerbytes);
            auto iter = ramPointerCollection.find(pt);

            if (iter != ramPointerCollection.end()){
                ramPointerCollection[pt].push_back(cpuaddress);
                return std::make_shared<RamPointer>(iter->first);
            }else {
                std::vector<void*> newList;
                newList.push_back(cpuaddress);
                ramPointerCollection.insert(std::make_pair(pt, newList));
                return std::make_shared<RamPointer>(ramPointerCollection.find(pt)->first);
            }
        }

        frame_id_t getAvailableFrame(){
            frame_id_t frame;
            if (!freeList.empty()){
                frame = freeList.front();
                freeList.pop_front();
                recentlyUsed[frame] = true;
                return frame;
            } else {
               while(recentlyUsed[clock_hand] == true){
                   recentlyUsed[clock_hand] = false;
                   incrementIterator(clock_hand);
               }
               recentlyUsed[clock_hand] = true;
               frame = clock_hand;
               incrementIterator(clock_hand);
               auto iter = std::find_if(framePageTable.begin(), framePageTable.end(),[&](const std::pair< pair<void*, size_t>, frame_id_t> &pair){
                    return pair.second == frame;
               });
               if (iter->second < 0 || iter->second > poolSize){
                    std::cerr << " frame number is wrong! \n";
               }
               framePageTable.erase(iter);
               gpuPageTable.erase(iter->first);
               return frame;
            }
        }

        void DeepCopy(void* startLoc, size_t numBytes){

            for (auto & ramPointerPair : ramPointerCollection){
                for (void* cpuRamPointer: ramPointerPair.second){
                    if (cpuRamPointer >= startLoc && cpuRamPointer < ((char*)startLoc + numBytes)){
                        copyFromDeviceToHost(cpuRamPointer, ramPointerPair.first.ramAddress, ramPointerPair.first.numBytes);
                        Array<Nothing>* array = (Array<Nothing>*)((char*)cpuRamPointer - ramPointerPair.first.headerBytes);
                        array->setRamPointerReferenceToNull();
                    }
                }
            }
        }

    private:
        void incrementIterator(int32_t& it){
            if (++it == poolSize){
                it = 0;
            }
            return;
        }

    private:

         /**
          * the buffer manager to help maintain gpu page table
          */
         PDBBufferManagerInterfacePtr bufferManager;

        /**
         * gpu_page_table for mapping CPU bufferManager page address to GPU bufferManager page address
         */
         //std::map<pair<void*,size_t>, void*> gpuPageTable;
         std::map < pair<void*, size_t>, void*> gpuPageTable;

         /**
          * framePageTable for mapping CPU bufferManager page address to GPU frame.
          */
         std::map< pair<void*, size_t>, frame_id_t > framePageTable;


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

          /**
           * True if the frame has recently been used
           */
          std::vector<bool> recentlyUsed;

          /**
           * Clock hand for mimic a LRU algorithm
           */
          int32_t clock_hand;


          /**
           * ============================================== Here is the part for mem allocator ==============================================
           */
          size_t bytesUsed = 0;

          frame_id_t allocatorPage = -1;

          std::map<pdb::RamPointer, std::vector<void*> > ramPointerCollection;
    };
}

#endif
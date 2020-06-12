#ifndef PDB_CUDA_MEM_MANAGER
#define PDB_CUDA_MEM_MANAGER

#include <map>
#include <list>
#include <set>
#include <assert.h>
#include <PDBBufferManagerInterface.h>
#include "PDBCUDAUtility.h"
#include "PDBRamPointer.h"
#include "ReaderWriterLatch.h"

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


        // TODO: replace the pageTableMutex to ReadWriteLock
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

        // TODO: replace the pageTableMutex to ReadWriteLock
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

            if (allocatorPages.size() == 0 && currFrame == -1){
                frame_id_t oneframe = getAvailableFrame();
                bytesUsed = 0;
                allocatorPages.push_back(oneframe);
                currFrame = oneframe;
            }

            if (memSize > (pageSize - bytesUsed)){
                frame_id_t oneframe = getAvailableFrame();
                bytesUsed = 0;
                allocatorPages.push_back(oneframe);
                currFrame = oneframe;
            }
            size_t start = bytesUsed;
            bytesUsed += memSize;
            return (void*)((char*)availablePosition[currFrame] + start);
        }

        RamPointerReference addRamPointerCollection(void* gpuaddress, void* cpuaddress, size_t numbytes, size_t headerbytes){
            mutex.WLock();
            if (ramPointerCollection.count(gpuaddress) != 0){
                ramPointerCollection[gpuaddress]->push_back_pointer(cpuaddress);
                mutex.WUnlock();
                return std::make_shared<RamPointerBase>(ramPointerCollection[gpuaddress]);
            } else {
                RamPointerPtr ptr = std::make_shared<RamPointer>(gpuaddress, numbytes, headerbytes);
                ptr->push_back_pointer(cpuaddress);
                ramPointerCollection[gpuaddress] = ptr;
                mutex.WUnlock();
                return std::make_shared<RamPointerBase>(ptr);
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
            for (auto& ramPointerPair : ramPointerCollection){
                for (void* cpuPointer: ramPointerPair.second->cpuPointers){
                    if (cpuPointer >= startLoc && cpuPointer < ((char*)startLoc + numBytes)){
                        copyFromDeviceToHost(cpuPointer, ramPointerPair.second->ramAddress, ramPointerPair.second->numBytes);
                        // TODO: here exist a better way
                        Array<Nothing>* array = (Array<Nothing>*)((char*)cpuPointer - ramPointerPair.second->headerBytes);
                        array->setRamPointerReferenceToNull();
                    }
                }
            }
        }

        /**
        void swapPage(){

            PDBPageHandle cpuPage = bufferManager->getPage();

            cpuPage->getBytes();

        }
        */


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

         //TODO: replace the pageTableMutex to ReadWriteLock
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

          ReaderWriterLatch mutex{};

          size_t bytesUsed = 0;

          frame_id_t currFrame = -1;

          std::vector<frame_id_t> allocatorPages;

          /**
           * This is a map between gpu memory address and the RamPointer object.
           * It keeps all the ramPointers we create using the RamPointerPtr
           */
          std::map<void*, pdb::RamPointerPtr> ramPointerCollection;
    };
}

#endif
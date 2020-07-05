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
    using page_id_t = int32_t;
    using frame_id_t  = int32_t;

    class PDBCUDAMemoryManager {
    public:

        /**
         *
         * @param buffer
         */
        PDBCUDAMemoryManager(PDBBufferManagerInterfacePtr buffer, int32_t NumOfthread, bool isManager) {
            if (isManager) {
                return;
            }
            clock_hand = 0;
            bufferManager = buffer;
            poolSize = NumOfthread + 2;
            pageSize = buffer->getMaxPageSize();
            for (size_t i = 0; i < poolSize; i++) {
                void *cudaPointer;
                cudaMalloc((void **) &cudaPointer, pageSize);
                availablePosition.push_back(cudaPointer);
                freeList.push_back(static_cast<int32_t>(i));
                recentlyUsed.push_back(false);
            }
        }

        /**
         *
         */
        ~PDBCUDAMemoryManager() {
            for (auto &iter: PageTable) {
                freeGPUMemory(&iter.second);
            }
            for (auto &frame: freeList) {
                cudaFree(availablePosition[frame]);
            }
        }

        /**
         *
         * @param pageAddress
         * @param objectAddress
         * @return
         */
        size_t getObjectOffset(void *pageAddress, void *objectAddress) {
            return (char *) objectAddress - (char *) pageAddress;
        }

        /**
         *
         * @param objectAddress
         * @return
         */
        pair<void *, size_t> getObjectPage(void *objectAddress) {
            pdb::PDBPagePtr whichPage = bufferManager->getPageForObject(objectAddress);
            if (whichPage == nullptr) {
                std::cout << "getObjectOffset: cannot get page for this object!\n";
                exit(-1);
            }
            void *pageAddress = whichPage->getBytes();
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
        void* handleInputObject(pair<void *, size_t> pageInfo, void *objectAddress, cudaStream_t cs) {
            //std::cout << (long) pthread_self() << " : pageInfo: " << pageInfo.first << "bytes: "<< pageInfo.second << std::endl;
            size_t cudaObjectOffset = getObjectOffset(pageInfo.first, objectAddress);
            if (PageTable.find(pageInfo) != PageTable.end()) {
                pageTableMutex.RLock();
                void *cudaObjectAddress = static_cast<char *>(PageTable[pageInfo]) + cudaObjectOffset;
                pageTableMutex.RUnlock();
                return cudaObjectAddress;
            } else {
                pageTableMutex.WLock();
                if (PageTable.find(pageInfo) != PageTable.end()) {
                    void *cudaObjectAddress = static_cast<char *>(PageTable[pageInfo]) + cudaObjectOffset;
                    pageTableMutex.WUnlock();
                    return cudaObjectAddress;
                } else {
                    void* cudaPointer = nullptr;
                    copyFromHostToDeviceAsync((void **)&cudaPointer, pageInfo.first, pageInfo.second, cs);
                    PageTable.insert(std::make_pair(pageInfo, cudaPointer));
                    pageTableMutex.WUnlock();
                    void *cudaObjectAddress = static_cast<char *>(cudaPointer) + cudaObjectOffset;
                    return cudaObjectAddress;
                }
            }
        }

        RamPointerReference handleInputObjectWithRamPointer(pair<void *, size_t> pageInfo, void *objectAddress, size_t size, cudaStream_t cs){

            size_t cudaObjectOffset = getObjectOffset(pageInfo.first, objectAddress);
            if (PageTable.find(pageInfo) != PageTable.end()){
                pageTableMutex.RLock();
                void *cudaObjectAddress = static_cast<char *>(PageTable[pageInfo]) + cudaObjectOffset;
                pageTableMutex.RUnlock();
                return addRamPointerCollection(cudaObjectAddress, objectAddress, size);
            } else {
                pageTableMutex.WLock();
                if (PageTable.find(pageInfo) != PageTable.end()){
                    void * cudaObjectAddress = static_cast<char*>(PageTable[pageInfo]) + cudaObjectOffset;
                    pageTableMutex.WUnlock();
                    return addRamPointerCollection(cudaObjectAddress, objectAddress, size);
                } else {
                    void* cudaPointer = nullptr;
                    frame_id_t oneframe = getAvailableFrame();
                    copyFromHostToDeviceAsyncWithOutMalloc(availablePosition[oneframe], pageInfo.first, pageInfo.second, cs);
                    cudaPointer = availablePosition[oneframe];
                    PageTable.insert(std::make_pair(pageInfo, cudaPointer));
                    pageTableMutex.WUnlock();
                    void* cudaObjectAddress = static_cast<char *>(cudaPointer) + cudaObjectOffset;
                    return addRamPointerCollection(cudaObjectAddress, objectAddress, size);
                }
            }
        }

        //TODO: operations to memMalloc() should be implemented as thread safe.
        void *memMalloc(size_t memSize) {
            memMallocMutex.WLock();
            if (allocatorPages.size() == 0 && currFrame == -1) {
                frame_id_t oneframe = getAvailableFrame();
                bytesUsed = 0;
                allocatorPages.push_back(oneframe);
                currFrame = oneframe;
            }
            if (memSize > (pageSize - bytesUsed)) {
                frame_id_t oneframe = getAvailableFrame();
                bytesUsed = 0;
                allocatorPages.push_back(oneframe);
                currFrame = oneframe;
            }
            size_t start = bytesUsed;
            bytesUsed += memSize;
            memMallocMutex.WUnlock();
            return static_cast<char *>(availablePosition[currFrame]) + start;
        }

        RamPointerReference
        addRamPointerCollection(void *gpuaddress, void *cpuaddress, size_t numbytes = 0, size_t headerbytes = 0) {
            RamPointerMutex.WLock();
            if (ramPointerCollection.count(gpuaddress) != 0) {
                ramPointerCollection[gpuaddress]->push_back_cpu_pointer(cpuaddress);
                RamPointerMutex.WUnlock();
                //std::cout << " already exist RamPointerCollection size: " << ramPointerCollection.size() << std::endl;
                return std::make_shared<RamPointerBase>(ramPointerCollection[gpuaddress]);
            } else {
                RamPointerPtr ptr = std::make_shared<RamPointer>(gpuaddress, numbytes, headerbytes);
                ptr->push_back_cpu_pointer(cpuaddress);
                ramPointerCollection[gpuaddress] = ptr;
                RamPointerMutex.WUnlock();
                //std::cout << " non exist RamPointerCollection size: " << ramPointerCollection.size() << std::endl;
                return std::make_shared<RamPointerBase>(ptr);
            }
        }


        frame_id_t getAvailableFrame() {
            frame_id_t frame;
            if (!freeList.empty()) {
                frame = freeList.front();
                freeList.pop_front();
                recentlyUsed[frame] = true;
                return frame;
            } else {
                while (recentlyUsed[clock_hand] == true) {
                    recentlyUsed[clock_hand] = false;
                    incrementIterator(clock_hand);
                }
                recentlyUsed[clock_hand] = true;
                frame = clock_hand;
                incrementIterator(clock_hand);
                auto iter = std::find_if(frameTable.begin(), frameTable.end(),
                                         [&](const std::pair<pair<void *, size_t>, frame_id_t> &pair) {
                                             return pair.second == frame;
                                         });
                if (iter->second < 0 || iter->second > poolSize) {
                    std::cerr << " frame number is wrong! \n";
                }
                frameTable.erase(iter);
                PageTable.erase(iter->first);
                return frame;
            }
        }

        void DeepCopyD2H(void *startLoc, size_t numBytes) {
            int count = 0;
            for (auto &ramPointerPair : ramPointerCollection) {
                for (void *cpuPointer: ramPointerPair.second->cpuPointers) {
                    if (cpuPointer >= startLoc && cpuPointer < (static_cast<char*> (startLoc) + numBytes)) {
                        std::cout <<  " thread info: " <<(long) pthread_self()  << " count: " << ++count << std::endl;

                        // TODO: optimize this with async way
                        copyFromDeviceToHost(cpuPointer, ramPointerPair.second->ramAddress,
                                             ramPointerPair.second->numBytes);

                        // TODO: here should have a better way
                        //Array<Nothing> *array = (Array<Nothing> *) ((char *) cpuPointer -
                        //                                            ramPointerPair.second->headerBytes);
                        //array->setRamPointerReferenceToNull();
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
        void incrementIterator(int32_t &it) {
            if (++it == poolSize) {
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
        //std::map<pair<void*,size_t>, void*> PageTable;
        std::map<pair<void *, size_t>, void *> PageTable;

        /**
         * framePageTable for mapping CPU bufferManager page address to GPU frame.
         */
        std::map<pair<void *, size_t>, frame_id_t> frameTable;


        /**
         * one latch to protect the gpuPageTable access
         */
        ReaderWriterLatch pageTableMutex;

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
        std::vector<void *> availablePosition;

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
         * === Here is the part for mem allocator ===
         */

        ReaderWriterLatch RamPointerMutex{};

        /**
         * This latch is for protecting `bytesUsed` and `currFrame` and `allocatorPages`
         */
        ReaderWriterLatch memMallocMutex{};

        size_t bytesUsed = 0;

        frame_id_t currFrame = -1;

        /**
         * This is a vector of all the pages for out parameter
         */
        std::vector<frame_id_t> allocatorPages;

        /**
         * This is a map between gpu memory address and the RamPointer object.
         * It keeps all the ramPointers we create using the RamPointerPtr
         */
        std::map<void *, pdb::RamPointerPtr> ramPointerCollection;
    };
}

#endif
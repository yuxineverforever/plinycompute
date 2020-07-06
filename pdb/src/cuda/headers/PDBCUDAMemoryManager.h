#ifndef PDB_CUDA_MEM_MANAGER
#define PDB_CUDA_MEM_MANAGER

#include <map>
#include <list>
#include <set>
#include <assert.h>
#include <PDBBufferManagerInterface.h>
#include "PDBCUDAUtility.h"
#include "PDBRamPointer.h"
#include "PDBCUDAPage.h"
#include "PDBCUDAConfig.h"
#include "PDBCUDAReplacer.h"
#include "PDBCUDACPUStorageManager.h"

#include "ReaderWriterLatch.h"

namespace pdb {
    class PDBCUDAMemoryManager;
    using PDBCUDAMemoryManagerPtr = std::shared_ptr<PDBCUDAMemoryManager>;

    class PDBCUDAMemoryManager {
    public:
        /**
         *
         * @param buffer - CPU buffer
         */
        PDBCUDAMemoryManager(PDBBufferManagerInterfacePtr buffer, int32_t pool_Size, bool isManager) {
            if (isManager) {
                return;
            }
            bufferManager = buffer;
            poolSize = pool_Size;
            pageSize = buffer->getMaxPageSize();

            pages = new PDBCUDAPage[poolSize];
            replacer = new ClockReplacer(poolSize);

            for (size_t i = 0; i < poolSize; i++) {
                void *cudaPointer;
                cudaMalloc((void **) &cudaPointer, pageSize);
                pages[i].setBytes(static_cast<char*>(cudaPointer));
                pages[i].setPageSize(pageSize);
                freeList.emplace_back(static_cast<int32_t>(i));
            }
        }

        /**
         *
         */
        ~PDBCUDAMemoryManager() {

            delete[] pages;
            delete replacer;

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
         * @return the offset of the object to its cpu page start address
         */
        size_t getObjectOffsetWithCPUPage(void *pageAddress, void *objectAddress) {
            return (char *) objectAddress - (char *) pageAddress;
        }

        /**
         *
         * @param objectAddress
         * @return the info of the cpu page containing this object
         */
        pair<void *, size_t> getObjectCPUPage(void *objectAddress) {
            pdb::PDBPagePtr whichPage = bufferManager->getPageForObject(objectAddress);
            if (whichPage == nullptr) {
                std::cout << "getObjectCPUPage: cannot get page for this object!\n";
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
            size_t cudaObjectOffset = getObjectOffsetWithCPUPage(pageInfo.first, objectAddress);
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
                    frame_id_t  oneframe = getAvailableFrame();
                    cudaPointer = availablePosition[oneframe];
                    PageTable.insert(std::make_pair(pageInfo, cudaPointer));
                    pageTableMutex.WUnlock();

                    copyFromHostToDeviceAsync(cudaPointer, pageInfo.first, pageInfo.second, cs);
                    void *cudaObjectAddress = static_cast<char *>(cudaPointer) + cudaObjectOffset;
                    return cudaObjectAddress;
                }
            }
        }

        RamPointerReference handleInputObjectWithRamPointer(pair<void *, size_t> pageInfo, void *objectAddress, size_t size, cudaStream_t cs){

            size_t cudaObjectOffset = getObjectOffsetWithCPUPage(pageInfo.first, objectAddress);
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
                    cudaPointer = availablePosition[oneframe];
                    PageTable.insert(std::make_pair(pageInfo, cudaPointer));
                    pageTableMutex.WUnlock();

                    copyFromHostToDeviceAsync(cudaPointer, pageInfo.first, pageInfo.second, cs);
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

        bool SwapPageOut(page_id_t page_id) {
            // TODO: implement it.
            return true;
        }

        void FindFramePlacement(frame_id_t& frame){
            // Try to find the available frame from free_list, if not, try to get from replacer
            if (!freeList.empty()) {
                frame = freeList.front();
                freeList.pop_front();
                return;
            }
            // find the available frame from replacer
            if (replacer->Size() > 0) {
                replacer->Victim(&frame);
                auto res = std::find_if(pageTable.begin(), pageTable.end(),
                                        [&](const std::pair<page_id_t, frame_id_t> &it) {
                                            return it.second == frame;
                                        });
                // if page is dirty, write it back to disk
                page_id_t toWriteBack = res->first;
                if (pages[frame].isDirty()) {
                    SwapPageOut(toWriteBack);
                }
                pageTable.erase(res);
            } else {
                std::cout << "No available memory in FetchPageImpl()!\n";
                exit(-1);
            }
        }

        bool UnpinPageImpl(page_id_t page_id, bool is_dirty) {
            auto iter = pageTable.find(page_id);
            if (iter == pageTable.end()){
                return false;
            }
            auto& thisPage = pages[iter->second];
            auto pincount = thisPage.GetPinCount();
            if (pincount<=0){
                return false;
            }
            thisPage.setDirty(is_dirty);
            thisPage.decrementPinCount();
            if (thisPage.GetPinCount()==0){
                replacer->Unpin(iter->second);
            }
            return true;
        }

        bool IsAllPagesPinned(){
            return pageTable.size() == poolSize && replacer->Size() == 0;
        }

        PDBCUDAPage* FetchPageImpl(page_id_t page_id){
            auto iter = pageTable.find(page_id);
            if (iter != pageTable.end()){
                replacer->Pin(iter->second);
                pages[iter->second].incrementPinCount();
                return &pages[iter->second];
            }
            if (IsAllPagesPinned()){
                return nullptr;
            }
            // if cannot find the page from page_table, then try to get the page from free_list or replacer.
            frame_id_t replacement;
            FindFramePlacement(replacement);
            //update the page table
            pageTable[page_id] = replacement;
            pages[replacement].Reset();
            pages[replacement].incrementPinCount();
            pages[replacement].setPageID(page_id);
            replacer->Pin(replacement);

            //TODO: change cpu_storage_manager
            cpu_storage_manager->ReadPage(page_id, pages[replacement].getBytes());
            return &pages[replacement];
        }

        PDBCUDAPage* NewPageImpl(page_id_t *page_id) {
            //TODO: change cpu_storage_manager
            *page_id = cpu_storage_manager->AllocatePage();

            if (IsAllPagesPinned()){
                return nullptr;
            }
            // find a available frame to place the page
            frame_id_t placement;
            FindFramePlacement(placement);
            pageTable[*page_id] = placement;
            pages[placement].Reset();
            pages[placement].incrementPinCount();
            pages[placement].setPageID(*page_id);
            replacer->Pin(placement);
            return &pages[placement];
        }

        bool DeletePageImpl(page_id_t page_id) {

            //TODO: change cpu_storage_manager
            cpu_storage_manager->DeallocatePage(page_id);
            auto iter = pageTable.find(page_id);
            if (iter==pageTable.end()){
                return true;
            } else {
                auto& thisPage = pages[iter->second];
                auto pincount = thisPage.GetPinCount();
                if (pincount==0){
                    freeList.push_back(iter->second);
                    pageTable.erase(iter);
                    thisPage.Reset();
                    return true;
                } else {
                    return false;
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

        /**
         * the buffer manager to help maintain gpu page table
         */
        PDBBufferManagerInterfacePtr bufferManager;


        /** Page table for keeping track of buffer pool pages. */
        std::unordered_map<page_id_t, frame_id_t> pageTable;

        /**  H2DPageMap for mapping CPU bufferManager page info to GPU bufferManager page ids */
        std::unordered_map<pair<void *, size_t>, page_id_t> H2DPageMap;

        /** array of all the pages */
        PDBCUDAPage* pages;

        /** one latch to protect the gpuPageTable access */
        ReaderWriterLatch pageTableMutex;

        /** the size of the pool  */
        size_t poolSize;

        /** the size of the page in gpu Buffer Manager  */
        size_t pageSize;

        /** here is the replacer for */
        ClockReplacer *replacer;

        /** Frames for holding the free memory */
        std::list<frame_id_t> freeList;


        PDBCUDACPUStorageManager* cpu_storage_manager;

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
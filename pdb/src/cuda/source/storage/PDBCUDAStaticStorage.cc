#include "storage/PDBCUDAStaticStorage.h"

namespace pdb{

    // TODO: we should have someway to "remember" input pages and unpin them.
    // TODO: principle: make sure when the computation is on, input/ouput pages are pinned. otherwise, pages should be unpinned.

    inline size_t PDBCUDAStaticStorage::getObjectOffsetWithCPUPage(void* pageAddress, void* objectAddress) {
        return (char *) objectAddress - (char *) pageAddress;
    }

    inline bool PDBCUDAStaticStorage::IsCPUPageMovedToGPU(pair<void*, size_t> pageInfo){
        return pageMap.find(pageInfo) != pageMap.end();
    }

    bool PDBCUDAStaticStorage::IsObjectOnGPU(void* objectAddress){
        auto pageInfo = getCPUPageFromObjectAddress(objectAddress);
        return pageMap.find(pageInfo) != pageMap.end();
    }

    pair<void*, size_t> PDBCUDAStaticStorage::getCPUPageFromObjectAddress(void* objectAddress) {
        // objectAddress must be a CPU RAM Pointer
        assert(isDevicePointer(objectAddress) == 0);

        pdb::PDBPagePtr whichPage = PDBCUDAMemoryManager::getCPUBufferManagerInterface()->getPageForObject(objectAddress);
        if (whichPage == nullptr) {
            std::cout << "getObjectCPUPage: cannot get page for this object!\n";
            exit(-1);
        }
        void *pageAddress = whichPage->getBytes();
        size_t pageBytes = whichPage->getSize();
        auto pageInfo = std::make_pair(pageAddress, pageBytes);
        return pageInfo;
    }

    std::pair<page_id_t, MemAllocateStatus> PDBCUDAStaticStorage::checkGPUPageTable(pair<void*, size_t> pageInfo){
        // If Page has been added, just return it.
        if (pageMap.find(pageInfo) != pageMap.end()){
            // return false means the GPU page is already created.
            return std::make_pair(pageMap[pageInfo], MemAllocateStatus::OLD);

        } else {
            // otherwise, grab a new page, insert to map and return pageID.
            page_id_t newPageID;
            PDBCUDAMemoryManager::get()->CreateNewPage(&newPageID);
            pageMap.insert(std::make_pair(pageInfo, newPageID));
            // return true means the GPU page is newly created.
            return std::make_pair(newPageID, MemAllocateStatus::NEW);
        }
    }

    void PDBCUDAStaticStorage::create(){
        s_store = new PDBCUDAStaticStorage;
    }

    PDBCUDAStaticStorage* PDBCUDAStaticStorage::get(){
        std::call_once(initFlag, PDBCUDAStaticStorage::create);
        assert(check() == true);
        return s_store;
    }

    bool PDBCUDAStaticStorage::check(){
        return s_store!= nullptr;
    }
}
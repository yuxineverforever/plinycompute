#include "storage/PDBCUDAStaticStorage.h"

namespace pdb{

    // TODO: we should have someway to "remember" input pages and unpin them.
    // TODO: principle: make sure when the computation is on, input/ouput pages are pinned. otherwise, pages should be unpinned.

    inline size_t PDBCUDAStaticStorage::getObjectOffsetWithCPUPage(void* pageAddress, void* objectAddress) {
        return (char *) objectAddress - (char *) pageAddress;
    }

    inline bool PDBCUDAStaticStorage::IsCPUPageMovedToGPU(pair<void*, size_t> pageInfo){
        return H2DPageMap.find(pageInfo) != H2DPageMap.end();
    }

    bool PDBCUDAStaticStorage::IsObjectOnGPU(void* objectAddress){
        auto pageInfo = getCPUPageFromObjectAddress(objectAddress);
        return H2DPageMap.find(pageInfo) != H2DPageMap.end();
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

    pair<page_id_t, GPUPageCreateStatus> PDBCUDAStaticStorage::getGPUPageFromCPUPage(pair<void*, size_t> pageInfo){

        // If Page has been added, just return it.
        if (H2DPageMap.find(pageInfo) != H2DPageMap.end()){

            // return false means the GPU page is already created.
            return std::make_pair(H2DPageMap[pageInfo], GPUPageCreateStatus::CREATED_PAGE);
        } else {
            // otherwise, grab a new page, insert to map and return pageID.
            page_id_t newPageID;
            PDBCUDAMemoryManager::get()->CreateNewPage(&newPageID);
            H2DPageMap.insert(std::make_pair(pageInfo, newPageID));

            // return true means the GPU page is newly created.
            return std::make_pair(newPageID, GPUPageCreateStatus::NOT_CREATED_PAGE);
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
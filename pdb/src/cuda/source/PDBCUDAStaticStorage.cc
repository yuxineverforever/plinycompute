#include "PDBCUDAStaticStorage.h"

namespace pdb{


    size_t PDBCUDAStaticStorage::getObjectOffsetWithCPUPage(void* pageAddress, void* objectAddress) {
        return (char *) objectAddress - (char *) pageAddress;
    }

    pair<void*, size_t> PDBCUDAStaticStorage::getObjectCPUPage(void* objectAddress) {
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

    void* PDBCUDAStaticStorage::handleInputObject(pair<void *, size_t> pageInfo, void *objectAddress, cudaStream_t cs) {

        size_t cudaObjectOffset = getObjectOffsetWithCPUPage(pageInfo.first, objectAddress);

        if (H2DPageMap.find(pageInfo) != H2DPageMap.end()) {
            void *cudaObjectAddress = static_cast<char *>(PDBCUDAMemoryManager::get()->FetchPageImpl(H2DPageMap[pageInfo])->getBytes()) + cudaObjectOffset;
            return cudaObjectAddress;

        } else {

            if (H2DPageMap.find(pageInfo) != H2DPageMap.end()) {
                void *cudaObjectAddress = static_cast<char *>(PDBCUDAMemoryManager::get()->FetchPageImpl(H2DPageMap[pageInfo])->getBytes()) + cudaObjectOffset;
                return cudaObjectAddress;
            } else {

                page_id_t newPageID;

                //TODO: pin the static pages for a long time
                PDBCUDAPage* newPage = PDBCUDAMemoryManager::get()->NewPageImpl(&newPageID);
                H2DPageMap.insert(std::make_pair(pageInfo, newPageID));
                char* cudaPointer = newPage->getBytes();

                copyFromHostToDeviceAsync(cudaPointer, pageInfo.first, pageInfo.second, cs);
                void* cudaObjectAddress = static_cast<char *>(cudaPointer) + cudaObjectOffset;
                return cudaObjectAddress;
            }
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
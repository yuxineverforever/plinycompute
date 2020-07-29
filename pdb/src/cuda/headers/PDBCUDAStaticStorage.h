#ifndef PDB_CUDA_STATIC_STORAGE
#define PDB_CUDA_STATIC_STORAGE

#include <PDBCUDAMemoryManager.h>
#include <PDBCUDAConfig.h>

/**
 * StaticStorage is for handling all the static space allocation. (input parameters)
 * The allocation unit is page
 */
namespace pdb{

class PDBCUDAStaticStorage{

public:

    PDBCUDAStaticStorage() = default;

    static size_t getObjectOffsetWithCPUPage(void* pageAddress, void* objectAddress) {
        return (char *) objectAddress - (char *) pageAddress;
    }

    static pair<void*, size_t> getObjectCPUPage(void* objectAddress) {
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

    static void* handleInputObject(pair<void *, size_t> pageInfo, void *objectAddress, cudaStream_t cs) {
        //std::cout << (long) pthread_self() << " : pageInfo: " << pageInfo.first << "bytes: "<< pageInfo.second << std::endl;
        size_t cudaObjectOffset = getObjectOffsetWithCPUPage(pageInfo.first, objectAddress);
        if (H2DPageMap.find(pageInfo) != H2DPageMap.end()) {
            pageTableMutex.RLock();
            void *cudaObjectAddress = static_cast<char *>(H2DPageMap[pageInfo]) + cudaObjectOffset;
            pageTableMutex.RUnlock();
            return cudaObjectAddress;
        } else {
            pageTableMutex.WLock();
            if (H2DPageMap.find(pageInfo) != H2DPageMap.end()) {
                void *cudaObjectAddress = static_cast<char *>(H2DPageMap[pageInfo]) + cudaObjectOffset;
                pageTableMutex.WUnlock();
                return cudaObjectAddress;
            } else {
                void* cudaPointer = nullptr;
                frame_id_t  oneframe = PDBCUDAMemoryManager::getCPUBufferManagerInterface()->getAvailableFrame();
                cudaPointer = availablePosition[oneframe];
                PageTable.insert(std::make_pair(pageInfo, cudaPointer));
                pageTableMutex.WUnlock();
                copyFromHostToDeviceAsync(cudaPointer, pageInfo.first, pageInfo.second, cs);
                void *cudaObjectAddress = static_cast<char *>(cudaPointer) + cudaObjectOffset;
                return cudaObjectAddress;
            }
        }
    }

private:

    /**  H2DPageMap for mapping CPU bufferManager page info to GPU bufferManager page ids */
    static std::map<pair<void *, size_t>, page_id_t> H2DPageMap;

    /** one latch to protect the gpuPageTable access */
    static ReaderWriterLatch pageTableMutex;

    friend class PDBCUDAMemoryManager;
};

}

#endif
#ifndef PDB_CUDA_STATIC_STORAGE
#define PDB_CUDA_STATIC_STORAGE

#include <storage/PDBCUDAMemoryManager.h>
#include <PDBCUDAConfig.h>

/**
 * StaticStorage is for handling all the static space allocation. (input parameters)
 * The allocation unit is page
 */

//TODO: let invoke remember the input pages/output page
namespace pdb{

enum GPUPageCreateStatus{
    CREATED_PAGE,
    NOT_CREATED_PAGE
};

class PDBCUDAStaticStorage{

public:
    PDBCUDAStaticStorage() = default;

    inline size_t getObjectOffsetWithCPUPage(void* pageAddress, void* objectAddress);

    pair<void*, size_t> getCPUPageFromObjectAddress(void* objectAddress);

    pair<page_id_t, GPUPageCreateStatus> getGPUPageFromCPUPage(pair<void*, size_t> pageInfo);

    inline bool IsCPUPageMovedToGPU(pair<void*, size_t> pageInfo);
    bool IsObjectOnGPU(void* objectAddress);

    static void create();
    static PDBCUDAStaticStorage* get();
    static inline bool check();

private:

    static PDBCUDAStaticStorage* s_store;

    static std::once_flag initFlag;

    /**  H2DPageMap for mapping CPU bufferManager page info to GPU bufferManager page ids */
    static std::map<pair<void *, size_t>, page_id_t> H2DPageMap;

    /** one latch to protect the gpuPageTable access */
    static ReaderWriterLatch pageTableMutex;

    friend class PDBCUDAMemoryManager;
};

}

#endif
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

enum MemAllocateStatus{
    OLD,
    NEW
};

enum MemAllocatePolicy{
    DYNAMIC,
    STATIC
};

class PDBCUDAStaticStorage{

public:
    PDBCUDAStaticStorage() = default;

    inline size_t getObjectOffsetWithCPUPage(void* pageAddress, void* objectAddress);

    pair<void*, size_t> getCPUPageFromObjectAddress(void* objectAddress);

    //TODO: the policy of checkGPUPageTable should be decided based on the size of pageInfo
    // If size == Mgr.PageSize, use STATIC
    // If size < Mgr.PageSize, use DYNAMIC

    std::pair<page_id_t, MemAllocateStatus> checkGPUPageTable(pair<void*, size_t> pageInfo);

    static void create();
    static PDBCUDAStaticStorage* get();
    static inline bool check();

    inline bool IsCPUPageMovedToGPU(pair<void*, size_t> pageInfo);
    bool IsObjectOnGPU(void* objectAddress);

private:

    static PDBCUDAStaticStorage* s_store;

    static std::once_flag initFlag;

    /**  H2DPageMap for mapping CPU bufferManager page info to GPU bufferManager page ids */
    /** Used for static allocation */
    std::map<pair<void *, size_t>, page_id_t> pageMap;

    /** one latch to protect the gpuPageTable access */
    ReaderWriterLatch pageTableMutex;

    friend class PDBCUDAMemoryManager;
};

}

#endif
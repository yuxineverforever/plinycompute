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

    static size_t getObjectOffsetWithCPUPage(void* pageAddress, void* objectAddress);

    static pair<void*, size_t> getObjectCPUPage(void* objectAddress);

    static void* handleInputObject(pair<void *, size_t> pageInfo, void *objectAddress, cudaStream_t cs);

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
#ifndef PDB_CUDA_RAM_POINTER_MANAGER
#define PDB_CUDA_RAM_POINTER_MANAGER

#include <iostream>
#include <PDBCUDAMemoryManager.h>
#include <PDBCUDAConfig.h>

/**
 * DynamicStorage is for handling all the dynamic space allocation. (e.g. RamPointer)
 * The allocation unit is byte
 */
namespace pdb{


class PDBCUDADynamicStorage{

public:

    PDBCUDADynamicStorage() = default;

    //TODO: operations to memMalloc() should be implemented as thread safe.
    //TODO: operations should be called from single stream.
    void* memMalloc(size_t size){}

    void memFree(void *ptr){

    }

    pdb::RamPointerReference keepMemAddress(void *gpuaddress, void *cpuaddress, size_t numbytes, size_t headerbytes){
    }

private:

    std::vector<page_id_t> dynamicPages;

    size_t bytesUsed = 0;

    /**
     * This is a map between gpu memory address and the RamPointer object. It keeps all the ramPointers we create using the RamPointerPtr
     */
    std::map<page_id_t, pdb::RamPointerPtr> ramPointerCollection;

    friend class PDBCUDAMemoryManager;
};
}

#endif
#ifndef PDB_CUDA_CPU_STORAGE_MANAGER
#define PDB_CUDA_CPU_STORAGE_MANAGER

#include <PDBCUDAConfig.h>
#include <list>
#include "cuda_runtime.h"
#include "helper_cuda.h"

namespace pdb{

    class PDBCUDACPUStorageManager{

    public:

        PDBCUDACPUStorageManager() = default;

        PDBCUDACPUStorageManager(int32_t PageNum = CPU_STORAGE_MANAGER_PAGE_NUM, size_t PageSize = 1024*1024*1024);

        ~PDBCUDACPUStorageManager() = default;

        void ReadPage(page_id_t page_id, char* page_data);

        page_id_t AllocatePage();

        void DeallocatePage(page_id_t page_id);

        void WritePage(page_id_t page_id, const char *page_data);

        RamPointerReference handleInputObjectWithRamPointer(pair<void *, size_t> pageInfo, void *objectAddress, size_t size, cudaStream_t cs);

        RamPointerReference addRamPointerCollection(void *gpuaddress, void *cpuaddress, size_t numbytes = 0, size_t headerbytes = 0);

        void DeepCopyD2H(void* startLoc, size_t numBytes);

    private:

        std::atomic<page_id_t>  next_page_id_;

        std::list<void*> freeList;

        std::map<page_id_t, void*> storageMap;
    };
}
#endif
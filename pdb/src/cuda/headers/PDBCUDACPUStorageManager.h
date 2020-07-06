#ifndef PDB_CUDA_CPU_STORAGE_MANAGER
#define PDB_CUDA_CPU_STORAGE_MANAGER

#include <PDBCUDAConfig.h>

namespace pdb{

    class PDBCUDACPUStorageManager{
    public:
        PDBCUDACPUStorageManager() = default;
        ~PDBCUDACPUStorageManager() = default;
        void ReadPage(page_id_t page_id, char *page_data);
        page_id_t AllocatePage() { return next_page_id_++; }

    private:
        page_id_t  next_page_id_ = 0;
    };
}
#endif
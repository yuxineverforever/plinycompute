#include <storage/PDBCUDACPUStorageManager.h>

namespace pdb{

    PDBCUDACPUStorageManager::PDBCUDACPUStorageManager(int32_t PageNum, size_t PageSize){

        for (size_t i = 0; i<PageNum;i++){
             void* page = malloc(PageSize);
             freeList.push_back(page);
        }
        next_page_id_ = 0;
    }

    void PDBCUDACPUStorageManager::ReadPage(page_id_t page_id, char* page_data){

    }

    void PDBCUDACPUStorageManager::WritePage(page_id_t page_id, const char *page_data){

    }

    page_id_t PDBCUDACPUStorageManager::AllocatePage() {
        return next_page_id_++;
    }

    void PDBCUDACPUStorageManager::DeallocatePage(page_id_t page_id);

    RamPointerReference PDBCUDACPUStorageManager::handleInputObjectWithRamPointer(pair<void *, size_t> pageInfo, void *objectAddress, size_t size, cudaStream_t cs){

    }

    RamPointerReference PDBCUDACPUStorageManager::addRamPointerCollection(void *gpuaddress, void *cpuaddress, size_t numbytes = 0, size_t headerbytes = 0) {
    }

    void PDBCUDACPUStorageManager::DeepCopyD2H(void *startLoc, size_t numBytes) {


        int count = 0;
        for (auto &ramPointerPair : ramPointerCollection) {
            for (void *cpuPointer: ramPointerPair.second->cpuPointers) {
                if (cpuPointer >= startLoc && cpuPointer < (static_cast<char*> (startLoc) + numBytes)) {
                    std::cout <<  " thread info: " <<(long) pthread_self()  << " count: " << ++count << std::endl;

                    // TODO: optimize this with async way
                    copyFromDeviceToHost(cpuPointer, ramPointerPair.second->ramAddress,
                                         ramPointerPair.second->numBytes);

                    // TODO: here should have a better way
                    //Array<Nothing> *array = (Array<Nothing> *) ((char *) cpuPointer -
                    //                                            ramPointerPair.second->headerBytes);
                    //array->setRamPointerReferenceToNull();
                }
            }
        }
    }



}

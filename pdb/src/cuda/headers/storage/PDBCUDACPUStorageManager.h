#ifndef PDB_CUDA_CPU_STORAGE_MANAGER
#define PDB_CUDA_CPU_STORAGE_MANAGER

#include <PDBCUDAConfig.h>

namespace pdb{

    class PDBCUDACPUStorageManager{
    public:

        PDBCUDACPUStorageManager() = default;
        ~PDBCUDACPUStorageManager() = default;

        void ReadPage(page_id_t page_id, char* page_data);

        page_id_t AllocatePage() {
            return next_page_id_++;
        }

        void DeallocatePage(page_id_t page_id);

        RamPointerReference handleInputObjectWithRamPointer(pair<void *, size_t> pageInfo, void *objectAddress, size_t size, cudaStream_t cs){
            size_t cudaObjectOffset = getObjectOffsetWithCPUPage(pageInfo.first, objectAddress);
            if (PageTable.find(pageInfo) != PageTable.end()){
                pageTableMutex.RLock();
                void *cudaObjectAddress = static_cast<char *>(PageTable[pageInfo]) + cudaObjectOffset;
                pageTableMutex.RUnlock();
                return addRamPointerCollection(cudaObjectAddress, objectAddress, size);
            } else {
                pageTableMutex.WLock();
                if (PageTable.find(pageInfo) != PageTable.end()){
                    void * cudaObjectAddress = static_cast<char*>(PageTable[pageInfo]) + cudaObjectOffset;
                    pageTableMutex.WUnlock();
                    return addRamPointerCollection(cudaObjectAddress, objectAddress, size);
                } else {
                    void* cudaPointer = nullptr;
                    frame_id_t oneframe = getAvailableFrame();
                    cudaPointer = availablePosition[oneframe];
                    PageTable.insert(std::make_pair(pageInfo, cudaPointer));
                    pageTableMutex.WUnlock();

                    copyFromHostToDeviceAsync(cudaPointer, pageInfo.first, pageInfo.second, cs);
                    void* cudaObjectAddress = static_cast<char *>(cudaPointer) + cudaObjectOffset;
                    return addRamPointerCollection(cudaObjectAddress, objectAddress, size);
                }
            }
        }

        RamPointerReference addRamPointerCollection(void *gpuaddress, void *cpuaddress, size_t numbytes = 0, size_t headerbytes = 0) {
            RamPointerMutex.WLock();
            if (ramPointerCollection.count(gpuaddress) != 0) {
                ramPointerCollection[gpuaddress]->push_back_cpu_pointer(cpuaddress);
                RamPointerMutex.WUnlock();
                //std::cout << " already exist RamPointerCollection size: " << ramPointerCollection.size() << std::endl;
                return std::make_shared<RamPointerBase>(ramPointerCollection[gpuaddress]);
            } else {
                RamPointerPtr ptr = std::make_shared<RamPointer>(gpuaddress, numbytes, headerbytes);
                ptr->push_back_cpu_pointer(cpuaddress);
                ramPointerCollection[gpuaddress] = ptr;
                RamPointerMutex.WUnlock();
                //std::cout << " non exist RamPointerCollection size: " << ramPointerCollection.size() << std::endl;
                return std::make_shared<RamPointerBase>(ptr);
            }
        }

        void DeepCopyD2H(void *startLoc, size_t numBytes) {
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

    private:
        page_id_t  next_page_id_ = 0;
    };
}
#endif
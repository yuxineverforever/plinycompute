#ifndef PDB_CUDA_MEM_MANAGER
#define PDB_CUDA_MEM_MANAGER

#include <map>
#include <PDBBufferManagerInterface.h>
#include "PDBCUDAUtility.h"

namespace pdb {

class PDBCUDAMemoryManager;
using PDBCUDAMemoryManagerPtr = std::shared_ptr<PDBCUDAMemoryManager>;
class PDBCUDAMemoryManager{
    public:
        PDBCUDAMemoryManager(PDBBufferManagerInterfacePtr buffer) {
            bufferManager = buffer;
        }
        /**
         * handleOneObject is trying to get the physical address of a gpu page which contains the object
         * @param objectAddress - the physical address for the object we are handling
         * @return void* - the address of object on the GPU page which contains the object
         */
        void* handleOneObject(void *objectAddress) {
            std::cout << "handleOneObject is called!\n";
            pdb::PDBPageHandle whichPage = bufferManager->getPageForObject(objectAddress);
            size_t objectOffset = (char*)objectAddress - (char*)whichPage->getBytes();
            if (gpu_page_table.count(whichPage->getBytes()) != 0) {
                std::cout << "handleOneObject: object is already on GPU\n";
                return (void*)((char *)gpu_page_table[whichPage->getBytes()] + objectOffset);
            } else {
                std::cout << "handleOneObject: object is not on GPU, move the page\n";
                whichPage->repin();
                void* startAddress = whichPage->getBytes();
                size_t numBytes = whichPage->getSize();
                void* cudaPointer;
                copyFromHostToDevice((void **) &cudaPointer, startAddress, numBytes);
                gpu_page_table.insert(std::make_pair(whichPage->getBytes(), cudaPointer));
                return (void*)((char*)cudaPointer + objectOffset);
            }
        }

    public:
        // the buffer manager to help maintain gpu page table
        PDBBufferManagerInterfacePtr bufferManager;

        // gpu_page_table for mapping CPU bufferManager page address to GPU bufferManager page address
        std::map<void*, void*> gpu_page_table;
        // map <pair <PDBSetPtr, size_t>, PDBPagePtr, PDBPageCompare> allPages;
    };

}

#endif
#ifndef PDB_CUDA_MEM_MANAGER
#define PDB_CUDA_MEM_MANAGER

#include <map>
#include <PDBBufferManagerInterface.h>
#include "PDBCUDAUtility.h"
#include "PDBCUDALatch.h"

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
            //std::cout << "handleOneObject is called!\n";
            pdb::PDBPageHandle whichPage = bufferManager->getPageForObject(objectAddress);
            if (whichPage == nullptr){
                std::cout << "handleOneObject return page is nullptr!\n";
                exit(-1);
            }
            void*  pageAddress = whichPage->getBytes();
            size_t pageBytes = whichPage->getSize();
            size_t objectOffset = (char*)objectAddress - (char*)pageAddress;
            bool isThere = (gpuPageTable.count(pageAddress) != 0);
            if (isThere) {
                //pageTableLatch.RLock();
                //std::cout << "handleOneObject: object is already on GPU\n";
                auto cudaObjectAddress = (void*)((char *)(gpuPageTable[pageAddress]) + objectOffset);
                //pageTableLatch.RUnlock();
                return cudaObjectAddress;
            } else {
                //pageTableLatch.WLock();
                //std::cout << "handleOneObject: object is not on GPU, move the page\n";
                void* cudaPointer;
                copyFromHostToDevice((void **) &cudaPointer, pageAddress, pageBytes);
                gpuPageTable.insert(std::make_pair(pageAddress, cudaPointer));
                auto cudaObjectAddress = (void*)((char*)cudaPointer + objectOffset);

                //pageTableLatch.WUnlock();
                return cudaObjectAddress;
            }
        }


    public:
         /**
          * the buffer manager to help maintain gpu page table
          */
        PDBBufferManagerInterfacePtr bufferManager;

        /**
         * gpu_page_table for mapping CPU bufferManager page address to GPU bufferManager page address
         * the PDBPageHandle needs to be stored here for keeping some anonymous page. So that, the anonymous page wont be freed.
         */
        std::map<void*, void*> gpuPageTable;

        /**
         * pageTableLatch
         */
        //ReaderWriterLatch pageTableLatch;
    };

}

#endif
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
         *
         * @param pageAddress
         * @param objectAddress
         * @return
         */
        size_t getObjectOffset(void* pageAddress, void* objectAddress){
            return (char*)objectAddress - (char*)pageAddress;
        }

        /**
         *
         * @param objectAddress
         * @return
         */
        pair<void*, size_t> getObjectPage(void *objectAddress){
            pdb::PDBPageHandle whichPage = bufferManager->getPageForObject(objectAddress);
            if (whichPage == nullptr){
                std::cout << "getObjectOffset: cannot get page for this object!\n";
                exit(-1);
            }
            void*  pageAddress = whichPage->getBytes();
            size_t pageBytes = whichPage->getSize();
            auto pageInfo = std::make_pair(pageAddress, pageBytes);
            if (objectPageHandles.count(pageInfo) == 0){
                objectPageHandles.insert(std::make_pair(pageInfo, whichPage));
            }
            return pageInfo;
        }

        /**
         *
         * @param pageInfo
         * @param objectAddress
         * @return
         */
        void* handleObject(pair<void*, size_t> pageInfo, void *objectAddress) {

            size_t cudaObjectOffset = getObjectOffset(pageInfo.first, objectAddress);

            if (gpuPageTable.count(pageInfo) != 0) {
                // std::cout << "handleInputObject: object is already on GPU\n";
                return (void*)((char *)(gpuPageTable[pageInfo]) + cudaObjectOffset);
            } else {
                // pageTableLatch.WLock();
                std::cout << "handleInputObject: object is not on GPU, move the page\n";
                void* cudaPointer;
                copyFromHostToDevice((void **) &cudaPointer, pageInfo.first, pageInfo.second);
                gpuPageTable.insert(std::make_pair(pageInfo, cudaPointer));
                return (void*)((char*)cudaPointer + cudaObjectOffset);
            }
        }

        /**
         *
         * @param pageInfo
         * @return
         */
        void* getCUDAPage(pair<void*, size_t> pageInfo){

            if (gpuPageTable.count(pageInfo) == 0){
                std::cout << "getCUDAPage: cannot get CUDA page for this CPU Page!\n";
                exit(-1);
            }
            return gpuPageTable[pageInfo];
        }

    public:
         /**
          * the buffer manager to help maintain gpu page table
          */
        PDBBufferManagerInterfacePtr bufferManager;

        /**
         * gpu_page_table for mapping CPU bufferManager page address to GPU bufferManager page address
         */
        std::map<pair<void*,size_t>, void*> gpuPageTable;

        /**
         * objectPageHandles - keep all the handles for a specific page, so that the pages will be unpinned.
         */
        std::map<pair<void*, size_t>, pdb::PDBPageHandle> objectPageHandles;

        /**
         * pageTableLatch
         */

        //ReaderWriterLatch pageTableLatch;
    };

}

#endif
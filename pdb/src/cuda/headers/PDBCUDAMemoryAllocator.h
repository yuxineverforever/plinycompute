#include <iostream>
#include "PDBRamPointer.h"

namespace pdb{

    class PDBCUDAMemoryAllocator{

        public:

            PDBCUDAMemoryAllocator() = default;

            static void* memMalloc(size_t size);

            static void memFree(void* ptr);

            static RamPointerReference keepMemAddress(void* gpuaddress, void* cpuaddress, size_t numbytes, size_t headerbytes);

        private:

            // Interactions with GPU Buffer Manager

    };

};
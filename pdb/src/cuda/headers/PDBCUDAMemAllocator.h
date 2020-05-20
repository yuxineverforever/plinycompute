#include <iostream>

class PDBCUDAMemAllocator{

public:
    void* MemMalloc(size_t size);
    void MemFree(void* freeMe);
};
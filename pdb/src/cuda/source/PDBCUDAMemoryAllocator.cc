
#include "PDBCUDAMemoryAllocator.h"
#include "PDBCUDAMemoryManager.h"

extern void *gpuMemoryManager;



/**
 *
 * @param memSize
 * @return
void* memMalloc(size_t memSize){
    if (allocatorPage == -1){
        frame_id_t oneframe = getAvailableFrame();
        bytesUsed = 0;
        allocatorPage = oneframe;
    }
    if (memSize > (pageSize - bytesUsed)){
        std::cerr << "Unable to allocator space : the space on page is not enough!\n";
    }
    size_t start = bytesUsed;
    bytesUsed += memSize;
    return (void*)((char*)availablePosition[allocatorPage] + start);
}


RamPointerReference addRamPointerCollection(void* gpuaddress, void* cpuaddress, size_t numbytes, size_t headerbytes){
    RamPointer pt(gpuaddress, numbytes, headerbytes);
    auto findIter = std::find(ramPointerCollection.begin(), ramPointerCollection.end(), pt);
    if (findIter != ramPointerCollection.end()){
        findIter->push_back_pointer(cpuaddress);
        return std::make_shared<RamPointer>(*findIter);
    } else {
        pt.push_back_pointer(cpuaddress);
        ramPointerCollection.push_back(pt);
        return std::make_shared<RamPointer>(ramPointerCollection.back());
    }
}
 **/
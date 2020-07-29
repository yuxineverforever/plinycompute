

void* memMalloc(size_t size){

    if (dynamicPages.size() == 0) {


        frame_id_t oneframe = getAvailableFrame();
        bytesUsed = 0;
        allocatorPages.push_back(oneframe);
        currFrame = oneframe;
    }
    if (memSize > (pageSize - bytesUsed)) {
        frame_id_t oneframe = getAvailableFrame();
        bytesUsed = 0;
        allocatorPages.push_back(oneframe);
        currFrame = oneframe;
    }
    size_t start = bytesUsed;
    bytesUsed += memSize;
    memMallocMutex.WUnlock();
    return static_cast<char *>(availablePosition[currFrame]) + start;
}

static void memFree(void *ptr){

}

static pdb::RamPointerReference keepMemAddress(void *gpuaddress, void *cpuaddress, size_t numbytes, size_t headerbytes){
    return ((pdb::PDBCUDAMemoryManager *) gpuMemoryManager)->addRamPointerCollection(gpuaddress, cpuaddress, numbytes,
                                                                                     headerbytes);
}
#ifndef PDB_CUDA_VA_INVOKER
#define PDB_CUDA_VA_INVOKER

#include <iostream>
#include <Handle.h>
#include "PDBVector.h"
#include <functional>
#include <numeric>
#include "utility/PDBCUDAUtility.h"
#include "PDBCUDAOpType.h"
#include "PDBCUDAOpInvoker.h"
#include "storage/PDBCUDAStaticStorage.h"
#include "stream/PDBCUDAStreamManager.h"

namespace pdb {

    // simply support vector-add operation and vector-add kernel for GPU
    class PDBCUDAVectorAddInvoker : public PDBCUDAInvoker {

    public:

        PDBCUDAVectorAddInvoker();

        bool invoke();

        void kernel(float* in1data, float* in2data, size_t N);

        void setInput(float* input, std::vector<size_t> &inputDim);

        void setOutput(float* output, std::vector<size_t> &outputDim);

        void cleanup();

        std::shared_ptr<pdb::RamPointerBase> LazyAllocationHandler(void* pointer, size_t size);

    public:
        // raw pointer and the dimension for the vector
        std::vector<std::pair<float*, std::vector<size_t> >> inputArguments;

        std::pair<float*, std::vector<size_t> > outputArguments;
        float *copyBackPara;

        cublasHandle_t cudaHandle;
        cudaStream_t cudaStream;

        std::vector<page_id_t> inputPages;
        PDBCUDAOpType op = PDBCUDAOpType::VectorAdd;
        PDBCUDAStreamManager* stream_instance;
        PDBCUDAStaticStorage* sstore_instance;
        PDBCUDAMemoryManager* memmgr_instance;
    };
}
#endif
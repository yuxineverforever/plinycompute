#ifndef PDB_CUDA_VA_INVOKER
#define PDB_CUDA_VA_INVOKER

#include <iostream>
#include <Handle.h>
#include "PDBVector.h"
#include <functional>
#include <numeric>
#include "PDBCUDAUtility.h"
#include "PDBCUDAOpType.h"
#include "PDBCUDAOpInvoker.h"

namespace pdb {

    // simply support vector-add operation and vector-add kernel for GPU
    class PDBCUDAVectorAddInvoker : public PDBCUDAOpInvoker {
        using T = float;

    public:

        PDBCUDAVectorAddInvoker();

        bool invoke();

        void cublasRouting(T *in1data, T *in2data, size_t N);

        void setInput(T *input, std::vector<size_t> &inputDim);

        void setOutput(T *output, std::vector<size_t> &outputDim);

        void cleanup();

        std::shared_ptr<pdb::RamPointerBase> LazyAllocationHandler(void* pointer);

    public:
        // raw pointer and the dimension for the vector
        std::vector<std::pair<T *, std::vector<size_t> >> inputParas;
        std::pair<T *, std::vector<size_t> > outputPara;
        T *copyBackPara;
        PDBCUDAOpType op = PDBCUDAOpType::VectorAdd;
        cublasHandle_t cudaHandle;
        cudaStream_t cudaStream;
    };
}
#endif
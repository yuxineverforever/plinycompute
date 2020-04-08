#ifndef PDB_CUDA_MM_INVOKER
#define PDB_CUDA_MM_INVOKER

#include <iostream>
#include <Handle.h>
#include <functional>
#include <numeric>
#include <utility>
#include "PDBVector.h"
#include "PDBCUDAUtility.h"
#include "PDBCUDAOpType.h"
#include "PDBCUDAOpInvoker.h"

// simply support two kind of operations
namespace pdb{

/**
 * PDBCUDAMatrixMultipleInvoker - A wrapper for cublas sgemm
 */
class PDBCUDAMatrixMultipleInvoker: public PDBCUDAOpInvoker{
    using T = float;

public:

    PDBCUDAMatrixMultipleInvoker();

    bool invoke();

    void cublasRouting(T* in1data, T* in2data, T* outdata, size_t in1NumRow, size_t in1NumCol, size_t in2NumCol);

    void setInput(T* input, std::vector<size_t>& inputDim);

    void setOutput(T* output, std::vector<size_t>& outputDim);

    void cleanup();

public:

    /**
     *
     */
    std::vector<std::pair<T*, std::vector<size_t> >> inputParas;

    /**
     *
     */
    std::pair<T *, std::vector<size_t> > outputPara;

    /**
     *
     */
    T* copyBackPara;

    /**
     *  for amortize the D2H copy overhead
     */
    pair<void*, size_t> pageToCopyBack = std::make_pair(nullptr, 0);

    PDBCUDAOpType op = PDBCUDAOpType::MatrixMultiple;

    cublasHandle_t cudaHandle;
};
}
#endif
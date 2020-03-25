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

//simply support Matrix Multiply kernel.
class PDBCUDAMatrixMultipleInvoker: public PDBCUDAOpInvoker{

    using T = float;

public:

    PDBCUDAMatrixMultipleInvoker() = default;

    bool invoke();

    void cublasRouting(T* in1data, T* in2data, T* outdata, size_t in1NumRow, size_t in1NumCol, size_t in2NumCol);

    void setInput(T* input, std::vector<size_t>& inputDim);

    void setOutput(T* output, std::vector<size_t>& outputDim);

    void cleanup();

public:

    std::vector<std::pair<T*, std::vector<size_t> >> InputParas;
    std::pair<T *, std::vector<size_t> > OutputPara;

    T* copyBackPara;

    PDBCUDAOpType op = PDBCUDAOpType::MatrixMultiple;
};
}
#endif
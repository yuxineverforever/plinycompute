#ifndef PDB_CUDA_GLOBAL_VARS
#define PDB_CUDA_GLOBAL_VARS
#include "PDBCUDAVectorAddInvoker.h"
#include "PDBCUDAMatrixMultipleInvoker.h"

void* gpuMemoryManager = nullptr;
pdb::PDBCUDAVectorAddInvoker vectorAddInvoker;
pdb::PDBCUDAMatrixMultipleInvoker matrixMultipleInvoker;

#endif
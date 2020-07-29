#ifndef PDB_CUDA_GLOBAL_VARS
#define PDB_CUDA_GLOBAL_VARS

#include "PDBCUDAStreamManager.h"
#include "PDBCUDAVectorAddInvoker.h"
#include "PDBCUDAMatrixMultipleInvoker.h"

void* gpuMemoryManager = nullptr;
void* gpuThreadManager = nullptr;

#endif
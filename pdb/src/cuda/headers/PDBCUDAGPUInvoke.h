#ifndef PDB_CUDA_GPU_INVOKE
#define PDB_CUDA_GPU_INVOKE

#include <iostream>
#include "PDBCUDAMatrixMultipleInvoker.h"
#include "PDBCUDAVectorAddInvoker.h"
#include <vector>

// `Out` vector should be reserved before passing as parameter
template<typename InvokerType, typename InputType, typename OutputType>
typename std::enable_if_t<is_base_of<pdb::PDBCUDAOpInvoker, InvokerType>::value && std::is_trivially_copyable<OutputType>::value && std::is_trivially_copyable<InputType>::value, bool>
SimpleTypeGPUInvoke(InvokerType& f, OutputType* Out, std::vector<size_t>& OutDim, InputType* In1, std::vector<size_t>& In1Dim, InputType* In2, std::vector<size_t>& In2Dim);

// This is for the case that Handle<SimpleType>
template <typename InvokerType, typename InputType, typename OutputType>
bool GPUInvoke(InvokerType& f, pdb::Handle<OutputType> Output, pdb::Handle<InputType> In1, pdb::Handle<InputType> In2);

// Should we wrap all the objects from pdb::Vector to a simple std::vector? The answer is not.
template <typename InvokerType, typename InputType, typename OutputType>
bool GPUInvoke(InvokerType& f, pdb::Handle<pdb::Vector<OutputType>> Out, pdb::Handle<pdb::Vector<InputType>> In1, pdb::Handle<pdb::Vector<InputType>> In2);

template <typename InvokerType, typename InputType, typename OutputType>
bool GPUInvoke(InvokerType& f, pdb::Handle<pdb::Vector<OutputType>> Out, std::vector<size_t>& OutDim, pdb::Handle<pdb::Vector<InputType>> In1, std::vector<size_t>& In1Dim);

template <typename InvokerType, typename InputType, typename OutputType>
bool GPUInvoke(InvokerType& f, pdb::Handle<pdb::Vector<OutputType>> Out, std::vector<size_t>& OutDim, pdb::Handle<pdb::Vector<InputType>> In1, std::vector<size_t>& In1Dim, pdb::Handle<pdb::Vector<InputType> > In2, std::vector<size_t>& In2Dim);

bool GPUInvoke(pdb::PDBCUDAMatrixMultipleInvoker& f, pdb::Handle<pdb::Vector<float>> Out, std::vector<size_t>& OutDim, pdb::Handle<pdb::Vector<float>> In1, std::vector<size_t>& In1Dim, pdb::Handle<pdb::Vector<float> > In2, std::vector<size_t>& In2Dim);

bool GPUInvoke(pdb::PDBCUDAVectorAddInvoker& f, pdb::Handle<pdb::Vector<float>> Out, std::vector<size_t>& OutDim, pdb::Handle<pdb::Vector<float>> In1, std::vector<size_t>& In1Dim);


#endif
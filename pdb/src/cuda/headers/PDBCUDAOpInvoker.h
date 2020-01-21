#ifndef PDB_CUDA_INVOKER
#define PDB_CUDA_INVOKER

#include <iostream>
#include <Handle.h>
#include "PDBVector.h"
#include "PDBCUDAUtility.h"

enum PDBCUDAOpType {
    SimpleAdd,
    SimpleMultiple,
    MatrixMultiple, //0
    VectorAdd, //1
};

// simply support two kind of operations
class PDBCUDAOpInvoker{
public:
    PDBCUDAOpInvoker();
    virtual bool invoke() = 0;
    virtual void setInput() = 0;
    virtual void setOutput() = 0;
public:
    PDBCUDAOpType op;
};

/*template <typename HandleType, typename RecordType>
size_t getAbsoluteOffset(Handle<HandleType>& input, Record<RecordType>* allocationBlock){
    return 0;
}*/

template <typename InputType, typename OutputType>
bool SimpleTypeGPUInvoke(PDBCUDAOpInvoker& f, OutputType* Out, InputType* In1, InputType* In2){
    if (!std::is_trivially_copyable<InputType>::value){
        std::cout<<"GPUInvoke just allow trivial copyable types\n";
    }
    f.setInput(In1);
    f.setInput(In2);
    f.setOutput(Out);
    bool res = f.invoke();
    return res;
}

template<typename InputType, typename OutputType>
bool SimpleTypeGPUInvoke(PDBCUDAOpInvoker& f, std::vector<OutputType> Out, std::vector<InputType> In1, std::vector<InputType> In2){
    if (!std::is_trivially_copyable<InputType>::value){
        std::cout<<"GPUInvoke just allow trivial copyable types\n";
    }
    auto In1data = In1.data();
    auto In2data = In2.data();
    auto Outdata = Out.data();
    f.setInput(In1data, In1.size());
    f.setInput(In2data, In2.size());
    f.setOutput(Outdata, Out.size());
    bool res = f.invoke();
    return res;
}

// This is for the case that Handle<SimpleType>
template <typename InputType, typename OutputType>
bool GPUInvoke(PDBCUDAOpInvoker& f, pdb::Handle<OutputType> Output, pdb::Handle<InputType> In1, pdb::Handle<InputType> In2){
    auto In1Object = (In1.getTarget())->getObject();
    auto In2Object = (In2.getTarget())->getObject();
    auto OutputObject = (Output.getTarget())->getObject();
    return SimpleTypeGPUInvoke(f, OutputObject, In1Object, In2Object);
}

// Should we wrap all the objects from pdb::Vector to a simple std::vector?
template <typename InputType, typename OutputType>
bool GPUInvoke(PDBCUDAOpInvoker& f, pdb::Handle<pdb::Vector<OutputType>> Out, pdb::Handle<pdb::Vector<InputType>> In1, pdb::Handle<pdb::Vector<InputType>> In2){

}



#include "PDBCUDAGPUInvoke.h"

// `Out` vector should be reserved before passing as parameter
template<typename InvokerType, typename InputType, typename OutputType>
typename std::enable_if_t<is_base_of<pdb::PDBCUDAOpInvoker, InvokerType>::value, bool>
SimpleTypeGPUInvoke(InvokerType& f, OutputType* Out, std::vector<size_t>& OutDim, InputType* In1, std::vector<size_t>& In1Dim, InputType* In2, std::vector<size_t>& In2Dim){
    if (!std::is_trivially_copyable<InputType>::value){
        std::cout<<"GPUInvoke just allow trivial copyable types\n";
    }
    f.setInput(In1, In1Dim);
    f.setInput(In2, In2Dim);
    f.setOutput(Out, OutDim);
    bool res = f.invoke();
    return res;
}

// This is for the case that Handle<SimpleType>
template <typename InvokerType, typename InputType, typename OutputType>
bool GPUInvoke(InvokerType& f, pdb::Handle<OutputType> Output, pdb::Handle<InputType> In1, pdb::Handle<InputType> In2){
    auto In1Object = (In1.getTarget())->getObject();
    std::vector<size_t> In1Dim{1};
    auto In2Object = (In2.getTarget())->getObject();
    std::vector<size_t> In2Dim{1};
    auto OutputObject = (Output.getTarget())->getObject();
    std::vector<size_t> OutDim{1};
    return SimpleTypeGPUInvoke(f, OutputObject, OutDim, In1Object, In1Dim, In2Object, In2Dim);
}

// Should we wrap all the objects from pdb::Vector to a simple std::vector? The answer is not.
template <typename InvokerType, typename InputType, typename OutputType>
bool GPUInvoke(InvokerType& f, pdb::Handle<pdb::Vector<OutputType>> Out, pdb::Handle<pdb::Vector<InputType>> In1, pdb::Handle<pdb::Vector<InputType>> In2){
    auto In1Object = In1->c_ptr();
    std::vector<size_t> In1Dim{In1->size()};
    auto In2Object = In2->c_ptr();
    std::vector<size_t> In2Dim{In2->size()};
    auto OutObject = Out->c_ptr();
    std::vector<size_t> OutDim{Out->size()};
    return SimpleTypeGPUInvoke(f, OutObject, OutDim, In1Object, In1Dim, In2Object, In2Dim);
}

/*
template <typename InvokerType, typename InputType, typename OutputType>
bool GPUInvoke(InvokerType& f, pdb::Handle<pdb::Vector<OutputType>> Out, std::vector<size_t>& OutDim, pdb::Handle<pdb::Vector<InputType>> In1, std::vector<size_t>& In1Dim, pdb::Handle<pdb::Vector<InputType>> In2, std::vector<size_t>& In2Dim){
    auto In1Object = In1->c_ptr();
    auto In2Object = In2->c_ptr();
    auto OutObject = Out->c_ptr();
    return SimpleTypeGPUInvoke(f, OutObject, OutDim, In1Object, In1Dim, In2Object, In2Dim);
}
*/

bool GPUInvoke(pdb::PDBCUDAMatrixMultipleInvoker& f, pdb::Handle<pdb::Vector<float>> Out, std::vector<size_t>& OutDim, pdb::Handle<pdb::Vector<float>> In1, std::vector<size_t>& In1Dim, pdb::Handle<pdb::Vector<float> > In2, std::vector<size_t>& In2Dim){
    auto In1Object = In1->c_ptr();
    auto In2Object = In2->c_ptr();
    auto OutObject = Out->c_ptr();
    return SimpleTypeGPUInvoke(f, OutObject, OutDim, In1Object, In1Dim, In2Object, In2Dim);
}
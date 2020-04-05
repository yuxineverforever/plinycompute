#include "PDBCUDAGPUInvoke.h"

/** SimpleTypeGPUInvoke deals with all the primitive types and invoke the gpu kernel for the input/output
 * `Out` vector should be reserved before passing as parameter
 * @tparam InvokerType - operator type (should be a derived type from PDBCUDAOp)
 * @tparam InputType - trivial copyable type for input params
 * @tparam OutputType - trivial copyable type for output params
 * @param f - operator
 * @param Out - output param
 * @param OutDim - output param dimension
 * @param In1 - input param 1
 * @param In1Dim - input param 1 dimension
 * @param In2 - input param 2
 * @param In2Dim - input param 2 dimension
 * @return bool - successful or not
 */
template<typename InvokerType, typename InputType, typename OutputType>
typename std::enable_if_t<is_base_of<pdb::PDBCUDAOpInvoker, InvokerType>::value && std::is_trivially_copyable<OutputType>::value && std::is_trivially_copyable<InputType>::value, bool>
SimpleTypeGPUInvoke(InvokerType& f, OutputType* Out, std::vector<size_t>& OutDim, InputType* In1, std::vector<size_t>& In1Dim, InputType* In2, std::vector<size_t>& In2Dim){
    f.setInput(In1, In1Dim);
    f.setInput(In2, In2Dim);
    f.setOutput(Out, OutDim);
    bool res = f.invoke();
    return res;
}

/** SimpleTypeGPUInvoke deals with all the primitive types and invoke the gpu kernel for the input/output
 * `Out` vector should be reserved before passing as parameter
 * @tparam InvokerType - operator type (should be a derived type from PDBCUDAOp)
 * @tparam InputType - trivial copyable type for input params
 * @tparam OutputType - trivial copyable type for output params
 * @param f - operator
 * @param Out - output param
 * @param OutDim - output param dimension
 * @param In1 - input param 1
 * @param In1Dim - input param 1 dimension
 * @return bool - successful or not
 */
template<typename InvokerType, typename InputType, typename OutputType>
typename std::enable_if_t<is_base_of<pdb::PDBCUDAOpInvoker, InvokerType>::value && std::is_trivially_copyable<OutputType>::value && std::is_trivially_copyable<InputType>::value, bool>
SimpleTypeGPUInvoke(InvokerType& f, OutputType* Out, std::vector<size_t>& OutDim, InputType* In1, std::vector<size_t>& In1Dim){
    f.setInput(In1, In1Dim);
    f.setOutput(Out, OutDim);
    bool res = f.invoke();
    return res;
}

/**
 * GPUInvoke for handling the case that both input/output param is Handle<SimpleType>
 * @tparam InvokerType - operator type (should be a derived type from PDBCUDAOp)
 * @tparam InputType - trivial copyable type for input params
 * @tparam OutputType - trivial copyable type for output params
 * @param f - operator
 * @param Out - output param
 * @param In1 - input param 1
 * @param In2 - input param 2
 * @return bool - successful or not
 */
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


/**
 * GPUInvoke for handling the case that dimensions for all the input/output params are 1 dimensional array
 * @tparam InvokerType - operator type (should be a derived type from PDBCUDAOp)
 * @tparam InputType - trivial copyable type for input params
 * @tparam OutputType - trivial copyable type for output params
 * @param f - operator
 * @param Out - output param
 * @param In1 - input param 1
 * @param In2 - input param 2
 * @return bool - successful or not
 */
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

/**
 * GPUInvoke just allow trivial copyable types. Handle 1 input param case.
 * @tparam InvokerType - operator type (should be a derived type from PDBCUDAOp)
 * @tparam InputType - trivial copyable type for input params
 * @tparam OutputType - trivial copyable type for output params
 * @param f - operator
 * @param Out - output param
 * @param OutDim - output param dimension
 * @param In1 - input param 1
 * @param In1Dim - input param 1 dimension
 * @return bool - successful or not
 */
template <typename InvokerType, typename InputType, typename OutputType>
bool GPUInvoke(InvokerType& f, pdb::Handle<pdb::Vector<OutputType>> Out, std::vector<size_t>& OutDim, pdb::Handle<pdb::Vector<InputType>> In1, std::vector<size_t>& In1Dim){
    auto In1Object = In1->c_ptr();
    auto OutObject = Out->c_ptr();
    return SimpleTypeGPUInvoke(f, OutObject, OutDim, In1Object, In1Dim);
}

/**
 * GPUInvoke just allow trivial copyable types. Handle 2 input param case.
 * @tparam InvokerType - operator type (should be a derived type from PDBCUDAOp)
 * @tparam InputType - trivial copyable type for input params
 * @tparam OutputType - trivial copyable type for output params
 * @param f - operator
 * @param Out - output param
 * @param OutDim - output param dimension
 * @param In1 - input param 1
 * @param In1Dim - input param 1 dimension
 * @param In2 - input param 2
 * @param In2Dim - input param 2 dimension
 * @return bool - successful or not
 */
template <typename InvokerType, typename InputType, typename OutputType>
bool GPUInvoke(InvokerType& f, pdb::Handle<pdb::Vector<OutputType>> Out, std::vector<size_t>& OutDim, pdb::Handle<pdb::Vector<InputType>> In1, std::vector<size_t>& In1Dim, pdb::Handle<pdb::Vector<InputType> > In2, std::vector<size_t>& In2Dim){
    auto In1Object = In1->c_ptr();
    auto In2Object = In2->c_ptr();
    auto OutObject = Out->c_ptr();
    return SimpleTypeGPUInvoke(f, OutObject, OutDim, In1Object, In1Dim, In2Object, In2Dim);
}


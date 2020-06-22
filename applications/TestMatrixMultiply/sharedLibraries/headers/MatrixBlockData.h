#pragma once

#include <Object.h>
#include <PDBVector.h>
#include <PDBCUDAVectorAddInvoker.h>
#include <PDBCUDAGPUInvoke.h>
#include <PDBCUDAMemoryAllocatorState.h>

namespace pdb {

// the sub namespace
namespace matrix {

class MatrixBlockData : public pdb::Object {

public:

  /**
   * The default constructor
   */
  MatrixBlockData(){
      isGPU = true;
  };

  MatrixBlockData(uint32_t numRows, uint32_t numCols, bool onGPU) : numRows(numRows), numCols(numCols), isGPU(onGPU){
    // allocate the data
    data = makeObject<Vector<float>>(numRows * numCols, numRows * numCols, onGPU, memAllocateState::INSTANT);
  }

  ENABLE_DEEP_COPY

  /**
   * The number of rows in the block
   */
  uint32_t numRows = 0;

  /**
   * The number of columns in the block
   */
  uint32_t numCols = 0;

  /**
   * is the data on GPU
   */
  bool isGPU = false;

  /**
   * The values of the block
   */
  Handle<Vector<float>> data;

  /**
   * Does the summation of the data
   * @param other - the other
   * @return
   */
  MatrixBlockData& operator+(MatrixBlockData& other) {
    float *myData = data->c_ptr();
    float *otherData = other.data->c_ptr();
    size_t length = numRows * numCols;
    vector<size_t> outdim = {length};
    vector<size_t> in1dim = {length};
    pdb::PDBCUDAOpType op = pdb::PDBCUDAOpType ::VectorAdd;
    GPUInvoke(op, data, outdim, other.data, in1dim);
    return *this;
  }

  MatrixBlockData& operator= (const MatrixBlockData& other){

      if (other.isGPU== true && isGPU == false){
            numCols = other.numCols;
            numRows = other.numRows;
            data = other.data;
        } else if (other.isGPU== false && isGPU == false){
            numCols = other.numCols;
            numRows = other.numRows;
            data = other.data;
            //data = makeObject<Vector<float>>(numRows * numCols, numRows * numCols, false);
        } else if (other.isGPU == true && isGPU == true){
            // This is for handling the case:
            // AggregationCombinerSink:61
            // *temp = (*it).value;
            // we do not really deep copy the data from input page to the output page.
            // instead we allocate space on output page and gpu.
            numCols = other.numCols;
            numRows = other.numRows;
            data = makeObject<Vector<float>>(numRows * numCols, numRows * numCols, isGPU, memAllocateState::LAZY);
        } else if (other.isGPU == false && isGPU == true){
            numCols = other.numCols;
            numRows = other.numRows;
            data = makeObject<Vector<float>>(numRows * numCols, numRows * numCols, isGPU, memAllocateState::LAZY);
        }
        return *this;
  }
};

}

}

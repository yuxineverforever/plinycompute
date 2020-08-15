#pragma once

#include <Object.h>
#include <PDBVector.h>
#include <operators/PDBCUDAVectorAddInvoker.h>
#include <PDBCUDAGPUInvoke.h>

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
    data = makeObject<Vector<float>>(numRows * numCols, numRows * numCols, onGPU);
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
    size_t length = numRows * numCols;
    vector<size_t> outdim = {length};
    vector<size_t> in1dim = {length};
    pdb::PDBCUDAOpType op = pdb::PDBCUDAOpType ::VectorAdd;
    GPUInvoke(op, data, outdim, other.data, in1dim);
    return *this;
  }
};

}

}

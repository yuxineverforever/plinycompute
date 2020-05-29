#pragma once

#include <Object.h>
#include <PDBVector.h>
#include <PDBCUDAVectorAddInvoker.h>
#include <PDBCUDAGPUInvoke.h>

namespace pdb {

// the sub namespace
namespace matrix {

class MatrixBlockData : public pdb::Object {
public:

  /**
   * The default constructor
   */
  MatrixBlockData() = default;

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
  bool isGPU;

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

    // get the data
    float *myData = data->c_ptr();
    float *otherData = other.data->c_ptr();

    size_t length = numRows * numCols;
    vector<size_t> outdim = {length};
    vector<size_t> in1dim = {length};

    pdb::PDBCUDAOpType op = pdb::PDBCUDAOpType ::VectorAdd;
    GPUInvoke(op, data, outdim, other.data, in1dim);

    /*
    // sum up the data
    for (int i = 0; i < numRows * numCols; i++) {
    (myData)[i] += (otherData)[i];
    }
    */

    // return me
    return *this;
}
};

}

}

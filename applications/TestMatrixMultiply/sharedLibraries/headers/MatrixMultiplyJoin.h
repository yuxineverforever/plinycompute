#pragma once
#include <LambdaCreationFunctions.h>
#include "JoinComp.h"
#include "MatrixBlock.h"
#include "PDBCUDAUtility.h"
#include "PDBCUDAOpInvoker.h"
#include "PDBCUDAMatrixMultipleInvoker.h"
#include "PDBCUDAGPUInvoke.h"
#include <mkl.h>

namespace pdb {

namespace matrix {

class MatrixMultiplyJoin : public JoinComp <MatrixMultiplyJoin, MatrixBlock, MatrixBlock, MatrixBlock> {
public:

    ENABLE_DEEP_COPY

  MatrixMultiplyJoin() = default;

  static Lambda <bool> getKeySelection (Handle <MatrixBlockMeta> in1, Handle <MatrixBlockMeta> in2) {
    return (makeLambdaFromMember (in1, colID) == makeLambdaFromMember (in2, rowID));
  }

  static Lambda <Handle <MatrixBlock>> getProjection (Handle <MatrixBlock> in1, Handle <MatrixBlock> in2) {
    return makeLambda (in1, in2, [] (Handle <MatrixBlock> &in1, Handle <MatrixBlock> &in2) {

      uint32_t I = in1->data.numRows;
      uint32_t J = in2->data.numCols;
      // K and L should be equal
      uint32_t K = in1->data.numCols;
      // make the output block
      Handle <MatrixBlock> out = makeObject<MatrixBlock>(in1->getRowID(), in2->getColID(), I, J);

      vector<size_t> outdim = {I,J};
      vector<size_t> in1dim = {I,K};
      vector<size_t> in2dim = {K,J};

      pdb::PDBCUDAOpType op = pdb::PDBCUDAOpType ::MatrixMultiple;
      GPUInvoke(op, out->data.data, outdim, in1->data.data, in1dim, in2->data.data, in2dim);
      return out;
    });
  }
};
}
}

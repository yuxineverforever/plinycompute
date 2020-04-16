#include <PDBClient.h>
#include <GenericWork.h>
#include "sharedLibraries/headers/MatrixBlock.h"
#include "sharedLibraries/headers/MatrixScanner.h"
#include "sharedLibraries/headers/MatrixMultiplyJoin.h"
#include "sharedLibraries/headers/MatrixMultiplyAggregation.h"
#include "sharedLibraries/headers/MatrixWriter.h"

using namespace pdb;
using namespace pdb::matrix;

// some constants for the test
const size_t blockSize = 1024;
const uint32_t matrixRows = 10000;
const uint32_t matrixColumns = 10000;
const uint32_t numRows = 16;
const uint32_t numCols = 16;

void initMatrix(pdb::PDBClient &pdbClient, const std::string &set) {

  // make the allocation block
  const pdb::UseTemporaryAllocationBlock tempBlock{blockSize * 1024 * 1024};

  // put the chunks here
  Handle<Vector<Handle<MatrixBlock>>> data = pdb::makeObject<Vector<Handle<MatrixBlock>>>();

  // fill the vector up
  for (uint32_t r = 0; r < numRows; r++) {
    for (uint32_t c = 0; c < numCols; c++) {

      // allocate a matrix
      Handle<MatrixBlock> myInt = makeObject<MatrixBlock>(r, c, matrixRows / numRows, matrixColumns / numCols);

      // init the values
      float *vals = myInt->data.data->c_ptr();
      for (int v = 0; v < (matrixRows / numRows) * (matrixColumns / numCols); ++v) {
        //vals[v] = 1.0f * v + 2.0f;
        vals[v] = 1.0f;
      }

      data->push_back(myInt);
    }
  }

  // init the records
  getRecord(data);

  // send the data a bunch of times
  pdbClient.sendData<MatrixBlock>("myData", set, data);
}


int main(int argc, char* argv[]) {    

  // make a client
  pdb::PDBClient pdbClient(8108, "localhost");

  /// 1. Register the classes

  // now, register a type for user data
  pdbClient.registerType("libraries/libMatrixBlock.so");
  pdbClient.registerType("libraries/libMatrixBlockData.so");
  pdbClient.registerType("libraries/libMatrixBlockMeta.so");
  pdbClient.registerType("libraries/libMatrixMultiplyAggregation.so");
  pdbClient.registerType("libraries/libMatrixMultiplyJoin.so");
  pdbClient.registerType("libraries/libMatrixScanner.so");
  pdbClient.registerType("libraries/libMatrixWriter.so");

  /// 2. Create the set

  // now, create a new database
  pdbClient.createDatabase("myData");

  // now, create the input and output sets
  pdbClient.createSet<MatrixBlock>("myData", "A");
  pdbClient.createSet<MatrixBlock>("myData", "B");
  pdbClient.createSet<MatrixBlock>("myData", "C");

  /// 3. Fill in the data (single threaded)

  initMatrix(pdbClient, "A");
  initMatrix(pdbClient, "B");

  /// 4. Make query graph an run query

  // for allocations
  const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

  auto total_begin = std::chrono::high_resolution_clock::now();
  Handle <Computation> readA = makeObject <MatrixScanner>("myData", "A");
  Handle <Computation> readB = makeObject <MatrixScanner>("myData", "B");
  Handle <Computation> join = makeObject <MatrixMultiplyJoin>();
  join->setInput(0, readA);
  join->setInput(1, readB);
  Handle<Computation> myAggregation = makeObject<MatrixMultiplyAggregation>();
  myAggregation->setInput(join);
  Handle<Computation> myWriter = makeObject<MatrixWriter>("myData", "C");
  myWriter->setInput(myAggregation);

  //TODO this is just a preliminary version of the execute computation before we add back the TCAP generation

  pdbClient.executeComputations({ myWriter });

  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Time Duration: " << std::chrono::duration_cast<std::chrono::duration<float>>(end - total_begin).count()
            << " secs." << std::endl;

  /// 5. Get the set from the

  // grab the iterator
  auto it = pdbClient.getSetIterator<MatrixBlock>("myData", "C");
  while(it->hasNextRecord()) {
    // grab the record
    auto r = it->getNextRecord();
    // write out the values
    float *values = r->data.data->c_ptr();
    for(int i = 0; i < r->data.numRows; ++i) {
      for(int j = 0; j < r->data.numCols; ++j) {
            //std::cout << values[i * r->data.numCols + j] << ", ";
      }
      std::cout << "\n";
    }
    std::cout << "\n\n";
  }
  // shutdown the server
  pdbClient.shutDownServer();

  return 0;
}
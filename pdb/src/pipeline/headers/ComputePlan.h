/*****************************************************************************
 *                                                                           *
 *  Copyright 2018 Rice University                                           *
 *                                                                           *
 *  Licensed under the Apache License, Version 2.0 (the "License");          *
 *  you may not use this file except in compliance with the License.         *
 *  You may obtain a copy of the License at                                  *
 *                                                                           *
 *      http://www.apache.org/licenses/LICENSE-2.0                           *
 *                                                                           *
 *  Unless required by applicable law or agreed to in writing, software      *
 *  distributed under the License is distributed on an "AS IS" BASIS,        *
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
 *  See the License for the specific language governing permissions and      *
 *  limitations under the License.                                           *
 *                                                                           *
 *****************************************************************************/

#ifndef ComputePlan_H
#define ComputePlan_H

#include "Computation.h"
#include "PDBString.h"
#include "Object.h"
#include "LogicalPlan.h"
#include "PDBVector.h"
#include "pipeline/Pipeline.h"
#include "ComputeInfo.h"

namespace pdb {

// this is the basic type that is sent around a PDB cluster to store a computation that PDB is to execute
class ComputePlan {

  // this is the compiled plan
  String TCAPComputation;

  // this is the list of Computation objects that are going to be used to power the plan
  Vector<Handle<Computation>> allComputations;

  // this data structure contains both the compiled TCAP string, as well as an index of all of the computations
  LogicalPlanPtr myPlan;

 public:

  ComputePlan() = default;

  // constructor, takes as input the string to execute, as well as the vector of computations
  ComputePlan(String &TCAPComputation, Vector<Handle<Computation>> &allComputations);

  // this compiles the TCAPComputation string, returning a LogicalPlan object.  The resuting object contains:
  //
  // (1) a graph of individual, SIMD-style operations.  This can be accessed via the getComputations () method.
  // (2) a data structure containing all of the actual Computations that implement those SIMD-style operations,
  //     as well as the Lambdas that are associated with each of those SIMD-style operations.  Particular Computation
  //     objects can be accessed via the getNode () method (note that the argument to getNode () is a string that
  //     names the computation; this string can be obtained via the getComputationName () method on the
  //     AtomicComputation objects stored in the graph of SIMD-style operations.
  //
  LogicalPlanPtr getPlan();

  // Note that once getPlan () has been called, ComputePlan object contains a C++ smart pointer inside of it.
  // IT IS VERY DANGEROUS TO SEND SUCH A POINTER ACCROSS THE NETWORK.  Hence, after calling getPlan () but before
  // this object is sent accross the network or written to disk, the following method MUST be called to avoid
  // sending the smart pointer.
  void nullifyPlanPointer();

  /**
   * Returns a processor for a join
   * @param joinTupleSetName
   * @param numNodes
   * @param numProcessingThreads
   * @param pageQueues
   * @param bufferManager
   * @return
   */
  PageProcessorPtr getProcessorForJoin(const std::string &joinTupleSetName,
                                       size_t numNodes,
                                       size_t numProcessingThreads,
                                       vector<PDBPageQueuePtr> &pageQueues,
                                       PDBBufferManagerInterfacePtr bufferManager);

  // this builds a pipeline between the Computation that produces sourceTupleSetName and the Computation
  // targetComputationName.  Since targetComputationName can have more than one input (in the case of a join,
  // for example) the pipeline to targetComputationName is built on the link producig targetTupleSetName.
  //
  // The lambda getPage is used by the pipeline to obtain new temp pages; it is assumed that a page returned
  // by getPage will remain pinned until either discardTempPage or writeBackPage are called.  The former is
  // called if the page can safely be destroyed because it has no useful data.  The latter is called if the
  // page stores a pdb :: Object that contains the result of the computation.

  PipelinePtr buildPipeline(std::string sourceTupleSetName,
                            const std::string &targetTupleSetName,
                            const PDBAbstractPageSetPtr &inputPageSet,
                            const PDBAnonymousPageSetPtr &outputPageSet,
                            std::map<ComputeInfoType, ComputeInfoPtr> &params,
                            size_t numNodes,
                            size_t numProcessingThreads,
                            uint64_t chunkSize,
                            uint64_t workerID);

  PipelinePtr buildAggregationPipeline(const std::string &targetTupleSetName,
                                       const PDBAbstractPageSetPtr &inputPageSet,
                                       const PDBAnonymousPageSetPtr &outputPageSet,
                                       uint64_t workerID);

  PipelinePtr buildBroadcastJoinPipeline(const string &targetTupleSetName,
                                    const PDBAbstractPageSetPtr &inputPageSet,
                                    const PDBAnonymousPageSetPtr &outputPageSet,
                                    uint64_t numThreads,
                                    uint64_t numNodes,
                                    uint64_t workerID);

};

}

#endif

#include "ComputePlan.cc"


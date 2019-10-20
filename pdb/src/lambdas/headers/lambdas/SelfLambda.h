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

#pragma once

#include "Handle.h"
#include <string>
#include "Ptr.h"
#include "TupleSet.h"
#include <vector>
#include "TupleSetMachine.h"

namespace pdb {

template<class ClassType>
class SelfLambda : public TypedLambdaObject<pdb::Ptr<ClassType>> {
public:

  // create an att access lambda; offset is the position in the input object where we are going to
  // find the input att
  SelfLambda(Handle<ClassType> &input) {
    inputTypeName = getTypeName<ClassType>();
    this->setInputIndex(0, -((input.getExactTypeInfoValue() + 1)));
  }

  std::string getTypeOfLambda() const override {
    return std::string("self");
  }

  std::string typeOfAtt() {
    return inputTypeName;
  }

  std::string getInputType() {
    return inputTypeName;
  }

  unsigned int getNumInputs() override {
    return 1;
  }

  /**
   * Returns the additional information about this lambda currently lambda type
   * @return the map
   */
  std::map<std::string, std::string> getInfo() override {

    // fill in the info
    return std::map<std::string, std::string>{
        std::make_pair ("lambdaType", getTypeOfLambda()),
    };
  };

  ComputeExecutorPtr getExecutor(TupleSpec &inputSchema,
                                 TupleSpec &attsToOperateOn,
                                 TupleSpec &attsToIncludeInOutput) override {

    // create the output tuple set
    TupleSetPtr output = std::make_shared<TupleSet>();

    // create the machine that is going to setup the output tuple set, using the input tuple set
    TupleSetSetupMachinePtr myMachine =
        std::make_shared<TupleSetSetupMachine>(inputSchema, attsToIncludeInOutput);

    // this is the input attribute that we will process
    std::vector<int> matches = myMachine->match(attsToOperateOn);
    int whichAtt = matches[0];

    // this is the output attribute
    int outAtt = (int) attsToIncludeInOutput.getAtts().size();

    return std::make_shared<ApplyComputeExecutor>(
        output,
        [=](TupleSetPtr input) {

          // set up the output tuple set
          myMachine->setup(input, output);

          // get the columns to operate on
          std::vector<Handle<ClassType>> &inputColumn = input->getColumn<Handle<ClassType>>(whichAtt);

          // setup the output column, if it is not already set up
          if (!output->hasColumn(outAtt)) {
            auto *outputCol = new std::vector<Ptr<ClassType>>;
            output->addColumn(outAtt, outputCol, true);
          }

          // get the output column
          std::vector<Ptr<ClassType>> &outColumn = output->getColumn<Ptr<ClassType>>(outAtt);

          // loop down the columns, setting the output
          auto numTuples = inputColumn.size();
          outColumn.resize(numTuples);
          for (int i = 0; i < numTuples; i++) {
            outColumn[i] = (ClassType *) &(*(inputColumn[i]));
          }

          return output;
        });
  }

  std::string inputTypeName;
};
}
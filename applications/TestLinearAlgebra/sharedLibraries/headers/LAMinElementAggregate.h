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
#ifndef LA_MINELEMENT_AGGREGATE_H
#define LA_MINELEMENT_AGGREGATE_H

// by Binhang, May 2017

#include "Lambda.h"
#include "LambdaCreationFunctions.h"
#include "AggregateComp.h"
#include "LAMinElementValueType.h"
#include "LAMinElementOutputType.h"
#include "MatrixBlock.h"


using namespace pdb;

class LAMinElementAggregate
    : public AggregateComp<LAMinElementOutputType, MatrixBlock, int, LAMinElementValueType> {

public:

    ENABLE_DEEP_COPY

    LAMinElementAggregate() {}

    // the key type must have == and size_t hash () defined
    Lambda<int> getKeyProjection(Handle<MatrixBlock> aggMe) override {
        return makeLambda(aggMe, [](Handle<MatrixBlock>& aggMe) { return 1; });
    }

    // the value type must have + defined
    Lambda<LAMinElementValueType> getValueProjection(Handle<MatrixBlock> aggMe) override {
        return makeLambdaFromMethod(aggMe, getMinElementValue);
    }
};


#endif

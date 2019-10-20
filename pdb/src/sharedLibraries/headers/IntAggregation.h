#include <utility>

#include <utility>

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

#include "AggregateComp.h"
#include "LambdaCreationFunctions.h"
#include "SumResult.h"

using namespace pdb;

class IntAggregation : public AggregateComp<SumResult, int, int, int> {
public:

  ENABLE_DEEP_COPY

  IntAggregation() = default;

  // the key type must have == and size_t hash () defined
  Lambda<int> getKeyProjection(Handle<int> aggMe) override {
    return makeLambda(aggMe, [](Handle<int>& aggMe) { return 0; });
  }

  // the value type must have + defined
  Lambda<int> getValueProjection(Handle<int> aggMe) override {
    return makeLambda(aggMe, [](Handle<int>& aggMe) { return 1; });
  }
};

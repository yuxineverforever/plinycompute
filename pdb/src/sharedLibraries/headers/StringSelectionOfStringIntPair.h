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

// by Jia, May 2017

#include "SelectionComp.h"
#include "StringIntPair.h"
#include "LambdaCreationFunctions.h"

using namespace pdb;

class StringSelectionOfStringIntPair : public SelectionComp<String, StringIntPair> {

 public:
  ENABLE_DEEP_COPY

  StringSelectionOfStringIntPair() = default;

  Lambda<bool> getSelection(Handle<StringIntPair> checkMe) override {
    return makeLambda(checkMe, [](Handle<StringIntPair>& checkMe) {
      return ((*checkMe).myInt % 3 == 0) && ((*checkMe).myInt < 1000);
    });
  }

  Lambda<Handle<String>> getProjection(Handle<StringIntPair> checkMe) override {
    return makeLambdaFromMember(checkMe, myString);
  }
};
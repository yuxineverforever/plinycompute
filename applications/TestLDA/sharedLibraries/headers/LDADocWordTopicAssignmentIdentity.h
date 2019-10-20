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
#ifndef LDA_DOC_WORD_TOPIC_IDENT_H
#define LDA_DOC_WORD_TOPIC_IDENT_H

#include "SelectionComp.h"
#include "Lambda.h"
#include "LDADocWordTopicAssignment.h"
#include "LambdaCreationFunctions.h"

using namespace pdb;

class LDADocWordTopicAssignmentIdentity : public SelectionComp<LDADocWordTopicAssignment, LDADocWordTopicAssignment> {

public:
    ENABLE_DEEP_COPY

    LDADocWordTopicAssignmentIdentity() {}

    Lambda<bool> getSelection(Handle<LDADocWordTopicAssignment> in) override {
        return makeLambda(in, [&](Handle<LDADocWordTopicAssignment>& in) { return true; });
    }

    Lambda<Handle<LDADocWordTopicAssignment>> getProjection(Handle<LDADocWordTopicAssignment> in) override {
        return makeLambda(in, [&](Handle<LDADocWordTopicAssignment>& in) { return in; });
    }
};

#endif

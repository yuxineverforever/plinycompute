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
#ifndef LDA_INITIAL_TOPIC_PROB_SELECT_H
#define LDA_INITIAL_TOPIC_PROB_SELECT_H

#include "Lambda.h"
#include "LambdaCreationFunctions.h"
#include "SelectionComp.h"
#include "PDBVector.h"
#include "SumResult.h"
#include "IntDoubleVectorPair.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <random>
#include <gsl/gsl_vector.h>

/* The class for initializing the topic-probability for a document */
using namespace pdb;
class LDAInitialTopicProbSelection : public SelectionComp<IntDoubleVectorPair, SumResult> {

private:
    Vector<double> prior;
    Handle<Vector<char>> myMem;

public:
    ENABLE_DEEP_COPY

    LDAInitialTopicProbSelection() {}
    LDAInitialTopicProbSelection(Vector<double> fromPrior) {

        this->prior = fromPrior;

	/* Prepare the random number genrator */
        /* Set up the gsl_rng *src */
        gsl_rng* src = gsl_rng_alloc(gsl_rng_mt19937);
        std::random_device rd;
        std::mt19937 gen(rd());
        gsl_rng_set(src, gen());

        /* Allocate space needed for myRand */
        int spaceNeeded = sizeof(gsl_rng) + src->type->size;
        myMem = makeObject<Vector<char>>(spaceNeeded, spaceNeeded);

        /* Copy src over */
        memcpy(myMem->c_ptr(), src, sizeof(gsl_rng));
        memcpy(myMem->c_ptr() + sizeof(gsl_rng), src->state, src->type->size);

        gsl_rng_free(src);
    }

    Lambda<bool> getSelection(Handle<SumResult> checkMe) override {
        return makeLambda(checkMe, [](Handle<SumResult>& checkMe) { return true; });
    }

    Lambda<Handle<IntDoubleVectorPair>> getProjection(Handle<SumResult> checkMe) override {
        return makeLambda(checkMe, [&](Handle<SumResult>& checkMe) {

            gsl_rng* rng = getRng();
            Handle<IntDoubleVectorPair> result = makeObject<IntDoubleVectorPair>();
            int topicNum = this->prior.size();
            result->setInt(checkMe->getKey());
            Handle<Vector<double>> mySamples = makeObject<Vector<double>>(topicNum, topicNum);

	    /* Sample for the topic-probability */
            gsl_ran_dirichlet(rng, topicNum, this->prior.c_ptr(), mySamples->c_ptr());

            result->setVector(mySamples);

            return result;
        });
    }

    /* Get the GSL RNG from myMem */
    gsl_rng* getRng() {
        gsl_rng* dst = (gsl_rng*)myMem->c_ptr();
        dst->state = (void*)(myMem->c_ptr() + sizeof(gsl_rng));
        dst->type = gsl_rng_mt19937;
        return dst;
    }

    Vector<double> getPrior() {
        return this->prior;
    }
};


#endif

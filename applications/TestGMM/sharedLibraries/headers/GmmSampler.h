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
#ifndef PDB_SAMPLER_H
#define PDB_SAMPLER_H

#include "KMeansDoubleVector.h"
#include <math.h>
#include <stdio.h>

/* This class wraps utilities for sampling data */
namespace pdb {

class GmmSampler {

public:
  static double numStd(int sampleSizeLowerBound) {
    // to make it tight
    if (sampleSizeLowerBound < 6.0) {
      return 12.0;
    } else if (sampleSizeLowerBound < 16.0) {
      return 9.0;
    } else {
      return 6.0;
    }
  }

  static double computeFractionForSampleSize(int sampleSizeLowerBound,
                                             long total, bool withReplacement) {

    if (withReplacement) {
      return fmax(sampleSizeLowerBound +
                      numStd(sampleSizeLowerBound) * sqrt(sampleSizeLowerBound),
                  1e-15) /
             total;
    } else {
      double fraction = (double)sampleSizeLowerBound / (double)(total);
      double delata = 1e-4;
      double gamma = -log(delata) / total;
      return fmin(1,
                  fmax(1e-10, fraction + gamma +
                                  sqrt(gamma * gamma + 2 * gamma * fraction)));
    }
  }

  // srand must be initialized before invoking below function
  static void randomizeInPlace(std::vector<Handle<KMeansDoubleVector>> &samples) {

    size_t mySize = samples.size();
    for (int i = mySize - 1; i >= 0; i--) {
      int j = rand() % (i + 1);
      Handle<KMeansDoubleVector> tmp = samples[j];
      samples[j] = samples[i];
      samples[i] = tmp;
    }
  }

  // srand must be initialized before invoking below function
  static void randomizeInPlace(Vector<Handle<KMeansDoubleVector>> &samples) {

    size_t mySize = samples.size();
    for (int i = mySize - 1; i >= 0; i--) {
      int j = rand() % (i + 1);
      Handle<KMeansDoubleVector> tmp = samples[j];
      samples[j] = samples[i];
      samples[i] = tmp;
    }
  }
};
}

#endif

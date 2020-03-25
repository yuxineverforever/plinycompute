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

#ifndef CAT_STO_GET_PAGE_FOR_OBJECT_REQ_H
#define CAT_STO_GET_PAGE_FOR_OBJECT_REQ_H

#include "Object.h"
#include "Handle.h"
#include "PDBString.h"
#include "PDBSet.h"
#include "BufManagerRequestBase.h"

// PRELOAD %BufGetPageForObjectRequest%

namespace pdb {
// encapsulates a request to obtain a type name from the catalog
    class BufGetPageForObjectRequest : public BufManagerRequestBase {

    public:

        BufGetPageForObjectRequest() = default;

        ~BufGetPageForObjectRequest() = default;

        BufGetPageForObjectRequest(void* thisObj) : objectAddress(thisObj) {
        }

        explicit BufGetPageForObjectRequest(const pdb::Handle<BufGetPageForObjectRequest>& copyMe) : BufManagerRequestBase(*copyMe) {
            // copy stuff
            objectAddress = copyMe->objectAddress;
        }

        ENABLE_DEEP_COPY

        void* objectAddress;
    };
}

#endif

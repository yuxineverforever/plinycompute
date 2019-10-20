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

#ifndef DELETE_SET_H
#define DELETE_SET_H

#include "Object.h"
#include "PDBString.h"
#include <vector>
#include <iostream>
#include <memory>

// PRELOAD %DeleteSet%

namespace pdb {

// this corresponds to a database set
class DeleteSet : public Object {

public:
    ENABLE_DEEP_COPY

    DeleteSet() {}
    ~DeleteSet() {}

    DeleteSet(std::string dbNameIn, std::string setNameIn) {
        dbName = dbNameIn;
        setName = setNameIn;
    }

    std::string whichDatabase() {
        return dbName;
    }

    std::string whichSet() {
        return setName;
    }

private:
    String dbName;
    String setName;
};
}

#endif

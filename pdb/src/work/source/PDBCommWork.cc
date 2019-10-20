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

#ifndef PDB_COMM_WORK_C
#define PDB_COMM_WORK_C

#include "PDBCommWork.h"
#include <iostream>
#include <PDBServer.h>

namespace pdb {

PDBCommunicatorPtr PDBCommWork::getCommunicator() {
    return myCommunicator;
}

void PDBCommWork::setGuts(PDBCommunicatorPtr toMe, pdb::PDBServer *server) {
    this->myCommunicator = std::move(toMe);
    this->server = server;
}
}

#endif

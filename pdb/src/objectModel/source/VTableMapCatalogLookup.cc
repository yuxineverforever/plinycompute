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

#ifndef VTABLEMAP_CAT_LOOKUP_CC
#define VTABLEMAP_CAT_LOOKUP_CC

#include <dlfcn.h>
#include <unistd.h>
#include <unistd.h>
#include "PDBDebug.h"
#include "PDBLogger.h"
#include <cctype>
#include "PDBCatalogClient.h"

namespace pdb {

// note: this should only be called while protected by a lock on the vTableMap
void* VTableMap::getVTablePtrUsingCatalog(int16_t objectTypeID) {

    // in this case, we do not have the vTable pointer for this type, so we will try to load it
    if (theVTable->catalog == nullptr) {
        if (theVTable->logger != nullptr) {
            if (objectTypeID >= 8191) {
                theVTable->logger->error(
                    std::string("unable to obtain shared library file for typeId=") +
                    std::to_string(objectTypeID));
            }
            return nullptr;
        }
    }

    // return if the type is unknown b/c it won't be found neither in
    // an .so library nor in the Catalog
    if (objectTypeID == 0) {
        PDB_COUT << "This is typeId=0, just return!!!!!  " << endl;
        return nullptr;
    }

    std::string sharedLibraryFile = "/var/tmp/objectFile.";
    sharedLibraryFile += to_string(getpid()) + "." + to_string(objectTypeID) + ".so";
    PDB_COUT << "VTableMap:: to get sharedLibraryFile =" << sharedLibraryFile << std::endl;
    if (theVTable->logger != nullptr) {
        theVTable->logger->debug(std::string("VTableMap:: to get sharedLibraryFile =") + sharedLibraryFile);
    }
    unlink(sharedLibraryFile.c_str());
    PDB_COUT << "VTableMap:: to get shared for objectTypeID=" << objectTypeID << std::endl;
    bool ret = theVTable->catalog->getSharedLibrary(objectTypeID, sharedLibraryFile);

    // we should stop here if someone else updated the VTable
    void* returnVal = theVTable->allVTables[objectTypeID];
    if (returnVal != nullptr) {
        return returnVal;
    }

    // we need check return value
    if (ret == false) {
        std::cout << "Error fixing VTableMap for objectTypeID=" << objectTypeID << std::endl;
        std::cout << "   This could be because: 1) the name used to retrieve the shared library "
                     "doesn't match the types in the catalog, or"
                  << std::endl;
        std::cout << "   2) a shared library for that type has not been registered in the catalog"
                  << std::endl;
        return nullptr;
    }

    // open up the shared library

    void* so_handle = dlopen(sharedLibraryFile.c_str(), RTLD_LOCAL | RTLD_LAZY);
    theVTable->so_handles.push_back(so_handle);

    if (!so_handle) {
        const char* dlsym_error = dlerror();
        if (theVTable->logger != nullptr)
            theVTable->logger->error("Cannot load Stored Data Type library: " + sharedLibraryFile +
                                     " error " + (std::string)dlsym_error + '\n');
        std::cout << "Error == " <<(std::string)dlsym_error << std::endl;
        // if we were able to open it
    } else {
        const char* dlsym_error = dlerror();

        // first we need to correctly set all of the global variables in the shared library
        typedef void setGlobalVars(Allocator*, VTableMap*, void*, void*);
        std::string getInstance = "setAllGlobalVariables";
        PDB_COUT << "to set global variables" << std::endl;
        setGlobalVars* setGlobalVarsFunc = (setGlobalVars*)dlsym(so_handle, getInstance.c_str());
        // see if we were able to get the function
        if ((dlsym_error = dlerror())) {
            if (theVTable->logger != nullptr)
                theVTable->logger->error(
                    "Error, can't set global variables in .so file; error is " +
                    (std::string)dlsym_error + "\n");
            std::cout << "ERROR: we were not able to get the function" << std::endl;
            return nullptr;
            // if we were able to, then run it
        } else {
            setGlobalVarsFunc(mainAllocatorPtr, theVTable, stackBase, stackEnd);
            PDB_COUT << "Successfully set global variables" << std::endl;
        }

        // get the function that will give us access to the vTable
        typedef void* getObjectVTable();
        getInstance = "getObjectVTable";
        getObjectVTable* getObjectFunc = (getObjectVTable*)dlsym(so_handle, getInstance.c_str());

        // see if we were able to get the function
        if ((dlsym_error = dlerror())) {
            if (theVTable->logger != nullptr)
                theVTable->logger->error("Error, can't load function getInstance (); error is " +
                                         (std::string)dlsym_error + "\n");
            std::cout << "ERROR: we were not able to load function getObjectVTable" << std::endl;
            return nullptr;
            // if we were able to, then run it
        } else {
            theVTable->allVTables[objectTypeID] = getObjectFunc();
            PDB_COUT << "VTablePtr for objectTypeID=" << objectTypeID
                     << " is set in allVTables to be " << theVTable->allVTables[objectTypeID]
                     << std::endl;
        }
    }
    return theVTable->allVTables[objectTypeID];
}

int16_t VTableMap::lookupTypeNameInCatalog(std::string objectTypeName) {

    PDB_COUT << "invoke lookupTypeNameInCatalog for objectTypeName=" << objectTypeName << std::endl;

    std::string error;
    auto type = theVTable->catalog->getType(objectTypeName, error);

    // if we find the type return -1 as the identifier
    if(type == nullptr) {
        // log the error
        PDB_COUT << error;
        return -1;
    }

    // return the id
    return (int16_t) type->id;
}


} /* namespace pdb */

#endif

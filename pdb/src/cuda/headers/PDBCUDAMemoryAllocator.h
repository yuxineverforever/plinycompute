#pragma once

#include <iostream>
#include "PDBRamPointer.h"

/**
 * here is one State flag for memory allocation.
 * INSTANT: get GPU space allocated instantly
 * LAZY: get GPU space allocated later
 */

void *memMalloc(size_t size);

void memFree(void *ptr);

pdb::RamPointerReference keepMemAddress(void *gpuaddress, void *cpuaddress, size_t numbytes, size_t headerbytes);


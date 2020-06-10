#pragma once
#include <iostream>
#include "PDBRamPointer.h"

void* memMalloc(size_t size);

void memFree(void* ptr);

pdb::RamPointerReference keepMemAddress(void* gpuaddress, void* cpuaddress, size_t numbytes, size_t headerbytes);


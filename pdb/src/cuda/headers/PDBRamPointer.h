#pragma once
#include <iostream>
#include <memory>
#include <list>

namespace pdb{
    // TODO: add comment
    // Here is the Ram Pointer which can point to both CPU/GPU RAM
    /**
     *
     */

    class RamPointer{

    public:

        RamPointer(void* physicalAddress, size_t numbytes, size_t headerbytes): ramAddress(physicalAddress), numBytes(numbytes), headerBytes(headerbytes){
        }
        ~RamPointer(){}
        void push_back_pointer(void* pointer){
            cpuPointers.push_back(pointer);
        }
        void delete_pointer(void* pointer){
            cpuPointers.remove(pointer);
        }
        inline bool operator== (const RamPointer& rp) const {
            return ramAddress == rp.ramAddress;
        }
        inline bool operator < (const RamPointer& rp) const {
            return ramAddress < rp.ramAddress;
        }
        inline bool operator > (const RamPointer& rp) const {
            return ramAddress > rp.ramAddress;
        }
    public:
        void* ramAddress;
        size_t numBytes;
        size_t headerBytes;
        std::list<void*> cpuPointers;
    };
    using RamPointerPtr = std::shared_ptr<RamPointer>;
}

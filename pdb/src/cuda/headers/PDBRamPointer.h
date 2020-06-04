#pragma once
#include <iostream>
#include <memory>
namespace pdb{

    // Here is the Ram Pointer which can point to both CPU/GPU RAM
    class RamPointer{

    public:

        RamPointer(void* physicalAddress, size_t numbytes, size_t headerbytes): ramAddress(physicalAddress), numBytes(numbytes), headerBytes(headerbytes){

        }

        RamPointer() = delete;

        inline bool operator== (const RamPointer& rp){
            return ramAddress == rp.ramAddress;
        }

    public:
        void* ramAddress;
        size_t numBytes;
        size_t headerBytes;
    };

    using RamPointerReference = std::shared_ptr<RamPointer>;
};

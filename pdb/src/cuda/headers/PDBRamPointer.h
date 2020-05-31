#include <iostream>
#include <memory>
namespace pdb{

    // Here is the Ram Pointer which can point to both CPU/GPU RAM
    class RamPointer{
    public:
        RamPointer(void* physicalAddress): ramAddress(physicalAddress){}

        RamPointer() = delete;

        inline bool operator== (const RamPointer& rp){
            return ramAddress == rp.ramAddress;
        }

    public:
        void* ramAddress;
    };
    using RamPointerReference = std::shared_ptr<RamPointer>;
};

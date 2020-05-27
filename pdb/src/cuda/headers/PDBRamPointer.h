#include <iostream>
#include <memory>
namespace pdb{

    // Here is the Ram Pointer which can point to both CPU/GPU RAM
    class RamPointer{
        public:
            void* ramAddress;
    };
    using RamPointerReference = std::shared_ptr<RamPointer>;
};

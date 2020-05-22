#include <iostream>
#include <memory>

namespace pdb{

    class RamPointer{

    public:
        void* ramAddress;
    };
    using RamPointerReference = std::shared_ptr<RamPointer>;
};

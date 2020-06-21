#pragma once

#include <iostream>
#include <memory>
#include <list>

namespace pdb {
    // TODO: add comment
    // Here is the Ram Pointer which can point to both CPU/GPU RAM
    /**
     *
     */

    class RamPointer {

    public:

        RamPointer(void *physicalAddress, size_t numbytes, size_t headerbytes) : ramAddress(physicalAddress),
                                                                                 numBytes(numbytes),
                                                                                 headerBytes(headerbytes) {
            refCount = 0;
            isDirty = false;
        }

        ~RamPointer() {
            std::cout << "RamPointer destructor!\n";
        }

        void push_back_pointer(void *pointer) {
            cpuPointers.push_back(pointer);
        }

        void delete_pointer(void *pointer) {
            cpuPointers.remove(pointer);
        }

        void setDirty() {
            isDirty = true;
        }

        inline bool operator==(const RamPointer &rp) const {
            return ramAddress == rp.ramAddress;
        }

        inline bool operator<(const RamPointer &rp) const {
            return ramAddress < rp.ramAddress;
        }

        inline bool operator>(const RamPointer &rp) const {
            return ramAddress > rp.ramAddress;
        }

        RamPointer &operator=(const RamPointer &rp) {
            ramAddress = rp.ramAddress;
            numBytes = rp.numBytes;
            headerBytes = rp.headerBytes;
            cpuPointers = rp.cpuPointers;
            refCount = rp.refCount;
            isDirty = rp.isDirty;
            return *this;
        }

    public:
        void *ramAddress;
        size_t numBytes;
        size_t headerBytes;
        std::list<void *> cpuPointers;
        int refCount;
        bool isDirty;
    };

    using RamPointerPtr = std::shared_ptr<RamPointer>;

    /**
     * This is just one simple wrapper for the RamPointer Class
     */
    class RamPointerBase {
    public:
        RamPointerBase(RamPointerPtr useMe) {
            ptr = useMe;
        }

        ~RamPointerBase(){
            std::cout << "RamPointerBase destructor!\n";
        }

        void push_back_pointer(void *pointer) {
            ptr->push_back_pointer(pointer);
        }

        void delete_pointer(void *pointer) {
            ptr->delete_pointer(pointer);
        }

        void* get_address() {
            return ptr->ramAddress;
        }

    private:
        RamPointerPtr ptr;
    };

    using RamPointerReference = std::shared_ptr<RamPointerBase>;

}

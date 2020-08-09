#pragma once

#include <iostream>
#include <memory>
#include <list>
#include "PDBCUDAConfig.h"

namespace pdb {

    // TODO: add comment
    // Here is the Ram Pointer which can point to both CPU/GPU RAM
    /**
     *
     */


    class RamPointer {

    public:

        RamPointer(void *physicalAddress, size_t numbytes, size_t headerBytes) : ramAddress(physicalAddress),
                                                                                 numBytes(numbytes),
                                                                                 headerBytes(headerBytes) {
            refCount = 0;
            isDirty = false;
        }

        ~RamPointer() {
            std::cout << "RamPointer destructor!\n";
        }

        void push_back_cpu_pointer(void *pointer) {
            cpuPointers.push_back(pointer);
        }

        void delete_cpu_pointer(void* pointer) {
            cpuPointers.remove(pointer);
        }

        void set_ram_pointer(void* pointer){
            ramAddress = pointer;
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

        // if the ramAddress == nullptr, it means the space is lazy allocated
        // and the ramAddress will be changed in the future
        void* ramAddress;
        size_t numBytes;
        size_t headerBytes;
        std::list<void *> cpuPointers;
        int refCount;
        bool isDirty;
    };


    class RamPointerWithOffset {

    public:

        RamPointerWithOffset(size_t offsetToPage, page_id_t pageID, size_t numBytes, size_t headerBytes) : offset(offsetToPage),
                                                                                 whichPage(pageID),
                                                                                 numBytes(numBytes),
                                                                                 headerBytes(headerBytes) {
            refCount = 0;
            isDirty = false;
        }

        ~RamPointerWithOffset() {
            std::cout << "RamPointerWithOffset destructor!\n";
        }

        void push_back_cpu_pointer(void *pointer) {
            cpuPointers.push_back(pointer);
        }

        void delete_cpu_pointer(void* pointer) {
            cpuPointers.remove(pointer);
        }

        void set_ram_pointer(size_t newOffset){
            offset = newOffset;
        }

        void setDirty() {
            isDirty = true;
        }

    public:

        // if the ramAddress == nullptr, it means the space is lazy allocated
        // and the ramAddress will be changed in the future
        size_t offset;
        page_id_t whichPage;

        size_t numBytes;
        size_t headerBytes;

        std::list<void *> cpuPointers;
        int refCount;
        bool isDirty;
    };

    using RamPointerPtr = std::shared_ptr<RamPointerWithOffset>;

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

        void push_back_cpu_pointer(void *pointer) {
            ptr->push_back_cpu_pointer(pointer);
        }

        void delete_cpu_pointer(void *pointer) {
            ptr->delete_cpu_pointer(pointer);
        }

        void* get_address() {
            return ptr->ramAddress;
        }

    private:
        RamPointerPtr ptr;
    };

    using RamPointerReference = std::shared_ptr<RamPointerBase>;

}

#ifndef PDB_CUDA_PAGE
#define PDB_CUDA_PAGE

#include <cstdint>

namespace pdb{

    using page_id_t = int32_t;
    static constexpr int32_t INVALID_PAGE_ID = -1;

    /**
     * NONE: default value
     * STATIC_PAGE: usually the page for `in` parameter
     * DYNAMIC_PAGE: usually the page for `out` parameter, for serving RamPointer
     */
    enum PDBCUDAPageType{
        NONE,
        STATIC_PAGE,
        DYNAMIC_PAGE
    };

    class PDBCUDAPage{

    public:

        PDBCUDAPage() = default;

        inline void setBytes(char* loc) { data = loc;}

        inline char* getBytes() { return data;}

        inline bool isDirty() { return is_dirty; }

        /** @return the page id of this page */
        inline page_id_t GetPageId() { return page_id; }

        /** @return the pin count of this page */
        inline int GetPinCount() { return pin_count; }

        /** increase the pin count */
        inline void incrementPinCount() { pin_count++; }

        /** decrease the pin count */
        inline void decrementPinCount() { pin_count--;}

        inline void setPageID(page_id_t id) { page_id = id;}

        inline void setPageType(PDBCUDAPageType type) { page_type = type;}

    private:

        PDBCUDAPageType page_type = NONE;
        page_id_t page_id = INVALID_PAGE_ID;
        char* data = nullptr;
        bool is_dirty = false;
        int pin_count = 0;
    };
};


#endif
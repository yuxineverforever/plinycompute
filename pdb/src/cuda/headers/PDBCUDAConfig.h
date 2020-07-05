#ifndef PDB_CUDA_CONFIG
#define PDB_CUDA_CONFIG
namespace pdb{

    using page_id_t = int32_t;
    using frame_id_t  = int32_t;
    using ref_bit = bool;
    using frame_ref_info = std::pair<frame_id_t, ref_bit>;
    using buffer_iter = std::list<frame_ref_info>::iterator;
};
#endif
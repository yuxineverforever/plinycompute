# create the target
add_pdb_integration_test(TestMatrixMultiply)
# add a build dependency to build-tests target
add_dependencies(build-integration-tests TestMatrixMultiply)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/tools/cmake")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.0/bin/nvcc")
set(CMAKE_CUDA_FLAGS "-arch=sm_75")

set (CUDA_INCLUDE_DIR "/usr/local/cuda/include")
set (CUDA_LIBRARY_DIRS "/usr/local/cuda-10.0/lib64/libcublas.so")

set(USE_MKL Yes)

find_package(MKL REQUIRED)
link_directories(${/home/yuxin/snap/intel/mkl/include/})
include_directories("/usr/local/cuda-10.0/samples/common/inc/")
include_directories("${PROJECT_SOURCE_DIR}/pdb/src/cuda/headers/")
include_directories(${MKL_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIR})

message("as ${MKL_LIBRARIES}")

option(USE_CUDA "Use CUDA" ON)
# compile all the objects
file(GLOB files "${PROJECT_SOURCE_DIR}/applications/TestMatrixMultiply/sharedLibraries/source/*.cc")
#file(GLOB cuda_cc_files "${PROJECT_SOURCE_DIR}/pdb/src/cuda/source/*.cc")
file(GLOB cuda_cu_files "${PROJECT_SOURCE_DIR}/pdb/src/cuda/source/*.cu")


foreach(file ${files})
    get_filename_component(fileName "${file}" NAME_WE)
    add_library(${fileName} MODULE ${file})
    target_link_libraries(${fileName} pdb-shared-common)
    target_link_libraries(${fileName} ${MKL_LIBRARIES})
    add_dependencies(shared-libraries ${fileName})
endforeach()

foreach (file ${cuda_cu_files})
    get_filename_component(fileName "${file}" NAME_WE)
    if (USE_CUDA)
        enable_language("CUDA")
        add_library(${fileName} SHARED ${file})
    endif ()
endforeach ()

target_link_libraries(TestMatrixMultiply ${MKL_LIBRARIES})

target_link_libraries(MatrixMultiplyJoin PDBCUDAUtility)
target_link_libraries(TestMatrixMultiply PDBCUDAUtility)

target_link_libraries(TestMatrixMultiply ${CUDA_ADD_CUBLAS_TO_TARGET})
#target_link_libraries(TestMatrixMultiply ${CUDA_LIBRARY_DIRS})

# add build dependencies to shared libraries it uses
add_dependencies(TestMatrixMultiply MatrixBlock)
add_dependencies(TestMatrixMultiply MatrixBlockData)
add_dependencies(TestMatrixMultiply MatrixBlockMeta)
add_dependencies(TestMatrixMultiply MatrixMultiplyAggregation)
add_dependencies(TestMatrixMultiply MatrixMultiplyJoin)
add_dependencies(TestMatrixMultiply MatrixScanner)
add_dependencies(TestMatrixMultiply MatrixWriter)

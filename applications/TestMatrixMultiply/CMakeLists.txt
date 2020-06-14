
# create the target
add_pdb_integration_test(TestMatrixMultiply)

# add a build dependency to build-tests target
add_dependencies(build-integration-tests TestMatrixMultiply)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/tools/cmake")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.0/bin/nvcc")
set(CMAKE_CUDA_FLAGS "-arch=sm_75")

# set CUDA path
set (CUDA_INCLUDE_DIR "/usr/local/cuda/include")
set (CUDA_BLAS_LIB "/usr/local/cuda-10.0/lib64/libcublas.so")
set (CUDA_DART_LIB "/usr/local/cuda-10.0/lib64/libcudart.so")

# use MKL and CUBLAS
set(USE_MKL Yes)
option(USE_CUDA "Use CUDA" ON)

# include MKL and cuda header files
find_package(MKL REQUIRED)
include_directories(${MKL_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIR})
link_directories(${/home/yuxin/snap/intel/mkl/include/})
include_directories("/usr/local/cuda-10.0/samples/common/inc/")
include_directories("${PROJECT_SOURCE_DIR}/pdb/src/cuda/headers/")
message("as ${MKL_LIBRARIES}")

# compile all the objects
file(GLOB files "${PROJECT_SOURCE_DIR}/applications/TestMatrixMultiply/sharedLibraries/source/*.cc")
file(GLOB cuda_source_files "${PROJECT_SOURCE_DIR}/pdb/src/cuda/source/*.cu" "${PROJECT_SOURCE_DIR}/pdb/src/cuda/source/*.cc")

# add all libraries
foreach(file ${files})
    get_filename_component(fileName "${file}" NAME_WE)
    add_library(${fileName} MODULE ${file})
    target_link_libraries(${fileName} pdb-shared-common)
    target_link_libraries(${fileName} ${MKL_LIBRARIES})
    add_dependencies(shared-libraries ${fileName})
endforeach()

include_directories(${CUDA_INCLUDE_DIR})
enable_language("CUDA")
add_library(CUDA_LIBRARIES SHARED ${cuda_source_files})

# link target to CUDA
target_link_libraries(MatrixBlockData cuda-obj)
target_link_libraries(MatrixMultiplyJoin cuda-obj)

# add build dependencies to shared libraries it uses
add_dependencies(TestMatrixMultiply MatrixBlock)
add_dependencies(TestMatrixMultiply MatrixBlockData)
add_dependencies(TestMatrixMultiply MatrixBlockMeta)
add_dependencies(TestMatrixMultiply MatrixMultiplyAggregation)
add_dependencies(TestMatrixMultiply MatrixMultiplyJoin)
add_dependencies(TestMatrixMultiply MatrixScanner)
add_dependencies(TestMatrixMultiply MatrixWriter)

# link target to MKL and CUDA libraries
target_link_libraries(TestMatrixMultiply ${MKL_LIBRARIES})
target_link_libraries(TestMatrixMultiply ${CUDA_DART_LIB})
target_link_libraries(TestMatrixMultiply ${CUDA_BLAS_LIB})
target_link_libraries(TestMatrixMultiply CUDA_LIBRARIES)

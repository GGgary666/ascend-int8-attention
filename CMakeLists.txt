cmake_minimum_required(VERSION 3.20)
project(sharedkv_int8 LANGUAGES C CXX)

if(DEFINED ENV{ASCEND_HOME_PATH})
    set(_cann_default "$ENV{ASCEND_HOME_PATH}")
else()
    set(_cann_default "/usr/local/Ascend/cann-9.2.0")
endif()
set(ASCEND_CANN_PACKAGE_PATH "${_cann_default}" CACHE PATH "CANN installation")
set(ENV{ASCEND_HOME_PATH} "${ASCEND_CANN_PACKAGE_PATH}")
set(CMAKE_ASC_ARCHITECTURES dav-2201 CACHE STRING "Validated NPU instruction architecture")
find_package(asc REQUIRED CONFIG PATHS "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake" NO_DEFAULT_PATH)
enable_language(ASC)

# Keep the p101 translation-unit ordering and every device source byte stable.
ascendc_library(sharedkv_c8 SHARED
    csrc/kernels/c8/c4.cpp csrc/kernels/c8/c4_small.cpp csrc/kernels/c8/c128.cpp)
ascendc_library(sharedkv_int8 SHARED
    csrc/kernels/cache_write.cpp csrc/kernels/sharedkv.cpp csrc/kernels/sharedkv_wide.cpp
    csrc/kernels/sharedkv_packed.cpp csrc/kernels/sharedkv_medium.cpp csrc/kernels/sharedkv_packed_sparse.cpp
    csrc/kernels/sharedkv_adaptive.cpp csrc/kernels/sharedkv_wide_trim.cpp csrc/kernels/sharedkv_paired.cpp
    csrc/kernels/sharedkv_paired_sparse.cpp csrc/kernels/sharedkv_dynamic.cpp
    csrc/kernels/sharedkv_dynamic_dense.cpp csrc/kernels/sharedkv_dynamic_grouped.cpp)

foreach(target sharedkv_c8 sharedkv_int8)
    target_include_directories(${target} PRIVATE
        "${ASCEND_CANN_PACKAGE_PATH}/include"
        "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/tikcfw")
    target_link_directories(${target} PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64")
    target_link_libraries(${target} PRIVATE ascendcl runtime ascendc_runtime)
endforeach()

##############################################################################
#
# CUDA Data Parallel Primitives (CUDPP) CMake file
#
###############################################################################

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../ext/moderngpu/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../ext/cub
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/app
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel
  ${CMAKE_CURRENT_SOURCE_DIR}/cta
  )

set(CCFILES
  cudpp.cpp
  cudpp_plan.cpp
  cudpp_manager.cpp
  ${CMAKE_SOURCE_DIR}/ext/moderngpu/src/mgpuutil.cpp
  )

set (HFILES
  cudpp_manager.h
  cudpp_plan.h
  cuda_util.h
  cudpp_util.h
  cudpp_globals.h
  cudpp_compact.h
  cudpp_compress.h
  cudpp_listrank.h
  cudpp_sa.h
  cudpp_mergesort.h
  cudpp_multisplit.h
  cudpp_radixsort.h
  cudpp_rand.h
  cudpp_reduce.h
  cudpp_stringsort.h
  cudpp_scan.h
  cudpp_segscan.h
  cudpp_spmvmult.h
  sharedmem.h
  )

set (CUHFILES
  cta/compress_cta.cuh
  cta/mergesort_cta.cuh
  cta/radixsort_cta.cuh
  cta/rand_cta.cuh
  cta/scan_cta.cuh
  cta/segmented_scan_cta.cuh
  cta/stringsort_cta.cuh
  kernel/compact_kernel.cuh
  kernel/compress_kernel.cuh
  kernel/listrank_kernel.cuh
  kernel/sa_kernel.cuh
  kernel/mergesort_kernel.cuh
  kernel/multisplit_kernel.cuh
  kernel/radixsort_kernel.cuh
  kernel/rand_kernel.cuh
  kernel/reduce_kernel.cuh
  kernel/segmented_scan_kernel.cuh
  kernel/spmvmult_kernel.cuh
  kernel/stringsort_kernel.cuh
  kernel/vector_kernel.cuh
  kernel/tridiagonal_kernel.cuh
  )

set(CUFILES
  app/reduce_app.cu
  app/compact_app.cu
  app/compress_app.cu
  app/listrank_app.cu
  app/sa_app.cu
  app/mergesort_app.cu
  app/multisplit_app.cu
  app/scan_app.cu
  app/segmented_scan_app.cu
  app/spmvmult_app.cu
  app/stringsort_app.cu
  app/radixsort_app.cu
  app/rand_app.cu
  app/tridiagonal_app.cu
  ${CMAKE_SOURCE_DIR}/ext/moderngpu/src/mgpucontext.cu
  )

set(HFILES_PUBLIC
  ${CMAKE_SOURCE_DIR}/include/cudpp.h
  ${CMAKE_SOURCE_DIR}/include/cudpp_config.h
  )

source_group("CUDA Source Files" FILES ${CUFILES})
source_group("CUDA Header Files" FILES ${CUHFILES})

set(GENCODE_SM12 -gencode=arch=compute_12,code=sm_12 -gencode=arch=compute_12,code=compute_12)
set(GENCODE_SM13 -gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_13,code=compute_13)
set(GENCODE_SM20 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20)
set(GENCODE_SM21 -gencode=arch=compute_20,code=sm_21 -gencode=arch=compute_20,code=compute_20)
set(GENCODE_SM30 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_30,code=compute_30)
set(GENCODE_SM35 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_35,code=compute_35)
set(GENCODE_SM50 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_50,code=compute_50)

#set(GENCODE -gencode=arch=compute_20,code=compute_20) # at least generate PTX

#option(CUDPP_GENCODE_SM12
#       "ON to generate code for Compute Capability 1.2 devices (e.g. GeForce GT 330M)"
#       OFF)

#option(CUDPP_GENCODE_SM13
#       "ON to generate code for Compute Capability 1.3 devices (e.g. Tesla C1060)"
#       OFF)

#option(CUDPP_GENCODE_SM20
#       "ON to generate code for Compute Capability 2.0 devices (e.g. Tesla C2050)"
#       OFF)

#option(CUDPP_GENCODE_SM21
#       "ON to generate code for Compute Capability 2.1 devices (e.g. NVS 4200M)"
#       OFF)

option(CUDPP_GENCODE_SM30
       "ON to generate code for Compute Capability 3.0 devices (e.g. Tesla K10)"
       OFF)

option(CUDPP_GENCODE_SM35
       "ON to generate code for Compute Capability 3.5 devices (e.g. Tesla K20)"
       ON)

option(CUDPP_GENCODE_SM50
       "ON to generate code for Compute Capability 5.0 devices (e.g. GeForce GTX 750)"
       OFF)

if (CUDPP_GENCODE_SM12)
  set(GENCODE ${GENCODE} ${GENCODE_SM12})
endif(CUDPP_GENCODE_SM12)

if (CUDPP_GENCODE_SM13)
  set(GENCODE ${GENCODE} ${GENCODE_SM13})
endif(CUDPP_GENCODE_SM13)

if (CUDPP_GENCODE_SM20)
  set(GENCODE ${GENCODE} ${GENCODE_SM20})
endif(CUDPP_GENCODE_SM20)

if (CUDPP_GENCODE_SM21)
  set(GENCODE ${GENCODE} ${GENCODE_SM21})
endif(CUDPP_GENCODE_SM21)

if (CUDPP_GENCODE_SM30)
  set(GENCODE ${GENCODE} ${GENCODE_SM30})
endif(CUDPP_GENCODE_SM30)

if (CUDPP_GENCODE_SM35)
  set(GENCODE ${GENCODE} ${GENCODE_SM35})
endif(CUDPP_GENCODE_SM35)

if (CUDPP_GENCODE_SM50)
  set(GENCODE ${GENCODE} ${GENCODE_SM50})
endif(CUDPP_GENCODE_SM50)

if (CUDA_VERBOSE_PTXAS)
  set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  exec_program(uname ARGS -v  OUTPUT_VARIABLE DARWIN_VERSION)
  string(REGEX MATCH "[0-9]+" DARWIN_VERSION ${DARWIN_VERSION})
  # message(STATUS "Detected Darwin kernel version: ${DARWIN_VERSION}")
  if ((DARWIN_VERSION GREATER 13) OR (DARWIN_VERSION EQUAL 13))
    set(APPLE_MAVERICKS_OR_LATER 1 INTERNAL)
    set(CMAKE_CXX_FLAGS -stdlib=libstdc++)
    set(CMAKE_C_FLAGS -stdlib=libstdc++)
    if(CUDA_VERSION_MAJOR EQUAL 6 AND CUDA_VERSION_MINOR LESS 5)
      set(CUDA_6_0 1)
    endif(CUDA_VERSION_MAJOR EQUAL 6 AND CUDA_VERSION_MINOR LESS 5)

    if(CUDA_VERSION_MAJOR LESS 6 OR CUDA_6_0)
      if(NOT CUDA_NVCC_FLAGS MATCHES "ccbin")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -ccbin;/usr/bin/clang)
      endif(NOT CUDA_NVCC_FLAGS MATCHES "ccbin")
    endif(CUDA_VERSION_MAJOR LESS 6 OR CUDA_6_0)

  endif ((DARWIN_VERSION GREATER 13) OR (DARWIN_VERSION EQUAL 13))
  # current CUDA on Mavericks uses libstdc++, not libc++
  # https://github.com/cudpp/cudpp/wiki/BuildingCUDPPWithMavericks

endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")


if (CUDPP_BUILD_SHARED_LIBS)
  message ( STATUS "Set export shared library.")
  add_definitions ( -Dcudpp_EXPORTS)
  add_definitions ( -Dcudpp_hash_EXPORTS)
endif()


cuda_add_library( ${TARGET_NAME1}  ${LIB_TYPE}
  ${CCFILES}
  ${HFILES}
  ${CUHFILES}
  ${HFILES_PUBLIC}
  ${CUFILES}
  OPTIONS ${GENCODE} ${VERBOSE_PTXAS}
  )

install(FILES ${HFILES_PUBLIC}
  DESTINATION include
  )

install(TARGETS ${TARGET_NAME1}
  DESTINATION lib
  EXPORT cudpp-targets
  )

install(EXPORT cudpp-targets
  DESTINATION lib
  )

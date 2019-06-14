##############################################################################
#
# CUDA Data Parallel Primitives (CUDPP) CMake file
#
###############################################################################

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/cudpp
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

set(CCFILES
  cudpp_hash.cpp
  debugging.cpp
  hash_compacting.cpp
  hash_multivalue.cpp
  hash_table.cpp
  mt19937ar.cpp
  )

set (HFILES
  debugging.h
  definitions.h
  hash_compacting.h
  hash_functions.h
  hash_multivalue.h
  hash_table.h
  mt19937ar.h
  )

set (CUHFILES
  hash_table.cuh
  )

set(CUFILES
  debugging.cu
  hash_compacting.cu
  hash_multivalue.cu
  hash_table.cu
  hash_functions.cu
  )

set(HFILES_PUBLIC
  ../../include/cudpp_hash.h
  )

source_group("CUDA Source Files" FILES ${CUFILES})
source_group("CUDA Header Files" FILES ${CUHFILES})

if (CUDA_VERBOSE_PTXAS)
  set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)

if (WIN32)
    add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
endif (WIN32)

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

set ( CUDPP_BUILD_SHARED_LIBS OFF CACHE BOOL "Use only static cuddp libs for GVDB" FORCE )

if (CUDPP_BUILD_SHARED_LIBS)
  message ( STATUS "Set export shared library.")
  add_definitions ( -Dcudpp_EXPORTS)
  add_definitions ( -Dcudpp_hash_EXPORTS)
endif()

# Autodetect Target Architecture
unset(CUDA_ARCH_BIN CACHE)
unset(CUDA_ARCH_PTX CACHE)
detect_installed_gpus(CUDA_ARCH_BIN)
set (CUDA_ARCH_BIN "30")
set (CUDA_ARCH_PTX "30")
message ( "CUDA_ARCH_BIN: ${CUDA_ARCH_BIN}")
message ( "CUDA_ARCH_PTX: ${CUDA_ARCH_PTX}")
# Take the detected version, and append the bin and ptx flags assuming ptx ver==bin ver
APPEND_TARGET_ARCH_FLAGS()

cuda_add_library(${TARGET_NAME2} ${LIB_TYPE}
  ${CCFILES}
  ${HFILES}
  ${CUHFILES}
  ${HFILES_PUBLIC}
  ${CUFILES}
  OPTIONS ${GENCODE_SM30} ${VERBOSE_PTXAS}
  )

target_link_libraries( ${TARGET_NAME2} ${TARGET_NAME1} )

install(FILES ${HFILES_PUBLIC}
  DESTINATION include
  )

install(TARGETS ${TARGET_NAME2}
  DESTINATION lib
  EXPORT cudpp_hash-targets
  )

install(EXPORT cudpp_hash-targets
  DESTINATION lib
  )


project(cudpp)

set(cudpp_VERSION_MAJOR 3)
set(cudpp_VERSION_MINOR 0)
set(cudpp_VERSION_PATCH 0)

cmake_minimum_required(VERSION 3.0)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../../../../LocalPackages/CUDA)
  Message(STATUS "Local CUDA detected. Using it")
  
  set(CUDA_TOOLKIT_VERSION "8.0" CACHE STRING "")
  set(CUDA_LOCAL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../../LocalPackages/CUDA/v${CUDA_TOOLKIT_VERSION}_win")
  get_filename_component(CUDA_ABS_PATH ${CUDA_LOCAL_PATH} ABSOLUTE )
  set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_ABS_PATH} CACHE PATH "")
endif()
find_package(CUDA REQUIRED)

set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#---------------- GET CUDA VERSION
if (CUDA_TOOLKIT_VERSION STREQUAL "8.0")
    set (CUDA_SUFFIX "cu8")
endif()
if (CUDA_TOOLKIT_VERSION STREQUAL "9.0")
    set (CUDA_SUFFIX "cu9")
endif()
if (CUDA_TOOLKIT_VERSION STREQUAL "10.0")
    set (CUDA_SUFFIX "cu10")
endif()

if (WIN32)
	set ( TARGET_NAME1 "cudpp_${MSVC_VERSION}${CUDA_SUFFIX}x" CACHE STRING "" FORCE)
	set ( TARGET_NAME2 "cudpp_hash_${MSVC_VERSION}${CUDA_SUFFIX}x" CACHE STRING "" FORCE)
else()
	set ( TARGET_NAME1 "cudpp_${CUDA_SUFFIX}" CACHE STRING "" FORCE)
	set ( TARGET_NAME2 "cudpp_hash_${CUDA_SUFFIX}" CACHE STRING "" FORCE)
endif()

option(CUDPP_BUILD_SHARED_LIBS
  "On to build shared libraries, off for static libraries."
  ON
  )

option(CUDPP_BUILD_APPLICATIONS
  "If on, builds the sample applications."
  OFF
  )

option(CUDA_VERBOSE_PTXAS
  "On to enable verbose output from the PTXAS assembler."
  OFF
  )

## Set the directory where the binaries will be stored
set(EXECUTABLE_OUTPUT_PATH
  ${PROJECT_BINARY_DIR}/bin
  CACHE PATH
  "Directory where all executables will be stored"
  )

## Set the directory where the libraries will be stored
set(LIBRARY_OUTPUT_PATH
  ${PROJECT_BINARY_DIR}/lib
  CACHE PATH
  "Directory where all the libraries will be stored"
  )

set(CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/cmake
  )

set(cudpp_INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/ext/moderngpu/include/
  ${CMAKE_SOURCE_DIR}/ext/cub/
  )
include_directories(${cudpp_INCLUDE_DIRS})

if (CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(bits 64)
else ()
  set(bits 32)
endif (CMAKE_SIZEOF_VOID_P EQUAL 8)
string(TOUPPER "${CMAKE_CONFIGURATION_TYPES}" configs)
foreach (config ${configs})
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config} "${EXECUTABLE_OUTPUT_PATH}")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${config} "${LIBRARY_OUTPUT_PATH}")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${config} "${LIBRARY_OUTPUT_PATH}")
  set(CMAKE_${config}_POSTFIX ${bits})
endforeach(config {configs})
set(CMAKE_DEBUG_POSTFIX "${bits}d")

if(CUDPP_BUILD_SHARED_LIBS)
  # User wants to build Dynamic Libraries, so change the LIB_TYPE variable to CMake keyword 'SHARED'
  set(LIB_TYPE SHARED)
else()
  set(LIB_TYPE STATIC)
  set(CUDPP_STATIC_LIB 1)
endif(CUDPP_BUILD_SHARED_LIBS)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/cudpp_config.h.in
  ${CMAKE_CURRENT_SOURCE_DIR}/include/cudpp_config.h
  )

set (CUDPP_APP_DATA_DIR
  "${CMAKE_SOURCE_DIR}/apps/data/"
  )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/apps/common/include/common_config.h.in
  ${CMAKE_CURRENT_SOURCE_DIR}/apps/common/include/common_config.h
  )

# copy WIN32 DLLs
if (WIN32)
if (bits EQUAL 64)
file(COPY bin/win32/glut32.dll bin/win64/glew64.dll DESTINATION bin/)
elseif (bits EQUAL 32)
file(COPY bin/win32/glut32.dll bin/win32/glew32.dll DESTINATION bin/)
endif (bits EQUAL 64)
endif (WIN32)

# copy data files
#file(COPY apps/data DESTINATION apps)

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

add_subdirectory(src/cudpp)
add_subdirectory(src/cudpp_hash)

if (CUDPP_BUILD_APPLICATIONS)
  add_subdirectory(apps/cudpp_testrig)
  add_subdirectory(apps/cudpp_hash_testrig)
  add_subdirectory(apps/simpleCUDPP)
  #add_subdirectory(apps/satGL)
endif (CUDPP_BUILD_APPLICATIONS)

# Make files that store configuration to help dependent CMake files use
# these libraries.
export(
  TARGETS ${TARGET_NAME1} ${TARGET_NAME2}
  FILE ${LIBRARY_OUTPUT_PATH}/cudpp-targets.cmake
  )

set(cudpp_INCLUDE_DIRS_CONFIG ${cudpp_INCLUDE_DIRS})
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cudpp-config.cmake.in
  ${LIBRARY_OUTPUT_PATH}/cudpp-config.cmake
  @ONLY
  )

set(cudpp_INCLUDE_DIRS_CONFIG "\${_install_dir}/include")
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cudpp-config.cmake.in
  ${LIBRARY_OUTPUT_PATH}/cudpp-config.cmake.install
  @ONLY
  )
install(FILES ${LIBRARY_OUTPUT_PATH}/cudpp-config.cmake.install
  DESTINATION lib
  RENAME cudpp-config.cmake
  )

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cudpp-config-version.cmake.in
  ${LIBRARY_OUTPUT_PATH}/cudpp-config-version.cmake
  @ONLY
  )
install(FILES ${LIBRARY_OUTPUT_PATH}/cudpp-config-version.cmake
  DESTINATION lib
  )

set(CPACK_PACKAGE_VERSION_MAJOR ${cudpp_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${cudpp_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${cudpp_VERSION_PATCH})
include(CPack)

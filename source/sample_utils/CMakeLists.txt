# Essentially, create a utils library that we can link against
cmake_minimum_required(VERSION 2.8)
project(gvdb_sample_utils)
add_compile_options(-std=c++11)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(CUDA REQUIRED)

set(GVDB_ROOT_DIR "~/packages/gvdb2/install")
find_package(GVDB REQUIRED)
include_directories(${GVDB_INCLUDE_DIRS})

if ( NOT DEFINED ASSET_PATH ) 
  get_filename_component ( _assets "${CMAKE_CURRENT_SOURCE_DIR}/../shared_assets" REALPATH )
  set ( ASSET_PATH ${_assets} CACHE PATH "Full path to gvdb/shared_assets/" ) 
endif()

option(GVDB_UTILS_BUILD_SHARED_LIBS "Build Shared Library for dynamic linking" TRUE)
option(REQUIRE_NVGUI "Build NVGUI" TRUE)
option(REQUIRE_CAM "Build CAM" TRUE)
option(REQUIRE_VEC "Build VEC" TRUE)
option(REQUIRE_OPTIX "Build OPTIX" FALSE)



if(GVDB_UTILS_BUILD_SHARED_LIBS)
  # User wants to build Dynamic Libraries, so change the LIB_TYPE variable to CMake keyword 'SHARED'
  set(LIB_TYPE SHARED)
  message("Building shared library")
else()
  set(LIB_TYPE STATIC)
  message("Building static library")
endif()

set(UTIL_SOURCE_FILES "")
set(UTIL_HEADER_FILES "")
set(UTIL_KERNEL_FILES "")

set(SAMPLE_UTIL_DIR ${PROJECT_SOURCE_DIR})

# OpenGL always required
find_package(OpenGL)
# Add PNG to build
message(STATUS " Using PNG")
list(APPEND UTIL_SOURCE_FILES "${SAMPLE_UTIL_DIR}/file_png.cpp")
list(APPEND UTIL_HEADER_FILES "${SAMPLE_UTIL_DIR}/file_png.h")

# Add TGA to build
message(STATUS " Using TGA")
list(APPEND UTIL_SOURCE_FILES "${SAMPLE_UTIL_DIR}/file_tga.cpp")

# Add GLEW to build
message(STATUS " Using GLEW")
list(APPEND UTIL_SOURCE_FILES "${SAMPLE_UTIL_DIR}/glew.c")

# Add Main to build
if(WIN32)
  list(APPEND UTIL_SOURCE_FILES "${SAMPLE_UTIL_DIR}/main_win.cpp")
else()
  list(APPEND UTIL_SOURCE_FILES "${SAMPLE_UTIL_DIR}/main_x11.cpp")
endif()
list(APPEND UTIL_HEADER_FILES "${SAMPLE_UTIL_DIR}/main.h")
list(APPEND UTIL_HEADER_FILES "${SAMPLE_UTIL_DIR}/platform.h")

if(REQUIRE_NVGUI)
  # Add NVGUI to build
  message(STATUS " Using NVGUI")
  list(APPEND UTIL_SOURCE_FILES "${SAMPLE_UTIL_DIR}/nv_gui.cpp")
  list(APPEND UTIL_HEADER_FILES "${SAMPLE_UTIL_DIR}/nv_gui.h")
endif()

if(REQUIRE_CAM)
  # Add Camera to build
  message(STATUS " Using CAM")
  list(APPEND UTIL_SOURCE_FILES "${SAMPLE_UTIL_DIR}/camera.cpp")
  list(APPEND UTIL_HEADER_FILES "${SAMPLE_UTIL_DIR}/camera.h")
endif()

if(REQUIRE_VEC)
  # Add Vec to build
  message(STATUS " Using VEC")
  list(APPEND UTIL_SOURCE_FILES "${SAMPLE_UTIL_DIR}/vec.cpp")
  list(APPEND UTIL_HEADER_FILES "${SAMPLE_UTIL_DIR}/vec.h")
endif()

if(REQUIRE_OPTIX)
  # Add OptiX to build Collect all OptiX utility files
  add_definitions(-DUSE_OPTIX_UTILS)
  message(STATUS " Using OPTIX_UTILS")
  file(GLOB UTIL_OPTIX_SOURCE_FILES "${SAMPLE_UTIL_DIR}/optix*.cpp")
  file(GLOB UTIL_OPTIX_HEADER_FILES "${SAMPLE_UTIL_DIR}/optix*.hpp")
  file(GLOB UTIL_OPTIX_KERNELS "${SAMPLE_UTIL_DIR}/optix*.cu"
            "${SAMPLE_UTIL_DIR}/optix*.cuh")
#   list(APPEND UTIL_SOURCE_FILES ${UTIL_OPTIX_SOURCE_FILES})
#   list(APPEND UTIL_HEADER_FILES ${UTIL_OPTIX_HEADER_FILES})
#   list(APPEND UTIL_KERNELS ${UTIL_OPTIX_KERNELS})
endif()

if(GVDB_UTILS_BUILD_SHARED_LIBS)
  message(STATUS "Set export shared library.")
  add_definitions(-Dgvdb_utils_EXPORTS)
endif()

cuda_add_library(gvdb_utils
                 ${LIB_TYPE}
                 ${UTIL_SOURCE_FILES}
                 ${UTIL_HEADER_FILES}
                 OPTIONS
                 ${GENCODE}
                 ${VERBOSE_PTXAS})
# Add as target definition so that it carries through to packages that import this package
target_compile_definitions(gvdb_utils PUBLIC ASSET_PATH="${ASSET_PATH}/" USE_GVDB=1)
target_link_libraries(gvdb_utils ${GVDB_LIBRARIES} ${OPENGL_LIBRARY})
set_target_properties(gvdb_utils PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

install(FILES ${UTIL_HEADER_FILES} DESTINATION include)
install(TARGETS gvdb_utils DESTINATION lib EXPORT gvdb_utils-targets)

install(EXPORT gvdb_utils-targets DESTINATION lib/cmake)
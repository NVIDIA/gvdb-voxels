# Copyright 2017 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.10) # Use Modern CMake

# Here's what the dependency graph of this project looks like:
#      .---gvdbCopy--. (copies files)
#      |             |
#      V             V
#     gvdb        gvdbPTX
# (builds lib)  (builds PTX)

project(gvdb LANGUAGES CUDA CXX C)

cmake_policy(SET CMP0072 NEW) # Prefer GLVND by default when available (CMake 3.11+)
find_package(OpenGL REQUIRED)
# Finds GLEW
include("cmake/FindGLEW.cmake")

# Print warning if not a 64-bit build
if(NOT (CMAKE_SIZEOF_VOID_P EQUAL 8))
    message(WARNING "gvdb_library: This project requires a 64-bit build. Subsequent configuration steps may fail.")
endif()

# Set the global minimum C++ standard for the project (this is C++11, since we use nullptr)
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# Set C++ standard required for CUDA
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Compile a couple of source files into a PTX object library.
# Creates an object library target named gvdbPTX.
set(GVDB_PTX_SOURCE_FILES "kernels/cuda_gvdb_copydata.cu" "kernels/cuda_gvdb_module.cu")
add_library(gvdbPTX OBJECT ${GVDB_PTX_SOURCE_FILES})
# Make sure it compiles to a PTX file.
set_target_properties(gvdbPTX PROPERTIES CUDA_PTX_COMPILATION ON)
# Private and public sources are compiled into the target.
# Public and interface sources are published into the list of sources for
# consuming targets
# Another way of thinking about this is that
# public = used for target + used from targets.
target_sources(gvdbPTX
    PUBLIC  "${CMAKE_CURRENT_LIST_DIR}/kernels/cuda_gvdb.cuh"
            "${CMAKE_CURRENT_LIST_DIR}/kernels/cuda_gvdb_dda.cuh"
            "${CMAKE_CURRENT_LIST_DIR}/kernels/cuda_gvdb_geom.cuh"
            "${CMAKE_CURRENT_LIST_DIR}/kernels/cuda_gvdb_nodes.cuh"
            "${CMAKE_CURRENT_LIST_DIR}/kernels/cuda_gvdb_operators.cuh"
            "${CMAKE_CURRENT_LIST_DIR}/kernels/cuda_gvdb_particles.cuh"
            "${CMAKE_CURRENT_LIST_DIR}/kernels/cuda_gvdb_raycast.cuh"
            "${CMAKE_CURRENT_LIST_DIR}/kernels/cuda_gvdb_scene.cuh"
            "${CMAKE_CURRENT_LIST_DIR}/kernels/cuda_math.cuh")

# Enable the --use_fast_math CUDA compiler flag.
target_compile_options(gvdbPTX
    PRIVATE --use_fast_math)

# Export our additional target include directories.
target_include_directories(gvdbPTX
    INTERFACE   $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/kernels>
                $<INSTALL_INTERFACE:include>) # expands to _output/include - note that these must be un-quoted!


# Create our main GVDB shared library (will output to libgvdb.so on Unix and gvdb.dll on Windows)
# add_library requires at least one file to be listed in CMake 3.10:
file(WRITE "${CMAKE_CURRENT_LIST_DIR}/dummy.cpp" "")

add_library(gvdb SHARED "${CMAKE_CURRENT_LIST_DIR}/dummy.cpp")

# Add sources - this tells our compiler which files to build, as well as which
# files to show in IDEs.
target_sources(gvdb
    PRIVATE src/app_perf.cpp
            src/gvdb_allocator.cpp
            src/gvdb_camera.cpp
            src/gvdb_cutils.cu
            src/gvdb_model.cpp
            src/gvdb_node.cpp
            src/gvdb_render_opengl.cpp
            src/gvdb_scene.cpp
            src/gvdb_types.cpp
            src/gvdb_vec.cpp
            src/gvdb_volume_3D.cpp
            src/gvdb_volume_base.cpp
            src/gvdb_volume_gvdb.cpp
            src/loader_ObjarReader.cpp
            src/loader_OBJReader.cpp
            src/loader_Parser.cpp
            src/string_helper.cpp
    PUBLIC  "${CMAKE_CURRENT_LIST_DIR}/src/app_perf.h"
            "${CMAKE_CURRENT_LIST_DIR}/src/gvdb.h"
            "${CMAKE_CURRENT_LIST_DIR}/src/gvdb_allocator.h"
            "${CMAKE_CURRENT_LIST_DIR}/src/gvdb_camera.h"
            "${CMAKE_CURRENT_LIST_DIR}/src/gvdb_cutils.cuh"
            "${CMAKE_CURRENT_LIST_DIR}/src/gvdb_model.h"
            "${CMAKE_CURRENT_LIST_DIR}/src/gvdb_node.h"
            "${CMAKE_CURRENT_LIST_DIR}/src/gvdb_render.h"
            "${CMAKE_CURRENT_LIST_DIR}/src/gvdb_scene.h"
            "${CMAKE_CURRENT_LIST_DIR}/src/gvdb_types.h"
            "${CMAKE_CURRENT_LIST_DIR}/src/gvdb_vec.h"
            "${CMAKE_CURRENT_LIST_DIR}/src/gvdb_volume_3D.h"
            "${CMAKE_CURRENT_LIST_DIR}/src/gvdb_volume_base.h"
            "${CMAKE_CURRENT_LIST_DIR}/src/gvdb_volume_gvdb.h"
            "${CMAKE_CURRENT_LIST_DIR}/src/loader_ObjarReader.h"
            "${CMAKE_CURRENT_LIST_DIR}/src/loader_OBJReader.h"
            "${CMAKE_CURRENT_LIST_DIR}/src/loader_Parser.h"
            "${CMAKE_CURRENT_LIST_DIR}/src/string_helper.h")

# Export our additional target include directories.
target_include_directories(gvdb
    PUBLIC      ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    INTERFACE   $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>
                $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/kernels>
                $<INSTALL_INTERFACE:include>)

# Make sure to link against CUDA (since we use the Driver API) and OpenGL
target_link_libraries(gvdb
    PUBLIC ${OPENGL_LIBRARIES})
if(WIN32)
    # Get the absolute path to cuda.lib from CMAKE_CUDA_COMPILER if it exists.
    # This avoids linking errors with Ninja and the samples.
    set(_CUDA_LIB "cuda.lib")
    if(CMAKE_CUDA_COMPILER)
      get_filename_component(_CUDA_BIN_DIR "${CMAKE_CUDA_COMPILER}" DIRECTORY)
      get_filename_component(_CUDA_DIR "${_CUDA_BIN_DIR}" DIRECTORY)
      string(PREPEND _CUDA_LIB "${_CUDA_DIR}/lib/x64/")
    endif()
    target_link_libraries(gvdb PUBLIC "${_CUDA_LIB}")
elseif(UNIX)
    find_package(X11 REQUIRED)
    target_link_libraries(gvdb
        PUBLIC  cuda
                ${X11_LIBRARIES})
endif()

macro(_gvdb_find variable path show_warning)
    if(NOT ${variable})
        file(GLOB _FOUND_FILES ${path})
        if(NOT _FOUND_FILES)
            if({$show_warning})
                message(WARNING "_gvdb_find_shared_library: Unable to find any files corresponding to the glob specification ${path}!")
            endif()
        else()
            list(GET _FOUND_FILES 0 ${variable}) # Set variable to the first element of the list
        endif()
    endif()
endmacro()

# Allow the developer to build with or without OpenVDB
set(GVDB_BUILD_OPENVDB OFF CACHE BOOL "If ON, builds GVDB with OpenVDB and #defines BUILD_OPENVDB.")
set(GVDB_OPENVDB_LIBRARIES_TO_COPY "")
if(GVDB_BUILD_OPENVDB)
    target_compile_definitions(gvdb PUBLIC BUILD_OPENVDB)
    if(WIN32)
        # Windows version - we assume you use vcpkg here or have set the
        # GVDB_OPENVDB_INCLUDE_DIR, GVDB_OPENVDB_LIB_RELEASE_DIR, GVDB_OPENVDB_LIB_DEBUG_DIR, and
        # GVDB_OPENVDB_DLL_RELEASE_DIR variables:
        # Helpful (hopefully!) warning message if someone's building this on Windows with OpenVDB but without using vcpkg
        if(NOT CMAKE_TOOLCHAIN_FILE)
            message(WARNING "GVDB_BUILD_OPENVDB was specified, but CMAKE_TOOLCHAIN_FILE wasn't specified. This is only an issue if you're using something like vcpkg. If you're using the CMake GUI, you'll want to specify the vcpkg toolchain in the configuration settings (delete the cache, press the Configure button, then choose 'Specify toolchain file for cross-compiling' and enter the path to vcpkg.cmake, then make sure to re-enable GVDB_BUILD_OPENVDB). If you're using the CMake command-line interface, you might need to specify -DCMAKE_TOOLCHAIN_FILE=<path to toolchain file>.")
        endif()
        
        # Linking OpenVDB with vcpkg is a bit odd, so this section is a bit more complex than the rest
        # of the CMakeFile. If you're not building with OpenVDB, this never gets included.
        #
        # As of this writing, vcpkg seems to assume you're running its CMake toolchain with a single 
        # configuration (e.g. Debug or Release), so find_package calls search first in the respective
        # debug or release directory in vcpkg's installed directory. Vcpkg also runs a useful PowerShell
        # script in the post-build process step that searches for referenced DLLs and automatically copies
        # them to the output directory. Here, it looks at your built executable's path, determines whether
        # you built in debug or release mode from that, and then searches in the relevant vcpkg directories.
        #
        # Unfortunately, it doesn't search in the release directory if it can't find the relevant file in
        # the debug directory, and silently fails if this is the case. At the same time, openvdb in vcpkg
        # has some quirks - for example, the debug version always uses the release version of Half and tbb.
        #
        # Here's how we handle this:
        # - Finding openvdb.h works normally.
        # - We find paths to separate debug and release libraries using the fact that (as of this writing!)
        # tbb's debug and release libraries and DLLs are named different things (tbb_debug and tbb). This
        # allows us to find vcpkg's install directories; these can also be overriden by the user.
        # - Finally, we make sure to always copy over the release versions of Half and tbb. When we're building
        # in debug mode, this allows us to include the DLLs vcpkg won't find, while this saves some work for
        # vcpkg in release mode.
        
        # Compile definitions: We're using the DLL of OpenVDB, instead of building it statically.
        # TBB has a #pragma comment(lib, tbb...) line in it, but this isn't set up to work properly,
        # so we explicitly link it instead:
        target_compile_definitions(gvdb PUBLIC __TBB_NO_IMPLICIT_LINKAGE)
        
        # Find the path to the include directory:
        if(NOT GVDB_OPENVDB_INCLUDE_DIR)
            find_path(GVDB_OPENVDB_INCLUDE_DIR openvdb/openvdb.h)
        endif()
        message(STATUS "Using this path for GVDB_OPENVDB_INCLUDE_DIR: ${GVDB_OPENVDB_INCLUDE_DIR}")
        target_include_directories(gvdb PUBLIC ${GVDB_OPENVDB_INCLUDE_DIR})
        
        # Find the debug and release directories containing vcpkg's .lib files:
        if(NOT GVDB_OPENVDB_LIB_RELEASE_DIR)
            find_path(GVDB_OPENVDB_LIB_RELEASE_DIR tbb.lib PATH_SUFFIXES lib)
        endif()
        if(NOT GVDB_OPENVDB_LIB_DEBUG_DIR)
            find_path(GVDB_OPENVDB_LIB_DEBUG_DIR tbb_debug.lib PATH_SUFFIXES debug/lib)
        endif()
        
        # Find the paths to our .lib files manually:
        _gvdb_find(GVDB_OPENVDB_LIB_DEBUG ${GVDB_OPENVDB_LIB_DEBUG_DIR}/openvdb_d.lib OFF)
        # Compatibility for OpenVDB < 7.0
        _gvdb_find(GVDB_OPENVDB_LIB_DEBUG ${GVDB_OPENVDB_LIB_DEBUG_DIR}/openvdb.lib ON)
        _gvdb_find(GVDB_TBB_LIB_DEBUG     ${GVDB_OPENVDB_LIB_DEBUG_DIR}/tbb_debug.lib ON)
        _gvdb_find(GVDB_BLOSC_LIB_DEBUG   ${GVDB_OPENVDB_LIB_DEBUG_DIR}/blosc.lib ON)
        _gvdb_find(GVDB_ZLIB_LIB_DEBUG    ${GVDB_OPENVDB_LIB_DEBUG_DIR}/zlib*.lib ON)
        _gvdb_find(GVDB_HALF_LIB_DEBUG    ${GVDB_OPENVDB_LIB_DEBUG_DIR}/Half-*.lib ON)
        
        _gvdb_find(GVDB_OPENVDB_LIB_RELEASE ${GVDB_OPENVDB_LIB_RELEASE_DIR}/openvdb.lib ON)
        _gvdb_find(GVDB_TBB_LIB_RELEASE     ${GVDB_OPENVDB_LIB_RELEASE_DIR}/tbb.lib ON)
        _gvdb_find(GVDB_BLOSC_LIB_RELEASE   ${GVDB_OPENVDB_LIB_RELEASE_DIR}/blosc.lib ON)
        _gvdb_find(GVDB_ZLIB_LIB_RELEASE    ${GVDB_OPENVDB_LIB_RELEASE_DIR}/zlib*.lib ON)
        _gvdb_find(GVDB_HALF_LIB_RELEASE    ${GVDB_OPENVDB_LIB_RELEASE_DIR}/Half-*.lib ON)
        
        target_link_libraries(gvdb PRIVATE
            debug ${GVDB_OPENVDB_LIB_DEBUG}
            debug ${GVDB_TBB_LIB_DEBUG}
            debug ${GVDB_BLOSC_LIB_DEBUG}
            debug ${GVDB_ZLIB_LIB_DEBUG}
            debug ${GVDB_HALF_LIB_DEBUG}
            optimized ${GVDB_OPENVDB_LIB_RELEASE}
            optimized ${GVDB_TBB_LIB_RELEASE}
            optimized ${GVDB_BLOSC_LIB_RELEASE}
            optimized ${GVDB_ZLIB_LIB_RELEASE}
            optimized ${GVDB_HALF_LIB_RELEASE}
        )
        
        # Find the path to the directory containing release versions of DLLs.
        if(NOT GVDB_OPENVDB_DLL_RELEASE_DIR)
            find_path(GVDB_OPENVDB_DLL_RELEASE_DIR tbb.dll PATH_SUFFIXES bin)
        endif()
        
        # Include the release versions of Half and tbb.
        _gvdb_find(GVDB_TBB_DLL_RELEASE ${GVDB_OPENVDB_DLL_RELEASE_DIR}/tbb.dll ON)
        _gvdb_find(GVDB_HALF_DLL_RELEASE ${GVDB_OPENVDB_DLL_RELEASE_DIR}/Half-*.dll ON)
        list(APPEND GVDB_OPENVDB_LIBRARIES_TO_COPY ${GVDB_TBB_DLL_RELEASE} ${GVDB_HALF_DLL_RELEASE})
    else()
        # On Linux systems, find_package(openvdb) should work, assuming OpenVDB has
        # been installed from source. By default, OpenVDB will be installed in /usr/local/lib,
        # but we allow the user to modify the search path for this.
        if(NOT GVDB_OPENVDB_INSTALL_PATH)
            # Try to find where libopenvdb.so has been installed.
            find_library(GVDB_OPENVDB_LIB libopenvdb.so)
            if(GVDB_OPENVDB_LIB)
                get_filename_component(GVDB_OPENVDB_INSTALL_PATH "${GVDB_OPENVDB_LIB}" DIRECTORY)
            else()
                set(GVDB_OPENVDB_INSTALL_PATH "/usr/local/lib")
            endif()
            message(STATUS "Note: GVDB_OPENVDB_INSTALL_PATH not specified; assuming OpenVDB has been installed in ${GVDB_OPENVDB_INSTALL_PATH} (which should contain cmake/OpenVDB).")
        endif()
        list(APPEND CMAKE_MODULE_PATH "${GVDB_OPENVDB_INSTALL_PATH}/cmake/OpenVDB")
        find_package(OpenVDB REQUIRED) # GVDB_BUILD_OPENVDB was specified!
        target_link_libraries(gvdb PUBLIC OpenVDB::openvdb)
    endif()
endif()

# Allow the developer to build with or without NanoVDB
set(GVDB_BUILD_NANOVDB OFF CACHE BOOL "If ON, builds GVDB with NanoVDB and #defines BUILD_NANOVDB.")
set(GVDB_NANOVDB_INCLUDE_DIR "" CACHE PATH "The path to NanoVDB (should be a folder that contains a nanovdb folder filled with include files")
if(GVDB_BUILD_NANOVDB)
  if(NOT GVDB_NANOVDB_INCLUDE_DIR)
    message(WARNING "GVDB_NANOVDB_INCLUDE_DIR needs to be specified for GVDB to find NanoVDB! (You're seeing this message because GVDB_BUILD_NANOVDB was ON, but GVDB_NANOVDB_INCLUDE_DIR was unfilled).")
  endif()
  target_compile_definitions(gvdb PUBLIC BUILD_NANOVDB)
  target_include_directories(gvdb PUBLIC ${GVDB_NANOVDB_INCLUDE_DIR})
  target_include_directories(gvdbPTX PUBLIC ${GVDB_NANOVDB_INCLUDE_DIR})
endif()

# Enable definitions: build with OpenGL (required),
# write DLL definitions as exports (instead of imports)
target_compile_definitions(gvdb
    PRIVATE BUILD_OPENGL
            GVDB_EXPORTS)

# GVDB currently compiles with GLEW using the static version -
# that is, by directly including glew.c and its header files.
# We also need to define GLEW_STATIC.
target_compile_definitions(gvdb PUBLIC GLEW_STATIC)
target_sources(gvdb PRIVATE ${GLEW_SOURCE})
target_include_directories(gvdb PRIVATE ${GLEW_INCLUDE_DIR})

# Allow GVDB to use e.g. CUDA_GVDB_COPYDATA as a macro for "cuda_gvdb_copydata.ptx"
foreach(_FILE ${GVDB_PTX_SOURCE_FILES})
	get_filename_component(_FILE ${_FILE} NAME) # cuda_gvdb_copydata.cu
	string(REPLACE ".cu" ".ptx" _FILE ${_FILE}) # cuda_gvdb_copydata.ptx
	string(REPLACE "." "_" _MACRO ${_FILE})     # cuda_gvdb_copydata_ptx
	string(TOUPPER ${_MACRO} _MACRO) # Yep, this goes input, then output!
	target_compile_definitions(gvdb PUBLIC
		${_MACRO}="${_FILE}")
endforeach()

# Finally, make our targets importable from the build directory.
export(TARGETS gvdb gvdbPTX FILE GVDBConfig.cmake)

# As one further improvement, we can copy all of the files we need into the library
# directory - essentially packaging this for applications that use gvdb without
# requiring an additional install step.
# To track dependencies correctly, we'll add another target that handles the file copies,
# and depends on gvdbPTX!
set(_GVDB_COPY_STAMP_FILES copiedPTX.stamp copiedShaders.stamp)
if(GVDB_BUILD_OPENVDB AND GVDB_OPENVDB_LIBRARIES_TO_COPY)
    # Copy OpenVDB and the libraries it depends upon
    add_custom_command(OUTPUT copiedSharedLibraries.stamp
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${GVDB_OPENVDB_LIBRARIES_TO_COPY} $<TARGET_FILE_DIR:gvdb>
    COMMAND ${CMAKE_COMMAND} -E touch copiedSharedLibraries.stamp
    COMMAND_EXPAND_LISTS)
    list(APPEND _GVDB_COPY_STAMP_FILES copiedSharedLibraries.stamp)
endif()

add_custom_target(gvdbCopy DEPENDS ${_GVDB_COPY_STAMP_FILES})

# Copy the PTX files to the library directory (this is where the dependency on gvdbPTX is introduced)
add_custom_command(OUTPUT copiedPTX.stamp
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_OBJECTS:gvdbPTX> $<TARGET_FILE_DIR:gvdb>
    COMMAND ${CMAKE_COMMAND} -E touch copiedPTX.stamp
    DEPENDS gvdbPTX COMMAND_EXPAND_LISTS)
# Finally, copy the shaders to the library directory
add_custom_command(OUTPUT copiedShaders.stamp
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_LIST_DIR}/shaders" $<TARGET_FILE_DIR:gvdb>
    COMMAND ${CMAKE_COMMAND} -E touch copiedShaders.stamp)
# We can then tell CMake that our libraries were built in this folder, so that
# subsequent find_package calls on the same system can find the libraries
# right away.
# export(PACKAGE gvdb)
# export(PACKAGE gvdbPTX)

# Install target
# Install near source tree - not in the default directory unless CMAKE_INSTALL_PREFIX
# has been set by the user!
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    get_filename_component(_INSTALLPATH "${CMAKE_CURRENT_SOURCE_DIR}/../../_output" REALPATH)
    # Forcing is only OK here because we've checked for CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT.
    set(CMAKE_INSTALL_PREFIX ${_INSTALLPATH} CACHE PATH "Default install path for GVDB" FORCE)
endif()

# Compute binary, include, and library install paths.
get_filename_component(BIN_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/bin REALPATH)
get_filename_component(INCLUDE_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/include REALPATH)
get_filename_component(LIB_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/lib REALPATH)

install(DIRECTORY "$<TARGET_FILE_DIR:gvdb>/" DESTINATION ${BIN_INSTALL_PATH} FILES_MATCHING PATTERN "*.dll" PATTERN "*.glsl" PATTERN "*.ptx" PATTERN "*.so")
install(DIRECTORY "$<TARGET_FILE_DIR:gvdb>/" DESTINATION ${LIB_INSTALL_PATH} FILES_MATCHING PATTERN "*.lib" PATTERN "*.exp" PATTERN "*.ilk" PATTERN "*.pdb")
get_target_property(TEMP_INTERFACE_SOURCES gvdb INTERFACE_SOURCES)
install(FILES ${TEMP_INTERFACE_SOURCES} DESTINATION ${INCLUDE_INSTALL_PATH})
get_target_property(TEMP_INTERFACE_SOURCES gvdbPTX INTERFACE_SOURCES)
install(FILES ${TEMP_INTERFACE_SOURCES} DESTINATION ${INCLUDE_INSTALL_PATH})

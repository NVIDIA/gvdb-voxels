cmake_minimum_required(VERSION 2.8)
if ( WIN32 )
   set(PROJNAME libgvdb)
else()
   set(PROJNAME gvdb)
endif()
Project(${PROJNAME})
Message(STATUS "-------------------------------")
Message(STATUS "Processing Project ${PROJNAME}:")

#####################################################################################
# Included Cmakes

if(NOT BASE_DIRECTORY)
  SET(BASE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()
include ( "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Helpers.cmake" )    
include ( "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Packages.cmake" )  
include_directories ("${CMAKE_CURRENT_SOURCE_DIR}/src")    

_add_project_definitions(${PROJNAME})

#####################################################################################
# GVDB Options
#
set ( USE_BITMASKS OFF CACHE BOOL "Use Bitmasks" )
if ( USE_BITMASKS ) 
   add_definitions(-DUSE_BITMASKS)
endif()

set ( USE_SIMULATION OFF CACHE BOOL "Use GVDB 2.0 Simulation" )
if ( USE_SIMULATION ) 
   add_definitions(-DSIM_CODE)
endif()

set ( USE_OPENVDB OFF CACHE BOOL "Use OpenVDB" FORCE)

set ( USE_NVTX OFF CACHE BOOL "Use NSIGHT Markers" )
if ( USE_NVTX ) 
   add_definitions(-DUSE_NVTX)
endif()

#####################################################################################
# Add CUDA
#
_add_package_CUDA()

#####################################################################################
# Add CUDPP Library
#
_add_package_CUDPP()

#####################################################################################
# Add OPENVDB Library (optional)
#
_add_package_OpenVDB()
 
#####################################################################################
# Source files for this project
#
file(GLOB SOURCE_FILES src/*.cpp src/*.hpp src/*.inl src/*.h src/*.c)

#####################################################################################
# Additional files for tools/helpers
#
SET(COMMON_SOURCE_FILES ${GLEW_SOURCE} )

#####################################################################################
# Library paths
#
# CMAKE_INSTALL_PREFIX -- path where library will be installed to

if ( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
   if ( WIN32 )
      get_filename_component ( _instpath "${CMAKE_CURRENT_SOURCE_DIR}/../../_output" REALPATH )
   else()
      get_filename_component ( _instpath "/usr/local/gvdb" REALPATH )
   endif()
   set ( CMAKE_INSTALL_PREFIX ${_instpath} CACHE PATH "default install path" FORCE)   
endif()

get_filename_component( LIB_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR} REALPATH)
get_filename_component( LIB_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/lib REALPATH)
get_filename_component( INCLUDE_PATH ${CMAKE_INSTALL_PREFIX}/include REALPATH)
get_filename_component( SHARE_PATH ${CMAKE_INSTALL_PREFIX}/lib REALPATH)

set ( EXECUTABLE_OUTPUT_PATH ${LIB_INSTALL_PATH} CACHE PATH "" FORCE )

#####################################################################################
# Custom Build for CUDA files
#
file(GLOB CUDA_FILES RELATIVE "${BASE_DIRECTORY}" kernels/*.cu kernels/*.cuh )
message ( STATUS "Located CUDA kernels: ${CUDA_FILES}" )

if ( NOT DEFINED CUDA_ARCH )
   SET( CUDA_ARCH "compute_30" CACHE STRING "CUDA Architecture target")
   SET( CUDA_CODE "sm_30" CACHE STRING "CUDA Code target")
endif ()

_COMPILEPTX ( SOURCES ${CUDA_FILES} TARGET_PATH ${LIB_OUTPUT_PATH} GENERATED CUDA_PTX GENPATHS CUDA_PTX_PATHS INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/kernels" OPTIONS -DSIM_CODE -arch=${CUDA_ARCH} -code=${CUDA_CODE} --ptxas-options=-v -O3 --use_fast_math --maxrregcount=64 )

#####################################################################################
# Executable
#
unset ( ALL_SOURCE_FILES )
list( APPEND ALL_SOURCE_FILES ${SOURCE_FILES} )
list( APPEND ALL_SOURCE_FILES ${COMMON_SOURCE_FILES} )
list( APPEND ALL_SOURCE_FILES ${PACKAGE_SOURCE_FILES} )

# Definitions
add_definitions(-DGVDB_EXPORTS)					# Export dll symbols

# Set the library type
OPTION (BUILD_SHARED_LIBS "Build Shared Libraries" ON)
set (LIB_TYPE STATIC)
if (BUILD_SHARED_LIBS)
  set (LIB_TYPE SHARED)
endif()

# Build with OpenGL
OPTION (BUILD_OPENGL "Build with OpenGL" ON)
if (BUILD_OPENGL)
	add_definitions(-DBUILD_OPENGL)  				# Build OpenGL
endif()

if( WIN32 AND NOT GLUT_FOUND)
  add_definitions(/wd4267) #remove size_t to int warning
  add_definitions(/wd4996) #remove printf warning
  add_definitions(/wd4244) #remove double to float conversion warning
  add_definitions(/wd4305) #remove double to float truncation warning
  add_library (${PROJNAME} ${LIB_TYPE} ${ALL_SOURCE_FILES} ${CUDA_FILES} )
else()
  add_library (${PROJNAME} ${LIB_TYPE} ${ALL_SOURCE_FILES} ${CUDA_FILES} ${PTX_FILES} )
endif()

set_target_properties( ${PROJNAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${LIB_OUTPUT_PATH} )
set_target_properties( ${PROJNAME} PROPERTIES VS_INTERMEDIATE_DIRECTORY_DEBUG ${LIB_OUTPUT_PATH}/Debug )
set_target_properties( ${PROJNAME} PROPERTIES VS_INTERMEDIATE_DIRECTORY_RELEASE ${LIB_OUTPUT_PATH}/Release )
set_target_properties( ${PROJNAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${LIB_OUTPUT_PATH} )

message ( STATUS "Build:   ${LIB_OUTPUT_PATH}" )
message ( STATUS "Lib:     ${LIB_INSTALL_PATH}" )
message ( STATUS "Inc:     ${INCLUDE_PATH}" )
message ( STATUS "Share:   ${SHARE_PATH}" )

#####################################################################################
# Linkage
#
_LINK ( PROJECT ${PROJNAME} OPT ${LIBRARIES_OPTIMIZED} DEBUG ${LIBRARIES_DEBUG} PLATFORM ${PLATFORM_LIBRARIES} )

################################################################
# Windows specific
#
if ( WIN32 )    
	# instruct CMake to automatically build INSTALL project in Visual Studio 
	set(CMAKE_VS_INCLUDE_INSTALL_TO_DEFAULT_BUILD 1)
	# generate an import library
	set ( GVDB_LINK_FLAGS "/implib:${LIB_OUTPUT_PATH}/$(Configuration)/libgvdb.lib" )
	set_target_properties ( ${PROJNAME} PROPERTIES LINK_FLAGS "${GVDB_LINK_FLAGS}" )
	message ( STATUS "Import Library: ${GVDB_LINK_FLAGS}" )  
	# visual studio source groups
	source_group(common FILES 
		${COMMON_SOURCE_FILES}  
	)
	source_group(CUDA FILES 
		${CUDA_FILES}
	)
	source_group(PTX FILES 
		${PTX_FILES}
	)
endif()

################################################################
# Install Binaries, Shaders and PTX
#
if ( USE_OPENVDB ) 
    package_openvdb_binaries()
   _INSTALL ( FILES ${OPENVDB_BINARIES} SOURCE ${OPENVDB_LIB_DIR} DESTINATION ${EXECUTABLE_OUTPUT_PATH} OUTPUT INSTALL_LIST  )
endif()
file( GLOB SHADERS shaders/*.glsl)
file( GLOB HEADER_FILES src/*.h )
file( GLOB CUH_FILES kernels/*.cuh )

_INSTALL ( FILES ${SHADERS} SOURCE "" DESTINATION ${EXECUTABLE_OUTPUT_PATH} OUTPUT INSTALL_LIST )
_INSTALL ( FILES ${CUDPP_DLL} SOURCE ${CUDPP_LIB_DIR} DESTINATION ${EXECUTABLE_OUTPUT_PATH} OUTPUT INSTALL_LIST )
_INSTALL_PTX ( FILES ${CUDA_PTX_PATHS} DESTINATION ${EXECUTABLE_OUTPUT_PATH} OUTPUT INSTALL_LIST )

install ( FILES ${HEADER_FILES} DESTINATION ${INCLUDE_PATH} )
install ( FILES ${CUH_FILES} DESTINATION ${INCLUDE_PATH} )
install ( FILES ${INSTALL_LIST} DESTINATION ${LIB_INSTALL_PATH} )
install ( TARGETS ${PROJNAME} DESTINATION ${LIB_INSTALL_PATH} )

#################################
# Done
message ( STATUS "\nGVDB INSTALL PATH: ${CMAKE_INSTALL_PREFIX}" )
message ( STATUS "GVDB will be installed to this path post-build (win32) or during make install (linux)")



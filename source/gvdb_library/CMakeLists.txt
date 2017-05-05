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
# offer the choice of having shared_sources as a sub-folder... good for packaging a sample
#
# if BASE_DIRECTORY not defined, it means this cmake file was called as the first entry point and not included

if(NOT BASE_DIRECTORY)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
    Message(STATUS "found sub-folder cmake")
    SET(BASE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeLists_include.txt)    
  elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake)
    SET(BASE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../..)
    INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/CMakeLists_include.txt)    
  else()
    SET(BASE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
    INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CMakeLists_include.txt)    
  endif()
endif()

include_directories ("${CMAKE_CURRENT_SOURCE_DIR}/src")    

include ( "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Helpers.cmake" )    

_add_project_definitions(${PROJNAME})

#####################################################################################
# Find CUDA

find_package(CUDA)

if ( CUDA_FOUND )
    message( STATUS "--> Using package CUDA (ver ${CUDA_VERSION})") 
    add_definitions(-DUSECUDA)    
    include_directories(${CUDA_TOOLKIT_INCLUDE})
    LIST(APPEND LIBRARIES_OPTIMIZED ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} )
    LIST(APPEND LIBRARIES_DEBUG ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} )
    LIST(APPEND PACKAGE_SOURCE_FILES ${CUDA_TOOLKIT_INCLUDE} )    
    source_group(CUDA FILES ${CUDA_TOOLKIT_INCLUDE} ) 
else()
	message ( FATAL_ERROR "---> Unable to find package CUDA")
endif()


#####################################################################################
# Find OPENVDB Library (optional)
#

if ( NOT DEFINED USE_OPENVDB )	  
	SET(USE_OPENVDB OFF CACHE BOOL "Use OpenVDB" FORCE) 
endif ()

_add_package_OpenVDB ()

if ( OPENVDB_FOUND )

	if ( USE_OPENVDB ) 
		#---- Add OpenVDB to project build flags   
		add_definitions(-DBUILD_OPENVDB)	 
		add_definitions(-DOPENVDB_STATICLIB)
		add_definitions(-DOPENVDB_USE_BLOSC)
   
		#---- Include paths to headers & library that OpenVDB depends on (comes with)
		message ( STATUS "Adding OpenVDB includes: ${OPENVDB_INCLUDE_DIR}" )
		include_directories ("${OPENVDB_INCLUDE_DIR}")
		include_directories ("${OPENVDB_INCLUDE_DIR}/IlmBase")
		include_directories ("${OPENVDB_INCLUDE_DIR}/tbb")         

		# add_definitions(-D_ITERATOR_DEBUG_LEVEL=0)   
	
	endif ()
else ()
   SET(USE_OPENVDB OFF CACHE BOOL "Use OpenVDB") 
endif()

 
#####################################################################################
# Source files for this project
#
file(GLOB SOURCE_FILES src/*.cpp src/*.hpp src/*.inl src/*.h src/*.c)

#####################################################################################
# additional files for tools/helpers
#
SET(COMMON_SOURCE_FILES ${GLEW_SOURCE}
)

#####################################################################################
# Library paths
#
# CMAKE_INSTALL_PREFIX -- path where library will be installed to

if ( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
   if (WIN32)
      get_filename_component ( _instpath "${CMAKE_CURRENT_SOURCE_DIR}/../.." REALPATH )
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

_COMPILEPTX ( SOURCES ${CUDA_FILES} TARGET_PATH ${LIB_OUTPUT_PATH} GENERATED PTX_FILES INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/kernels" OPTIONS -arch=${CUDA_ARCH} -code=${CUDA_CODE} --ptxas-options=-v -O3 --use_fast_math --maxrregcount=64 )

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
set_property(GLOBAL PROPERTY DEBUG_CONFIGURATIONS Debug) 

foreach (loop_var IN ITEMS ${LIBRARIES_OPTIMIZED} )   
   target_link_libraries ( ${PROJNAME} optimized ${loop_var} )
endforeach()

foreach (loop_var IN ITEMS ${LIBRARIES_DEBUG} )
   target_link_libraries ( ${PROJNAME} debug ${loop_var} )
endforeach()

target_link_libraries( ${PROJNAME} general ${PLATFORM_LIBRARIES} )

string (REPLACE ";" "\n   " OUTSTR "${LIBRARIES_DEBUG}")
message ( STATUS "Libraries used:\n  ${OUTSTR}" )

################################################################
# Install Library
#
#--- Install public headers
file( GLOB HEADER_FILES src/*.h )
file( GLOB CUH_FILES kernels/*.cuh )
_INSTALL ( FILES ${HEADER_FILES} DESTINATION ${INCLUDE_PATH} )   
_INSTALL ( FILES ${CUH_FILES} DESTINATION ${INCLUDE_PATH} )  

#--- Install binaries (.so/.dll/.lib)
if ( WIN32 )    
    set ( GVDB_LINK_FLAGS "/implib:${LIB_OUTPUT_PATH}/$(Configuration)/libgvdb.lib" )
	set_target_properties ( ${PROJNAME} PROPERTIES LINK_FLAGS "${GVDB_LINK_FLAGS}" )
	message ( STATUS "Import Library: ${GVDB_LINK_FLAGS}" )
   _INSTALL ( FILES "${LIB_OUTPUT_PATH}/$(Configuration)/libgvdb.dll" DESTINATION ${LIB_INSTALL_PATH} )   
   _INSTALL ( FILES "${LIB_OUTPUT_PATH}/$(Configuration)/libgvdb.lib" DESTINATION ${LIB_INSTALL_PATH} )   
   if ( USE_OPENVDB )
	  _INSTALL ( FILES "${OPENVDB_DLL}" DESTINATION ${LIB_INSTALL_PATH} )
   endif ()
else()
   install ( TARGETS ${PROJNAME} DESTINATION ${LIB_INSTALL_PATH} )
endif()

#--- Install Shaders
file(GLOB SHADERS shaders/*.glsl)
message ( STATUS "Shaders used: ${SHADERS}" )
_INSTALL ( FILES ${SHADERS} DESTINATION ${SHARE_PATH} )

#--- Install PTX files
unset ( PTX_FIXED )
if ( WIN32 )   
  foreach ( _file IN ITEMS ${PTX_FILES} )
     list ( APPEND PTX_FIXED "${LIB_OUTPUT_PATH}/$(Configuration)/${_file}" )
  endforeach() 
else()
  foreach ( _file IN ITEMS ${PTX_FILES} )
    get_filename_component ( _ptxpath ${_file} DIRECTORY )
    get_filename_component ( _ptxbase ${_file} NAME_WE )
    string ( SUBSTRING ${_ptxbase} 27 -1 _ptxname )
    set ( _fixed "${_ptxpath}/${_ptxname}.ptx" )
    add_custom_command ( TARGET ${PROJNAME} PRE_LINK
      COMMAND ${CMAKE_COMMAND} -E copy  ${_file} ${_fixed}
      )      
    list ( APPEND PTX_FIXED ${_fixed} )
  endforeach()
endif()
message ( STATUS "PTX files: ${PTX_FIXED}" )
_INSTALL ( FILES ${PTX_FIXED} DESTINATION ${SHARE_PATH} )



#####################################################################################
# Source groups for Visual Studio
#
source_group(common FILES 
  ${COMMON_SOURCE_FILES}  
)
source_group(CUDA FILES 
  ${CUDA_FILES}
)
source_group(PTX FILES 
  ${PTX_FILES}
)

message ( STATUS "\nGVDB INSTALL PATH: ${CMAKE_INSTALL_PREFIX}" )
message ( STATUS "GVDB will be installed to this path post-build (win32) or during make install (linux)")



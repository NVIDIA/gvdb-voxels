cmake_minimum_required(VERSION 2.8)
set(PROJNAME gJetsonTX)
Project(${PROJNAME})
Message(STATUS "-------------------------------")
Message(STATUS "Processing Project ${PROJNAME}:")

####################################################################################
# Bootstrap
#
set( BASE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )
find_path ( HELPERS "Helpers.cmake" HINTS 
    ${CMAKE_MODULE_PATH}
    ${BASE_DIRECTORY}/sample_utils
    ${BASE_DIRECTORY}/../sample_utils
    ${BASE_DIRECTORY}/../source/sample_utils
    ${BASE_DIRECTORY}/../../source/sample_utils
)
if ( ${HELPERS} STREQUAL "HELPERS-NOTFOUND" )
    set ( CMAKE_MODULE_PATH "***FULL PATH TO***/gvdb/sample_utils/" CACHE PATH "Full path to gvdb/sample_utils/" )
    message ( FATAL_ERROR "\n
    Please set the CMAKE_MODULE_PATH 
    to the full path of for /gvdb/sample_utils/ above
    and configure again." )
endif()
get_filename_component ( CMAKE_MODULE_PATH ${HELPERS} REALPATH )
set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} CACHE PATH "Full path to gvdb/sample_utils/" )
set( EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR} CACHE PATH "Executable path" )
if ( NOT DEFINED ASSET_PATH ) 
  get_filename_component ( _assets "${CMAKE_MODULE_PATH}/../../source/shared_assets" REALPATH )
  set ( ASSET_PATH ${_assets} CACHE PATH "Full path to gvdb/shared_assets/" )
endif()

include( ${CMAKE_MODULE_PATH}/Helpers.cmake )     # Cross-Platform functions

#####################################################################################
# Sample requirements

set ( REQUIRE_PNG "1" )
set ( REQUIRE_TGA "1" )
set ( REQUIRE_GLEW "1" )
set ( REQUIRE_MAIN "1" )
set ( REQUIRE_NVGUI "1" )

#####################################################################################
# Find GVDB
#
find_package(GVDB)

if ( GVDB_FOUND )
	message( STATUS "--> Using package GVDB (inc: ${GVDB_INCLUDE_DIR}) ")    
    include_directories( ${GVDB_INCLUDE_DIR} )
	add_definitions(-DUSE_GVDB)
    if(WIN32)
      LIST(APPEND LIBRARIES_OPTIMIZED ${GVDB_LIB_DIR}/${GVDB_LIB} )
      LIST(APPEND LIBRARIES_DEBUG ${GVDB_LIB_DIR}/${GVDB_LIB} )
    endif(WIN32)
    LIST(APPEND PACKAGE_SOURCE_FILES ${GVDB_INCLUDE_DIR}/${GVDB_HEADERS} )  	
    source_group(GVDB FILES ${GVDB_INCLUDE_DIR}/${GVDB_HEADERS} ) 
 else()
    message( FATAL_ERROR "--> Unable to find GVDB") 
 endif()

####################################################################################
# Find CUDA
#
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../../../../LocalPackages/CUDA)
  Message(STATUS "Local CUDA detected. Using it")
  #TODO: if using VS 2015, point to CUDA v8
  set(CUDA_TOOLKIT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../LocalPackages/CUDA/v7.5_win")
endif()
find_package(CUDA)

if ( CUDA_FOUND )
    message( STATUS "--> Using package CUDA (ver ${CUDA_VERSION})") 
	add_definitions(-DUSE_CUDA)			
    include_directories(${CUDA_TOOLKIT_INCLUDE})	
	LIST(APPEND LIBRARIES_OPTIMIZED ${CUDA_CUDA_LIBRARY} )
	LIST(APPEND LIBRARIES_DEBUG ${CUDA_CUDA_LIBRARY} )
	LIST(APPEND PACKAGE_SOURCE_FILES ${CUDA_TOOLKIT_INCLUDE} )    
	source_group(CUDA FILES ${CUDA_TOOLKIT_INCLUDE} ) 
else()
   message ( FATAL_ERROR "---> Unable to find package CUDA")
endif()

####################################################################################
# Compile PTX Files
#
file(GLOB CUDA_FILES RELATIVE "${BASE_DIRECTORY}" *.cu *.cuh)
message ( STATUS "Build CUDA kernels: ${CUDA_FILES}" )

_COMPILEPTX ( SOURCES ${CUDA_FILES} TARGET_PATH ${EXECUTABLE_OUTPUT_PATH} GENERATED PTX_FILES INCLUDE "${GVDB_INCLUDE_DIR},${CMAKE_CURRENT_SOURCE_DIR}" OPTIONS -arch=compute_30 -code=sm_30 --ptxas-options=-v -O3 --use_fast_math --maxrregcount=128  )

####################################################################################
# Find Sample Utils
#
find_package(Utils)

#####################################################################################
# Require OpenGL
#
IF(WIN32)
  LIST(APPEND LIBRARIES_OPTIMIZED "opengl32.lib" )
  LIST(APPEND LIBRARIES_DEBUG "opengl32.lib" )
ENDIF()

#####################################################################################
# Source files for this project
#
file(GLOB SOURCE_FILES *.cpp *.hpp *.inl *.h *.c)

#####################################################################################
# Executable
#
unset ( ALL_SOURCE_FILES )
list( APPEND ALL_SOURCE_FILES ${SOURCE_FILES} )
list( APPEND ALL_SOURCE_FILES ${COMMON_SOURCE_FILES} )
list( APPEND ALL_SOURCE_FILES ${PACKAGE_SOURCE_FILES} )
list( APPEND ALL_SOURCE_FILES ${UTIL_SOURCE_FILES} )
list( APPEND ALL_SOURCE_FILES ${CUDA_FILES} )

if ( NOT DEFINED WIN32 )

  message ( STATUS "Processor: ${CMAKE_SYSTEM_PROCESSOR}" )
  
  if ( ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64" )
    find_library(CUDALIB cuda HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    set(libdeps GL GLEW X11 ${GVDB_ROOT_DIR}/lib/libgvdb.so ${CUDALIB} )
  else()
    find_library(GVDBLIB gvdb HINTS {$GVDB_ROOT_DIR}/lib )
    message ( STATUS "Find library GVDB: ${GVDBLIB}" )
 
    find_library(NVTOOLSEXT nvToolsExt HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    find_library(CUDART cudart HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    set(libdeps GL GLEW X11 ${GVDBLIB} cuda ${NVTOOLSEXT} ${CUDART})
  endif()
  LIST(APPEND LIBRARIES_OPTIMIZED ${libdeps})
  LIST(APPEND LIBRARIES_DEBUG ${libdeps})
ENDIF()

include_directories ("${CMAKE_CURRENT_SOURCE_DIR}")    
add_definitions(-DGVDB_IMPORTS -DGLEW_STATIC -DGLEW_NO_GLU)  
add_definitions(-DASSET_PATH="${ASSET_PATH}/")
add_executable (${PROJNAME} ${ALL_SOURCE_FILES} ${CUDA_FILES} ${PTX_SOURCES} )
set_property ( TARGET ${PROJNAME} APPEND PROPERTY DEPENDS ${PTX_SOURCES} )

if ( MSVC  )
    set_target_properties( ${PROJNAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH} )
    set_target_properties( ${PROJNAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${EXECUTABLE_OUTPUT_PATH} )
    set_target_properties( ${PROJNAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${EXECUTABLE_OUTPUT_PATH} )    	
	source_group(CUDA FILES ${CUDA_FILES})
    source_group(PTX FILES  ${PTX_FILES})
endif ()

#####################################################################################
# Install paths
#
# CMAKE_INSTALL_PREFIX -- path where library will be installed to

if ( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
   #if (WIN32)
      get_filename_component ( _instpath "${CMAKE_CURRENT_SOURCE_DIR}/../../_output" REALPATH )
   #else()
   #   get_filename_component ( _instpath "/usr/local/gvdb" REALPATH )
   #endif()
   set ( CMAKE_INSTALL_PREFIX ${_instpath} CACHE PATH "default install path" FORCE)   
endif()
get_filename_component( BIN_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/bin REALPATH)

#####################################################################################
# Install Binaries
#
_INSTALL ( FILES ${GVDB_GLSL} SOURCE ${GVDB_SHARE_DIR} DESTINATION ${EXECUTABLE_OUTPUT_PATH} )
_INSTALL ( FILES ${GVDB_PTX} SOURCE ${GVDB_SHARE_DIR} DESTINATION ${EXECUTABLE_OUTPUT_PATH} )
_INSTALL ( FILES ${GVDB_DLL} SOURCE ${GVDB_LIB_DIR} DESTINATION ${EXECUTABLE_OUTPUT_PATH} )
_INSTALL ( FILES ${GVDB_EXTRA} SOURCE ${GVDB_LIB_DIR} DESTINATION ${EXECUTABLE_OUTPUT_PATH} )

foreach ( _file IN ITEMS ${GVDB_GLSL} )
   list ( APPEND GVDB_GLSL_FILES "${GVDB_SHARE_DIR}/${_file}" )
endforeach() 
foreach ( _file IN ITEMS ${GVDB_PTX} )
   list ( APPEND GVDB_PTX_FILES "${GVDB_SHARE_DIR}/${_file}" )
endforeach() 
foreach ( _file IN ITEMS ${GVDB_DLL} )
   list ( APPEND GVDB_DLL_FILES "${GVDB_LIB_DIR}/${_file}" )
endforeach() 
foreach ( _file IN ITEMS ${GVDB_EXTRA} )
   list ( APPEND GVDB_EXTRA_FILES "${GVDB_LIB_DIR}/${_file}" )
endforeach() 
# INSTALL project : for the CMake official install
install ( FILES ${GVDB_GLSL_FILES} DESTINATION ${BIN_INSTALL_PATH} )
install ( FILES ${GVDB_PTX_FILES} DESTINATION ${BIN_INSTALL_PATH} )
install ( FILES ${GVDB_DLL_FILES} DESTINATION ${BIN_INSTALL_PATH} )
install ( FILES ${GVDB_EXTRA_FILES} DESTINATION ${BIN_INSTALL_PATH} )
install ( TARGETS ${PROJNAME} DESTINATION ${BIN_INSTALL_PATH} )

#####################################################################################
# Install PTX to share location
#  
_INSTALL_PTX ( FILES ${PTX_FILES_PATH} DESTINATION ${EXECUTABLE_OUTPUT_PATH} )
foreach ( _file IN ITEMS ${PTX_FILES} )
   list ( APPEND PTX_FILES_FILES "${EXECUTABLE_OUTPUT_PATH}/${_file}" )
endforeach() 
_CMAKE_INSTALL_PTX ( FILES ${PTX_FILES_FILES} DESTINATION ${BIN_INSTALL_PATH} )


#####################################################################################
# Library dependencies
#
set_property(GLOBAL PROPERTY DEBUG_CONFIGURATIONS Debug) 

foreach (loop_var IN ITEMS ${LIBRARIES_OPTIMIZED} )   
   target_link_libraries ( ${PROJNAME} optimized ${loop_var} )
endforeach()

foreach (loop_var IN ITEMS ${LIBRARIES_DEBUG} )
   target_link_libraries ( ${PROJNAME} debug ${loop_var} )
endforeach()

message ( STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}" )
message ( STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}" )
message ( STATUS "EXECUTABLE_OUTPUT_PATH: ${EXECUTABLE_OUTPUT_PATH}" )



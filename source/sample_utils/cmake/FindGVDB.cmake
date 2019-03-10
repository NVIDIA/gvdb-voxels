
#####################################################################################
# Find GVDB
#
unset(GVDB_FOUND CACHE)
unset(GVDB_INCLUDE_DIR CACHE)

if ( CUDA_VERSION )
	SET ( CUDA_SUFFIX "cu${CUDA_VERSION}" )
else()
	message ( FATAL_ERROR "\nNVIDIA CUDA not found.\n" )
endif()

if ( NOT DEFINED GVDB_ROOT_DIR )
  if (WIN32)
    get_filename_component ( BASEDIR "${CMAKE_MODULE_PATH}/../../_output" REALPATH )
  else()
    get_filename_component ( BASEDIR "/usr/local/gvdb/" REALPATH )
  endif()
  set ( GVDB_ROOT_DIR ${BASEDIR} CACHE PATH "Location of GVDB library" FORCE)
endif()
message ( STATUS "Searching for GVDB at.. ${GVDB_ROOT_DIR}")

#-- Paths to GVDB Library (cached so user can modify)
set ( GVDB_INCLUDE_DIR "${GVDB_ROOT_DIR}/include" CACHE PATH "Path to include files" FORCE)
set ( GVDB_LIB_DIR "${GVDB_ROOT_DIR}/lib" CACHE PATH "Path to libraries" FORCE)	
set ( GVDB_SHARE_DIR "${GVDB_ROOT_DIR}/lib" CACHE PATH "Path to share files" FORCE)	

find_path(GVDB_INCLUDE_DIR NAMES gvdb.h PATHS ${GVDB_ROOT_DIR} ${CMAKE_MODULE_PATH}/../../_output /usr/local/gvdb/)
file(GLOB GVDB_PTX ${GVDB_SHARE_DIR}/*.ptx)
file(GLOB GVDB_GLSL ${GVDB_SHARE_DIR}/*.glsl)
find_library(gvdb_lib NAMES gvdb libgvdb PATHS ${GVDB_LIB_DIR})
find_library(cudpp_lib NAMES cudpp libcudpp cudpp_${CUDA_SUFFIX} libcudpp_${CUDA_SUFFIX} PATHS ${GVDB_LIB_DIR})
find_library(cudpp_hash_lib NAMES cudpp_hash libcudpp_hash cudpp_hash_${CUDA_SUFFIX} libcudpp_hash_${CUDA_SUFFIX} PATHS ${GVDB_LIB_DIR})


include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set GVDB_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(GVDB  DEFAULT_MSG GVDB_INCLUDE_DIR GVDB_PTX GVDB_GLSL gvdb_lib cudpp_lib cudpp_hash_lib)
mark_as_advanced(GVDB_INCLUDE_DIR GVDB_PTX GVDB_GLSL gvdb_lib cudpp_lib cudpp_hash_lib)

set( GVDB_INCLUDE_DIRS ${GVDB_INCLUDE_DIR})
set( GVDB_LIBRARIES ${gvdb_lib} ${cudpp_lib} ${cudpp_hash_lib})

#-- We do not want user to modified these vars, but helpful to show them
message ( STATUS "  GVDB_ROOT_DIR: ${GVDB_ROOT_DIR}" )
message (  "  GVDB_LIB:  ${GVDB_LIBRARIES}" )
message ( STATUS "  GVDB_PTX:  ${GVDB_PTX}" )
message ( STATUS "  GVDB_GLSL: ${GVDB_GLSL}" )

mark_as_advanced(GVDB_FOUND)
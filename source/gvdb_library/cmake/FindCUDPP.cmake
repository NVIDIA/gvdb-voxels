
#####################################################################################
# Find CUDPP
#
unset(CUDPP_FOUND CACHE)
unset(CUDPP_INCLUDE_DIR CACHE)

if ( NOT DEFINED CUDPP_ROOT_DIR )
  if (WIN32)
    get_filename_component ( BASEDIR "${CMAKE_MODULE_PATH}/../../shared_cudpp" REALPATH )
  else()
    get_filename_component ( BASEDIR "/usr/local/cudpp/" REALPATH )
  endif()
  set ( CUDPP_ROOT_DIR ${BASEDIR} CACHE PATH "Location of cuDPP library" FORCE)
endif()
set( CUDPP_FOUND "YES" )

if ( CUDPP_ROOT_DIR )

    #-- Paths to CUDPP Library (cached so user can modify)
	set ( CUDPP_INCLUDE_DIR "${CUDPP_ROOT_DIR}/include" CACHE PATH "Path to include files" FORCE)
	set ( CUDPP_LIB_DIR "${CUDPP_ROOT_DIR}/lib" CACHE PATH "Path to libraries" FORCE)		

	#-------- Locate Header files
    set ( OK_H "0" )
	_FIND_FILE ( CUDPP_HEADERS CUDPP_INCLUDE_DIR "cudpp.h" "cudpp.h" OK_H )	
	if ( OK_H EQUAL 1 ) 
	    message ( STATUS "  Found. CUDPP Header files. ${CUDPP_INCLUDE_DIR}" )
	else()
	    message ( "  NOT FOUND. CUDPP Header files" )
		set ( CUDPP_FOUND "NO" )
	endif ()

    #-------- Locate Library	
	if (WIN32)
		message ( STATUS "  Locating: cudpp_${MSVC_VERSION}${CUDA_SUFFIX}x64.lib, MSVC: ${MSVC_VERSION}, CUDA: ${CUDA_SUFFIX}" )
	else()
		message ( STATUS "  Locating: libcudpp_${CUDA_SUFFIX}.so, CUDA: ${CUDA_SUFFIX}")
	endif()
     set ( OK_REL "0" )
     _FIND_FILE ( LIB1_REL CUDPP_LIB_DIR "cudpp_${MSVC_VERSION}${CUDA_SUFFIX}x64.lib" "libcudpp_${CUDA_SUFFIX}.so" OK_REL)
     _FIND_FILE ( LIB2_REL CUDPP_LIB_DIR "cudpp_hash_${MSVC_VERSION}${CUDA_SUFFIX}x64.lib" "libcudpp_hash_${CUDA_SUFFIX}.so" OK_REL )

	set (OK_DLL "0")
	_FIND_FILE( CUDPP_DLL CUDPP_LIB_DIR "cudpp_${MSVC_VERSION}${CUDA_SUFFIX}x64.dll" " " OK_DLL)
	_FIND_FILE( CUDPP_DLL CUDPP_LIB_DIR "cudpp_hash_${MSVC_VERSION}${CUDA_SUFFIX}x64.dll" " " OK_DLL)

	if (OK_REL EQUAL 2)
		message ( STATUS "  Found LIBs (Release): ${LIB1_REL} ${LIB2_REL}" )
		set ( CUDPP_LIB1_REL "${LIB1_REL}" CACHE INTERNAL "" FORCE)
		set ( CUDPP_LIB2_REL "${LIB2_REL}" CACHE INTERNAL "" FORCE)
		set (CUDPP_LIBRARIES ${CUDPP_LIBRARIES} ${CUDPP_LIB1_REL} ${CUDPP_LIB2_REL})
		set ( OK_DBG "0" )
		_FIND_FILE ( LIB1_DEBUG CUDPP_LIB_DIR "cudpp_${MSVC_VERSION}${CUDA_SUFFIX}x64d.lib" "libcudppd_${CUDA_SUFFIX}.so" OK_DBG )
		_FIND_FILE ( LIB2_DEBUG CUDPP_LIB_DIR "cudpp_hash_${MSVC_VERSION}${CUDA_SUFFIX}x64d.lib" "libcudpp_hashd_${CUDA_SUFFIX}.so" OK_DBG  )
	endif()

	if (OK_DLL EQUAL 2)
		message ( STATUS "  Found DLLs (Release): ${CUDPP_DLL}")
		set ( CUDPP_DLL "${CUDPP_DLL}" CACHE PATH "" FORCE)
		set (CUDPP_LIBRARIES ${CUDPP_LIBRARIES} ${CUDPP_DLL})
		set ( OK_DBG "0" )
		_FIND_FILE( LIB1_DEBUG CUDPP_LIB_DIR "cudpp_${MSVC_VERSION}${CUDA_SUFFIX}x64d.dll" "cudpp_${CUDA_SUFFIX}d.so" OK_DLL)
		_FIND_FILE( LIB2_DEBUG CUDPP_LIB_DIR "cudpp_hash_${MSVC_VERSION}${CUDA_SUFFIX}x64d.dll" "cudpp_hash_${CUDA_SUFFIX}d.so" OK_DLL)
	endif()

	if (OK_DBG EQUAL 2)
		message ( STATUS "  Found LIBs (Debug): ${LIB1_DEBUG} ${LIB2_DEBUG}" )
		set (CUDPP_LIBRARIES ${CUDPP_LIBRARIES} ${CUDPP_LIB1_DEBUG} ${CUDPP_LIB2_DEBUG})
		set ( CUDPP_LIB1_DEBUG "${LIB1_DEBUG}" CACHE INTERNAL "" FORCE)
		set ( CUDPP_LIB2_DEBUG "${LIB2_DEBUG}" CACHE INTERNAL "" FORCE)
	endif()

	if ( (NOT OK_DBG EQUAL 2) AND (NOT OK_REL EQUAL 2) )
		message ( "  NOT FOUND. Missing CUDPP .lib files. Built and install 'shared_cudpp' prior to gvdb" )
		set ( CUDPP_FOUND "NO" )
	endif()
endif()

if ( ${CUDPP_FOUND} STREQUAL "NO" )
   message( FATAL_ERROR "
      Please set CUDPP_ROOT_DIR to the root location
      of installed CUDPP library containing /include and /lib.
      Not found at CUDPP_ROOT_DIR: ${CUDPP_ROOT_DIR}\n"
   )
endif()

set ( CUDPP_LIB_DIR ${CUDPP_LIB_DIR} CACHE INTERNAL "" FORCE)

#-- We do not want user to modified these vars, but helpful to show them
message ( STATUS "  CUDPP_ROOT_DIR:    ${CUDPP_ROOT_DIR}" )
message ( STATUS "  CUDPP_INCLUDE_DIR: ${CUDPP_INCLUDE_DIR}" )
message ( STATUS "  CUDPP_LIB1:        ${CUDPP_LIB1_DEBUG}, ${CUDPP_LIB1_REL}" )
message ( STATUS "  CUDPP_LIB2:        ${CUDPP_LIB2_DEBUG}, ${CUDPP_LIB2_REL}" )
message ( STATUS "  CUDPP_LIBRARIES:   ${CUDPP_LIBRARIES}" )

mark_as_advanced(CUDPP_FOUND)







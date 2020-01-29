
set(VERSION "1.3.3")


if ( MSVC_VERSION )
	# MSVC_VERSION:
	# https://en.wikipedia.org/wiki/Microsoft_Visual_C%2B%2B
	# 1600 - VS 11.0 - 2010
	# 1700 - VS 11.0 - 2012
	# 1800 - VS 12.0 - 2013
	# 1900 - VS 14.0 - 2015
	# 1910 - VS 14.1 - 2017
	# 1911 - VS 14.1 - 2017
	# 1912 - VS 15.0 - 2017
   	message ( STATUS "MSVC_VERSION = ${MSVC_VERSION}" )
	if ( ${MSVC_VERSION} EQUAL "1600" ) 
	   SET ( MSVC_YEAR "2010" )
	endif()
	if ( ${MSVC_VERSION} EQUAL "1700" ) 
	   SET ( MSVC_YEAR "2012" )
	endif()
	if ( ${MSVC_VERSION} EQUAL "1800" ) 
	   SET ( MSVC_YEAR "2013" )
	endif()
	if ( ${MSVC_VERSION} EQUAL "1900" ) 
	   SET ( MSVC_YEAR "2015" )
	endif()
	if ( ${MSVC_VERSION} EQUAL "1910" ) 
	   SET ( MSVC_YEAR "2017" )
	endif()
	if ( ${MSVC_VERSION} EQUAL "1911" ) 
	   SET ( MSVC_YEAR "2017" )
	endif()
	if ( ${MSVC_VERSION} EQUAL "1912" ) 
	   SET ( MSVC_YEAR "2017" )    # USE LIBS FOR VS2015 due to cl.exe issue
	endif()
endif()

# Set the default build to Release.  Note this doesn't do anything for the VS
# default build target.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

#####################################################################################
function(_make_relative FROM TO OUT)
  #message(STATUS "FROM = ${FROM}")
  #message(STATUS "TO = ${TO}")
  
  get_filename_component(FROM ${FROM} ABSOLUTE)
  get_filename_component(TO ${TO} ABSOLUTE)
  
  string(REPLACE "/" ";" FROM_LIST ${FROM})
  string(REPLACE "/" ";" TO_LIST ${TO})
  
  #message(STATUS "FROM = ${FROM_LIST}")
  #message(STATUS "TO = ${TO_LIST}")
  
  list(LENGTH FROM_LIST flen)
  math(EXPR flen "${flen} - 1" )
  #message(STATUS "flen = ${flen}")
  list(LENGTH TO_LIST tlen)
  math(EXPR tlen "${tlen} - 1" )
  #message(STATUS "tlen = ${tlen}")
  
  set(REL_LIST)
  foreach(loop_var RANGE ${flen})
    #message(STATUS "i = ${loop_var}")
    if ((loop_var GREATER tlen) OR (loop_var EQUAL tlen))
      list(APPEND REL_LIST "..")
      #message(STATUS "descend")
    else()
      list(GET FROM_LIST ${loop_var} f)
      list(GET TO_LIST ${loop_var} t)
      #message(STATUS "f = ${f}")
      #message(STATUS "t = ${t}")
      if (${f} STREQUAL ${t})
        set(begin ${loop_var})
        #message(STATUS "equal")
      else()
        list(APPEND REL_LIST "..")
        #message(STATUS "descend")
      endif()
    endif()
  endforeach(loop_var)
  if (begin)
     math(EXPR begin "${begin} + 1" )
  endif()
  
  #message(STATUS "---")
  
  foreach(loop_var RANGE ${begin} ${tlen})
    #message(STATUS "i = ${loop_var}")
    #message(STATUS "t = ${t}")
    #message(STATUS "ascend")
    list(GET TO_LIST ${loop_var} t)
    list(APPEND REL_LIST ${t})
  endforeach(loop_var)
  
  #message(STATUS "relative = ${REL_LIST}")

  string (REPLACE ";" "/" _TMP_STR "${REL_LIST}")
  set (${OUT} "${_TMP_STR}" PARENT_SCOPE)
endfunction()

macro(_add_project_definitions name)
  if(MSVC)
    _make_relative("${EXECUTABLE_OUTPUT_PATH}/config" "${CMAKE_CURRENT_SOURCE_DIR}" TOPROJECT)
  else()
    _make_relative("${EXECUTABLE_OUTPUT_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}" TOPROJECT)
  endif()
  
  #message(STATUS "${TOPROJECT}")
  
  add_definitions(-DPROJECT_RELDIRECTORY="${TOPROJECT}/")
  add_definitions(-DPROJECT_ABSDIRECTORY="${CMAKE_CURRENT_SOURCE_DIR}/")
  add_definitions(-DPROJECT_NAME="${name}")  
  
endmacro(_add_project_definitions)

#####################################################################################
if(UNIX) 
  set(OS "linux")
  add_definitions(-DLINUX)
else(UNIX)
  if(APPLE)
  else(APPLE)
    if(WIN32)
      set(OS "win")
      add_definitions(-DNOMINMAX)
      if(MEMORY_LEAKS_CHECK)
        add_definitions(-DMEMORY_LEAKS_CHECK)
      endif()
    endif(WIN32)
  endif(APPLE)
endif(UNIX)


if (MSVC90)
  include_directories(${BASE_DIRECTORY}/stdint_old_msvc)
endif(MSVC90)



#######################################################
# LOCAL PACKAGES
#
macro ( _FIND_LOCALPACKAGE_CUDA ) 
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../../../../LocalPackages/CUDA)
    Message(STATUS "Local CUDA detected. Using it")
    set(CUDA_LOCALPACK_VER "9.0" CACHE STRING "CUDA Version")
    set(CUDA_TOOLKIT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../LocalPackages/CUDA/v${CUDA_LOCALPACK_VER}_win")
  endif()
endmacro()

macro ( _FIND_LOCALPACKAGE_OPTIX )
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../../../../LocalPackages/Optix)
    Message(STATUS "Local Optix detected. Using it")
    set(OPTIX_LOCALPACK_VER "5.0.0" CACHE STRING "OptiX Version")
    set(OPTIX_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../LocalPackages/Optix/Optix_${OPTIX_LOCALPACK_VER}_win64")
  endif()
endmacro ()

#####################################################################################
# Optional GVDB
#
macro(_add_package_GVDB)

  message( STATUS "<-- Searching for package GVDB")    
  find_package(GVDB)

  if ( GVDB_FOUND )
	message( STATUS "--> Using package GVDB. ${GVDB_LIB_DIR} ")    
    include_directories( ${GVDB_INCLUDE_DIR} )
    add_definitions(-DUSE_GVDB)
	add_definitions(-DSIM_CODE)
    if (WIN32)
       LIST(APPEND LIBRARIES_OPTIMIZED ${GVDB_LIB_DIR}/${GVDB_LIB} )
       LIST(APPEND LIBRARIES_DEBUG ${GVDB_LIB_DIR}/${GVDB_LIB} )
	else ()
	   find_library( LIBGVDB gvdb HINTS ${GVDB_LIB_DIR} )
	   LIST(APPEND LIBRARIES_OPTIMIZED ${LIBGVDB} )
       LIST(APPEND LIBRARIES_DEBUG ${LIBGVDB} )
	endif()
    LIST(APPEND PACKAGE_SOURCE_FILES ${GVDB_INCLUDE_DIR}/${GVDB_HEADERS} )  	
    source_group(GVDB FILES ${GVDB_INCLUDE_DIR}/${GVDB_HEADERS} ) 
 else()
    message( FATAL_ERROR "--> Unable to find GVDB") 
 endif()

endmacro()


#####################################################################################
# Optional ZLIB
#
macro(_add_package_ZLIB)
  if(EXISTS ${BASE_DIRECTORY}/zlib)
    set(ZLIB_ROOT ${BASE_DIRECTORY}/zlib)
  endif()
  Message(STATUS "--> using package ZLIB")
  find_package(ZLIB)
  if(ZLIB_FOUND)
      include_directories(${ZLIB_INCLUDE_DIR})
      LIST(APPEND PACKAGE_SOURCE_FILES
        ${ZLIB_HEADERS}
        )
      LIST(APPEND LIBRARIES_OPTIMIZED ${ZLIB_LIBRARY})
      LIST(APPEND LIBRARIES_DEBUG ${ZLIB_LIBRARY})
  else()
    Message(WARNING "ZLIB not available. setting NOGZLIB define")
    add_definitions(-DNOGZLIB)
  endif()
endmacro()


#####################################################################################
# Optional Utils package
#
macro(_add_package_Utils)
   
   find_package(Utils)

   if ( UTILS_FOUND )   
		
		if (WIN32)
			# Windows platform
			if(REQUIRE_OPENGL)				
				add_definitions(-DBUILD_OPENGL)
				set(PLATFORM_LIBRARIES ${OPENGL_LIBRARY} )				
			endif()
		else()
			# Linux platform
			if(REQUIRE_OPENGL)
				add_definitions(-DBUILD_OPENGL)
				set(PLATFORM_LIBRARIES GL GLU GLEW X11)				
			endif()
			if(USE_NVTX)
				find_library(LIBNVTX nvToolsExt HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
				LIST(APPEND LIBRARIES_OPTIMIZED ${LIBNVTX})
				LIST(APPEND LIBRARIES_DEBUG {LIBNVTX})
			endif()			
		endif()		
	endif()

endmacro()


#####################################################################################
# Optional OptiX package
#
macro(_add_package_Optix)

	Message(STATUS "<-- Searching for package OptiX")
	
	_FIND_LOCALPACKAGE_OPTIX()

	find_package(Optix)  
	
	if (OPTIX_FOUND)
      if ( NOT DEFINED USE_OPTIX )	       
	       SET(USE_OPTIX ON CACHE BOOL "Use OPTIX")
      endif()
	  if ( USE_OPTIX ) 			
			add_definitions(-DUSE_OPTIX)			
			add_definitions(-DBUILD_OPTIX)	
			include_directories(${OPTIX_INCLUDE_DIR})
			if (WIN32)
				LIST(APPEND LIBRARIES_OPTIMIZED ${OPTIX_LIB_DIR}/${OPTIX_LIB1} )
				LIST(APPEND LIBRARIES_OPTIMIZED ${OPTIX_LIB_DIR}/${OPTIX_LIB2} )
				LIST(APPEND LIBRARIES_DEBUG ${OPTIX_LIB_DIR}/${OPTIX_LIB1} )
				LIST(APPEND LIBRARIES_DEBUG ${OPTIX_LIB_DIR}/${OPTIX_LIB2} )
			else()
				find_library(LIBOPTIX optix HINTS ${OPTIX_LIB_DIR})
				find_library(LIBOPTIXU optixu HINTS ${OPTIX_LIB_DIR})
				LIST(APPEND LIBRARIES_OPTIMIZED ${LIBOPTIX} ${LIBOPTIXU} )
				LIST(APPEND LIBRARIES_DEBUG ${LIBOPTIX} ${LIBOPTIXU} )
			endif()
			LIST(APPEND PACKAGE_SOURCE_FILES ${OPTIX_INCLUDE_DIR}/${OPTIX_HEADERS} )      

			_COMPILEPTX ( SOURCES ${UTIL_OPTIX_KERNELS} TARGET_PATH ${EXECUTABLE_OUTPUT_PATH} GENERATED UTIL_OPTIX_PTX GENPATHS UTIL_OPTIX_PTX_PATHS INCLUDE "${OPTIX_ROOT_DIR}/include/,${GVDB_INCLUDE_DIR},${CMAKE_CURRENT_SOURCE_DIR}" OPTIONS -arch=compute_30 -code=sm_30 --ptxas-options=-v -O3 --use_fast_math --maxrregcount=128 )
			message (STATUS "  OptiX FILES:   ${UTIL_OPTIX_FILES}")
			message (STATUS "  OptiX KERNELS: ${UTIL_OPTIX_KERNELS}")
			message (STATUS "  OptiX PTX:     ${UTIL_OPTIX_PTX}")
			Message(STATUS "--> Using package OptiX\n")

	  endif()	  
    else()
      Message(STATUS "Warning: OptiX not found. The gInteractiveOptix sample will not work without OptiX.")
	  SET(USE_OPTIX OFF CACHE BOOL "Use Optix" FORCE)
 endif()
endmacro()

#####################################################################################
# 
#
set(TINYTHREADPP_DIRECTORY ${BASE_DIRECTORY}/tinythreadpp)

macro(_add_package_tinythreadpp)
  Message(STATUS "--> using package TinyThread++")
  include_directories(${TINYTHREADPP_DIRECTORY}/src)
  set(PROJECT_TINYTHREADPP ${TINYTHREADPP_DIRECTORY}/src/fast_mutex.h ${TINYTHREADPP_DIRECTORY}/src/tinythread.h ${TINYTHREADPP_DIRECTORY}/src/tinythread.cpp)

  LIST(APPEND PACKAGE_SOURCE_FILES
        ${PROJECT_TINYTHREADPP}
        )
endmacro(_add_package_tinythreadpp)

#####################################################################################
# Glew : source or lib
if(GLEW_SOURCE)
  message(STATUS "found Glew source code. Using it instead of library")
  add_definitions(-DGLEW_STATIC)
else()
    message(STATUS "using GLEW library")
    LIST(APPEND PLATFORM_LIBRARIES ${GLEW_LIBRARY})
endif()
add_definitions(-DGLEW_NO_GLU)

 ####################################################################################
 # XF86
if (UNIX)
 LIST(APPEND PLATFORM_LIBRARIES "Xxf86vm")
endif()

#####################################################################################
# NSight
#
# still need the include directory when no use of NSIGHT: for empty #defines
macro(_add_package_NSight)
  Message(STATUS "--> using package NSight")
  include_directories(
      ${BASE_DIRECTORY}/NSight
  )
  if(SUPPORT_NVTOOLSEXT)
    link_directories(
        ${BASE_DIRECTORY}/NSight
    )
    LIST(APPEND PACKAGE_SOURCE_FILES 
      ${BASE_DIRECTORY}/NSight/NSightEvents.h
      ${BASE_DIRECTORY}/NSight/nvToolsExt.h
    )
    add_definitions(-DSUPPORT_NVTOOLSEXT)
    if(ARCH STREQUAL "x86")
      SET(NSIGHT_DLL ${BASE_DIRECTORY}/NSight/nvToolsExt32_1.dll)
      SET(NSIGHT_LIB ${BASE_DIRECTORY}/NSight/nvToolsExt32_1.lib)
    else()
      SET(NSIGHT_DLL ${BASE_DIRECTORY}/NSight/nvToolsExt64_1.dll)
      SET(NSIGHT_LIB ${BASE_DIRECTORY}/NSight/nvToolsExt64_1.lib)
    endif()
    LIST(APPEND LIBRARIES_OPTIMIZED ${NSIGHT_LIB})
    LIST(APPEND LIBRARIES_DEBUG ${NSIGHT_LIB})
  endif()
endmacro()

## -- no shared_sources to include (-Rama)
#  include_directories(
#  ${BASE_DIRECTORY}/shared_sources
#  )

#####################################################################################
# Macro to download a file from a URL
#
macro(_download_file _URL _TARGET _FORCE)
  if(${_FORCE} OR (NOT EXISTS ${_TARGET}))
    Message(STATUS "downloading ${_URL} ==> ${_TARGET}")
    file(DOWNLOAD ${_URL} ${_TARGET} SHOW_PROGRESS)
  else()
    Message(STATUS "model ${_TARGET} already loaded...")
  endif()
endmacro()
#
# example: _download_files("${FILELIST}"  "http://..." "${BASE_DIRECTORY}/shared_external/..." ${MODELS_DOWNLOAD_FORCE} )
#
macro(_download_files _FILELIST _URL _TARGET _FORCE )
  foreach(_FILE ${_FILELIST})
    if(${_FORCE} OR (NOT EXISTS "${_TARGET}/${_FILE}"))
      Message(STATUS "*******************************************")
      Message(STATUS "downloading ${_URL}/${_FILE}\n ==>\n ${_TARGET}")
      Message(STATUS "*******************************************")
      file(DOWNLOAD ${_URL}/${_FILE} ${_TARGET}/${_FILE} SHOW_PROGRESS)
    else()
      Message(STATUS "model ${_FILE} already loaded...")
    endif()
  endforeach(_FILE)
endmacro()


set(VERSION "1.3.3")

set(SUPPORT_NVTOOLSEXT OFF CACHE BOOL "Use NSight for custom markers")
if(WIN32)
  SET( MEMORY_LEAKS_CHECK OFF CACHE BOOL "Check for Memory leaks" )
  SET(USE_GLFW OFF CACHE BOOL "Use GLFW instead of our own simple Window management")
else(WIN32)
  SET(USE_GLFW ON CACHE BOOL "Use GLFW instead of our own simple Window management")
endif(WIN32)

SET(RESOURCE_DIRECTORY "${BASE_DIRECTORY}/resources")
add_definitions(-DRESOURCE_DIRECTORY="${RESOURCE_DIRECTORY}/")

Message(STATUS "BASE_DIRECTORY = ${BASE_DIRECTORY}")
Message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")

# Specify the list of directories to search for cmake modules.
set(CMAKE_MODULE_PATH
    ${BASE_DIRECTORY}/cmake
)

set( CMAKE_FIND_ROOT_PATH "")

if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
  set (ARCH "x64" CACHE STRING "CPU Architecture")
else ()
  set (ARCH "x86" CACHE STRING "CPU Architecture")
endif()

set(EXECUTABLE_OUTPUT_PATH
    ${BASE_DIRECTORY}/bin_${ARCH}
    CACHE PATH
    "Directory where executables will be stored"
)


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

  set(KERNELS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/kernels/)
  file(GLOB KERNELS_SOURCE_FILES RELATIVE ${KERNELS_SOURCE_DIR} kernels/*.cu)
  foreach( _FILE ${KERNELS_SOURCE_FILES} )
    string(REPLACE ".cu" ".ptx" _FILE_PTX ${_FILE})
    string(TOUPPER ${_FILE_PTX} _FILE_UPPER)
    string(REPLACE "." "_" _MACRO ${_FILE_UPPER})
    add_definitions(-D${_MACRO}="${CMAKE_INSTALL_PREFIX}/lib/${_FILE_PTX}")
  endforeach ( _FILE )

  set(SHADERS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders/)
  file(GLOB SHADERS_SOURCE_FILES RELATIVE ${SHADERS_SOURCE_DIR} shaders/*.glsl)
  foreach( _FILE ${SHADERS_SOURCE_FILES} )
    string(TOUPPER ${_FILE} _FILE_UPPER)
    string(REPLACE "." "_" _MACRO ${_FILE_UPPER})
    add_definitions(-D${_MACRO}="${CMAKE_INSTALL_PREFIX}/lib/${_FILE}")
  endforeach ( _FILE )

  
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


# Macro for adding files close to the executable
macro(_copy_files_to_target target thefiles)
    if(WIN32)	    
        foreach (FFF ${thefiles} )          
            add_custom_command(
              TARGET ${target} POST_BUILD
              COMMAND ${CMAKE_COMMAND} -E copy ${FFF} $<TARGET_FILE_DIR:${target}>                
            )          
        endforeach()
    endif()
endmacro()

# ===============> OpenGL
find_package(OpenGL)

# ===============> GLEW
if (NOT APPLE)
  find_package(GLEW REQUIRED)
  if(NOT GLEW_FOUND)
    message(WARNING "Try to set GLEW_LOCATION")
  else()
    include_directories(${GLEW_INCLUDE_DIR} )
  endif()
endif()


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

macro ( _FIND_LOCALPACKAGE_OPENVDB ) 
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../../../../LocalPackages/OpenVDB)
    Message(STATUS "Local OpenVDB detected. Using it")    
	set(OPENVDB_LOCALPACK_VER "4.0.1" CACHE STRING "OpenVDB Version")
    set(OPENVDB_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../LocalPackages/OpenVDB/OpenVDB_${OPENVDB_LOCALPACK_VER}_vs2015" )	
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
# Optional OpenVDB
#
macro(_add_package_OpenVDB)

   Message(STATUS "\n<-- Searching for OpenVDB")		

  _FIND_LOCALPACKAGE_OPENVDB ()

  if ( NOT DEFINED USE_OPENVDB )	  
	SET(USE_OPENVDB OFF CACHE BOOL "Use OpenVDB") 
  endif ()

  find_package(OpenVDB)  
  if (OPENVDB_FOUND)
      if ( NOT DEFINED USE_OPENVDB )	       
	       SET(USE_OPENVDB ON CACHE BOOL "Use OpenVDB")
      endif()
	  if ( USE_OPENVDB ) 
		  Message(STATUS "--> Using package OpenVDB")
		  
		  add_definitions(-DUSEOPENVDB)
		  add_definitions(-DOPENEXR_DLL)
		  add_definitions(-DOPENVDB_3_ABI_COMPATIBLE)
		  add_definitions(-DBUILD_OPENVDB)	 
		  add_definitions(-DOPENVDB_STATICLIB)
		  add_definitions(-DOPENVDB_USE_BLOSC)

		  message ( STATUS "Adding OpenVDB includes: ${OPENVDB_INCLUDE_DIR}" )
		  include_directories(${OPENVDB_INCLUDE_DIR})
		  include_directories ("${OPENVDB_INCLUDE_DIR}")
		  include_directories ("${OPENVDB_INCLUDE_DIR}/IlmBase")
		  include_directories ("${OPENVDB_INCLUDE_DIR}/tbb")     
		  LIST(APPEND LIBRARIES_OPTIMIZED ${OPENVDB_LIB_RELEASE} )
		  LIST(APPEND LIBRARIES_DEBUG ${OPENVDB_LIB_DEBUG} )		

		  if ( OPENVDB_LIB_DIR STREQUAL "" ) 
		     message ( FATAL_ERROR "OpenVDB package found, but OpenVDB library directory not found. OPENVDB_LIB_DIR." )		   
		  endif ()

		if (MSVC_VERSION EQUAL 1600)
		   set ( MSVCX "vc10" )
		endif()
		if (MSVC_VERSION EQUAL 1700)
		   set ( MSVCX "vc11" )
		endif()
		if (MSVC_VERSION EQUAL 1800)
		   set ( MSVCX "vc12" )
		endif()
		if (MSVC_VERSION EQUAL 1900)
		   set ( MSVCX "vc14" )
		endif()

		  LIST(APPEND LIBRARIES_OPTIMIZED "${OPENVDB_LIB_DIR}/Blosc.lib" )	
		  LIST(APPEND LIBRARIES_DEBUG "${OPENVDB_LIB_DIR}/Blosc.lib" )	
		  LIST(APPEND LIBRARIES_OPTIMIZED "${OPENVDB_LIB_DIR}/Half.lib" )	
		  LIST(APPEND LIBRARIES_DEBUG "${OPENVDB_LIB_DIR}/Half.lib" )	
		  LIST(APPEND LIBRARIES_OPTIMIZED "${OPENVDB_LIB_DIR}/zlib.lib" )	
		  LIST(APPEND LIBRARIES_DEBUG "${OPENVDB_LIB_DIR}/zlibd.lib" )	
		  LIST(APPEND LIBRARIES_OPTIMIZED "${OPENVDB_LIB_DIR}/tbb.lib" )
		  LIST(APPEND LIBRARIES_DEBUG "${OPENVDB_LIB_DIR}/tbb_debug.lib" )		  
		  LIST(APPEND LIBRARIES_OPTIMIZED "${OPENVDB_LIB_DIR}/boost_system-${MSVCX}0-mt-1_64.lib" )
		  LIST(APPEND LIBRARIES_DEBUG "${OPENVDB_LIB_DIR}/boost_system-${MSVCX}0-mt-gd-1_64.lib" )
		  LIST(APPEND LIBRARIES_OPTIMIZED "${OPENVDB_LIB_DIR}/boost_thread-${MSVCX}0-mt-1_64.lib" )
		  LIST(APPEND LIBRARIES_DEBUG "${OPENVDB_LIB_DIR}/boost_thread-${MSVCX}0-mt-gd-1_64.lib" )
		  
		  LIST(APPEND PACKAGE_SOURCE_FILES ${OPENVDB_HEADERS} )
	  endif ()
 else()
     SET(USE_OPENVDB OFF CACHE BOOL "Use OpenVDB" FORCE)
 endif()
endmacro()


#####################################################################################
# Optional CUDA package
#

macro(_set_cuda_suffix)
	#------ CUDA VERSION
	if ( CUDA_VERSION ) 	  
	  if ( ${CUDA_VERSION} EQUAL "8.0" )
		SET ( CUDA_SUFFIX "cu8" )
	  endif ()
	  if ( ${CUDA_VERSION} EQUAL "9.0" )
		SET ( CUDA_SUFFIX "cu9" )
	  endif ()
	  if ( ${CUDA_VERSION} EQUAL "9.1" )
		SET ( CUDA_SUFFIX "cu9" )
	  endif ()
	else()
	  message ( FATAL_ERROR "\nNVIDIA CUDA not found.\n" )
	endif()
endmacro()

macro(_add_package_CUDA)

	Message(STATUS "\n<-- Searching for CUDA")		

	_FIND_LOCALPACKAGE_CUDA ()

	find_package(CUDA)

	if ( CUDA_FOUND )
		_set_cuda_suffix()
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

endmacro()

#####################################################################################
# Optional CUDPP package
#
macro(_add_package_CUDPP)

  Message(STATUS "\n<-- Searching for CUDPP")		

  find_package(CUDPP)  

  if (CUDPP_FOUND)
      if ( NOT DEFINED USE_CUDPP )	       
	       SET(USE_CUDPP ON CACHE BOOL "Use CUDPP")
      endif()
	  if ( USE_CUDPP ) 
		Message(STATUS "--> Using package CUDPP")		
		add_definitions(-DUSE_CUDPP)			
		include_directories( ${CUDPP_INCLUDE_DIR} )		
		link_directories( ${CUDPP_LIB_DIR} )
		if ( NOT ${CUDPP_LIB1_REL} STREQUAL "")
			LIST(APPEND LIBRARIES_OPTIMIZED ${CUDPP_LIB1_REL} )		
		endif()
		if ( NOT ${CUDPP_LIB2_REL} STREQUAL "")
			LIST(APPEND LIBRARIES_OPTIMIZED ${CUDPP_LIB2_REL} )
		endif()
		if ( NOT ${CUDPP_LIB1_DEBUG} STREQUAL "")
			LIST(APPEND LIBRARIES_DEBUG ${CUDPP_LIB1_DEBUG} )
		endif()
		if ( NOT ${CUDPP_LIB2_DEBUG} STREQUAL "")
			LIST(APPEND LIBRARIES_DEBUG ${CUDPP_LIB2_DEBUG} )	
		endif()
		LIST(APPEND PACKAGE_SOURCE_FILES ${CUDPP_INCLUDE_DIR}/${CUDPP_HEADERS} ) 
  	  else()
		Message(FATAL_ERROR "--> NOT USING package CUDPP. Set USE_CUDPP true.")		
   endif ()
else ()
   SET(USE_CUDPP OFF CACHE BOOL "Use CUDPP") 
endif()

endmacro()

#####################################################################################
# Optional OptiX package
#
macro(_add_package_Optix)
  find_package(Optix)  
  if (OPTIX_FOUND)
      if ( NOT DEFINED USE_OPTIX )	       
	       SET(USE_OPTIX ON CACHE BOOL "Use OPTIX")
      endif()
	  if ( USE_OPTIX ) 
			Message(STATUS "--> Using package OptiX")
			add_definitions(-DUSEOPTIX)
			include_directories(${OPTIX_INCLUDE_DIR})
			LIST(APPEND LIBRARIES_OPTIMIZED ${OPTIX_LIB} )
			LIST(APPEND LIBRARIES_DEBUG ${OPTIX_LIB} )
			LIST(APPEND PACKAGE_SOURCE_FILES ${OPTIX_HEADERS} )      
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
#
set(PLATFORM_LIBRARIES
    ${OPENGL_LIBRARY}
)
set(COMMON_SOURCE_FILES)
LIST(APPEND COMMON_SOURCE_FILES
    ${BASE_DIRECTORY}/resources.h
    ${BASE_DIRECTORY}/resources.rc
)

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

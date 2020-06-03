# Try to find OptiX project dll/so and headers
#

# outputs
unset(OPTIX_SHARED_LIBS CACHE)
unset(OPTIX_LIB CACHE)
unset(OPTIX_LIBU CACHE)
unset(OPTIX_FOUND CACHE)
unset(OPTIX_INCLUDE_DIR CACHE)

message(STATUS "Finding OptiX...")

macro ( folder_list result curdir )
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  foreach ( child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
        LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()

macro(_check_version_on_folder checkdir bestver bestpath)
  string ( REGEX MATCH "OptiX SDK.*([0-9]+).([0-9]+).([0-9]+)" result "${checkdir}" )
  if ( "${result}" STREQUAL "${checkdir}" )
    # found a path with versioning 
    SET ( ver "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}" )
	# GVDB 1.1.1's samples haven't been ported to OptiX 7.0 yet. But GVDB itself
	# (not the samples) should be compatible with all versions of OptiX.
    if ( (ver VERSION_GREATER bestver) AND (ver VERSION_LESS "7.0.0") )
      SET ( bestver ${ver} )
      SET ( bestpath "${basedir}/${checkdir}" )
    endif ()
  endif()
endmacro()

macro(_find_version_path targetVersion targetPath searchList )
  unset ( targetVersion )
  unset ( targetPath )
  SET ( bestver "0.0.0" )
  SET ( bestpath "" )
  
  Message(STATUS "searchList: ${searchList}" )
  foreach ( basedir ${searchList} )
    folder_list ( dirList ${basedir} )  
    MESSAGE(STATUS "Folder List : ${dirList}")
    Message(STATUS "dirList: ${dirList}" )
    foreach ( checkdir ${dirList} )
      _check_version_on_folder(${checkdir} bestver bestpath)
    endforeach ()   
  endforeach ()  
  SET ( ${targetVersion} "${bestver}" )
  SET ( ${targetPath} "${bestpath}" )
endmacro()

# Finds a file named either fileNameWin or fileNameUnix (depending on the system)
# in searchFolder and appends it to the list in targetVar. If the file was found,
# adds 1 to numFound.
macro(_find_file targetList fileNameWin fileNameUnix searchFolder numFound)
  unset(fileName)
  unset(fileNameLength)
  unset(foundFile CACHE)
  
  # Choose file name
  if(WIN32)
    set(fileName ${fileNameWin})
  elseif(UNIX)
    set(fileName ${fileNameUnix})
  else()
    message("FindOptix.cmake:_find_files: Neither WIN32 or UNIX was defined!")
    return()
  endif()
  
  unset(fileNameLength)
  string(LENGTH ${fileName} fileNameLength)
  if(fileNameLength EQUAL 0)
    return()
  endif()
  
  find_file(foundFile ${fileName} ${searchFolder})
  
  if(NOT ("${foundFile}" STREQUAL "foundFile-NOTFOUND"))
    # Add one to numFound
    math(EXPR ${numFound} "${${numFound}}+1")
  endif()
  
  list(APPEND ${targetList} "${foundFile}")
endmacro()

# Main code for finding OptiX

if ( NOT OPTIX_ROOT_DIR )
  # -------------------------------------------------------------------
  # Locate OPTIX by version
  set(OPTIX_LOCATION "C:/ProgramData/NVIDIA Corporation/") 

  STRING(REGEX REPLACE "\\\\" "/" OPTIX_LOCATION "${OPTIX_LOCATION}") 

  set ( SEARCH_PATHS
    "${OPTIX_LOCATION}" # this could be set to C:\ProgramData\NVIDIA Corporation; best version will be taken.
  )

  message(STATUS "  OptiX search paths: : ${SEARCH_PATHS}")

  # Default search
  if ( WIN32 )
    set(OPTIX_ROOT_DIR "")
    set (OPTIX_VERSION "0.0.0" )
  else ()
     set (OPTIX_ROOT_DIR "/usr/local/optix" CACHE PATH "" FORCE)
     set (OPTIX_VERSION "4.0.0" )
  endif()
  
  if(WIN32)
    _find_version_path ( OPTIX_VERSION OPTIX_ROOT_DIR "${SEARCH_PATHS}" )
    message(STATUS "OptiX version string: : ${OPTIX_VERSION}")
  else()
	message(STATUS "OptiX root directory (Linux) : ${OPTIX_ROOT_DIR}")
  endif()
else()
  message(STATUS "OPTIX_ROOT_DIR was set to ${OPTIX_ROOT_DIR}. Detecting version number.")
  string ( REGEX MATCH "OptiX SDK.*([0-9]+).([0-9]+).([0-9]+)" _RESULT "${OPTIX_ROOT_DIR}" )
  set(OPTIX_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
  message(STATUS "Detected OptiX version (OPTIX_VERSION): ${OPTIX_VERSION}")
endif()

#-------- Locate Libs
set ( OPTIX_BIN_DIR "${OPTIX_ROOT_DIR}/bin64" CACHE PATH "Optix Binary path" FORCE)
set ( OPTIX_LIB_DIR "${OPTIX_ROOT_DIR}/lib64" CACHE PATH "Optix Library path" FORCE)
set ( OPTIX_INCLUDE_DIR "${OPTIX_ROOT_DIR}/include" CACHE PATH "Optix Header path" FORCE )
set ( OPTIX_FOUND "YES" )

message ( STATUS "  OptiX Lib:     ${OPTIX_LIB_DIR}" )
message ( STATUS "  OptiX Bin:     ${OPTIX_BIN_DIR}" )
message ( STATUS "  OptiX Include: ${OPTIX_INCLUDE_DIR}" )

# Shared libraries (TODO: Clean this up)
set(OK_DLL "0")
if(WIN32)
  _find_file(OPTIX_SHARED_LIBS "optix.${OPTIX_VERSION}.dll" "" ${OPTIX_BIN_DIR} OK_DLL)
  _find_file(OPTIX_SHARED_LIBS "optixu.${OPTIX_VERSION}.dll" "" ${OPTIX_BIN_DIR} OK_DLL)
else()
  _find_file(OPTIX_SHARED_LIBS "" "liboptix.so" ${OPTIX_LIB_DIR} OK_DLL)
  _find_file(OPTIX_SHARED_LIBS "" "liboptixu.so" ${OPTIX_LIB_DIR} OK_DLL)
endif()

if(NOT OK_DLL EQUAL 2)
  message("  OptiX NOT FOUND. Could not find at least one of optix.${OPTIX_VERSION}.dll or optixu.${OPTIX_VERSION}.dll in ${OPTIX_BIN_DIR}.")
  set(OPTIX_FOUND "NO")
endif()

#Libraries
set(OK_LIB "0")
_find_file(OPTIX_LIB "optix.${OPTIX_VERSION}.lib" "liboptix.so" ${OPTIX_LIB_DIR} OK_LIB)
_find_file(OPTIX_LIBU "optixu.${OPTIX_VERSION}.lib" "liboptixu.so" ${OPTIX_LIB_DIR} OK_LIB)
if( NOT OK_LIB EQUAL 2 )
   message("  OptiX NOT FOUND. Could not find at least one of optix.${OPTIX_VERSION}.lib or optixu.${OPTIX_VERSION}.lib in ${OPTIX_LIB_DIR}.")
   set(OPTIX_FOUND "NO")
endif()

# set(OK_HEADERS "0")
# _find_file(OPTIX_HEADERS "optix.h" "optix.h" ${OPTIX_INCLUDE_DIR} OK_HEADERS )
# if( NOT OK_HEADERS EQUAL 1 )	
#    message("  OptiX NOT FOUND. Could not find the OptiX header file, optix.h, in ${OPTIX_INCLUDE_DIR}.")
#    set(OPTIX_FOUND "NO")
# endif()

if ( OPTIX_FOUND STREQUAL "NO" )
  message(FATAL_ERROR "
      OPTIX not found. Please specify OPTIX_ROOT_DIR
      containing the include, lib64, bin64 folders. 
      Not found at OPTIX_ROOT_DIR: ${OPTIX_ROOT_DIR}\n"
  )
else()
  message ( STATUS "  OptiX Location: ${OPTIX_ROOT_DIR}" )
endif()

message(STATUS "  OptiX Libs: ${OPTIX_LIB}, ${OPTIX_LIBU}")
message(STATUS "  OptiX Shared Libraries: ${OPTIX_SHARED_LIBS}")
# message(STATUS "  OptiX Headers: ${OPTIX_HEADERS}")

SET(OPTIX_SHARED_LIBS ${OPTIX_SHARED_LIBS} CACHE STRING "" FORCE)
SET(OPTIX_LIB ${OPTIX_LIB} CACHE STRING "" FORCE)
SET(OPTIX_LIBU ${OPTIX_LIBU} CACHE STRING "" FORCE)

mark_as_advanced( OPTIX_FOUND )

# Try to find OptiX project dll/so and headers
#

# outputs
unset(OPTIX_DLL CACHE)
unset(OPTIX_LIB1 CACHE)
unset(OPTIX_LIB2 CACHE)
unset(OPTIX_FOUND CACHE)
unset(OPTIX_INCLUDE_DIR CACHE)

set ( OPTIX_MAX_VER "5.0.0" CACHE STRING "" )

macro ( folder_list result curdir substring )
  FILE(GLOB children RELATIVE ${curdir} "${curdir}/*")      
  SET(dirlist "")
  foreach ( child ${children})  
    IF(IS_DIRECTORY ${curdir}/${child})
        LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()

macro(_find_version_path targetVersion targetPath rootName avoidName searchList platform )
  unset ( targetVersion )
  unset ( targetPath )
  SET ( bestver "0.0.0" )
  SET ( bestpath "" )
  foreach ( searchdir ${searchList} )
    get_filename_component ( basedir ${searchdir} ABSOLUTE )    
    folder_list ( dirList ${basedir} ${platform} )	
	foreach ( checkdir ${dirList} )	    	    
	    string ( TOLOWER ${checkdir} checklower )			 		
		string ( FIND ${checklower} ${avoidName} result )
		if ( ${result} EQUAL "-1" )		    
			string ( REGEX MATCH "${rootName}(.*)([0-9]+).([0-9]+).([0-9]+)(.*)$" result "${checklower}" )
			if ( "${result}" STREQUAL "${checklower}" )
			   # found a path with versioning 
			   SET ( ver "${CMAKE_MATCH_2}.${CMAKE_MATCH_3}.${CMAKE_MATCH_4}" )
			   if ( ver GREATER bestver AND NOT (ver GREATER ${OPTIX_MAX_VER}) )
	  			 SET ( bestver ${ver} )
	  			 SET ( bestpath "${basedir}/${checkdir}" )				 
	  		 endif ()
			endif()	  
		endif()
	  endforeach ()		
  endforeach ()  
  SET ( ${targetVersion} "${bestver}" )
  SET ( ${targetPath} "${bestpath}" )
endmacro()

macro(_find_files targetVar incDir dllName dllName64 folder)
  unset ( fileList )
  if(ARCH STREQUAL "x86")
      file(GLOB fileList "${${incDir}}/../${folder}${dllName}")
      list(LENGTH fileList NUMLIST)
      if(NUMLIST EQUAL 0)
        file(GLOB fileList "${${incDir}}/${folder}${dllName}")
      endif()
  else()
      file(GLOB fileList "${${incDir}}/../${folder}${dllName64}")
      list(LENGTH fileList NUMLIST)
      if(NUMLIST EQUAL 0)
        file(GLOB fileList "${${incDir}}/${folder}${dllName64}")
      endif()
  endif()  
  list(LENGTH fileList NUMLIST)
  if(NUMLIST EQUAL 0)    
     set (${targetVar} "NOTFOUND")    
  endif()
  list(APPEND ${targetVar} ${fileList} )  

  # message ( "File list: ${${targetVar}}" )		#-- debugging
endmacro()

 # Locate OptiX 

if ( NOT OPTIX_ROOT_DIR )
  # Default search
  if ( WIN32 )
     set (OPTIX_ROOT_DIR "c:/ProgramData/NVIDIA Corporation/OptiX SDK 4.0.0/" CACHE PATH "Optix Location" FORCE) 
     set (OPTIX_VERSION "4.0.0" )
  else ()
     set (OPTIX_ROOT_DIR "/usr/local/optix" CACHE PATH "" FORCE)
     set (OPTIX_VERSION "4.0.0" )
  endif()
endif()

#-------- Locate Libs
set ( OPTIX_BIN_DIR "${OPTIX_ROOT_DIR}/bin64" CACHE PATH "Optix Binary path" FORCE)
set ( OPTIX_LIB_DIR "${OPTIX_ROOT_DIR}/lib64" CACHE PATH "Optix Library path" FORCE)
set ( OPTIX_INCLUDE_DIR "${OPTIX_ROOT_DIR}/include" CACHE PATH "Optix Header path" FORCE )
set ( OPTIX_FOUND "YES" )

message ( STATUS "  OptiX Lib:     ${OPTIX_LIB_DIR}" )
message ( STATUS "  OptiX Bin:     ${OPTIX_BIN_DIR}" )
message ( STATUS "  OptiX Include: ${OPTIX_INCLUDE_DIR}" )

set ( OK_DLL "0" )
_FIND_FILE ( OPTIX_DLL OPTIX_BIN_DIR "optix.1.dll" "" OK_DLL )
_FIND_FILE ( OPTIX_DLL OPTIX_BIN_DIR "optixu.1.dll" "" OK_DLL )
if( NOT OK_DLL EQUAL 2 )	
   message ( INFO "  OptiX NOT FOUND. Could not find OptiX dll/so" )
   set ( OPTIX_FOUND "NO" )
endif()

set ( OK_LIB "0" )
_FIND_FILE ( OPTIX_LIB1 OPTIX_LIB_DIR "optix.lib" "liboptix.so" OK_LIB )
_FIND_FILE ( OPTIX_LIB2 OPTIX_LIB_DIR "optixu.lib" "liboptixu.so" OK_LIB )
if( NOT OK_LIB EQUAL 2 )	
   message ( INFO "  OptiX NOT FOUND. Could not find OptiX lib" )
   set ( OPTIX_FOUND "NO" )
endif()

set ( OK_H "0" )
_FIND_FILE ( OPTIX_HEADERS OPTIX_INCLUDE_DIR "optix.h" "optix.h" OK_H )
if( NOT OK_H EQUAL 1 )	
   message ( INFO "  OptiX NOT FOUND. Could not find OptiX headers" )
   set ( OPTIX_FOUND "NO" )
endif()

if ( OPTIX_FOUND STREQUAL "NO" )
  message(FATAL_ERROR "
      OPTIX not found. Please specify OPTIX_ROOT_DIR
      containing the include, lib64, bin64 folders. 
      Not found at OPTIX_ROOT_DIR: ${OPTIX_ROOT_DIR}\n"
  )
else()
  message ( STATUS "  OptiX Location: ${OPTIX_ROOT_DIR}" )
endif()

message ( STATUS "  OptiX Libs:    ${OPTIX_LIB1}, ${OPTIX_LIB2}" )
message ( STATUS "  OptiX Bins:    ${OPTIX_DLL}" )

SET(OPTIX_DLL ${OPTIX_DLL} CACHE STRING "" FORCE)
SET(OPTIX_LIB1 ${OPTIX_LIB1} CACHE STRING "" FORCE)
SET(OPTIX_LIB2 ${OPTIX_LIB2} CACHE STRING "" FORCE)

mark_as_advanced( OPTIX_FOUND )


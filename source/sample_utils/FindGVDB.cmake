
#####################################################################################
# Find GVDB
#
unset(GVDB_FOUND CACHE)
unset(GVDB_INCLUDE_DIR CACHE)

if ( NOT DEFINED GVDB_ROOT_DIR )
  if (WIN32)
    get_filename_component ( BASEDIR "${CMAKE_MODULE_PATH}/../../_output" REALPATH )
  else()
    get_filename_component ( BASEDIR "/usr/local/gvdb/" REALPATH )
  endif()
  set ( GVDB_ROOT_DIR ${BASEDIR} CACHE PATH "Location of GVDB library" FORCE)
endif()
message ( STATUS "Searching for GVDB at.. ${GVDB_ROOT_DIR}")
set( GVDB_FOUND "YES" )

if ( GVDB_ROOT_DIR )

    #-- Paths to GVDB Library (cached so user can modify)
	set ( GVDB_INCLUDE_DIR "${GVDB_ROOT_DIR}/include" CACHE PATH "Path to include files" FORCE)
	set ( GVDB_LIB_DIR "${GVDB_ROOT_DIR}/lib" CACHE PATH "Path to libraries" FORCE)	
	set ( GVDB_SHARE_DIR "${GVDB_ROOT_DIR}/lib" CACHE PATH "Path to share files" FORCE)	

	#-------- Locate Header files
        set ( OK_H "0" )
	_FIND_FILE ( GVDB_HEADERS GVDB_INCLUDE_DIR "gvdb.h" "gvdb.h" OK_H )
	if ( OK_H EQUAL 1 ) 
	    message ( STATUS "  Found. GVDB Header files. ${GVDB_INCLUDE_DIR}" )
	else()
	    message ( "  NOT FOUND. GVDB Header files" )
		set ( GVDB_FOUND "NO" )
	endif ()

    #-------- Locate Library	
        set ( OK_DLL 0 )	
        set ( OK_LIB 0 )	
	_FIND_FILE ( LIST_DLL GVDB_LIB_DIR "libgvdb.dll" "libgvdb.so" OK_DLL )	
  	_FIND_FILE ( LIST_LIB GVDB_LIB_DIR "libgvdb.lib" "libgvdb.so" OK_LIB )
	if ( (${OK_DLL} EQUAL 1) AND (${OK_LIB} EQUAL 1) ) 
	   message ( STATUS "  Found. GVDB Library. ${GVDB_LIB_DIR}" )	   
	else()
	   set ( GVDB_FOUND "NO" )	   
	   message ( "  NOT FOUND. GVDB Library. (so/dll or lib missing)" )	   
	endif()

	if ( NOT DEFINED WIN32 )
           set ( OK_CUDPP 0 )
     	   _FIND_FILE ( LIST_CUDPP GVDB_LIB_DIR "" "libcudpp.so" OK_CUDPP )	        
    	   _FIND_FILE ( LIST_CUDPP GVDB_LIB_DIR "" "libcudpp_hash.so" OK_CUDPP )	     
        endif()

	#-------- Locate PTX/GLSL
        set ( OK_PTX 0 )	
        set ( OK_GLSL 0 )	
	_FIND_MULTIPLE( LIST_PTX GVDB_SHARE_DIR "ptx" "ptx" OK_PTX )  
         _FIND_MULTIPLE( LIST_GLSL GVDB_SHARE_DIR "glsl" "glsl" OK_GLSL )    
	if ( (${OK_PTX} EQUAL 2) AND (${OK_GLSL} GREATER 2) ) 
	   message ( STATUS "  Found. GVDB Ptx/Glsl. ${GVDB_SHARE_DIR}" )	   
	else()
	   set ( GVDB_FOUND "NO" )	   
	   message ( "  NOT FOUND. GVDB Ptx/Glsl. (ptx or glsl missing)" )	   	   
	endif()	
	
	#-------- Locate Support DLLs
        set ( OK_EXTRA "0" )		
	_FIND_MULTIPLE ( LIST_EXTRA GVDB_LIB_DIR "dll" "so" OK_EXTRA )	

endif()
 
if ( ${GVDB_FOUND} STREQUAL "NO" )
   message( FATAL_ERROR "
      Please set GVDB_ROOT_DIR to the root location 
      of installed GVDB library containing /include and /lib.
      Not found at GVDB_ROOT_DIR: ${GVDB_ROOT_DIR}\n"
   )
endif()

list ( APPEND LIST_LIB ${LIST_CUDPP})

set ( GVDB_DLL ${LIST_DLL} CACHE INTERNAL "" FORCE)
set ( GVDB_LIB ${LIST_LIB} CACHE INTERNAL "" FORCE)
set ( GVDB_PTX ${LIST_PTX} CACHE INTERNAL "" FORCE)
set ( GVDB_GLSL ${LIST_GLSL} CACHE INTERNAL "" FORCE)
set ( GVDB_EXTRA ${LIST_EXTRA} CACHE INTERNAL "" FORCE)

#-- Create a list of all binary files needed for exes
unset ( LIST_FULL )	
LIST ( APPEND LIST_FULL ${LIST_DLL} )
LIST ( APPEND LIST_FULL ${LIST_PTX} )	
LIST ( APPEND LIST_FULL ${LIST_GLSL} )	
LIST ( APPEND LIST_FULL ${LIST_EXTRA} )	
LIST ( APPEND LIST_FULL ${LIST_CUDPP} )	
set ( GVDB_LIST ${LIST_FULL} CACHE INTERNAL "" )

#-- We do not want user to modified these vars, but helpful to show them
message ( STATUS "  GVDB_ROOT_DIR: ${GVDB_ROOT_DIR}" )
message ( STATUS "  GVDB_DLL:  ${GVDB_DLL}" )
message ( STATUS "  GVDB_LIB:  ${GVDB_LIB}" )
message ( STATUS "  GVDB_PTX:  ${GVDB_PTX}" )
message ( STATUS "  GVDB_GLSL: ${GVDB_GLSL}" )
message ( STATUS "  GVDB_EXTRA:${GVDB_EXTRA}" )

mark_as_advanced(GVDB_FOUND)







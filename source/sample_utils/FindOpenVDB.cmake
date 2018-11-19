
# Try to find OpenVDB project dll/so and headers
#

# outputs
unset(OPENVDB_DLL CACHE)
unset(OPENVDB_LIB CACHE)
unset(OPENVDB_FOUND CACHE)
unset(OPENVDB_INCLUDE_DIR CACHE)
unset(OPENVDB_LIB_DIR CACHE)
unset(OPENVDB_LIB_DEBUG CACHE)
unset(OPENVDB_LIB_RELEASE CACHE)

macro(_find_files targetVar incDir dllName )
  unset ( fileList )
  file(GLOB fileList "${${incDir}}/${dllName}")
  list(LENGTH fileList NUMLIST)
  message ( STATUS "locate: ${${incDir}}/${dllName}, found: ${NUMLIST}" )  
  if(NUMLIST EQUAL 0)
    message(FATAL_ERROR "MISSING: unable to find ${targetVar} files (${folder}${dllName}, ${folder}${dllName64})" )    
  else()
    list(APPEND ${targetVar} ${fileList} )  
  endif()  

  # message ( "File list: ${${targetVar}}" )		#-- debugging
endmacro()

FUNCTION(package_openvdb_binaries)
   list ( APPEND OPENVDB_LIST "blosc.dll")
   list ( APPEND OPENVDB_LIST "boost_system-vc140-mt-1_64.dll")
   list ( APPEND OPENVDB_LIST "boost_system-vc140-mt-gd-1_64.dll")
   list ( APPEND OPENVDB_LIST "boost_thread-vc140-mt-1_64.dll")
   list ( APPEND OPENVDB_LIST "boost_thread-vc140-mt-gd-1_64.dll")
   list ( APPEND OPENVDB_LIST "Half.dll")
   list ( APPEND OPENVDB_LIST "Iex.dll")
   list ( APPEND OPENVDB_LIST "IexMath.dll")
   list ( APPEND OPENVDB_LIST "IlmImf.dll")
   list ( APPEND OPENVDB_LIST "IlmThread.dll")
   list ( APPEND OPENVDB_LIST "Imath.dll")
   list ( APPEND OPENVDB_LIST "openvdb.dll")
   list ( APPEND OPENVDB_LIST "openvdb_d.dll")
   list ( APPEND OPENVDB_LIST "tbb.dll")
   list ( APPEND OPENVDB_LIST "tbb_debug.dll")
   list ( APPEND OPENVDB_LIST "zlib.dll")
   list ( APPEND OPENVDB_LIST "zlibd.dll")
   set ( OPENVDB_BINARIES ${OPENVDB_LIST} PARENT_SCOPE)      
ENDFUNCTION()

if (OPENVDB_ROOT_DIR AND USE_OPENVDB)

  if (WIN32) 	 
    set ( OPENVDB_LIB_DIR "${OPENVDB_ROOT_DIR}/lib64" CACHE PATH "path" )
	set ( OPENVDB_INCLUDE_DIR "${OPENVDB_ROOT_DIR}/include" CACHE PATH "path" )
	#-- get_filename_component ( LIB_PATH "${OPENVDB_LIB_RELEASE}" DIRECTORY ) 

    #-------- Locate DLL
	_find_files( OPENVDB_DLL OPENVDB_LIB_DIR "Blosc.dll" )   
	_find_files( OPENVDB_DLL OPENVDB_LIB_DIR "Half.dll" )    
	_find_files( OPENVDB_DLL OPENVDB_LIB_DIR "Iex.dll" )    
	_find_files( OPENVDB_DLL OPENVDB_LIB_DIR "IexMath.dll" )    
	_find_files( OPENVDB_DLL OPENVDB_LIB_DIR "IlmThread.dll" )    
	_find_files( OPENVDB_DLL OPENVDB_LIB_DIR "Imath.dll" )    	
	_find_files( OPENVDB_DLL OPENVDB_LIB_DIR "tbb.dll" )  
	_find_files( OPENVDB_DLL OPENVDB_LIB_DIR "tbb_debug.dll" )  

	#-------- Locate LIBS
    _find_files( OPENVDB_LIB_DEBUG OPENVDB_LIB_DIR "openvdb_d.lib" )    
	_find_files( OPENVDB_LIB_RELEASE OPENVDB_LIB_DIR "openvdb.lib" )	
	
  endif(WIN32)

  if (UNIX)
    _find_files( OPENVDB_LIB OPENVDB_ROOT_DIR "libopenvdb.so" ) 
	set(OPENVDB_DLL ${OPENVDB_LIB})
  endif(UNIX)

  #-------- Locate HEADERS
  _find_files( OPENVDB_HEADERS OPENVDB_INCLUDE_DIR "Openvdb/openvdb.h" )

  if (OPENVDB_DLL)
	  set( OPENVDB_FOUND "YES" )      
	  
  endif()
  message(STATUS "--> Found OpenVDB at ${OPENVDB_ROOT_DIR}." )
else()

  set ( OPENVDB_ROOT_DIR "SET-OPENVDB_ROOT_DIR" CACHE PATH "")
  if ( USE_OPENVDB) 
  
      message(STATUS "--> WARNING: OPENVDB not found. Some samples requiring OpenVDB may not run.
        Set OPENVDB_ROOT_DIR to full path of library, and enable the USE_OPENVDB option." )
  else()
      message(STATUS "--> NOTE: OPENVDB is not enabled. Enable with the USE_OPENVDB option." )
  endif()

endif()

include(FindPackageHandleStandardArgs)

SET(OPENVDB_DLL ${OPENVDB_DLL} CACHE PATH "path")
SET(OPENVDB_LIB_DEBUG ${OPENVDB_LIB_DEBUG} CACHE PATH "path")
SET(OPENVDB_LIB_RELEASE ${OPENVDB_LIB_RELEASE} CACHE PATH "path")

mark_as_advanced( OPENVDB_FOUND )


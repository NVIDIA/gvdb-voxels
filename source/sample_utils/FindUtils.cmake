

#####################################################################################
# Find Sample Utils
#
unset( UTILS_FOUND CACHE)

#------ Locate LoadPNG
#
set ( SAMPLE_UTIL_DIR "${CMAKE_MODULE_PATH}" CACHE PATH "Path to sample_utils" )

set ( OK_UTIL "0" )
_FIND_FILE ( UTIL_FILES SAMPLE_UTIL_DIR "nv_gui.h" "nv_gui.h" OK_UTIL )

if ( OK_UTIL EQUAL 1 )
    set ( UTILS_FOUND "YES" )
    message ( STATUS "--> Using sample utils. ${SAMPLE_UTIL_DIR}") 
	include_directories( ${SAMPLE_UTIL_DIR} )		
	if ( REQUIRE_PNG )				
		LIST( APPEND UTIL_SOURCE_FILES "${SAMPLE_UTIL_DIR}/file_png.cpp" )
		LIST( APPEND UTIL_SOURCE_FILES "${SAMPLE_UTIL_DIR}/file_png.h" )
	endif()
	if ( REQUIRE_TGA )
		LIST( APPEND UTIL_SOURCE_FILES "${SAMPLE_UTIL_DIR}/file_tga.cpp" )	
	endif()
	if ( REQUIRE_GLEW )
		LIST( APPEND UTIL_SOURCE_FILES "${SAMPLE_UTIL_DIR}/glew.c" )	
	endif()
	if ( REQUIRE_MAIN )
		IF(WIN32)
  			LIST( APPEND UTIL_SOURCE_FILES "${SAMPLE_UTIL_DIR}/main_win.cpp" )
	    ELSE()
	  		LIST( APPEND UTIL_SOURCE_FILES "${SAMPLE_UTIL_DIR}/main_x11.cpp" )
		ENDIF()
		LIST( APPEND UTIL_SOURCE_FILES "${SAMPLE_UTIL_DIR}/main.h" )
	endif()
	if ( REQUIRE_NVGUI )
		LIST( APPEND UTIL_SOURCE_FILES "${SAMPLE_UTIL_DIR}/nv_gui.cpp" )	
		LIST( APPEND UTIL_SOURCE_FILES "${SAMPLE_UTIL_DIR}/nv_gui.h" )				
	endif()
	if ( REQUIRE_CAM )
		LIST( APPEND UTIL_SOURCE_FILES "${SAMPLE_UTIL_DIR}/camera.cpp" )	
		LIST( APPEND UTIL_SOURCE_FILES "${SAMPLE_UTIL_DIR}/camera.h" )				
	endif()
	if ( REQUIRE_VEC )
		LIST( APPEND UTIL_SOURCE_FILES "${SAMPLE_UTIL_DIR}/vec.cpp" )	
		LIST( APPEND UTIL_SOURCE_FILES "${SAMPLE_UTIL_DIR}/vec.h" )				
	endif()
else()
    set ( UTILS_FOUND "NO" )	
	message ( FATAL_ERROR "
Sample Utils not found. Please set the CMAKE_MODULE_PATH  \
and the SAMPLE_UTIL_DIR to the location of sample_utils path,  \
which contains nv_gui, file_png, file_tga, main_win, main_x11, \
and cmake helpers. \n
     ") 
endif()

mark_as_advanced( UTILS_FOUND )

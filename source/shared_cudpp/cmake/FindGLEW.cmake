#
# Try to find GLEW library and include path.
# Once done this will define
#
# GLEW_FOUND
# GLEW_INCLUDE_PATH
# GLEW_LIBRARY
# 
IF (WIN32)
    FIND_PATH( GLEW_INCLUDE_PATH GL/glew.h
        ${GLEW_ROOT_DIR}/include
        DOC "The directory where GL/glew.h resides")
    if (CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(GLEWNAMES glew GLEW glew64 glew64s)
    else ()
        set(GLEWNAMES glew GLEW glew32 glew32s)
    endif (CMAKE_SIZEOF_VOID_P EQUAL 8)  
    
    FIND_LIBRARY( GLEW_LIBRARY
        NAMES ${GLEWNAMES}
	PATHS
        ${GLEW_ROOT_DIR}/bin
        ${GLEW_ROOT_DIR}/lib
        DOC "The GLEW library")
ELSE (WIN32)
  FIND_PATH( GLEW_INCLUDE_PATH GL/glew.h
		/usr/include
		/usr/local/include
		/sw/include
		/opt/local/include
    ${GLEW_ROOT_DIR}/include
		DOC "The directory where GL/glew.h resides")
	FIND_LIBRARY( GLEW_LIBRARY
		NAMES GLEW libGLEW
		PATHS
		/usr/lib64
		/usr/lib
    /usr/local/lib64
		/usr/local/lib
		/sw/lib
		/opt/local/lib
        ${GLEW_ROOT_DIR}/lib
		DOC "The GLEW library")
ENDIF (WIN32)

IF (GLEW_INCLUDE_PATH AND GLEW_LIBRARY)
  SET( FOUND_GLEW 1)
ELSE (GLEW_INCLUDE_PATH AND GLEW_LIBRARY)
  SET( FOUND_GLEW 0)
ENDIF (GLEW_INCLUDE_PATH AND GLEW_LIBRARY)

MARK_AS_ADVANCED( FOUND_GLEW )
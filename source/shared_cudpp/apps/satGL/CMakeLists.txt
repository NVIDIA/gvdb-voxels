###############################################################################
#
# Build script for project
#
###############################################################################

set(CCFILES
  main.cpp
  )

set(CUFILES
  satGL.cu
  )

source_group("CUDA Source Files" FILES ${CUFILES})
  
# to help find libs...
set(CMAKE_PREFIX_PATH ../common)
  
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

if (NOT FOUND_GLEW)
  message(REPORT_ERROR " satGL requires the GLEW library (http://glew.sourceforge.net/)")
endif (NOT FOUND_GLEW)

include_directories(  
  ../common/include
  ${OPENGL_INCLUDE_DIR}
  ${GLUT_INCLUDE_DIR}
  ${GLEW_INCLUDE_DIR}
  )

cuda_add_executable(satGL ${CCFILES} ${CUFILES})

target_link_libraries(satGL
  cudpp
  ${OPENGL_LIBRARIES}
  ${GLUT_LIBRARIES}
  ${GLEW_LIBRARY}
  )

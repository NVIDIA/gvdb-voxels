cmake_minimum_required(VERSION 3.10)

# Here's what the dependency graph of this sample looks like:
#           gPointFusion (copies files)
#           |          |
#           V          V
# gPointFusionApp  gPointFusionPTX
#  (builds app)     (builds PTX)

set(PROJECT_NAME gPointFusion)
set(PROJECT_NAME_APP "${PROJECT_NAME}App")
set(PROJECT_NAME_PTX "${PROJECT_NAME}PTX")
# Make sure to include CUDA, so that GVDB header files are processed correctly, and C, so that GLEW compiles correctly:
project(${PROJECT_NAME} LANGUAGES CUDA CXX C)

# If the gvdb target hasn't yet been defined (for instance, if this is
# being called from a build_all script), import the GVDB project directly:
if(NOT TARGET gvdb)
    # One fast way to add GVDB as a build dependency of the project is to add its subdirectory as follows:
    add_subdirectory("../gvdb_library" "$CMAKE_CURRENT_BINARY_DIR}/../gvdb_library")
    # Alternatively, if GVDB has already been built, then we can import its GVDBConfig.cmake list file:
    # find_package(GVDB CONFIG REQUIRED
    #    PATHS "${CMAKE_CURRENT_BINARY_DIR}/../gvdb_library")
endif()


# Sample-specific requirements:
set ( REQUIRE_OPENGL "1" )
set ( REQUIRE_PNG "1" )
set ( REQUIRE_TGA "1" )
set ( REQUIRE_GLEW "1" )
set ( REQUIRE_MAIN "1" )
set ( REQUIRE_NVGUI "1" )
set ( REQUIRE_OPTIX "1" )

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/../sample_utils")
find_package(Utils REQUIRED)

# Find OptiX
find_package(Optix REQUIRED)


#---------------------------------------------------------------------PTX files
# Compile the fluid system's source files into a PTX object library.
add_library(${PROJECT_NAME_PTX}
    OBJECT
    ${UTIL_OPTIX_KERNELS})
# The library also needs to compile point_fusion_cuda.cu.
target_sources(${PROJECT_NAME_PTX}
    PRIVATE point_fusion_cuda.cu
            point_fusion_cuda.cuh)
# Make sure it compiles to PTX files.
set_target_properties(${PROJECT_NAME_PTX} PROPERTIES CUDA_PTX_COMPILATION ON)
# Enable the --use_fast_math CUDA compiler flag.
target_compile_options(${PROJECT_NAME_PTX} PRIVATE --use_fast_math)
# Include gvdbPTX's include directories:
target_include_directories(${PROJECT_NAME_PTX}
    PRIVATE $<TARGET_PROPERTY:gvdbPTX,INTERFACE_INCLUDE_DIRECTORIES>)
# It also needs to include GVDB's include directories because it uses gvdb_vec.h:
target_include_directories(${PROJECT_NAME_PTX}
    PRIVATE $<TARGET_PROPERTY:gvdb,INTERFACE_INCLUDE_DIRECTORIES>)
# Add OptiX's include file directory to the search paths:
target_include_directories(${PROJECT_NAME_PTX} PRIVATE "${OPTIX_ROOT_DIR}/include")

#------------------------------------------------------------------main project
# Add the executable and set its name and sources:
add_executable(${PROJECT_NAME_APP} main_point_fusion.cpp)
set_property(TARGET ${PROJECT_NAME_APP} PROPERTY OUTPUT_NAME ${PROJECT_NAME})
target_sources(${PROJECT_NAME_APP}
    PRIVATE cutil_math.h)

# Then add the utils and OptiX files and kernels to the list of source files to build:
target_sources(${PROJECT_NAME_APP}
    PRIVATE ${UTIL_SOURCE_FILES}
            ${UTIL_OPTIX_FILES})

# Tell NVGUI to use GVDB's vector and camera classes:
target_compile_definitions(${PROJECT_NAME_APP}
    PRIVATE USE_GVDB="1"
            BUILD_OPENGL="1") # Build with GVDB's OpenGL utilities

# Link the GVDB library into the build:
target_link_libraries(${PROJECT_NAME_APP} PUBLIC gvdb)
# Also add a dependency on gvdbCopy (this makes sure that files are copied correctly)
add_dependencies(${PROJECT_NAME_APP} gvdbCopy)
 
# Add OptiX's libraries to the build:
add_definitions(-DUSE_OPTIX)
add_definitions(-DBUILD_OPTIX)
target_link_libraries(${PROJECT_NAME_APP} PUBLIC ${OPTIX_LIB} ${OPTIX_LIBU})
# Add OptiX's include file directory to the search paths:
target_include_directories(${PROJECT_NAME_APP} PRIVATE "${OPTIX_ROOT_DIR}/include")

# Set asset path definition:
if(NOT DEFINED ASSET_PATH) 
  get_filename_component(_assets "${CMAKE_CURRENT_SOURCE_DIR}/../shared_assets" REALPATH)
  set(ASSET_PATH ${_assets} CACHE PATH "Full path to gvdb/shared_assets/")  
endif()
target_compile_definitions(${PROJECT_NAME_APP}
    PRIVATE ASSET_PATH="${ASSET_PATH}/")

#------------------------------------------------------------------shared files
# These are run every time the main project is built:
add_custom_target(${PROJECT_NAME} ALL DEPENDS copiedPTX.stamp copiedOptiX.stamp copiedGVDB.stamp)
# Tell Visual Studio how to start the application:
set_target_properties(${PROJECT_NAME} PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:${PROJECT_NAME_APP}>
    VS_DEBUGGER_COMMAND $<TARGET_FILE:${PROJECT_NAME_APP}>)
# Copy the PTX files to the executable directory (this is where the dependencies on the PTX files are introduced):
add_custom_command(OUTPUT copiedPTX.stamp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_OBJECTS:${PROJECT_NAME_PTX}> $<TARGET_FILE_DIR:${PROJECT_NAME_APP}>
    COMMAND ${CMAKE_COMMAND} -E touch copiedPTX.stamp
    DEPENDS ${PROJECT_NAME_PTX} ${PROJECT_NAME_APP}
    COMMAND_EXPAND_LISTS)
# Copy the OptiX DLLs to the executable directory:
add_custom_command(OUTPUT copiedOptiX.stamp
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${OPTIX_SHARED_LIBS} $<TARGET_FILE_DIR:${PROJECT_NAME_APP}>
    COMMAND ${CMAKE_COMMAND} -E touch copiedOptiX.stamp
    COMMAND_EXPAND_LISTS)
# Finally, copy the GVDB library itself to the executable directory:
add_custom_command(OUTPUT copiedGVDB.stamp
    COMMAND ${CMAKE_COMMAND} -E copy_directory $<TARGET_FILE_DIR:gvdb> $<TARGET_FILE_DIR:${PROJECT_NAME_APP}>
    COMMAND ${CMAKE_COMMAND} -E touch copiedGVDB.stamp)

#------------------------------------------------------------install (optional)
get_filename_component(BIN_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/bin REALPATH)
install(DIRECTORY "$<TARGET_FILE_DIR:${PROJECT_NAME_APP}>/" DESTINATION ${BIN_INSTALL_PATH} FILES_MATCHING PATTERN "*.dll" PATTERN "*.glsl" PATTERN "*.ptx" PATTERN "*.so" PATTERN "*.exe" REGEX "/[^.]+$")

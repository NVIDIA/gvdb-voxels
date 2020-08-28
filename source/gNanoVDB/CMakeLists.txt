cmake_minimum_required(VERSION 3.10)

# Make sure to include CUDA, so that GVDB header files are processed correctly, and C, so that GLEW compiles correctly.
set(PROJECT_NAME gNanoVDB)
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

# Add PTX files as a library
set(PROJECT_PTX_NAME "${PROJECT_NAME}PTX")
add_library(${PROJECT_PTX_NAME} OBJECT cuda_export_nanovdb.cu)
set_target_properties(${PROJECT_PTX_NAME} PROPERTIES CUDA_PTX_COMPILATION ON)
# Include gvdbPTX's include directories and make sure it's built beforehand
# (this is like using target_link_libraries without linking)
add_dependencies(${PROJECT_PTX_NAME} gvdbPTX)
target_include_directories(${PROJECT_PTX_NAME}
    PRIVATE $<TARGET_PROPERTY:gvdbPTX,INTERFACE_INCLUDE_DIRECTORIES>)

# Main project
add_executable(${PROJECT_NAME} main_nanovdb.cpp gvdb_export_nanovdb.cpp gvdb_export_nanovdb.h)
add_dependencies(${PROJECT_NAME} ${PROJECT_PTX_NAME})

# Link with GVDB:
target_link_libraries(${PROJECT_NAME} gvdb)

# Tell Visual Studio to set its debugger working directory to the executable directory:
set_property(TARGET ${PROJECT_NAME}
    PROPERTY VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:${PROJECT_NAME}>)


# Sample-specific requirements
set ( REQUIRE_OPENGL "1" )
set ( REQUIRE_PNG "1" )
set ( REQUIRE_TGA "0" )
set ( REQUIRE_GLEW "1" )
set ( REQUIRE_MAIN "0" )
set ( REQUIRE_NVGUI "0" )
# Tell NVGUI to use GVDB's vector and camera classes:
target_compile_definitions(${PROJECT_NAME}
    PRIVATE USE_GVDB="1"
            BUILD_OPENGL="1") # Build with GVDB's OpenGL utilities

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/../sample_utils")
find_package(Utils REQUIRED)
# Then add the utils to the list of source files to build:
target_sources(${PROJECT_NAME}
    PRIVATE ${UTIL_SOURCE_FILES})

# Set asset path definition
if ( NOT DEFINED ASSET_PATH ) 
  get_filename_component ( _assets "${CMAKE_CURRENT_SOURCE_DIR}/../shared_assets" REALPATH )
  set ( ASSET_PATH ${_assets} CACHE PATH "Full path to gvdb/shared_assets/" )  
endif()
target_compile_definitions(${PROJECT_NAME}
    PRIVATE ASSET_PATH="${ASSET_PATH}/")

# Finally, copy the GVDB library itself to the executable directory (this also adds
# a source-level dependency on gvdbCopy, which makes sure the files are copied correctly)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/copiedGVDB.stamp
    COMMAND ${CMAKE_COMMAND} -E copy_directory $<TARGET_FILE_DIR:gvdb> $<TARGET_FILE_DIR:${PROJECT_NAME}>
    COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/copiedGVDB.stamp
    DEPENDS gvdbCopy)
set_property(SOURCE main_nanovdb.cpp APPEND PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/copiedGVDB.stamp)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/copiedPTX.stamp
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_OBJECTS:${PROJECT_PTX_NAME}> $<TARGET_FILE_DIR:${PROJECT_NAME}>
    COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/copiedPTX.stamp
    COMMAND_EXPAND_LISTS)
set_property(SOURCE main_nanovdb.cpp APPEND PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/copiedPTX.stamp)

# Optionally, install the application into the BIN_INSTALL_PATH set by gvdb_library's CMakeLists.
get_filename_component(BIN_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/bin REALPATH)
install(DIRECTORY "$<TARGET_FILE_DIR:${PROJECT_NAME}>/" DESTINATION ${BIN_INSTALL_PATH} FILES_MATCHING PATTERN "*.dll" PATTERN "*.glsl" PATTERN "*.ptx" PATTERN "*.so" PATTERN "*.exe" REGEX "/[^.]+$")

#----- copy_cuda_bins
# This macro copies all binaries for the CUDA library to the target executale location. 
#
macro(_copy_cuda_bins projname )	
	add_custom_command(
		TARGET ${projname} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy ${CUDA_DLL} $<TARGET_FILE_DIR:${PROJNAME}>   
	)
endmacro()


#------------------------------------ CROSS-PLATFORM PTX COMPILE 
#
# _COMPILEPTX( SOURCES file1.cu file2.cu TARGET_PATH <path where ptxs should be stored> GENERATED_FILES ptx_sources NVCC_OPTIONS -arch=sm_20)
# Generates ptx files for the given source files. ptx_sources will contain the list of generated files.
#
FUNCTION( _COMPILEPTX )
  set(options "")
  set(oneValueArgs TARGET_PATH GENERATED GENPATHS INCLUDE)  
  set(multiValueArgs OPTIONS SOURCES)
  CMAKE_PARSE_ARGUMENTS( _COMPILEPTX "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  
  # Match the bitness of the ptx to the bitness of the application
  set( MACHINE "--machine=32" )
  if( CMAKE_SIZEOF_VOID_P EQUAL 8)
    set( MACHINE "--machine=64" )
  endif()
  unset ( PTX_FILES CACHE )
  unset ( PTX_FILES_PATH CACHE )

  set( USE_DEBUG_PTX OFF CACHE BOOL "Enable CUDA debugging with NSight")  
  if ( USE_DEBUG_PTX )
	 set ( DEBUG_FLAGS ";-g;-G;-D_DEBUG;-DEBUG")
  else()
	 set ( DEBUG_FLAGS "")
  endif()
  
  if ( WIN32 ) 
		# Windows - PTX compile
		file ( MAKE_DIRECTORY "${_COMPILEPTX_TARGET_PATH}/Debug" )
		file ( MAKE_DIRECTORY "${_COMPILEPTX_TARGET_PATH}/Release" )
		string (REPLACE ";" " " _COMPILEPTX_OPTIONS "${_COMPILEPTX_OPTIONS}")  
		separate_arguments( _OPTS WINDOWS_COMMAND "${_COMPILEPTX_OPTIONS}" )
		message ( STATUS "NVCC Options: ${_COMPILEPTX_OPTIONS}" )  
		message ( STATUS "NVCC Include: ${_COMPILEPTX_INCLUDE}" )

        set ( INCL "-I\"${_COMPILEPTX_INCLUDE}\"" )

		# Custom build rule to generate ptx files from cuda files
		FOREACH( input ${_COMPILEPTX_SOURCES} )
			get_filename_component( input_ext ${input} EXT )									# Input extension
			get_filename_component( input_without_ext ${input} NAME_WE )						# Input base
			if ( ${input_ext} STREQUAL ".cu" )			
				
				# Set output names
				set( output "${input_without_ext}.ptx" )							# Output name
				set( output_with_path "${_COMPILEPTX_TARGET_PATH}/$(Configuration)/${input_without_ext}.ptx" )	# Output with path
				set( output_with_quote "\"${output_with_path}\"" )
				LIST( APPEND PTX_FILES ${output} )		# Append to output list
				LIST( APPEND PTX_FILES_PATH ${output_with_path} )
    
				message( STATUS "NVCC Compile: ${CUDA_NVCC_EXECUTABLE} ${MACHINE} --ptx ${_COMPILEPTX_OPTIONS} ${input} ${INCL} -o ${output_with_path} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}")
    
				add_custom_command(
					OUTPUT  ${output_with_path}
					MAIN_DEPENDENCY ${input}
					COMMAND ${CUDA_NVCC_EXECUTABLE} ${MACHINE} --ptx ${_OPTS} ${input} ${INCL} -o ${output_with_quote} WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
				)			
			endif()
		ENDFOREACH( )
  else ()
		# Linux - PTX compile
		file ( MAKE_DIRECTORY "${_COMPILEPTX_TARGET_PATH}" )
    FOREACH(input ${_COMPILEPTX_SOURCES})
      get_filename_component( input_ext ${input} EXT )									# Input extension
      get_filename_component( input_without_ext ${input} NAME_WE )						# Input base
      if ( ${input_ext} STREQUAL ".cu" )			
        # Set output names
        set( output "${input_without_ext}.ptx" ) # Output name
        set( output_with_path "${_COMPILEPTX_TARGET_PATH}/${input_without_ext}.ptx" )	# Output with path

        set( compile_target_ptx "${input_without_ext}_PTX")
        set( custom_command_var "${input_without_ext}_OUTPUT")
        # compile ptx
        cuda_compile_ptx(custom_command_var ${input} OPTIONS "${DEBUG_FLAGS}")
        # This will only configure file generation, we need to add a target to
        # generate a file cuda_generated_<counter>_${input_without_ext}.ptx
        # Add custom command to rename to simply ${input_without_ext}.ptx
        add_custom_command(OUTPUT ${output_with_path}
                          COMMAND ${CMAKE_COMMAND} -E rename ${custom_command_var} ${output_with_path}
                          DEPENDS ${custom_command_var})
        add_custom_target(${compile_target_ptx} ALL DEPENDS ${input} ${output_with_path} SOURCES ${input})

        # Add this output file to list of generated ptx files
        LIST(APPEND PTX_FILES ${output})
        LIST(APPEND PTX_FILES_PATH ${output_with_path} )
      endif()
    ENDFOREACH()
  endif()

  set( ${_COMPILEPTX_GENERATED} ${PTX_FILES} PARENT_SCOPE)
  set( ${_COMPILEPTX_GENPATHS} ${PTX_FILES_PATH} PARENT_SCOPE)

ENDFUNCTION()

#------------------------------------ CROSS-PLATFORM INSTALL
function( _INSTALL )   
  set (options "")
  set (oneValueArgs DESTINATION SOURCE OUTPUT )
  set (multiValueArgs FILES )
  CMAKE_PARSE_ARGUMENTS(_INSTALL "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if ( _INSTALL_SOURCE ) 
     set ( _INSTALL_SOURCE "${_INSTALL_SOURCE}/" )	  
  endif()
  set ( OUT_LIST ${${_INSTALL_OUTPUT}} )

  if ( WIN32 )      
      # Windows - copy to desintation at post-build
      file ( MAKE_DIRECTORY "${_INSTALL_DESTINATION}/" )
      foreach (_file ${_INSTALL_FILES} )	     
          message ( STATUS "Install: ${_INSTALL_SOURCE}${_file} -> ${_INSTALL_DESTINATION}" )
          add_custom_command(
            TARGET ${PROJNAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${_INSTALL_SOURCE}${_file} ${_INSTALL_DESTINATION}
          )          
 	  list ( APPEND OUT_LIST "${_INSTALL_SOURCE}${_file}" )
      endforeach()    
  else ()
      # Linux 
      if ( _INSTALL_SOURCE )	   
	  foreach ( _file ${_INSTALL_FILES} )
             list ( APPEND OUT_LIST "${_INSTALL_SOURCE}${_file}" )
    	  endforeach()
      else()
	     list ( APPEND OUT_LIST ${_INSTALL_FILES} )
      endif() 
  endif( )
  set ( ${_INSTALL_OUTPUT} ${OUT_LIST} PARENT_SCOPE )
   
endfunction()

#------------------------------------------------- CROSS-PLATFORM INSTALL PTX
#
function( _INSTALL_PTX )   
  set (options "")
  set (oneValueArgs DESTINATION OUTPUT )
  set (multiValueArgs FILES )
  CMAKE_PARSE_ARGUMENTS(_INSTALL_PTX "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set ( OUT_LIST ${${_INSTALL_PTX_OUTPUT}} )

  unset ( PTX_FIXED )

  if ( WIN32 )   

    foreach ( _file IN ITEMS ${_INSTALL_PTX_FILES} )
	get_filename_component ( _ptxbase ${_file} NAME_WE )
 	get_filename_component ( _ptxpath ${_file} DIRECTORY )
 	get_filename_component ( _ptxparent ${_ptxpath} DIRECTORY )   # parent directory
	set ( _fixed "${_ptxparent}/${_ptxbase}.ptx" )                # copy to parent to remove compile time $(Configuration) path
  	add_custom_command ( TARGET ${PROJNAME} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy  ${_file} ${_fixed}
        )
	list ( APPEND PTX_FIXED ${_file} )     # NOTE: Input of FILES must be list of ptx *with paths*	
	list ( APPEND OUT_LIST ${_fixed} )
    endforeach()

  else()

    foreach ( _file IN ITEMS ${_INSTALL_PTX_FILES} )
      get_filename_component ( _ptxpath ${_file} DIRECTORY )
      get_filename_component ( _ptxbase ${_file} NAME_WE )
      get_filename_component ( _ptxparent ${_ptxpath} DIRECTORY )    # parent directory
      set ( _fixed "${_file}" )
      list ( APPEND PTX_FIXED ${_fixed} )
      list ( APPEND OUT_LIST ${_fixed} )
    endforeach()
  endif()
  
  # Install PTX
  message ( STATUS "PTX files: ${PTX_FIXED}" )
  _INSTALL ( FILES ${PTX_FIXED} DESTINATION ${_INSTALL_PTX_DESTINATION} )

  set ( ${_INSTALL_PTX_OUTPUT} ${OUT_LIST} PARENT_SCOPE )

endfunction()


#----------------------------------------------- CROSS-PLATFORM FIND FILES
# Find one or more of a specific file in the given folder
# Returns the file name w/o path

macro(_FIND_FILE targetVar searchDir nameWin64 nameLnx cnt)
  unset ( fileList )  
  unset ( nameFind )
  unset ( targetVar )  
  if ( WIN32 ) 
     SET ( nameFind ${nameWin64} )
  else()
     SET ( nameFind ${nameLnx} )
  endif()
  if ( "${nameFind}" STREQUAL ""  )
    MATH (EXPR ${cnt} "${${cnt}}+1" )	
  else()
    file(GLOB fileList "${${searchDir}}/${nameFind}")  
    list(LENGTH fileList NUMLIST)  
    if (NUMLIST GREATER 0)	
       MATH (EXPR ${cnt} "${${cnt}}+1" )	
       list(APPEND ${targetVar} ${nameFind} )
    endif() 
  endif()
endmacro()

#----------------------------------------------- CROSS-PLATFORM FIND MULTIPLE
# Find all files in specified folder with the given extension.
# This creates a file list, where each entry is only the filename w/o path
# Return the count of files
macro(_FIND_MULTIPLE targetVar searchDir extWin64 extLnx cnt)    
  unset ( fileList )    
  unset ( targetVar ) 
  unset ( ${cnt} )
  set ( ${cnt} "0" )
  if ( WIN32 ) 
     SET ( extFind ${extWin64} )
  else()
     SET ( extFind ${extLnx} )
  endif()
  file( GLOB fileList "${${searchDir}}/*.${extFind}")  
  list( LENGTH fileList NUMLIST)
  math( EXPR ${cnt} "${${cnt}}+${NUMLIST}" )  
  foreach ( _file ${fileList} )
     get_filename_component ( fname ${_file} NAME )
     list( APPEND ${targetVar} ${fname} )
  endforeach()
endmacro()

#----------------------------------------------- LIST ALL source
function(_LIST_ALL_SOURCE )   
  set (options "")
  set (oneValueArgs "" )
  set (multiValueArgs FILES )
  CMAKE_PARSE_ARGUMENTS(_LIST_ALL_SOURCE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  unset ( SOURCE_LIST )
  foreach ( _file IN ITEMS ${_LIST_ALL_SOURCE_FILES} )
     message ( STATUS "Source: ${_file}")			# uncomment to check source files
     list ( APPEND SOURCE_LIST ${_file} )
  endforeach()

  set ( ALL_SOURCE_FILES ${SOURCE_LIST} PARENT_SCOPE )
endfunction()

function(_LINK ) 
    set (options "")
    set (multiValueArgs PROJECT OPT DEBUG PLATFORM )
    CMAKE_PARSE_ARGUMENTS(_LINK "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

	set_property(GLOBAL PROPERTY DEBUG_CONFIGURATIONS Debug PARENT_SCOPE ) 

	set (PROJ_NAME ${_LINK_PROJECT})

	foreach (loop_var IN ITEMS ${_LINK_PLATFORM} )
		target_link_libraries( ${PROJ_NAME} general ${loop_var} )	
		list (APPEND LIBLIST ${loop_var})
	endforeach() 	

	foreach (loop_var IN ITEMS ${_LINK_DEBUG} )
		target_link_libraries ( ${PROJ_NAME} debug ${loop_var} )
		list (APPEND LIBLIST ${loop_var})
	endforeach()
	
	foreach (loop_var IN ITEMS ${_LINK_OPT} )   
		target_link_libraries ( ${PROJ_NAME} optimized ${loop_var} )
	endforeach()
	
	string (REPLACE ";" "\n   " OUTSTR "${LIBLIST}")
	message ( STATUS "Libraries used:\n   ${OUTSTR}" )
endfunction()

macro(_MSVC_PROPERTIES)
	# Instruct CMake to automatically build INSTALL project in Visual Studio 
	set(CMAKE_VS_INCLUDE_INSTALL_TO_DEFAULT_BUILD 1)

	set_target_properties( ${PROJNAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH} )
    set_target_properties( ${PROJNAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${EXECUTABLE_OUTPUT_PATH} )
    set_target_properties( ${PROJNAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${EXECUTABLE_OUTPUT_PATH} )    	

	# Set startup PROJECT	
	if ( (${CMAKE_MAJOR_VERSION} EQUAL 3 AND ${CMAKE_MINOR_VERSION} GREATER 5) OR (${CMAKE_MAJOR_VERSION} GREATER 3) )
		message ( STATUS "VS Startup Project: ${CMAKE_CURRENT_BINARY_DIR}, ${PROJNAME}")
		set_property ( DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJNAME} )
	endif()		

	# Source groups
	source_group(CUDA FILES ${CUDA_FILES})
    source_group(PTX FILES  ${PTX_FILES})
	source_group(OPTIX FILES ${UTIL_OPTIX_KERNELS})
endmacro ()

macro(_DEFAULT_INSTALL_PATH)
	if ( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
	   if (WIN32)
		  get_filename_component ( _instpath "${CMAKE_CURRENT_SOURCE_DIR}/../../_output" REALPATH )
	   else()
		  get_filename_component ( _instpath "/usr/local/gvdb/_output" REALPATH )
	   endif()
	   set ( CMAKE_INSTALL_PREFIX ${_instpath} CACHE PATH "default install path" FORCE)   
	endif()
	get_filename_component( BIN_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/bin REALPATH)
endmacro()

function(REMOVE_DUPES ARG_STR OUTPUT)
  set(ARG_LIST ${ARG_STR})
  separate_arguments(ARG_LIST)
  list(REMOVE_DUPLICATES ARG_LIST)
  string (REGEX REPLACE "([^\\]|^);" "\\1 " _TMP_STR "${ARG_LIST}")
  string (REGEX REPLACE "[\\](.)" "\\1" _TMP_STR "${_TMP_STR}") #fixes escaping
  set (${OUTPUT} "${_TMP_STR}" PARENT_SCOPE)
endfunction()


################################################################################################
# A function for automatic detection of GPUs installed  (if autodetection is enabled)
# Usage:
#   detect_installed_gpus(out_variable)
function(detect_installed_gpus out_variable)
  if(NOT CUDA_gpu_detect_output)
    set(__cufile ${PROJECT_BINARY_DIR}/detect_cuda_archs.cu)

    file(WRITE ${__cufile} ""
      "#include <cstdio>\n"
      "int main()\n"
      "{\n"
      "  int count = 0;\n"
      "  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
      "  if (count == 0) return -1;\n"
      "  for (int device = 0; device < count; ++device)\n"
      "  {\n"
      "    cudaDeviceProp prop;\n"
      "    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))\n"
      "      std::printf(\"%d.%d \", prop.major, prop.minor);\n"
      "  }\n"
      "  return 0;\n"
      "}\n")

    execute_process(COMMAND "${CUDA_NVCC_EXECUTABLE}" "--run" "${__cufile}"
      WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
      RESULT_VARIABLE __nvcc_res OUTPUT_VARIABLE __nvcc_out
      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(__nvcc_res EQUAL 0)
      string(REPLACE "2.1" "2.1(2.0)" __nvcc_out "${__nvcc_out}")
      message("BEFORE DUPE ${__nvcc_out}!!")
      REMOVE_DUPES(${__nvcc_out} dedup)
      message("AFTER DUPE ${dedup}")
      set(CUDA_gpu_detect_output ${dedup} CACHE INTERNAL "Returned GPU architectures from detect_gpus tool" FORCE)
    endif()
  endif()

  if(NOT CUDA_gpu_detect_output)
    message(STATUS "Automatic GPU detection failed. Building for all known architectures.")
    set(${out_variable} ${CUDA_known_gpu_archs} PARENT_SCOPE)
  else()
    set(${out_variable} ${CUDA_gpu_detect_output} PARENT_SCOPE)
  endif()
endfunction()


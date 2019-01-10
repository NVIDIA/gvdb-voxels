// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// -------------------------------------------------------------
#include <cuda_runtime_api.h>   // for cudaDeviceProp
#include "cudpp_testrig_options.h"

#define CUDPP_APP_COMMON_IMPL
#include "commandline.h"

extern cudaDeviceProp devProps;

using namespace cudpp_app;

/**
 * Sets "global" options in testOptions given command line
 *  -debug: sets bool <var>debug</var>. Usage is application-dependent.
 *  -op=OP: sets char * <var>op</var> to OP
 *  -iterations=#: sets int <var>numIterations</var> to #
 *  -skiplongtests: set to skip long tests that might otherwise trigger a
 *   watchdog timer
 *  -dir=<path>: sets the search path for cudppRand test inputs
 */
void setOptions(int argc, const char **argv, testrigOptions &testOptions)
{
    testOptions.debug = false;
    commandLineArg(testOptions.debug, argc, argv, "debug");

    if (checkCommandLineFlag(argc, argv, "multiscan"))
        testOptions.algorithm = "multiscan";
    else if (checkCommandLineFlag(argc, argv, "scan"))
        testOptions.algorithm = "scan";
    else if (checkCommandLineFlag(argc, argv, "segscan"))
        testOptions.algorithm = "segscan";
    else if (checkCommandLineFlag(argc, argv, "compact"))
        testOptions.algorithm = "compact";
    else if (checkCommandLineFlag(argc, argv, "sort"))
        testOptions.algorithm = "sort";
    else if (checkCommandLineFlag(argc, argv, "reduce"))
        testOptions.algorithm = "reduce";
    else if (checkCommandLineFlag(argc, argv, "spmv"))
        testOptions.algorithm = "spmv";
    else if (checkCommandLineFlag(argc, argv, "rand"))
        testOptions.algorithm = "rand";
    else if (checkCommandLineFlag(argc, argv, "tridiagonal"))
        testOptions.algorithm = "tridiagonal";
    else if (checkCommandLineFlag(argc, argv, "mtf"))
        testOptions.algorithm = "mtf";
    else if (checkCommandLineFlag(argc, argv, "bwt"))
        testOptions.algorithm = "bwt";
    else if (checkCommandLineFlag(argc, argv, "compress"))
        testOptions.algorithm = "compress";
    else if(checkCommandLineFlag(argc, argv, "sa"))
        testOptions.algorithm = "sa";
    else if(checkCommandLineFlag(argc, argv, "multisplit"))
        testOptions.algorithm = "multisplit";

    testOptions.op = "sum";
    commandLineArg(testOptions.op, argc, argv, "op");

    testOptions.numIterations = numTestIterations;
    commandLineArg(testOptions.numIterations, argc, argv, "iterations");

    /* currently: skiplongtests if GPU has a timeout and <= 2 SMs */
    testOptions.skiplongtests =
        checkCommandLineFlag(argc, argv, "skiplongtests") ||
        (devProps.kernelExecTimeoutEnabled &&
         (devProps.multiProcessorCount <= 2));

    testOptions.dir = "";
    commandLineArg(testOptions.dir, argc, argv, "dir");
}

bool hasOptions(int argc, const char**argv)
{
    std::string temp;
    if (commandLineArg(temp, argc, argv, "op") ||
        checkCommandLineFlag(argc, argv, "float") ||
        checkCommandLineFlag(argc, argv, "int") ||
        checkCommandLineFlag(argc, argv, "uint") ||
        checkCommandLineFlag(argc, argv, "double") ||
        checkCommandLineFlag(argc, argv, "longlong") ||
        checkCommandLineFlag(argc, argv, "ulonglong") ||
        checkCommandLineFlag(argc, argv, "char") ||
        checkCommandLineFlag(argc, argv, "uchar") ||
        checkCommandLineFlag(argc, argv, "backward") ||
        checkCommandLineFlag(argc, argv, "forward") ||
        checkCommandLineFlag(argc, argv, "inclusive") ||
        checkCommandLineFlag(argc, argv, "exclusive") ||
        checkCommandLineFlag(argc, argv, "keysonly"))
        return true;
    return false;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

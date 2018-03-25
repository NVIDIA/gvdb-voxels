// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: $
// $Date: $
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// -------------------------------------------------------------

/**
 * @file
 * test_compact.cu
 *
 * @brief Host testrig routines to exercise cudpp's compact functionality.
 */

#include <stdio.h>
#include <time.h>
#include <limits.h>
#include <string.h>
#include <cuda_runtime_api.h>

#include "cudpp.h"

#include "cudpp_testrig_options.h"
#include "cudpp_testrig_utils.h"
#include "cuda_util.h"
#include "stopwatch.h"
#include "comparearrays.h"
#include "commandline.h"
#include "compact_gold.h"

using namespace cudpp_app;

/**
 * testCompact exercises cudpp's compact functionality.
 * Possible command line arguments:
 * - --forward, --backward: sets direction of compact
 * - --n=#: number of elements in input
 * - --prob=#: fraction (0.0-1.0) of elements that are valid (default: 0.3)
 * - Also "global" options (see setOptions)
 * @param argc Number of arguments on the command line, passed
 * directly from main
 * @param argv Array of arguments on the command line, passed directly
 * from main
 * @return Number of tests that failed regression (0 for all pass)
 * @see setOptions, cudppCompact
 */
template <typename T>
int compactTest(int argc, const char **argv, 
                const CUDPPConfiguration &config, 
                testrigOptions &testOptions)
{
    int retval = 0;

    cudpp_app::StopWatch timer;

    bool quiet = checkCommandLineFlag(argc, (const char**)argv, "quiet");   
   
    unsigned int test[] = {39, 128, 256, 512, 1000, 1024, 1025, 32768, 45537, 65536, 131072,
        262144, 500001, 524288, 1048577, 1048576, 1048581, 2097152, 4194304, 8388608};
    int numTests = sizeof(test) / sizeof(test[0]);
    int numElements = test[numTests-1]; // maximum test size

    bool oneTest = false;
    if (commandLineArg(numElements, argc, (const char**) argv, "n"))
    {
        oneTest = true;
        numTests = 1;
        test[0] = numElements;
    }

    float probValid = 0.3f;
    commandLineArg(probValid, argc, (const char**) argv, "prob");

    CUDPPResult result = CUDPP_SUCCESS;
    CUDPPHandle theCudpp;
    result = cudppCreate(&theCudpp);
    if (result != CUDPP_SUCCESS)
    {
        if (!quiet)
            fprintf(stderr, "Error initializing CUDPP Library.\n");
        retval = (oneTest) ? 1 : numTests;
        return retval;
    }

    CUDPPHandle plan;
    result = cudppPlan(theCudpp, &plan, config, numElements, 1, 0);

    if (result != CUDPP_SUCCESS)
    {
        if (!quiet)
            fprintf(stderr, "Error creating plan for Compact\n");
        retval = (oneTest) ? 1 : numTests;
        return retval;
    }

    unsigned int memSize = sizeof(T) * numElements;
    
    // allocate host memory to store the input data
    T* h_data = (T*) malloc( memSize);
    unsigned int *h_isValid = (unsigned int*) malloc(sizeof(unsigned int) * numElements);

    // allocate and compute reference solution
    T* reference = (T*) malloc( memSize);

    // allocate device memory input and output arrays
    T* d_idata     = NULL;
    T* d_odata     = NULL;
    unsigned int* d_isValid   = NULL;
    size_t* d_numValid  = NULL;

    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_idata, memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_odata, memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_isValid, sizeof(unsigned int) * numElements));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_numValid, sizeof(size_t)));

    size_t numValidElements = 0;

    // numTests = numTests;
    for (int k = 0; k < numTests; ++k)
    {
        if (!quiet)
        {
           printf("Running a%s stream compact of %d %s elements\n",
                  (config.options & CUDPP_OPTION_BACKWARD) ? " backward" : "",
                  test[k],
                  datatypeToString(config.datatype));
           fflush(stdout);
        }

        //srand((unsigned int)time(NULL));
        srand(222);

        for( unsigned int i = 0; i < test[k]; ++i)
        {
            if (rand() / (float)RAND_MAX > probValid)
                h_isValid[i] = 0;
            else
                h_isValid[i] = 1;
            h_data[i] = (T)(rand() + 1);
        }

        memset(reference, 0, sizeof(T) * test[k]);
        size_t c_numValidElts =
            compactGold( reference, h_data, h_isValid, test[k], config);
        CUDA_SAFE_CALL( cudaMemcpy(d_idata, h_data, sizeof(T) * test[k],
                                   cudaMemcpyHostToDevice) );

        CUDA_SAFE_CALL( cudaMemcpy(d_isValid, h_isValid, sizeof(unsigned int) * test[k],
                                   cudaMemcpyHostToDevice) );

        CUDA_SAFE_CALL( cudaMemset(d_odata, 0, sizeof(T) * test[k]));

        // run once to avoid timing startup overhead.
        cudppCompact(plan, d_odata, d_numValid, d_idata, d_isValid, test[k]);

        timer.reset();
        timer.start();
        for (int i = 0; i < testOptions.numIterations; i++)
        {
            cudppCompact(plan, d_odata, d_numValid, d_idata, d_isValid, test[k]);
        }
        cudaThreadSynchronize();
        timer.stop();

        // get number of valid elements back to host
        CUDA_SAFE_CALL( cudaMemcpy(&numValidElements, d_numValid, sizeof(size_t), 
                                   cudaMemcpyDeviceToHost) );

        // allocate host memory to store the output data

        T* o_data = (T*) malloc( sizeof(T) * numValidElements);

        // copy result from device to host
        CUDA_SAFE_CALL(cudaMemcpy(o_data, d_odata,
                                  sizeof(T) * numValidElements,
                                  cudaMemcpyDeviceToHost));
        // check if the result is equivalent to the expected soluion
        if (!quiet)
            printf("numValidElements: %ld\n", numValidElements);
            
        bool result = compareArrays( reference, o_data, (unsigned int)numValidElements, 0.001f);

        free(o_data);

        if (c_numValidElts != numValidElements)
        {
            retval += 1;
            if (!quiet)
            {
                printf("Number of valid elements does not match reference solution.\n");
                printf("Test FAILED\n");
            }
        }
        else
        {
            retval += result ? 0 : 1;
            if (!quiet)
            {
                printf("test %s\n", result ? "PASSED" : "FAILED");
            }
        }
        if (!quiet)
        {
            printf("Average execution time: %f ms\n",
                   timer.getTime() / testOptions.numIterations);
        }
        else
            printf("\t%10d\t%0.4f\n", test[k], timer.getTime() / testOptions.numIterations);
    }
    if (!quiet)
        printf("\n");

    result = cudppDestroyPlan(plan);
    if (result != CUDPP_SUCCESS)
    {
        if (!quiet)
            printf("Error destroying CUDPPPlan for Scan\n");
    }

    result = cudppDestroy(theCudpp);
    if (result != CUDPP_SUCCESS)
    {
        if (!quiet)
            printf("Error shutting down CUDPP Library.\n");
    }

    // cleanup memory
    free( h_data);
    free( h_isValid);
    free( reference);
    cudaFree( d_odata);
    cudaFree( d_idata);
    cudaFree( d_isValid);
    cudaFree( d_numValid);
    return retval;
}

int testCompact(int argc, const char **argv, const CUDPPConfiguration *configPtr)
{
    testrigOptions testOptions;
    setOptions(argc, argv, testOptions);

    CUDPPConfiguration config;
    config.algorithm = CUDPP_COMPACT;
        
    if (configPtr != NULL)
    {
        config = *configPtr;
    }
    else
    {
        CUDPPOption direction = CUDPP_OPTION_FORWARD;

        config.datatype = getDatatypeFromArgv(argc, argv);
            
        if (checkCommandLineFlag(argc, argv, "backward"))
        {
            direction = CUDPP_OPTION_BACKWARD;
        } 
    }

    switch(config.datatype)
    {
    case CUDPP_CHAR:
        return compactTest<char>(argc, argv, config, testOptions);
        break;
    case CUDPP_UCHAR:
        return compactTest<unsigned char>(argc, argv, config, testOptions);
        break;    
    case CUDPP_INT:
        return compactTest<int>(argc, argv, config, testOptions);
        break;
    case CUDPP_UINT:
        return compactTest<unsigned int>(argc, argv, config, testOptions);
        break;
    case CUDPP_FLOAT:
        return compactTest<float>(argc, argv, config, testOptions);
        break;
    case CUDPP_DOUBLE:
        return compactTest<double>(argc, argv, config, testOptions);
        break;
    case CUDPP_LONGLONG:
        return compactTest<long long>(argc, argv, config, testOptions);
        break;
    case CUDPP_ULONGLONG:
        return compactTest<unsigned long long>(argc, argv, config, testOptions);
        break;
    default:
        return 0;
        break;
    }
    return 0;
}

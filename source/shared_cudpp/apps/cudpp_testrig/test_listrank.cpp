// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: $
// $Date: $
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// -------------------------------------------------------------

/**
 * @file
 * test_listrank.cpp
 *
 * @brief Host testrig routines to exercise cudpp's listrank functionality.
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
#include "listrank_gold.h"

using namespace cudpp_app;

/**
 * testListRank exercises cudpp's listrank functionality.
 * Possible command line arguments:
 * - --n=#: number of elements in input
 * @param argc Number of arguments on the command line, passed
 * directly from main
 * @param argv Array of arguments on the command line, passed directly
 * from main
 * @return Number of tests that failed regression (0 for all pass)
 * @see setOptions, cudppListRank
 */
template <typename T>
int listRankTest(int argc, const char **argv, const CUDPPConfiguration &config,
                 const testrigOptions &testOptions)
{
    int retval = 0;

    cudpp_app::StopWatch timer;

    bool quiet = checkCommandLineFlag(argc, (const char**)argv, "quiet");   

    unsigned int test[] = {39, 128, 256, 512, 1000, 1024, 1025, 32768, 45537, 65536, 131072,
        262144, 500001, 524288, 1048577, 1048576, 1048581};
    int numTests = sizeof(test) / sizeof(test[0]);
    int numElements = test[numTests-1]; // maximum test size

    bool oneTest = false;
    if (commandLineArg(numElements, argc, (const char**) argv, "n"))
    {
        oneTest = true;
        numTests = 1;
        test[0] = numElements;
    }

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
            fprintf(stderr, "Error creating plan for ListRank\n");
        retval = (oneTest) ? 1 : numTests;
        return retval;
    }

    unsigned int memSize = sizeof(T) * numElements;

    // allocate host memory to store the input data
    unsigned int head   = 0;
    T* h_values    = (T*) malloc( memSize);
    int* h_tmpvalues    = (int*) malloc( sizeof(int) * numElements);
    int* h_next_indices = (int*) malloc( sizeof(int) * numElements);

    // allocate and compute reference solution
    T* reference = (T*) malloc( memSize);

    // allocate device memory input and output arrays
    T* d_ivalues        = NULL;
    int* d_inextindices = NULL;
    T* d_ovalues        = NULL;

    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_ivalues, memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_inextindices, sizeof(int) * numElements));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_ovalues, memSize));

    for (int k = 0; k < numTests; ++k)
    {
        if (!quiet)
        {
            printf("Running a listrank of %u %s nodes\n",
                test[k],
                datatypeToString(config.datatype));
            fflush(stdout);
        }

        // create a random linked-list of test[k] nodes
        int* tracked_prev_indices = (int*) malloc(sizeof(int)*test[k]);
        for(int i=0; i<(int)test[k]; i++){
            h_tmpvalues[i] = i;
            h_next_indices[i] = 0;
        }
        // shuffle
        for(int i=0;i<(int)test[k];i++){
            int other = i + (rand()%(test[k]-i));
            // this only swaps the value
            int tmp_value       = h_tmpvalues[i];
            int other_value     = h_tmpvalues[other];

            h_tmpvalues[i]         = other_value;
            h_tmpvalues[other]     = tmp_value;

            tracked_prev_indices[(other_value+1)%test[k]] = i;
        }
        // now we find indices
        for(int i=0; i<(int)test[k]; i++)
        {
            int value = h_tmpvalues[i];
            int my_previous = tracked_prev_indices[value];

            if(value==0)
                h_next_indices[my_previous] = -1;
            else
                h_next_indices[my_previous] = i;
        }
        // find head
        for(unsigned int i=0; i<test[k]; i++){
            int value = h_tmpvalues[i];
            if(value==0){
                head = i;
            }
        }
        for (unsigned int i=0; i<test[k]; i++){
            h_values[i] = (T)h_tmpvalues[i];
        }
        free( tracked_prev_indices);

        memset(reference, 0, sizeof(T) * test[k]);
        listRankGold<T>( reference, h_values, h_next_indices, head, test[k]);
        
        CUDA_SAFE_CALL( cudaMemcpy(d_ivalues, h_values, sizeof(T) * test[k],
                                   cudaMemcpyHostToDevice) );

        CUDA_SAFE_CALL( cudaMemcpy(d_inextindices, h_next_indices, sizeof(int) * test[k],
                                   cudaMemcpyHostToDevice) );

        CUDA_SAFE_CALL( cudaMemset(d_ovalues, 0, sizeof(T) * test[k]));

        // run once to avoid timing startup overhead.
        cudppListRank(plan, d_ovalues, d_ivalues, d_inextindices, head, test[k]);

        timer.reset();
        timer.start();
        for (int i = 0; i < testOptions.numIterations; i++)
        {
            cudppListRank(plan, d_ovalues, d_ivalues, d_inextindices, head, test[k]);
        }
        cudaThreadSynchronize();
        timer.stop();

        // allocate host memory to store the output data
        T* o_data = (T*) malloc( sizeof(T) * test[k]);

        // copy result from device to host
        CUDA_SAFE_CALL(cudaMemcpy(o_data, d_ovalues,
                                  sizeof(T) * test[k],
                                  cudaMemcpyDeviceToHost));
            
        bool result = compareArrays<T>( reference, o_data, test[k]);

        free(o_data);

        retval += result ? 0 : 1;
        if (!quiet)
        {
            printf("test %s\n", result ? "PASSED" : "FAILED");
        }
        if (!quiet)
        {
            printf("Average execution time: %f ms\n",
                timer.getTime() / testOptions.numIterations);
        }
        else
            printf("\t%10d\t%0.4f\n", test[k], timer.getTime() / testOptions.numIterations);
    }

    result = cudppDestroyPlan(plan);
    if (result != CUDPP_SUCCESS)
    {
        if (!quiet)
            printf("Error destroying CUDPPPlan for ListRank\n");
    }

    result = cudppDestroy(theCudpp);
    if (result != CUDPP_SUCCESS)
    {
        if (!quiet)
            printf("Error shutting down CUDPP Library.\n");
    }

    // cleanup memory
    free( h_values);
    free( h_tmpvalues);
    free( h_next_indices);
    free( reference);
    cudaFree( d_ivalues);
    cudaFree( d_inextindices);
    cudaFree( d_ovalues);
    return retval;
}

int testListRank(int argc, const char **argv, 
                 const CUDPPConfiguration *configPtr)
{
    testrigOptions testOptions;
    setOptions(argc, argv, testOptions);

    CUDPPConfiguration config;
    config.algorithm = CUDPP_LISTRANK;
    config.options = 0;

    if (configPtr != NULL)
    {
        config = *configPtr;
    }
    else
    {
        config.datatype = CUDPP_INT;
    }

    switch(config.datatype)
    {
    case CUDPP_CHAR:
        return listRankTest<char>(argc, argv, config, testOptions);
        break;
    case CUDPP_UCHAR:
        return listRankTest<unsigned char>(argc, argv, config, testOptions);
        break;    
    case CUDPP_INT:
        return listRankTest<int>(argc, argv, config, testOptions);
        break;
    case CUDPP_UINT:
        return listRankTest<unsigned int>(argc, argv, config, testOptions);
        break;
    case CUDPP_FLOAT:
        return listRankTest<float>(argc, argv, config, testOptions);
        break;
    case CUDPP_LONGLONG:
        return listRankTest<long long>(argc, argv, config, testOptions);
        break;
    case CUDPP_ULONGLONG:
        return listRankTest<unsigned long long>(argc, argv, config, testOptions);
        break;
    default:
        return 0;
        break;
    }
    return 0;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

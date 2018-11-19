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
 * test_scan.cu
 *
 * @brief Host testrig routines to exercise cudpp's scan functionality.
 */

#include <stdio.h>
#include <time.h>
#include <limits.h>
#include <cstring>
#include <cuda_runtime_api.h>

#include "cudpp.h"
#include "cudpp_testrig_options.h"
#include "cudpp_testrig_utils.h"
#include "cuda_util.h"
#include "comparearrays.h"
#include "stopwatch.h"
#include "commandline.h"

#include "scan_gold.h"

using namespace cudpp_app;

/**
 * testScan exercises cudpp's unsegmented scan functionality.
 * Possible command line arguments:
 * - --op=OP: sets scan operation to OP (sum, max, min and multiply.)
 * - --forward, --backward: sets direction of scan
 * - --exclusive, --inclusive: sets exclusivity of scan
 * - --n=#: number of elements in scan
 * - Also "global" options (see setOptions)
 * @param argc Number of arguments on the command line, passed
 * directly from main
 * @param argv Array of arguments on the command line, passed directly
 * from main
 * @param configPtr Configuration for scan, set by caller
 * @return Number of tests that failed regression (0 for all pass)
 * @see CUDPPConfiguration, setOptions, cudppScan
 */
template <typename T>
int scanTest(int argc, const char **argv, const CUDPPConfiguration &config,
             const testrigOptions &testOptions)
{
    int retval = 0;

    cudpp_app::StopWatch timer;

    bool quiet = checkCommandLineFlag(argc, (const char**) argv, "quiet");

    bool oneTest = false;
    int numElements;
    if (commandLineArg(numElements, argc, (const char**) argv, "n"))
    {
        oneTest = true;
    }

    unsigned int test[] = {39, 128, 256, 512, 1000, 1024, 1025, 32768, 45537,
                           65536, 131072, 262144, 500001, 524288, 1048577,
                           1048576, 1048581, 2097152, 4194304, 8388608};

    int numTests = sizeof(test) / sizeof(test[0]);
    numElements = test[numTests-1]; // maximum test size

    if (oneTest)
    {
        test[0] = numElements;
        numTests = 1;
    }

    // Initialize CUDPP
    CUDPPResult result = CUDPP_SUCCESS;
    CUDPPHandle theCudpp;
    result = cudppCreate(&theCudpp);
    if (result != CUDPP_SUCCESS)
    {
        fprintf(stderr, "Error initializing CUDPP Library\n");
        retval = (oneTest) ? 1 : numTests;
        return retval;
    }

    CUDPPHandle plan;

    result = cudppPlan(theCudpp, &plan, config, numElements, 1, 0);

    if (result != CUDPP_SUCCESS)
    {
        fprintf(stderr, "Error creating plan for Scan\n");
        retval = (oneTest) ? 1 : numTests;
        return retval;
    }

    unsigned int memSize = sizeof(T) * numElements;

    // allocate host memory to store the input data
    T* i_data = (T*) malloc( memSize);

    // allocate host memory to store the output data
    T* o_data = (T*) malloc( memSize);

    // host memory to store input flags

    // initialize the input data on the host
    for(int i = 0; i < numElements; ++i)
    {
        // @TODO: Not thrilled that we're only scanning 0s and 1s --JDO
        i_data[i] = 1;//(T)(rand() & 1);
    }

    unsigned int *i_flags = 0;
    unsigned int *d_iflags = 0;
    if (config.algorithm == CUDPP_SEGMENTED_SCAN)
    {
        i_flags = (unsigned int*)malloc(sizeof(unsigned int) * numElements);
        CUDA_SAFE_CALL( cudaMalloc( (void**) &d_iflags,
                                    sizeof(unsigned int) * numElements));
    }

    // allocate and compute reference solution
    T* reference = (T*) malloc( memSize);

    // allocate device memory input and output arrays
    T* d_idata     = NULL;
    T* d_odata     = NULL;

    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_idata, memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_odata, memSize));

    // copy host memory to device input array
    CUDA_SAFE_CALL( cudaMemcpy(d_idata, i_data, memSize,
                               cudaMemcpyHostToDevice) );
    // initialize all the other device arrays to be safe
    CUDA_SAFE_CALL( cudaMemcpy(d_odata, o_data, memSize,
                               cudaMemcpyHostToDevice) );

    for (int k = 0; k < numTests; ++k)
    {
        if (config.algorithm == CUDPP_SEGMENTED_SCAN)
        {
            int numFlags = 4;
            memset(i_flags, 0, sizeof(unsigned int) * test[k]);
            // Generate flags
            for(int i = 0; i < numFlags; ++i)
            {
                unsigned int idx;

                // The flag at the first position is implicitly set
                // so try to generate non-zero positions
                while((idx = (unsigned int)
                       ((test[k] - 1) * (rand() / (double)RAND_MAX)))
                      == 0)
                {
                }

                i_flags[idx] = 1;
            }

            // Copy flags to GPU
            CUDA_SAFE_CALL( cudaMemcpy(d_iflags, i_flags,
                                       sizeof(unsigned int) * test[k],
                                       cudaMemcpyHostToDevice) );
        }

        char op[10];
        switch (config.op)
        {
        case CUDPP_ADD:
            strcpy(op, "sum");
            break;
        case CUDPP_MULTIPLY:
            strcpy(op, "multiply");
            break;
        case CUDPP_MAX:
            strcpy(op, "max");
            break;
        case CUDPP_MIN:
            strcpy(op, "min");
            break;
        case CUDPP_OPERATOR_INVALID:
            fprintf(stderr, "testScan called with invalid operator\n");
            break;
        }

        if (!quiet)
        {
            printf("Running a%s%s %s%s-scan of %d %s elements\n",
                   (config.options & CUDPP_OPTION_BACKWARD) ? " backward" : "",
                   (config.options & CUDPP_OPTION_INCLUSIVE) ? " inclusive" : 
                   "",
                   (config.algorithm == CUDPP_SEGMENTED_SCAN) ? "segmented " : 
                   "",
                   op,
                   test[k],
                   datatypeToString(config.datatype));
            fflush(stdout);
        }

        timer.reset();
        timer.start();

        if (config.algorithm == CUDPP_SEGMENTED_SCAN)
        {
            if (config.op == CUDPP_ADD)
                computeSegmentedSumScanGold( reference, i_data, i_flags, 
                                             test[k], config);
            else if (config.op == CUDPP_MULTIPLY)
                computeSegmentedMultiplyScanGold( reference, i_data, i_flags, 
                                                  test[k], config);
            else if (config.op == CUDPP_MAX)
                computeSegmentedMaxScanGold( reference, i_data, i_flags, 
                                             test[k], config);     
            else if (config.op == CUDPP_MIN)
                computeSegmentedMinScanGold( reference, i_data, i_flags, 
                                             test[k], config);                
        }
        else
        {
            if (config.op == CUDPP_ADD)
                computeSumScanGold( reference, i_data, test[k], config);
            else if (config.op == CUDPP_MULTIPLY)
                computeMultiplyScanGold( reference, i_data, test[k], config);
            else if (config.op == CUDPP_MAX)
                computeMaxScanGold( reference, i_data, test[k], config);
            else if (config.op == CUDPP_MIN)
                computeMinScanGold( reference, i_data, test[k], config);
        }

        timer.stop();

        if (!quiet)
            printf("CPU execution time = %f\n", timer.getTime());
        timer.reset();

        // Run the scan
        // run once to avoid timing startup overhead.
        if (config.algorithm == CUDPP_SEGMENTED_SCAN)
            cudppSegmentedScan(plan, d_odata, d_idata, d_iflags, test[k]);
        else
            cudppScan(plan, d_odata, d_idata, test[k]);

        timer.start();
        for (int i = 0; i < testOptions.numIterations; i++)
        {
            if (config.algorithm == CUDPP_SEGMENTED_SCAN)
                cudppSegmentedScan(plan, d_odata, d_idata, d_iflags, test[k]);
            else
                cudppScan(plan, d_odata, d_idata, test[k]);
        }
        cudaThreadSynchronize();
        timer.stop();

        // copy result from device to host
        CUDA_SAFE_CALL(cudaMemcpy( o_data, d_odata, sizeof(T) * test[k],
                                   cudaMemcpyDeviceToHost));

        // check if the result is equivalent to the expected solution
        bool result = compareArrays( reference, o_data, test[k], 0.001f);

        retval += result ? 0 : 1;
        if (!quiet)
        {
            printf("test %s\n", result ? "PASSED" : "FAILED");
            printf("Average execution time: %f ms\n",
                   timer.getTime() / testOptions.numIterations);
        }
        else
        {
            printf("\t%10d\t%0.4f\n", test[k], 
                   timer.getTime() / testOptions.numIterations);
        }
        if (testOptions.debug)
        {
            printArray(i_data, numElements);
            printArray(o_data, numElements);
        }
    }
    if (!quiet)
        printf("\n");

    result = cudppDestroyPlan(plan);

    if (result != CUDPP_SUCCESS)
    {
        printf("Error destroying CUDPPPlan for Scan\n");
    }

    result = cudppDestroy(theCudpp);

    if (result != CUDPP_SUCCESS)
    {
        printf("Error shutting down CUDPP Library\n");
    }

    // cleanup memory
    free(i_data);
    free(o_data);
    free(reference);
    if (i_flags) free(i_flags);
    cudaFree(d_odata);
    cudaFree(d_idata);
    if (d_iflags) cudaFree(d_iflags);
    return retval;
}


/**
 * testMultiSumScan exercises cudpp's multiple-unsegmented-scan functionality.
 * @param argc Number of arguments on the command line, passed
 * directly from main
 * @param argv Array of arguments on the command line, passed directly
 * from main
 * @return Number of tests that failed regression (0 for all pass)
 * @see cudppMultiScan
 */
template <class T>
int multiscanTest(int argc, const char **argv, const CUDPPConfiguration &config,
                  const testrigOptions &testOptions, cudaDeviceProp devProps)
{
    int retval = 0;

    cudpp_app::StopWatch timer;

    bool quiet = checkCommandLineFlag(argc, (const char**) argv, "quiet");

    unsigned int test[] = {39, 128, 256, 512, 1000, 1024, 1025, 32768, 45537, 
                           65536, 131072, 262144, 500001, 524288, 1048577, 
                           1048576, 1048581, 2097152, 4194304, 8388608};

    int numTests = sizeof(test) / sizeof(test[0]);

    int numElements = test[numTests-1]; // maximum test size
    int numRows = 10;

    if (commandLineArg(numElements, argc, (const char**) argv, "n"))
    {
        test[0] = numElements;
        numTests = 1;
    }
    bool fixedNumRows = false;
    if (commandLineArg(numRows, argc, (const char**) argv, "r"))
    {
        fixedNumRows = true;
    }

    char op[10];
    switch (config.op)
    {
    case CUDPP_ADD:
        strcpy(op, "sum");
        break;
    case CUDPP_MULTIPLY:
        strcpy(op, "multiply");
        break;
    case CUDPP_MAX:
        strcpy(op, "max");
        break;
    case CUDPP_MIN:
        strcpy(op, "min");
        break;
    case CUDPP_OPERATOR_INVALID:
        fprintf(stderr, "testScan called with invalid operator\n");
        break;
    }

    CUDPPResult ret;
    CUDPPHandle theCudpp;
    ret = cudppCreate(&theCudpp);

    if (ret != CUDPP_SUCCESS)
    {
        fprintf(stderr, "Error Initializing CUDPP Library.\n");
        retval = 1;
        return retval;
    }

    for (int k = 0; k < numTests; ++k)
    {    
        size_t freeMem, totalMem;
        CUDA_SAFE_CALL(cudaMemGetInfo(&freeMem, &totalMem));
        unsigned int memNeeded = test[k] * numRows * sizeof(T) * 3;
        while (memNeeded > freeMem) 
        {
            numRows /= 2;
            memNeeded = test[k] * numRows * sizeof(T) * 3;
        }

        if (numRows < 1) {
            fprintf(stderr,
                    "multiscanTest: Error, not enough memory to run test\n");
            break;
        }

        if (!quiet)
        {
            printf("Running a%s%s %s-multiscan of %d %s elements in %d rows\n",
                   (config.options & CUDPP_OPTION_BACKWARD) ? " backward" : "",
                   (config.options & CUDPP_OPTION_INCLUSIVE) ? " inclusive" :
                   "",
                   op,
                   test[k],
                   datatypeToString(config.datatype),
                   numRows);
            fflush(stdout);
        }

        size_t kPitch = test[k] * sizeof(T);
        size_t hmemSize = numRows * kPitch;

        // allocate host memory to store the input data
        T* i_data = (T*) malloc(hmemSize);
        if (!i_data) {
            printf("Error, out of host memory\n");
            break;
        }

        // allocate host memory to store the output data
        T* o_data = (T*) malloc(hmemSize);
        if (!o_data) {
            printf("Error, out of host memory\n");
            break;
        }

        for( unsigned int i = 0; i < test[k] * numRows; ++i)
        {
            i_data[i] = (T)(rand() & 1);
            o_data[i] = -1;
        }

        // allocate and compute reference solution
        T* reference = (T*) malloc(hmemSize);
        if (!reference) {
            printf("Error, out of host memory\n");
            break;
        }

        if (config.op == CUDPP_ADD)
            computeMultiRowScanGold<T, OperatorAdd<T> >( reference, i_data, 
                                                         test[k], numRows, 
                                                         config);
        else if (config.op == CUDPP_MULTIPLY)
            computeMultiRowScanGold<T, OperatorMultiply<T> >( reference, i_data,
                                                              test[k], numRows, 
                                                              config);
        else if (config.op == CUDPP_MAX)
            computeMultiRowScanGold<T, OperatorMax<T> >( reference, i_data, 
                                                         test[k], numRows, 
                                                         config);
        else if (config.op == CUDPP_MIN)
            computeMultiRowScanGold<T, OperatorMin<T> >( reference, i_data, 
                                                         test[k], numRows, 
                                                         config);
        
        // allocate device memory input and output arrays
        T* d_idata     = NULL;
        T* d_odata     = NULL;
        size_t d_ipitch = 0;
        size_t d_opitch = 0;

        CUDA_SAFE_CALL(cudaMallocPitch((void**) &d_idata, &d_ipitch,
                                       kPitch, numRows));
        CUDA_SAFE_CALL(cudaMallocPitch((void**) &d_odata, &d_opitch,
                                       kPitch, numRows));
        // copy host memory to device input array
        CUDA_SAFE_CALL(cudaMemcpy2D(d_idata, d_ipitch, i_data, kPitch, kPitch,
                                    numRows, cudaMemcpyHostToDevice));
        // initialize all the other device arrays to be safe
        CUDA_SAFE_CALL(cudaMemcpy2D(d_odata, d_ipitch, o_data, kPitch, kPitch,
                                    numRows, cudaMemcpyHostToDevice));

        size_t rowPitch = d_ipitch / sizeof(T);

        CUDPPHandle multiscanPlan = 0;
        ret = cudppPlan(theCudpp, &multiscanPlan, config, test[k], numRows,
                        rowPitch);

        if (ret != CUDPP_SUCCESS)
        {
            fprintf(stderr, "Error creating CUDPP Plan for multi-row Scan.\n");
            retval = 1;
            return retval;
        }

        // run once to avoid timing startup overhead.
        cudppMultiScan(multiscanPlan, d_odata, d_idata, test[k], numRows);

        timer.start();
        for (int i = 0; i < testOptions.numIterations; i++)
        {
            cudppMultiScan(multiscanPlan, d_odata, d_idata, test[k], numRows);
        }
        cudaThreadSynchronize();
        timer.stop();

        // copy result from device to host
        CUDA_SAFE_CALL(cudaMemcpy2D( o_data, kPitch, d_odata, d_opitch,
                                     kPitch, numRows, cudaMemcpyDeviceToHost));

        // check if the result is equivalent to the expected solution
        bool result = compareArrays( reference, o_data, test[k]*numRows,
                                     0.001f);
        retval += (true == result) ? 0 : 1;
        if (!quiet)
        {
            printf("test %s\n", result ? "PASSED" : "FAILED");
            printf("Average execution time: %f ms\n",
                   timer.getTime() / testOptions.numIterations);
        }
        else
        {
            printf("\t%10d\t%0.4f\n", test[k],
                   timer.getTime() / testOptions.numIterations);
        }

        ret = cudppDestroyPlan(multiscanPlan);

        if (ret != CUDPP_SUCCESS)
        {
            printf("Error destroying CUDPPPlan for Multiscan\n");
        }

        // cleanup memory
        free(i_data);
        free(o_data);
        free(reference);
        cudaFree(d_odata);
        cudaFree(d_idata);
    }

    ret = cudppDestroy(theCudpp);

    if (ret != CUDPP_SUCCESS)
    {
        printf("Error shutting down CUDPP Library.\n");
    }

    return retval;
}

int testScan(int argc, const char **argv, const CUDPPConfiguration *configPtr,
             bool multiRow, cudaDeviceProp deviceProperties)
{
    testrigOptions testOptions;
    setOptions(argc, argv, testOptions);

    CUDPPConfiguration config;
    config.algorithm = CUDPP_SCAN;
    if (testOptions.algorithm == "segscan")
        config.algorithm = CUDPP_SEGMENTED_SCAN;

    if (multiRow) {
        testOptions.algorithm = "multiscan";
        config.algorithm = CUDPP_SCAN;
    }

    if (configPtr != NULL)
    {
        config = *configPtr;
    }
    else
    {
        CUDPPOption direction = CUDPP_OPTION_FORWARD;
        CUDPPOption inclusivity = CUDPP_OPTION_EXCLUSIVE;

        //default sum scan
        config.op = CUDPP_ADD;
        config.datatype = getDatatypeFromArgv(argc, argv);

        if (testOptions.op == "max")
        {
            config.op = CUDPP_MAX;
        }
        else if (testOptions.op == "min")
        {
            config.op = CUDPP_MIN;
        }
        else if (testOptions.op == "multiply")
        {
            config.op = CUDPP_MULTIPLY;
        }

        if (checkCommandLineFlag(argc, argv, "backward"))
        {
            direction = CUDPP_OPTION_BACKWARD;
        }

        if (checkCommandLineFlag(argc, argv, "exclusive"))
        {
            inclusivity = CUDPP_OPTION_EXCLUSIVE;
        }

        if (checkCommandLineFlag(argc, argv, "inclusive"))
        {
            inclusivity = CUDPP_OPTION_INCLUSIVE;
        }

        config.options = direction | inclusivity;
    }

    switch(config.datatype)
    {
    case CUDPP_INT:
        if (testOptions.algorithm == "multiscan")
            return multiscanTest<int>(argc, argv, config, testOptions,
                                      deviceProperties);
        else
            return scanTest<int>(argc, argv, config, testOptions);

        break;
    case CUDPP_UINT:
        if (testOptions.algorithm == "multiscan")
            return multiscanTest<unsigned int>(argc, argv, config, testOptions,
                                               deviceProperties);
        else
            return scanTest<unsigned int>(argc, argv, config, testOptions);
        break;
    case CUDPP_FLOAT:
        if (testOptions.algorithm == "multiscan")
            return multiscanTest<float>(argc, argv, config, testOptions,
                                        deviceProperties);
        else
            return scanTest<float>(argc, argv, config, testOptions);
        break;
    case CUDPP_DOUBLE:
        if (testOptions.algorithm == "multiscan")
            return multiscanTest<double>(argc, argv, config, testOptions,
                                         deviceProperties);
        else
            return scanTest<double>(argc, argv, config, testOptions);
        break;
    case CUDPP_LONGLONG:
        if (testOptions.algorithm == "multiscan")
            return multiscanTest<long long>(argc, argv, config, testOptions,
                                            deviceProperties);
        else
            return scanTest<long long>(argc, argv, config, testOptions);
        break;
    case CUDPP_ULONGLONG:
        if (testOptions.algorithm == "multiscan")
            return multiscanTest<unsigned long long>(argc, argv, config,
                                                     testOptions,
                                                     deviceProperties);
        else
            return scanTest<unsigned long long>(argc, argv, config,
                                                testOptions);
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

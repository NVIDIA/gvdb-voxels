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
 * test_reduce.cu
 *
 * @brief Host testrig routines to exercise cudpp's reduction functionality.
 */

#include <cstring>
#include <iostream>
#include <cuda_runtime_api.h>

#include "cudpp.h"
#include "cudpp_testrig_options.h"
#include "cudpp_testrig_utils.h"
#include "cuda_util.h"
#include "commandline.h"

using namespace cudpp_app;

template <class Oper, typename T>
void computeReduceGold( T* out, const T* idata, const unsigned int len)
{
    Oper op;
    T sum = op.identity();

    for (unsigned int i = 0; i < len; i++)
    {
        sum = op(sum, idata[i]);
    }
    *out = sum;
}

template <typename T>
int reduceTest(int argc, const char **argv, const CUDPPConfiguration &config,
               const testrigOptions &testOptions)
{
    int retval = 0;

    int numElements = 8388608; // maximum test size

    bool quiet = checkCommandLineFlag(argc, argv, "quiet");

    bool oneTest = false;
    if (commandLineArg(numElements, argc, argv, "n"))
    {
        oneTest = true;
    }

    unsigned int test[] = {39, 128, 256, 512, 1000, 1024, 1025, 32768, 45537, 65536, 131072,
        262144, 500001, 524288, 1048577, 1048576, 1048581, 2097152, 4194304, 8388608};

    int numTests = sizeof(test) / sizeof(test[0]);
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

    if(result != CUDPP_SUCCESS)
    {
        printf("Error in plan creation\n");
        retval = numTests;
        cudppDestroyPlan(plan);
        cudppDestroy(theCudpp);
        return retval;
    }

    unsigned int memSize = sizeof(T) * numElements;

    // allocate host memory to store the input data
    T* i_data = new T[numElements];

    // initialize the input data on the host
    float range = 100.f;
    if (config.op == CUDPP_MULTIPLY)
    {

        if ((config.datatype == CUDPP_FLOAT) ||
            (config.datatype == CUDPP_DOUBLE))   range = 1.f;
        else                                     range = 2.f;
    }
    
    if (config.datatype != CUDPP_FLOAT && config.datatype != CUDPP_DOUBLE)
        range = (float)(sizeof(T)*8);
        
    VectorSupport<T>::fillVector(i_data, numElements, range);
    
    T reference = 0;

    // allocate device memory input and output arrays
    T* d_idata     = (T *) NULL;
    T* d_odata     = (T *) NULL;

    CUDA_SAFE_CALL( cudaMalloc( (void **) &d_idata, memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void **) &d_odata, sizeof(T)));

    // copy host memory to device input array
    CUDA_SAFE_CALL( cudaMemcpy(d_idata, i_data, memSize, cudaMemcpyHostToDevice) );

    CUDA_SAFE_CALL( cudaMemset(d_odata, 0, sizeof(T)) );

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    
    for (int k = 0; k < numTests; ++k)
    {
        char op[10];
        switch (config.op)
        {
        case CUDPP_ADD:
        default:
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
        }
        char dt[10];
        switch (config.datatype)
        {
        case CUDPP_UINT:
        default:
            strcpy(dt, "uint");
            break;
        case CUDPP_INT:
            strcpy(dt, "int");
            break;
        case CUDPP_FLOAT:
            strcpy(dt, "float");
            break;
        case CUDPP_DOUBLE:
            strcpy(dt, "double");
            break;
        case CUDPP_LONGLONG:
            strcpy(dt, "longlong");
            break;
        case CUDPP_ULONGLONG:
            strcpy(dt, "ulonglong");
            break;
        }

        if (!quiet)
        {
            printf("Running a %s-reduction of %d %s elements\n", 
                   op, test[k], dt);
            fflush(stdout);
        }

        if (config.op == CUDPP_ADD)
            computeReduceGold<OperatorAdd<T>, T>( &reference, i_data, test[k]);
        else if (config.op == CUDPP_MULTIPLY)
            computeReduceGold<OperatorMultiply<T>, T>( &reference, i_data, test[k]);
        else if (config.op == CUDPP_MAX)
            computeReduceGold<OperatorMax<T>, T>( &reference, i_data, test[k]);
        else if (config.op == CUDPP_MIN)
            computeReduceGold<OperatorMin<T>, T>( &reference, i_data, test[k]);

        // Run the reduction
        // run once to avoid timing startup overhead.
        cudppReduce(plan, d_odata, d_idata, test[k]);

        CUDA_SAFE_CALL( cudaEventRecord(startEvent, 0) );
        for (int i = 0; i < testOptions.numIterations; i++)
        {
            cudppReduce(plan, d_odata, d_idata, test[k]);
        }
        CUDA_SAFE_CALL( cudaEventRecord(stopEvent, 0) );
        CUDA_SAFE_CALL( cudaEventSynchronize(stopEvent) );

        float time = 0;
        CUDA_SAFE_CALL( cudaEventElapsedTime(&time, startEvent, stopEvent));
        time /= testOptions.numIterations;
        double bandwidth = 1.0e-6 * test[k] * sizeof(T) / time;

        T o_data = 0;
        // copy result from device to host
        CUDA_SAFE_CALL(cudaMemcpy( &o_data, d_odata, sizeof(T), cudaMemcpyDeviceToHost));

        double threshold = (config.op == CUDPP_MULTIPLY && (config.datatype == CUDPP_FLOAT || config.datatype == CUDPP_DOUBLE)) ? 1 : test[k] * 5e-6;
        bool correct = (fabs((double)reference - (double)o_data) < threshold);

        // correct result?
        retval += (correct) ? 0 : 1;

        if (!quiet)
        {
            printf("test %s\n", (correct) ? "PASSED" : "FAILED");
            if (!correct)
                std::cout << o_data << " != " << reference << " (ref)" 
                          << std::endl;
            printf("Average execution time: %f ms, ", time);
            printf(": %0.4f GB/s\n", bandwidth);
        }
        else
        {
            printf("\t%10d\t%0.4f\t%0.4f\n", test[k], time, bandwidth);
        }
    }

    result = cudppDestroyPlan(plan);

    if (result != CUDPP_SUCCESS)
    {   
        printf("Error destroying CUDPPPlan for Scan\n");
        retval = numTests;
    }

    result = cudppDestroy(theCudpp);

    if (result != CUDPP_SUCCESS)
    {   
        printf("Error shutting down CUDPP Library.\n");
        retval = numTests;
    }

    delete [] i_data;
    cudaFree(d_odata);
    cudaFree(d_idata);
    return retval;
}

int testReduce(int argc, const char **argv, const CUDPPConfiguration *configPtr)
{
    testrigOptions testOptions;
    setOptions(argc, argv, testOptions);

    CUDPPConfiguration config;
    config.algorithm = CUDPP_REDUCE;
    config.options = 0;

    if (configPtr != NULL)
    {
        config = *configPtr;
    }
    else
    {
        //default sum scan
        config.op = CUDPP_ADD;

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

        // default float
        config.datatype = getDatatypeFromArgv(argc, argv);
    }

    switch(config.datatype)
    {
    case CUDPP_INT:
        return reduceTest<int>(argc, argv, config, testOptions);
        break;
    case CUDPP_UINT:
        return reduceTest<unsigned int>(argc, argv, config, testOptions);
        break;
    case CUDPP_FLOAT:
        return reduceTest<float>(argc, argv, config, testOptions);
        break;
    case CUDPP_DOUBLE:
        return reduceTest<double>(argc, argv, config, testOptions);
        break;
    case CUDPP_LONGLONG:
        return reduceTest<long long>(argc, argv, config, testOptions);
        break;
    case CUDPP_ULONGLONG:
        return reduceTest<unsigned long long>(argc, argv, config, testOptions);
        break;
    default:
        return 0;
        break;
    }
    return 0;
}

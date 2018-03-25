// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// ------------------------------------------------------------- 

#include <stdio.h>
#include <math.h>
#include <cuda_runtime_api.h>

#include "cudpp.h"
#include "cudpp_testrig_options.h"
#include "cudpp_testrig_utils.h"
#include "cuda_util.h"
#include "commandline.h"

#ifdef WIN32
#undef min
#undef max
#endif

#include <limits>


using namespace cudpp_app;

template <typename T>
int radixSortTest(CUDPPHandle theCudpp, CUDPPConfiguration config, size_t *tests, 
                  unsigned int numTests, size_t numElements, 
                  testrigOptions testOptions, bool quiet)
{
    int retval = 0;
    
    T *h_keys, *h_keysSorted, *d_keys;
    unsigned int *h_values, *h_valuesSorted, *d_values;

    h_keys       = (T*)malloc(numElements*sizeof(T));
    h_keysSorted = (T*)malloc(numElements*sizeof(T));
    h_values     = 0;                   
    h_valuesSorted = 0;

    if (config.options & CUDPP_OPTION_KEY_VALUE_PAIRS)      
    {
        h_values       = (unsigned int*)malloc(numElements*sizeof(unsigned int));
        h_valuesSorted = (unsigned int*)malloc(numElements*sizeof(unsigned int));

        for(unsigned int i=0; i < numElements; ++i)                     
            h_values[i] = i;            
    }                                                                                                                                   

    // Fill up with some random data   
    if (config.datatype != CUDPP_FLOAT && config.datatype != CUDPP_DOUBLE)
        VectorSupport<T>::fillVectorKeys(h_keys, numElements, 32);         
    else
        VectorSupport<T>::fillVector(h_keys, numElements, std::numeric_limits<float>::max() );         

    CUDA_SAFE_CALL(cudaMalloc((void **)&d_keys, numElements*sizeof(T)));
    if (config.options & CUDPP_OPTION_KEY_VALUE_PAIRS)
    {
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_values, numElements*sizeof(unsigned int)));
    }
    else
    {
        d_values = 0;
    }
    
    CUDPPHandle plan;   
    CUDPPResult result = cudppPlan(theCudpp, &plan, config, numElements, 1, 0);     

    if(result != CUDPP_SUCCESS)
    {
        printf("Error in plan creation\n");
        retval = numTests;
        cudppDestroyPlan(plan);
        return retval;
    }

    // run multiple iterations to compute an average sort time
    cudaEvent_t start_event, stop_event;
    CUDA_SAFE_CALL( cudaEventCreate(&start_event) );
    CUDA_SAFE_CALL( cudaEventCreate(&stop_event) );

    for (unsigned int k = 0; k < numTests; ++k)
    {
        if(numTests == 1)
            tests[0] = numElements;
            
        if (!quiet)
        {
            printf("Running a %s radix sort of %ld %s %s\n",
                  (config.options & CUDPP_OPTION_BACKWARD) ? " backward" : "forward",
                  tests[k],
                  datatypeToString(config.datatype),
                  (config.options & CUDPP_OPTION_KEY_VALUE_PAIRS) ? "key-value pairs" : "keys");
            fflush(stdout);
        }                                        
            
        float totalTime = 0;

        for (int i = 0; i < testOptions.numIterations; i++)
        {
            CUDA_SAFE_CALL(cudaMemcpy(d_keys, h_keys, tests[k] * sizeof(T), cudaMemcpyHostToDevice));
            if(config.options & CUDPP_OPTION_KEY_VALUE_PAIRS)
            {
                CUDA_SAFE_CALL( cudaMemcpy((void*)d_values, 
                                           (void*)h_values, 
                                           tests[k] * sizeof(unsigned int), 
                                           cudaMemcpyHostToDevice) );
            }

            CUDA_SAFE_CALL( cudaEventRecord(start_event, 0) );

            cudppRadixSort(plan, d_keys, (void*)d_values, tests[k]);

            CUDA_SAFE_CALL( cudaEventRecord(stop_event, 0) );
            CUDA_SAFE_CALL( cudaEventSynchronize(stop_event) );

            float time = 0;
            CUDA_SAFE_CALL( cudaEventElapsedTime(&time, start_event, stop_event));
            totalTime += time;
        }
        
        CUDA_CHECK_ERROR("testradixSort - cudppRadixSort");

        // copy results
        CUDA_SAFE_CALL(cudaMemcpy(h_keysSorted, d_keys, tests[k] * sizeof(T), cudaMemcpyDeviceToHost));
        if (config.options & CUDPP_OPTION_KEY_VALUE_PAIRS)
        {
            CUDA_SAFE_CALL( cudaMemcpy((void*)h_valuesSorted, 
                                       (void*)d_values, 
                                       tests[k] * sizeof(unsigned int), 
                                       cudaMemcpyDeviceToHost) );
        }
        else
            h_values = 0;               

        retval += VectorSupport<T>::verifySort(h_keysSorted, h_valuesSorted, h_keys, tests[k], 
                                               (config.options & CUDPP_OPTION_BACKWARD) != 0);

        if(!quiet)
        {                         
            printf("test %s\n", (retval == 0) ? "PASSED" : "FAILED");
            printf("Average execution time: %f ms\n", totalTime / testOptions.numIterations);
        }
        else
        {
            printf("\t%10ld\t%0.4f\n", tests[k], totalTime / testOptions.numIterations);
        }
    }
    printf("\n");

    CUDA_CHECK_ERROR("after radixsort");

    result = cudppDestroyPlan(plan);

    if (result != CUDPP_SUCCESS)
    {   
        printf("Error destroying CUDPPPlan for Scan\n");
        retval = numTests;
    }

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    cudaFree(d_keys);
    free(h_keys);
    free(h_keysSorted);
    
    if (config.options & CUDPP_OPTION_KEY_VALUE_PAIRS)
    {
        cudaFree(d_values);
        free(h_values);     
        free(h_valuesSorted);
    }

    return retval;
}

/**
 * testRadixSort tests cudpp's radix sort
 * Possible command line arguments:
 * - --keysonly, tests only a set of keys
 * - --keyval, tests a set of keys with associated values
 * - --n=#, number of elements in sort
 * @param argc Number of arguments on the command line, passed
 * directly from main
 * @param argv Array of arguments on the command line, passed directly
 * from main
 * @param configPtr Configuration for scan, set by caller
 * @return Number of tests that failed regression (0 for all pass)
 * @see cudppSort
*/
int testRadixSort(int argc, const char **argv, const CUDPPConfiguration *configPtr)
{

    int cmdVal;
    int retval = 0;
    
    bool quiet = checkCommandLineFlag(argc, argv, "quiet");        
    testrigOptions testOptions;
    setOptions(argc, argv, testOptions);        
    
    CUDPPConfiguration config;
    config.algorithm = CUDPP_SORT_RADIX;
    config.datatype = CUDPP_UINT;
    config.options = CUDPP_OPTION_KEY_VALUE_PAIRS;
             
    size_t test[] = {39, 128, 256, 512, 513, 1000, 1024, 1025, 32768, 
                     45537, 65536, 131072, 262144, 500001, 524288, 
                     1048577, 1048576, 1048581, 2097152, 4194304, 
                     8388608};
    
    int numTests = sizeof(test)/sizeof(test[0]);
    
    size_t numElements = test[numTests - 1];

    if(configPtr != NULL)
    {
        config = *configPtr;
    }
    else
    {
        config.datatype = getDatatypeFromArgv(argc, argv);

        bool keysOnly = checkCommandLineFlag(argc, argv, "keysonly");     
        bool backward = checkCommandLineFlag(argc, argv, "backward");
        
        config.options = CUDPP_OPTION_KEY_VALUE_PAIRS;
        
        if(keysOnly) 
            config.options = CUDPP_OPTION_KEYS_ONLY;
            
        if (backward)
            config.options |= CUDPP_OPTION_BACKWARD;   
    }

    if( commandLineArg( cmdVal, argc, (const char**)argv, "n" ) )
    { 
        numElements = cmdVal;
        numTests = 1;                           
    }
    
    CUDPPResult result = CUDPP_SUCCESS;  
    CUDPPHandle theCudpp;
    result = cudppCreate(&theCudpp);
    if(result != CUDPP_SUCCESS)
    {
        printf("Error initializing CUDPP Library.\n");
        retval = numTests;
        return retval;
    }
        
    switch(config.datatype)
    {        
    case CUDPP_CHAR:
        retval = radixSortTest<char>(theCudpp, config, test, numTests, numElements, testOptions, quiet);    
        break;
    case CUDPP_UCHAR:
        retval = radixSortTest<unsigned char>(theCudpp, config, test, numTests, numElements, testOptions, quiet);    
        break;
    case CUDPP_INT:
        retval = radixSortTest<int>(theCudpp, config, test, numTests, numElements, testOptions, quiet);
        break;
    case CUDPP_UINT:
        retval = radixSortTest<unsigned int>(theCudpp, config, test, numTests, numElements, testOptions, quiet);
        break;
    case CUDPP_FLOAT:   
        retval = radixSortTest<float>(theCudpp, config, test, numTests, numElements, testOptions, quiet);
        break;
    case CUDPP_DOUBLE:   
        retval = radixSortTest<double>(theCudpp, config, test, numTests, numElements, testOptions, quiet);
        break;
    case CUDPP_LONGLONG:   
        retval = radixSortTest<long long>(theCudpp, config, test, numTests, numElements, testOptions, quiet);
        break;
    case CUDPP_ULONGLONG:   
        retval = radixSortTest<unsigned long long>(theCudpp, config, test, numTests, numElements, testOptions, quiet);
        break;
    default:
        break;
    }

    result = cudppDestroy(theCudpp);

    if (result != CUDPP_SUCCESS)
    {   
        printf("Error shutting down CUDPP Library.\n");
        retval = numTests;
    }
                          
    return retval;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

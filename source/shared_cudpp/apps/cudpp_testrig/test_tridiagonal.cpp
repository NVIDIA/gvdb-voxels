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
 * test_tridiagonal.cpp
 *
 * @brief Host testrig routines to exercise cudpp's tridiagonal solver functionality.
 */

#include <memory.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <cstdlib>
#include <cstdio>

#include <cuda_runtime_api.h>
#include "cuda_util.h"
#include "commandline.h"

#include "cudpp_testrig_options.h"
#include "cudpp_testrig_utils.h"
#include "tridiagonal_gold.h"
#include "stopwatch.h"

using namespace cudpp_app;

template <typename T>
int testTridiagonalDataType(int argc, const char** argv, CUDPPConfiguration &config)
{
    bool quiet = checkCommandLineFlag(argc, argv, "quiet");

    int retval = 0;
    CUDPPHandle tridiagonalPlan = 0;
    CUDPPResult result;
    
    CUDPPHandle theCudpp;
    result = cudppCreate(&theCudpp);
    if(result != CUDPP_SUCCESS)
    {
        printf("Error initializing CUDPP Library.\n");
        retval = 1;
        return retval;
    }

    result = cudppPlan(theCudpp, &tridiagonalPlan, config, 0, 0, 0);
    if (CUDPP_SUCCESS != result)
    {
        printf("Error creating CUDPPPlan here\n");
        exit(-1);
    }

    bool oneTest = false;
    int numSystems = 512;
    int systemSize = 512;

    if (commandLineArg(systemSize, argc, argv, "n"))
    {
        oneTest = true;
    }

    int systemSizes[] = { 5, 32, 39, 128, 177, 255, 256, 500, 512 };

    int numTests = sizeof(systemSizes) / sizeof(int);

    if (oneTest)
    {
        systemSizes[0] = systemSize;
        numTests = 1;
    }

    for (int k = 0; k < numTests; k++)
    {
        systemSize = systemSizes[k];
        const unsigned int memSize = sizeof(T)*numSystems*systemSize;

        T* a = (T*) malloc(memSize);
        T* b = (T*) malloc(memSize);
        T* c = (T*) malloc(memSize);
        T* d = (T*) malloc(memSize);
        T* x1 = (T*) malloc(memSize);
        T* x2 = (T*) malloc(memSize);

        for (int i = 0; i < numSystems; i++)
        {
            testGeneration(&a[i*systemSize], &b[i*systemSize], &c[i*systemSize], &d[i*systemSize], &x1[i*systemSize], systemSize);
        }

        // allocate device memory input and output arrays
        T* d_a;
        T* d_b;
        T* d_c;
        T* d_d;
        T* d_x;

        CUDA_SAFE_CALL( cudaMalloc( (void**) &d_a,memSize));
        CUDA_SAFE_CALL( cudaMalloc( (void**) &d_b,memSize));
        CUDA_SAFE_CALL( cudaMalloc( (void**) &d_c,memSize));
        CUDA_SAFE_CALL( cudaMalloc( (void**) &d_d,memSize));
        CUDA_SAFE_CALL( cudaMalloc( (void**) &d_x,memSize));

       // copy host memory to device input array
        CUDA_SAFE_CALL( cudaMemcpy( d_a, a, memSize, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL( cudaMemcpy( d_b, b, memSize, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL( cudaMemcpy( d_c, c, memSize, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL( cudaMemcpy( d_d, d, memSize, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL( cudaMemcpy( d_x, x1, memSize, cudaMemcpyHostToDevice));

        // warm up the GPU to avoid the overhead time for the next timing
        CUDPPResult err = cudppTridiagonal(tridiagonalPlan, 
                                           d_a, 
                                           d_b, 
                                           d_c, 
                                           d_d, 
                                           d_x, 
                                           systemSize, 
                                           numSystems);

       if (err == CUDPP_ERROR_INSUFFICIENT_RESOURCES)
       {
           printf("System size %d (%s) is too large for this GPU -- skipping (PASSED)\n", 
                  systemSize, config.datatype == CUDPP_FLOAT ? "fp32" : "fp64");
           continue;
       }
       else if (err!= CUDPP_SUCCESS) 
       {
           printf("Error running cudppTridiagonal\n");
           retval++;
           continue;
       }
        
        if (!quiet)
            printf("Running a %s CR-PCR tridiagonal solver solving %d "
                   "systems of %d equations\n", 
                   config.datatype == CUDPP_FLOAT ? "fp32" : "fp64",
                   numSystems, systemSize);
        
        cudpp_app::StopWatch timer;
        timer.reset();
        timer.start();
        
        err = cudppTridiagonal(tridiagonalPlan, 
                               d_a, 
                               d_b, 
                               d_c, 
                               d_d, 
                               d_x, 
                               systemSize, 
                               numSystems);

        if (err != CUDPP_SUCCESS) 
        {
            printf("Error running cudppTridiagonal\n");
            retval++;
            continue;
        }
        cudaThreadSynchronize();

        timer.stop();            
        if (!quiet)
            printf("GPU execution time: %f ms\n", timer.getTime());
        else
            printf("%f\n", timer.getTime());
        
        // copy result from device to host
        CUDA_SAFE_CALL( cudaMemcpy(x2, d_x, memSize, cudaMemcpyDeviceToHost));

        // cleanup memory
        CUDA_SAFE_CALL(cudaFree(d_a));
        CUDA_SAFE_CALL(cudaFree(d_b));
        CUDA_SAFE_CALL(cudaFree(d_c));
        CUDA_SAFE_CALL(cudaFree(d_d));
        CUDA_SAFE_CALL(cudaFree(d_x));

        timer.reset();
        timer.start();
        
        serialManySystems<T>(a,b,c,d,x1,systemSize,numSystems);

        timer.stop();            
        if (!quiet)
            printf("CPU execution time: %f ms\n", timer.getTime());
        
        int failed = compareManySystems<T>(x1, x2, systemSize, numSystems, 0.001f);
        retval += failed;
        
        if (!quiet)
        {
            if (failed == 0)
                printf("test PASSED\n");
            else
                printf("test FAILED\n");
            printf("\n");
        }

        free(a);
        free(b);
        free(c);
        free(d);
        free(x1);
        free(x2);
    }

    return retval;
    
}

int testTridiagonal(int argc, const char** argv, const CUDPPConfiguration *configPtr)
{
    int retval = 0;

    testrigOptions testOptions;
    setOptions(argc, argv, testOptions);

    CUDPPConfiguration config;
    config.algorithm = CUDPP_TRIDIAGONAL;
    config.options = 0;

    if (configPtr != NULL)
    {
        config = *configPtr;
    }
    else
    {    
        config.datatype = getDatatypeFromArgv(argc, argv);
    }
    
    
    if (config.datatype == CUDPP_FLOAT)
        retval = testTridiagonalDataType<float>(argc, argv, config);
    else if (config.datatype == CUDPP_DOUBLE)
        retval = testTridiagonalDataType<double>(argc, argv, config);  
    
    return retval;
    
}

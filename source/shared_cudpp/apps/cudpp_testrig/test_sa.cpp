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
 * test_sa.cpp
 *
 * @brief Host testrig routines to exercise cudpp's suffix array functionality.
 */

#include <cstring>
#include <iostream>
#include <cuda_runtime_api.h>
#include <time.h>

#include "cudpp.h"
#include "cudpp_testrig_options.h"
#include "cudpp_testrig_utils.h"
#include "cuda_util.h"
#include "stopwatch.h"
#include "commandline.h"
#include "comparearrays.h"

#include "sparse.h"
using namespace cudpp_app;


int suffixArrayTest(int argc, const char **argv,
                    const CUDPPConfiguration &config,
                    const testrigOptions &testOptions)
{
    int retval = 0;

    cudpp_app::StopWatch timer;

    bool quiet = checkCommandLineFlag(argc, (const char**) argv, "quiet");
    unsigned int test[] = {39, 128, 256, 512, 513, 1000, 1024, 1025, 32768,
                           45537, 65536, 131072, 262144, 500001, 524288,
                           1048577, 1048576, 1048581};

    int numTests = sizeof(test) / sizeof(test[0]);
    int numElements;
    size_t freeMem, totalMem;
    CUDA_SAFE_CALL(cudaMemGetInfo(&freeMem, &totalMem));
    unsigned int memNeeded = test[numTests-1] * sizeof(unsigned char);
    while (memNeeded > 0.015 * freeMem) {
         numTests -= 1;    
	 memNeeded = test[numTests-1] * sizeof(unsigned char);
         if(numTests == 0) {
	       fprintf(stderr,
                       "suffixArrayTest: Error, not enough memory to run test\n");
               return retval;
	 }
    }   

    numElements = test[numTests-1] + numTests; // maximum test size
    if(testOptions.skiplongtests)
    {
          numTests -= 3;
    }

    bool oneTest = false;

    if (commandLineArg(numElements, argc, (const char**) argv, "n"))
    {
        oneTest = true;
        numTests = 1;
        test[0] = numElements;
    }

    // Initialize CUDPP
    CUDPPHandle plan;
    CUDPPResult result = CUDPP_SUCCESS;
    CUDPPHandle theCudpp;
    result = cudppCreate(&theCudpp);


    if (result != CUDPP_SUCCESS)
    {
        if (!quiet)
           fprintf(stderr, "Error initializing CUDPP Library\n");
        retval = (oneTest) ? 1 : numTests;
        return retval;
    }

    result = cudppPlan(theCudpp, &plan, config, numElements, 1, 0);

    if(result != CUDPP_SUCCESS)
    {
        if (!quiet)
           fprintf(stderr, "Error in plan creation\n");
        retval = (oneTest) ? 1 : numTests;
        return retval;
    }

    // allocate host memory to store input data
    unsigned char* i_data = new unsigned char[numElements];
    unsigned int* reference = new unsigned int[numElements+3];

    // allocate device memory input and output arrays
    unsigned char* d_idata = (unsigned char *) NULL;
    unsigned int* d_odata = (unsigned int *) NULL;
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_idata,
                              numElements*sizeof(unsigned char)));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_odata,
                              (numElements+1)*sizeof(unsigned int)));
    for(int k=0; k<numTests; k++)
    {
        if(!quiet)
        {
            printf("Running a Suffix Array test of %u %s nodes\n",
                    test[k], datatypeToString(config.datatype));
            fflush(stdout);
        }

        // initialize the input data on the host
        srand(95835);
        for(int j=0; j<test[k]; ++j)
            i_data[j] = (unsigned char)(rand()%128+1);

        CUDA_SAFE_CALL(cudaMemcpy(d_idata, i_data,
                                  sizeof(unsigned char) * test[k],
                                  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemset(d_odata, 0,
                                  sizeof(unsigned int) * (test[k]+1)));

        // allocate host memory to store the output data
        unsigned int* o_data =
                (unsigned int*) malloc(sizeof(unsigned int) * test[k]);
        memset(reference, 0, sizeof(unsigned int) * (test[k]+3));

        computeSaGold(i_data, reference, test[k]);

        // Run the SA
	// run once to avoid timing startup overhead.
        result = cudppSuffixArray(plan, d_idata, d_odata, test[k]);
      
        if (result != CUDPP_SUCCESS)
        {
          if(!quiet)
            printf("Error in cudppSuffixArray call in testSa (make sure your device is at"
                " least compute version 2.0)\n");
          retval = numTests;
        } else {
          timer.reset();
          timer.start();
          for(int i=0; i<testOptions.numIterations; i++)
              cudppSuffixArray(plan, d_idata, d_odata, test[k]);

          CUDA_SAFE_CALL(cudaThreadSynchronize());
          timer.stop();
        }

        CUDA_SAFE_CALL(cudaMemcpy(o_data, d_odata + 1,
                                  sizeof(unsigned int) * test[k],
                                  cudaMemcpyDeviceToHost));
        bool result = compareArrays<unsigned int> (reference, o_data, test[k]);

        free(o_data);

        retval += result ? 0 : 1;
        if(!quiet)
        {
            printf("test %s\n", result ? "PASSED" : "FAILED");
            printf("Average execution time: %f ms\n",
                   timer.getTime() / testOptions.numIterations);
        } else
            printf("\t%10d\t%0.4f\n", test[k],
                   timer.getTime() / testOptions.numIterations);
    }

    result = cudppDestroyPlan(plan);
    if (result != CUDPP_SUCCESS)
    {
        if (!quiet)
           printf("Error destroying CUDPPPlan for Suffix Array\n");
    }

    result = cudppDestroy(theCudpp);
    if (result != CUDPP_SUCCESS)
    {
         if(!quiet)
            printf("Error shutting down CUDPP Library.\n");
    }

    delete [] reference;
    delete [] i_data;
    cudaFree(d_odata);
    cudaFree(d_idata);
    return retval;
}



int testSuffixArray(int argc, const char **argv,
                    const CUDPPConfiguration *configPtr)
{
    testrigOptions testOptions;
    setOptions(argc, argv, testOptions);

    CUDPPConfiguration config;
    config.algorithm = CUDPP_SA;
    config.options = 0;

    if (configPtr != NULL)
    {
        config = *configPtr;
    }
    else
    {
        config.datatype = CUDPP_UCHAR;
    }

    return suffixArrayTest(argc, argv, config, testOptions);
}


// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:



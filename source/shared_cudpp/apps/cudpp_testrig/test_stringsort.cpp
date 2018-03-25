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


int verifyStringSort(unsigned int *valuesSorted,
                     unsigned char* stringVals, size_t numElements,
                     int stringSize, unsigned char termC)
{
    int retval = 0;

    for(unsigned int i = 0; i < numElements-1; ++i)
    {
        unsigned int add1, add2;
        add1 = valuesSorted[i];
        add2 = valuesSorted[i+1];

        unsigned char c1, c2;

        do
        {
            c1 = (stringVals[add1]);
            c2 = (stringVals[add2]);


            add1++;
            add2++;

        }
        while(c1 == c2 && c1 != termC && c2 != termC &&
              add1 < stringSize && add2 < stringSize);

        if (c1 > c2)
        {
            printf("Error comparing index %d to %d (%d > %d) "
                   "(add1 %d add2 %d)\n",
                   i, i+1, c1, c2, valuesSorted[i], valuesSorted[i+1]);
            return 1;
        }

    }
    return retval;
}

int verifyPackedStringSort(unsigned int *valuesSorted,
                           unsigned int* packedStringVals, size_t numElements,
                           unsigned int stringSize, unsigned char termC)
{
    int retval = 0;

    for(unsigned int i = 0; i < numElements-1; ++i)
    {
        unsigned int add1, add2;
        add1 = valuesSorted[i];
        add2 = valuesSorted[i+1];

        unsigned int c1, c2;

        do
        {
            c1 = (packedStringVals[add1]);
            c2 = (packedStringVals[add2]);


            add1++;
            add2++;

        }
        while(c1 == c2 && (c1&255) != termC && (c2&255) != termC &&
              add1 < stringSize && add2 < stringSize);

        if(c1 > c2)
        {
            printf("Error comparing index %d to %d (%d > %d) "
                   "(add1 %d add2 %d)\n",
                   i, i+1, c1, c2, valuesSorted[i], valuesSorted[i+1]);
            return 1;
        }

    }
    return retval;
}

int stringSortTest(CUDPPHandle theCudpp, CUDPPConfiguration config,
                   size_t *tests, unsigned int numTests, size_t numElements,
                   testrigOptions testOptions, bool quiet)
{
    int retval = 0;
    srand(44);

    unsigned int  *h_valuesSorted, *h_valSend, *d_address, *h_valAligned;
    unsigned int *string_length;
    unsigned char *d_stringVals;
    unsigned int *d_packedStringVals;
    unsigned char *stringVals;
    unsigned int *packedStringVals;
    unsigned int *h_alignedKeys;
    unsigned int *d_alignedKeys;;
    config.algorithm = CUDPP_SORT_STRING;
    config.datatype = CUDPP_UINT;
    config.options = CUDPP_OPTION_FORWARD;

    unsigned int maxStringLength = 14;


    unsigned int stringSize = 0;
    unsigned int packedStringSize = 0;
    h_valSend = (unsigned int*)malloc(numElements*sizeof(unsigned int));
    h_valAligned = (unsigned int*)malloc(numElements*sizeof(unsigned int));
    h_alignedKeys = (unsigned int*)malloc(numElements*sizeof(unsigned int));
    h_valuesSorted = (unsigned int*)malloc(numElements*sizeof(unsigned int));
    string_length = (unsigned int*)malloc(numElements*sizeof(unsigned int));
    unsigned int* unique_qualifier_length =
        (unsigned int*) malloc(numElements*sizeof(unsigned int));
    for(unsigned int i=0; i < numElements; ++i)
    {
        int append = i+1;
        unique_qualifier_length[i] = 0;
        while(append > 0)
        {
            append /=255;
            unique_qualifier_length[i]++;
        }

        string_length[i] =
            3 + (rand()%maxStringLength) + unique_qualifier_length[i];
        h_valSend[i] = i == 0 ? 0 : string_length[i-1];
        stringSize += string_length[i];
        packedStringSize += ((string_length[i]+3)>>2);
    }
    stringVals = (unsigned char*) malloc(sizeof(unsigned char)*stringSize);
    packedStringVals =
        (unsigned int*) malloc(sizeof(unsigned int)*(packedStringSize));
    unsigned int index = 0;
    unsigned int aIndex = 0;
    //printf("%lu elements and %d characters\n", numElements, stringSize);
    unsigned int temp = 0;
    unsigned char c;
    for(unsigned int i = 0; i < numElements; ++i)
    {
        unsigned int packedVal = 0;
        unsigned int count = 0;
        h_valAligned[i] = aIndex;

        for(unsigned int j = 0;
            j < (string_length[i]-unique_qualifier_length[i])-1 ;
            j++)
        {
            packedVal = packedVal << 8;
            c = (rand()%254)+1;
            stringVals[index++] = c;
            packedVal += c;
            count++;

            if((count & 3) == 0)
            {
                packedStringVals[aIndex++] = packedVal;
                packedVal = 0;
            }
        }

        int append = i+1;
        for(int k = 0; k < unique_qualifier_length[i]; k++)
        {
            packedVal = packedVal << 8;
            count++;

            if ((append&255) == 0)
                append++;

            stringVals[index++] = (append&255);
            packedVal += (append&255);
            append /=255;
            if((count&3) == 0)
            {
                packedStringVals[aIndex++] = packedVal;
                packedVal = 0;
            }

        }

        stringVals[index++] = 0;
        packedStringVals[aIndex++] = packedVal << 8;


        if( i > 0 )
            h_valSend[i] += h_valSend[i-1];

    }


    for(int i = 0; i < numElements; i++)
    {
        int address = h_valAligned[i];
        h_alignedKeys[i] = packedStringVals[address];
    }


    CUDA_SAFE_CALL(cudaMalloc((void **) &d_alignedKeys,
                              numElements*sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_address,
                              numElements*sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_stringVals,
                              stringSize*sizeof(unsigned char)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_packedStringVals,
                              packedStringSize*sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMemcpy(d_stringVals,
                              stringVals,
                              stringSize*sizeof(unsigned char),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_packedStringVals,
                              packedStringVals,
                              packedStringSize*sizeof(unsigned int),
                              cudaMemcpyHostToDevice));

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

            printf("Running a string sort of %ld keys\n", tests[k]);
            fflush(stdout);
        }

        float totalTime = 0;

        for (int i = 0; i < testOptions.numIterations; i++)
        {
            CUDA_SAFE_CALL( cudaMemcpy(d_address, h_valSend,
                                       tests[k] * sizeof(unsigned int),
                                       cudaMemcpyHostToDevice) );
            CUDA_SAFE_CALL( cudaEventRecord(start_event, 0) );


            cudppStringSort(plan, d_stringVals, d_address, 0, tests[k],
                            stringSize);

            CUDA_SAFE_CALL( cudaEventRecord(stop_event, 0) );
            CUDA_SAFE_CALL( cudaEventSynchronize(stop_event) );

            float time = 0;
            CUDA_SAFE_CALL( cudaEventElapsedTime(&time, start_event,
                                                 stop_event));
            totalTime += time;
        }

        CUDA_CHECK_ERROR("teststringSort - cudppStringSort");

        // copy results

        CUDA_SAFE_CALL( cudaMemcpy(h_valuesSorted,
                                   d_address,
                                   numElements * sizeof(unsigned int),
                                   cudaMemcpyDeviceToHost));

        CUDA_SAFE_CALL( cudaMemcpy(stringVals,
                                   d_stringVals,
                                   stringSize * sizeof(unsigned char),
                                   cudaMemcpyDeviceToHost));

        retval += verifyStringSort(h_valuesSorted,
                                   stringVals, tests[k], stringSize, 0);

        //Verify that the keys make sense
        //TODO: Verify that all strings are in correct order using addresses

        if(!quiet)
        {
            printf("test %s\n", (retval == 0) ? "PASSED" : "FAILED");
            printf("Average execution time: %f ms\n",
                   totalTime / testOptions.numIterations);
        }
        else
        {
            printf("\t%10ld\t%0.4f\n", tests[k],
                   totalTime / testOptions.numIterations);
        }
    }
    printf("\n");


    for (unsigned int k = 0; k < numTests; ++k)
    {
        if(numTests == 1)
            tests[0] = numElements;

        if (!quiet)
        {

            printf("Running a packed string sort of %ld keys\n", tests[k]);
            fflush(stdout);
        }

        float totalTime = 0;


        for (int i = 0; i < testOptions.numIterations; i++)
        {


            CUDA_SAFE_CALL( cudaMemcpy(d_address, h_valAligned,
                                       tests[k] * sizeof(unsigned int),
                                       cudaMemcpyHostToDevice) );
            CUDA_SAFE_CALL( cudaMemcpy(d_alignedKeys, h_alignedKeys,
                                       tests[k] * sizeof(unsigned int),
                                       cudaMemcpyHostToDevice) );
            CUDA_SAFE_CALL( cudaEventRecord(start_event, 0) );


            cudppStringSortAligned(plan, d_alignedKeys, d_address,
                                   d_packedStringVals, tests[k],
                                   packedStringSize);


            CUDA_SAFE_CALL( cudaEventRecord(stop_event, 0) );
            CUDA_SAFE_CALL( cudaEventSynchronize(stop_event) );

            float time = 0;
            CUDA_SAFE_CALL( cudaEventElapsedTime(&time, start_event,
                                                 stop_event));
            totalTime += time;
        }

        CUDA_CHECK_ERROR("teststringSort - cudppStringSort");


        // copy results

        CUDA_SAFE_CALL( cudaMemcpy(h_valuesSorted,
                                   d_address,
                                   numElements * sizeof(unsigned int),
                                   cudaMemcpyDeviceToHost));

        CUDA_SAFE_CALL( cudaMemcpy(packedStringVals,
                                   d_packedStringVals,
                                   packedStringSize * sizeof(unsigned char),
                                   cudaMemcpyDeviceToHost));

        retval += verifyPackedStringSort(h_valuesSorted, packedStringVals,
                                         tests[k], packedStringSize, 0);

        if(!quiet)
        {
            printf("test %s\n", (retval == 0) ? "PASSED" : "FAILED");
            printf("Average execution time: %f ms\n",
                   totalTime / testOptions.numIterations);
        }
        else
        {
            printf("\t%10ld\t%0.4f\n", tests[k],
                   totalTime / testOptions.numIterations);
        }
    }


    CUDA_CHECK_ERROR("after stringsort");

    result = cudppDestroyPlan(plan);


    if (result != CUDPP_SUCCESS)
    {
        printf("Error destroying CUDPPPlan for StringSort\n");
        retval = numTests;
    }

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    cudaFree(d_address);
    cudaFree(d_stringVals);
    cudaFree(d_alignedKeys);
    cudaFree(d_packedStringVals);

    free(h_valSend);
    free(h_valAligned);
    free(h_alignedKeys);
    free(h_valuesSorted);
    free(unique_qualifier_length);
    free(stringVals);
    free(packedStringVals);
    free(string_length);;

    return retval;
}


/**
 * testStringSort tests cudpp's merge sort
 * Possible command line arguments:
 * - -n=#, number of elements in sort
 * @param argc Number of arguments on the command line, passed
 * directly from main
 * @param argv Array of arguments on the command line, passed directly
 * from main
 * @param configPtr Configuration for scan, set by caller
 * @return Number of tests that failed regression (0 for all pass)
 * @see cudppSort
 */
int testStringSort(int argc, const char **argv,
                   const CUDPPConfiguration *configPtr)
{

    int cmdVal;
    int retval = 0;

    bool quiet = checkCommandLineFlag(argc, argv, "quiet");
    testrigOptions testOptions;
    setOptions(argc, argv, testOptions);

    CUDPPConfiguration config;
    config.algorithm = CUDPP_SORT_STRING;
    config.datatype = CUDPP_UINT;


    size_t test[] = {39, 128, 256, 512, 513, 1000, 1024, 1025, 32768,
                     45537, 65536, 131072, 262144, 500001, 524288,
                     1048577, 1048576, 1048581, 2097152, 4194304};

    int numTests = sizeof(test)/sizeof(test[0]);

    // small GPUs are susceptible to running out of memory,
    // restrict the tests to only those where we have enough
    size_t freeMem, totalMem;
    CUDA_SAFE_CALL(cudaMemGetInfo(&freeMem, &totalMem));
    printf("freeMem: %d, totalMem: %d\n", int(freeMem), int(totalMem));
    while (freeMem < 90 * test[numTests - 1]) // 90B/item appears to be enough
    {
        numTests--;
        if (numTests <= 0)
        {
            // something has gone very wrong
            printf("Not enough free memory to run any stringsort tests.\n");
            return -1;
        }
    }


    size_t numElements = test[numTests - 1];


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

    retval = stringSortTest(theCudpp, config, test, numTests, numElements,
                            testOptions, quiet);
    result = cudppDestroy(theCudpp);
    return retval;

}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

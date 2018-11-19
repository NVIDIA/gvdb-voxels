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
 * test_rand.cu
 *
 * @brief Host testrig routines to exercise cudpp's rand functionality.
 */

#include <cstdio>
#include <cstring>
#include <cuda_runtime_api.h>
#include "cudpp_testrig_options.h"
#include "cudpp_testrig_utils.h"
#include "cuda_util.h"
#include "stopwatch.h"
#include "findfile.h"
#include "commandline.h"
#include "common_config.h"

using namespace cudpp_app;
 
//windows uses \ as the path, so we must adjust our original path for this
//also if you're using Visual Studio, the path is only two directories up rather than three
#if defined (__linux__) || defined (__APPLE__) || defined (MACOSX)
    #define MD5_DEFAULT_PATH "../../../apps/data/"
#else
    #define MD5_DEFAULT_PATH "..\\..\\apps\\data\\"
    #define MD5_DEFAULT_PATH2  "..\\..\\..\\apps\\data\\"
#endif

//#define MD5_WRITE_BINARY
//#define MD5_WRITE_ASCII
#define MD5_TEST_BINARY



///////////////////////////////////////////////////////////////////////
//globals
char regDir[100];       //a string holding the directory name of the regression data

///////////////////////////////////////////////////////////////////////
//declaration forward

bool file_exists(const char * filename)
{
    if (FILE * file = fopen(filename, "r"))
    {
        fclose(file);
        return true;
    }
    return false;
}

//this function makes the proper path file name
void constructFileName(char * fileName, testrigOptions & testOptions, unsigned int size, const char * path)
{
    //check first to see if user has inputted a string name
    if(testOptions.dir == "")
    {
        //use default location for the files
        strcpy(fileName, path);

#if defined(MD5_WRITE_BINARY) || defined(MD5_TEST_BINARY) 
        sprintf(fileName, "%smd5_regression_%u.dat",fileName, size); 
#else
        sprintf(fileName, "%smd5_regression_%u.txt",fileName, size); 
#endif
    }
    else
    {
        //use user defined path
        sprintf(fileName, "%smd5_regression_%u.dat",testOptions.dir.c_str(), size);
    }
} //end constructFileName

bool searchForFile(unsigned int numElements, FILE ** randFile, char * path, testrigOptions & testOptions, bool quiet)
{
    char fileName[100];
    char fullFileName[200];


    sprintf(fileName,"md5_regression_%u.dat", numElements);

    constructFileName(fullFileName, testOptions, numElements, MD5_DEFAULT_PATH);
    strcpy(path, fullFileName);

    //search and see which one of them exists
    if(file_exists(fileName))
    {
        if(!quiet)
            printf("%s found in local directory!  Testing against this file.\n", fileName);
        *randFile = fopen(fileName, "rb");
        return true;
    }
    if(file_exists(fullFileName))
    {
        *randFile = fopen(fullFileName,"rb");
        return true;
    }

    // search in CMAKE-configured app data path
    constructFileName(fullFileName, testOptions, numElements, CUDPP_APP_DATA_DIR);
    strcpy(path, fullFileName);
    if(file_exists(fullFileName))
    {
        *randFile = fopen(fullFileName,"rb");
        return true;
    }

    /*
    this part added for windows: path in Visual Studio and command line is different!  
    */
#if defined(_WIN32) || defined(WIN32)
    constructFileName(fullFileName, testOptions, numElements, MD5_DEFAULT_PATH2);
    strcpy(path, fullFileName);
    if(file_exists(fullFileName))
    {
        *randFile = fopen(fullFileName,"rb");
        return true;
    }
#endif

    //final desperate attempt, if we can't find it, try to find it via the data directory
    char dataPath[100];
    if (!findDir("cudpp", "data", dataPath) )
        return false;

    constructFileName(fullFileName, testOptions, numElements, dataPath);
    if(file_exists(fullFileName))
    {
        if(!quiet) 
            printf("the file was found in the dir: %s\n", dataPath);
        *randFile = fopen(fullFileName,"rb");
        return true;
    }
    return false;
}

/////////////////////////////////END DIRECTORY FINDING/////////////////////////

int
testRandMD5(int argc, const char** argv)
{
    int retval =0;
    unsigned int seed = 9999;   //constant seed
    testrigOptions testOptions;
    setOptions(argc, argv, testOptions);

    bool quiet = checkCommandLineFlag(argc, (const char**) argv, "quiet");

    unsigned int test[] = {39, 128, 256, 512, 1000, 1024, 1025, 32768, 45537, 65536, 131072,
        262144, 500001, 524288, 1048577, 1048576, 1048581, 2097152, 4194304, 8388608};

    int numTests = sizeof(test) / sizeof(test[0]);

    unsigned int *md5GPU1;

    unsigned int * md5GPUHost1;
    unsigned int * md5FromFile;

    FILE * randFile = NULL;
    char fileName[200] = "";

    //initialize the CUDPP config
    CUDPPConfiguration config;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_UINT;
    config.algorithm = CUDPP_RAND_MD5;
    config.options = 0;

    CUDPPHandle randPlan = 0;
    CUDPPResult result;

    CUDPPHandle theCudpp;
    result = cudppCreate(&theCudpp);
    if(result != CUDPP_SUCCESS)
    {
        printf("Error initializing CUDPP Library.\n");
        retval = numTests;
        return retval;
    }

    if(!quiet && testOptions.dir == "")
    {
#if defined (__linux__) || defined (__APPLE__) || defined (MACOSX)
        printf("MD5 Rand test: no user specified path for rand testing.  Looking in local directory and ");
        printf(MD5_DEFAULT_PATH);
        printf(" for files.\n");
#else
        printf("MD5 Rand test: no user specified path for rand testing.  Looking in local directory, ");
        printf(MD5_DEFAULT_PATH);
        printf(" and ");
        printf(MD5_DEFAULT_PATH2);
        printf(" for files.\n");
#endif
    }

    StopWatch timer;

    for(int i=0; i<numTests; i++)
    {
        //here we read in the file to test the bits against
        randFile = NULL;

        if(!searchForFile(test[i],&randFile, fileName, testOptions, quiet))
        {
            if(!quiet)
            {
                printf("No regression test file found for size %u\n", test[i]);
            }
            if(randFile != NULL) fclose(randFile);
            continue;
        }
        else
            if(!quiet)
                printf("Found regression file for size %u.\n",test[i]);

        if(randFile == NULL)
        {
            if (!quiet)
            {
                printf("error opening file: %s. Is it a binary data file?\n",fileName);
                retval++;
            }
        }

        CUDA_SAFE_CALL(cudaMalloc((void**)&md5GPU1, test[i] * sizeof(unsigned int)));
        md5GPUHost1 = (unsigned int *) malloc(sizeof(unsigned int) * test[i]);
        md5FromFile = (unsigned int *) malloc(sizeof(unsigned int) * test[i]);

        for(unsigned int j=0; j<test[i]; j++)
            md5FromFile[j] = 0;
        result = cudppPlan(theCudpp, &randPlan, config, test[i], 1,0);

        if (CUDPP_SUCCESS != result)
        {
            printf("Error creating CUDPPPlan\n");
            exit(-1);
        }

        //tell the user the block size if not quiet
        if(!quiet)
        {
             //figure out how many elements are needed in this array
            unsigned int devOutputsize = test[i] / 4;
            devOutputsize += (test[i] %4 == 0) ? 0 : 1; //used for overflow

            //now figure out block size
            unsigned int blockSize = 128;
            if(devOutputsize < 128) blockSize = devOutputsize;

            unsigned int n_blocks = 
                devOutputsize/blockSize + (devOutputsize%blockSize == 0 ? 0:1);  

            printf("Generating %u random numbers using %u %u-thread blocks\n", test[i], n_blocks, blockSize);
        }
        
        cudppRandSeed(randPlan, seed);
        
        timer.reset();
        timer.start();
        cudppRand(randPlan, md5GPU1, test[i]);
        timer.stop();
        
        if(quiet)
        {
            // print out basic information here
            printf("\t%10u\t%0.4f%5c\n", test[i], timer.getTime(),' ');
        }
        else
        {
            printf("%u pseudorandom numbers generated in %f ms\n", test[i], timer.getTime());
        }
        // copy the data back
        CUDA_SAFE_CALL(cudaMemcpy(md5GPUHost1, md5GPU1, sizeof(unsigned int) * test[i],cudaMemcpyDeviceToHost));


    #if defined(MD5_WRITE_BINARY) || defined(MD5_WRITE_ASCII)
        searchForFile(test[i],&randFile,fileName, testOptions, quiet);
        printf("writing file to: %s\n\n",fileName);
        if(randFile == NULL)
            printf("something wrong here!\n");
        if(randFile)
            fclose(randFile);
#ifdef MD5_WRITE_BINARY
        randFile = fopen(fileName, "wb");
#else
        randFile = fopen(fileName, "w");
#endif

        if(randFile == NULL)
        {
            printf("something wrong here!\n%s\n", fileName);
        }
        for(int j=0; j<test[i]; j++)
        {
#ifdef MD5_WRITE_BINARY  
            fwrite(&md5GPUHost1[j], sizeof(unsigned int), 1, randFile);
#else 
            fprintf(randFile, "%u\n", md5GPUHost1[j]);
#endif
        }
#else
        
        fread(md5FromFile, sizeof(unsigned int), test[i], randFile);

        //check here to make sure it is correct or not
        bool cleanTest = true;
        int numPass = 0;
        for(unsigned int j=0; j<test[i]; j++)
        {
            if(md5GPUHost1[j] != md5FromFile[j])
            {
 //            printf("test failed on %d:\t%u %u\n", test[i], md5GPUHost1[j], md5FromFile[j]);
               cleanTest = false;
            }//end if
            else
                numPass++;
        }   
        
        if(!cleanTest)
        {
            retval++;
            if(!quiet)
            {
                printf("Passed: %u / %u\n", numPass, test[i]);
                printf("Test FAILED\n\n");
            }
        }
        else
        {
            if(!quiet)
            {
                printf("Test PASSED\n\n");
            }
         }

#endif

        //at the end of the loop, we clean up
        CUDA_SAFE_CALL(cudaFree(md5GPU1));
        free(md5GPUHost1);
        free(md5FromFile);
        if(randFile != NULL) fclose(randFile);
        result = cudppDestroyPlan(randPlan);
        if (CUDPP_SUCCESS != result)
        {
            printf("Error destroying CUDPPPlan\n");
            exit(-1);
        }
    }

    result = cudppDestroy(theCudpp);
    if (CUDPP_SUCCESS != result)
    {
        printf("Error shutting down CUDPP Library.\n");
        exit(-1);
    }

    if(!quiet)
        printf("%u total tests failed in rand regression test.\n", retval);

    return retval;
}


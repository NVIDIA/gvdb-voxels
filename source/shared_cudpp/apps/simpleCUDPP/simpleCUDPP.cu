// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/*
 * This is a basic example of how to use the CUDPP library.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "cudpp.h"

#include <string>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

extern "C" 
void computeSumScanGold( float *reference, const float *idata, 
                        const unsigned int len,
                        const CUDPPConfiguration &config);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    runTest( argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    if (argc > 1) {
        std::string arg = argv[1];
        size_t pos = arg.find("=");
        if (arg.find("device") && pos != std::string::npos) {
            dev = atoi(arg.c_str() + (pos + 1));
        }
    }
    if (dev < 0) dev = 0;
    if (dev > deviceCount-1) dev = deviceCount - 1;
    cudaSetDevice(dev);

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
    {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
               prop.name, (int)prop.totalGlobalMem, (int)prop.major, 
               (int)prop.minor, (int)prop.clockRate);
    }

    unsigned int numElements = 32768;
    unsigned int memSize = sizeof( float) * numElements;

    // allocate host memory
    float* h_idata = (float*) malloc( memSize);
    // initalize the memory
    for (unsigned int i = 0; i < numElements; ++i) 
    {
        h_idata[i] = (float) (rand() & 0xf);
    }

    // allocate device memory
    float* d_idata;
    cudaError_t result = cudaMalloc( (void**) &d_idata, memSize);
    if (result != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(result));
        exit(-1);
    }
    
    // copy host memory to device
    result = cudaMemcpy( d_idata, h_idata, memSize, cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(result));
        exit(-1);
    }
     
    // allocate device memory for result
    float* d_odata;
    result = cudaMalloc( (void**) &d_odata, memSize);
    if (result != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(result));
        exit(-1);
    }

    // Initialize the CUDPP Library
    CUDPPHandle theCudpp;
    cudppCreate(&theCudpp);

    CUDPPConfiguration config;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_FLOAT;
    config.algorithm = CUDPP_SCAN;
    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
    
    CUDPPHandle scanplan = 0;
    CUDPPResult res = cudppPlan(theCudpp, &scanplan, config, numElements, 1, 0);  

    if (CUDPP_SUCCESS != res)
    {
        printf("Error creating CUDPPPlan\n");
        exit(-1);
    }

    // Run the scan
    res = cudppScan(scanplan, d_odata, d_idata, numElements);
    if (CUDPP_SUCCESS != res)
    {
        printf("Error in cudppScan()\n");
        exit(-1);
    }

    // allocate mem for the result on host side
    float* h_odata = (float*) malloc( memSize);
    // copy result from device to host
    result = cudaMemcpy( h_odata, d_odata, memSize, cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(result));
        exit(-1);
    }
    
    // compute reference solution
    float* reference = (float*) malloc( memSize);
    computeSumScanGold( reference, h_idata, numElements, config);

    // check result
    bool passed = true;
    for (unsigned int i = 0; i < numElements; i++)
        if (reference[i] != h_odata[i]) passed = false;
        
    printf( "Test %s\n", passed ? "PASSED" : "FAILED");

    res = cudppDestroyPlan(scanplan);
    if (CUDPP_SUCCESS != res)
    {
        printf("Error destroying CUDPPPlan\n");
        exit(-1);
    }

    // shut down the CUDPP library
    cudppDestroy(theCudpp);
    
    free( h_idata);
    free( h_odata);
    free( reference);
    cudaFree(d_idata);
    cudaFree(d_odata);
}

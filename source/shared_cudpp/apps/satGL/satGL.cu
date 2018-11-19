// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "cudpp.h"
#include "cuda_util.h"

#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  include <windows.h>
#endif
#include <cuda_gl_interop.h>

#include <cuda.h>
#if CUDA_VERSION < 3000
#define USE_CUDA_GRAPHICS_INTEROP 0
#else
#define USE_CUDA_GRAPHICS_INTEROP 1
#endif


int width = 0;
int height = 0;
size_t d_satPitch = 0;
size_t d_satPitchInElements = 0;
CUDPPConfiguration config = { CUDPP_SCAN, 
                              CUDPP_ADD, 
                              CUDPP_FLOAT, 
                              CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE };
CUDPPHandle theCudpp;
CUDPPHandle scanPlan;

float *SATs[2][3];
cudaEvent_t timerStart, timerStop;

extern "C"
__host__ void initialize(int width, int height)
{   
    size_t dpitch = width * sizeof(float);

    CUDA_CHECK_ERROR("Before CUDA initialization");

    CUDA_SAFE_CALL( cudaMallocPitch( (void**) &SATs[0][0], &d_satPitch, dpitch, height));
    CUDA_SAFE_CALL( cudaMallocPitch( (void**) &SATs[0][1], &d_satPitch, dpitch, height));
    CUDA_SAFE_CALL( cudaMallocPitch( (void**) &SATs[0][2], &d_satPitch, dpitch, height));
    CUDA_SAFE_CALL( cudaMallocPitch( (void**) &SATs[1][0], &d_satPitch, dpitch, height));
    CUDA_SAFE_CALL( cudaMallocPitch( (void**) &SATs[1][1], &d_satPitch, dpitch, height));
    CUDA_SAFE_CALL( cudaMallocPitch( (void**) &SATs[1][2], &d_satPitch, dpitch, height));

    d_satPitchInElements = d_satPitch / sizeof(float);

    // Initialize CUDPP
    cudppCreate(&theCudpp);
    
    if (CUDPP_SUCCESS != cudppPlan(theCudpp, &scanPlan, config, width, height, d_satPitchInElements))
    {
        printf("Error creating CUDPPPlan.\n");
    }

    CUDA_SAFE_CALL( cudaEventCreate(&timerStart) );
    CUDA_SAFE_CALL( cudaEventCreate(&timerStop) );
    
}

extern "C"
__host__ void finalize()
{
    if (CUDPP_SUCCESS != cudppDestroyPlan(scanPlan))
    {
        printf("Error destroying CUDPPPlan.\n");
    }

    // shut down CUDPP
    if (CUDPP_SUCCESS != cudppDestroy(theCudpp))
    {
        printf("Error destroying CUDPP.\n");
    }

    CUDA_SAFE_CALL( cudaEventDestroy(timerStart) );
    CUDA_SAFE_CALL( cudaEventDestroy(timerStop) );
}

__global__ 
void deinterleaveRGBA8toFloat32(float *out_R,
                                float *out_G,
                                float *out_B,
                                unsigned int *in_RGBA8, 
                                size_t pitch,
                                size_t width,
                                size_t height)
{
    unsigned int xIndex = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    unsigned int yIndex = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;

    //if (xIndex < width && yIndex < height)
    //{
        unsigned int indexIn  = __mul24(__mul24(blockDim.x, gridDim.x), yIndex) + xIndex;
        unsigned int indexOut = __mul24(pitch, yIndex) + xIndex;

        unsigned int pixel_int = in_RGBA8[indexIn];
        //uchar4 pixel = *((uchar4*)&pixel_int);
        //pixel_int >>= 8;
        out_R[indexOut] = (float)(pixel_int & 255) / 255.0f;
        pixel_int >>= 8;
        out_G[indexOut] = (float)(pixel_int & 255) / 255.0f;
        pixel_int >>= 8;
        out_B[indexOut] = (float)(pixel_int & 255) / 255.0f;
        
    //}                       
}


__global__ 
void interleaveFloat32toRGBAfp32(float4 *out,
                                 float *in_R,
                                 float *in_G,
                                 float *in_B,
                                 size_t pitch,
                                 size_t width,
                                 size_t height)
{
    unsigned int xIndex = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    unsigned int yIndex = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;

    //if (xIndex < width && yIndex < height)
    //{
        unsigned int indexIn  = __mul24(pitch, yIndex) + xIndex;
        unsigned int indexOut = __mul24(__mul24(blockDim.x, gridDim.x), yIndex) + xIndex;

        float4 pixel;
        pixel.x = in_R[indexIn];
        pixel.y = in_G[indexIn];
        pixel.z = in_B[indexIn];
        pixel.w = 0;
        out[indexOut] = pixel;
    //}                       
}

template <typename T, int block_width, int block_height>
__global__ void transpose(T *out_R,
                          T *out_G,
                          T *out_B,
                          T *in_R,
                          T *in_G,                          
                          T *in_B,
                          size_t pitch,
                          size_t width,
                          size_t height)
{
    __shared__ T block[3][block_width*block_height];

    unsigned int xBlock = __mul24(blockDim.x, blockIdx.x);
    unsigned int yBlock = __mul24(blockDim.y, blockIdx.y);
    unsigned int xIndex = xBlock + threadIdx.x;
    unsigned int yIndex = yBlock + threadIdx.y;
    unsigned int index_out, index_transpose;

    if (xIndex < width && yIndex < height)
    {
        // load block into smem
        unsigned int index_in  = 
                __mul24(pitch, yBlock + threadIdx.y) + 
                xBlock + threadIdx.x;

        unsigned int index_block = __mul24(threadIdx.y, block_width) + threadIdx.x;
        block[0][index_block] = in_R[index_in];
        block[1][index_block] = in_G[index_in];
        block[2][index_block] = in_B[index_in];
       
        index_transpose = __mul24(threadIdx.x, block_width) + threadIdx.y;
        
        index_out = __mul24(pitch, xBlock + threadIdx.y) +
            yBlock + threadIdx.x;
    }

    __syncthreads();

    if (xIndex < width && yIndex < height)
    {
        // write it out (transposed) into the new location
        out_R[index_out] = block[0][index_transpose];
        out_G[index_out] = block[1][index_transpose];
        out_B[index_out] = block[2][index_transpose];
    }
}


////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
extern "C"
#if USE_CUDA_GRAPHICS_INTEROP
void process( cudaGraphicsResource **pgres, int width, int height, int radius) 
#else
void process( int pbo_in, int pbo_out, int width, int height, int radius) 
#endif
{
    unsigned int *in_data;
    float *out_data;

#if USE_CUDA_GRAPHICS_INTEROP
    CUDA_SAFE_CALL(cudaGraphicsMapResources(2, pgres, 0));
    size_t size;
    CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&in_data, &size, pgres[0]));
    CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&out_data, &size, pgres[1]));
#else
    CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&in_data, pbo_in));
    CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&out_data, pbo_out));
#endif

    dim3 block(16, 16, 1);
    dim3 grid(width / block.x, height / block.y, 1);

    CUDA_SAFE_CALL( cudaEventRecord(timerStart) );    

    deinterleaveRGBA8toFloat32<<<grid, block, 0>>>(SATs[0][0], 
                                                   SATs[0][1], 
                                                   SATs[0][2], 
                                                   in_data, 
                                                   d_satPitchInElements,
                                                   width, height);

    // scan rows
    cudppMultiScan(scanPlan, SATs[1][0], SATs[0][0], width, height);
    cudppMultiScan(scanPlan, SATs[1][1], SATs[0][1], width, height);
    cudppMultiScan(scanPlan, SATs[1][2], SATs[0][2], width, height);

    // transpose so columns become rows
    transpose<float, 16, 16><<<grid, block, 0>>>
        (SATs[0][0], SATs[0][1], SATs[0][2], 
         SATs[1][0], SATs[1][1], SATs[1][2], d_satPitchInElements, width, height);

    // scan columns
    cudppMultiScan(scanPlan, SATs[1][0], SATs[0][0], width, height);
    cudppMultiScan(scanPlan, SATs[1][1], SATs[0][1], width, height);
    cudppMultiScan(scanPlan, SATs[1][2], SATs[0][2], width, height);
    
    interleaveFloat32toRGBAfp32<<<grid, block, 0>>>((float4*)out_data, 
                                                    SATs[1][0], 
                                                    SATs[1][1], 
                                                    SATs[1][2], 
                                                    d_satPitchInElements,
                                                    width, height);

    CUDA_SAFE_CALL(cudaEventRecord(timerStop));

#if USE_CUDA_GRAPHICS_INTEROP
    CUDA_SAFE_CALL(cudaGraphicsUnmapResources(2, pgres, 0));
#else
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject( pbo_in));
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject( pbo_out));
#endif

    float ms;
    CUDA_SAFE_CALL( cudaEventSynchronize(timerStop) );
    CUDA_SAFE_CALL( cudaEventElapsedTime(&ms, timerStart, timerStop) );
    char msg[100];
    sprintf(msg, "CUDPP Summed-Area Table: %0.3f ms to create SAT", ms);
    glutSetWindowTitle(msg);
    
    CUDA_CHECK_ERROR("process");
}


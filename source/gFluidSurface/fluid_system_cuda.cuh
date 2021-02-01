//-----------------------------------------------------------------------------
// FLUIDS v.3 - SPH Fluid Simulator for CPU and GPU
// Copyright (C) 2012-2013. Rama Hoetzlein, http://fluids3.com
//
// NVIDIA(R) GVDB VOXELS
// Copyright 2017 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#ifndef DEF_KERN_CUDA
	#define DEF_KERN_CUDA

	#include <curand.h>
	#include <curand_kernel.h>
	#include <stdio.h>
	#include <math.h>

	#define CUDA_KERNEL
	#include "fluid.h"

	#define EPSILON				0.00001f
	#define GRID_UCHAR			0xFF
	#define GRID_UNDEF			4294967295
	#define TOTAL_THREADS		1000000
	#define BLOCK_THREADS		256
	#define MAX_NBR				80		
	#define FCOLORA(r,g,b,a)	( (uint((a)*255.0f)<<24) | (uint((b)*255.0f)<<16) | (uint((g)*255.0f)<<8) | uint((r)*255.0f) )

	typedef unsigned int		uint;
	typedef unsigned short int	ushort;
	typedef unsigned char		uchar;
	
	extern "C" {
		__global__ void insertParticles ( int pnum );		
		__global__ void countingSortFull ( int pnum );		
		__global__ void computeQuery ( int pnum );	
		__global__ void computePressure ( int pnum );		
		__global__ void computeForce ( int pnum );	
		__global__ void advanceParticles ( float time, float dt, float ss, int numPnts );
		__global__ void emitParticles ( float frame, int emit, int numPnts );
		__global__ void randomInit ( int seed, int numPnts );
		__global__ void sampleParticles ( float* brick, uint3 res, float3 bmin, float3 bmax, int numPnts, float scalar );	
		__global__ void prefixFixup ( uint *input, uint *aux, int len);
		__global__ void prefixSum ( uint* input, uint* output, uint* aux, int len, int zeroff );
		__global__ void countActiveCells ( int pnum );		
	}

#endif

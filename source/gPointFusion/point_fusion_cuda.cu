//----------------------------------------------------------------------------------
//
// FLUIDS v.3 - SPH Fluid Simulator for CPU and GPU
// Copyright (C) 2012-2013. Rama Hoetzlein, http://fluids3.com
//
// BSD 3-clause:
// Redistribution and use in source and binary forms, with or without modification, 
// are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this 
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this 
//    list of conditions and the following disclaimer in the documentation and/or 
//    other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors may 
//    be used to endorse or promote products derived from this software without specific 
//   prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
// SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT 
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) 
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR 
// TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//----------------------------------------------------------------------------------

#define CUDA_KERNEL
#include "point_fusion_cuda.cuh"
#include "cutil_math.h"			// cutil32.lib
#include <string.h>
#include <assert.h>

struct ALIGN(16) Obj {
	float3		pos;
	float3		size;
	float3		loc;
	uint		clr;
};

struct ALIGN(16) ScanInfo {
	int*		objGrid;
	int*		objCnts;
	Obj*		objList;
	float3*		pntList;
	uint*		pntClrs;
	int3		gridRes;
	float3		gridSize;	
	float3		cams;
	float3		camu;
	float3		camv;
	uint*		rnd_seeds;
};
__device__ ScanInfo		scan;
__device__ int			pntout;

// Generate random unsigned int in [0, 2^24)
static __host__ __device__ __inline__ unsigned int lcg(unsigned int &prev)
{
  const unsigned int LCG_A = 1664525u;
  const unsigned int LCG_C = 1013904223u;
  prev = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
}
static __host__ __device__ __inline__ float rnd(unsigned int &prev)
{
  return ((float) lcg(prev) / (float) 0x01000000);
}

// Get view ray
inline __device__ float3 getViewRay ( float x, float y )
{
	float3 v = x*scan.camu + y*scan.camv + scan.cams;  
	return normalize(v);
}

#define NOHIT			1.0e10f

// Ray box intersection
inline __device__ float3 rayBoxIntersect ( float3 rpos, float3 rdir, float3 vmin, float3 vmax )
{
	register float ht[8];
	ht[0] = (vmin.x - rpos.x)/rdir.x;
	ht[1] = (vmax.x - rpos.x)/rdir.x;
	ht[2] = (vmin.y - rpos.y)/rdir.y;
	ht[3] = (vmax.y - rpos.y)/rdir.y;
	ht[4] = (vmin.z - rpos.z)/rdir.z;
	ht[5] = (vmax.z - rpos.z)/rdir.z;
	ht[6] = fmax(fmax(fmin(ht[0], ht[1]), fmin(ht[2], ht[3])), fmin(ht[4], ht[5]));
	ht[7] = fmin(fmin(fmax(ht[0], ht[1]), fmax(ht[2], ht[3])), fmax(ht[4], ht[5]));	
	ht[6] = (ht[6] < 0 ) ? 0.0 : ht[6];
	return make_float3( ht[6], ht[7], (ht[7]<ht[6] || ht[7]<0) ? NOHIT : 0 );
}

#define COLOR(r,g,b)	( (uint((b)*255.0f)<<16) | (uint((g)*255.0f)<<8) | uint((r)*255.0f) ) 

float3 __device__ __inline__ jitter_sample ()
{	 
	uint index = (threadIdx.y % 128) * 128 + (threadIdx.x % 128);
    unsigned int seed  = scan.rnd_seeds[ index ]; 
    float uu = rnd( seed );
    float vv = rnd( seed );
	float ww = rnd( seed );   
	scan.rnd_seeds[ index ] = seed;
    return make_float3(uu,vv,ww);
}

extern "C" __global__ void scanBuildings ( float3 pos, int3 res, int num_obj, float tmax )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= res.x || y >= res.y ) return;

	float3 jit = jitter_sample();
	float3 dir = getViewRay( float(x+jit.x)/float(res.x), float(y+jit.y)/float(res.y) );
	
	int gcell = int(pos.z/scan.gridSize.y) * scan.gridRes.x + int(pos.x/scan.gridSize.x);
	if ( gcell < 0 || gcell > scan.gridRes.x*scan.gridRes.y)  return;

	Obj* bldg;
	float3 t, tnearest;
	uint clr = 0;

	tnearest.x = NOHIT;	

	//for (int n=0; n < scan.objCnts[gcell]; n++) {
//		bldg = scan.objList + (scan.objGrid[gcell] + n);

	for (int n=0; n < num_obj; n++) {
		bldg = scan.objList + n;
		if ( bldg != 0 ) {
			t = rayBoxIntersect ( pos, dir, bldg->pos, bldg->pos + bldg->size );
			if ( t.x < tnearest.x && t.x < tmax && t.z != NOHIT ) {
				tnearest = t;
				clr = bldg->clr;
			}
		}
	}
	if ( tnearest.x == NOHIT) { scan.pntList[ y*res.x + x] = make_float3(0,0,0); return; }

	atomicAdd(&pntout, 1);
	
	scan.pntList[ y*res.x + x] = pos + tnearest.x * dir;
	scan.pntClrs[ y*res.x + x] = clr;	
}




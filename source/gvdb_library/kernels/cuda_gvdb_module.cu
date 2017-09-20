
//--------------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2017, NVIDIA Corporation. 
//
// Redistribution and use in source and binary forms, with or without modification, 
// are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer 
//    in the documentation and/or  other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived 
//    from this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
// BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
// SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE 
// OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// Version 1.0: Rama Hoetzlein, 5/1/2017
//----------------------------------------------------------------------------------
// File: cuda_gvdb_raycast.cu
//
// GVDB Raycasting
// - RayDeep		- deep volume sampling
// - RaySurface		- surface hit raycast
// - RayLevelSet	- level set raycast
// - Raytrace		- raytrace a bundle of arbitrary rays
// - Section3D		- 3D cross-section render
// - Section2D		- 2D cross-section render
//-----------------------------------------------

#include <stdio.h>
#include "cuda_math.cuh"

//----------------------------------------- GVDB Data Structure
#define CUDA_PATHWAY

#include "cuda_gvdb_scene.cuh"		// GVDB Scene
#include "cuda_gvdb_nodes.cuh"		// GVDB Node structure
#include "cuda_gvdb_geom.cuh"		// GVDB Geom helpers
#include "cuda_gvdb_dda.cuh"		// GVDB DDA 
#include "cuda_gvdb_raycast.cuh"	// GVDB Raycasting
//-----------------------------------------

// Operator functions
#include "cuda_gvdb_operators.cuh"

// Particle functions
#include "cuda_gvdb_particles.cuh"


inline __device__ float4 performPhongShading( float4& hclr, float3 hit, float3 norm, float3 rdir )
{
	float diff = 1.0f;
	float amb = 0.0f;
  
	// shadow ray
	if (SCN_SHADOWAMT > 0) 
	{  
		float3 lightdir = normalize(scn.light_pos - hit);
		float3 H = normalize(scn.campos - hit + lightdir);
    
		float ndotl = dot(norm, lightdir);
		diff = max(0.0f, ndotl) * SCN_SHADOWAMT;
		amb = (1.0f - SCN_SHADOWAMT);
		
		rdir = make_float3(NOHIT, NOHIT, NOHIT);
		rayCast ( SCN_SHADE, gvdb.top_lev, 0, hit + norm*gvdb.voxelsize*2.0, lightdir, rdir, norm, hclr, (SCN_SHADE==SHADE_VOXEL) ? raySurfaceVoxelBrick : raySurfaceTrilinearBrick );
		diff *= (rdir.z == NOHIT) ? 1 : 0;		
	}
	return make_float4( fxyz(hclr) * (diff + amb), 1.0 );
}

// Raytracing functions
extern "C" __global__ void gvdbRayDeep ( uchar4* outBuf )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= scn.width || y >= scn.height ) return;

	float4 clr = make_float4(0,0,0,1);
	float3 hit = make_float3(0,0, NOHIT);
	float3 norm;
	float3 rdir = getViewRay(float(x + 0.5f) / float(scn.width), float(y + 0.5f) / float(scn.height));

	// ray deep sampling	
	rayCast ( SHADE_VOLUME, gvdb.top_lev, 0, scn.campos, rdir, hit, norm, clr, rayDeepBrick );
	clr = lerp4 ( SCN_BACKCLR, clr, 1.0-clr.w );	// final color
	
	outBuf [ y*scn.width + x ] = make_uchar4( clr.x*255, clr.y*255, clr.z*255, (1.0-clr.w)*255 );	
}



// Render the volume data by raycasting

extern "C" __global__ void gvdbRaySurfaceVoxel ( uchar4* outBuf )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= scn.width || y >= scn.height ) return;
		
	float3 hit = make_float3(NOHIT,NOHIT,NOHIT);	
	float3 norm;
	float3 rdir = getViewRay(float(x + 0.5f) / float(scn.width), float(y + 0.5f) / float(scn.height));
	float4 clr = make_float4(1,1,1,1);
	
	// ray surface hit
	rayCast ( SCN_SHADE, gvdb.top_lev, 0, scn.campos, rdir, hit, norm, clr, raySurfaceVoxelBrick );	
	if ( hit.z != NOHIT) {		
		clr = performPhongShading ( clr, hit, norm, rdir );		
	} else {		
		clr = SCN_BACKCLR;		// background color
	}	
	outBuf[y*scn.width + x] = make_uchar4(clr.x*255, clr.y*255, clr.z*255, clr.w*255);
}

// Render the volume data by raycasting
extern "C" __global__ void gvdbRaySurfaceTrilinear ( uchar4* outBuf )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= scn.width || y >= scn.height ) return;
		
	float3 hit = make_float3(NOHIT,NOHIT,NOHIT);	
	float3 norm;
	float3 rdir = getViewRay(float(x + 0.5f) / float(scn.width), float(y + 0.5f) / float(scn.height));
	float4 clr = make_float4(1,1,1,1);
	
	// ray surface hit
	rayCast ( SCN_SHADE, gvdb.top_lev, 0, scn.campos, rdir, hit, norm, clr, raySurfaceTrilinearBrick );	
	if ( hit.z != NOHIT) {		
		clr = performPhongShading ( clr, hit, norm, rdir );		
	} else {		
		clr = SCN_BACKCLR;			// background color
	}	
	outBuf[y*scn.width + x] = make_uchar4(clr.x*255, clr.y*255, clr.z*255, clr.w*255);
}
// Render the volume data by raycasting
extern "C" __global__ void gvdbRaySurfaceTricubic ( uchar4* outBuf )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= scn.width || y >= scn.height ) return;
		
	float3 hit = make_float3(NOHIT,NOHIT,NOHIT);	
	float3 norm;
	float3 rdir = getViewRay(float(x + 0.5f) / float(scn.width), float(y + 0.5f) / float(scn.height));
	float4 clr = make_float4(1,1,1,1);
	
	// ray surface hit
	rayCast ( SCN_SHADE, gvdb.top_lev, 0, scn.campos, rdir, hit, norm, clr, raySurfaceTricubicBrick );	
	if ( hit.z != NOHIT) {		
		clr = performPhongShading ( clr, hit, norm, rdir );		
	} else {		
		clr = SCN_BACKCLR;				// background color
	}	
	outBuf[y*scn.width + x] = make_uchar4(clr.x*255, clr.y*255, clr.z*255, clr.w*255);
}

// Render the volume data by raycasting
extern "C" __global__ void gvdbRaySurfaceDepth ( uchar4* outBuf )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= scn.width || y >= scn.height ) return;
		
	float3 hit = make_float3(NOHIT,NOHIT,NOHIT);	
	float3 norm;
	float3 rdir = getViewRay(float(x + 0.5f) / float(scn.width), float(y + 0.5f) / float(scn.height));
	float4 clr = make_float4(1,1,1,1);
	
	// ray surface hit
	rayCast ( SCN_SHADE, gvdb.top_lev, 0, scn.campos, rdir, hit, norm, clr, raySurfaceDepthBrick );
	if ( hit.z != NOHIT) {		
		clr = performPhongShading ( clr, hit, norm, rdir );
	} else {		
		clr = SCN_BACKCLR;					// background color
	}

	outBuf[y*scn.width + x] = make_uchar4(clr.x*255, clr.y*255, clr.z*255, clr.w*255 );
}

// Render the volume data by raycasting
extern "C" __global__ void gvdbRayLevelSet ( uchar4* outBuf )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= scn.width || y >= scn.height ) return;
	
	float3 hit = make_float3(NOHIT, NOHIT, NOHIT);
	float4 clr = make_float4(1,1,1,1);
	float3 norm;
	float3 rdir = getViewRay(float(x + 0.5f) / float(scn.width), float(y + 0.5f) / float(scn.height));

	// Raycast Level Set
	rayCast ( 0, gvdb.top_lev, 0, scn.campos, rdir, hit, norm, clr, rayLevelSetBrick );

	if ( hit.z != NOHIT) {		
		float3 lightdir = normalize ( scn.light_pos - hit );

		// shading		
		float3 eyedir		= normalize ( scn.campos - hit );
		float3 H			= normalize ( eyedir + lightdir );			
		float diffuse		= 0.4 * max(0.0f, dot( norm, lightdir ));
		float spec			= 0.3 * pow( max(0.0, dot( norm, H)), 24);	
		float3 env			= 0.3 * (make_float3(250,223,150) + (make_float3(183,232,254)-make_float3(250,223,150))*(norm.y*0.5+0.5) ) / 255.0;			

		// shadow ray		
		float3 h2 = make_float3(NOHIT,1,1);
		float3 n2;
		rayCast ( 0, gvdb.top_lev, 0, hit + norm*gvdb.voxelsize*2.0, lightdir, h2, n2, clr, rayLevelSetBrick );		
		clr.x = (diffuse+spec) * ((h2.x==NOHIT) ? 1 : 0);
		clr = make_float4( clr.x, clr.y, clr.z, 1);		
	} else {
		clr = SCN_BACKCLR;		// background color
	}	
	outBuf [ y*scn.width + x ] = make_uchar4( clr.x*255, clr.y*255, clr.z*255, clr.w*255 );		
}

// Render the volume data by raycasting
extern "C" __global__ void gvdbRayEmptySkip ( uchar4* outBuf )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= scn.width || y >= scn.height ) return;

	float4 clr = make_float4(1,1,1,1);
	float3 hit = make_float3(NOHIT,NOHIT,NOHIT);	
	float3 norm; 
	
	float3 rdir = getViewRay(float(x + 0.5f) / float(scn.width), float(y + 0.5f) / float(scn.height));

	// Empty skipping
	rayCast ( 0, gvdb.top_lev, 0, scn.campos, rdir, hit, norm, clr, rayEmptySkipBrick );

	if ( hit.z != NOHIT) {	
		clr = make_float4( hit * 0.01, 1 );
	} else {		
		clr = SCN_BACKCLR;	
	}	
	outBuf [ y*scn.width + x ] = make_uchar4( clr.x*255, clr.y*255, clr.z*255, 255 );	
}


// Raytrace a bundle of rays
extern "C" __global__ void gvdbRaytrace ( int num_rays, ScnRay* rays, float bias )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;	
	if ( x >= num_rays ) return;

	// raytrace
	rays[x].hit = make_float3(NOHIT, NOHIT, NOHIT);
	float4 hclr = make_float4(1,1,1,1);
	rayCast ( SCN_SHADE, gvdb.top_lev, 0, rays[x].orig, rays[x].dir, rays[x].hit, rays[x].normal, hclr, raySurfaceTricubicBrick );

	if ( rays[x].hit.z != NOHIT ) rays[x].hit -= rays[x].dir * bias;
}

// Render a cross section of the volume data in 3D
extern "C" __global__ void gvdbSection3D ( uchar4* outBuf )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= scn.width || y >= scn.height ) return;

	float4 clr = make_float4(1,1,1,0);
	float3 norm;
	float3 rdir = getViewRay(float(x + 0.5f) / float(scn.width), float(y + 0.5f) / float(scn.height));

	// raytrace with cross-section plane	
	float3 wpos = scn.campos;
	float t = rayPlaneIntersect ( wpos, rdir, SCN_SLICE_NORM, SCN_SLICE_PNT );			// hit section plane
	if ( t > 0 ) {																		// yes..
		wpos = scn.campos + t*rdir;														// get point of surface
		
		float3 offs, vmin, vdel; uint64 nid;
		VDBNode* node = getNodeAtPoint ( wpos, &offs, &vmin, &vdel, &nid );				// find vdb node at point
		if ( node != 0x0 ) {															
			//---- debugging: show apron
			// float3 p = offs + (wpos-vmin)*(34.0/16.0) - make_float3(gvdb.atlas_apron);	
			// clr = transfer ( tex3D ( volTexIn, p.x, p.y, p.z ) );
			t = getTrilinear ( wpos, offs, vmin, vdel );								// t <= voxel value
			clr = transfer ( t );														// clr at point on surface
			if ( gvdb.clr_chan != CHAN_UNDEF ) {										
				float3 p = offs + (wpos - vmin)/vdel;								
				clr *= make_float4( make_float3( getColor(gvdb.clr_chan, p) ), 1.0 );
			}
		} else {
			t = 0;																		// set t=0, no voxel value found
		}
	}


	// 3D surface raytrace 
	float3 hit = make_float3(NOHIT,NOHIT,NOHIT);		
	float4 hclr = make_float4(1,1,1,1);			
	// using previous wpos (set above) to start ray, trace beyond section plane to get 3D surface hit 
	rayCast ( SHADE_TRILINEAR, gvdb.top_lev, 0, wpos, rdir, hit, norm, hclr, raySurfaceTrilinearBrick );		
	if ( hit.z != NOHIT) {												// 3D surface hit..
		float3 lightdir		= normalize ( scn.light_pos - hit );							
		float ds			= (t > gvdb.thresh.x) ? 1 : 0.8*max(0.0f, dot( norm, lightdir )); // if voxel value on section plane is inside surface, no diffuse shading
		clr = lerp4( hclr * ds, clr, clr.w );							// blend 3D surface with cross-section clr
	} else {											
		clr = lerp4( SCN_BACKCLR, clr, clr.w );							// no 3D hit. blend background with cross-section
	}	
	outBuf [ y*scn.width + x ] = make_uchar4( clr.x*255, clr.y*255, clr.z*255, 255 );
}

// Render a section of the volume data in 2D 
extern "C" __global__ void gvdbSection2D ( uchar4* outBuf )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= scn.width || y >= scn.height ) return;

	float4 clr = make_float4(1,1,1,0);	
	float3 bgclr = make_float3 ( 0, 0, 0 );
	float3 wpos;
	float3 spnt = make_float3( float(x)*2.0/scn.width - 1.0, 0, float(y)*2.0/scn.height - 1.0);

	wpos = SCN_SLICE_PNT + spnt * SCN_SLICE_NORM;

	// get leaf node at hit point
	float3 offs, vmin, vdel;
	uint64 nid;
	VDBNode* node = getNodeAtPoint ( wpos, &offs, &vmin, &vdel, &nid );
	if ( node == 0x0 ) { outBuf [ y*scn.width + x ] = make_uchar4(bgclr.x*255, bgclr.y*255, bgclr.z*255, 255); return; }

	// get tricubic data value
	clr = transfer ( getTrilinear ( wpos, offs, vmin, vdel ) );	
	bgclr = lerp3 ( bgclr, make_float3(clr.x,clr.y,clr.z), clr.w );
	
	outBuf [ y*scn.width + x ] = make_uchar4( bgclr.x*255, bgclr.y*255, bgclr.z*255, 255 );	
}
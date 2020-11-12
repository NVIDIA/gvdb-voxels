
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
#include "radixsort_kernel.cuh"
//----------------------------------------- 
   
// Operator functions
#include "cuda_gvdb_operators.cuh"

// Particle functions 
#include "cuda_gvdb_particles.cuh"
     
inline __device__ float4 performPhongShading( VDBInfo* gvdb, uchar chan, float3 shit, float3 snorm, float4 sclr, gvdbBrickFunc_t brickfunc )
{
	if ( shit.z == NOHIT) 		// no surface hit
		return SCN_BACKCLR;

	// phong
	float3 lightdir = normalize(scn.light_pos - shit);  
	float diff = 0.9 * max(0.0f, dot(snorm, lightdir) );
	float amb = 0.1f;

	// shadow ray   
	if (SCN_SHADOWAMT > 0) { 		
		float3 hit2 = make_float3(0,0,NOHIT);		
		float4 hclr2 = make_float4(0,0,0,1);
		float3 norm2;
		rayCast ( gvdb, chan, shit + snorm * SCN_SHADOWBIAS, lightdir, hit2, norm2, hclr2, brickfunc );	// shadow ray
		diff = (hit2.z==NOHIT ? diff : diff*(1.0-SCN_SHADOWAMT) );
	}
	return make_float4( fxyz(sclr) * (diff + amb), 1.0 );
}

// Raytracing functions
extern "C" __global__ void gvdbRayDeep ( VDBInfo* gvdb, uchar chan, uchar4* outBuf )
{

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= scn.width || y >= scn.height ) return;

	float4 clr = make_float4(0,0,0,1);
	float3 hit = make_float3(0,0,NOHIT);
	float3 norm;
	float3 rpos = getViewPos();
	float3 rdir = getViewRay(float(x + 0.5f) / float(scn.width), float(y + 0.5f) / float(scn.height));

	// ray deep sampling	
	rayCast ( gvdb, chan,  rpos, rdir, hit, norm, clr, rayDeepBrick );
	clr = make_float4( lerp3(SCN_BACKCLR, clr, 1.0-clr.w), 1.0-clr.w );
	
	outBuf [ y*scn.width + x ] = make_uchar4( clr.x*255, clr.y*255, clr.z*255, clr.w*255 );
}



// Render the volume data by raycasting
extern "C" __global__ void gvdbRaySurfaceVoxel ( VDBInfo* gvdb, uchar chan, uchar4* outBuf )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= scn.width || y >= scn.height ) return;

	float3 hit = make_float3(NOHIT,NOHIT,NOHIT);	
	float3 norm;
	float3 rpos = getViewPos();
	float3 rdir = getViewRay(float(x + 0.5f) / float(scn.width), float(y + 0.5f) / float(scn.height));
	float4 clr = make_float4(1,1,1,1);
	
	// ray surface hit
	rayCast ( gvdb, chan, rpos, rdir, hit, norm, clr, raySurfaceVoxelBrick );	
	clr = performPhongShading ( gvdb, chan, hit, norm, clr, raySurfaceVoxelBrick );		

	outBuf[y*scn.width + x] = make_uchar4(clr.x*255, clr.y*255, clr.z*255, clr.w*255);
}

// Render the volume data by raycasting
extern "C" __global__ void gvdbRaySurfaceTrilinear ( VDBInfo* gvdb, uchar chan, uchar4* outBuf )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= scn.width || y >= scn.height ) return;
		
	float3 hit = make_float3(NOHIT,NOHIT,NOHIT);	
	float3 norm;
	float3 rpos = getViewPos();
	float3 rdir = getViewRay(float(x + 0.5f) / float(scn.width), float(y + 0.5f) / float(scn.height));
	float4 clr = make_float4(1,1,1,1);
	
	// ray surface hit
	rayCast ( gvdb, chan, rpos, rdir, hit, norm, clr, raySurfaceTrilinearBrick );	
	clr = performPhongShading ( gvdb, chan, hit, norm, clr, raySurfaceTrilinearBrick );		
	
	outBuf[y*scn.width + x] = make_uchar4(clr.x*255, clr.y*255, clr.z*255, clr.w*255);
}
// Render the volume data by raycasting
extern "C" __global__ void gvdbRaySurfaceTricubic ( VDBInfo* gvdb, uchar chan, uchar4* outBuf )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= scn.width || y >= scn.height ) return;
		
	float3 hit = make_float3(NOHIT,NOHIT,NOHIT);	
	float3 norm;
	float3 rpos = getViewPos();
	float3 rdir = getViewRay(float(x + 0.5f) / float(scn.width), float(y + 0.5f) / float(scn.height));
	float4 clr = make_float4(1,1,1,1);
	
	// ray surface hit
	rayCast (  gvdb, chan, rpos, rdir, hit, norm, clr, raySurfaceTricubicBrick );	
	clr = performPhongShading ( gvdb, chan, hit, norm, clr, raySurfaceTrilinearBrick );
	
	outBuf[y*scn.width + x] = make_uchar4(clr.x*255, clr.y*255, clr.z*255, clr.w*255);
}

// Render the volume data by raycasting
extern "C" __global__ void gvdbRaySurfaceDepth ( VDBInfo* gvdb, uchar chan, uchar4* outBuf )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= scn.width || y >= scn.height ) return;
		
	float3 hit = make_float3(NOHIT,NOHIT,NOHIT);	
	float3 norm;
	float3 rpos = getViewPos();
	float3 rdir = getViewRay(float(x + 0.5f) / float(scn.width), float(y + 0.5f) / float(scn.height));
	float4 clr = make_float4(1,1,1,1);
	
	// ray surface hit
	//   *NOTE*: raySurfaceDepthBrick not yet implemented

	rayCast (gvdb, chan, rpos, rdir, hit, norm, clr, raySurfaceTrilinearBrick );
	clr = performPhongShading ( gvdb, chan,  hit, norm, clr, raySurfaceTrilinearBrick  );

	outBuf[y*scn.width + x] = make_uchar4(clr.x*255, clr.y*255, clr.z*255, clr.w*255 );
}

// Render the volume data by raycasting
extern "C" __global__ void gvdbRayLevelSet ( VDBInfo* gvdb, uchar chan, uchar4* outBuf )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= scn.width || y >= scn.height ) return;
	
	float3 hit = make_float3(0,0,NOHIT);	
	float4 clr = make_float4(1,1,1,1);
	float3 norm;
	float3 rpos = getViewPos();
	float3 rdir = getViewRay(float(x + 0.5f) / float(scn.width), float(y + 0.5f) / float(scn.height));

	// Raycast Level Set
	rayCast (  gvdb, chan, rpos, rdir, hit, norm, clr, rayLevelSetBrick );
	clr = performPhongShading ( gvdb, chan, hit, norm, clr, rayLevelSetBrick  );		
	
	outBuf [ y*scn.width + x ] = make_uchar4( clr.x*255, clr.y*255, clr.z*255, clr.w*255 );		
}

// Render the volume data by raycasting
extern "C" __global__ void gvdbRayEmptySkip ( VDBInfo* gvdb, uchar chan, uchar4* outBuf )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= scn.width || y >= scn.height ) return;

	float4 clr = make_float4(1,1,1,1);
	float3 hit = make_float3(NOHIT,NOHIT,NOHIT);	
	float3 norm; 
	
	float3 rpos = getViewPos();
	float3 rdir = getViewRay(float(x + 0.5f) / float(scn.width), float(y + 0.5f) / float(scn.height));


	// Empty skipping
	rayCast ( gvdb, chan, rpos, rdir, hit, norm, clr, rayEmptySkipBrick );

	if ( hit.z != NOHIT) {	
		clr = make_float4( hit * 0.01, 1 );
	} else {		
		clr = SCN_BACKCLR;	
	}	
	outBuf [ y*scn.width + x ] = make_uchar4( clr.x*255, clr.y*255, clr.z*255, 255 );	
}


// Raytrace a bundle of rays
extern "C" __global__ void gvdbRaytrace ( VDBInfo* gvdb, uchar chan, int num_rays, ScnRay* rays, float bias )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;	
	if ( x >= num_rays ) return;

	// raytrace
	rays[x].hit = make_float3(NOHIT, NOHIT, NOHIT);
	float4 hclr = make_float4(1,1,1,1);
	rayCast ( gvdb, chan,  rays[x].orig, rays[x].dir, rays[x].hit, rays[x].normal, hclr, raySurfaceTrilinearBrick );

	if ( rays[x].hit.z != NOHIT ) rays[x].hit -= rays[x].dir * bias;
}

// Render a cross section of the volume data in 3D
extern "C" __global__ void gvdbSection3D ( VDBInfo* gvdb, uchar chan, uchar4* outBuf )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= scn.width || y >= scn.height ) return;

	float4 clr = make_float4(1,1,1,0);
	float3 norm;
	float3 rdir = getViewRay(float(x + 0.5f) / float(scn.width), float(y + 0.5f) / float(scn.height));

	// raytrace with cross-section plane	
	float3 wpos = getViewPos();
	float t = rayPlaneIntersect ( wpos, rdir, SCN_SLICE_NORM, SCN_SLICE_PNT );			// hit section plane
	if ( t > 0 ) {																		// yes..
		wpos += t*rdir;																	// get point of surface
		
		float3 offs, vmin; uint64 nid;
		VDBNode* node = getNodeAtPoint ( gvdb, wpos, &offs, &vmin, &nid );				// find vdb node at point
		if ( node != 0x0 ) {															
			//---- debugging: show apron
			// float3 p = offs + (wpos-vmin)*(34.0/16.0) - make_float3(gvdb.atlas_apron);	
			// clr = transfer ( tex3D ( volTexIn, p.x, p.y, p.z ) );
			t = getTrilinear ( gvdb, chan, wpos, offs, vmin );								// t <= voxel value
			clr = transfer ( gvdb, t );														// clr at point on surface
			if ( gvdb->clr_chan != CHAN_UNDEF ) {										
				float3 p = offs + (wpos - vmin);								
				clr *= make_float4( make_float3( getColor(gvdb, gvdb->clr_chan, p) ), 1.0 );
			}
		} else {
			t = 0;																		// set t=0, no voxel value found
		}
	}

	// 3D surface raytrace 
	float3 hit = make_float3(NOHIT,NOHIT,NOHIT);		
	float4 hclr = make_float4(1,1,1,1);			
	// using previous wpos (set above) to start ray, trace beyond section plane to get 3D surface hit 
	rayCast ( gvdb, chan,  wpos, rdir, hit, norm, hclr, raySurfaceTrilinearBrick );		
	if ( hit.z != NOHIT) {												// 3D surface hit..
		float3 lightdir		= normalize ( scn.light_pos - hit );							
		float ds			= (t > SCN_THRESH) ? 1 : 0.8*max(0.0f, dot( norm, lightdir )); // if voxel value on section plane is inside surface, no diffuse shading
		clr = lerp4( hclr * ds, clr, clr.w );							// blend 3D surface with cross-section clr
	} else {											
		clr = lerp4( SCN_BACKCLR, clr, clr.w );							// no 3D hit. blend background with cross-section
	}	
	outBuf [ y*scn.width + x ] = make_uchar4( clr.x*255, clr.y*255, clr.z*255, 255 );
}

// Render a section of the volume data in 2D 
extern "C" __global__ void gvdbSection2D ( VDBInfo* gvdb, uchar chan, uchar4* outBuf )
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
	float3 offs, vmin;
	uint64 nid;
	VDBNode* node = getNodeAtPoint ( gvdb, wpos, &offs, &vmin, &nid );
	if ( node == 0x0 ) { outBuf [ y*scn.width + x ] = make_uchar4(bgclr.x*255, bgclr.y*255, bgclr.z*255, 255); return; }

	// get tricubic data value
	clr = transfer ( gvdb, getTrilinear ( gvdb, chan, wpos, offs, vmin ) );
	bgclr = lerp3 ( bgclr, make_float3(clr.x,clr.y,clr.z), clr.w );
	
	outBuf [ y*scn.width + x ] = make_uchar4( bgclr.x*255, bgclr.y*255, bgclr.z*255, 255 );	
}
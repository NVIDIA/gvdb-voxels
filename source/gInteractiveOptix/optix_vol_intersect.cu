//--------------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2017, NVIDIA Corporation
//
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
// 
// Version 1.0: Rama Hoetzlein, 5/1/2017
//----------------------------------------------------------------------------------

#include "optix_extra_math.cuh"			
#include "texture_fetch_functions.h"

//------------------- GVDB Structure
#define OPTIX_PATHWAY
#include "cuda_gvdb_scene.cuh"		// GVDB Scene
#include "cuda_gvdb_nodes.cuh"		// GVDB Node structure
#include "cuda_gvdb_geom.cuh"		// GVDB Geom helpers
#include "cuda_gvdb_dda.cuh"		// GVDB DDA 
#include "cuda_gvdb_raycast.cuh"	// GVDB Raycasting
//--------------------


rtBuffer<float3>		  brick_buffer;

rtDeclareVariable(uint,	  mat_id, , );
rtDeclareVariable(float3, light_pos, , );

rtDeclareVariable(float3, back_hit_point,	attribute back_hit_point, ); 
rtDeclareVariable(float3, front_hit_point,	attribute front_hit_point, ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal,	attribute shading_normal, ); 
rtDeclareVariable(float4, deep_color,		attribute deep_color, ); 

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

struct PerRayData_radiance
{
	float3	result;
	float	length; 
	float	alpha;
	int		depth;
	int		rtype;
};

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );

//------ Intersection Program

RT_PROGRAM void vol_intersect( int primIdx )
{
	float3 hit = make_float3(NOHIT,NOHIT,NOHIT);	
	float3 norm = make_float3(0,0,0);
	float4 clr = make_float4(0,0,0,0);	
	float t;

	//-- Ray march		
	float4 hclr;
	rayCast ( SCN_SHADE, gvdb.top_lev, 0, ray.origin, ray.direction, hit, norm, hclr, raySurfaceBrick );
	if ( hit.z == NOHIT) return;	
	t = length ( hit - ray.origin );

	// report intersection to optix
	if ( rtPotentialIntersection( t ) ) {	

		shading_normal = norm;		
		geometric_normal = norm;
		front_hit_point = hit + shading_normal*gvdb.voxelsize;
		back_hit_point  = hit - shading_normal*gvdb.voxelsize*5;
		deep_color = make_float4(1,1,1,1);
		if ( prd_radiance.rtype == SHADOW_RAY ) deep_color.w = (hit.x==NOHIT) ? 1 : 0;

		rtReportIntersection( mat_id );
	}
}

RT_PROGRAM void vol_deep( int primIdx )
{
	float3 hit = make_float3(NOHIT,NOHIT,NOHIT);	
	float3 norm = make_float3(0,1,0);
	float4 clr = make_float4(0,0,0,1);	
	if ( prd_radiance.rtype == MESH_RAY ) return;	

	// ---- Debugging
	// Uncomment this code to demonstrate tracing of the bounding box 
	// surrounding the volume.
	/*hit = rayBoxIntersect ( ray.origin, ray.direction, gvdb.bmin, gvdb.bmax );
	if ( hit.z == NOHIT ) return;
	if ( rtPotentialIntersection ( hit.x ) ) {
		shading_normal = norm;		
		geometric_normal = norm;
		front_hit_point = ray.origin + hit.x * ray.direction;
		back_hit_point  = ray.origin + hit.y * ray.direction;
		deep_color = make_float4( front_hit_point/200.0, 0.5);	
		rtReportIntersection( 0 );		
	}
	return;*/

	//-- Raycast
	rayCast ( SHADE_VOLUME, gvdb.top_lev, 0, ray.origin, ray.direction, hit, norm, clr, rayDeepBrick );
	if ( hit.z == NOHIT) return;	

	if ( rtPotentialIntersection( hit.x ) ) {

		shading_normal = norm;		
		geometric_normal = norm;
		front_hit_point = ray.origin + hit.x * ray.direction;
		back_hit_point  = ray.origin + hit.y * ray.direction;
		deep_color = make_float4 ( fxyz(clr), 1.0-clr.w );

		rtReportIntersection( 0 );			
	}
}

RT_PROGRAM void vol_levelset ( int primIdx )
{
	float3 hit = make_float3(NOHIT,1,1);	
	float3 norm = make_float3(0,0,0);
	float4 clr = make_float4(0,0,0,0);	
	float t;

	//-- Ray march		
	rayCast ( 0, gvdb.top_lev, 0, ray.origin, ray.direction, hit, norm, clr, rayLevelSetBrick );
	if ( hit.x == NOHIT) return;	
	t = length ( hit - ray.origin );

	// report intersection to optix
	if ( rtPotentialIntersection( t ) ) {	

		shading_normal = norm;		
		geometric_normal = norm;
		front_hit_point = hit + shading_normal*gvdb.voxelsize;
		back_hit_point  = hit - shading_normal*gvdb.voxelsize*5;
		deep_color = make_float4(1,1,1,1);
		if ( prd_radiance.rtype == SHADOW_RAY ) deep_color.w = (hit.x==NOHIT) ? 1 : 0;

		rtReportIntersection( mat_id );
	}
}


RT_PROGRAM void vol_bounds (int primIdx, float result[6])
{
	// AABB bounds is just the brick extents	
	optix::Aabb* aabb = (optix::Aabb*) result;
	aabb->m_min = brick_buffer[ primIdx*2 ];
	aabb->m_max = brick_buffer[ primIdx*2+1 ];
}


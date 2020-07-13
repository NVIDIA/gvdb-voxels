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


#include "optix_gvdb.cuh"

rtBuffer<float3>		  brick_buffer;

rtDeclareVariable(uint,	  mat_id, , );
rtDeclareVariable(float3, light_pos, , );

rtDeclareVariable(float3, back_hit_point,	attribute back_hit_point, ); 
rtDeclareVariable(float3, front_hit_point,	attribute front_hit_point, ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal,	attribute shading_normal, ); 
rtDeclareVariable(float4, deep_color,		attribute deep_color, ); 

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

struct RayInfo
{
	float3	result;
	float	length; 
	float	alpha;
	int		depth;
	int		rtype;
};
rtDeclareVariable( RayInfo, ray_info, rtPayload, );

//------ Intersection Program

RT_PROGRAM void vol_intersect( int primIdx )
{
	float3 hit = make_float3(NOHIT,NOHIT,NOHIT);	
	float3 norm = make_float3(0,0,0);	
	float4 hclr = make_float4(1,1,1,1);
	float t;

	//-- Ray march
	// If the Optix transform node has been set up correctly, then the ray is in GVDB's coordinate system.
	const float3 orig = ray.origin;
	const float3 dir = ray.direction;
	rayCast(&gvdbObj, gvdbChan, orig, dir, hit, norm, hclr, raySurfaceTrilinearBrick);
	if ( hit.z == NOHIT) return;	
	t = length ( hit - ray.origin );

	// report intersection to optix
	if ( rtPotentialIntersection( t ) ) {	

		shading_normal = norm;		
		geometric_normal = norm;
		// Transform from GVDB's coordinate space to OptiX's coordinate space
		front_hit_point = hit + shading_normal * 2;
		back_hit_point = hit - shading_normal * 4;
		deep_color = hclr;
		//if ( ray_info.rtype == SHADOW_RAY ) deep_color.w = (hit.x!=NOHIT) ? 0 : 1;

		rtReportIntersection( mat_id );
	}
}

RT_PROGRAM void vol_deep( int primIdx )
{
	float3 hit = make_float3(0,0,NOHIT);	
	float3 norm = make_float3(0,1,0);
	float4 clr = make_float4(0,0,0,1);	
	if (ray_info.rtype == MESH_RAY ) return;

	float3 orig = ray.origin;
	float3 dir = ray.direction;

	// ---- Debugging
	// Uncomment this code to demonstrate tracing of the bounding box 
	// surrounding the volume.
	/* hit = rayBoxIntersect ( orig, dir, gvdbObj.bmin, gvdbObj.bmax );
	if ( hit.z == NOHIT ) return;
	const float t2 = hit.x;
	if ( rtPotentialIntersection ( t2 ) ) {
		shading_normal = norm;		
		geometric_normal = norm;
		front_hit_point = orig + hit.x * dir;
		back_hit_point  = orig + hit.y * dir;
		deep_color = make_float4( front_hit_point/200.0, 0.5);	
		rtReportIntersection( 0 );
	}
	return; */

	//-- Raycast		
	rayCast ( &gvdbObj, gvdbChan, orig, dir, hit, norm, clr, rayDeepBrick );	
	if ( hit.x==0 && hit.y == 0) return;

	// Note that rayDeepBrick sets hit.x and hit.y to the front and back brick intersection points in GVDB's coordinate
	// system, in contrast to the other functions in this file.
	const float t = hit.x;

	if ( rtPotentialIntersection( t ) ) {

		shading_normal = norm;		
		geometric_normal = norm;
		// Transform from GVDB's coordinate space to the application's coordinate space
		front_hit_point = orig + hit.x * dir;
		back_hit_point  = orig + hit.y * dir;
		deep_color = make_float4 ( fxyz(clr), 1.0-clr.w );		

		rtReportIntersection( 0 );			
	}
}

RT_PROGRAM void vol_levelset ( int primIdx )
{
	float3 hit = make_float3(NOHIT,NOHIT,NOHIT);	
	float3 norm = make_float3(0,0,0);
	float4 hclr = make_float4(0,1,0,1);	
	float t;
	
	//if (ray_info.rtype == SHADOW_RAY && ray_info.depth >= 1) return;

	// Transform from application space to GVDB's coordinate space
	float3 orig = ray.origin;
	float3 dir = ray.direction;

	//-- Ray march		
	if (ray_info.rtype == REFRACT_RAY) {		
		if (ray_info.depth == 2) return;
		rayCast(&gvdbObj, gvdbChan, orig, dir, hit, norm, hclr, raySurfaceTrilinearBrick);
	}	else {
		rayCast(&gvdbObj, gvdbChan, orig, dir, hit, norm, hclr, rayLevelSetBrick);
	}
	if ( hit.z == NOHIT) return;

	// Transform from GVDB's coordinate space to application space
	t = length ( hit - ray.origin );

	// report intersection to optix
	if ( rtPotentialIntersection( t ) ) {	

		shading_normal = norm;		
		geometric_normal = norm;
		front_hit_point = hit;
		back_hit_point = hit - shading_normal * .2;
		deep_color = hclr;

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


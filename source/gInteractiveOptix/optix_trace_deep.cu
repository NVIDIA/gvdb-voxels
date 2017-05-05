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

#define ANY_RAY			0
#define	SHADOW_RAY		1
#define VOLUME_RAY		2
#define MESH_RAY		3

rtDeclareVariable(float3,       light_pos, , );

rtDeclareVariable(rtObject,     top_object, , );
rtDeclareVariable(float,        scene_epsilon, , );
rtDeclareVariable(unsigned int, shadow_enable, , );
rtDeclareVariable(unsigned int, mirror_enable, , );
rtDeclareVariable(unsigned int, cone_enable, , );
rtDeclareVariable(int,          max_depth, , );

rtDeclareVariable(float3,		shading_normal,		attribute shading_normal, ); 
rtDeclareVariable(float3,		front_hit_point,	attribute front_hit_point, );
rtDeclareVariable(float3,		back_hit_point,		attribute back_hit_point, );
rtDeclareVariable(float4,		deep_color,			attribute deep_color, );
rtDeclareVariable(int,			obj_type,			attribute obj_type, );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(uint2,        launch_index, rtLaunchIndex, );
rtDeclareVariable(unsigned int, sample, , );

rtBuffer<unsigned int, 2>       rnd_seeds;

struct PerRayData_radiance
{
	float3	result;
	float	length; 
	float	alpha;
	int		depth;
	int		rtype;
};

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );

// -----------------------------------------------------------------------------

static __device__ __inline__ float3 TraceRay (float3 origin, float3 direction, int depth, int rtype, float& length )
{
  optix::Ray ray = optix::make_Ray( origin, direction, 0, 0.0f, RT_DEFAULT_MAX );
  PerRayData_radiance prd;
  prd.length = 0.f;
  prd.depth = depth; 
  prd.rtype = rtype;
  rtTrace( top_object, ray, prd );  
  length = prd.length;
  return prd.result;
}

float3 __device__ __inline__ jitter_sample ( const uint2& index )
{	 
    volatile unsigned int seed  = rnd_seeds[ index ]; // volatile workaround for cuda 2.0 bug
    unsigned int new_seed  = seed;
    float uu = rnd( new_seed )-0.5f;
    float vv = rnd( new_seed )-0.5f;
	float ww = rnd( new_seed )-0.5f;
    rnd_seeds[ index ] = new_seed;	
    return make_float3(uu,vv,ww);
}

RT_PROGRAM void trace_deep ()
{
	// Volumetric material

	// We arrive here after vol_deep has already traced into the gvdb volume.
	// - deep_color is the accumulated color along the volume ray
	// - front_hit_point is the start point of the volume
	// - back_hit_point is the ending point of the volume
	
	// Blending with polygonal objects is achieved by stochastically 
	// tracing a MESH_RAY from a random point inside the volume toward the background.
	float rlen;	
	float3 jit = jitter_sample ( make_uint2( (launch_index.x + sample) % blockDim.x, (launch_index.y + sample) % blockDim.y) );	
	float3 pos = front_hit_point + (jit.x+0.5f) * (back_hit_point - front_hit_point);
	float3 bgclr = TraceRay ( pos, ray.direction, 1, MESH_RAY, rlen );	

	// Result is blending of background color and the volume color (deep_color)	
	prd_radiance.result = lerp3 ( bgclr, fxyz(deep_color), deep_color.w );
	prd_radiance.length = length ( back_hit_point - ray.origin );
	prd_radiance.alpha = deep_color.w;

	// prd_radiance.result = fhp/200.0;			-- debugging
}

// -----------------------------------------------------------------------------

//
// Attenuates shadow rays for shadowing transparent objects
//
RT_PROGRAM void trace_shadow ()
{
	// rtype is SHADOW_RAY
	prd_radiance.alpha = 0; //deep_color.w;
}

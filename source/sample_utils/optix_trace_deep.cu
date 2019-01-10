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
#define REFRACT_RAY		4

struct Material {
	char		name[64];
	int			id;
	float		light_width;		// light scatter
	
	float3		amb_color;
	float3		env_color;			// 0.5,0.5,0.5
	float3		diff_color;			// .6,.7,.7
	float3		spec_color;			// 3,3,3
	float		spec_power;			// 400		

	float		shadow_width;		// shadow scatter
	float		shadow_bias;

	float		refl_width;			// reflect scatter
	float3		refl_color;			// 1,1,1		
	float		refl_bias;

	float		refr_width;			// refract scatter
	float		refr_ior;			// 1.2
	float3		refr_color;			// .35, .4, .4
	float		refr_amount;		// 10
	float		refr_offset;		// 15
	float		refr_bias;
};

rtDeclareVariable(float3,       light_pos, , );
rtDeclareVariable(Material,		mat, , );

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

struct RayInfo
{
	float3	result;
	float	length; 
	float	alpha;
	int		depth;
	int		rtype;
};

rtDeclareVariable(RayInfo,		rayinfo, rtPayload, );

// -----------------------------------------------------------------------------

static __device__ __inline__ float3 TraceRay (float3 origin, float3 direction, int depth, int rtype, float& length )
{
  optix::Ray ray = optix::make_Ray( origin, direction, 0, 0.0f, RT_DEFAULT_MAX );
  RayInfo rayi;
  rayi.length = 0.f;
  rayi.depth = depth;
  rayi.rtype = rtype;
  rayi.alpha = 1.f;
  rtTrace( top_object, ray, rayi );
  length = rayi.length;
  return (rtype == SHADOW_RAY) ? make_float3(rayi.alpha, rayi.alpha, rayi.alpha) : rayi.result;
}

float3 __device__ __inline__ jitter_sample ()
{	 
	uint2 index = make_uint2(launch_index.x & 0x7F, launch_index.y & 0x7F);
	unsigned int seed = rnd_seeds[index];  	
	float uu = rnd(seed) - 0.5f;
	float vv = rnd(seed) - 0.5f;
	float ww = rnd(seed) - 0.5f;
	rnd_seeds[index] = seed;
	return make_float3(uu, vv, ww);
}

RT_PROGRAM void trace_deep ()
{
	// Volumetric material

	// We arrive here after vol_deep has already traced into the gvdb volume.
	// - deep_color is the accumulated color along the volume ray
	// - front_hit_point is the start point of the volume
	// - back_hit_point is the ending point of the volume

	// Blending with polygonal objects 
	float plen;		
	float3 pos = ray.origin; 
	float3 bgclr = TraceRay ( pos, ray.direction, rayinfo.depth, MESH_RAY, plen );	// poly ray
	float vlen = length(front_hit_point - ray.origin);		// volume ray
	float a = deep_color.w;
	//if (plen < vlen) { a = 0; vlen = plen; }

	// Result is blending of background/poly color and the volume color (deep_color)		
	rayinfo.result = lerp3 ( bgclr, fxyz(deep_color), a );
	rayinfo.length = vlen;
	rayinfo.alpha = a;

	// prd_radiance.result = fhp/200.0;			-- debugging
}

// -----------------------------------------------------------------------------

//
// Attenuates shadow rays for shadowing transparent objects
//
RT_PROGRAM void trace_shadow ()
{
	// rtype is SHADOW_RAY
	rayinfo.alpha = deep_color.w;
}

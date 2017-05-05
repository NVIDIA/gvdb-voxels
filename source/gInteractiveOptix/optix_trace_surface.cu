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

struct Material {
	char		name[64];
	float		light_width;		// light scatter
	float		shadow_width;		// shadow scatter
	float3		env_color;			// 0.5,0.5,0.5
	float3		diff_color;			// .6,.7,.7
	float3		spec_color;			// 3,3,3
	float		spec_power;			// 400		
	float		refl_width;			// reflect scatter
	float3		refl_color;			// 1,1,1		
	float		refr_width;			// refract scatter
	float		refr_ior;			// 1.2
	float3		refr_color;			// .35, .4, .4
	float		refr_amount;		// 10
	float		refr_offset;		// 15
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

static __device__ __inline__ float ShadowRay( float3 origin, float3 direction, bool bDeep )
{
  optix::Ray ray = optix::make_Ray( origin, direction, 0, 0.0f, RT_DEFAULT_MAX );
  PerRayData_radiance prd;  
  prd.alpha = 1.0f;
  prd.depth = 2;
  prd.rtype = bDeep ? ANY_RAY : SHADOW_RAY;
  rtTrace( top_object, ray, prd );
  return prd.alpha;
}

static __device__ __inline__ float3 exp( const float3& x )
{
  return make_float3(exp(x.x), exp(x.y), exp(x.z));
}

// -----------------------------------------------------------------------------

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

RT_PROGRAM void trace_surface ()
{
	// geometry vectors
	const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal)); // normal  
	const float3 fhp = rtTransformPoint(RT_OBJECT_TO_WORLD, front_hit_point);
	const float3 bhp = rtTransformPoint(RT_OBJECT_TO_WORLD, back_hit_point);
	const float3 raydir = ray.direction;                                            // incident direction
	float3 lightdir, spos, refldir, reflclr, shadowclr;
	float3 jit = jitter_sample ( make_uint2( (launch_index.x + sample) % blockDim.x, (launch_index.y + sample) % blockDim.y) );
	float ndotl, refldist;
	float d = length ( fhp - ray.origin );						// approximation for screen coverage of pixel in world space
	float3 dxyz = make_float3(0.75) * d / 1280.0;

	lightdir =	normalize ( normalize (light_pos - fhp) + jit * mat.light_width );
	ndotl = dot( n, lightdir );	

	// shading		
	float3 H			= normalize ( -raydir + lightdir );			
	float3 diffuse		= mat.diff_color * max(0.0f, ndotl );
	float3 spec			= mat.spec_color * pow( max(0.0f, dot( n, H )), (float) mat.spec_power );	
	
	if ( prd_radiance.depth <= 1 ) {
		// reflection sample		
		refldir = normalize ( normalize ( 2 * ndotl * n - lightdir ) + jit * mat.refl_width );
		reflclr = TraceRay ( fhp, refldir, 2, ANY_RAY, refldist ) * mat.refl_color;
		
		// refraction sample	
		// optix::refract ( refrdir, raydir, n, mat.refr_ior );
		// refrdir = normalize ( refrdir + jit * mat.refr_width );
		// refrclr = TraceRay ( bhp, refrdir, 2, ANY_RAY, refrdist ) + mat.refr_color;
		// refrclr *= mat.refr_amount / (refrdist + mat.refr_offset); 
		  
		// shadow sample		
		spos = fhp + jit*dxyz + n;				// jittered shadow sample				
		lightdir =	normalize ( normalize (light_pos - spos) + jit*mat.shadow_width);	// compute light direction at shadow sample	
		shadowclr = TraceRay ( spos, lightdir, 2, ANY_RAY, refldist );

	} else {
		reflclr = make_float3(0,0,0);		
		shadowclr = make_float3(1,1,1);
	}
	prd_radiance.result = (diffuse+spec)*shadowclr + reflclr;
	prd_radiance.length = d;
	prd_radiance.alpha = deep_color.w;
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

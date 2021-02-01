//-----------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2017 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
// 
// Version 1.0: Rama Hoetzlein, 5/1/2017
//-----------------------------------------------------------------------------

#include "optix_gvdb.cuh"

rtTextureSampler<float4, 2>		envmap;

rtDeclareVariable(float3,		cam_pos, , );

rtDeclareVariable(uint2,		launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2,		launch_dim,   rtLaunchDim, );
rtBuffer<unsigned int, 2>       rnd_seeds;

rtDeclareVariable(optix::Ray,	ray, rtCurrentRay, );

struct RayInfo
{
	float3	result;
	float	length; 
	float	alpha;
	int		depth;
	int		rtype;
};

rtDeclareVariable(RayInfo,				rayinfo,	rtPayload, );

// -----------------------------------------------------------------------------

float3 __device__ __inline__ jitter_sample ()
{	 
	uint2 index = make_uint2(launch_index.x & 0x7F, launch_index.y & 0x7F);
	unsigned int seed = rnd_seeds[index];	 
	float uu = rnd(seed) - 0.5f;
	float vv = rnd(seed) - 0.5f;
	float ww = rnd(seed) - 0.5f;
	return make_float3(uu, vv, ww);
}

float3 __device__ __inline__ sampleEnv(float3 dir)
{
	// envmap is a texture containing the top half of the skydome, in
	// warped spherical coordinates:
	// (x, y, z) = (cos(pi*u)*v, 1-v, sin(pi*u)*v)
	float u = atan2f(dir.x, dir.z) * M_1_PIf;
	float v = 1.0 - dir.y;
	// Use a single color for the sea
	return (v > 1.0f) ? make_float3(.1f, .1f, .1f) : make_float3( tex2D(envmap, u, v) );
}

RT_PROGRAM void miss()
{
	rayinfo.result = sampleEnv(ray.direction) * make_float3(SCN_BACKCLR);
	rayinfo.length = 1.0e10;
}

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
	float u = atan2f(dir.x, dir.z) * M_1_PIf;
	float v = 1.0 - dir.y;
	return (v < 0) ? make_float3(.1, .1, .1) : make_float3( tex2D(envmap, u, v) );
}

RT_PROGRAM void miss()
{
	rayinfo.result = sampleEnv(ray.direction) * make_float3(SCN_BACKCLR);
	rayinfo.length = 1.0e10;
}

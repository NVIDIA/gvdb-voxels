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

//rtTextureSampler<float4, 2>		envmap;

rtDeclareVariable(float3,        cam_pos, , );
rtDeclareVariable(float3,        cam_U, , );
rtDeclareVariable(float3,        cam_V, , );
rtDeclareVariable(float3,        cam_W, , );

rtDeclareVariable(uint2,		launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2,		launch_dim,   rtLaunchDim, );
rtBuffer<unsigned int, 2>       rnd_seeds;

rtDeclareVariable(optix::Ray,	ray, rtCurrentRay, );

struct PerRayData_radiance
{
	float3	result;
	float	length; 
	float	alpha;
	int		depth;
	int		rtype;
};

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_radiance, prd_shadow, rtPayload, );

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

RT_PROGRAM void miss()
{
	float2 d = make_float2(launch_index) / make_float2(launch_dim); // - 0.5f;  

	float3 clr = make_float3 ( 0.0, fabs(ray.direction.y)*0.15, fabs(ray.direction.y)*0.2 );
	clr += jitter_sample ( launch_index ) * 0.05;    

	prd_radiance.result = clr;
}

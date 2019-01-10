
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
#include "texture_fetch_functions.h"			// from OptiX SDK

struct PerRayData_radiance
{
	float3	result;
	float	length; 
	float	alpha;
	int		depth;
	int		rtype;
};

rtDeclareVariable(float3,        cam_pos, , );
rtDeclareVariable(float3,		 cam_U, , );
rtDeclareVariable(float3,		 cam_V, , );
rtDeclareVariable(float3,		 cam_S, , );
rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(unsigned int,  sample, , );
rtBuffer<float3, 2>              output_buffer;
rtDeclareVariable(rtObject,      top_object, , );
rtBuffer<unsigned int, 2>        rnd_seeds;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

float3 __device__ __inline__ jitter_sample ()
{	 
	uint2 index = make_uint2(launch_index.x % 128, launch_index.y % 128);
    unsigned int seed  = rnd_seeds[ index ]; 
    float uu = rnd( seed )-0.5f;
    float vv = rnd( seed )-0.5f;
	float ww = rnd( seed )-0.5f;    
    return make_float3(uu,vv,ww);
}


__device__ float3 getViewRay(float x, float y, float pixsize)
{
	float3 v = x*cam_U + y*cam_V + cam_S + jitter_sample()*make_float3(pixsize, pixsize, pixsize);
	return normalize(v);
}

RT_PROGRAM void trace_primary ()
{
  float2 d = make_float2(launch_index) / make_float2(launch_dim);
  float pixsize = length ( cam_U ) / launch_dim.x;	
  float3 ray_direction;
  float3 result;

  PerRayData_radiance prd;
  prd.length = 0.f;
  prd.alpha = 1.f;
  prd.depth = 0;
  prd.rtype = 0;	// ANY_RAY

  int initial_samples = 1;
  
  if ( sample <= initial_samples ) {
	  result = make_float3(0,0,0);	  
	  for (int n=0; n < initial_samples; n++ ) {
		  ray_direction = getViewRay(d.x, d.y, pixsize);
		  optix::Ray ray = optix::make_Ray( cam_pos, ray_direction, 0, 0.0f, RT_DEFAULT_MAX);
		  rtTrace( top_object, ray, prd );
		  result += prd.result;
	  }
	  prd.result = result / float(initial_samples);
  } else {	  
	  ray_direction = getViewRay(d.x, d.y, pixsize);
	  optix::Ray ray = optix::make_Ray( cam_pos, ray_direction, 0, 0.0f, RT_DEFAULT_MAX);
	  rtTrace( top_object, ray, prd );
	  prd.result = (output_buffer[launch_index]*(sample-1) + prd.result) / float(sample);
  }

  output_buffer[launch_index] = prd.result;
}

RT_PROGRAM void exception()
{
  const unsigned int code = rtGetExceptionCode();
  //rtPrintf( "Exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
  printf( "Exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
  rtPrintExceptionDetails();
  output_buffer[launch_index] = bad_color;
}

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
rtDeclareVariable(float3,        cam_U, , );
rtDeclareVariable(float3,        cam_V, , );
rtDeclareVariable(float3,        cam_W, , );
rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(float,         scene_epsilon, , );
rtDeclareVariable(unsigned int,  sample, , );
rtBuffer<float3, 2>              output_buffer;
rtDeclareVariable(rtObject,      top_object, , );
rtBuffer<unsigned int, 2>        rnd_seeds;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

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

RT_PROGRAM void trace_primary ()
{
  float2 d = make_float2(launch_index) / make_float2(launch_dim) - 0.5f;  
  float pixsize = length ( cam_U ) / launch_dim.x;	
  float3 ray_direction;
  float3 result;

  PerRayData_radiance prd;
  prd.length = 0.f;
  prd.depth = 0;
  prd.rtype = 0;	// ANY_RAY

  int initial_samples = 1;
  
  if ( sample <= initial_samples ) {
	  result = make_float3(0,0,0);	  
	  for (int n=0; n < initial_samples; n++ ) {
		  ray_direction = normalize (d.x*cam_U + d.y*cam_V + cam_W + jitter_sample ( launch_index )*make_float3(pixsize,pixsize,pixsize) );
		  optix::Ray ray = optix::make_Ray( cam_pos, ray_direction, 0, 0.0f, RT_DEFAULT_MAX);
		  rtTrace( top_object, ray, prd );
		  result += prd.result;
	  }
	  prd.result = result / float(initial_samples);
  } else {	  
	  ray_direction = normalize (d.x*cam_U + d.y*cam_V + cam_W + jitter_sample ( launch_index )*make_float3(pixsize,pixsize,pixsize) );
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
  output_buffer[launch_index] = bad_color;
}

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

#ifndef EXTRA_MATH_H
#define EXTRA_MATH_H

#include <cuda.h>
#include <math.h>
#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
using namespace optix;

#define ANY_RAY			0
#define	SHADOW_RAY		1
#define VOLUME_RAY		2
#define MESH_RAY		3

//--------- Extra type defs
typedef uchar3				bool3;
typedef unsigned char		uchar;
typedef unsigned short		ushort;
typedef unsigned int		uint;
typedef unsigned long		ulong;
typedef unsigned long long	uint64;

//--------- Extra vec3 operators

inline __host__ __device__ float3 operator*(int3 &a, float3 b)
{
     return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__ float3 fabs3 ( float3 a )
{
	return make_float3 ( fabs(a.x), fabs(a.y), fabs(a.z) );
}
inline __host__ __device__ float3 floor3 ( float3 a )
{
	return make_float3 ( floorf(a.x), floorf(a.y), floorf(a.z) );
}
inline __host__ __device__ int3 iabs3 ( int3 a )
{
	return make_int3 ( abs(a.x), abs(a.y), abs(a.z) );
}
inline __host__ __device__ int3 isign3 ( float3 a )
{
	return make_int3 ( (a.x > 0) ? 1 : -1, (a.y > 0) ? 1 : -1, (a.z > 0) ? 1 : -1 );
	//return make_int3 ( copysignf(1,a.x), copysignf(1,a.y), copysignf(1,a.z) );
}
inline __host__ __device__ float3 fyzx ( float3 a )
{
	return make_float3 ( a.y, a.z, a.x );
}
inline __host__ __device__ float3 fzxy ( float3 a )
{
	return make_float3 ( a.z, a.x, a.y );
}
inline __host__ __device__ float3 fxyz ( float4 a )
{
	return make_float3 ( a.x, a.y, a.z );
}

inline __host__ __device__ bool3 make_bool3 ( uchar a, uchar b, uchar c )
{
	return make_uchar3( a, b, c );
}

inline __host__ __device__ bool3 lessThan3 ( float3 a, float3 b )
{
	return make_bool3 ( uchar(a.x < b.x), uchar(a.y < b.y), uchar(a.z < b.z) );
}
inline __host__ __device__ bool3 lessThanEqual3 ( float3 a, float3 b )
{
	return make_bool3 ( uchar(a.x <= b.x), uchar(a.y <= b.y), uchar(a.z <= b.z) );
}
inline __host__ __device__ float3 lerp3 ( float3 a, float3 b, float t )
{
	return make_float3 ( a.x+t*(b.x-a.x), a.y+t*(b.y-a.y), a.z+t*(b.z-a.z) );
}
inline __host__ __device__ float4 lerp4 ( float4 a, float4 b, float t )
{
	return make_float4 ( a.x+t*(b.x-a.x), a.y+t*(b.y-a.y), a.z+t*(b.z-a.z), a.w+t*(b.w-a.w)  );
}
inline __host__ __device__ float4 make_float4 ( uchar4 a )
{
	return make_float4 ( a.x/255.f, a.y/255.f, a.z/255.f, a.w/255.f );
}
////////////////////////////////////////////////////////////////////////////
// 4x4 matrix operations
////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float3 mmult(float3 vec, float* mtx)
{
	float3 p;
	p.x = vec.x * mtx[0] + vec.y*mtx[4] + vec.z*mtx[8] + mtx[12];
	p.y = vec.x * mtx[1] + vec.y*mtx[5] + vec.z*mtx[9] + mtx[13];
	p.z = vec.x * mtx[2] + vec.y*mtx[6] + vec.z*mtx[10] + mtx[14];
	return p;
}

//--------- Random number generator

// Generate random unsigned int in [0, 2^24)
static __host__ __device__ __inline__ unsigned int lcg(unsigned int &prev)
{
  const unsigned int LCG_A = 1664525u;
  const unsigned int LCG_C = 1013904223u;
  prev = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
}

static __host__ __device__ __inline__ unsigned int lcg2(unsigned int &prev)
{
  prev = (prev*8121 + 28411)  % 134456;
  return prev;
}

// Generate random float in [0, 1)
static __host__ __device__ __inline__ float rnd(unsigned int &prev)
{
  return ((float) lcg(prev) / (float) 0x01000000);
}

//--------- Intersection refinement

// Plane intersection -- used for refining triangle hit points. Skips zero denom check (for rays perpindicular to plane normal)
// since we know that the ray intersects the plane.
static __device__ __inline__ float intersectPlane( const optix::float3& origin, const optix::float3& direction, const optix::float3& normal, const optix::float3& point )
{
  return -( optix::dot( normal, origin-point ) ) / optix::dot( normal, direction );
}

// Offset the hit point using integer arithmetic
static __device__ __inline__ optix::float3 offset( const optix::float3& hit_point, const optix::float3& normal )
{
  using namespace optix;
  const float epsilon = 1.0e-4f;
  const float offset  = 4096.0f*2.0f;

  float3 offset_point = hit_point;
  if( (__float_as_int( hit_point.x )&0x7fffffff)  < __float_as_int( epsilon ) ) {
    offset_point.x += epsilon * normal.x;
  } else {
    offset_point.x = __int_as_float( __float_as_int( offset_point.x ) + int(copysign( offset, hit_point.x )*normal.x) );
  }

  if( (__float_as_int( hit_point.y )&0x7fffffff) < __float_as_int( epsilon ) ) {
    offset_point.y += epsilon * normal.y;
  } else {
    offset_point.y = __int_as_float( __float_as_int( offset_point.y ) + int(copysign( offset, hit_point.y )*normal.y) );
  }

  if( (__float_as_int( hit_point.z )&0x7fffffff)  < __float_as_int( epsilon ) ) {
    offset_point.z += epsilon * normal.z;
  } else {
    offset_point.z = __int_as_float( __float_as_int( offset_point.z ) + int(copysign( offset, hit_point.z )*normal.z) );
  }

  return offset_point;
}

// Refine the hit point to be more accurate and offset it for reflection and refraction ray starting points.
static __device__ __inline__ void refine_and_offset_hitpoint ( const optix::float3& original_hit_point, const optix::float3& direction, const optix::float3& normal, const optix::float3& p, optix::float3& back_hit_point, optix::float3& front_hit_point )
{
  using namespace optix;  
  float  refined_t          = intersectPlane( original_hit_point, direction, normal, p );  // Refine hit point
  float3 refined_hit_point  = original_hit_point + refined_t*direction;  
  if( dot( direction, normal ) > 0.0f ) {						
    back_hit_point  = offset( refined_hit_point,  normal );		// Offset hit point
    front_hit_point = offset( refined_hit_point, -normal );
  } else {
    back_hit_point  = offset( refined_hit_point, -normal );
    front_hit_point = offset( refined_hit_point,  normal );
  }
}

#endif
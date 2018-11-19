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

// This is to be plugged into an RTgeometry object to represent
// a triangle mesh with a vertex buffer of triangle soup (triangle list)
// with an interleaved position, normal, texturecoordinate layout.

rtBuffer<float3> vertex_buffer;     
rtBuffer<float3> normal_buffer;
rtBuffer<float2> texcoord_buffer;
rtBuffer<int3>   vindex_buffer;    // position indices 
rtBuffer<int3>   nindex_buffer;    // normal indices
rtBuffer<int3>   tindex_buffer;    // texcoord indices
rtBuffer<uint>   mindex_buffer;    // per-face material index
rtDeclareVariable(float3, back_hit_point, attribute back_hit_point, ); 
rtDeclareVariable(float3, front_hit_point, attribute front_hit_point, ); 
rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
rtDeclareVariable(float4, deep_color, attribute deep_color, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

struct RayInfo
{
	float3	result;
	float	length;
	float	alpha;
	int		depth;
	int		rtype;
};
rtDeclareVariable(RayInfo, ray_info, rtPayload, );


RT_PROGRAM void mesh_intersect( int primIdx )
{
  int3 v_idx = vindex_buffer[primIdx];

  float3 p0 = vertex_buffer[ v_idx.x ];
  float3 p1 = vertex_buffer[ v_idx.y ];
  float3 p2 = vertex_buffer[ v_idx.z ];

  // Intersect ray with triangle
  float3 n;
  float  t, beta, gamma;
  if( intersect_triangle( ray, p0, p1, p2, n, t, beta, gamma ) ) {

    if(  rtPotentialIntersection( t ) ) {

      // Calculate normals and tex coords 
      float3 geo_n = normalize( n );
      int3 n_idx = nindex_buffer[ primIdx ];
      
	  shading_normal = geo_n;
	  
	  // INTERPOLATED NORMALS
	  if ( normal_buffer.size() == 0 || n_idx.x < 0 || n_idx.y < 0 || n_idx.z < 0 ) {
        shading_normal = geo_n;
      } else {
        float3 n0 = normal_buffer[ n_idx.x ];
        float3 n1 = normal_buffer[ n_idx.y ];
        float3 n2 = normal_buffer[ n_idx.z ];
        shading_normal = normalize( n1*beta + n2*gamma + n0*(1.0f-beta-gamma) );
      }

      geometric_normal = geo_n;

      int3 t_idx = tindex_buffer[ primIdx ];
      if ( texcoord_buffer.size() == 0 || t_idx.x < 0 || t_idx.y < 0 || t_idx.z < 0 ) {
        texcoord = make_float3( 0.0f, 0.0f, 0.0f );
      } else {

        float2 t0 = texcoord_buffer[ t_idx.x ];
        float2 t1 = texcoord_buffer[ t_idx.y ];
        float2 t2 = texcoord_buffer[ t_idx.z ];
        texcoord = make_float3( t1*beta + t2*gamma + t0*(1.0f-beta-gamma) );
      }

      refine_and_offset_hitpoint( ray.origin + t*ray.direction, ray.direction,
                                  geo_n, p0,
                                  back_hit_point, front_hit_point );
	  
	  deep_color = make_float4(0, 0, 0, 0);	  

      rtReportIntersection(mindex_buffer[primIdx]);
    }
  }
}


RT_PROGRAM void mesh_bounds (int primIdx, float result[6])
{
  const int3 v_idx = vindex_buffer[primIdx];

  const float3 v0 = vertex_buffer[ v_idx.x ];
  const float3 v1 = vertex_buffer[ v_idx.y ];
  const float3 v2 = vertex_buffer[ v_idx.z ];
  const float  area = length(cross(v1-v0, v2-v0));

  optix::Aabb* aabb = (optix::Aabb*)result;

  if(area > 0.0f && !isinf(area)) {
    aabb->m_min = fminf( fminf( v0, v1), v2 );
    aabb->m_max = fmaxf( fmaxf( v0, v1), v2 );
  } else {
    aabb->invalidate();
  }
}


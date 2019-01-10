
#include <stdio.h>
#include "cuda_math.cuh"

typedef unsigned char		uchar;
typedef unsigned int		uint;
typedef unsigned short		ushort;
typedef unsigned long		ulong;
typedef unsigned long long	uint64;

//-------------------------------- GVDB Data Structure
#define CUDA_PATHWAY
#include "cuda_gvdb_scene.cuh"		// GVDB Scene
#include "cuda_gvdb_nodes.cuh"		// GVDB Node structure
#include "cuda_gvdb_geom.cuh"		// GVDB Geom helpers
#include "cuda_gvdb_dda.cuh"		// GVDB DDA 
#include "cuda_gvdb_raycast.cuh"	// GVDB Raycasting
//--------------------------------


inline __host__ __device__ float3 reflect3 (float3 i, float3 n)
{
	return i - 2.0f * n * dot(n,i);
}

// Custom raycast kernel
extern "C" __global__ void raycast_kernel ( VDBInfo* gvdb, uchar chan, uchar4* outBuf )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= scn.width || y >= scn.height ) return;
	
	float3 hit = make_float3(NOHIT,NOHIT,NOHIT);	
	float4 clr = make_float4(1,1,1,1);
	float3 norm;
	float3 rdir = normalize ( getViewRay ( (float(x)+0.5)/scn.width, (float(y)+0.5)/scn.height ) );	

	// Ray march - trace a ray into GVDB and find the closest hit point
	rayCast ( gvdb, chan, scn.campos, rdir, hit, norm, clr, raySurfaceTrilinearBrick );

	if ( hit.z != NOHIT) {		
		float3 lightdir = normalize ( scn.light_pos - hit );

		// Shading - custom look 
		float3 eyedir	= normalize ( scn.campos - hit );		
		float3 R		= normalize ( reflect3 ( eyedir, norm ) );		// reflection vector
		float diffuse	= max(0.0f, dot( norm, lightdir ));		
		float refl		= min(1.0f, max(0.0f, R.y ));		
		clr = diffuse*0.6 + refl * make_float4(0,0.3,0.7, 1.0);

	} else {
		clr = make_float4 ( 0.0, 0.0, 0.1, 1.0 );
	}	
	outBuf [ y*scn.width + x ] = make_uchar4( clr.x*255, clr.y*255, clr.z*255, 255 );	
}

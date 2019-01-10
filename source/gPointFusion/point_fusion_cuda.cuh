
#ifndef DEF_KERN_CUDA
	#define DEF_KERN_CUDA

	#include <stdio.h>
	#include <math.h>

	#define CUDA_KERNEL
	typedef unsigned int		uint;
	typedef unsigned short int	ushort;
	typedef unsigned char		uchar;

	#define ALIGN(x)	__align__(x)
	
	extern "C" {
		__global__ void scanBuildings ( float3 pos, int3 res, int num_obj, float tmax );
	}

#endif

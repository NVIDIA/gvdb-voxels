/*
  FLUIDS v.1 - SPH Fluid Simulator for CPU and GPU
  Copyright (C) 2008. Rama Hoetzlein, http://www.rchoetzlein.com

  ZLib license
  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

#ifndef DEF_FLUID
	#define DEF_FLUID
	
	#include <cuda.h>
	#include <curand.h>
	#include "gvdb_vec.h"
	using namespace nvdb;

	typedef	unsigned int		uint;	
	typedef	unsigned short int	ushort;	

	struct NList {
		int num;
		int first;
	};
	struct Fluid {						// offset - TOTAL: 72 (must be multiple of 12)
		Vector3DF		pos;			// 0
		Vector3DF		vel;			// 12
		Vector3DF		veleval;		// 24
		Vector3DF		force;			// 36
		float			pressure;		// 48
		float			density;		// 52
		int				grid_cell;		// 56
		int				grid_next;		// 60
		uint			clr;			// 64
		uint			state;			// 68
	};

	#define FPOS		0		// particle buffers
	#define FVEL		1
	#define FVEVAL		2
	#define FFORCE		3
	#define FPRESS		4
	#define FDENSITY	5
	#define FAGE		6
	#define FCLR		7
	#define FGCELL		8
	#define FGNDX		9
	#define FGNEXT		10
	#define FNBRNDX		11		// particle neighbors (optional)
	#define FNBRCNT		12
	#define FCLUSTER	13	
	#define FGRID		14		// uniform acceleration grid
	#define FGRIDCNT	15
	#define	FGRIDOFF	16
	#define FGRIDACT	17
	#define FSTATE		18
	#define FBRICK		19
	#define FPARAMS		20		// fluid parameters
	#define FAUXARRAY1	21		// auxiliary arrays (prefix sums)
	#define FAUXSCAN1   22
	#define FAUXARRAY2	23
	#define FAUXSCAN2	24
	#define MAX_BUF		25

	#ifdef CUDA_KERNEL
		#define	CALLFUNC	__device__
	#else
		#define CALLFUNC
	#endif		

	// Particle & Grid Buffers
	struct FBufs {
#ifndef CUDA_KERNEL
		inline ~FBufs() {
			for (int i = 0; i < MAX_BUF; i++) {
				if (mcpu[i] != nullptr) {
					free(mcpu[i]);
				}
				if (mgpu[i] != 0) {
					cuMemFree(mgpu[i]);
				}
			}
		}
#endif

		inline CALLFUNC void setBuf(int n, char* buf) {
			if (mcpu[n] != nullptr) {
				free(mcpu[n]);
			}
			mcpu[n] = buf;
		}

		#ifdef CUDA_KERNEL
			// on device, access data via gpu pointers 
			inline CALLFUNC Vector3DF* bufV3(int n)		{ return (Vector3DF*) mgpu[n]; }
			inline CALLFUNC float3* bufF3(int n)		{ return (float3*) mgpu[n]; }
			inline CALLFUNC float*  bufF (int n)		{ return (float*)  mgpu[n]; }
			inline CALLFUNC uint*   bufI (int n)		{ return (uint*)   mgpu[n]; }
			inline CALLFUNC char*   bufC (int n)		{ return (char*)   mgpu[n]; }			
		#else
			// on host, access data via cpu pointers
			inline CALLFUNC Vector3DF* bufV3(int n)		{ return (Vector3DF*) mcpu[n]; }
			inline CALLFUNC float3* bufF3(int n)		{ return (float3*) mcpu[n]; }
			inline CALLFUNC float*  bufF (int n)		{ return (float*)  mcpu[n]; }
			inline CALLFUNC uint*   bufI (int n)		{ return (uint*)   mcpu[n]; }
			inline CALLFUNC char*   bufC (int n)		{ return (char*)   mcpu[n]; }				
		#endif	

		char* mcpu[MAX_BUF] = { nullptr };

		#ifdef CUDA_KERNEL
			char* mgpu[MAX_BUF] = { nullptr };		// on device, pointer is local 
		#else			
			CUdeviceptr		mgpu[MAX_BUF] = { 0 };		// on host, gpu is a device pointer
			CUdeviceptr		gpu (int n )	{ return mgpu[n]; }
			CUdeviceptr*	gpuptr (int n )	{ return &mgpu[n]; }		
		#endif			
	};

	// Temporary sort buffer offsets
	#define BUF_POS			0
	#define BUF_VEL			(sizeof(float3))
	#define BUF_VELEVAL		(BUF_VEL + sizeof(float3))
	#define BUF_FORCE		(BUF_VELEVAL + sizeof(float3))
	#define BUF_PRESS		(BUF_FORCE + sizeof(float3))
	#define BUF_DENS		(BUF_PRESS + sizeof(float))
	#define BUF_GCELL		(BUF_DENS + sizeof(float))
	#define BUF_GNDX		(BUF_GCELL + sizeof(uint))
	#define BUF_CLR			(BUF_GNDX + sizeof(uint))

	#define OFFSET_POS		0
	#define OFFSET_VEL		12
	#define OFFSET_VELEVAL	24
	#define OFFSET_FORCE	36
	#define OFFSET_PRESS	48
	#define OFFSET_DENS		52
	#define OFFSET_CELL		56
	#define OFFSET_GCONT	60
	#define OFFSET_CLR		64	

	// Fluid Parameters (stored on both host and device)
	struct FParams {
		int				numThreads, numBlocks;
		int				gridThreads, gridBlocks;	

		int				szPnts, szHash, szGrid;
		int				stride, pnum;
		int				chk;
		float			pdist, pmass, prest_dens;
		float			pextstiff, pintstiff;
		float			pradius, psmoothradius, r2, psimscale, pvisc;
		float			pforce_min, pforce_max, pforce_freq, pground_slope;
		float			pvel_limit, paccel_limit, pdamp;
		float3			pboundmin, pboundmax, pgravity;
		float			AL, AL2, VL, VL2;

		float			d2, rd2, vterm;		// used in force calculation		 
		
		float			poly6kern, spikykern, lapkern, gausskern;

		float3			gridSize, gridDelta, gridMin, gridMax;
		int3			gridRes, gridScanMax;
		int				gridSrch, gridTotal, gridAdjCnt, gridActive;
		int				gridAdj[64];

		int3			brickRes;
		int				pemit;
	};

#endif /*PARTICLE_H_*/

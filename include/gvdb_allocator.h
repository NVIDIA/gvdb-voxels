//--------------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2017, NVIDIA Corporation. 
//
// Redistribution and use in source and binary forms, with or without modification, 
// are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer 
//    in the documentation and/or  other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived 
//    from this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
// BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
// SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE 
// OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// Version 1.0: Rama Hoetzlein, 5/1/2017
//----------------------------------------------------------------------------------

#ifndef DEF_ALLOCATOR
	#define DEF_ALLOCATOR

	#include "gvdb_types.h"		
	#include "gvdb_vec.h"
	#include <vector>
	#include <cuda.h>

	// Maximum number of GVDB Pool levels
	#define MAX_POOL		10

	// Global CUDA helpers
	#define MIN_RUNTIME_VERSION		4010
	#define MIN_COMPUTE_VERSION		0x20
	extern void				StartCuda ( int devid, bool verbose );	
	extern GVDB_API bool	cudaCheck ( CUresult e, char* func, char* api);
	extern GVDB_API float	cudaGetFreeMem ();

	namespace nvdb {

	class Allocator;
	
	// Pool Pointer
	// Smart pointer for all CPU/GPU pointers 
	struct GVDB_API DataPtr {
		DataPtr () { type=T_UCHAR; num=0; max=0; size=0; stride=0; cpu=0; glid=0; grsc=0; gpu=0; }		
		char		type;				// data type
		char		apron;				// apron size
		uint64		num, max;			// element count
		uint64		size;				// size of data
		uint64		stride;				// stride of data	
		Vector3DI	subdim;				// subdim		
		Allocator*	alloc;				// allocator instance
		char*		cpu;				// cpu pointer		
		int			glid;				// gpu opengl id
		CUgraphicsResource	grsc;		// gpu graphics resource (cuda)		
		CUarray		garray;				// gpu array (cuda)
		CUdeviceptr	gpu;				// gpu pointer (cuda)		
	};	
	
	// Element conversions
	// Used to pack/unpack the group, level, and index from a pool reference
	inline uint64 Elem ( uchar grp, uchar lev, uint64 ndx )	{ return uint64(grp) | (uint64(lev) << 8) | (uint64(ndx) << 16); }
	inline uchar ElemGrp ( uint64 id )						{ return uchar(id & 0xFF); }
	inline uchar ElemLev ( uint64 id )						{ return uchar((id>>8) & 0xFF); }
	inline uint64 ElemNdx ( uint64 id )						{ return id >> 16; }	

	// Allocator
	// Primary memory handler for GVDB
	class Allocator {
	public:
		Allocator();
		
		// Pool functions
		void	PoolCreate ( uchar grp, uchar lev, uint64 width, uint64 initmax, bool bGPU );		// create a pool		
		void	PoolReleaseAll ();																	// release all pools
		void	PoolCommit ( int grp, int lev );
		void	PoolCommitAll ();
		void	PoolCommitAtlasMap ();
		void	PoolEmptyAll ();
		void	PoolClearCPU();
		void	PoolFetchAll();
		void	PoolFetch(int grp, int lev );

		uint64	PoolAlloc ( uchar grp, uchar lev, bool bGPU );		// allocate on pool
		void	PoolFree ( uint64 id );								// free from pool		
		char*	PoolData ( uint64 id );								// get data ptr
		char*	PoolData ( uchar grp, uchar lev, uint64 ndx );
		uint64* PoolData64 ( uint64 id );		
		uint64	getPoolCnt ( uchar grp, uchar lev )	{ return mPool[grp][lev].num; }
		uint64  getPoolMax ( uchar grp, uchar lev ) { return mPool[grp][lev].max; }
		char*	getPoolCPU ( uchar grp, uchar lev ) { return mPool[grp][lev].cpu; }
		uint64  getPoolSize ( uchar grp, uchar lev ) { return mPool[grp][lev].size; }
		CUdeviceptr	getPoolGPU ( uchar grp, uchar lev )	{ return mPool[grp][lev].gpu; }
		uint64	getPoolWidth ( uchar grp, uchar lev );					// get pool width		
		int		getPoolMem ();		
		void	PoolWrite ( FILE* fp, uchar grp, uchar lev );
		void	PoolRead ( FILE* fp, uchar grp, uchar lev, int cnt, int wid );
		
		// Texture functions
		bool	TextureCreate ( uchar chan, uchar dtype, Vector3DI res, bool bCPU, bool bGL );
		void	AllocateTextureGPU ( DataPtr& p, uchar dtype, Vector3DI res, bool bGL, uint64 preserve );
		void	AllocateTextureCPU ( DataPtr& p, uint64 sz, bool bCPU, uint64 preserve );
		void	AllocateAtlasMap ( int stride, Vector3DI axiscnt );

		// Atlas functions
		bool	AtlasCreate ( uchar chan, uchar dtype, Vector3DI leafdim, Vector3DI axiscnt, char apr, uint64 map_wid, bool bCPU, bool bGL );		
		bool	AtlasResize ( uchar chan, uint64 max_leaf );
		bool	AtlasResize ( uchar chan, int cx, int cy, int cz );
		void	AtlasSetNum ( uchar chan, int n );
		void	AtlasReleaseAll ();
		void	AtlasEmptyAll ();
		bool	AtlasAlloc ( uchar chan, Vector3DI& val );
		void	AtlasFill ( uchar chan );		
		void	AtlasCommit ( uchar chan );										// commit CPU atlas data to GPU
		void	AtlasCommitFromCPU ( uchar chan, uchar* src );					// host-to-device copy from 3D to 3D (entire vol)				
		void	AtlasAppendLinearCPU ( uchar chan, int n, float* src );			// CPU only, append 3D data linearly to end of atlas (no GPU update)
		void	AtlasCopyTex ( uchar chan, Vector3DI val, const DataPtr& src );		// device-to-device copy 3D sub-vol into 3D 
		void	AtlasCopyTexZYX ( uchar chan, Vector3DI val, const DataPtr& src );	// device-to-device copy 3D sub-vol into 3D, ZYX order 		
		void	AtlasCopyLinear ( uchar chan, Vector3DI offset, CUdeviceptr gpu_buf );
		void	AtlasRetrieveSlice ( uchar chan, int y, int sz, CUdeviceptr tempmem, uchar* dest );
		void	AtlasWriteSlice ( uchar chan, int slice, int sz, CUdeviceptr gpu_buf, uchar* cpu_src );
		void	AtlasRetrieveTexXYZ ( uchar chan, Vector3DI val, DataPtr& buf );		
		int		getAtlasMem ();
		void	AtlasWrite ( FILE* fp, uchar chan );		
		void	AtlasRead ( FILE* fp, uchar chan, uint64 asize );

		//void	CreateImage ( DataPtr& p, nvImg& img );
		void	CreateMemLinear ( DataPtr& p, char* dat, int sz );
		void	CreateMemLinear ( DataPtr& p, char* dat, int stride, int cnt, bool bCPU );
		void    FreeMemLinear ( DataPtr& p );
		void    RetrieveMem ( DataPtr& p);
		void    CommitMem ( DataPtr& p);

		// OpenGL functions
		void	AtlasRetrieveGL ( uchar chan, char* dest );


		// Query functions
		char*		getAtlasNode ( uchar chan, Vector3DI val );
		CUdeviceptr getAtlasMapGPU ( uchar chan )		{ return mAtlasMap[chan].gpu; }

		int		getSize ( uchar dtype );
		int		getNumAtlas ()					{ return (int) mAtlas.size(); }
		DataPtr	getAtlas ( uchar chan )			{ return mAtlas[chan]; }	
		float*	getAtlasCPU ( uchar chan )		{ return (float*) mAtlas[chan].cpu; }
		int		getAtlasGLID ( uchar chan )		{ return mAtlas[chan].glid; }
		uint64  getAtlasSize ( uchar chan )		{ return (uint64) mAtlas[chan].size; }
		Vector3DI getAtlasPos ( uchar chan, uint64 id );
		Vector3DI getAtlasRes ( uchar chan );
		int		getAtlasBrickres ( uchar chan);		
		int		getNumLevels ()		{ return (int) mPool[0].size(); }
		DataPtr* getPool(uchar grp, uchar lev);

	private:

		std::vector< DataPtr >		mPool[ MAX_POOL ];
		std::vector< DataPtr >		mAtlas;
		std::vector< DataPtr >		mAtlasMap;

		int							mVFBO[2];

		static bool					bAllocator;
		static CUmodule				cuAllocatorModule;
		static CUfunction			cuFillTex;	
		static CUfunction			cuCopyTexC;
		static CUfunction			cuCopyTexF;
		static CUfunction			cuCopyBufToTexC;
		static CUfunction			cuCopyBufToTexF;
		static CUfunction			cuCopyTexZYX;
		static CUfunction			cuRetrieveTexXYZ;
		static CUfunction			cuSliceTexToBufF;
		static CUfunction			cuSliceTexToBufC;
		static CUfunction			cuSliceBufToTexF;		
		static CUfunction			cuSliceBufToTexC;
		
		static CUsurfref			cuSurfWrite;
		static CUtexref				cuSurfReadC;
		static CUtexref				cuSurfReadF;
		
	};

	}

#endif



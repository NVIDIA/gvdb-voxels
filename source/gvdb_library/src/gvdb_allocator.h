//-----------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2016 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
// 
// Version 1.0: Rama Hoetzlein, 5/1/2017
// Version 1.1: Rama Hoetzlein, 3/25/2018
//-----------------------------------------------------------------------------

#ifndef DEF_ALLOCATOR
	#define DEF_ALLOCATOR

	#include "gvdb_types.h"		
	#include "gvdb_vec.h"
	#include <vector>
	#include <cuda.h>
	using namespace nvdb;

	// Maximum number of GVDB Pool levels
	#define MAX_POOL		10

	// Global CUDA helpers
	#define MIN_RUNTIME_VERSION		4010
	#define MIN_COMPUTE_VERSION		0x20
	extern void				StartCuda( int devsel, CUcontext ctxsel, CUdevice& dev, CUcontext& ctx, CUstream* strm, bool verbose );
	extern GVDB_API bool	cudaCheck ( CUresult e, const char* obj, const char* method, const char* apicall, const char* arg, bool bDebug);
	extern GVDB_API Vector3DF cudaGetMemUsage(); // Returns (used MBs, free MBs, total MBs) of GPU memory.

	namespace nvdb {

	class Allocator;
	
	// Pool Pointer
	// Smart pointer for all CPU/GPU pointers 
	struct GVDB_API DataPtr {
		DataPtr ();		
		char		type;				// data type
		char		apron;				// apron size
		char		filter;				// filter mode
		char		border;				// border mode
		uint64		max;				// max element count
		uint64		lastEle;			// total element count
		uint64		usedNum;			// used element count
		uint64		size;				// size of data
		uint64		stride;				// stride of data	
		Vector3DI	subdim;				// subdim		
		Allocator*	alloc;				// allocator instance
		char*		cpu;				// cpu pointer		
		int			glid;				// gpu opengl id
		CUgraphicsResource	grsc;		// gpu graphics resource (cuda)		
		CUarray		garray;				// gpu array (cuda)
		CUdeviceptr	gpu;				// gpu pointer (cuda)	
		CUtexObject tex_obj;			// gpu texture object
		CUsurfObject surf_obj;			// gpu surface object
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
		~Allocator();
		
		// Pool functions
		void	PoolCreate ( uchar grp, uchar lev, uint64 width, uint64 initmax, bool bGPU );		// create a pool		
		void	PoolReleaseAll ();																	// release all pools
		void	PoolCommit ( int grp, int lev );
		void	PoolCommitAll ();		
		void	PoolEmptyAll ();

		void	PoolClearCPU();
		void	PoolFetchAll();
		void	PoolFetch(int grp, int lev );

		uint64	PoolAlloc ( uchar grp, uchar lev, bool bGPU );		// allocate on pool
		void	PoolFree ( uint64 id );								// free from pool		
		char*	PoolData ( uint64 id );								// get data ptr
		char*	PoolData ( uchar grp, uchar lev, uint64 ndx );
		uint64* PoolData64 ( uint64 id );		
		uint64	getPoolUsedCnt ( uchar grp, uchar lev )	{ return mPool[grp][lev].usedNum; }
		uint64	getPoolTotalCnt ( uchar grp, uchar lev )	{ return mPool[grp][lev].lastEle; }
		uint64  getPoolMax ( uchar grp, uchar lev ) { return mPool[grp][lev].max; }
		char*	getPoolCPU ( uchar grp, uchar lev ) { return mPool[grp][lev].cpu; }
		uint64  getPoolSize ( uchar grp, uchar lev ) { return mPool[grp][lev].size; }
		CUdeviceptr	getPoolGPU ( uchar grp, uchar lev )	{ return mPool[grp][lev].gpu; }
		uint64	getPoolWidth ( uchar grp, uchar lev );					// get pool width		
		uint64	getPoolMem ();
		void	PoolWrite ( FILE* fp, uchar grp, uchar lev );
		void	PoolRead ( FILE* fp, uchar grp, uchar lev, int cnt, int wid );
		
		void	SetPoolUsedCnt ( uchar grp, uchar lev, uint64 cnt ) { mPool[grp][lev].usedNum = cnt;}

		// Texture functions
		bool	TextureCreate ( uchar chan, uchar dtype, Vector3DI res, bool bCPU, bool bGL );
		void	AllocateTextureGPU ( DataPtr& p, uchar dtype, Vector3DI res, bool bGL, uint64 preserve );
		void	AllocateTextureCPU ( DataPtr& p, uint64 sz, bool bCPU, uint64 preserve );		

		// Atlas functions
		bool	AtlasCreate ( uchar chan, uchar dtype, Vector3DI leafdim, Vector3DI axiscnt, char apr, uint64 map_wid, bool bCPU, bool bGL );		
		bool	AtlasResize ( uchar chan, uint64 max_leaf );
		bool	AtlasResize ( uchar chan, int cx, int cy, int cz );
		void	AtlasSetNum ( uchar chan, int n );
		void	AtlasSetFilter ( uchar chan, int filter, int border );
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
		uint64	getAtlasMem (); // Returns the number of megabytes of memory used by the atlas.
		void	AtlasWrite ( FILE* fp, uchar chan );		
		void	AtlasRead ( FILE* fp, uchar chan, uint64 asize );

		//void	CreateImage ( DataPtr& p, nvImg& img );
		void	CreateMemLinear ( DataPtr& p, char* dat, int sz );
		void	CreateMemLinear ( DataPtr& p, char* dat, int stride, uint64 cnt, bool bCPU, bool bAllocHost = false  );
		void    FreeMemLinear ( DataPtr& p );
		void    RetrieveMem ( DataPtr& p);
		void    CommitMem ( DataPtr& p);

		// OpenGL functions
		void	AtlasRetrieveGL ( uchar chan, char* dest );

		// Atlas Mapping		
		void	AllocateAtlasMap(int stride, Vector3DI axiscnt);
		void	PoolCommitAtlasMap();
		char*	getAtlasMapNode (uchar chan, Vector3DI val);
		CUdeviceptr getAtlasMapGPU(uchar chan) { return (mAtlasMap.size()==0) ? 0 : mAtlasMap[chan].gpu; }

		// Neighbor Table
		void	AllocateNeighbors(uint64 cnt);		
		void	CommitNeighbors();
		DataPtr* getNeighborTable()				{ return &mNeighbors; }		

		// Query functions		
		int		getSize ( uchar dtype );
		int		getNumAtlas ()					{ return (int) mAtlas.size(); }
		DataPtr	getAtlas ( uchar chan )			{ return mAtlas[chan]; }	
		float*	getAtlasCPU ( uchar chan )		{ return (float*) mAtlas[chan].cpu; }
		int		getAtlasGLID ( uchar chan )		{ return mAtlas[chan].glid; }
		uint64  getAtlasSize ( uchar chan )		{ return (uint64) mAtlas[chan].size; }
		Vector3DI getAtlasPos ( uchar chan, uint64 id );
		Vector3DI getAtlasRes ( uchar chan );
		Vector3DI getAtlasPackres(uchar chan);
		Vector3DI getAtlasCnt(uchar chan)		{ return mAtlas[chan].subdim; }		
		int		getAtlasBrickres ( uchar chan);				
		int		getAtlasBrickwid ( uchar chan);
		int		getNumLevels ()		{ return (int) mPool[0].size(); }
		DataPtr* getPool(uchar grp, uchar lev);

		void CopyChannel(int chanDst, int chanSrc);

		void SetStream(CUstream strm) { mStream = strm;  }
		CUstream getStream() { return mStream; }
		void SetDebug(bool b) { mbDebug = b; }

	private:

		std::vector< DataPtr >		mPool[ MAX_POOL ];
		std::vector< DataPtr >		mAtlas;
		std::vector< DataPtr >		mAtlasMap;
		DataPtr						mNeighbors;
		bool						mbDebug;

		int							mVFBO[2];

		CUstream					mStream;

		CUmodule					cuAllocatorModule;
		CUfunction					cuFillTex;	
		CUfunction					cuCopyTexC;
		CUfunction					cuCopyTexF;
		CUfunction					cuCopyBufToTexC;
		CUfunction					cuCopyBufToTexF;
		CUfunction					cuCopyTexZYX;
		CUfunction					cuRetrieveTexXYZ;
	};

	}

#endif



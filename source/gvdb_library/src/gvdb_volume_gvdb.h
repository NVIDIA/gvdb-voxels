//-----------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2016 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
// 
// Version 1.0: Rama Hoetzlein, 5/1/2017
// Version 1.1: Rama Hoetzlein, 3/25/2018
//-----------------------------------------------------------------------------

#ifndef DEF_VOL_GVDB
	#define DEF_VOL_GVDB

	#include "gvdb_types.h"
	#include "gvdb_node.h"	
	#include "gvdb_volume_base.h"
	#include "gvdb_allocator.h"		

	using namespace nvdb;

	#ifdef BUILD_OPENVDB
		#include <openvdb/openvdb.h>

		// OpenVDB <3,3,3,4> support
		using FloatTree34 = openvdb::tree::Tree5<float, 3, 3, 3, 4>::Type;
		using Vec3fTree34 = openvdb::tree::Tree5<openvdb::Vec3f, 3, 3, 3, 4>::Type;
		using FloatGrid34 = openvdb::Grid<FloatTree34>;
		using Vec3fGrid34 = openvdb::Grid<Vec3fTree34>;
		using GridType34 = FloatGrid34;
		using TreeType34F = FloatGrid34::TreeType;
		using TreeType34VF = Vec3fGrid34::TreeType;
	
		// OpenVDB <5,4,3> support
		using FloatGrid543 = openvdb::FloatGrid;
		using Vec3fGrid543 = openvdb::Vec3fGrid;
		using GridType543 = FloatGrid543;
		using TreeType543F = FloatGrid543::TreeType;
		using TreeType543VF = Vec3fGrid543::TreeType;
	#endif

	#define MAXLEV			10

	class Volume3D;

	namespace nvdb {

	struct AtlasNode {
		Vector3DI	mPos;	
		int			mLeafNode;
	};

	struct Stat {
		Stat ()	{ num=0; cover=0; occupy=0; mem_node=0; mem_mask=0; mem_compact=0; mem_full=0;}
		slong	num;			// number of nodes at this level
		slong	cover;			// total coverage of all nodes (addressable bits)
		slong	occupy;			// number of set bits (occupied bits)
		slong	mem_node;		// memory used 
		slong	mem_mask;
		slong	mem_compact;
		slong	mem_full;
	};
	typedef std::vector<Stat>	statVec;

	// Contains information for a GVDB volume.
	// Levels vary from 0 (bricks) to top_lev (root node).
	struct ALIGN(16) VDBInfo {
		int				dim[MAXLEV]; // Log base 2 of lateral resolution of each node per level
		int				res[MAXLEV]; // Lateral resolution of each node per level
		Vector3DF		vdel[MAXLEV]; // How many voxels on a side a child of each level covers
		Vector3DI		noderange[MAXLEV]; // How many voxels on a side a node of each level covers
		int				nodecnt[MAXLEV]; // Total number of allocated nodes per level
		int				nodewid[MAXLEV]; // Size of a node at each level in bytes
		int				childwid[MAXLEV]; // Size of the child list per node at each level in bytes
		CUdeviceptr		nodelist[MAXLEV]; // GPU pointer to each level's pool group 0 (nodes)
		CUdeviceptr		childlist[MAXLEV]; // GPU pointer to each level's pool group 1 (child lists)
		CUdeviceptr		atlas_map; // GPU pointer to the atlas map (which maps from atlas to world space)
		Vector3DI		atlas_cnt; // Number of bricks on each axis of the atlas
		Vector3DI		atlas_res; // Total resolution in voxels of the atlas
		int				atlas_apron; // Apron size
		int				brick_res; // Resolution of a single brick
		int				apron_table[8]; // Unused
		int				top_lev; // Top level (i.e. tree spans from voxels to level 0 to level top_lev)
		int				max_iter; // Unused
		float			epsilon; // Epsilon used for voxel ray tracing
		bool			update; // Whether this information needs to be updated from the latest volume data
		uchar			clr_chan; // Index of the color channel for rendering color information
		Vector3DF		bmin; // Inclusive minimum of axis-aligned bounding box in voxels
		Vector3DF		bmax; // Inclusive maximum of axis-aligned bounding box in voxels
		CUtexObject		volIn[MAX_CHANNEL]; // Texture reference (read plus interpolation) to atlas per channel
		CUsurfObject	volOut[MAX_CHANNEL]; // Surface reference (read and write) to atlas per channel
	};

	struct ALIGN(16) ScnInfo {
		int			width;
		int			height;
		float		camnear;
		float   	camfar;
		Vector3DF	campos;
		Vector3DF	cams;
		Vector3DF	camu;
		Vector3DF	camv;
		Vector3DF	light_pos;
		Vector3DF	slice_pnt;
		Vector3DF	slice_norm;
		Vector3DF	shadow_params;
		Vector4DF	backclr;
		float		xform[16];
		float		invxform[16];
		float		invxrot[16];
		float		bias;		
		char		shading;
		char		filtering;		
		int			frame;
		int			samples;				
		Vector3DF	extinct;
		Vector3DF	steps;
		Vector3DF	cutoff;
		Vector3DF	thresh;
		CUdeviceptr	transfer;
		CUdeviceptr outbuf;
		CUdeviceptr dbuf;		
	};

	// CUDA modules
	#define MODL_PRIMARY			0

	// Raytracing
	#define FUNC_RAYDEEP			0
	#define FUNC_RAYVOXEL			1
	#define FUNC_RAYTRILINEAR		2
	#define FUNC_RAYTRICUBIC		3
	#define FUNC_RAYLEVELSET		4
	#define FUNC_EMPTYSKIP			5
	#define FUNC_SECTION2D			6
	#define FUNC_SECTION3D			7
	#define FUNC_RAYTRACE			8	
	#define FUNC_RAYSURFACE_DEPTH	9
	
	// Sorting & Sampling
	#define FUNC_PREFIXSUM			20		// sorting
	#define FUNC_PREFIXFIXUP		21			
	#define FUNC_RESAMPLE			22
	#define FUNC_REDUCTION			23
	#define FUNC_DOWNSAMPLE			24
	
	// Points & Triangles
	#define FUNC_INSERT_POINTS		30
	#define FUNC_SORT_POINTS		31	
	#define FUNC_INSERT_TRIS		32
	#define FUNC_SORT_TRIS			33
	#define FUNC_VOXELIZE			34	
	#define	FUNC_SCALE_PNT_POS		35
	#define	FUNC_CONV_AND_XFORM		36
	
	// Gather & Scatter
	#define FUNC_COUNT_SUBCELL		40
	#define FUNC_INSERT_SUBCELL		41		
	#define FUNC_INSERT_SUBCELL_FP16	42
	#define FUNC_CALC_SUBCELL_POS	43
	#define FUNC_SET_FLAG_SUBCELL	44
	#define FUNC_SPLIT_POS			45
	#define	FUNC_MAP_EXTRA_GVDB		46
	#define FUNC_ADD_SUPPORT_VOXEL	47
	#define FUNC_INSERT_SUPPORT_POINTS	48
	#define FUNC_SCATTER_DENSITY	49
	#define FUNC_SCATTER_AVG_COL	50
	#define FUNC_GATHER_DENSITY		51	
	#define FUNC_GATHER_LEVELSET	52
	#define FUNC_GATHER_LEVELSET_FP16		53
	
	// Topology
	#define FUNC_FIND_ACTIV_BRICKS	60
	#define FUNC_BITONIC_SORT		61
	#define FUNC_CALC_BRICK_ID		62
	#define FUNC_FIND_UNIQUE		66
	#define FUNC_COMPACT_UNIQUE		67
	#define FUNC_LINK_BRICKS		68
	
	// Incremental Topology
	#define FUNC_CALC_EXTRA_BRICK_ID	70
	#define FUNC_CALC_INCRE_BRICK_ID	71
	#define FUNC_CALC_INCRE_EXTRA_BRICK_ID	72
	#define FUNC_DELINK_LEAF_BRICKS		73
	#define FUNC_DELINK_BRICKS			74
	#define FUNC_MARK_LEAF_NODE			75

	#define FUNC_READ_GRID_VEL		97
	#define FUNC_CHECK_VAL			98

	// Apron Updates
	#define FUNC_UPDATEAPRON_F		100
	#define FUNC_UPDATEAPRON_F3		101
	#define FUNC_UPDATEAPRON_F4		102
	#define FUNC_UPDATEAPRON_C		103
	#define FUNC_UPDATEAPRON_C3		104
	#define FUNC_UPDATEAPRON_C4		105
	#define FUNC_UPDATEAPRONFACES_F		106
	#define FUNC_UPDATEAPRONFACES_F3	107
	#define FUNC_UPDATEAPRONFACES_F4	108

	// Operators
	#define FUNC_FILL_F				110		
	#define FUNC_FILL_C				111	
	#define FUNC_FILL_C4			112
	#define FUNC_SMOOTH				113		
	#define FUNC_NOISE				114
	#define FUNC_GROW				115
	#define FUNC_CLR_EXPAND			116
	#define FUNC_EXPANDC			117

	#define MAX_FUNC				255

	#define SR_HIT_OFFSET			0
	#define SR_ORIG_OFFSET			24
	#define SR_CLR_OFFSET			48
	#define SR_PNODE_OFFSET			52
	#define SR_PNDX_OFFSET			56

	// Auxiliary buffers
	#define AUX_PNTPOS				0
	#define AUX_PNTCLR				1
	#define AUX_PNODE				2
	#define AUX_PNDX				3
	#define AUX_GRIDCNT				4
	#define AUX_GRIDOFF				5
	#define AUX_ARRAY1				6
	#define AUX_SCAN1				7
	#define AUX_ARRAY2				8
	#define AUX_SCAN2				9
	#define AUX_COLAVG				10
	#define AUX_VOXELIZE			11
	#define AUX_VERTEX_BUF			12
	#define AUX_ELEM_BUF			13
	#define AUX_TRI_BUF				14
	#define AUX_PNTSORT				15
	#define AUX_PNTDIR				16
	#define AUX_DATA2D				18
	#define AUX_DATA3D				19
	#define AUX_MATRIX4F			20
	#define AUX_DOWNSAMPLED			21

	// Topology
	#define AUX_PBRICKDX			30
	#define AUX_ACTIVBRICKCNT		31
	#define AUX_BRICK_LEVXYZ		32
	#define AUX_RANGE_RES			33
	#define AUX_MARKER				34
	#define AUX_RADIX_PRESCAN		35
	#define AUX_SORTED_LEVXYZ		36
	#define AUX_TMP					37
	#define AUX_UNIQUE_CNT			38
	#define AUX_MARKER_PRESUM		39
	#define AUX_UNIQUE_LEVXYZ		40
	#define AUX_LEVEL_CNT			41

	// Incremental Topology
	#define AUX_EXTRA_BRICK_CNT		42
	#define	AUX_NODE_MARKER			43
	#define AUX_PNTVEL				44
	#define	AUX_DIV					45

	// Gathering
	#define AUX_SUBCELL_CNT			49
	#define AUX_SUBCELL_PREFIXSUM	50
	#define AUX_SUBCELL_PNTS		51
	#define AUX_SUBCELL_POS			52
	#define AUX_SUBCELL_PNT_POS		53
	#define AUX_SUBCELL_PNT_VEL		54
	#define AUX_SUBCELL_PNT_CLR		55
	#define AUX_SUBCELL_MAPPING		56	
	#define AUX_SUBCELL_FLAG		57
	#define AUX_SUBCELL_NID			58
	#define AUX_SUBCELL_OBS_NID		59	
	
	// CG
	#define	AUX_VOLUME				60
	#define	AUX_CG					61
	#define	AUX_INNER_PRODUCT		62
	#define	AUX_TEXTURE_MAX			63
	#define	AUX_TEXTURE_MAX_TMP		64

	#define AUX_BOUNDING_BOX		65
	#define AUX_WORLD_POS_X			66
	#define AUX_WORLD_POS_Y			67
	#define AUX_WORLD_POS_Z			68
	
	#define AUX_VEL_BOUNDING_BOX	69
	#define AUX_WORLD_VEL_X			70
	#define AUX_WORLD_VEL_Y			71
	#define AUX_WORLD_VEL_Z			72

	// Testing
	#define AUX_TEST				75
	#define AUX_TEST_1				76
	#define AUX_OUT1				77
	#define AUX_OUT2				78

	#define MAX_AUX					96
		
	// Ray object
	struct ALIGN(16) ScnRay {		//							offset	bytes
		Vector3DF	hit;			// hit point				0		12 bytes
		Vector3DF	normal;			// normal at hit point		12		12 bytes
		Vector3DF	orig;			// ray origin				24		12 bytes
		Vector3DF	dir;			// ray direction			36		12 bytes
		uint		clr;			// color					48		4 bytes
		uint		pnode;			// point sorting			52		4 bytes
		uint		pndx;			// point sorting			56		4 bytes
	};	
	struct ALIGN(16) Extents {
		int			lev;
		Vector3DF	vmin, vmax;	
		Vector3DI	imin, imax, ires;
		Vector3DF	iwmin, iwmax;
		Vector3DF	cover;
		int			icnt;
	};

	
	class GVDB_API VolumeGVDB : public VolumeBase {
	public:
			VolumeGVDB ();
			~VolumeGVDB ();
			
			// Setup
			void SetCudaDevice ( int devid, CUcontext ctx=NULL );
			void Initialize ();			
			void Clear ();	
			void SetProfile ( bool bCPU, bool bGPU ) ;
			void SetDebug(bool dbg);
			void LoadFunction ( int fid, std::string func, int mid, std::string ptx );
			void StartRasterGL ();
			void SetModule ();
			void SetModule ( CUmodule module );			
			void SetEpsilon(float eps, int maxiter) { mEpsilon = eps; mMaxIter = maxiter; mVDBInfo.update = true; }			
			Vector3DI getVersion();
						
			CUcontext getContext() { return mContext; }
			CUdevice getDevice() { return mDevice; }
			
			// Raytracing
			void Render ( char shade_mode = SHADE_TRILINEAR, uchar in_channel = 0, uchar outbuf = 0 );	
			void RenderKernel ( CUfunction user_kernel, uchar in_channel = 0, uchar outbuf = 0);			
    	    void Raytrace ( DataPtr rays, uchar chan, char shading, int frame, float bias );
			char* getDataPtr ( int i, DataPtr dat )		{ return (dat.cpu + (i*dat.stride)); }			
			
			// Compute
			// For numIterations iterations, runs the native compute kernel with index
			// `effect`, assuming it has the signature
			// `effect(VDBInfo* gvdb, int3 atlasRes, uchar channel, float p1, float p2, float p3)`,
			// then if `bUpdateApron` is true, updates the apron after each iteration
			// with boundary value `boundval`.
			// If `skipOverAprons` is true, it uses a grid of size `mPool->getAtlasResPacked(...)`;
			// otherwise, it uses a grid of size `mPool->getAtlasRes(...)`.
			// Passes the elements of `parameters` to the kernel as p1, p2, and p3.
			void Compute ( int effect, uchar channel, int num_iterations, Vector3DF parameters, bool bUpdateApron, bool skipOverAprons, float boundval = 0.0f );
			// Runs a custom user compute kernel, `user_kernel`, in the module `user_module`,
			// passing in argument `channel`. This must have the function signature
			// `func(VDBInfo* gvdb, int3 atlasRes, uchar channel)`
			// If `bUpdateApron` is true, it updates all apron channels
			// afterwards. If `skipOverAprons` is true, it uses a grid of size
			// `mPool->getAtlasResPacked(...)`; otherwise, it uses a grid of size `mPool->getAtlasRes(...)`.
			void ComputeKernel ( CUmodule user_module, CUfunction user_kernel, uchar channel, bool bUpdateApron, bool skipOverAprons);
			void Resample ( uchar chan, Matrix4F xform, Vector3DI in_res, char in_aux, Vector3DF inr, Vector3DF outr );			
			Vector3DF Reduction(uchar chan);
			void DownsampleCPU(Matrix4F xform, Vector3DI in_res, char in_aux, Vector3DI out_res, Vector3DF out_max, char out_aux, Vector3DF inr, Vector3DF outr);
			
			// File I/O
			bool LoadBRK ( std::string fname );
			// Configures and loads a GVDB volume from a VDB file.
			// Supports <5, 4, 3> and <3, 3, 3, 4> floating-point grids.
			// This can also read Vec3 grids, but vectors are converted
			// to their magnitudes and stored as floats.
			bool LoadVDB ( std::string fname );
			bool LoadVBX ( const std::string fname, int force_maj=0, int force_min=0 );
			void SaveVBX ( const std::string fname );
			// Saves channel 0 of the current volume as an OpenVDB file, which must have
			// the T_FLOAT format. Supports <5, 4, 3> and <3, 3, 3, 4> grids, which are
			// chosen automatically. 
			void SaveVDB ( std::string fname );
			bool ImportVTK ( std::string fname, std::string field, Vector3DI& res );
			void WriteObj ( char* fname );
			void AddPath ( const char* path );
			bool FindFile ( std::string fname, char* path );		
			void ConvertBitmaskToNonBitmask(int levs);

			// Topology Functions
			void Configure ( int r4, int r3, int r2, int r1, int r0 );		// Initialize VDB configuration
			void Configure ( int levs, int* r, int* ncnt, bool use_masks=false );
			void DestroyChannels ();
			void SetChannelDefault ( int cx, int cy, int cz )	{ mDefaultAxiscnt.Set(cx,cy,cz); }
			void SetApron ( int n )	 { mApron = n;}
			void AddChannel ( uchar chan, int dt, int apron, int filter=F_LINEAR, int border=F_CLAMP, Vector3DI axiscnt = Vector3DI(0,0,0) );
			void FillChannel ( uchar chan, Vector4DF val );
			void ClearAllChannels ();
			void ClearChannel(uchar chan);
			uchar GetChannelType(uchar channel); // Returns the type code (e.g. T_FLOAT) of the given channel.
			slong Reparent ( int lev, slong prevroot_id, Vector3DI pos, bool& bNew );		// Reparent tree with new root			
			slong ActivateSpace ( Vector3DF pos );
			slong ActivateSpace ( slong nodeid, Vector3DI pos, bool& bNew, slong stopnode = ID_UNDEFL, int stoplev = 0 );	// Active leaf at given location
			slong ActivateSpaceAtLevel ( int lev, Vector3DI pos );
			Vector3DI GetCoveringNode ( int lev, Vector3DI pos, Vector3DI& range );
			void ComputeBounds ();
			void ClearAtlasAccess ();
			void SetupAtlasAccess ();
			void FinishTopology (bool pCommitPool = true, bool pComputeBounds = true);						
			void UpdateAtlas ();
			void UpdateApron ();
			void UpdateApron ( uchar chan, float boundval = 0.0f, bool changeCtx = true );
			void UpdateApronFaces(uchar chan);
			void SetColorChannel ( uchar chan );
			// API-facing version of Allocator::AtlasRetrieveBrickXYZ.
			void AtlasRetrieveBrickXYZ(uchar channel, Vector3DI minimumCorner, DataPtr& dest);

			void SetBounds(Vector3DF pMin, Vector3DF pMax);

			// Topology
			void ClearPoolCPU();
			void FetchPoolCPU();
			int DetermineDepth(Vector3DI& pos);
			slong FindParent(int lev,  Vector3DI pos);
			void FindActivBricks(int lev, int rootlev, int num_pnts, Vector3DF orig, Vector3DI rootPos);
			void ActivateBricksGPU(int pNumPnts, float pRadius, Vector3DF pOrig, int pRootLev, Vector3DI pRootPos);
			void RadixSortByByte(int pNumPnts, int pLevDepth);
			void FindUniqueBrick(int pNumPnts, int pLevDepth, int& pNumUniqueBrick);

			void ActivateExtraBricksGPU(int pNumPnts, float pRadius, Vector3DF pOrig, int pRootLev, Vector3DI pRootPos);
			void ActivateIncreBricksGPU(int pNumPnts, float pRadius, Vector3DF pOrig, int pRootLev, Vector3DI pRootPos, bool bAccum=false);

			void RebuildTopology(int pNumPnts, float pRadius, Vector3DF pOrig);
			void RebuildTopologyCPU(int pNumPnts, Vector3DF pOrig, Vector3DF* pPos);
			void AccumulateTopology(int pNumPnts, float pRadius, Vector3DF pOrig, int iDepth=1 );
			void RequestFullRebuild(bool tf) { mRebuildTopo = tf;  }
			void SetDiv ( DataPtr div );

			int GetLevels() { return mPool->getNumLevels(); }

			void ReadGridVel( int N);
			void CheckVal ( float slice, int chanVx, int chanVy, int chanVz, int chanVxOld, int chanVyOld, int chanVzOld );

			// Nodes
			slong AllocateNode ( int lev );
			void  SetupNode ( slong nodeid, int lev, Vector3DF pos, bool marker = true );
			slong AddChildNode ( slong nodeid, Vector3DF ppos, int plev, uint32 i, Vector3DI pos );
			slong InsertChild ( slong nodeid, slong child, uint32 i );			
			void DebugNode ( slong nodeid );
			void ClearMapping ();
			void AssignMapping ( Vector3DI brickpos, Vector3DI pos, int leafid );
			void UpdateNeighbors();
			float getValue ( slong nodeid, Vector3DF pos, float* atlas );
			// Gets a Node struct from a group, level, and index.
			nvdb::Node* getNode ( int grp, int lev, slong ndx )		{ return (Node*) mPool->PoolData ( Elem(grp,lev,ndx) ); }			
			// Gets a Node struct from a pool reference.
			nvdb::Node* getNode ( slong nodeid )		{ return (Node*) mPool->PoolData ( nodeid ); }			
			// Get the `ndx`th child of the node `curr`.
			nvdb::Node* getChild (Node* curr, uint ndx);
			// Get the child that corresponds to bit `b`. If there is no child at that bit, returns
			// nullptr (0x0).
			nvdb::Node* getChildAtBit ( Node* curr, uint b);
			// Get the pool reference to the child that corresponds to bit `b`. If there is no child
			// at that bit, or if curr's childList was undefined, returns ID_UNDEF64.
			uint64 getChildRefAtBit(Node* curr, uint b);
			bool  isActive(Vector3DI wpos);
			bool  isActive(Vector3DI wpos, slong nodeid);	// recursive
			// Returns true if the given pool reference is at level = 0.
			bool  isLeaf ( slong nodeid )		{ return ElemLev ( nodeid )==0; }
			// Gets the child node reference at the specific bit position of the given node reference.
			uint64 getChildNode ( slong nodeid, uint b );
			// Gets the bit position and local 3D position of the given child node reference within
			// the given parent node reference.
			uint32 getChildOffset ( slong  nodeid, slong childid, Vector3DI& pos );
			// Returns true and sets `bit` to the corresponding bit index if the index-space position `pos` is inside
			// the given node, for false otherwise
			bool getPosInNode ( slong curr_id, Vector3DI pos, uint32& bit );

			// Render Buffers
			void AddRenderBuf ( int chan, int width, int height, int byteperpix );
			void ResizeRenderBuf ( int chan, int width, int height, int byteperpix );
			void AddDepthBuf ( int chan, int width, int height );
			void ResizeDepthBuf ( int chan, int width, int height );
			void ReadRenderBuf ( int chan, unsigned char* outptr );		

			// Prepare
			void PrepareVDB ();
			void PrepareVDBPartially ();	// for activate brick use only, no atlas
			void RetrieveVDB();
			void PrepareRender ( int w, int h, char shading );		
			
			void getUsage(Vector3DF& ext, Vector3DF& vox, Vector3DF& used, Vector3DF& free);	// Quick memory info
			Vector3DF MemoryUsage(std::string name, std::vector<std::string>& outlist);			// Detailed info
			void Measure ( bool bPrint );			
			float MeasurePools ();

			// Voxelization
			Extents ComputeExtents ( Node* node );
			Extents ComputeExtents ( int lev, Vector3DF obj_min, Vector3DF obj_max );			
			void SolidVoxelize ( uchar chan, Model* model, Matrix4F* xform, float val_surf, float val_inside, float vthresh=0.0 );
			int VoxelizeNode ( Node* node, uchar chan, Matrix4F* xform, float bdiv, float val_surf, float val_inside, float vthresh = 0.0);
			int ActivateRegion ( int lev, Extents& e );
			int ActivateRegionFromAux(Extents& e, int auxid, uchar dt, float vthresh);
			int ActivateHalo(Extents& e);
			void SurfaceVoxelizeGL ( uchar chan, Model* model, Matrix4F* xform );   // OpenGL voxelize
			void AuxGeometryMap ( Model* model, int vertaux, int elemaux );
			void AuxGeometryUnmap ( Model* model, int vertaux, int elemaux );

			void Synchronize();
			
			// OpenGL Functions							
			void WriteDepthTexGL ( int chan, int glid );
			void WriteRenderTexGL ( int chan, int glid );
			void ReadRenderTexGL ( int chan, int glid );			
			void ValidateOpenGL ();			
			void UseOpenGLAtlas ( bool tf );
			int  getAtlasGLID ( int chan )	{ return mPool->getAtlasGLID ( chan ); }

			// Internal Data Accessors
			// Gets the size of a VDBInfo struct
			int  getVDBSize ()				{ return sizeof(mVDBInfo); }
			// Gets a pointer to the volume's VDBInfo struct
			char* getVDBInfo ()				{ return (char*) &mVDBInfo; }
			// Gets a pointer to the volume's GPU copy of the VDBInfo struct
			CUdeviceptr getCUVDBInfo()      { return cuVDBInfo; }
			// Gets the size of a ScnInfo struct
			int  getScnSize()				{ return sizeof(mScnInfo); }
			// Gets a pointer to the volume's ScnInfo struct
			char* getScnInfo()				{ return (char*) &mScnInfo; }			

			// Data Operations
			void CleanAux(int id);
			void CleanAux();
			void PrepareAux ( int id, int cnt, int stride, bool bZero, bool bCPU=false );
			void PrepareV3D ( Vector3DI ires, uchar dtype );
			void AllocData ( DataPtr& ptr, int cnt, int stride, bool bCPU=true );
			void FreeData ( DataPtr& ptr );
			void RetrieveData ( DataPtr ptr );			
			void CommitData ( DataPtr ptr );			
			void CommitData ( DataPtr& ptr, int cnt, char* cptr, int offs, int stride );
			void SetDataCPU ( DataPtr& dat, int cnt, char* cptr, int offs, int stride );	
			void SetDataGPU ( DataPtr& dat, int cnt, CUdeviceptr gptr, int offs, int stride );
			char* getDataCPU ( DataPtr ptr, int n, int stride );
			void PrefixSum ( CUdeviceptr outArray, CUdeviceptr inArray, int numElements );
			void SetPoints ( DataPtr& pntpos, DataPtr& pntvel, DataPtr& clrpos );
			void InsertPoints ( int num_pnts, Vector3DF trans, bool bPrefix=false );					
			Vector3DI InsertTriangles ( Model* model, Matrix4F* xform, float& ydiv );

			void SetSupportPoints ( DataPtr& pntpos, DataPtr& dirpos );
			void InsertSupportPoints ( int num_pnts, float offset, Vector3DF trans, bool bPrefix=false );
			void AddSupportVoxel ( int num_pnts, float radius, float offset, float amp, Vector3DF trans, bool expand = true, bool avgColor = false );
			void PrintPool(uchar grp, uchar lev);

			void MapExtraGVDB (int subcell_size);
			void InsertPointsSubcell( int subcell_size, int num_pnts, float pRadius, Vector3DF trans, int& pSCPntsLength );
			void InsertPointsSubcell_FP16(int subcell_size, int num_pnts, float radius, Vector3DF trans, int& pSCPntsLength);			

			void ScalePntPos(int num_pnts, float scale);
			void ScatterDensity			(int num_pnts, float radius, float amp, Vector3DF trans, bool expand = true, bool avgColor = false );			
			void GatherDensity			(int subcell_size, int num_pnts, float radius, Vector3DF trans, int& pSCPntsLength, int chanDensity,  int chanClr, bool bAccum=false);
			void GatherLevelSet			(int subcell_size, int num_pnts, float radius, Vector3DF trans, int& pSCPntsLength, int chanLevelset, int chanClr, bool bAccum=false);
			void GatherLevelSet_FP16	(int subcell_size, int num_pnts, float radius, Vector3DF trans, int& pSCPntsLength, int chanLevelSet, int chanClr);			
			void ConvertAndTransform(DataPtr& psrc, char psrcbits, DataPtr& pdest, char pdestbits, int num_pnts, Vector3DF wMin, Vector3DF wMax, Vector3DF trans, Vector3DF scal);
			
			// Misc info
			void GetBoundingBox( int num_pnts, Vector3DF pTrans );
			Vector3DF getBoundMin() { return mPosMin; }
			Vector3DF getBoundMax() { return mPosMax; }			
			void GetMinMaxVel(int num_pnts);
			void CopyChannel(int chanDst, int chanSrc);
			char* GetTestPtr();

			// Write .OBJ 
			void setupVerts ( int gv[], Vector3DI g, int r1, int r2 );
			void enableVerts ( int*& vgToVert, std::vector<Vector3DF>& verts, Vector3DF vm, int gv[] );
			void writeCube ( FILE* fp, unsigned char vpix[], slong& numfaces, int gv[], int*& vgToVert, slong vbase );

			// Helpers
			void CommitTransferFunc ();
			void TimerStart ();
			float TimerStop ();
			void CheckData ( std::string msg, CUdeviceptr ptr, int dt, int stride, int cnt );

			// VDB Access
			bool isOn (slong nodeid, uint32 n );
			uint64 getMaskSize(int lev)	{ uint64 sz = getVoxCnt(lev) / 8; return (lev==0) ? 0 : ((sz < 8 ) ? 8 : sz); }		// Mask Size of level						

			// VDB Configuration			
			void SetVDBConfig ( int lev, int i )		{ mVCFG[lev] = i; }
			// Gets the nearest node in index-space at the given level.
			Vector3DI getNearestAbsVox ( int lev, Vector3DF pnt );
			// Gets the log base 2 branching factor per side of a node at the given level.
			int getLD(int lev)			{ return mLogDim[lev]; }
			// Gets the branching factor per side (number of child nodes or child voxels per side) of a node
			// at the given level.
			int getRes(int lev)			{ return (1 << mLogDim[lev]); }
			// Gets the number of child nodes or child voxels of a node at the given level.
			uint64 getVoxCnt(int lev)	{ uint64 r = uint64(1) << mLogDim[lev]; return r*r*r; }
			// Returns the index of a position within a node at the given level.
			// Suppose a node has resolution r, and its children are numbered from 0 to r^3-1.
			// This returns the number of the child with local coordinate `pos` (in [0, r-1]^3).
			int getBitPos ( int lv, Vector3DI pos )			{ int res=getRes(lv);	return (pos.z*res + pos.y)*res+ pos.x; }
			// Inverts `getBitPos`.
			// Suppose a node has resolution r, and its children are numbered from 0 to r^3-1.
			// Given that number, this returns the child's local position (in [0, s-1]^3).
			Vector3DI getPosFromBit ( int lv, uint32 b )	{ 
					int logr = mLogDim[lv]; 					
					uint32 mask = (uint32(1) << logr) - 1;		
					int z = (b & (mask << 2*logr) ) >> (2*logr);
					int y = (b & (mask << logr) ) >> logr;
					int x = (b & mask);
					return Vector3DI(x,y,z);
			}
			// Gets the size a node at the given level covers in voxels.
			// Same except for return type as `getRange` since 1.1.1.
			Vector3DF getCover(int lv)	{ return Vector3DF(getRange(lv)); }
			// Gets the lateral size a node at the given level covers in voxels.
			Vector3DI getRange(int lv)	{ 
					if ( lv==-1 ) return Vector3DI(1,1,1);
					Vector3DI r = getRes3DI(0);		// brick res
					for ( int l=1; l<=lv; l++) r *= getRes3DI(l);
					return r;
			}
			// Gets the resolution (branching factor per side) of a node as a `Vector3DI`.
			Vector3DI getRes3DI(int lv)	{ int r = (1 << mLogDim[lv]); return Vector3DI(r,r,r); }
			// Gets a color for the given level.
			Vector3DF getClrDim(int lv) { return mClrDim[lv]; }
			// Gets the `id`th auxiliary buffer.
			DataPtr& getAux(int id) { return mAux[id]; }

			void	SetSimBounds(Vector3DI b) {mSimBounds = b;}
			void	SetCollisionObj(VolumeGVDB* obs) {cuOBSVDBInfo = obs->cuVDBInfo; mHasObstacle = true;}

			void verbosef(const char * fmt, ...) {
				if (!mbVerbose) return;			// check if verbose
				va_list  vlist;
				va_start(vlist, fmt);
				gprintf2(vlist, fmt, 0);
			}
			// Gets the total number of nodes at a given level.
			uint64	getNumNodes(int lev) { return getNumTotalNodes(lev); }
			// Gets the number of used nodes at a given level.
			uint64	getNumUsedNodes ( int lev );
			// Gets the total number of nodes at a given level.
			uint64	getNumTotalNodes ( int lev );
			// Gets node `n` at level `lev`.
			Node*	getNodeAtLevel ( int n, int lev );
			// Gets the pool reference of the leaf (level 0 node) that is a child of the pool reference `nodeid` and
			// contains the point `pos` (in voxel coordinates). Returns ID_UNDEFL if no such leaf exists.
			uint64	getNodeAtPoint ( uint64 nodeid, Vector3DF pos);
			
			// Gets the inclusive lower bound of the node's bounding box in voxel coordinates (i.e. `node->mPos`).
			// This function's return type may be changed to `Vector3DI` in the future.
			Vector3DF getWorldMin ( Node* node );
			// Gets the exclusive upper bound of the node's bounding box in voxel coordinates
			// (i.e. `getWorldMin` plus the size of this node).
			// This function's return type may be changed to `Vector3DI` in the future.
			Vector3DF getWorldMax ( Node* node );
			// Gets the inclusive lower bound of the `VolumeGVDB`'s bounding box in voxel coordinates.
			// Same as getVoxMin since 2020-07-07.
			// This function's return type may be changed to `Vector3DI` in the future.
			Vector3DF getWorldMin();
			// Gets the inclusive upper bound of the `VolumeGVDB`'s bounding box in voxel coordinates.
			// Same as getVoxMax since 2020-07-07.
			// This function's return type may be changed to `Vector3DI` in the future.
			Vector3DF getWorldMax();
			
			//-- Grid Transform - arbitrary transforms on volume (replaces Voxsize)
			// Sets the transform from points in GVDB's coordinate system to the application's coordinate system.
			// This transform consists of the following operations in order:
			// - Add `pretrans` to the point
			// - Multiply the result's coordinates by `scal`
			// - Rotate the result around the X axis by angs.x, around the Y axis by angs.x, and then around the Z axis
			// by angs.z (as in `Matrix4F::RotateTZYXS`)
			// - Add `trans` to the result
			void SetTransform(Vector3DF pretrans, Vector3DF scal, Vector3DF angs, Vector3DF trans);
			// Returns the transform as a 4x4 matrix. If `v` is a vector, then computing `v`*`mXform` transforms the
			// point `v` from GVDB's coordinate system to the application's coordinate system.
			Matrix4F& getTransform() { return mXform; }	

			int		getMaskBytes(Node* node) { int r = getRes(node->mLev); return imax(((uint64)r*r*r) >> 3, 1); }		// divide by bits per byte (2^3=8)
			uint64	getMaskWords(Node* node) { int r = getRes(node->mLev); return imax(((uint64)r*r*r) >> 6, 1); }		// divide by bits per 64-bit word (2^6=64)			
			uint64* getMask(Node* node)		{ return (uint64*) &node->mMask; }
			int		getNumChild(Node* node) { return (node->mLev == 0) ? 0 : countOn( node ); }
			
			// Bit operations:
			// Based on Bithacks (Sean Eron Anderson, http://graphics.stanford.edu/~seander/bithacks.html)
			// 
			inline uint64 numBitsOn(byte v)
			{
				static const byte numBits[256] = {
					#define B2(n)  n,     n+1,     n+1,     n+2
					#define B4(n)  B2(n), B2(n+1), B2(n+1), B2(n+2)
					#define B6(n)  B4(n), B4(n+1), B4(n+1), B4(n+2)
					B6(0), B6(1), B6(1),   B6(2)
				};
				return numBits[v];
			}
			inline uint64 numBitsOff(byte v) { return numBitsOn((byte)~v); }
			inline uint64 numBitsOn(uint32 v)
			{
				v = v - ((v >> 1) & 0x55555555U);
				v = (v & 0x33333333U) + ((v >> 2) & 0x33333333U);
				return ((v + (v >> 4) & 0xF0F0F0FU) * 0x1010101U) >> 24;
			}
			inline uint64 numBitsOff(uint32 v) { return numBitsOn(~v); }
			inline uint64 numBitsOn(uint64 v)
			{
				v = v - ((v >> 1) & UINT64_C(0x5555555555555555));
				v = (v & UINT64_C(0x3333333333333333)) + ((v >> 2) & UINT64_C(0x3333333333333333));
				return ((v + (v >> 4) & UINT64_C(0xF0F0F0F0F0F0F0F)) * UINT64_C(0x101010101010101)) >> 56;
			}
			inline uint64 numBitsOff(uint64 v) { return numBitsOn((uint64)~v); }
			inline uint64 firstBitOn(byte v)
			{
				assert(v);		// make sure not 0
				static const byte DeBruijn[8] = { 0, 1, 6, 2, 7, 5, 4, 3 };
				return DeBruijn[byte((v & -v) * 0x1DU) >> 5];
			}
			inline uint64 firstBitOn(uint32 v)
			{
				assert(v);
				static const byte DeBruijnBitPos[32] = {
					0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
					31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
				};
				return DeBruijnBitPos[uint32((int(v) & -int(v)) * 0x077CB531U) >> 27];
			}
			inline uint64 firstBitOn(uint64 v)
			{
				assert(v);
				static const byte DeBruijn[64] = {
					0,   1,  2, 53,  3,  7, 54, 27, 4,  38, 41,  8, 34, 55, 48, 28,
					62,  5, 39, 46, 44, 42, 22,  9, 24, 35, 59, 56, 49, 18, 29, 11,
					63, 52,  6, 26, 37, 40, 33, 47, 61, 45, 43, 21, 23, 58, 17, 10,
					51, 25, 36, 32, 60, 20, 57, 16, 50, 31, 19, 15, 30, 14, 13, 12,
				};
				return DeBruijn[uint64((sint64(v) & -sint64(v)) * UINT64_C(0x022FDD63CC95386D)) >> 58];
			}
			inline uint64 lastBitOn(uint32 v)
			{
				static const byte DeBruijn[32] = {
					0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30,
					8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31
				};
				v |= v >> 1; // first round down to one less than a power of 2
				v |= v >> 2;
				v |= v >> 4;
				v |= v >> 8;
				v |= v >> 16;
				return DeBruijn[uint32(v * 0x07C4ACDDU) >> 27];
			}
			void clearMask( Node* node )
			{
				node->mMask = 0;
				memset(&node->mMask, 0, getMaskBytes(node) );
			}
			// set operator
			void set ( Node* node, Node* op2) {
				uint64* w1 = (uint64*)&node->mMask;
				uint64* we = (uint64*)&node->mMask + getMaskWords(node);
				uint64* w2 = (uint64*)&op2->mMask;
				for (; w1 != we; )
					*w1++ = *w2++;
			}			
			bool isEqual ( Node* node, Node* op2)
			{
				int sz = (int) getMaskWords(node);
				uint64* w1 = (uint64*)&node->mMask;
				uint64* we = (uint64*)&node->mMask + sz;
				uint64* w2 = (uint64*)&op2->mMask;
				for (; w1 != we && (*w1++ == *w2++); );
				return w1 == we;
			}			
			uint32 countOn( Node* node )
			{
				uint32 sum = 0;
				uint64* w1 = (uint64*)&node->mMask;
				uint64* we = (uint64*)&node->mMask + getMaskWords(node);
				for (; w1 != we; ) sum += (int)numBitsOn(*w1++);
				return sum;
			}
			uint64 countOn( Node* node, uint32 b)
			{
				uint64 sum = 0;
				uint64* w1 = (uint64*)&node->mMask;
				uint64* we = (uint64*)&node->mMask + (b >> 6);
				for (; w1 != we; ) sum += (int)numBitsOn(*w1++);
				uint64 w2 = *w1;
				w2 = w2 & ((uint64(1) << (b & 63)) - 1);
				sum += numBitsOn(w2);
				return sum;
			}
			uint64 countOff( Node* node ) { return getMaskBytes(node)*8 - countOn(node); }
			uint32 countToIndex(Node* node, uint64 count)
			{
				uint64 sum = 0;
				uint64* w1 = (uint64*)&node->mMask;
				uint64* we = (uint64*)&node->mMask + getMaskWords(node);
				uint32 bits = 0;
				count++;				
				for (; sum < count; bits++) {
					if (isOn(node, bits)) sum++;
				}
				return bits-1;
			}
			void setAll(Node* node, bool on)
			{
				const uint64 val = on ? ~uint64(0) : uint64(0);
				uint64* w1 = (uint64*)&node->mMask;
				uint64* we = (uint64*)&node->mMask + getMaskWords(node);
				for (; w1 != we; ) *w1++ = val;
			}
			void setOn (Node* node, uint32 n)  { (&node->mMask)[n >> 6] |= uint64(1) << (n & 63); }
			void setOff(Node* node, uint32 n)  { (&node->mMask)[n >> 6] &= ~(uint64(1) << (n & 63)); }			
			bool isOn ( Node* node, uint64 n)  { return (((uint64*)&node->mMask)[n >> 6] & (uint64(1) << (n & 63))) != 0; }
			bool isOff( Node* node, uint64 n)  { return (((uint64*)&node->mMask)[n >> 6] & (uint64(1) << (n & 63))) == 0; }
			
			void SetBias(float b) { m_bias = b; }

	protected:
			// VDB Settings
			int				mLogDim[MAXLEV];	// internal res config
			Vector3DF		mClrDim[MAXLEV];
			int				mVCFG[MAXLEV];		// user selected vdb config
			int				mApron;
			Matrix4F		mXForm;
			bool			mbGlew;
			bool			mbUseGLAtlas;
			bool			mbDebug;
			Vector3DI		mAtlasResize;
			Vector3DI		mDefaultAxiscnt;
						
			// Root node
			uint64			mRoot;
			Vector3DI		mPnt;

			// Scene 
			ScnInfo			mScnInfo;
			CUdeviceptr		cuScnInfo;

			// VDB Data Structure
			VDBInfo			mVDBInfo;			
			CUdeviceptr		cuVDBInfo;

			bool			mHasObstacle;
			CUdeviceptr		cuOBSVDBInfo; // Non-owning pointer to VDBInfo used for collision

			// CUDA kernels
			CUmodule		cuModule[5];
			CUfunction		cuFunc[ MAX_FUNC ];

			// CUDA pointers		
			CUdeviceptr		cuXform;
			CUdeviceptr		cuDebug;

			std::vector< Vector3DF >	leaf_pos;
			std::vector< uint64 >		leaf_ptr;

			// Auxiliary buffers
			DataPtr			mAux[MAX_AUX];		// Auxiliary
			std::string		mAuxName[MAX_AUX];

			Volume3D*		mV3D;			// Volume 3D

			// Dummy frame buffer
			int mDummyFrameBuffer;

			// CUDA Device & Context
			int				mDevSelect;
			CUcontext		mContext;
			CUdevice		mDevice;
			CUstream		mStream;

			bool			mRebuildTopo;
			int				mCurrDepth;
			Vector3DF		mPosMin, mPosMax, mPosRange;
			Vector3DF		mVelMin, mVelMax, mVelRange;

			Vector3DI		mSimBounds;
			float			mEpsilon;
			int				mMaxIter;

			// Grid Transform
			Vector3DF		mPretrans, mAngs, mTrans, mScale;
			Matrix4F		mXform, mInvXform, mInvXrot;

			const char*		mRendName[SHADE_MAX];

			float			m_bias;

#ifdef BUILD_OPENVDB
			// Internal function for loading bricks from an OpenVDB grid into
			// a GVDB grid. Reads OpenVDB <5, 4, 3> and <3, 3, 3, 4> grids,
			// and converts grids with float3 values into grids containing
			// their lengths. GridType is e.g. FloatGrid34.
			template<class GridType>
			bool LoadVDBInternal(openvdb::GridBase::Ptr& baseGrid);

			// Internal function for saving the GVDB grid as an OpenVDB file
			// with filename `fname`. TreeType is e.g. FloatTree34.
			template<class TreeType>
			void SaveVDBInternal(std::string& fname);
#endif // #ifdef BUILD_OPENVDB
		};

	}

#endif




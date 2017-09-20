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

#ifndef DEF_VOL_GVDB
	#define DEF_VOL_GVDB

	#include "gvdb_types.h"
	#include "gvdb_node.h"	
	#include "gvdb_volume_base.h"
	#include "gvdb_allocator.h"		
	using namespace nvdb;

	#ifdef BUILD_OPENVDB
		#include <openvdb/openvdb.h>
		using namespace openvdb::v3_0_0;

		// OpenVDB <3,3,3,4> support
		typedef openvdb::tree::Tree<openvdb::tree::RootNode<openvdb::tree::InternalNode<openvdb::tree::InternalNode<openvdb::tree::InternalNode<openvdb::tree::LeafNode<float,4>,3>,3>,3>>> FloatTree34; 
		typedef openvdb::tree::Tree<openvdb::tree::RootNode<openvdb::tree::InternalNode<openvdb::tree::InternalNode<openvdb::tree::InternalNode<openvdb::tree::LeafNode<openvdb::Vec3f,4>,3>,3>,3>>> Vec3fTree34; 
		typedef openvdb::Grid<FloatTree34>		FloatGrid34; 
		typedef openvdb::Grid<Vec3fTree34>		Vec3fGrid34; 
		typedef FloatGrid34						GridType34;
		typedef FloatGrid34::TreeType			TreeType34F;
		typedef Vec3fGrid34::TreeType			TreeType34VF;
	
		// OpenVDB <5,4,3> support
		typedef openvdb::FloatGrid				FloatGrid543;
		typedef openvdb::Vec3fGrid				Vec3fGrid543;
		typedef FloatGrid543					GridType543;
		typedef FloatGrid543::TreeType			TreeType543F;
		typedef Vec3fGrid543::TreeType			TreeType543VF;

	#endif

	#define MAXLEV			10

	class OVDBGrid;
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

	struct ALIGN(16) VDBInfo {
		int			dim[MAXLEV];
		int			res[MAXLEV];
		Vector3DF	vdel[MAXLEV];
		Vector3DF	voxelsize;
		Vector3DI	noderange[MAXLEV];
		int			nodecnt[MAXLEV];
		int			nodewid[MAXLEV];
		int			childwid[MAXLEV];		
		CUdeviceptr	nodelist[MAXLEV];		
		CUdeviceptr	childlist[MAXLEV];	
		CUdeviceptr atlas_map;					
		Vector3DI	atlas_cnt;
		Vector3DI	atlas_res;
		int			atlas_apron;
		int			brick_res;		
		int			apron_table[8];
		int			top_lev;
		bool		update;
		uchar		clr_chan;		
		Vector3DF	bmin;
		Vector3DF	bmax;
		Vector3DF	thresh;
		CUdeviceptr transfer;		
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
		Vector4DF 	camivprow0;
		Vector4DF 	camivprow1;
		Vector4DF 	camivprow2;
		Vector4DF 	camivprow3;
		Vector3DF	light_pos;
		Vector3DF	slice_pnt;
		Vector3DF	slice_norm;
		float		bias;
		float		shadow_amt;
		char		shading;
		char		filtering;		
		int			frame;
		int			samples;				
		Vector3DF	extinct;
		Vector3DF	steps;
		Vector3DF	cutoff;
		CUdeviceptr outbuf;
		CUdeviceptr dbuf;
		Vector4DF	backclr;
	};

	// CUDA modules
	#define MODL_PRIMARY			0
	#define MODL_TRANSFERS			1

	// CUDA kernels
	#define FUNC_RAYDEEP			0		// raytracing
	#define FUNC_RAYVOXEL			1
	#define FUNC_RAYTRILINEAR		2
	#define FUNC_RAYTRICUBIC		3
	#define FUNC_RAYLEVELSET		4
	#define FUNC_EMPTYSKIP			5
	#define FUNC_SECTION2D			6
	#define FUNC_SECTION3D			7
	#define FUNC_RAYTRACE			8	
	#define FUNC_RAYSURFACE_DEPTH	9
	
	#define FUNC_PREFIXSUM			50		// sorting
	#define FUNC_PREFIXFIXUP		51			
	#define FUNC_INSERT_POINTS		52		// points
	#define FUNC_SORT_POINTS		53	
	#define FUNC_SCATTER_DENSITY	54
	#define FUNC_SCATTER_AVG_COL	55
	#define FUNC_INSERT_TRIS		56		// triangles
	#define FUNC_SORT_TRIS			57
	#define FUNC_VOXELIZE			58	
	#define FUNC_GATHER_DENSITY		59
	#define FUNC_GATHER_VELOCITY	60
	#define FUNC_RESAMPLE			61
	#define FUNC_ADD_SUPPORT_VOXEL	62
	#define FUNC_INSERT_SUPPORT_POINTS  63
	#define FUNC_DOWNSAMPLE			64

	#define FUNC_UPDATEAPRON_F		100		// apron updates
	#define FUNC_UPDATEAPRON_F3		101
	#define FUNC_UPDATEAPRON_F4		102
	#define FUNC_UPDATEAPRON_C		103
	#define FUNC_UPDATEAPRON_C3		104
	#define FUNC_UPDATEAPRON_C4		105
	
	#define FUNC_FILL_F				150		// operators
	#define FUNC_FILL_C				151	
	#define FUNC_FILL_C4			152
	#define FUNC_SMOOTH				153		
	#define FUNC_NOISE				154
	#define FUNC_GROW				155
	#define FUNC_CLR_EXPAND			156
	#define FUNC_EXPANDC			157

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
	#define AUX_DATA3D				17
	#define AUX_MATRIX4F			18
	#define AUX_DOWNSAMPLED			19

	#define MAX_AUX					64
		
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
			
			// Setup
			void SetCudaDevice ( int devid );
			void Initialize ();			
			void Clear ();	
			void SetVoxelSize ( float vx, float vy, float vz );
			void SetProfile ( bool pf ) ;
			void LoadFunction ( int fid, std::string func, int mid, std::string ptx );
			void StartRasterGL ();
			void SetModule ();
			void SetModule ( CUmodule module );			
			void getMemory ( float& voxels, float& overhead, float& effective );
			
			// Raytracing
			void Render ( uchar rbuf, char shading, char filtering, int frame, int sample, int max_samples, float samt, uchar dbuf = 255 );	
			void RenderKernel ( uchar rbuf, CUfunction user_kernel, char shading, char filtering, int frame, int sample, int max_samples, float samt );			
			void Raytrace ( DataPtr rays, char shading, int frame, float bias );
			char* getDataPtr ( int i, DataPtr dat )		{ return (dat.cpu + (i*dat.stride)); }
			
			// Compute
			void Compute ( int effect, uchar chan, int iter, Vector3DF parm, bool bUpdateApron );
			void ComputeKernel ( CUmodule user_module, CUfunction user_kernel, uchar chan, bool bUpdateApron );
			void Resample ( uchar chan, Matrix4F xform, Vector3DI in_res, char in_aux, Vector3DF inr, Vector3DF outr );			
			void DownsampleCPU(Matrix4F xform, Vector3DI in_res, char in_aux, Vector3DI out_res, Vector3DF out_max, char out_aux, Vector3DF inr, Vector3DF outr);
			
			// File I/O
			bool LoadBRK ( std::string fname );
			bool LoadVDB ( std::string fname );
			bool LoadVBX ( std::string fname );
			void SaveVBX ( std::string fname );
			void SaveVDB ( std::string fname );
			bool ImportVTK ( std::string fname, std::string field, Vector3DI& res );
			void WriteObj ( char* fname );
			void AddPath ( std::string path );
			bool FindFile ( std::string fname, char* path );		

			// Topology Functions
			void Configure ( int r4, int r3, int r2, int r1, int r0 );		// Initialize VDB configuration
			void Configure ( int levs, int* r, int* ncnt);		
			void DestroyChannels ();
			void SetChannelDefault ( int cx, int cy, int cz )	{ mDefaultAxiscnt.Set(cx,cy,cz); }
			void SetApron ( int n )	 { mApron = n;}
			void AddChannel ( uchar chan, int dt, int apron, Vector3DI axiscnt = Vector3DI(0,0,0) );
			void FillChannel ( uchar chan, Vector4DF val );
			slong Reparent ( int lev, slong prevroot_id, Vector3DI pos, bool& bNew );		// Reparent tree with new root			
			slong ActivateSpace ( Vector3DF pos );
			slong ActivateSpace ( slong nodeid, Vector3DI pos, bool& bNew, slong stopnode = ID_UNDEFL, int stoplev = 0 );	// Active leaf at given location
			slong ActivateSpaceAtLevel ( int lev, Vector3DF pos );
			Vector3DI GetCoveringNode ( int lev, Vector3DI pos, Vector3DI& range );
			void ComputeBounds ();
			void ClearAtlasAccess ();
			void SetupAtlasAccess ();
			void FinishTopology ();						
			void UpdateAtlas ();
			void ClearAtlas ();			
			void UpdateApron ();
			void UpdateApron ( uchar chan );
			void SetColorChannel ( uchar chan );

			// Nodes
			slong AllocateNode ( int lev );
			void  SetupNode ( slong nodeid, int lev, Vector3DF pos);
			slong AddChildNode ( slong nodeid, Vector3DF ppos, int plev, uint32 i, Vector3DI pos );
			slong InsertChild ( slong nodeid, slong child, uint32 i );			
			void DebugNode ( slong nodeid );
			void ClearMapping ();
			void AssignMapping ( Vector3DI brickpos, Vector3DI pos, int leafid );
			float getValue ( slong nodeid, Vector3DF pos, float* atlas );
			nvdb::Node* getNode ( int grp, int lev, slong ndx )		{ return (Node*) mPool->PoolData ( Elem(grp,lev,ndx) ); }
			nvdb::Node* getNode ( slong nodeid )		{ return (Node*) mPool->PoolData ( nodeid ); }			
			bool  isLeaf ( slong nodeid )		{ return ElemLev ( nodeid )==0; }
			slong getChildNode ( slong nodeid, uint b );
			slong getChildOffset ( slong  nodeid, slong childid, Vector3DI& pos );
			bool getPosInNode ( slong curr_id, Vector3DI pos, uint32& bit );

			// Render Buffers
			void AddRenderBuf ( int chan, int width, int height, int byteperpix );
			void ResizeRenderBuf ( int chan, int width, int height, int byteperpix );
			void AddDepthBuf ( int chan, int width, int height );
			void ResizeDepthBuf ( int chan, int width, int height );
			void ReadRenderBuf ( int chan, unsigned char* outptr );		

			// Prepare
			void PrepareVDB ();
			void PrepareRender ( int w, int h, char shading, char filtering, int frame, int samples, float samt, uchar dbuf = 255 );
			void SetVoxels ( VolumeGVDB* vdb, std::vector<Vector3DI> poslist, float val );			
			void Measure ( bool bPrint );
			void Measure ( statVec& stats, slong nodeid );			
			float MeasurePools ();

			// Voxelization
			Extents ComputeExtents ( Node* node );
			Extents ComputeExtents ( int lev, Vector3DF obj_min, Vector3DF obj_max );			
			void SolidVoxelize ( uchar chan, Model* model, Matrix4F* xform, uchar val_surf, uchar val_inside );
			int VoxelizeNode ( Node* node, uchar chan, Matrix4F* xform, float bdiv, uchar val_surf, uchar val_inside );
			int ActivateRegion ( Extents& e );
			int ActivateRegionFromAux ( Extents& e, int auxid, uchar dt, float vthresh=0.0f );
			void ActiveRegionSparse(Extents& s, float* srcbuf);
			void SurfaceVoxelizeGL ( uchar chan, Model* model, Matrix4F* xform );   // OpenGL voxelize
			void AuxGeometryMap ( Model* model, int vertaux, int elemaux );
			void AuxGeometryUnmap ( Model* model, int vertaux, int elemaux );

			
			// OpenGL Functions					
			void PrepareScreenTexGL ();			
			void RenderGetResultGL ();			
			void WriteDepthTexGL ( int chan, int glid );
			void WriteRenderTexGL ( int chan, int glid );
			void ReadRenderTexGL ( int chan, int glid );			
			void ValidateOpenGL ();			
			void UseOpenGLAtlas ( bool tf );
			int  getAtlasGLID ( int chan )	{ return mPool->getAtlasGLID ( chan ); }
			int  getVDBSize ()				{ return sizeof(mVDBInfo); }
			char* getVDBInfo ()				{ return (char*) &mVDBInfo; }

			// Data Operations
			void PrepareAux ( int id, int cnt, int stride, bool bZero, bool bCPU=false );
			void PrepareV3D ( Vector3DI ires, uchar dtype );
			void AllocData ( DataPtr& ptr, int cnt, int stride, bool bCPU=true );
			void RetrieveData ( DataPtr ptr );			
			void CommitData ( DataPtr ptr );			
			void CommitData ( DataPtr& ptr, int cnt, char* cptr, int offs, int stride );
			void SetDataCPU ( DataPtr& dat, int cnt, char* cptr, int offs, int stride );	
			void SetDataGPU ( DataPtr& dat, int cnt, CUdeviceptr gptr, int offs, int stride );
			char* getDataCPU ( DataPtr ptr, int n, int stride );
			void PrefixSum ( CUdeviceptr outArray, CUdeviceptr inArray, int numElements );
			void SetPoints ( DataPtr pntpos, DataPtr clrpos );
			void InsertPoints ( int num_pnts, Vector3DF trans, bool bPrefix=false );		
			void ScatterPointDensity ( int num_pnts, float radius, float amp, Vector3DF trans, bool expand = true, bool avgColor = false );			
			void GatherPointDensity ( int num_pnts, float radius, int chan );

			Vector3DI InsertTriangles ( Model* model, Matrix4F* xform, float& ydiv );

			// Kui
			void SetSupportPoints ( DataPtr pntpos, DataPtr dirpos );
			void InsertSupportPoints ( int num_pnts, float offset, Vector3DF trans, bool bPrefix=false );
			void AddSupportVoxel ( int num_pnts, float radius, float offset, float amp, Vector3DF trans, bool expand = true, bool avgColor = false );

			// Write .OBJ 
			void setupVerts ( int gv[], Vector3DI g, int r1, int r2 );
			void enableVerts ( int*& vgToVert, std::vector<Vector3DF>& verts, Vector3DF vm, int gv[] );
			void writeCube ( FILE* fp, unsigned char vpix[], slong& numfaces, int gv[], int*& vgToVert, slong vbase );

			// Helpers
			void CommitTransferFunc ();
			void TimerStart ();
			float TimerStop ();
			void CheckData ( std::string msg, CUdeviceptr ptr, int dt, int stride, int cnt );

			// VDB Configuration			
			void SetVDBConfig ( int lev, int i )		{ mVCFG[lev] = i; }
			Vector3DI getNearestAbsVox ( int lev, Vector3DF pnt );
			int getLD(int lev)			{ return mLogDim[lev]; }							// Logres
			int getRes(int lev)			{ return (1 << mLogDim[lev]); }						// Resolution of level
			uint64 getVoxCnt(int lev)	{ uint64 r = uint64(1) << mLogDim[lev]; return r*r*r; }		// # of Voxels of level
			uint64 getMaskSize(int lev)	{ uint64 sz = getVoxCnt(lev) / 8; return (sz < 8 ) ? 8 : sz; }		// Mask Size of level						
			int getBitPos ( int lv, Vector3DI pos )			{ int res=getRes(lv);	return (pos.z*res + pos.y)*res+ pos.x; }
			Vector3DI getPosFromBit ( int lv, uint32 b )	{ 
					int logr = mLogDim[lv]; 					
					uint32 mask = (uint32(1) << logr) - 1;		
					int z = (b & (mask << 2*logr) ) >> (2*logr);
					int y = (b & (mask << logr) ) >> logr;
					int x = (b & mask);
					return Vector3DI(x,y,z);
			}
			Vector3DF getCover(int lv)	{ return mVoxsize * Vector3DF(getRange(lv)); }
			Vector3DI getRange(int lv)	{ 
					if ( lv==-1 ) return Vector3DI(1,1,1);
					Vector3DI r = getRes3DI(0);		// brick res
					for ( int l=1; l<=lv; l++) r *= getRes3DI(l);
					return r;
			}
			Vector3DI getRes3DI(int lv)	{ int r = (1 << mLogDim[lv]); return Vector3DI(r,r,r); }
			Vector3DF getClrDim(int lv) { return mClrDim[lv]; }
			DataPtr& getAux(int id) { return mAux[id];  }

			void verbosef(const char * fmt, ...) {
				if (!mbVerbose) return;			// check if verbose
				va_list  vlist;
				va_start(vlist, fmt);
				gprintf2(vlist, fmt, 0);
			}

			int		getNumNodes ( int lev );
			Node*	getNodeAtLevel ( int n, int lev );
			Vector3DF getWorldMin ( Node* node );
			Vector3DF getWorldMax ( Node* node );
			Vector3DF getVoxelSize() { return mVoxsize; }
			
	protected:
			// VDB Settings
			int				mLogDim[MAXLEV];	// internal res config
			Vector3DF		mClrDim[MAXLEV];
			int				mVCFG[MAXLEV];		// user selected vdb config
			int				mApron;
			Matrix4F		mXForm;
			bool			mbGlew;
			bool			mbUseGLAtlas;
			Vector3DI		mAtlasResize;
			Vector3DI		mDefaultAxiscnt;
						
			// Root node
			uint64			mRoot;
			Vector3DI		mPnt;

			VDBInfo			mVDBInfo;
			ScnInfo			mScnInfo;

			// CUDA kernels
			CUmodule		cuModule[5];
			CUfunction		cuFunc[ MAX_FUNC ];

			// CUDA pointers
			CUtexref		cuSurfReadTransfer;
			CUtexref		cuSurfRead;
			CUsurfref		cuSurfWrite;
			CUdeviceptr		cuOutBuf;
			CUdeviceptr		cuVDBInfo;
			CUdeviceptr		cuScnInfo;
			CUdeviceptr		cuXform;
			CUdeviceptr		cuDebug;

			CUtexObject		mTexIn[10];
			CUsurfObject	mTexOut[10];
			CUdeviceptr		cuTexIn;
			CUdeviceptr		cuTexOut;

			float			mTreeMem;

			int				mTime;

			std::vector< Vector3DF >	leaf_pos;
			std::vector< uint64 >		leaf_ptr;

			// Auxiliary buffers
			DataPtr			mAux[MAX_AUX];		// Auxiliary
			
			OVDBGrid*		mOVDB;			// OpenVDB grid	
			Volume3D*		mV3D;			// Volume 3D

			// Dummy frame buffer
			int mDummyFrameBuffer;

	};
	extern VolumeGVDB*	gVDB;

	}

#endif




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

#include "gvdb_allocator.h"
#include "gvdb_volume_3D.h"
#include "gvdb_volume_gvdb.h"
#include "gvdb_render.h"
#include "gvdb_node.h"
#include "app_perf.h"
#include "string_helper.h"

#if !defined(_WIN32)
#	include <GL/glx.h>
#endif


using namespace nvdb;

#define MAJOR_VERSION		1
#define MINOR_VERSION		0

#ifdef BUILD_OPENVDB
	// Link GVDB to OpenVDB for loading .vdb files
	#pragma message ( "Building OpenVDB." )	
	#ifdef OPENVDB_USE_BLOSC
		#pragma message ( "  OPENVDB_USE_BLOSC = Yes" )	
	#else
		#pragma message ( "  OPENVDB_USE_BLOSC = NO" )	
	#endif
	#include <openvdb\openvdb.h>
	#include <openvdb\io\Stream.h>
	#include <openvdb\tree\LeafNode.h>		// access to leaf buffers
	#include <openvdb/tools/ValueTransformer.h>
	#include <fstream>
	using namespace openvdb::v3_0_0;
#endif

// #define GVDB_DEBUG			// brings back and prints debug buffer from kernels

VolumeGVDB*	nvdb::gVDB = 0x0;

#define	MRES	2048

#ifdef BUILD_OPENVDB
	// OpenVDB helper
	class OVDBGrid {
	public:

		FloatGrid543::Ptr			grid543F;			// grids
		Vec3fGrid543::Ptr			grid543VF;
		FloatGrid34::Ptr			grid34F;			
		Vec3fGrid34::Ptr			grid34VF;

		TreeType543F::LeafCIter		iter543F;			// iterators
		TreeType543VF::LeafCIter	iter543VF;
		TreeType34F::LeafCIter		iter34F;
		TreeType34VF::LeafCIter		iter34VF;
															// buffers
		openvdb::v3_0_0::tree::LeafNode<float, 3U>::Buffer buf3U;			// 2^3 leaf res
		openvdb::v3_0_0::tree::LeafNode<Vec3f, 3U>::Buffer buf3VU;
		openvdb::v3_0_0::tree::LeafNode<float, 4U>::Buffer buf4U;			// 2^4 leaf res
		openvdb::v3_0_0::tree::LeafNode<Vec3f, 4U>::Buffer buf4VU;
	};
#endif

void VolumeGVDB::TimerStart ()		{ PERF_START(); }
float VolumeGVDB::TimerStop ()		{ return PERF_STOP(); }

VolumeGVDB::VolumeGVDB ()
{
	TimeX start;

	gVDB = this;
	mPool = 0x0;
	mScene = 0x0;
	mOVDB = 0x0;
	mV3D = 0x0;
	mAtlasResize.Set ( 0, 20, 0 );
	mVoxsize.Set ( 1, 1, 1 );		// default voxel size
	mApron = 1;						// default apron

	mRoot = ID_UNDEFL;
	mTime = 0;	
	mbUseGLAtlas = false;

	mVDBInfo.update = true;	
	mVDBInfo.clr_chan = CHAN_UNDEF;

	mbProfile = false;
	mbVerbose = false;

	mDummyFrameBuffer = -1;

	for (int n=0; n < 5; n++ ) cuModule[n] = (CUmodule) -1;
	for (int n=0; n < MAX_FUNC; n++ ) cuFunc[n] = (CUfunction) -1;
	for (int n=0; n < 10; n++ ) { mTexIn[n] = ID_UNDEFL; mTexOut[n] = ID_UNDEFL; }
}

void VolumeGVDB::SetProfile ( bool pf ) 
{
	mbProfile = pf; 	
	PERF_SET ( pf, 0 );
	PERF_INIT ( 64, pf, pf, pf, 0, "" );
}

// Loads a CUDA function into memory from ptx file
void VolumeGVDB::LoadFunction ( int fid, std::string func, int mid, std::string ptx )
{
	char cptx[512];		strcpy ( cptx, ptx.c_str() );
	char cfn[512];		strcpy ( cfn, func.c_str() );

	if ( cuModule[mid] == (CUmodule) -1 ) 
		cudaCheck ( cuModuleLoad ( &cuModule[mid], cptx ), cptx, "ModuleLoad" );	
	if ( cuFunc[fid] == (CUfunction) -1 )
		cudaCheck ( cuModuleGetFunction ( &cuFunc[fid], cuModule[mid], cfn ), cfn, "ModuleGetFunction" );	
}

// Set the current CUDA device, load all GVDB kernels
void VolumeGVDB::SetCudaDevice ( int devid ) 
{
	size_t len = 0;

	StartCuda ( devid, mbVerbose );

	//--- Load cuda kernels
	// Raytracing
	LoadFunction ( FUNC_RAYDEEP,			"gvdbRayDeep",					MODL_PRIMARY, "cuda_gvdb_module.ptx" );
	LoadFunction ( FUNC_RAYVOXEL,			"gvdbRaySurfaceVoxel",			MODL_PRIMARY, "cuda_gvdb_module.ptx" );
	LoadFunction ( FUNC_RAYTRILINEAR,		"gvdbRaySurfaceTrilinear",		MODL_PRIMARY, "cuda_gvdb_module.ptx" );
	LoadFunction ( FUNC_RAYTRICUBIC,		"gvdbRaySurfaceTricubic",		MODL_PRIMARY, "cuda_gvdb_module.ptx" );
	LoadFunction ( FUNC_RAYSURFACE_DEPTH,	"gvdbRaySurfaceDepth",			MODL_PRIMARY, "cuda_gvdb_module.ptx" );
	LoadFunction ( FUNC_RAYLEVELSET,		"gvdbRayLevelSet",				MODL_PRIMARY, "cuda_gvdb_module.ptx" );
	LoadFunction ( FUNC_EMPTYSKIP,			"gvdbRayEmptySkip",				MODL_PRIMARY, "cuda_gvdb_module.ptx" );
	LoadFunction ( FUNC_SECTION2D,			"gvdbSection2D",				MODL_PRIMARY, "cuda_gvdb_module.ptx" );
	LoadFunction ( FUNC_SECTION3D,			"gvdbSection3D",				MODL_PRIMARY, "cuda_gvdb_module.ptx" );	
	LoadFunction ( FUNC_RAYTRACE,			"gvdbRaytrace",					MODL_PRIMARY, "cuda_gvdb_module.ptx" );
	
	// Sorting / Points / Triangles
	LoadFunction ( FUNC_PREFIXSUM,			"prefixSum",					MODL_PRIMARY, "cuda_gvdb_module.ptx" );
	LoadFunction ( FUNC_PREFIXFIXUP,		"prefixFixup",					MODL_PRIMARY, "cuda_gvdb_module.ptx" );
	LoadFunction ( FUNC_INSERT_POINTS,		"gvdbInsertPoints",				MODL_PRIMARY, "cuda_gvdb_module.ptx" );
	LoadFunction ( FUNC_SORT_POINTS,		"gvdbSortPoints",				MODL_PRIMARY, "cuda_gvdb_module.ptx" );	
	LoadFunction ( FUNC_SCATTER_DENSITY,	"gvdbScatterPointDensity",		MODL_PRIMARY, "cuda_gvdb_module.ptx" );
	LoadFunction ( FUNC_SCATTER_AVG_COL,	"gvdbScatterPointAvgCol",		MODL_PRIMARY, "cuda_gvdb_module.ptx" );
	LoadFunction ( FUNC_INSERT_TRIS,		"gvdbInsertTriangles",			MODL_PRIMARY, "cuda_gvdb_module.ptx" );
	LoadFunction ( FUNC_SORT_TRIS,			"gvdbSortTriangles",			MODL_PRIMARY, "cuda_gvdb_module.ptx" );
	LoadFunction ( FUNC_VOXELIZE,			"gvdbVoxelize",					MODL_PRIMARY, "cuda_gvdb_module.ptx" );
	LoadFunction ( FUNC_GATHER_DENSITY,		"gvdbGatherPointDensity",		MODL_PRIMARY, "cuda_gvdb_module.ptx" );
	LoadFunction ( FUNC_GATHER_VELOCITY,	"gvdbGatherPointVelocity",		MODL_PRIMARY, "cuda_gvdb_module.ptx" );
	LoadFunction ( FUNC_RESAMPLE,			"gvdbResample",					MODL_PRIMARY, "cuda_gvdb_module.ptx" );
	LoadFunction ( FUNC_DOWNSAMPLE,			"gvdbDownsample",				MODL_PRIMARY, "cuda_gvdb_module.ptx" );

	LoadFunction ( FUNC_ADD_SUPPORT_VOXEL,	"gvdbAddSupportVoxel",			MODL_PRIMARY, "cuda_gvdb_module.ptx" );
	LoadFunction ( FUNC_INSERT_SUPPORT_POINTS, "gvdbInsertSupportPoints",	MODL_PRIMARY, "cuda_gvdb_module.ptx" );

	// Apron Updates
	LoadFunction ( FUNC_UPDATEAPRON_F,		"gvdbUpdateApronF",				MODL_PRIMARY, "cuda_gvdb_module.ptx" );
	LoadFunction ( FUNC_UPDATEAPRON_F4,		"gvdbUpdateApronF4",			MODL_PRIMARY, "cuda_gvdb_module.ptx" );
	LoadFunction ( FUNC_UPDATEAPRON_C,		"gvdbUpdateApronC",				MODL_PRIMARY, "cuda_gvdb_module.ptx" );
	LoadFunction ( FUNC_UPDATEAPRON_C4,		"gvdbUpdateApronC4",			MODL_PRIMARY, "cuda_gvdb_module.ptx" );
	
	// Operators
	LoadFunction ( FUNC_FILL_F,				"gvdbOpFillF",					MODL_PRIMARY, "cuda_gvdb_module.ptx" );
	LoadFunction ( FUNC_FILL_C,				"gvdbOpFillC",					MODL_PRIMARY, "cuda_gvdb_module.ptx" );
	LoadFunction ( FUNC_FILL_C4,			"gvdbOpFillC4",					MODL_PRIMARY, "cuda_gvdb_module.ptx" );
	LoadFunction ( FUNC_SMOOTH,				"gvdbOpSmooth",					MODL_PRIMARY, "cuda_gvdb_module.ptx" );
	LoadFunction ( FUNC_NOISE,				"gvdbOpNoise",					MODL_PRIMARY, "cuda_gvdb_module.ptx" );	
	LoadFunction ( FUNC_CLR_EXPAND,			"gvdbOpClrExpand",				MODL_PRIMARY, "cuda_gvdb_module.ptx" );	
	LoadFunction ( FUNC_EXPANDC,			"gvdbOpExpandC",				MODL_PRIMARY, "cuda_gvdb_module.ptx" );	

	SetModule ( cuModule[MODL_PRIMARY] );	
}

// Reset to default module
void VolumeGVDB::SetModule ()
{
	SetModule ( cuModule[MODL_PRIMARY] );	
}

// Set to a user-defined module (application)
void VolumeGVDB::SetModule ( CUmodule module )
{
	cudaCheck ( cuCtxSynchronize (), "cuCtxSync", "SetModule" );

	ClearAtlasAccess ();

	size_t len = 0;
	cudaCheck ( cuModuleGetGlobal ( &cuVDBInfo, &len,	module, "gvdb" ),		"cuModuleGetGlobal(cuVDBInfo)", "SetModule" );
	cudaCheck ( cuModuleGetGlobal ( &cuScnInfo, &len,	module, "scn" ),		"cuModuleGetGlobal(cuScnInfo)", "SetModule" );	

	cudaCheck ( cuModuleGetSurfRef( &cuSurfWrite,		module, "volTexOut" ),	"cuModuleGetSurfRef", "SetModule" );
	cudaCheck ( cuModuleGetTexRef ( &cuSurfRead,		module, "volTexIn" ),	"cuModuleGetTexRef", "SetModule" );

	cudaCheck ( cuModuleGetGlobal ( &cuTexIn,  &len,	module, "volIn" ),		"cuModuleGetGlobal(cuTexIn)", "SetModule" );	
	cudaCheck ( cuModuleGetGlobal ( &cuTexOut,  &len,	module, "volOut" ),		"cuModuleGetGlobal(cuTexOut)", "SetModule" );	

	cudaCheck ( cuModuleGetGlobal ( &cuXform,  &len,	module, "cxform" ),		"cuModuleGetGlobal(cuXform)", "SetModule" );	
	cudaCheck ( cuModuleGetGlobal ( &cuDebug,  &len,	module, "cdebug" ),		"cuModuleGetGlobal(cuDebug)", "SetModule" );	

	SetupAtlasAccess ();

	mVDBInfo.update = true;
}

float* ConvertToScalar ( int cnt, float* inbuf, float* outbuf, float& vmin, float& vmax )
{
	float* outv = outbuf;
	Vector3DF* vec = (Vector3DF*) inbuf;	
	float val;	
	for ( int n=0; n < cnt; n++ ) {
		val = vec->Length();
		if ( val < vmin ) vmin = val;
		if ( val > vmax ) vmax = val;		
		*outv++ = val;
		vec++;		
	}
	return outbuf;
}

// Simple Importer for ASCII VTK files
// - Only supports STRUCTURED_POINTS
bool VolumeGVDB::ImportVTK ( std::string fname, std::string field, Vector3DI& res )
{
	char buf[2048];
	strcpy (buf, fname.c_str() );
	FILE* fp = fopen ( buf, "rt" );
	std::string lin, word;
	bool bAscii = false;	
	float f;
	float* vox = 0x0;
	long cnt = 0, max_cnt = 0;
	char status = 'X'; 

	verbosef ( "Loading VTK: %s\n", buf );

	while ( !feof(fp) && status!='D' ) {
		if ( fgets ( buf, 2048, fp ) == NULL ) break;
		lin = std::string(buf);

		word = strSplit ( lin, " \n" );
		if ( word.compare("vtk")==0 ) continue;
		if ( word.compare("ASCII")==0 ) {bAscii=true; continue;}
		if ( word.compare("DATASET")==0 ) {
			word = strSplit ( lin, " \n" );
			if (word.compare("STRUCTURED_POINTS")==0) continue;
			gprintf ( "ERROR: ImportVTK currently only supports STRUCTURED_POINTS\n" );
			return false;
		}
		if ( word.compare("DIMENSIONS")==0 ) {
			word = strSplit ( lin, " \n" ); strIsNum(word, f ); res.x = f-1;
			word = strSplit ( lin, " \n" ); strIsNum(word, f ); res.y = f-1;
			word = strSplit ( lin, " \n" ); strIsNum(word, f ); res.z = f-1;
			verbosef ( "  Res: %d, %d, %d\n", res.x, res.y, res.z );
			PrepareAux ( AUX_DATA3D, res.x*res.y*res.z, sizeof(float), false, true );
			vox = (float*) mAux[AUX_DATA3D].cpu;
			continue;
		}
		if ( word.compare("FIELD")==0 ) {
			// get next line
			if ( fgets ( buf, 2048, fp ) == NULL ) break;
			lin = std::string(buf);
			word = strSplit ( lin, " \n" );
			// if desired field is found, start reading
			if ( word.compare ( field ) == 0 ) status = 'R';
			word = strSplit ( lin, " \n" );
			word = strSplit ( lin, " \n" ); strIsNum(word, f); max_cnt = f;
			verbosef ( "  Reading: %s, %ld\n", field.c_str(), max_cnt );
			continue;
		}
		if ( word.compare("SPACING")==0 ) continue;
		if ( word.compare("ORIGIN")==0 ) continue;
		if ( word.compare("CELL_DATA")==0 ) continue;
		if ( word.compare("SCALARS")==0 ) continue;
		if ( word.compare("LOOKUP_TABLE")==0 ) continue;

		if ( status=='R' ) {
			while ( strIsNum(word, f) && !lin.empty() ) {
				*vox++ = f;				
				word = strSplit( lin, " \n" );
			}
			cnt = vox - (float*) mAux[AUX_DATA3D].cpu;
			if ( cnt >= max_cnt) status = 'D';		// done
		}
	}
	verbosef ( "  Values Read: %d\n", cnt );

	// Commit data to GPU
	CommitData ( mAux[AUX_DATA3D] );

	return true;
}

// Load a VBX file
bool VolumeGVDB::LoadVBX ( std::string fname )
{
	char buf[2048];
	strcpy ( buf, fname.c_str() );
	FILE* fp = fopen ( buf, "rb" );
	
	if ( mbProfile ) PERF_PUSH ( "Read VBX" );	
	
	verbosef ( "LoadVBX: %s\n", fname.c_str() );
		verbosef ( "Sizes: char %d, int %d, u64 %d, float %d\n", sizeof(char), sizeof(int), sizeof(uint64), sizeof(float) );

	// Read VDB config
	uchar major, minor;
	int num_grids;
	char grid_name[512];
	char grid_components;
	char grid_dtype;
	char grid_compress;
	char grid_topotype;
	int  grid_reuse;
	char grid_layout;
	int levels, leafcnt, apron;
	int num_chan;
	Vector3DI axisres, axiscnt, leafdim;
	int ld[MAXLEV], res[MAXLEV];
	Vector3DI range[MAXLEV];
	int cnt0[MAXLEV], cnt1[MAXLEV];
	int width0[MAXLEV], width1[MAXLEV];
	Vector3DF voxelsize;	
	uint64 atlas_sz, root;

	std::vector<uint64> grid_offs;

	//--- gvdb header
	fread ( &major, sizeof(uchar), 1, fp );					// major version
	fread ( &minor, sizeof(uchar), 1, fp );					// minor version
	fread ( &num_grids, sizeof(int), 1, fp );				// number of grids
	for (int n=0; n < num_grids; n++ ) {
		grid_offs.push_back(0);
		fread ( &grid_offs[n], sizeof(uint64), 1, fp );		// grid offsets
	}

	for (int n=0; n < num_grids; n++ ) {
		
		//---- grid header
		fread ( &grid_name, 256, 1, fp );					// grid name		
		fread ( &grid_dtype, sizeof(uchar), 1, fp );		// grid data type
		fread ( &grid_components, sizeof(uchar), 1, fp );	// grid components
		fread ( &grid_compress, sizeof(uchar), 1, fp );		// grid compression (0=none, 1=blosc, 2=..)
		fread ( &voxelsize, sizeof(float), 3, fp );			// voxel size
		fread ( &leafcnt, sizeof(int), 1, fp );				// total brick count
		fread ( &leafdim.x, sizeof(int), 3, fp );			// brick dimensions
		fread ( &apron, sizeof(int), 1, fp );				// brick apron
		fread ( &num_chan, sizeof(int), 1, fp );			// number of channels
		fread ( &atlas_sz, sizeof(uint64), 1, fp );			// total atlas size (all channels)
		fread ( &grid_topotype, sizeof(uchar), 1, fp );		// topology type? (0=none, 1=reuse, 2=gvdb, 3=..)
		fread ( &grid_reuse, sizeof(int), 1, fp);			// topology reuse
		fread ( &grid_layout, sizeof(uchar), 1, fp);		// brick layout? (0=atlas, 1=brick)
		fread ( &axiscnt.x, sizeof(int), 3, fp );			// atlas brick count
		fread ( &axisres.x, sizeof(int), 3, fp );			// atlas res
		
	
		//---- topology section
		fread ( &levels, sizeof(int), 1, fp );				// num levels
		fread ( &root, sizeof(uint64), 1, fp );			// root id	
		for (int n=0; n < levels; n++ ) {				
			fread ( &ld[n], sizeof(int), 1, fp );
			fread ( &res[n], sizeof(int), 1, fp );
			fread ( &range[n].x, sizeof(int), 1, fp );
			fread ( &range[n].y, sizeof(int), 1, fp );
			fread ( &range[n].z, sizeof(int), 1, fp );
			fread ( &cnt0[n], sizeof(int), 1, fp );			
			fread ( &width0[n], sizeof(int), 1, fp );
			fread ( &cnt1[n], sizeof(int), 1, fp );
			fread ( &width1[n], sizeof(int), 1, fp );			
		}	
		if ( width0[0] != sizeof(nvdb::Node) ) {
			gprintf ( "ERROR: VBX file contains nodes incompatible with current gvdb_library.\n" );
			gprintf ( "       Size in file: %d,  Size in library: %d\n", width0[0], sizeof(nvdb::Node) );
			gerror ();
		}
	
		// Initialize GVDB
		Configure ( levels, ld, cnt0 );
		SetVoxelSize ( voxelsize.x, voxelsize.y, voxelsize.z );
		mRoot = root;		// must be set after initialize

		// Read topology
		for (int n=0; n < levels; n++ ) 
			mPool->PoolRead ( fp, 0, n, cnt0[n], width0[n] );
		for (int n=0; n < levels; n++ )
			mPool->PoolRead ( fp, 1, n, cnt1[n], width1[n] );

		FinishTopology ();

		// Atlas section
		DestroyChannels ();
		
		// Read atlas into GPU slice-by-slice to conserve CPU and GPU mem		
		for (int chan = 0 ; chan < num_chan; chan++ ) {			
		
			uint64 cpos = ftell ( fp );				

			int chan_type, chan_stride;
			fread ( &chan_type, sizeof(int), 1, fp );
			fread ( &chan_stride, sizeof(int), 1, fp );
			
			AddChannel ( chan, chan_type, apron, axiscnt );		// provide axiscnt
					
			mPool->AtlasSetNum ( chan, cnt0[0] );		// assumes atlas contains all bricks (all are resident)

			DataPtr slice;			
			mPool->CreateMemLinear ( slice, 0x0, chan_stride, axisres.x*axisres.y, true );
			for (int z = 0; z < axisres.z; z++ ) {
				fread ( slice.cpu, slice.size, 1, fp );
				mPool->AtlasWriteSlice ( chan, z, slice.size, slice.gpu, (uchar*) slice.cpu );		// transfer from GPU, directly into CPU atlas				
			}
			mPool->FreeMemLinear ( slice );	
		}
		UpdateAtlas ();
	}	

	if ( mbProfile ) PERF_POP ();

	return true;
}

// Set the current color channel for rendering
void VolumeGVDB::SetColorChannel ( uchar chan )
{
	mVDBInfo.clr_chan = chan;
}

// Clear device access to atlases
void VolumeGVDB::ClearAtlasAccess ()
{
	if ( mPool==0x0 ) return;

	int num_chan = mPool->getNumAtlas();
	for (int chan=0; chan < num_chan; chan++ ) {
		if ( mTexIn[chan] != ID_UNDEFL ) {
			cudaCheck ( cuTexObjectDestroy ( mTexIn[chan] ), "cuTexObjectDestroy", "ClearAtlasAccess" );		
			mTexIn[chan] = ID_UNDEFL;
		}
		if ( mTexOut[chan] != ID_UNDEFL ) {
			cudaCheck ( cuSurfObjectDestroy ( mTexOut[chan] ), "cuSurfObjectDestroy", "ClearAtlasAccess" );
			mTexOut[chan] = ID_UNDEFL;
		}
	}
	cudaCheck ( cuCtxSynchronize (), "cuCtxSync", "ClearAtlasAccess" );

}

// Setup device access to atlases
void VolumeGVDB::SetupAtlasAccess ()
{	
	if ( mPool == 0x0 ) return;
	if ( mPool->getNumAtlas() == 0 ) return;

	//-- Texture Access using TexRefs
	DataPtr atlas = mPool->getAtlas(0);		
	cudaCheck ( cuTexRefSetFilterMode ( cuSurfRead, CU_TR_FILTER_MODE_LINEAR ), "cuTexRefSetFilterMode", "SetupAtlasAccess" );
	cudaCheck ( cuTexRefSetArray ( cuSurfRead,  reinterpret_cast<CUarray>(atlas.garray), CU_TRSA_OVERRIDE_FORMAT  ), "cuTexRefSetArray", "SetupAtlasAccess" );	
	cudaCheck ( cuSurfRefSetArray( cuSurfWrite, reinterpret_cast<CUarray>(atlas.garray), 0 ), "cuSurfRefSetArray", "SetupAtlasAccess" );

	//-- Texture Access using CUDA Bindless
	int num_chan = mPool->getNumAtlas();
	for (int chan=0; chan < num_chan; chan++ ) {
	
		CUDA_RESOURCE_DESC resDesc;
		memset ( &resDesc, 0, sizeof(resDesc) );
		resDesc.resType = CU_RESOURCE_TYPE_ARRAY;
		resDesc.res.array.hArray = mPool->getAtlas(chan).garray;
		resDesc.flags = 0;		

		CUDA_TEXTURE_DESC texDesc;
		memset ( &texDesc, 0, sizeof(texDesc) );				
		switch ( mPool->getAtlas(chan).type ) {
		case T_FLOAT: case T_FLOAT3: case T_FLOAT4:
			texDesc.filterMode = CU_TR_FILTER_MODE_POINT;
			texDesc.flags = 0;
			break;
		case T_UCHAR: case T_UCHAR3: case T_UCHAR4:	case T_INT: case T_INT3: case T_INT4:
			texDesc.filterMode = CU_TR_FILTER_MODE_POINT;
			texDesc.flags = CU_TRSF_READ_AS_INTEGER;		// read as integer
			break;
		}
		texDesc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
		texDesc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
		texDesc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;		
		
		cudaCheck ( cuTexObjectCreate ( &mTexIn[chan], &resDesc, &texDesc, NULL ), "cuTexObjectCreate", "SetupAtlasAccess" );		
		cudaCheck ( cuSurfObjectCreate ( &mTexOut[chan], &resDesc ), "cuSurfObjectCreate", "SetupAtlasAccess" );							
	}	
	// Transmit tex/surf objects to CUDA variables
	cudaCheck ( cuMemcpyHtoD ( cuTexIn, &mTexIn[0], sizeof(CUtexObject) * 10  ), "cuMemcpyHtoD(cuTexIn)", "SetupAtlasAccess" );
	cudaCheck ( cuMemcpyHtoD ( cuTexOut, &mTexOut[0], sizeof(CUsurfObject) * 10 ), "cuMemcpyHtoD(cuTexOut)", "SetupAtlasAccess" );	

	cudaCheck ( cuCtxSynchronize (), "cuCtxSync", "SetupAtlasAccess" );
}


// Finish a topology update
void VolumeGVDB::FinishTopology ()
{
	// compute bounds
	ComputeBounds ();

	// commit topology
	mPool->PoolCommitAll ();

	// update VDB data on gpu 
	mVDBInfo.update = true;	
}

// Clear the atlas. Fill all channels with 0	
void VolumeGVDB::ClearAtlas ()
{
	if ( mbProfile ) PERF_PUSH ( "Clear Atlas" );
	for (int n=0; n < mPool->getNumAtlas(); n++ )
		mPool->AtlasFill ( n );	

	if ( mbProfile ) PERF_POP ();
}

// Save a VBX file
void VolumeGVDB::SaveVBX ( std::string fname )
{
	int cnt[2], width[2];
	Vector3DI range;
	char buf[512];
	strcpy ( buf, fname.c_str() );
	
	FILE* fp = fopen ( buf, "wb" );

	uchar major = MAJOR_VERSION;
	uchar minor = MINOR_VERSION;
	
	if ( mbProfile ) PERF_PUSH ( "Saving VBX" );	

	verbosef ( "  Saving VBX (ver %d.%d)\n", major, minor );

	int levels = mPool->getNumLevels();

	int		num_grids = 1;	
	char	grid_name[512]; 
	sprintf( grid_name, "" );	
	char	grid_components = 1;						// one component
	char	grid_dtype = 'f';							// float
	char	grid_compress = 0;							// no compression
	char	grid_topotype = 2;							// gvdb topology
	int		grid_reuse = 0;
	char	grid_layout = 0;							// atlas layout

	int		leafcnt = mPool->getPoolCnt(0,0);			// brick count
	int		res = getRes(0);
	Vector3DI leafdim = Vector3DI(res,res,res);			// brick resolution
	int		apron	= mPool->getAtlas(0).apron;			// brick apron
	Vector3DI axiscnt = mPool->getAtlas(0).subdim;		// atlas count
	Vector3DI axisres = mPool->getAtlasRes(0);			// atlas res	
	slong	atlas_sz = mPool->getAtlas(0).size;			// atlas size
	
	std::vector<uint64>	grid_offs;
	for (int n=0; n < num_grids; n++)
		grid_offs.push_back ( 0 );

	///--- gvdb file header
	fwrite ( &major, sizeof(uchar), 1, fp );				// major version
	fwrite ( &minor, sizeof(uchar), 1, fp );				// minor version
	fwrite ( &num_grids, sizeof(int), 1, fp );				// number of grids - future expansion
	uint64 grid_table = ftell ( fp );						// position of grid table
	for (int n=0; n < num_grids; n++ ) {
		fwrite ( &grid_offs[n], sizeof(uint64), 1, fp );		// grid offsets (populated later)
	}
	int num_chan = mPool->getNumAtlas();

	for (int n=0; n < num_grids; n++ ) {
		grid_offs[n] = ftell ( fp );						// record grid offset

		//---- grid header
		fwrite ( &grid_name, 256, 1, fp );					// grid name		
		fwrite ( &grid_dtype, sizeof(uchar), 1, fp );		// grid data type
		fwrite ( &grid_components, sizeof(uchar), 1, fp );	// grid components
		fwrite ( &grid_compress, sizeof(uchar), 1, fp );	// grid compression (0=none, 1=blosc, 2=..)
		fwrite ( &mVoxsize.x, sizeof(float), 3, fp );		// voxel size
		fwrite ( &leafcnt, sizeof(int), 1, fp );			// total brick count
		fwrite ( &leafdim.x, sizeof(int), 3, fp );			// brick dimensions
		fwrite ( &apron, sizeof(int), 1, fp );				// brick apron
		fwrite ( &num_chan, sizeof(int), 1, fp );			// number of channels
		fwrite ( &atlas_sz, sizeof(uint64), 1, fp );			// total atlas size (all channels)
		fwrite ( &grid_topotype, sizeof(uchar), 1, fp );	// topology type? (0=none, 1=reuse, 2=gvdb, 3=..)
		fwrite ( &grid_reuse, sizeof(int), 1, fp);			// topology reuse
		fwrite ( &grid_layout, sizeof(uchar), 1, fp);		// brick layout? (0=atlas, 1=brick)
		fwrite ( &axiscnt.x, sizeof(int), 3, fp );			// atlas axis count
		fwrite ( &axisres.x, sizeof(int), 3, fp );			// atlas res			

		//---- topology section
		fwrite ( &levels, sizeof(int), 1, fp );				// num levels
		fwrite ( &mRoot, sizeof(uint64), 1, fp );			// root id	
		for (int n=0; n < levels; n++ ) {				
			res = getRes(n); range = getRange(n);			
			width[0] = mPool->getPoolWidth(0,n);
			width[1] = mPool->getPoolWidth(1,n);
			cnt[0] = mPool->getPoolCnt(0,n);
			cnt[1] = mPool->getPoolCnt(1,n);
			fwrite ( &mLogDim[n], sizeof(int), 1, fp );
			fwrite ( &res, sizeof(int), 1, fp );
			fwrite ( &range.x, sizeof(int), 1, fp );
			fwrite ( &range.y, sizeof(int), 1, fp );
			fwrite ( &range.z, sizeof(int), 1, fp );
			fwrite ( &cnt[0],   sizeof(int), 1, fp );			
			fwrite ( &width[0], sizeof(int), 1, fp );		
			fwrite ( &cnt[1],   sizeof(int), 1, fp );
			fwrite ( &width[1], sizeof(int), 1, fp );			
		}	
		for (int n=0; n < levels; n++ )						// write pool 0 
			mPool->PoolWrite ( fp, 0, n );			
		for (int n=0; n < levels; n++ )						// write pool 1 
			mPool->PoolWrite ( fp, 1, n );

		//---- atlas section
		// readback slice-by-slice from gpu to conserve CPU and GPU mem	

		for (int chan = 0 ; chan < num_chan; chan++ ) {
			DataPtr slice;
			int chan_type = mPool->getAtlas(chan).type ;
			int chan_stride = mPool->getSize ( chan_type ); 
			uint64 cpos = ftell ( fp );				

			fwrite ( &chan_type, sizeof(int), 1, fp );
			fwrite ( &chan_stride, sizeof(int), 1, fp );
			mPool->CreateMemLinear ( slice, 0x0, chan_stride, axisres.x*axisres.y, true );

			for (int z = 0; z < axisres.z; z++ ) {
				mPool->AtlasRetrieveSlice ( chan, z, slice.size, slice.gpu, (uchar*) slice.cpu );		// transfer from GPU, directly into CPU atlas		
				fwrite ( slice.cpu, slice.size, 1, fp );
			}
			mPool->FreeMemLinear ( slice );
		}
	}
	// update grid offsets table
	for (int n=0; n < num_grids; n++ ) {
		fseek ( fp, 6 + n*sizeof(uint64), SEEK_SET );
		fwrite ( &grid_offs[n], sizeof(uint64), 1, fp );		// grid offsets
	}
		
	fclose ( fp );	

	if ( mbProfile ) PERF_POP ();
}

// Compute bounding box of entire volume.
// - This is done by finding the min/max of all bricks
void VolumeGVDB::ComputeBounds ()
{
	Vector3DI range = getRange(0);
	Node* curr = getNode ( 0, 0, 0 );	
	mVoxMin = curr->mPos;
	mVoxMax = mVoxMin;
	for (int n=0; n < mPool->getPoolCnt(0,0); n++ ) {
		curr = getNode ( 0, 0, n );
		if ( curr->mPos.x < mVoxMin.x ) mVoxMin.x = curr->mPos.x;
		if ( curr->mPos.y < mVoxMin.y ) mVoxMin.y = curr->mPos.y;
		if ( curr->mPos.z < mVoxMin.z ) mVoxMin.z = curr->mPos.z;		
		if ( curr->mPos.x + range.x > mVoxMax.x ) mVoxMax.x = curr->mPos.x + range.x;
		if ( curr->mPos.y + range.y > mVoxMax.y ) mVoxMax.y = curr->mPos.y + range.y;
		if ( curr->mPos.z + range.z > mVoxMax.z ) mVoxMax.z = curr->mPos.z + range.z;		
	}
	mObjMin = mVoxMin;	mObjMin *= mVoxsize;
	mObjMax = mVoxMax;  mObjMax *= mVoxsize;
	mVoxRes = mVoxMax;  mVoxRes -= mVoxMin;

	if ( mVoxRes.x > mVoxResMax.x ) mVoxResMax.x = mVoxRes.x;
	if ( mVoxRes.y > mVoxResMax.y ) mVoxResMax.y = mVoxRes.y;
	if ( mVoxRes.z > mVoxResMax.z ) mVoxResMax.z = mVoxRes.z;
}

#ifdef BUILD_OPENVDB
	
	void vdbSkip( OVDBGrid* ovg, int leaf_start, int gt, bool isFloat )
	{
		switch ( gt ) {
		case 0:	
			if ( isFloat )  { ovg->iter543F = ovg->grid543F->tree().cbeginLeaf();  for (int j=0; ovg->iter543F && j < leaf_start; j++) ++ovg->iter543F; }
			else			{ ovg->iter543VF = ovg->grid543VF->tree().cbeginLeaf(); for (int j=0; ovg->iter543VF && j < leaf_start; j++) ++ovg->iter543VF; }
			break;
		case 1:	
			if ( isFloat )	{ ovg->iter34F = ovg->grid34F->tree().cbeginLeaf();  for (int j=0; ovg->iter34F && j < leaf_start; j++) ++ovg->iter34F; }
			else			{ ovg->iter34VF = ovg->grid34VF->tree().cbeginLeaf(); for (int j=0; ovg->iter34VF && j < leaf_start; j++) ++ovg->iter34VF; }
			break;
		};
	}

	bool vdbCheck( OVDBGrid* ovg, int gt, bool isFloat )
	{
		switch ( gt ) {
		case 0: return ( isFloat ? ovg->iter543F.test() : ovg->iter543VF.test() );	break;
		case 1: return ( isFloat ? ovg->iter34F.test() : ovg->iter34VF.test() );	break;
		};
		return false;
	}
	void vdbOrigin ( OVDBGrid* ovg, Coord& orig, int gt, bool isFloat )
	{
		switch ( gt ) {
		case 0: if ( isFloat) (ovg->iter543F)->getOrigin( orig ); else (ovg->iter543VF)->getOrigin( orig );	break;
		case 1: if ( isFloat) (ovg->iter34F)->getOrigin( orig ); else (ovg->iter34VF)->getOrigin( orig );	break;
		};
	}

	void vdbNext ( OVDBGrid* ovg, int gt, bool isFloat )
	{
		switch ( gt ) {
		case 0: if ( isFloat) ovg->iter543F.next();	else ovg->iter543VF.next();		break;
		case 1: if ( isFloat) ovg->iter34F.next();	else ovg->iter34VF.next();		break;
		};
	}

#endif

	
#ifdef WRITE_TEST
	openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create ( 0.0 );
	grid->setTransform ( openvdb::math::Transform::createLinearTransform( 0.12 ) );
	grid->setGridClass ( openvdb::GRID_FOG_VOLUME );
	grid->setName ( "Rama's test" );

	GridType::Accessor accessor = grid.get()->getAccessor ();

	for (int x=0; x < 32; x++ ) {
		for (int y=0; y < 32; y++ ) {
			for (int z=0; z < 32; z++ ) {
				float d = (1.0-sqrt((x-16.0)*(x-16.0) + (y-16.0)*(y-16.0) + (z-16.0)*(z-16.0))/16.0) * 0.10;
				if ( x>16 && y>16 && z>16 ) d = 0;
				accessor.setValue ( Coord(x,y,z), d );
			}
		}
	}

	// axis test
	//for (int x=0; x < 6; x++ ) accessor.setValue ( Coord(x,0,0), 1.0 );	
	//for (int y=0; y < 3; y++ ) accessor.setValue ( Coord(0,y,0), 1.0 );
	//for (int z=0; z < 11; z++ ) accessor.setValue ( Coord(0,0,z), 1.0 ); 
		
	openvdb::io::File file ( "test2.vdb" );
	openvdb::GridPtrVec grids;
	grids.push_back ( grid );
	file.write ( grids );
	file.close ();

	verbosef ( "Done\n" );

#endif

// Load a raw BRK file
bool VolumeGVDB::LoadBRK ( std::string fname )
{
	char fn[1024], buf[1024];
	strcpy ( fn, fname.c_str() );
	FILE* fp = fopen ( fn, "rb" );

	slong leaf;
	int leaf_cnt = 0;
	Vector3DF bmin, bmax;
	Vector3DI bres, bndx, bcurr;
	int brkcnt;
	float* brick = 0x0;
	bcurr.Set ( 0, 0, 0 );	

	Volume3D vtemp ( mScene );

	sprintf ( buf, "Reading BRK %s", fname.c_str() );
	verbosef ( "  %s\n", buf );
	
	if ( mbProfile ) PERF_PUSH ( buf );	
	
	fread ( &brkcnt, sizeof(int), 1, fp );
	verbosef ( "    Number of bricks: %d\n", brkcnt );

	// Read first brick for res
	fread ( &bndx, sizeof(Vector3DI), 1, fp );
	fread ( &bmin, sizeof(Vector3DF), 1, fp );
	fread ( &bmax, sizeof(Vector3DF), 1, fp );
	fread ( &bres, sizeof(Vector3DI), 1, fp );		// needed to create atlas
	bcurr = bres;
	if ( brick != 0x0 ) free ( brick );
	brick = (float*) malloc ( bres.x*bres.y*bres.z * sizeof(float) );	// allocate brick memory
	vtemp.Resize ( T_FLOAT, bres, 0x0, false );
	fseek ( fp, 0, SEEK_SET );
	fread ( &brkcnt, sizeof(int), 1, fp );

	// Adjust VDB config if necessary	
	int res = bres.x;
	mVCFG[4] = 0;
	while ( res >>= 1 ) ++mVCFG[4];
	verbosef ( "    Leaf res: %d (leaf log2=%d)\n", bres.x, mVCFG[4] );

	// Initialize VDB
	Vector3DF voxelsize ( 1, 1, 1 );
	Configure ( mVCFG[0], mVCFG[1], mVCFG[2], mVCFG[3], mVCFG[4] );
	SetVoxelSize ( voxelsize.x, voxelsize.y, voxelsize.z );
	SetApron ( 1 );
	
	// Create atlas
	mPool->AtlasReleaseAll ();	
	if ( mbProfile ) PERF_PUSH ( "Create Atlas" );	
	
	int side = ceil ( pow ( brkcnt, 1/3.0f ) );		// number of leafs along one axis	
	Vector3DI axiscnt (side, side, side);
	mPool->AtlasCreate ( 0, T_FLOAT, bres, axiscnt, mApron, sizeof(AtlasNode), false, mbUseGLAtlas );
	if ( mbProfile ) PERF_POP ();
	
	float vmin = +1.0e20, vmax = -1.0e20;

	// Read all bricks
	Vector3DF t;

	if ( mbProfile ) PERF_PUSH ( "Load Bricks" );	
	for (int n=0; n < brkcnt; n++ ) {
		
		// Read brick dimensions
		PERF_START ();
		fread ( &bndx, sizeof(Vector3DI), 1, fp );
		fread ( &bmin, sizeof(Vector3DF), 1, fp );
		fread ( &bmax, sizeof(Vector3DF), 1, fp );
		fread ( &bres, sizeof(Vector3DI), 1, fp );		
		if ( bcurr.x != bres.x || bcurr.y != bres.y || bcurr.z != bres.z ) {
			gprintf ( "ERROR: Bricks do not have same resolution.\n" );
			exit (-1);
		}
		// Read brick
		fread ( brick, sizeof(float), bres.x*bres.y*bres.z, fp );
		t.x += PERF_STOP ();

		// Activate space
		PERF_START ();
		bool bnew = false;
		leaf = ActivateSpace ( mRoot, bndx, bnew );
		t.y += PERF_STOP ();

		if ( leaf != ID_UNDEFL ) {

			PERF_START ();
			// Copy data from CPU into 3D Texture
			vtemp.SetDomain ( bmin, bmax );
			vtemp.CommitFromCPU ( brick );					

			// Create VDB Atlas value and sub-copy 3D texture into it
			Vector3DI brickpos;
			Node* node;
			if ( mPool->AtlasAlloc ( 0, brickpos ) ) {
				node = getNode ( leaf );
				node->mValue = brickpos; 
				// gprintf ( "%d %d %d: %d %d %d, %lld\n", node->mPos.x, node->mPos.y, node->mPos.z, brickpos.x, brickpos.y, brickpos.z, leaf );				
				mPool->AtlasCopyTex ( 0, brickpos, vtemp.getPtr() );
			}
			leaf_cnt++;
			t.z += PERF_STOP ();
		}
	}
	if ( mbProfile ) PERF_POP ();

	verbosef( "    Read Brk: %f ms\n", t.x );
	verbosef( "    Activate: %f ms\n", t.y );
	verbosef( "    To Atlas: %f ms\n", t.z );

	mVoxsize = voxelsize;
	ComputeBounds ();

	// Commit all node pools to gpu
	if ( mbProfile ) PERF_PUSH ( "Commit" );		
	mPool->PoolCommitAll ();		
	if ( mbProfile ) PERF_POP ();

	if ( mbProfile ) PERF_POP ();

	FinishTopology ();
	UpdateApron ();

	return true;
}


// Load an OpenVDB file
// - Supports <5,4,3> (default) and <3,3,3,4> trees
bool VolumeGVDB::LoadVDB ( std::string fname )
{
	
#ifdef BUILD_OPENVDB

	openvdb::initialize ();	
	FloatGrid34::registerGrid();	
	//FloatGridVF34::registerGrid();

	mOVDB = new OVDBGrid;

	CoordBBox box;
	Coord orig;
	Vector3DF p0, p1;

	if ( mbProfile ) PERF_PUSH ( "Clear grid" );

	if ( mOVDB->grid543F != 0x0 ) 	mOVDB->grid543F.reset();		
	if ( mOVDB->grid543VF != 0x0 ) 	mOVDB->grid543VF.reset();		
	if ( mOVDB->grid34F != 0x0 ) 	mOVDB->grid34F.reset();		
	if ( mOVDB->grid34VF != 0x0 ) 	mOVDB->grid34VF.reset();				

	if ( mbProfile ) PERF_POP ();

	if ( mbProfile ) PERF_PUSH ( "Load VDB" );	

	// Read .vdb file	

	verbosef ( "   Reading OpenVDB file.\n" );
	openvdb::io::File* vdbfile = new openvdb::io::File ( fname );
	vdbfile->open();	
	
	// Read grid		
	openvdb::GridBase::Ptr baseGrid;
	openvdb::io::File::NameIterator nameIter = vdbfile->beginName();	
	std::string name = vdbfile->beginName().gridName();
	for ( openvdb::io::File::NameIterator nameIter = vdbfile->beginName(); nameIter != vdbfile->endName(); ++nameIter ) {
		verbosef ( "   Grid: %s\n", nameIter.gridName().c_str() );
		if ( nameIter.gridName().compare( getScene()->mVName ) == 0 ) name = getScene()->mVName;
	}	
	verbosef ( "   Loading Grid: %s\n", name.c_str() );
	baseGrid = vdbfile->readGrid ( name ); 
	
	if ( mbProfile ) PERF_POP ();

	// Initialize GVDB config
	Vector3DF voxelsize;
	int gridtype = 0;

	bool isFloat = false;

	verbosef ( "   Configuring GVDB.\n");
	if ( baseGrid->isType< FloatGrid543 >() ) {
		gridtype = 0;
		isFloat = true;
		mOVDB->grid543F = openvdb::gridPtrCast< FloatGrid543 >(baseGrid);	
		voxelsize.Set ( mOVDB->grid543F->voxelSize().x(), mOVDB->grid543F->voxelSize().y(), mOVDB->grid543F->voxelSize().z() );			
		Configure ( 5, 5, 5, 4, 3 );
	}
	if ( baseGrid->isType< Vec3fGrid543 >() ) {
		gridtype = 0;
		isFloat = false;
		mOVDB->grid543VF = openvdb::gridPtrCast< Vec3fGrid543 >(baseGrid);	
		voxelsize.Set ( mOVDB->grid543VF->voxelSize().x(), mOVDB->grid543VF->voxelSize().y(), mOVDB->grid543VF->voxelSize().z() );	
		Configure ( 5, 5, 5, 4, 3 );
	}	 
	if ( baseGrid->isType< FloatGrid34 >() ) {
		gridtype = 1;
		isFloat = true;
		mOVDB->grid34F = openvdb::gridPtrCast< FloatGrid34 >(baseGrid);	
		voxelsize.Set ( mOVDB->grid34F->voxelSize().x(), mOVDB->grid34F->voxelSize().y(), mOVDB->grid34F->voxelSize().z() );	
		Configure ( 3, 3, 3, 3, 4 );
	}
	if ( baseGrid->isType< Vec3fGrid34 >() ) {
		gridtype = 1;
		isFloat = false;
		mOVDB->grid34VF = openvdb::gridPtrCast< Vec3fGrid34 >(baseGrid);	
		voxelsize.Set ( mOVDB->grid34VF->voxelSize().x(), mOVDB->grid34VF->voxelSize().y(), mOVDB->grid34VF->voxelSize().z() );	
		Configure ( 3, 3, 3, 3, 4 );
	}
	SetVoxelSize ( voxelsize.x, voxelsize.y, voxelsize.z );
	SetApron ( 1 );

	float pused = MeasurePools ();
	verbosef( "   Topology Used: %6.2f MB\n", pused );
	
	slong leaf;
	int leaf_start = 0;				// starting leaf		gScene.mVLeaf.x;		
	int n, leaf_max, leaf_cnt = 0;
	Vector3DF vclipmin, vclipmax, voffset;
	vclipmin = getScene()->mVClipMin;
	vclipmax = getScene()->mVClipMax;	

	// Determine Volume bounds
	verbosef( "   Compute volume bounds.\n");
	vdbSkip ( mOVDB, leaf_start, gridtype, isFloat );
	for (leaf_max=0; vdbCheck ( mOVDB, gridtype, isFloat ); ) {
		vdbOrigin ( mOVDB, orig, gridtype, isFloat );
		p0.Set ( orig.x(), orig.y(), orig.z() );
		if ( p0.x > vclipmin.x && p0.y > vclipmin.y && p0.z > vclipmin.z && p0.x < vclipmax.x && p0.y < vclipmax.y && p0.z < vclipmax.z ) {		// accept condition
			if ( leaf_max== 0 ) {
				mVoxMin = p0; mVoxMax = p0; 
			} else {
				if ( p0.x < mVoxMin.x ) mVoxMin.x = p0.x;
				if ( p0.y < mVoxMin.y ) mVoxMin.y = p0.y;
				if ( p0.z < mVoxMin.z ) mVoxMin.z = p0.z;
				if ( p0.x > mVoxMax.x ) mVoxMax.x = p0.x;
				if ( p0.y > mVoxMax.y ) mVoxMax.y = p0.y;
				if ( p0.z > mVoxMax.z ) mVoxMax.z = p0.z;				
			}			
			leaf_max++;
		}
		vdbNext ( mOVDB, gridtype, isFloat );
	}	
	voffset = mVoxMin * -1;		// offset to positive space (hack)	
	
	// Activate Space
	if ( mbProfile ) PERF_PUSH ( "Activate" );	
	n = 0;
	verbosef ( "   Activating space.\n");

	vdbSkip ( mOVDB, leaf_start, gridtype, isFloat );
	for (leaf_max=0; vdbCheck ( mOVDB, gridtype, isFloat ) ; ) {
			
		// Read leaf position
		vdbOrigin ( mOVDB, orig, gridtype, isFloat );
		p0.Set ( orig.x(), orig.y(), orig.z() );
		p0 += voffset;

		if ( p0.x > vclipmin.x && p0.y > vclipmin.y && p0.z > vclipmin.z && p0.x < vclipmax.x && p0.y < vclipmax.y && p0.z < vclipmax.z ) {		// accept condition
			// only accept those in clip volume
			bool bnew = false;
			leaf = ActivateSpace ( mRoot, p0, bnew );
			leaf_ptr.push_back ( leaf );				
			leaf_pos.push_back ( p0 );
			if ( leaf_max==0 ) { 
				mVoxMin = p0; mVoxMax = p0; 
				verbosef ( "   First leaf: %d  (%f %f %f)\n", leaf_start+n, p0.x, p0.y, p0.z );
			}
			leaf_max++;				
		}
		vdbNext ( mOVDB, gridtype, isFloat );
		n++;
	}	

	// Finish Topology
	FinishTopology ();
	
	if ( mbProfile ) PERF_POP ();		// Activate

	// Resize Atlas
	verbosef ( "   Create Atlas. Free before: %6.2f MB\n", cudaGetFreeMem() );
	if ( mbProfile ) PERF_PUSH ( "Atlas" );		
	DestroyChannels ();
	AddChannel ( 0, T_FLOAT, mApron );
	UpdateAtlas ();
	if ( mbProfile ) PERF_POP ();
	verbosef ( "   Create Atlas. Free after:  %6.2f MB, # Leaf: %d\n", cudaGetFreeMem(), leaf_max );

	// Resize temp 3D texture to match leaf res
	int res0 = getRes ( 0 );	
	Vector3DI vres0 ( res0, res0, res0 );		// leaf resolution
	Vector3DF vrange0 = getRange(0);
	
	Volume3D vtemp ( mScene ) ;
	vtemp.Resize ( T_FLOAT, vres0, 0x0, false );

	vclipmin = getScene()->mVClipMin;
	vclipmax = getScene()->mVClipMax;	

	// Read brick data
	if ( mbProfile ) PERF_PUSH ( "Read bricks" );	

	// Advance to starting leaf		
	vdbSkip ( mOVDB, leaf_start, gridtype, isFloat );

	float* src;
	float* src2 = (float*) malloc ( res0*res0*res0*sizeof(float) );		// velocity field
	float mValMin, mValMax;
	mValMin = 1.0E35; mValMax = -1.0E35; 

	// Fill atlas from leaf data
	int percl = 0, perc = 0;
	verbosef ( "   Loading bricks.\n");
	for (leaf_cnt=0; vdbCheck ( mOVDB, gridtype, isFloat ); ) {

		// read leaf position
		vdbOrigin ( mOVDB, orig, gridtype, isFloat );		
		p0.Set ( orig.x(), orig.y(), orig.z() );
		p0 += voffset;

		if ( p0.x > vclipmin.x && p0.y > vclipmin.y && p0.z > vclipmin.z && p0.x < vclipmax.x && p0.y < vclipmax.y && p0.z < vclipmax.z ) {		// accept condition			
			
			// get leaf	
			if ( gridtype==0 ) {
				if ( isFloat ) {
					mOVDB->buf3U = (*mOVDB->iter543F).buffer();				
					src = mOVDB->buf3U.getData();					
				} else {
					mOVDB->buf3VU = (*mOVDB->iter543VF).buffer();
					src = ConvertToScalar ( res0*res0*res0, (float*) mOVDB->buf3VU.getData(), src2, mValMin, mValMax );				
				}			
			} else {
				if ( isFloat ) {
					mOVDB->buf4U = (*mOVDB->iter34F).buffer();				
					src = mOVDB->buf4U.getData();
				} else {
					mOVDB->buf4VU = (*mOVDB->iter34VF).buffer();
					src = ConvertToScalar ( res0*res0*res0, (float*) mOVDB->buf4VU.getData(), src2, mValMin, mValMax );				
				}
			}			
			// Copy data from CPU into temporary 3D texture
			vtemp.SetDomain ( leaf_pos[leaf_cnt], leaf_pos[leaf_cnt] + vrange0 );				
			vtemp.CommitFromCPU ( src );

			// Copy from 3D texture into Atlas brick
			Node* node = getNode ( leaf_ptr[leaf_cnt] );
			mPool->AtlasCopyTexZYX ( 0, node->mValue , vtemp.getPtr() );			
			
			// Progress percent
			leaf_cnt++; perc = int(leaf_cnt*100 / leaf_max);
			if ( perc != percl ) { verbosef ( "%d%%%% ", perc ); percl = perc; }
		}
		vdbNext ( mOVDB, gridtype, isFloat );
	}

	if ( mbProfile ) PERF_POP ();
	verbosef ( "    Value Range: %f %f\n", mValMin, mValMax );

	UpdateApron ();

	free ( src2 );

	// vdbfile->close ();
	// delete vdbfile;
	// delete mOVDB;
	// mOVDB = 0x0; 

	return true;

#else

	gprintf ( "ERROR: Unable to load .vdb file. OpenVDB library not linked.\n");
	return false;

#endif

}

#ifdef BUILD_OPENVDB

	typedef openvdb::tree::RootNode<openvdb::tree::InternalNode<openvdb::tree::InternalNode<openvdb::tree::InternalNode<openvdb::tree::LeafNode<float,4>,3>,3>,3>>  TreeConfig34;
	typedef openvdb::tree::Tree< TreeConfig34 > FloatTree34; 
	typedef openvdb::Grid< FloatTree34 >		FloatGrid34;
	typedef FloatTree34::RootNodeType			Root34; 
	typedef FloatTree34::LeafNodeType			Leaf34; 
	typedef Leaf34::ValueType					Value34;

	struct Activator {
		static inline void op(const FloatGrid34::ValueAllIter& iter) {
			if ( iter.getValue() != 0.0 )
			  iter.setActiveState ( true );
		}
	};

#endif

// Save OpenVDB file
void VolumeGVDB::SaveVDB ( std::string fname )
{
	// Save VDB file
	Vector3DI pos;

#ifdef BUILD_OPENVDB

	// create a vdb tree
	FloatTree34* tptr = new FloatTree34 ( 0.0 );
	FloatTree34::Ptr tree ( tptr );
	Leaf34* leaf;
	Value34* leafbuf;	
	int res = getRes(0);
	int sz = getVoxCnt(0) * sizeof(float);			// data per leaf = (res^3) floats
	
	DataPtr p;
	mPool->CreateMemLinear ( p, 0x0, 1, sz, true );	

	int leafcnt = mPool->getPoolCnt(0,0);			// leaf count
	Node* node;
	for (int n=0; n < leafcnt; n++ ) {
		node = getNode ( 0, 0, n );
		pos = node->mPos;
		leaf = tree->touchLeaf ( Coord(pos.x, pos.y, pos.z) );
		leaf->setValuesOff ();		
		leafbuf = leaf->buffer().getData();	
	
		mPool->AtlasRetrieveTexXYZ ( 0, node->mValue, p );

		memcpy ( leafbuf, p.cpu, sz );			// set leaf voxels
	}			
	verbosef( "  Leaf count: %d\n", tree->leafCount() );

	// create a vdb grid
	if ( mbProfile ) PERF_PUSH ( "Creating grid" );	
	mOVDB->grid34F = FloatGrid34::create ( tree );
	mOVDB->grid34F->setGridClass (openvdb::GRID_FOG_VOLUME );
	mOVDB->grid34F->setName ( "density" );	
	mOVDB->grid34F->setTransform ( openvdb::math::Transform::createLinearTransform(1.0) );	
	mOVDB->grid34F->addStatsMetadata ();
	openvdb::tools::foreach ( mOVDB->grid34F->beginValueAll(), Activator ::op);
	verbosef( "  Leaf count: %d\n", mOVDB->grid34F->tree().leafCount() );
	if ( mbProfile ) PERF_POP ();

	if ( mbProfile ) PERF_PUSH ( "Writing grids" );	
	openvdb::io::File* vdbfile = new openvdb::io::File ( fname );
	vdbfile->setGridStatsMetadataEnabled ( true );
	vdbfile->setCompression ( openvdb::io::COMPRESS_NONE );
	openvdb::GridPtrVec grids;
	   
	grids.push_back ( mOVDB->grid34F );
	vdbfile->write ( grids );	
	vdbfile->close ();
	if ( mbProfile ) PERF_POP ();
#endif

}

// Add a search path for assets
void VolumeGVDB::AddPath ( std::string path )
{
	mScene->AddPath ( path );
}
bool VolumeGVDB::FindFile ( std::string fname, char* path )
{
	return mScene->FindFile ( fname, path );
}

// Initialize GVDB
// - Creates a scene object and a memory pool allocator
// - Default transfer function and settings
void VolumeGVDB::Initialize ()
{
	mScene = new Scene;				// Scene object

	mScene->SetCamera ( new Camera3D );		// Default camera
	mScene->SetLight ( 0, new Light );		// Default light
	mScene->SetVolumeRange ( 0.1, 0, 1 );	// Default transfer range
	mScene->LinearTransferFunc ( 0, 1, Vector4DF(0,0,0,0), Vector4DF(1,1,1,0.1) );		// Default transfer function
	
	CommitTransferFunc ();			// Commit transfer func to GPU

	mPool = new Allocator;			// Allocator object

	SetChannelDefault ( 8, 8, 8 );
}

// Configure VDB tree (5-level)
void VolumeGVDB::Configure ( int q4, int q3, int q2, int q1, int q0 )
{
	int r[5], n[5];
	r[0] = q0; r[1] = q1; r[2] = q2; r[3] = q3; r[4] = q4; 

	n[0] = 4;				// leaf max
	int cnt = 4;
	n[1] = cnt;		cnt >>= 1;
	n[2] = cnt;		cnt >>= 1;
	n[3] = cnt;		cnt >>= 1;
	n[4] = cnt;
	
	Configure ( 5, r, n );	
}

// Configure VDB tree (N-level)
void VolumeGVDB::Configure ( int levs, int* r, int* numcnt )
{
	if ( mPool == 0 || mScene == 0 ) {
		gprintf ( "ERROR: Initialize not called.\n" );
		gerror ();
	}

	// This vector defines the VDB configuration
	int* maxcnt = (int*) malloc ( levs * sizeof(int) );
	for (int n=0; n < levs; n++) {
		mLogDim[n] = (r[n]==0) ? 1 : r[n];
		maxcnt[n] = (numcnt[n]==0) ? 1 : numcnt[n];
	}

	mClrDim[0] = Vector3DF(0,0,1);		// blue
	mClrDim[1] = Vector3DF(0,1,0);		// green
	mClrDim[2] = Vector3DF(1,0,0);		// red
	mClrDim[3] = Vector3DF(1,1,0);		// yellow
	mClrDim[4] = Vector3DF(1,0,1);		// purple
	mClrDim[5] = Vector3DF(0,1,1);		// aqua
	mClrDim[6] = Vector3DF(1,0.5,0);	// orange
	mClrDim[7] = Vector3DF(0,0.5,1);	// green-blue
	mClrDim[8] = Vector3DF(0.7,0.7,0.7);  // grey

	// Initialize memory pools
	int hdr = sizeof(Node);

	mPool->PoolReleaseAll();
	 
	// node & mask list
	mPool->PoolCreate ( 0, 0, hdr,					maxcnt[0], true );			
	for (int n=1; n < levs; n++ ) 
		mPool->PoolCreate ( 0, n, hdr+getMaskSize(n), maxcnt[n], true );

	// child lists	
	mPool->PoolCreate ( 1, 0, 0, 0, true );								
	for (int n=1; n < levs; n++ ) 	
		mPool->PoolCreate ( 1, n, sizeof(uint64)*getVoxCnt(n), maxcnt[n], true );

	mVoxResMax.Set ( 0, 0, 0 );

	// Clear tree and create default root
	Clear ();
}

// Add a data channel (voxel attribute)
void VolumeGVDB::AddChannel ( uchar chan, int dt, int apron, Vector3DI axiscnt )
{
	if (axiscnt.x==0 && axiscnt.y==0 && axiscnt.z==0) {
		if ( chan == 0 ) 	axiscnt = mDefaultAxiscnt;
		else				axiscnt = mPool->getAtlas(0).subdim;
	}
	mApron = apron;

	mPool->AtlasCreate ( chan, dt, getRes3DI(0), axiscnt, apron, sizeof(AtlasNode), false, mbUseGLAtlas );

}

// Fill data channel
void VolumeGVDB::FillChannel ( uchar chan, Vector4DF val )
{
	switch ( mPool->getAtlas(chan).type ) {
	case T_FLOAT:	Compute ( FUNC_FILL_F, chan, 1, val, false );	break;	
	case T_UCHAR:	Compute ( FUNC_FILL_C, chan, 1, val, false );	break;
	case T_UCHAR4:	Compute ( FUNC_FILL_C4, chan, 1, val, false );	break;	
	};
}

// Destroy all channels
void VolumeGVDB::DestroyChannels ()
{
	mPool->AtlasReleaseAll ();		
	SetColorChannel ( -1 );
}

// Clear GVDB without changing config
void VolumeGVDB::Clear ()
{
	// Empty VDB data (keep pools)
	mPool->PoolEmptyAll ();		// does not free pool mem
	mRoot = ID_UNDEFL;

	// Empty atlas & atlas map
	mPool->AtlasEmptyAll ();	// does not free atlas
	
	mPnt.Set ( 0, 0, 0 );
}

// Allocate a new VDB node
slong VolumeGVDB::AllocateNode ( int lev )
{
	return mPool->PoolAlloc ( 0, lev, true );	
}

// Setup a VDB node
void VolumeGVDB::SetupNode ( slong nodeid, int lev, Vector3DF pos )
{
	Node* node = getNode ( nodeid );
	node->mLev = lev;	
	node->mPos = pos;	
	node->mChildList = ID_UNDEFL;
	node->mParent = ID_UNDEFL;
	node->mValue = Vector3DI(-1,-1,-1);
	if ( lev > 0 ) node->clearMask ();
}

// Clear atlas mapping
void VolumeGVDB::ClearMapping ()
{ 
	// This function ensures that the atlas mapping, for unused bricks in the atlas,
	// maps to an undefined value which is checked by kernels.

	DataPtr a = mPool->getAtlas ( 0 );		// atlas
	Vector3DI axiscnt = a.subdim;			// number of leaves along atlas axis
	AtlasNode* an;	
	Vector3DI b;							// 3D brick index			0 < b < atlas_cnt
	Vector3DI brickpos;						// pos of brick in atlas	0 < brickpos < atlas res

	for (b.z=0; b.z < axiscnt.z; b.z++ ) {
		for (b.y=0; b.y < axiscnt.y; b.y++ ) {
			for (b.x=0; b.x < axiscnt.x; b.x++ ) {
				brickpos = b * int(a.stride + a.apron*2) + a.apron; 
				an = (AtlasNode*) mPool->getAtlasNode ( 0, brickpos );
				an->mLeafNode = ID_UNDEFL;
				an->mPos.Set ( ID_UNDEFL, ID_UNDEFL, ID_UNDEFL );
			}
		}
	}
}

// Assign an atlas mapping
void VolumeGVDB::AssignMapping ( Vector3DI brickpos, Vector3DI pos, int leafid )
{
	AtlasNode* an = (AtlasNode*) mPool->getAtlasNode ( 0, brickpos );
	an->mPos = pos;
	an->mLeafNode = leafid;
}

// Mandelbulb! (3D fractal)
float Mandelbulb ( Vector3DF s )
{
	const int iter = 24;
	const float bail = 8.0;
	float pwr = 8.0;
	float theta, phi, zr;

	Vector3DF z = s;
	float dr = 1.0;
	float r = 0.0;
	
	for (int i = 0; i < iter; i++ ) {		
		r = z.Length ();
		if ( r > bail ) break;

		theta = asin(z.z/r) * pwr;
		phi = atan2(z.y, z.x) * pwr;
		zr = pow( (double) r, (float) pwr-1.0 );
		dr = zr*pwr*dr + 1.0;
		zr *= r;		

		z = Vector3DF( cos(theta)*cos(phi), cos(theta)*sin(phi), sin(theta) );
		z *= zr;
		z += s;
	}
	return -0.5*log(r)*r / dr;
}

// Update the atlas
// - Resize atlas if needed
// - Assign new nodes to atlas
// - Update atlas mapping and pools
void VolumeGVDB::UpdateAtlas ()
{
	if ( mbProfile ) PERF_PUSH ( "Update Atlas" );

	Vector3DI brickpos;
	Node* node;
	int leafcnt = mPool->getPoolCnt(0,0);

	// Resize atlas
	int amax = mPool->getAtlas(0).max;
	if ( leafcnt > amax || (leafcnt < amax && ++mAtlasResize.x==mAtlasResize.y) ) {
		mAtlasResize.x = 0;
		if ( mbProfile ) PERF_PUSH ( "Resize Atlas" );
		for (int n=0; n < mPool->getNumAtlas(); n++ )
			mPool->AtlasResize ( n, leafcnt );
		if ( mbProfile ) PERF_POP ();
	}

	// Assign new nodes to atlas
	if ( mbProfile ) PERF_PUSH ( "Assign Atlas" );
	for (int n=0; n < leafcnt; n++ ) {
		node = getNode ( 0, 0, n );
		if ( node->mValue.x == -1 ) {					// node not yet assigned to atlas			
			if ( mPool->AtlasAlloc ( 0, brickpos ) )	// assign to atlas brick
				node->mValue = brickpos;
		}
	}
	if ( mbProfile ) PERF_POP ();

	// Reallocate Atlas Map (if needed)	
	// -- Note: subdim = full size of atlas including unused bricks
	mPool->AllocateAtlasMap ( sizeof(AtlasNode), mPool->getAtlas(0).subdim );

	// ensure mapping for unused bricks
	ClearMapping ();

	// Build Atlas Mapping	
	if ( mbProfile ) PERF_PUSH ( "Atlas Mapping" );
	int brickcnt = mPool->getAtlas(0).num;
	int brickres = mPool->getAtlasBrickres(0);
	Vector3DI atlasres = mPool->getAtlasRes(0);
	Vector3DI atlasmax = atlasres - brickres + mPool->getAtlas(0).apron; 
	for (int n=0; n < brickcnt; n++ ) {
		Node* node = getNode ( 0, 0, n );
		if ( node->mValue.x == -1 ) continue;
		if ( node->mValue.x > atlasmax.x || node->mValue.y > atlasmax.y || node->mValue.z > atlasmax.z ) {
			gprintf ( "ERROR: Node value exceeds atlas res. node: %d, val: %d %d %d, atlas: %d %d %d\n", n, node->mValue.x, node->mValue.y, node->mValue.z, atlasres.x, atlasres.y, atlasres.z );
			gerror ();
		}
		AssignMapping ( node->mValue, node->mPos, n );
	}
	if ( mbProfile ) PERF_POP ();

	// Commit to GPU
	if ( mbProfile ) PERF_PUSH ( "Commit Atlas Map" );
	mPool->PoolCommitAtlasMap ();		// Commit Atlas Map (HtoD)
	
	mPool->PoolCommit ( 0, 0 );			// Commit brick nodes *with new values* (HtoD)

	SetupAtlasAccess ();				// Setup CUDA Bindless pointers to Atlas (HtoD)
	
	if ( mbProfile ) PERF_POP ();

	if ( mbProfile ) PERF_POP ();

	if ( mbVerbose ) Measure ( true );
}

// Add a child to a node
slong VolumeGVDB::AddChildNode(slong nodeid, Vector3DF ppos, int plev, uint32 i, Vector3DI pos)
{
	// allocate new internal or leaf node
	slong child = AllocateNode(plev - 1);

	// determine absolute node pos at correct level	
	Vector3DI range = getRange(plev - 1);
	Vector3DI p = getPosFromBit(plev, i);
	p *= range;
	p += ppos;

	// position child
	SetupNode(child, plev - 1, p);

	// add to list of own children
	return InsertChild(nodeid, child, i);
}

// Reparent the VDB tree
slong VolumeGVDB::Reparent(int lev, slong prevroot_id, Vector3DI pos, bool& bNew)
{
	Node* prevroot = getNode(prevroot_id);
	Vector3DI prevroot_pos = prevroot->mPos;
	Vector3DI p, pos1, pos2, range;
	bool bCover = false;

	// find a level node which covers both the child (former root)
	// and the new position
	while (!bCover && lev < MAXLEV)
	{
		lev++;
		pos1 = GetCoveringNode(lev, pos, range);
		pos2 = GetCoveringNode(lev, prevroot_pos, range);
		bCover = (pos1.x == pos2.x && pos1.y == pos2.y && pos1.z == pos2.z);
	}
	if (lev >= MAXLEV) {
		return ID_UNDEFL;						// level limit exceeded, return existing root
	}
	// create new covering root
	uint64 newroot_id = AllocateNode(lev);
	if (newroot_id == ID_UNDEFL)
		return ID_UNDEFL;

	SetupNode(newroot_id, lev, pos1);

	// insert prevroot into new root	
	bool bn = false;				// prev path does not create new leaf, so ignore bnew
	ActivateSpace(newroot_id, prevroot_pos, bn, prevroot_id);	// use stopnode to connect paths

	// insert new pos into new root
	uint64 leaf_id = ActivateSpace(newroot_id, pos, bNew );		// new path may create new leaf (return bNew)

	// update root
	mRoot = newroot_id;

	return getNode(leaf_id)->mParent;			// return parent of the new pos
}

// Activate region of space at 3D position
slong VolumeGVDB::ActivateSpace ( Vector3DF pos )
{
	pos /= mVoxsize;
	
	Vector3DI brickpos;
	bool bnew = false;
	slong node_id = ActivateSpace ( mRoot, pos, bnew, ID_UNDEFL, 0 );
	if ( node_id == ID_UNDEFL ) return ID_UNDEFL;
	if ( !bnew ) return node_id; // exiting node. return
	return ID_UNDEFL;
}

// Activate region of space at 3D position down to a given level
slong VolumeGVDB::ActivateSpaceAtLevel ( int lev, Vector3DF pos )
{
	pos /= mVoxsize;	
	Vector3DI brickpos;
	bool bnew = false;
	slong node_id = ActivateSpace ( mRoot, pos, bnew, ID_UNDEFL, lev );		// specify level to stop
	if ( node_id == ID_UNDEFL ) return ID_UNDEFL;
	if ( !bnew ) return node_id; // exiting node. return
	return ID_UNDEFL;
}

// Activate space
// - 'nodeid'    Starting sub-tree for activation
// - 'pos'       Index-space position to activate
// - 'bNew'      Returns true if the brick was added
// - 'stopnode'  Specific node to stop activation
// - 'stoplev'   Specific level to stop activation
slong VolumeGVDB::ActivateSpace ( slong nodeid, Vector3DI pos, bool& bNew, slong stopnode, int stoplev  )
{

	Vector3DI p;
	Vector3DI range;
	slong childid;
	uint32 b;
	
	if ( mRoot == ID_UNDEFL && nodeid == mRoot ) {
		// Create new root 
		p = GetCoveringNode ( stoplev, pos, range );
		mRoot = AllocateNode ( stoplev );
		SetupNode ( mRoot, stoplev, p );
		nodeid = mRoot;
	}

	// Activate recursively to leaf
	Node* curr = getNode ( nodeid );
	

	if ( getPosInNode ( nodeid, pos, b ) ) {

		// check stop node
		if (stopnode != ID_UNDEFL) {
			Node* stopn = getNode(stopnode);
			Vector3DI sp = stopn->mPos;
			if (pos.x == sp.x && pos.y == sp.y && pos.z == sp.z && !curr->isOn(b) && curr->mLev == stopn->mLev + 1) {
				// if same position as stopnode, bit is not on, 
				return InsertChild( nodeid, stopnode, b);
			}
		}
		// check stop level
		if ( curr->mLev == stoplev ) return nodeid;

		// point is inside this node, add children
		if ( !curr->isOn ( b ) ) {		// not on yet - create new child			
			childid = AddChildNode ( nodeid, curr->mPos, curr->mLev, b, pos );
			if ( curr->mLev==1 ) bNew = true;
		} else {						// already on - insert into existing child
			uint32 ndx = curr->countOn ( b );
			childid = getChildNode ( nodeid, ndx );			
		}	
		if ( isLeaf ( childid ) ) return childid;
		return ActivateSpace ( childid, pos, bNew, stopnode, stoplev );
	} else {
		// point is outside this node		
		uint64 parent = curr->mParent;
		if ( parent == ID_UNDEFL ) {
			parent = Reparent ( curr->mLev, nodeid, pos, bNew );	// make make new lev0 nodes
			if ( parent == ID_UNDEFL ) return ID_UNDEFL;
		}		
		// active point inside the (possibly new) parent 
		return ActivateSpace ( parent, pos, bNew, stopnode, stoplev );
	}	
}

// Get bit position in node given 3D local brick-space index
bool VolumeGVDB::getPosInNode ( slong curr_id, Vector3DI pos, uint32& bit )
{
	Node* curr = getNode ( curr_id );
	Vector3DI res = getRes3DI ( curr->mLev );
	Vector3DI p = pos; p -= curr->mPos; 
	Vector3DI range = getRange ( curr->mLev );

	if (p.x >= 0 && p.y >=0 && p.z >=0 && p.x < range.x && p.y < range.y && p.z < range.z ) {
		// point is inside this node
		p *= res;
		p /= range;
		bit = getBitPos ( curr->mLev, p );
		return true;
	} else {
		bit = 0;
		return false;
	} 
}

const char* binaryStr (uint64 x)
{
	static char b[65];
	int p = 0;
    for (uint64 z = uint64(1) << 63; z > 0; z >>= 1) {
        b[p++] = ((x & z) == z) ? '1' : '0';
    }
	b[64] = '\0';
    return b;
}

// Debug a node
void VolumeGVDB::DebugNode ( slong nodeid )
{
	Node* curr = getNode ( nodeid );

	int b = -1;	
	//int grp = ElemGrp ( nodeid );
	//int lev = ElemLev ( nodeid );
	int ndx = ElemNdx ( nodeid );
	
	if ( curr->mLev < 4 && curr->mParent != ID_UNDEFL ) {
		Vector3DF range = getRange ( curr->mLev+1 );
		Vector3DI p = curr->mPos; p -= getNode(curr->mParent)->mPos; p *= getRes3DI ( curr->mLev+1 ); p /= range;
		b = getBitPos ( curr->mLev+1, p );
	}
	if ( curr->mLev > 0 ) {		
		gprintf ( "%*s L%d #%d, Bit: %d, Pos: %d %d %d, Mask: %s\n", (5-curr->mLev)*2, "", (int) curr->mLev, ndx, b, curr->mPos.x, curr->mPos.y, curr->mPos.z, binaryStr( *curr->getMask() ) );
		for (int n=0; n < curr->getNumChild(); n++ ) {
			slong childid = getChildNode ( nodeid, n );
			DebugNode ( childid );
		}
	} else {
		gprintf ( "%*s L%d #%d, Bit: %d, Pos: %d %d %d, Atlas: %d %d %d\n", (5-curr->mLev)*2, "", (int) curr->mLev, ndx, b, curr->mPos.x, curr->mPos.y, curr->mPos.z, curr->mValue.x, curr->mValue.y, curr->mValue.z);
	}
}

// Get the index-space position of the corner of a node covering the given 'pos'
Vector3DI VolumeGVDB::GetCoveringNode ( int lev, Vector3DI pos, Vector3DI& range )
{
	Vector3DI nodepos;
	range = getRange( lev );			// range of node

	if ( lev == MAXLEV-1 ) {		
		nodepos = Vector3DI(0,0,0);
		//nodepos = Vector3DI( -range.x/2, -range.y/2, -range.z/2 );  // highest level root must cover origin
	} else {
		nodepos = pos;
		nodepos /= range;	
		nodepos = Vector3DI(nodepos) * range;	// determine nearest next-level block
		if ( pos.x < nodepos.x ) nodepos.x -= range.x;
		if ( pos.y < nodepos.y ) nodepos.y -= range.y;
		if ( pos.z < nodepos.z ) nodepos.z -= range.z;
	}	
	return nodepos;
}

// Insert child into a child list
slong VolumeGVDB::InsertChild ( slong nodeid, slong childid, uint32 i )
{
	Node* curr = getNode ( nodeid );
	Node* child = getNode ( childid );
	child->mParent = nodeid;	// set parent of child

	// determine child bit position	
	assert ( ! curr->isOn ( i ) );			// check if child already exists
	uint64 p = curr->countOn ( i );
	uint64 cnum = curr->getNumChild();		// existing children count
	curr->setOn ( i );

	// add child list if doesn't exist
	if ( curr->mChildList == ID_UNDEFL ) {
		curr->mChildList = mPool->PoolAlloc ( 1, curr->mLev, true );		
	}
	uint64 max_child = mPool->getPoolWidth( 1, curr->mLev ) / sizeof(uint64);
	if ( cnum + 1 > max_child ) {
		gprintf ( "ERROR: Number of children exceed max of %d (lev %d)\n", max_child, curr->mLev );
		gerror ();
	}
	
	// insert into child list
	uint64* clist = mPool->PoolData64 ( curr->mChildList );
	if ( p < cnum ) {
		//for (uint64* j = clist + cnum; j > clist + p; j-- ) 
		//	*j = *(j-1);
		memmove ( clist + p+1, clist + p, (cnum-p)*sizeof(uint64) );
		*(clist + p) = childid;
	} else {
		*(clist + cnum) = childid;		
	}
	int cnt = curr->countOn();			// count children again

	return childid;
}

// Get child node at bit position
slong VolumeGVDB::getChildNode ( slong nodeid, uint b )
{
	Node* curr = getNode ( nodeid );
	uint64* clist = mPool->PoolData64 ( curr->mChildList );
	return *(clist + b);
}

// Get child a given local 3D brick position
slong VolumeGVDB::getChildOffset ( slong nodeid, slong childid, Vector3DI& pos )
{
	Node* curr = getNode ( nodeid );
	Node* child = getNode ( childid );
	pos = child->mPos;
	pos -= curr->mPos;
	pos *= getRes3DI ( curr->mLev );
	pos /= getRange ( curr->mLev );
	return getBitPos ( curr->mLev, pos );
}

// Set voxel size
void VolumeGVDB::SetVoxelSize ( float vx, float vy, float vz )
{
	mVoxsize.Set ( vx, vy, vz );
}


void VolumeGVDB::writeCube (  FILE* fp, unsigned char vpix[], slong& numfaces, int gv[], int*& vgToVert, slong vbase )
{
	// find absolute vert indices for this cube
	// * map from local grid index to local vertex id, then add base index for this leaf	
	long v[8];
	vbase++;	// * .obj file indices are base 1;
	v[0] = vbase + vgToVert[ gv[0] ];
	v[1] = vbase + vgToVert[ gv[1] ];
	v[2] = vbase + vgToVert[ gv[2] ];
	v[3] = vbase + vgToVert[ gv[3] ];
	v[4] = vbase + vgToVert[ gv[4] ];
	v[5] = vbase + vgToVert[ gv[5] ];
	v[6] = vbase + vgToVert[ gv[6] ];
	v[7] = vbase + vgToVert[ gv[7] ];

	if ( vpix[1] == 0 )	{fprintf(fp, "f %d//0 %d//0 %d//0 %d//0\n", v[1], v[2], v[6], v[5] );  numfaces++; }	// x+	
	if ( vpix[2] == 0 ) {fprintf(fp, "f %d//1 %d//1 %d//1 %d//1\n", v[0], v[3], v[7], v[4] );  numfaces++; }	// x-
	if ( vpix[3] == 0 ) {fprintf(fp, "f %d//2 %d//2 %d//2 %d//2\n", v[2], v[3], v[7], v[6] );  numfaces++; }	// y+
	if ( vpix[4] == 0 ) {fprintf(fp, "f %d//3 %d//3 %d//3 %d//3\n", v[1], v[0], v[4], v[5] );  numfaces++; }	// y- 
	if ( vpix[5] == 0 )	{fprintf(fp, "f %d//4 %d//4 %d//4 %d//4\n", v[4], v[5], v[6], v[7] );  numfaces++; }	// z+
	if ( vpix[6] == 0 ) {fprintf(fp, "f %d//5 %d//5 %d//5 %d//5\n", v[0], v[1], v[2], v[3] );  numfaces++; }	// z-	
}

void VolumeGVDB::enableVerts ( int*& vgToVert, std::vector<Vector3DF>& verts, Vector3DF vm, int gv[] )
{
	if ( vgToVert[ gv[0] ] == -1) {	vgToVert[ gv[0] ] = (int) verts.size ();	verts.push_back ( vm );		}
	if ( vgToVert[ gv[1] ] == -1) {	vgToVert[ gv[1] ] = (int) verts.size ();	verts.push_back ( vm+Vector3DF(1,0,0) ); 	}
	if ( vgToVert[ gv[2] ] == -1) {	vgToVert[ gv[2] ] = (int) verts.size ();	verts.push_back ( vm+Vector3DF(1,1,0) ); 	}
	if ( vgToVert[ gv[3] ] == -1) {	vgToVert[ gv[3] ] = (int) verts.size ();	verts.push_back ( vm+Vector3DF(0,1,0) ); 	}
	if ( vgToVert[ gv[4] ] == -1) {	vgToVert[ gv[4] ] = (int) verts.size ();	verts.push_back ( vm+Vector3DF(0,0,1) ); 	}
	if ( vgToVert[ gv[5] ] == -1) {	vgToVert[ gv[5] ] = (int) verts.size ();	verts.push_back ( vm+Vector3DF(1,0,1) ); 	}
	if ( vgToVert[ gv[6] ] == -1) {	vgToVert[ gv[6] ] = (int) verts.size ();	verts.push_back ( vm+Vector3DF(1,1,1) ); 	}
	if ( vgToVert[ gv[7] ] == -1) {	vgToVert[ gv[7] ] = (int) verts.size ();	verts.push_back ( vm+Vector3DF(0,1,1) ); 	}
}
void VolumeGVDB::setupVerts ( int gv[], Vector3DI g, int r1, int r2 )
{
	int g0 = (g.z*r1 + g.y)*r1 + g.x ;		// local index of first vertex
	gv[0] = g0;
	gv[1] = g0 + 1;
	gv[2] = g0 + r1+1;
	gv[3] = g0 + r1;
	gv[4] = g0 + r2;
	gv[5] = g0 + r2+1;
	gv[6] = g0 + r2+r1+1;
	gv[7] = g0 + r2+r1;
}

void VolumeGVDB::WriteObj ( char* fname )
{
	// Allocate CPU memory for leaf
	/*int res0 = getRes ( 0 );	
	Vector3DI vres0 ( res0, res0, res0 );	
	Vector3DF vrange0 = getRange(0);
	unsigned char* vdat = (unsigned char*) malloc ( res0*res0*res0 );
	unsigned char vpix[8];
	Vector3DF vmin;

	FILE* fp = fopen( fname, "w");

	fprintf(fp, "# Wavefront OBJ format\n\n");  
			
	// Output 6 normals (shared by all cubes)
	fprintf(fp, "vn  1  0  0\n" );	// x+
	fprintf(fp, "vn -1  0  0\n" );	// x-	
	fprintf(fp, "vn  0  1  0\n" );	// y+
	fprintf(fp, "vn  0 -1  0\n" );	// y-
	fprintf(fp, "vn  0  0  1\n" );	// z+
	fprintf(fp, "vn  0  0 -1\n" );	// z-

	slong numverts = 0, numfaces = 0, numvox = 0;

	// Create map from grid points to vertices
	Vector3DI g;
	int gv[8];
	int r1 = (res0+1);
	int r2 = (res0+1)*(res0+1);
	int r3 = r2*r1;
	int* vgToVert = (int*) malloc ( r3 * sizeof(int) );	
	std::vector< Vector3DF >  verts;	

	// Loop over every leaf in GVDB	
	LeafNode* leaf = 0x0;
	Vol3D* vox = 0x0;
	gprintf ( "Write OBJ file: %s\n", fname );
	for (int ln=0; ln < mLeaves.size(); ln++ ) {
		gprintf ( "  Writing: %d of %d (%f%%)\n", ln, mLeaves.size(), float(ln*100.f)/mLeaves.size() );

		// Get leaf
		leaf = mLeaves[ ln ];	
		vox = leaf->getVox( 0 );

		// Transfer leaf data to CPU
		vox->MemRetrieve ( vox->mData, vdat );

		// Find all active vertices in leaf		
		for (int n=0; n < r3; n++ ) vgToVert[n] = -1;
		verts.clear ();
		for (g.z=0; g.z < res0; g.z++)  {
			for (g.y=0; g.y < res0; g.y++) {
				for (g.x = 0; g.x < res0; g.x++) {
					vpix[0] = *(vdat + (g.z*res0 + g.y)*res0 + g.x );
					if ( vpix[0] > 0 ) {						
						vmin = vox->mVolMin;				// world coordinates of leaf
						vmin /= vrange0;					//  (reduce voxel to 1 unit size, for better rendering)
						vmin *= vres0; 
						vmin += g;						
						setupVerts ( gv, g, r1, r2 );						
						enableVerts ( vgToVert, verts, vmin, gv );
					}
				}
			}
		}

		// Write vertices
		long vbase = numverts;			// remember first vertex in this leaf 
		fprintf ( fp, "#---- leaf: %d %lld\n", ln, vbase );
		for (int v=0; v < verts.size(); v++ ) {			
			fprintf(fp, "v %d %d %d\n", (int) verts[v].x, (int) verts[v].y, (int) verts[v].z );
			numverts++;
		}

		// Write out faces from each voxel in leaf
		int goff = 0;
		for (g.z=0; g.z < res0; g.z++)  {
			for (g.y=0; g.y < res0; g.y++) {
				for (g.x = 0; g.x < res0; g.x++) {
					goff = (g.z*res0 + g.y)*res0 + g.x;
					vpix[0] = *(vdat + goff);					
					if ( vpix[0] > 0 ) {
						vpix[1] = (g.x==res0-1) ?	0 : *(vdat + goff + 1);			// x+;
						vpix[2] = (g.x==0) ?		0 : *(vdat + goff - 1);			// x-;
						vpix[3] = (g.y==res0-1) ?	0 : *(vdat + goff + res0);		// y+;
						vpix[4] = (g.y==0) ?		0 : *(vdat + goff - res0);		// y-;
						vpix[5] = (g.z==res0-1) ?	0 : *(vdat + goff + res0*res0);	// z+;
						vpix[6] = (g.z==0) ?		0 : *(vdat + goff - res0*res0);	// z-;
						setupVerts ( gv, g, r1, r2 );
						writeCube ( fp, vpix, numfaces, gv, vgToVert, vbase );
						numvox++;					
					}
				}
			}
		}
	}
	
	fprintf(fp,     "# %ld voxels.\n", numvox ); 		
	fprintf(fp,     "# %ld vertices.\n", numverts); 		
	fprintf(fp,     "# %ld faces.\n", numfaces); 				
	
	verbosef( "Write OBJ complete: %s\n", fname );
	verbosef( "  %lld voxels.\n", numvox ); 		
	verbosef( "  %lld vertices.\n", numverts); 	
	verbosef( "  %lld faces.\n", numfaces); 	

	fclose(fp);	

	// free temporary memory
	free ( vdat );
	free ( vgToVert );*/
}

// Validate OpenGL
// - Set the current OpenGL context for GL operations
void VolumeGVDB::ValidateOpenGL ()
{
	#ifdef BUILD_OPENGL
		
		if ( !mbGlew ) {
			mbGlew = true;
			glewExperimental = GL_TRUE;
			glewInit ();
		}
#if defined(_WIN32)
		HGLRC ctx = wglGetCurrentContext ();
#else
		void* ctx = glXGetCurrentContext();
#endif
		if ( ctx == NULL ) {
			gprintf ( "ERROR: Cannot validate OpenGL. No context active.\n" );
			return;
		}
	#endif
}

// Start Raster GL
// - Enable the GL rasterization pipeline. 
// - Load GLSL shaders
void VolumeGVDB::StartRasterGL ()
{
	#ifdef BUILD_OPENGL
		ValidateOpenGL ();
		makeSimpleShader ( mScene, "simple.vert.glsl", "simple.frag.glsl");
		makeVoxelizeShader ( mScene, "voxelize.vert.glsl", "voxelize.frag.glsl", "voxelize.geom.glsl" );
	#endif
}

// Create OpenGL atlases (optional)
void VolumeGVDB::UseOpenGLAtlas ( bool tf )
{
	mbUseGLAtlas = tf;
	#ifdef BUILD_OPENGL
		if ( mbUseGLAtlas ) ValidateOpenGL();
	#endif
}

// Prepare 3D texture for intermediate sub-volume
void VolumeGVDB::PrepareV3D ( Vector3DI ires, uchar dtype )
{
	if ( mV3D == 0x0 ) {
		mV3D = new Volume3D ( mScene ) ;		
		mV3D->Resize ( dtype, ires, 0x0, true );	
		mV3D->PrepareRasterGL ( true );
	} else {		
		if ( !mV3D->hasSize( ires, dtype ) ) {
			mV3D->Resize ( dtype, ires, 0x0, true );	
			mV3D->PrepareRasterGL ( (ires.x==0) ? false : true );
		}
	}	
}

Vector3DI VolumeGVDB::getNearestAbsVox ( int lev, Vector3DF pnt )
{
	Vector3DF cover = getCover(lev); 	
	return pnt / cover;
}

// Compute index-space extents given world bounding box
Extents VolumeGVDB::ComputeExtents ( int lev, Vector3DF obj_min, Vector3DF obj_max )
{
	// Compute index extents given world extents
	Extents e;
	e.lev = lev;
	e.vmin = obj_min;									// world-space bounds
	e.vmax = obj_max;
	e.cover = getCover(lev-1);
	e.imin = e.vmin / e.cover;							// absolute index-space extents of children
	e.imax = (e.vmax / e.cover) - Vector3DI(1,1,1);
	e.ires = e.imax; e.ires -= e.imin; e.ires += Vector3DI(1, 1, 1);
	e.icnt = e.ires.x * e.ires.y * e.ires.z;
	return e;		
}

// Compute index-space extents given a node
Extents VolumeGVDB::ComputeExtents ( Node* node )
{
	// Compute index and world extent given node
	Extents e;
	e.lev = node->mLev;
	Vector3DI range = getRange(e.lev-1);	
	e.cover = getCover(e.lev-1);	
	e.imin = node->mPos / range;						// absolute index-space extents of children
	e.imax = node->mPos / range + getRes3DI(e.lev);	
	e.vmin = Vector3DF(e.imin) * e.cover;				// world-space bounds of children
	e.vmax = Vector3DF(e.imax) * e.cover;	
	e.ires = e.imax; e.ires -= e.imin;
	e.icnt = e.ires.x * e.ires.y * e.ires.z;
	return e;		
}

// Map an OpenGL VBO to auxiliary geometry
void VolumeGVDB::AuxGeometryMap ( Model* model, int vertaux, int elemaux )
{
	DataPtr* vaux = &mAux[vertaux];
	DataPtr* eaux = &mAux[elemaux];
	size_t vsize, esize;

    cudaCheck(cuGraphicsGLRegisterBuffer( &vaux->grsc, model->vertBufferID, CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY), "cuGraphicsGLRegisterBuffer", "AuxGeometryMap" );
    cudaCheck(cuGraphicsMapResources(1, &vaux->grsc, 0), "cudaGraphicsMapResources", "AuxGeometryMap" );
    cudaCheck(cuGraphicsResourceGetMappedPointer ( &vaux->gpu, &vsize, vaux->grsc ), "cuGraphicsSubResourceGetMappedArray", "AuxGeometryMap" );
    //cudaCheck(cuGraphicsUnmapResources(1, &vaux->grsc, 0), "cudaGraphicsUnmapRsrc (AuxGeom)");
	vaux->num = model->vertCount;
	vaux->stride = model->vertStride;
	vaux->size = vsize;

    cudaCheck(cuGraphicsGLRegisterBuffer( &eaux->grsc, model->elemBufferID, CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY), "cuGraphicsGLRegisterBuffer", "AuxGeometryMap" );
    cudaCheck(cuGraphicsMapResources(1, &eaux->grsc, 0), "cudaGraphicsMapResources", "AuxGeometryMap" );
    cudaCheck(cuGraphicsResourceGetMappedPointer ( &eaux->gpu, &esize, eaux->grsc ), "cuGraphicsSubResourceGetMappedArray", "AuxGeometryMap" );
    //cudaCheck(cuGraphicsUnmapResources(1, &eaux->grsc, 0), "cudaGraphicsUnmapRsrc (AuxGeom)");
	eaux->num = model->elemCount;
	eaux->stride = model->elemStride;
	eaux->size = esize;
}

// Unmap an OpenGL VBO
void VolumeGVDB::AuxGeometryUnmap ( Model* model, int vertaux, int elemaux )
{
	DataPtr* vaux = &mAux[vertaux];
	DataPtr* eaux = &mAux[elemaux];
	
    cudaCheck(cuGraphicsUnmapResources(1, &vaux->grsc, 0), "cudaGraphicsUnmapRsrc", "AuxGeometryUnmap" );
    cudaCheck(cuGraphicsUnmapResources(1, &eaux->grsc, 0), "cudaGraphicsUnmapRsrc", "AuxGeometryUnmap" );
}

// Activate a region of space
int VolumeGVDB::ActivateRegion ( Extents& e )
{
	Vector3DF pos;
	uint64 leaf;		
	int cnt = 0;
	
	for (int z=e.imin.z; z <= e.imax.z; z++ )
		for (int y=e.imin.y; y <= e.imax.y; y++ )
			for (int x=e.imin.x; x <= e.imax.x; x++ ) {
				pos.Set(x,y,z); pos *= e.cover;
				leaf = ActivateSpaceAtLevel ( e.lev-1, pos );
				cnt++;
			}

	return cnt;
}

// Activate a region of space from an auxiliary byte buffer
int VolumeGVDB::ActivateRegionFromAux ( Extents& e, int auxid, uchar dt, float vthresh )
{
	Vector3DF pos;
	uint64 leaf;	
	char* vdat = mAux[auxid].cpu;			// Get AUX data		
	int cnt = 0;
	
	switch (dt) {
	case T_UCHAR: {
		for (int z=e.imin.z; z <= e.imax.z; z++ )
			for (int y=e.imin.y; y <= e.imax.y; y++ )
				for (int x=e.imin.x; x <= e.imax.x; x++ ) {
					pos.Set(x,y,z); pos *= e.cover;					
					uchar vset = *(vdat + ((z-e.imin.z)*e.ires.y + (y-e.imin.y))*e.ires.x + (x-e.imin.x));
					if ( vset > vthresh ) { leaf = ActivateSpaceAtLevel ( e.lev-1, pos ); cnt++; }
				}
		} break;
	case T_FLOAT: {
		for (int z=e.imin.z; z <= e.imax.z; z++ )
			for (int y=e.imin.y; y <= e.imax.y; y++ )
				for (int x=e.imin.x; x <= e.imax.x; x++ ) {
					pos.Set(x,y,z); pos *= e.cover;					
					float vset = *( ((float*) vdat) + ((z-e.imin.z)*e.ires.y + (y-e.imin.y))*e.ires.x + (x-e.imin.x));
					if ( vset > vthresh) {
						leaf = ActivateSpaceAtLevel ( e.lev-1, pos ); cnt++; 
					}
				}
		} break;
	};
	return cnt;
}


// Check data returned from GPU
void VolumeGVDB::CheckData ( std::string msg, CUdeviceptr ptr, int dt, int stride, int cnt )
{
	char* dat = (char*) malloc ( stride*cnt );
	cudaCheck ( cuMemcpyDtoH ( dat, ptr, stride*cnt ), "cuMemcpyDtoH", "CheckData" );

	gprintf ( "%s\n", msg.c_str() );
	char* p = dat;
	for (int n=0; n < cnt; n++ ) {
		switch ( dt ) {
		case T_FLOAT:		gprintf ( "%d: %f\n", n, *(float*) dat );		break;
		case T_INT:			gprintf ( "%d: %d\n", n, *(int*) dat );			break;
		case T_FLOAT3:		
			float3 v = *(float3*) dat;
			gprintf ( "%d: %f %f %f\n", n, v.x, v.y, v.z );			break;
		};
		dat += stride;
	}

}

// Voxelize a node from polygons
int VolumeGVDB::VoxelizeNode ( Node* node, uchar chan, Matrix4F* xform, float bdiv, uchar val_surf, uchar val_inside  )
{
	int cnt = 0;
	Extents e = ComputeExtents ( node );

/*	// OpenGL voxelize	
	uchar dt = mPool->getAtlas(chan).type;				
	PrepareV3D ( e.ires, dt );												// Preapre 3D Texture for hardware raster						
	mV3D->SurfaceVoxelizeFastGL ( e.vmin, e.vmax, xform );						// Voxelize with graphics pipeline				
	if ( node->mLev==0 ) {													// Generate voxels in a brick..
		mPool->AtlasCopyTex ( chan, node->mValue, mV3D->getPtr() );			//   Copy voxels into GVDB atlas
		cudaCheck ( cuCtxSynchronize(), "sync(AtlasCopyData)" );			//   Must sync after CUDA copy
	} else {																// OR Generate children..
		PrepareAux ( AUX_VOXELIZE, e.icnt, mPool->getSize(dt), true, true );//   Prepare CPU buffer for data retrieve		
		mV3D->RetrieveGL ( mAux[AUX_VOXELIZE].cpu );						//   Retrieve into CPU buffer		
		cnt = ActivateRegionFromAux ( e, AUX_VOXELIZE, dt );				//   Activate nodes at next level down		
	} */

	// CUDA voxelize
	uchar dt = mPool->getAtlas(chan).type;				
	PrepareAux ( AUX_VOXELIZE, e.icnt, mPool->getSize(dt), true, true );	//   Prepare buffer for data retrieve		

	Vector3DI block ( 8, 8, 8 );
	Vector3DI grid ( int(e.ires.x/block.x), int(e.ires.y/block.y), int(e.ires.z)/block.z );		
	int vcnt = mAux[AUX_VERTEX_BUF].num;
	int ecnt = mAux[AUX_ELEM_BUF].num;
	int bmax = mAux[AUX_GRIDOFF].num;
	int tcnt = mAux[AUX_TRI_BUF].num;
	void* args[11] = { &e.vmin, &e.vmax, &e.ires, &mAux[AUX_VOXELIZE].gpu, &val_surf, &val_inside, &bdiv, &bmax, &mAux[AUX_GRIDCNT].gpu, &mAux[AUX_GRIDOFF].gpu, &mAux[AUX_TRI_BUF].gpu };

	//CheckData ( "AUX_GRIDCNT", mAux[AUX_GRIDCNT].gpu, T_INT, sizeof(int), bmax );
	//CheckData ( "AUX_GRIDOFF", mAux[AUX_GRIDOFF].gpu, T_INT, sizeof(int), bmax );
	//CheckData ( "AUX_TRI_BUF", mAux[AUX_TRI_BUF].gpu, T_FLOAT3, sizeof(float3), tcnt );

	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_VOXELIZE], grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args, NULL ), "cuLaunchKernel", "VoxelizeNode" );

	if ( node->mLev==0 ) {
		mPool->AtlasCopyLinear ( chan, node->mValue, mAux[AUX_VOXELIZE].gpu );
	} else {
		RetrieveData ( mAux[AUX_VOXELIZE] );		
		cnt = ActivateRegionFromAux ( e, AUX_VOXELIZE, dt );
	} 
	return cnt;
}

// SolidVoxelize - Voxelize a polygonal mesh to a sparse volume
void VolumeGVDB::SolidVoxelize ( uchar chan, Model* model, Matrix4F* xform, uchar val_surf, uchar val_inside )
{
	TimerStart();
	
	AuxGeometryMap ( model, AUX_VERTEX_BUF, AUX_ELEM_BUF );					// Setup VBO for CUDA (interop)
	
	// Prepare model geometry for use by CUDA
	cudaCheck ( cuMemcpyHtoD ( cuXform, xform->GetDataF(), sizeof(float)*16), "cuMemcpyHtoD(xform)", "SolidVoxelize" );	// Send transform
	
	// Identify model bounding box
	model->ComputeBounds ( *xform, 0.1 );
	mObjMin = model->objMin; mObjMax = model->objMax;
	mVoxMin = mObjMin;	mVoxMin /= mVoxsize;
	mVoxMax = mObjMax;  mVoxMax /= mVoxsize;
	mVoxRes = mVoxMax; mVoxRes -= mVoxMin;

	// VDB Hierarchical Rasterization	
	Clear ();									// creates a new root	

	// Voxelize all nodes in bounding box at starting level		
	int N = 3;
	Extents e = ComputeExtents ( N, mObjMin, mObjMax );			// start - level N
	ActivateRegion ( e );									// activate - level N-1
	PrepareV3D ( Vector3DI(8,8,8), 0 );

	// Voxelize at each level	
	int node_cnt, cnt;
	Node* node;	
	for (int lev = N-1; lev >= 0; lev-- ) {						// scan - level N-1 down to 0
		node_cnt = getNumNodes(lev);
		cnt = 0;		
		// Insert triangles into bins
		float ydiv = getCover(lev).y;							// use brick boundaries for triangle sorting
		Vector3DI tcnts = InsertTriangles ( model, xform, ydiv );	

		// Voxelize each node at this level
		for (int n = 0; n < node_cnt; n++ ) {					// get each node at current
			node = getNodeAtLevel ( n, lev );
			cnt += VoxelizeNode ( node, chan, xform, ydiv, val_surf, val_inside );	// Voxelize each node
		}
		if ( lev==1 ) {											// Finish and Update atlas before doing bricks			
			FinishTopology();					
			UpdateAtlas();			
		}		
		verbosef("Voxelized.. lev: %d, nodes: %d, new: %d\n", lev, node_cnt, cnt );
	}	

	// Update apron
	UpdateApron ();	
	cudaCheck ( cuCtxSynchronize (), "cuCtxSync", "SolidVoxelize" );

	PrepareV3D ( Vector3DI(0,0,0), 0 );

	AuxGeometryUnmap ( model, AUX_VERTEX_BUF, AUX_ELEM_BUF );

	float msec = TimerStop();
	verbosef( "Voxelize Complete: %4.2f\n", msec );
}

// Insert triangles into auxiliary bins
Vector3DI VolumeGVDB::InsertTriangles ( Model* model, Matrix4F* xform, float& ydiv )
{
	// Identify model bounding box
	model->ComputeBounds ( *xform, 0.1 );	
	int ybins = int(model->objMax.y / ydiv)+1;						// y divisions align with lev0 brick boundaries	
	
	PrepareAux ( AUX_GRIDCNT, ybins, sizeof(uint), true, true );	// allow return to cpu
	PrepareAux ( AUX_GRIDOFF, ybins, sizeof(uint), false, false );
	int vcnt = mAux[AUX_VERTEX_BUF].num;			// input
	int ecnt = mAux[AUX_ELEM_BUF].num;
	int tri_cnt = 0;								// output
		
	Vector3DI block ( 512, 1, 1 );
	Vector3DI grid ( int(ecnt/block.x)+1, 1, 1 );	
	void* args[7] = { &ydiv, &ybins, &mAux[AUX_GRIDCNT].gpu, &vcnt, &ecnt, &mAux[AUX_VERTEX_BUF].gpu, &mAux[AUX_ELEM_BUF].gpu };
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_INSERT_TRIS], grid.x, 1, 1, block.x, 1, 1, 0, NULL, args, NULL ), "cuLaunchKernel(INSERT_TRIS)", "InsertTriangles" );

	// Prefix sum for bin offsets	
	PrefixSum ( mAux[AUX_GRIDCNT].gpu, mAux[AUX_GRIDOFF].gpu, ybins );
	
	// Sort triangles into bins (deep copy)		
	RetrieveData ( mAux[AUX_GRIDCNT] );
	uint* cnt = (uint*) mAux[AUX_GRIDCNT].cpu;
	for (int n=0; n < ybins; n++ ) tri_cnt += cnt[n];				// get total - should return from prefixsum
		
	// Reset grid counts
	cudaCheck ( cuMemsetD8 ( mAux[AUX_GRIDCNT].gpu, 0, sizeof(uint)*ybins ), "cuMemsetD8", "InsertTriangles" );

	// Prepare output triangle buffer
	PrepareAux ( AUX_TRI_BUF, tri_cnt, sizeof(Vector3DF)*3, false, false );
	// verbosef ( "ybins: %d, tri_cnt: %d\n", ybins, tri_cnt );

	// Deep copy sorted tris into output buffer
	block.Set ( 512, 1, 1 );
	grid.Set ( int(tri_cnt/block.x)+1, 1, 1 );	
	void* args2[10] = { &ydiv, &ybins, &mAux[AUX_GRIDCNT].gpu, &mAux[AUX_GRIDOFF].gpu, &tri_cnt, &mAux[AUX_TRI_BUF].gpu, &vcnt, &ecnt, &mAux[AUX_VERTEX_BUF].gpu, &mAux[AUX_ELEM_BUF].gpu };
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_SORT_TRIS], grid.x, 1, 1, block.x, 1, 1, 0, NULL, args2, NULL ), "cuLaunch(SORT_TRIS)", "InsertTriangles" );

	return Vector3DI( ybins, tri_cnt, ecnt);	// return: number of bins, total inserted tris, original tri count
}

// Voxelize a mesh as surface voxels using OpenGL hardware rasterizer
void VolumeGVDB::SurfaceVoxelizeGL ( uchar chan, Model* model, Matrix4F* xform )
{
	#ifdef BUILD_OPENGL

	Volume3D vtemp ( mScene ) ;		
	std::vector< Vector3DF >	leaf_pos;
	std::vector< uint64 >		leaf_ptr;
	slong leaf;
	int res0 = getRes ( 0 );	
	Vector3DI vres0 ( res0, res0, res0 );		// leaf resolution
	Vector3DF vrange0 = getRange(0);

	// VDB Hierarchical Rasterization
	PERF_PUSH ( "clear" );
	Clear ();									// creates a new root
	PERF_POP ();

	// Configure model
	model->ComputeBounds ( *xform, 0.1 );
	mObjMin = model->objMin; mObjMax = model->objMax;
	mVoxMin = mObjMin;	mVoxMin /= mVoxsize;
	mVoxMax = mObjMax;  mVoxMax /= mVoxsize;
	mVoxRes = mVoxMax; mVoxRes -= mVoxMin;
	
	// Determine which leaf voxels to activate
	if ( mbProfile ) PERF_PUSH ( "activate" );	

	int res1 = getRes ( 1 );	
	Vector3DI vres1 ( res1, res1, res1 );		// large-scale resolution = level 1
	Vector3DF vmin0, vmin1, vmax1;	
	Vector3DF vrange1 = getRange(1);
	Vector3DI vstart, vstop;	
	float* vdat = (float*) malloc ( res1*res1*res1*sizeof(float) );
	unsigned char vpix[8], c[8];
	
	vstart = mVoxMin;  vstart /= vrange1;
	vstop  = mVoxMax;  vstop  /= vrange1;

	// Resize temporary volume to level 1
	vtemp.Resize ( T_FLOAT, vres1, 0x0, true );		// use opengl
	
	verbosef( "  L1 range: %d %d %d, %d %d %d\n", vstart.x, vstart.y, vstart.z, vstop.x, vstop.y, vstop.z);

	vtemp.PrepareRasterGL ( true );

	// Scan over domain with level-1 rasterize	
	bool bnew = false;
	for ( int z1 = vstart.z; z1 <= vstop.z; z1++ ) {		
		for ( int y1 = vstart.y; y1 <= vstop.y; y1++ ) {
			for ( int x1 = vstart.x; x1 <= vstop.x; x1++ ) {

				// Rasterize at level 1 (green) 
				vmin1 = Vector3DI(x1+0, y1+0, z1+0); vmin1 *= vrange1;
				vmax1 = Vector3DI(x1+1, y1+1, z1+1); vmax1 *= vrange1;				
				vtemp.SurfaceVoxelizeFastGL ( vmin1*mVoxsize, vmax1*mVoxsize, xform );				

				// Readback the voxels
				vtemp.RetrieveGL ( (char*) vdat );

				// Identify leaf nodes at level-1
				for ( int z0=0; z0 < res1; z0++ ) {
					for ( int y0=0; y0 < res1; y0++ ) {
						for ( int x0=0; x0 < res1; x0++ ) {
							vpix[0] = *(vdat + (z0*res1 + y0)*res1 + x0);
							if ( vpix[0] > 0 ) {
								vmin0.x = vmin1.x + x0 * vrange1.x / res1;
								vmin0.y = vmin1.y + y0 * vrange1.y / res1;
								vmin0.z = vmin1.z + z0 * vrange1.z / res1;										
								leaf = ActivateSpace ( mRoot, vmin0, bnew );	
								if ( leaf != ID_UNDEFL ) {
									vmin0 *= mVoxsize;
									leaf_pos.push_back ( vmin0 );
									leaf_ptr.push_back ( leaf );
								}
							} else {
								// correction when not using conservative raster									
								c[1]=c[2]=c[3]=c[4]=c[5]=c[6] = 0;
								vpix[1] = (x0==res1-1) ? 0 :	*(vdat + ( z0*res1 +	 y0)	*res1 +	(x0+1));
								vpix[2] = (y0==res1-1) ? 0 :	*(vdat + ( z0*res1 +	(y0+1))	*res1 +	 x0);
								vpix[3] = (z0==res1-1) ? 0 :	*(vdat + ((z0+1)*res1 +  y0)	*res1 +	 x0);
								vpix[4] = (x0==0) ? 0 :		*(vdat + ( z0*res1 +	 y0)	*res1 +	(x0-1));
								vpix[5] = (y0==0) ? 0 :		*(vdat + ( z0*res1 +	(y0-1))	*res1 +	 x0);
								vpix[6] = (z0==0) ? 0 :		*(vdat + ((z0-1)*res1 +  y0)	*res1 +	 x0);
								vpix[7] = (vpix[1] > 0) + (vpix[2] > 0) + (vpix[3] > 0) + (vpix[4] > 0) + (vpix[5] > 0 ) + (vpix[6] > 0 );								
								
								if ( vpix[7] > 0 ) {
									vmin0.x = vmin1.x + x0 * vrange1.x / res1;
									vmin0.y = vmin1.y + y0 * vrange1.y / res1;
									vmin0.z = vmin1.z + z0 * vrange1.z / res1;									
									leaf = ActivateSpace ( mRoot, vmin0, bnew );
									if ( leaf != ID_UNDEFL ) {
										vmin0 *= mVoxsize;
										leaf_pos.push_back ( vmin0 );
										leaf_ptr.push_back ( leaf );
									}
								}
								c[1]=c[2]=c[3]=c[4]=c[5]=c[6] = 0;
								vpix[1] = (x0==res1-1) ? (++c[1]==1) :	*(vdat + ( z0*res1 +	 y0)	*res1 +	(x0+1));
								vpix[2] = (y0==res1-1) ? (++c[2]==1) :	*(vdat + ( z0*res1 +	(y0+1))	*res1 +	 x0);
								vpix[3] = (z0==res1-1) ? (++c[3]==1) :	*(vdat + ((z0+1)*res1 +  y0)	*res1 +	 x0);
								vpix[4] = (x0==0) ? (++c[4]==1) :		*(vdat + ( z0*res1 +	 y0)	*res1 +	(x0-1));
								vpix[5] = (y0==0) ? (++c[5]==1) :		*(vdat + ( z0*res1 +	(y0-1))	*res1 +	 x0);
								vpix[6] = (z0==0) ? (++c[6]==1) :		*(vdat + ((z0-1)*res1 +  y0)	*res1 +	 x0);
								vpix[7] = (vpix[1] > 0) + (vpix[2] > 0) + (vpix[3] > 0) + (vpix[4] > 0) + (vpix[5] > 0 ) + (vpix[6] > 0 );
								c[7] = (c[1]+c[2]+c[3]+c[4]+c[5]+c[6] == 3) ? 1 : 0;
								if ( vpix[7] >= 3 + c[7] ) {
									vmin0.x = vmin1.x + x0 * vrange1.x / res1;
									vmin0.y = vmin1.y + y0 * vrange1.y / res1;
									vmin0.z = vmin1.z + z0 * vrange1.z / res1;									
									leaf = ActivateSpace ( mRoot, vmin0, bnew );
									if ( leaf != ID_UNDEFL ) {
										vmin0 *= mVoxsize;
										leaf_pos.push_back ( vmin0 );
										leaf_ptr.push_back ( leaf );
									}
								}
							}
						}
					}
				}
			}
		}
	}	
	if ( mbProfile ) PERF_POP ();

	FinishTopology ();

	UpdateAtlas ();

	cudaCheck ( cuCtxSynchronize(), "cuCtxSync", "SurfaceVoxelizeGL" );

	vtemp.PrepareRasterGL ( false );				

	verbosef( "  # of leaves: %d\n", leaf_ptr.size() );
	if ( leaf_ptr.size()==0 ) {
		gprintf ( "  ERROR: No data generated.\n" );
		free ( vdat );
		return; 
	}
	
	// Create atlas if none exists
	if ( chan >= mPool->getNumAtlas() ) {
		gprintf ( "  ERROR: Channel %d not defined for SurfaceVoxelizeGL. Call AddChannel first.\n", (int) chan );
		free ( vdat );
		return;		
	}

	// Resize temp 3D texture to match leaf res	
	vtemp.Resize ( T_FLOAT, vres0, 0x0, true );

	if ( leaf_ptr.size() > 0 ) {


		// Rasterize each leaf node
		if ( mbProfile ) PERF_PUSH ( "rasterize" );		
		vtemp.PrepareRasterGL ( true );

		for (int n=0; n < leaf_ptr.size(); n++ ) {

			// Rasterize into temporary 3D texture						
			vtemp.SurfaceVoxelizeFastGL ( leaf_pos[n], leaf_pos[n] + vrange0*mVoxsize, xform );			

			// Copy 3D texture in VDB atlas
			Node* node = getNode ( leaf_ptr[n] );
			if ( node == 0x0 || node->mValue.x == -1 ) {
				gprintf ( "WARNING: Node or value is empty in SurfaceVoxelizeGL.\n" );
			} else {
				mPool->AtlasCopyTex ( chan, node->mValue, vtemp.getPtr() );			
			}
		}
		vtemp.PrepareRasterGL ( false );			

		if ( mbProfile ) PERF_POP ();
	}
	glFinish ();

	cudaCheck ( cuCtxSynchronize(), "cuCtxSync", "SurfaceVoxelizeGL" );

	UpdateApron ();

	free ( vdat ); 	

	#endif
}

// Measure node statistics (recursive)
void VolumeGVDB::Measure ( statVec& stats, slong nodeid )
{
	if ( nodeid == ID_UNDEFL ) return;
	Node* node = getNode ( nodeid );		
	int l = node->mLev;
	int sz = getRes(l)*getRes(l)*getRes(l);
	if ( isLeaf (nodeid) ) {
		// leaf				
		stats[l].cover += sz;
		stats[l].occupy = 0;	// unknown
		stats[l].mem_node += (slong) sizeof(Node);
		stats[l].mem_mask = 0;
		stats[l].mem_compact = 0;
		stats[l].mem_full = 0;
		stats[l].num++;
	} else {
		// internal
		stats[l].cover += sz;
		stats[l].occupy += node->getNumChild();
		stats[l].mem_node += (slong) sizeof(Node);
		stats[l].mem_mask += (slong) node->getMaskBytes();
		stats[l].mem_compact += (slong) node->getNumChild()*sizeof(uint64);
		stats[l].mem_full += (slong) node->getMaskBytes()*8 * sizeof(uint64);
		stats[l].num++;				
		for (int n=0; n < node->getNumChild(); n++ ) {			
			Measure ( stats, getChildNode( nodeid, n ) );
		}
	}
}

// Measure pools
float VolumeGVDB::MeasurePools ()
{
	uint64 pool0_total=0, pool1_total=0;
	int levs = mPool->getNumLevels ();
	float MB = 1024.0*1024.0;

	for (int n=0; n < levs; n++ ) {
		pool0_total += mPool->getPoolSize ( 0, n );
		pool1_total += mPool->getPoolSize ( 1, n );
	}
	return (pool0_total + pool1_total) / MB;
}

// Measure GVDB topology & data
void VolumeGVDB::Measure ( bool bPrint )
{	
	float tuse_nodes=0, tuse_masks=0, tuse_full=0, tuse_compact=0;
	float tmax_nodes=0, tmax_masks=0, tmax_full=0, tmax_compact=0;
	int	node_total = 0, node_max = 0, ave_child, max_child;
	Vector3DI axisres, axiscnt;
	int leafdim;
	uint64 atlas_sz = 0;	
	mTreeMem = 0;

	int levs = mPool->getNumLevels ();

	//--- Measure
	Stat t;
	statVec	stats;	
	for (int n=0; n < levs; n++ ) 
		stats.push_back( t );
	Measure ( stats, mRoot );


	//--- Print	
	if ( !bPrint ) return;
	gprintf ( "  EXTENTS:\n" );
	gprintf ( "   Volume Res: %.0f x %.0f x %.0f\n", mVoxRes.x, mVoxRes.y, mVoxRes.z );
	gprintf ( "   Volume Max: %.0f x %.0f x %.0f\n", mVoxResMax.x, mVoxResMax.y, mVoxResMax.z );
	gprintf ( "   Bound Min:  %f x %f x %f\n", mObjMin.x, mObjMin.y, mObjMin.z );
	gprintf ( "   Bound Max:  %f x %f x %f\n", mObjMax.x, mObjMax.y, mObjMax.z );
	gprintf ( "   Voxelsize:  %f x %f x %f\n", mVoxsize.x, mVoxsize.y, mVoxsize.z );
	gprintf ( "  VDB CONFIG: <%d, %d, %d, %d, %d>:\n", getLD(4), getLD(3), getLD(2), getLD(1), getLD(0) );
	gprintf (     "             # Nodes / # Pool    Max Node  Ave Child  Max Child\n" );
	for (int n=0; n < levs; n++ ) {
		if ( stats[n].num==0 ) stats[n].num = 1;
		if ( stats[n].cover==0 ) stats[n].cover = 1;
		
		// nodes in use 
		node_total += stats[n].num;
		tuse_nodes += stats[n].num * sizeof(Node);
		tuse_masks += stats[n].num * (mPool->getPoolWidth(0, n) - sizeof(Node));
		tuse_compact += stats[n].mem_compact;
		tuse_full += stats[n].mem_full;
		
		// nodes in memory		
		node_max += mPool->getPoolMax(0, n);
		tmax_nodes += mPool->getPoolMax(0, n) * sizeof(Node);
		tmax_masks += mPool->getPoolMax(0, n) * (mPool->getPoolWidth(0, n) - sizeof(Node));
		tmax_full  += mPool->getPoolMax(0, n) * mPool->getPoolWidth(1,n);
		
		// child averages
		ave_child = int(stats[n].mem_compact / (stats[n].num*sizeof(uint64)) );
		max_child = int(stats[n].mem_full    / (stats[n].num*sizeof(uint64)) );

		gprintf ( "   Level %d: %8d %8d %10d %10d %10d\n", n, (int) stats[n].num, mPool->getPoolCnt(0,n), mPool->getPoolMax(0, n), ave_child, max_child );		
	}
	if ( bPrint ) {
		float MB = 1024.0*1024.0;	// convert to MB
		gprintf ( "   Ratio Interior/Total: %4.2f%%%%\n", float(node_total-stats[0].num)*100.0f / node_total );

		// Atlas info
		if ( mPool->getNumAtlas() > 0 ) {
			leafdim = mPool->getAtlas(0).stride;				// voxel res of one brick
			axiscnt = mPool->getAtlas(0).subdim;				// number of bricks in atlas			
			axisres = axiscnt * (leafdim + mApron*2);			// number of voxels in atlas
			atlas_sz = mPool->getAtlas(0).size;			

			gprintf ( "  ATLAS STORAGE\n" );
			gprintf ( "   Atlas Res:     %d x %d x %d  LeafCnt: %d x %d x %d  LeafDim: %d^3\n", axisres.x, axisres.y, axisres.z, axiscnt.x, axiscnt.y, axiscnt.z, leafdim );
			Vector3DI vb = mVoxResMax;
			vb /= Vector3DI(leafdim, leafdim, leafdim);
			int vbrk = vb.x*vb.y*vb.z;					// number of bricks covering bounded world domain
			int sbrk = axiscnt.x*axiscnt.y*axiscnt.z;	// number of bricks stored in atlas
			int abrk = stats[0].num;
			gprintf ( "   Vol Extents:   %d bricks,  %5.2f million voxels\n", vbrk, float(vbrk)*leafdim*leafdim*leafdim / 1000000.0f );
			gprintf ( "   Atlas Storage: %d bricks,  %5.2f million voxels\n", abrk, float(abrk)*leafdim*leafdim*leafdim / 1000000.0f );
			gprintf ( "   Atlas Active:  %d bricks,  %5.2f million voxels\n", stats[0].num, float(stats[0].num)*leafdim*leafdim*leafdim / 1000000.0f );
			gprintf ( "   Occupancy:     %6.2f%%%% \n", float(stats[0].num)*100.0f / vbrk );
		}
		float vol_total, vol_dense;		
		vol_total = 0; vol_dense = 0;		
		gprintf ( "  MEMORY USAGE:\n");
		gprintf ( "   Topology Nodes:    %6.2f MB (%6.2f MB active)\n", tmax_nodes/MB, tuse_nodes/MB);
		gprintf ( "   Topology Bitmasks: %6.2f MB (%6.2f MB active)\n", tmax_masks/MB, tuse_masks/MB);
		gprintf ( "   Topology Pointers: %6.2f MB (%6.2f MB active)\n", tmax_full/MB,  tuse_full/MB);
		gprintf ( "   Topology Total:    %6.2f MB (%6.2f MB active)\n", (tmax_nodes+tmax_masks+tmax_full)/MB, (tuse_nodes+tuse_masks+tuse_full)/MB );
		gprintf ( "   Atlas:\n" );		
		int bpv;
		for (int n=0; n < mPool->getNumAtlas(); n++ ) {
			bpv = mPool->getSize(mPool->getAtlas(n).type) ;
			gprintf ( "     Channel %d:       %6.2f MB (%d bytes/vox)\n", n, float(mPool->getAtlas(n).size) / MB, bpv );
			vol_total += mPool->getAtlas(n).size;
			vol_dense += (mVoxResMax.x * mVoxResMax.y * mVoxResMax.z) * bpv;
		}
		gprintf ( "   Atlas Total:       %6.2f MB (Dense: %10.2f MB)\n", vol_total / MB, vol_dense / MB );		
		gprintf ( "   Volume Total:      %6.2f MB \n", (tmax_nodes+tmax_masks+tmax_full + vol_total) / MB );

	}
}

// Get memory usage
void VolumeGVDB::getMemory ( float& voxels, float& overhead, float& effective )
{
	Measure ( false );

	// all measurements in MB
	voxels = (float) mPool->getAtlasMem();
	overhead = (float) mTreeMem;
	effective = float(mVoxRes.x*mVoxRes.y*mVoxRes.z*4.0) / (1024.0*1024.0);
}

// Prepare VDB data on GPU before kernel launch
void VolumeGVDB::PrepareVDB ()
{
	if ( mPool->getNumAtlas() == 0 ) {
		gprintf ( "ERROR: No atlas created.\n" );
	}

	// Send VDB info
	if ( mVDBInfo.update ) {			
		mVDBInfo.update = false;
		int tlev = 1;
		int levs = mPool->getNumLevels ();
		for (int n = levs-1; n >= 0; n-- ) {				
			mVDBInfo.dim[n]			= mLogDim[n];
			mVDBInfo.res[n]			= getRes(n);
			mVDBInfo.vdel[n]		= Vector3DF(getRange(n)) * mVoxsize;	mVDBInfo.vdel[n] /= Vector3DF( getRes3DI(n) );
			mVDBInfo.noderange[n]	= getRange(n);		// integer (cannot send cover)
			mVDBInfo.nodecnt[n]		= mPool->getPoolCnt(0, n);
			mVDBInfo.nodewid[n]		= mPool->getPoolWidth(0, n);
			mVDBInfo.childwid[n]	= mPool->getPoolWidth(1, n);
			mVDBInfo.nodelist[n]	= mPool->getPoolGPU(0, n);
			mVDBInfo.childlist[n]	= mPool->getPoolGPU(1, n);								
			if ( mVDBInfo.nodecnt[n] == 1 ) tlev = n;		// get top level for rendering
		}
		mVDBInfo.voxelsize			= mVoxsize;		
		mVDBInfo.atlas_map			= mPool->getAtlasMapGPU(0);		
		mVDBInfo.atlas_apron		= mPool->getAtlas(0).apron;	
		mVDBInfo.atlas_cnt			= mPool->getAtlas(0).subdim;		// number of bricks on each axis of atlas
		mVDBInfo.atlas_res			= mPool->getAtlasRes(0);			// total resolution of atlas
		mVDBInfo.brick_res			= mPool->getAtlasBrickres(0);		// resolution of single brick (sides are all same)
		int blkres = mPool->getAtlasBrickres(0);		
		for (int n=0; n < mApron; n++ ) {
			mVDBInfo.apron_table[ n ] = n;
			mVDBInfo.apron_table[ (mApron*2-1)-n ] = (blkres-1) - n;
		}
		mVDBInfo.top_lev			= tlev;		
		mVDBInfo.bmin				= mObjMin;
		mVDBInfo.bmax				= mObjMax;
		mVDBInfo.thresh				= getScene()->mVThreshold;
		mVDBInfo.transfer			= getTransferFuncGPU();
		if ( mVDBInfo.transfer == 0 ) {
			gprintf ( "Error: Transfer function not on GPU. Must call CommitTransferFunc.\n" );
			gerror ();
		}
		cudaCheck ( cuMemcpyHtoD ( cuVDBInfo, &mVDBInfo, sizeof(VDBInfo) ), "cuMemcpyHtoD(VDBInfo)", "PrepareVDB" );
		//cudaCheck ( cuCtxSynchronize (), "cuCtxSync(PrepareVDB)" );
	}		
}


bool glCheck()
{
	GLenum error = glGetError();
	if (error != GL_NO_ERROR) {
		gprintf("GL ERROR\n");
		gerror();
		return false;
	}
	return true;
}

void VolumeGVDB::WriteDepthTexGL(int chan, int glid)
{
	if ( mbProfile ) PERF_PUSH ( "WriteDepthGL" );

	if (mDummyFrameBuffer == -1) {
		glGenFramebuffers(1, (GLuint*)&mDummyFrameBuffer);
	}

	// Set current framebuffer to input buffer:
	glBindFramebuffer(GL_FRAMEBUFFER, mDummyFrameBuffer);

	// Setup frame buffer so that we can copy depth info from the depth target bound
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, glid, 0);

	// Copy contents of depth target into _depthBuffer for use in CUDA rendering:
	glBindBuffer(GL_PIXEL_PACK_BUFFER, mRenderBuf[chan].glid);
	//glReadPixels(0, 0, mRenderBuf[chan].stride, mRenderBuf[chan].max / mRenderBuf[chan].stride, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, 0);
	glReadPixels(0, 0, mRenderBuf[chan].stride, mRenderBuf[chan].max / mRenderBuf[chan].stride, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	glCheck();

	// Clear render target:
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	if ( mbProfile ) PERF_POP ();
}

void VolumeGVDB::WriteRenderTexGL ( int chan, int glid )
{
  if ( mbProfile ) PERF_PUSH ( "ReadTexGL" );

  if (mRenderBuf[chan].garray == 0) {
    // Prepare render target
    mRenderBuf[chan].glid = glid;

    // Cuda-GL interop to get target CUarray				
    cudaCheck(cuGraphicsGLRegisterImage(&mRenderBuf[chan].grsc, glid, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST), "cuGraphicsGLRegisterImage", "WriteRenderTexGL" );
    cudaCheck(cuGraphicsMapResources(1, &mRenderBuf[chan].grsc, 0), "cudaGraphicsMapResources", "WriteRenderTexGL" );
    cudaCheck(cuGraphicsSubResourceGetMappedArray(&mRenderBuf[chan].garray, mRenderBuf[chan].grsc, 0, 0), "cuGraphicsSubResourceGetMappedArray", "WriteRenderTexGL" );
    cudaCheck(cuGraphicsUnmapResources(1, &mRenderBuf[chan].grsc, 0), "cudaGraphicsUnmapRsrc", "WriteRenderTexGL" );
  }
  int bpp = mRenderBuf[chan].size / mRenderBuf[chan].max;
  int width = mRenderBuf[chan].stride * bpp;
  int height = mRenderBuf[chan].max / mRenderBuf[chan].stride;

  // Cuda Memcpy2D to transfer cuda output buffer
  // into opengl texture as a CUarray
  CUDA_MEMCPY2D cp = { 0 };
  cp.dstMemoryType = CU_MEMORYTYPE_DEVICE;		// from cuda buffer
  cp.dstDevice = mRenderBuf[chan].gpu;
  cp.srcMemoryType = CU_MEMORYTYPE_ARRAY;			// to CUarray (opengl texture)
  cp.srcArray = mRenderBuf[chan].garray;
  cp.WidthInBytes = width;
  cp.Height = height;

  cudaCheck(cuMemcpy2D(&cp), "cuMemcpy2D", "WriteRenderTexGL");

  cuCtxSynchronize();

  if ( mbProfile ) PERF_POP ();
}

void VolumeGVDB::ReadRenderTexGL ( int chan, int glid )
{
	if ( mbProfile ) PERF_PUSH ( "ReadTexGL" );

	if ( mRenderBuf[chan].garray == 0 ) {
		// Prepare render target
		mRenderBuf[chan].glid = glid;

		// Cuda-GL interop to get target CUarray				
		cudaCheck ( cuGraphicsGLRegisterImage ( &mRenderBuf[chan].grsc, glid, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST ), "cuGraphicsGLRegisterImage", "ReadRenderTexGL" );	
		cudaCheck ( cuGraphicsMapResources(1, &mRenderBuf[chan].grsc, 0), "cudaGraphicsMapResources", "ReadRenderTexGL" );
		cudaCheck ( cuGraphicsSubResourceGetMappedArray ( &mRenderBuf[chan].garray, mRenderBuf[chan].grsc, 0, 0 ), "cuGraphicsSubResourceGetMappedArray", "ReadRenderTexGL" );				
		cudaCheck ( cuGraphicsUnmapResources(1, &mRenderBuf[chan].grsc, 0), "cudaGraphicsUnmapRsrc", "ReadRenderTexGL" );
	}
	int bpp = mRenderBuf[chan].size / mRenderBuf[chan].max;
	int width = mRenderBuf[chan].stride * bpp;
	int height = mRenderBuf[chan].max / mRenderBuf[chan].stride;

	// Cuda Memcpy2D to transfer cuda output buffer
	// into opengl texture as a CUarray
	CUDA_MEMCPY2D cp = {0};
	cp.srcMemoryType = CU_MEMORYTYPE_DEVICE;		// from cuda buffer
	cp.srcDevice = mRenderBuf[chan].gpu;
	cp.dstMemoryType = CU_MEMORYTYPE_ARRAY;			// to CUarray (opengl texture)
	cp.dstArray = mRenderBuf[chan].garray;
	cp.WidthInBytes = width;
	cp.Height = height;
	
	cudaCheck ( cuMemcpy2D ( &cp ), "cuMemcpy2D", "ReadRenderTexGL");

	cuCtxSynchronize ();

	if ( mbProfile ) PERF_POP ();
}

// Add a depth buffer
void VolumeGVDB::AddDepthBuf(int chan, int width, int height)
{
	mRenderBuf.resize(chan + 1);
	
	mRenderBuf[chan].alloc = mPool;
	mRenderBuf[chan].cpu = 0x0;				// no cpu residence yet
	mRenderBuf[chan].num = 0;
	mRenderBuf[chan].garray = 0;
	mRenderBuf[chan].grsc = 0;
	mRenderBuf[chan].glid = -1;
	mRenderBuf[chan].gpu = 0x0;
	
	ResizeDepthBuf( chan, width, height );
}

// Add a render buffer
void VolumeGVDB::AddRenderBuf ( int chan, int width, int height, int byteperpix )
{
	mRenderBuf.resize ( chan+1 );	

	mRenderBuf[chan].alloc = mPool;
	mRenderBuf[chan].cpu = 0x0;				// no cpu residence yet
	mRenderBuf[chan].num = 0;	
	mRenderBuf[chan].garray = 0;
	mRenderBuf[chan].grsc = 0;
	mRenderBuf[chan].glid = -1;
	mRenderBuf[chan].gpu = 0x0;

	ResizeRenderBuf ( chan, width, height, byteperpix );
}

// Resize a depth buffer
void VolumeGVDB::ResizeDepthBuf( int chan, int width, int height )
{
	if (chan >= mRenderBuf.size()) {
		gprintf("ERROR: Attempt to resize depth buf that doesn't exist.");
	}
	if (mRenderBuf[chan].glid != -1)
	{
		glDeleteBuffers(1, (GLuint*)&mRenderBuf[chan].glid);
		mRenderBuf[chan].glid = -1;
	}
	glGenBuffers(1, (GLuint*)&mRenderBuf[chan].glid);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, mRenderBuf[chan].glid);
	glBufferData(GL_PIXEL_PACK_BUFFER, width * height * sizeof(GLuint), NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

	mRenderBuf[chan].max = width * height;
	mRenderBuf[chan].size = width * height * sizeof(GLuint);
	mRenderBuf[chan].stride = width;
	
	cudaCheck(cuGraphicsGLRegisterBuffer(&mRenderBuf[chan].grsc, mRenderBuf[chan].glid, 0), "cuGraphicsGLRegisterBuffer", "ResizeDepthBuf" );
	cudaCheck(cuGraphicsMapResources(1, &mRenderBuf[chan].grsc, 0), "cudaGraphicsMapResources", "ResizeDepthBuf" );
	cudaCheck(cuGraphicsResourceGetMappedPointer(&mRenderBuf[chan].gpu, &mRenderBuf[chan].size, mRenderBuf[chan].grsc), "cuGraphicsResourceGetMappedPointer", "ResizeDepthBuf" );
	cudaCheck(cuGraphicsUnmapResources(1, &mRenderBuf[chan].grsc, 0), "cudaGraphicsUnmapRsrc", "ResizeDepthBuf" );
}

// Resize a render buffer
void VolumeGVDB::ResizeRenderBuf ( int chan, int width, int height, int byteperpix )
{
	if ( chan >= mRenderBuf.size() ) {
		gprintf ( "ERROR: Attempt to resize render buf that doesn't exist." );
	}
	mRenderBuf[chan].max = width * height; 
	mRenderBuf[chan].size = width * height * byteperpix; 
	mRenderBuf[chan].stride = width;
	if ( chan == 0 ) getScene()->SetRes ( width, height );
	
	size_t sz = mRenderBuf[chan].size;
	if ( mRenderBuf[chan].gpu != 0x0 ) { 
		cudaCheck ( cuMemFree ( mRenderBuf[chan].gpu ), "cuMemFree", "ResizeRenderBuf" ); 
	}
	cudaCheck ( cuMemAlloc ( &mRenderBuf[chan].gpu, sz ), "cuMemAlloc", "ResizeRenderBuf" );

	if ( mRenderBuf[chan].garray != 0x0 ) {		
		cudaCheck ( cuGraphicsUnregisterResource ( mRenderBuf[chan].grsc ), "cuGraphicsUnregisterResource", "ResizeRenderBuf" );
		mRenderBuf[chan].grsc = 0x0;
		mRenderBuf[chan].garray = 0x0;
	}

	cuCtxSynchronize ();
}

// Read render buffer to CPU
void VolumeGVDB::ReadRenderBuf ( int chan, unsigned char* outptr )
{
	if ( mbVerbose ) PERF_PUSH ( "ReadBuf" );
	mRenderBuf[chan].cpu = (char*) outptr;
	mPool->RetrieveMem ( mRenderBuf[chan] );		// transfer dev to host
	if ( mbVerbose ) PERF_POP ();
}

// Prepare scene data for rendering kernel
void VolumeGVDB::PrepareRender ( int w, int h, char shading, char filtering, int frame, int samples, float samt, uchar dbuf )
{
	// Send scene data
	Camera3D* cam = getScene()->getCamera();
	mScnInfo.width		= w;
	mScnInfo.height		= h;				
	mScnInfo.camnear	= cam->getNear ();
	mScnInfo.camfar 	= cam->getFar();
	mScnInfo.campos		= cam->origRayWorld;
	mScnInfo.cams		= cam->tlRayWorld;
	mScnInfo.camu		= cam->trRayWorld; mScnInfo.camu -= mScnInfo.cams;
	mScnInfo.camv		= cam->blRayWorld; mScnInfo.camv -= mScnInfo.cams;
	mScnInfo.camivprow0 = cam->invviewproj_matrix.GetRowVec(0);
	mScnInfo.camivprow1 = cam->invviewproj_matrix.GetRowVec(1);
	mScnInfo.camivprow2 = cam->invviewproj_matrix.GetRowVec(2);
	mScnInfo.camivprow3 = cam->invviewproj_matrix.GetRowVec(3);
	mScnInfo.bias		= 0.01f;
	mScnInfo.light_pos	= getScene()->getLight()->getPos();
	Vector3DF crs 		= Vector3DF(0,0,0);		// gView.getCursor(); --- mouse cursor
	mScnInfo.slice_pnt  = getScene()->getSectionPnt();
	mScnInfo.slice_norm = getScene()->getSectionNorm();
	mScnInfo.shading	= shading;
	mScnInfo.shadow_amt = samt;
	mScnInfo.filtering	= filtering;
	mScnInfo.frame		= frame;
	mScnInfo.samples	= samples;
	mScnInfo.backclr	= getScene()->getBackClr ();
	mScnInfo.extinct	= getScene()->getExtinct ();
	mScnInfo.steps		= getScene()->getSteps ();
	mScnInfo.cutoff		= getScene()->getCutoff ();
	mScnInfo.outbuf		= -1;			// NOT USED  (was mRenderBuf[0].gpu;)
	mScnInfo.dbuf 		= dbuf == 255 ? NULL : mRenderBuf[dbuf].gpu;

	cudaCheck ( cuMemcpyHtoD ( cuScnInfo, &mScnInfo, sizeof(ScnInfo) ), "cuMemcpyHtoD(ScnInfo)", "PrepareRender" );
}

// Render using custom user kernel
void VolumeGVDB::RenderKernel ( uchar rbuf, CUfunction user_kernel, char shading, char filtering, int frame, int sample, int max_samples, float samt )
{
	if (mbProfile) PERF_PUSH ( "Render" );
	
	int width = mRenderBuf[rbuf].stride;
	int height = mRenderBuf[rbuf].max / width;	

	// Send Scene info (camera, lights)
	PrepareRender ( width, height, shading, filtering, frame, max_samples, samt );

	// Send VDB Info & Atlas
	PrepareVDB ();												

	// Run CUDA User-Kernel
	Vector3DI block ( 16, 16, 1 );
	Vector3DI grid ( int(width/block.x)+1, int(height/block.y)+1, 1);		
	void* args[1] = { &mRenderBuf[rbuf].gpu };
	cudaCheck ( cuLaunchKernel ( user_kernel, grid.x, grid.y, 1, block.x, block.y, 1, 0, NULL, args, NULL ), "cuLaunch(user)", "RenderKernel" );
	
	if (mbProfile) PERF_POP ();
}

// Render using native kernel
void VolumeGVDB::Render ( uchar rbuf, char shading, char filtering, int frame, int sample, int max_samples, float samt, uchar dbuf )
{
	int width = mRenderBuf[rbuf].stride;
	int height = mRenderBuf[rbuf].max / width;	
	if ( shading==SHADE_OFF ) {
		cudaCheck ( cuMemsetD8 ( mRenderBuf[rbuf].gpu, 0, width*height*4 ), "cuMemsetD8", "Render" );
		return;
	}
	
	if (mbProfile) PERF_PUSH ( "Render" );

	// Send Scene info (camera, lights)
	PrepareRender ( width, height, shading, filtering, frame, max_samples, samt, dbuf );

	// Send VDB Info & Atlas
	PrepareVDB ();												

	// Prepare kernel
	Vector3DI block ( 16, 16, 1);
	Vector3DI grid ( int(width/block.x)+1, int(height/block.y)+1, 1);		
	void* args[1] = { &mRenderBuf[rbuf].gpu };
	int kern;
	switch ( shading ) {
	case SHADE_VOXEL:		kern = FUNC_RAYVOXEL;		break;
	case SHADE_TRILINEAR:	kern = FUNC_RAYTRILINEAR;	break;
	case SHADE_TRICUBIC:	kern = FUNC_RAYTRICUBIC;	break;
	case SHADE_EMPTYSKIP:	kern = FUNC_EMPTYSKIP;		break;
	case SHADE_SECTION2D:	kern = FUNC_SECTION2D;		break;
	case SHADE_SECTION3D:	kern = FUNC_SECTION3D;		break;
	case SHADE_LEVELSET:	kern = FUNC_RAYLEVELSET;	break;
	case SHADE_VOLUME:		kern = FUNC_RAYDEEP;		break;
	}
	
	// Run Raytracing kernel
	cudaCheck ( cuLaunchKernel ( cuFunc[kern], grid.x, grid.y, 1, block.x, block.y, 1, 0, NULL, args, NULL ), "cuRaycast", "Render" );
	
	cudaCheck ( cuCtxSynchronize (), "cuCtxSync", "Render" );

	if (mbProfile) PERF_POP ();
}

// Explicit raytracing
void VolumeGVDB::Raytrace ( DataPtr rays, char shading, int frame, float bias )
{
	if (mbProfile) PERF_PUSH ( "Raytrace" );

	// Send scene data
	PrepareRender ( 1, 1, shading, 0, frame, 1, 0 );

	// Send VDB Info & Atlas
	PrepareVDB ();												

	// Run CUDA GVDB Raytracer
	int cnt = rays.num;
	Vector3DI block ( 64, 1, 1);
	Vector3DI grid ( int(cnt/block.x)+1, 1, 1);		
	void* args[3] = { &cnt, &rays.gpu, &bias };
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_RAYTRACE], grid.x, 1, 1, block.x, 1, 1, 0, NULL, args, NULL ), "cuLaunch(RAYTRACE)", "Raytrace" );	 	

	if (mbProfile) PERF_POP ();

}

// Update apron (for all channels)
void VolumeGVDB::UpdateApron ()
{
	for (int n=0; n < mPool->getNumAtlas(); n++ )
		UpdateApron ( n );
}

// Update apron (one channel)
void VolumeGVDB::UpdateApron ( uchar chan )
{ 	
	if ( mApron == 0 ) return;

	if ( mbProfile ) PERF_PUSH ("UpdateApron");

	// Send VDB Info	
	PrepareVDB ();			

	// Determine grid and block dims
	Vector3DI atlasres = mPool->getAtlasRes( chan );		// size of atlas
	int brickres = mPool->getAtlasBrickres( chan );			// dimension of brick (including apron)
	Vector3DI axiscnt = mPool->getAtlas( chan ).subdim;		// number of bricks each axis

	Vector3DI block ( 8, 8, mApron*2 );	
	Vector3DI grid  ( int(atlasres.x/block.x)+1, int(atlasres.y/block.y)+1, int(atlasres.z/block.z)+1 );	
	int axis, kern;

	switch ( mPool->getAtlas(chan).type ) {
	case T_FLOAT:	kern = FUNC_UPDATEAPRON_F;		break;
	case T_FLOAT3:  kern = FUNC_UPDATEAPRON_F4;		break;		// F3 is implemented as F4 
	case T_FLOAT4:  kern = FUNC_UPDATEAPRON_F4;		break;
	case T_UCHAR:	kern = FUNC_UPDATEAPRON_C;		break;
	case T_UCHAR4:	kern = FUNC_UPDATEAPRON_C4;		break;
	}	

	void* args[4] = { &axis, &atlasres, &chan, &brickres };
	axis = 0; 
	cudaCheck ( cuLaunchKernel ( cuFunc[kern], axiscnt.x, grid.y, grid.z, block.z, block.x, block.y, 0, NULL, args, NULL ), "cuLaunch(UpdateApron[x])", "UpdateApron" );		
	axis = 1;
	cudaCheck ( cuLaunchKernel ( cuFunc[kern], grid.x, axiscnt.y, grid.z, block.x, block.z, block.y, 0, NULL, args, NULL ), "cuLaunch(UpdateApron[y])", "UpdateApron" );
	axis = 2;	
	cudaCheck ( cuLaunchKernel ( cuFunc[kern], grid.x, grid.y, axiscnt.z, block.x, block.y, block.z, 0, NULL, args, NULL ), "cuLaunch(UpdateApron[z])", "UpdateApron" );

	if ( mbProfile ) PERF_POP ();
}

// Run a custom user compute kernel
void VolumeGVDB::ComputeKernel ( CUmodule user_module, CUfunction user_kernel, uchar chan, bool bUpdateApron )
{
	if ( mbProfile ) PERF_PUSH ("ComputeKernel");

	SetModule ( user_module );

	// Send VDB Info (*to user module*)
	PrepareVDB ();

	// Determine grid and block dims (must match atlas bricks)	
	Vector3DI block ( 8, 8, 8 );
	Vector3DI res = mPool->getAtlasRes( chan );
	Vector3DI grid ( int(res.x/block.x)+1, int(res.y/block.y)+1, int(res.z/block.z)+1 );	

	void* args[2] = { &res, &chan };
	cudaCheck ( cuLaunchKernel ( user_kernel, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args, NULL ), "cuLaunch(UserKernel)", "ComputeKernel" );	
	
	if ( bUpdateApron ) {
		cudaCheck ( cuCtxSynchronize(), "cuCtxSync", "ComputeKernel" );
		SetModule ();			// Update apron must be called from default module
		UpdateApron ();				
	}

	if ( mbProfile ) PERF_POP();
}

// Run a native compute kernel
void VolumeGVDB::Compute ( int effect, uchar chan, int iter, Vector3DF parm, bool bUpdateApron )
{ 
	if ( mbProfile ) PERF_PUSH ("Compute");

	// Send VDB Info	
	PrepareVDB ();

	// Determine grid and block dims (must match atlas bricks)	
	Vector3DI block ( 8, 8, 8 );
	Vector3DI res = mPool->getAtlasRes( chan );
	Vector3DI grid ( int(res.x/block.x)+1, int(res.y/block.y)+1, int(res.z/block.z)+1 );	

	void* args[5] = { &res, &chan, &parm.x, &parm.y, &parm.z };	
	
	for (int n=0; n < iter; n++ ) {
		cudaCheck ( cuLaunchKernel ( cuFunc[effect], grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args, NULL ), "cuLaunch(Effect)", "Compute" );	
		if ( bUpdateApron ) UpdateApron ( chan );		// update the apron
	}
		
	if ( mbProfile ) PERF_POP();
}

void VolumeGVDB::ActiveRegionSparse(Extents& e, float* srcbuf)
{
	ActivateRegionFromAux(e, AUX_VOXELIZE, T_FLOAT);
}


void VolumeGVDB::DownsampleCPU(Matrix4F xform, Vector3DI in_res, char in_aux, Vector3DI out_res, Vector3DF out_max, char out_aux, Vector3DF inr, Vector3DF outr)
{
	PrepareAux(out_aux, out_res.x*out_res.y*out_res.z, sizeof(float), true, true);
	
	// Determine grid and block dims
	Vector3DI block(8, 8, 8);	
	Vector3DI grid(int(out_res.x / block.x) + 1, int(out_res.y / block.y) + 1, int(out_res.z / block.z) + 1);
	
	// Send transform matrix to cuda
	PrepareAux(AUX_MATRIX4F, 16, sizeof(float), true, true);
	memcpy(mAux[AUX_MATRIX4F].cpu, xform.GetDataF(), 16 * sizeof(float));
	CommitData(mAux[AUX_MATRIX4F]);

	void* args[8] = { &in_res, &mAux[in_aux].gpu, &out_res, &out_max, &mAux[out_aux].gpu, &mAux[AUX_MATRIX4F].gpu, &inr, &outr };
	cudaCheck(cuLaunchKernel(cuFunc[FUNC_DOWNSAMPLE], grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args, NULL), "cuLaunch(Downsample)", "DownsampleCPU");
	
	// Retrieve data back to CPU
	RetrieveData( mAux[out_aux] );
}

void VolumeGVDB::Resample ( uchar chan, Matrix4F xform, Vector3DI in_res, char in_aux, Vector3DF inr, Vector3DF outr )
{
	PrepareVDB ();

	// Determine grid and block dims (must match atlas bricks)	
	Vector3DI block ( 8, 8, 8 );
	Vector3DI res = mPool->getAtlasRes( chan );
	Vector3DI grid ( int(res.x/block.x)+1, int(res.y/block.y)+1, int(res.z/block.z)+1 );	

	// Send transform matrix to cuda
	PrepareAux ( AUX_MATRIX4F, 16, sizeof(float), true, true );
	memcpy ( mAux[AUX_MATRIX4F].cpu, xform.GetDataF(), 16*sizeof(float) );
	CommitData ( mAux[AUX_MATRIX4F] );

	void* args[7] = { &res, &chan, &in_res, &mAux[in_aux].gpu, &mAux[AUX_MATRIX4F].gpu, &inr, &outr };
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_RESAMPLE], grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args, NULL ), "cuLaunch(Resample)", "Resample" );	

}

float VolumeGVDB::getValue ( slong nodeid, Vector3DF pos, float* atlas )
{
	// Recurse to find value
	Node* curr = getNode ( nodeid );	
	Vector3DF p = pos; p -= curr->mPos;
	Vector3DF range = getRange ( curr->mLev );

	if (p.x >= 0 && p.y >=0 && p.z >=0 && p.x < range.x && p.y < range.y && p.z < range.z ) {
		
		// point is inside this node
		p *= getRes ( curr->mLev );
		p /= range;

		if ( isLeaf ( nodeid ) ) {
			
			Vector3DF atlaspos = curr->mValue;
			Vector3DI atlasres = mPool->getAtlasRes( 0 );
			p += atlaspos;

			return atlas [ (int(p.z)*atlasres.y + int(p.y) )*atlasres.x + int(p.x) ];

		} else {
			
			uint32 i = getBitPos ( curr->mLev, p );
			uint32 b = curr->countOn ( i );
			slong childid;			
			if ( curr->isOn ( i ) ) {
				childid = getChildNode ( nodeid, b );
				return getValue ( childid, pos, atlas );
			}			
		}
	} 
	return 0.0;	
}

#define SCAN_BLOCKSIZE		512				// <--- must match cuda_gvdb_particles.cuh header

void VolumeGVDB::PrefixSum ( CUdeviceptr inArray, CUdeviceptr outArray, int numElem )
{
	PrepareAux ( AUX_ARRAY1, SCAN_BLOCKSIZE << 1, sizeof(uint), false );
	PrepareAux ( AUX_SCAN1,  SCAN_BLOCKSIZE << 1, sizeof(uint), false );
	PrepareAux ( AUX_ARRAY2, SCAN_BLOCKSIZE << 1, sizeof(uint), false );
	PrepareAux ( AUX_SCAN2,  SCAN_BLOCKSIZE << 1, sizeof(uint), false );

	int naux = SCAN_BLOCKSIZE << 1;
	int len1 = numElem / naux;
	int blks1 = int( numElem / naux )+1;
	int blks2 = int( blks1 / naux )+1;
	int threads = SCAN_BLOCKSIZE;
	int zon = 1;	
	
	void* argsA[5] = {&inArray, &outArray, &mAux[AUX_ARRAY1].gpu, &numElem, &zon };
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_PREFIXSUM], blks1, 1, 1, threads, 1, 1, 0, NULL, argsA, NULL ), "cuPrefixSumA", "PrefixSum" );

	void* argsB[5] = { &mAux[AUX_ARRAY1].gpu, &mAux[AUX_SCAN1].gpu, &mAux[AUX_ARRAY2].gpu, &len1, &zon };
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_PREFIXSUM], blks2, 1, 1, threads, 1, 1, 0, NULL, argsB, NULL ), "cuPrefixSumB", "PrefixSum" );

	CUdeviceptr nptr = {0};
	void* argsC[5] = { &mAux[AUX_ARRAY2].gpu, &mAux[AUX_SCAN2].gpu, &nptr, &blks2, &zon };
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_PREFIXSUM], 1, 1, 1, threads, 1, 1, 0, NULL, argsC, NULL ), "cuPrefixSumC", "PrefixSum" );		

	void* argsD[3] = { &mAux[AUX_SCAN1].gpu, &mAux[AUX_SCAN2].gpu, &len1 };
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_PREFIXFIXUP], blks2, 1, 1, threads, 1, 1, 0, NULL, argsD, NULL ), "cuPrefixFixupC", "PrefixSum" );
	
	void* argsE[3] = { &outArray, &mAux[AUX_SCAN1].gpu, &numElem };
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_PREFIXFIXUP], blks1, 1, 1, threads, 1, 1, 0, NULL, argsE, NULL ), "cuPrefixFixupB", "PrefixSum" );	
}

void VolumeGVDB::AllocData ( DataPtr& ptr, int cnt, int stride, bool bCPU )
{
	mPool->CreateMemLinear ( ptr, 0x0, stride, cnt, bCPU );
}
void VolumeGVDB::CommitData ( DataPtr ptr )
{
	if ( mbProfile ) PERF_PUSH ( "Commit Data" );	
	mPool->CommitMem ( ptr );
	if ( mbProfile ) PERF_POP ();
}
void VolumeGVDB::CommitData ( DataPtr& dat, int cnt, char* cpubuf, int offs, int stride )
{
	dat.cpu = cpubuf;
	dat.num = cnt;
	dat.max = cnt;
	dat.stride = stride;
	dat.subdim.Set ( offs, stride, 0 );

	if (dat.gpu == 0x0) {
		gprintf( "ERROR: Buffer not allocated on GPU. May be missing call to AllocData.\n");
		gerror();
		return; 
	}

	if ( mbProfile ) PERF_PUSH ( "Commit Data" );	
	mPool->CommitMem ( dat );
	if ( mbProfile ) PERF_POP ();
}
void VolumeGVDB::SetDataCPU ( DataPtr& dat, int cnt, char* cptr, int offs, int stride )
{
	dat.num = cnt;
	dat.max = cnt;
	dat.cpu = cptr;	
	dat.stride = stride;
	dat.subdim.Set ( offs, stride, 0 );
}
void VolumeGVDB::SetDataGPU ( DataPtr& dat, int cnt, CUdeviceptr gptr, int offs, int stride )
{
	dat.num = cnt;
	dat.max = cnt;
	dat.gpu = gptr;
	dat.stride = stride;
	dat.subdim.Set ( offs, stride, 0 );
}

void VolumeGVDB::RetrieveData ( DataPtr ptr )
{
	if ( mbProfile ) PERF_PUSH ( "Retrieve Data" );	
	mPool->RetrieveMem ( ptr );
	if ( mbProfile ) PERF_POP ();
}
char* VolumeGVDB::getDataCPU ( DataPtr ptr, int n, int stride )
{
	return (ptr.cpu + n*stride);
}
void VolumeGVDB::CommitTransferFunc ()
{
	mPool->CreateMemLinear ( mTransferPtr, (char*) mScene->getTransferFunc(), 16384*sizeof(Vector4DF) );
	mVDBInfo.update = true;		// tranfer func pointer may have changed for next render
}

void VolumeGVDB::SetPoints ( DataPtr pntpos, DataPtr clrpos )
{
	mAux[AUX_PNTPOS] = pntpos;
	mAux[AUX_PNTCLR] = clrpos;
}

void VolumeGVDB::SetSupportPoints ( DataPtr pntpos, DataPtr dirpos )
{
	mAux[AUX_PNTPOS] = pntpos;
	mAux[AUX_PNTDIR] = dirpos;
}

void VolumeGVDB::PrepareAux ( int id, int cnt, int stride, bool bZero, bool bCPU )
{
	if ( mAux[id].num < cnt || mAux[id].stride != stride ) {
		mPool->CreateMemLinear ( mAux[id], 0x0, stride, cnt, bCPU );
	}
	if ( bZero ) {
		cudaCheck ( cuMemsetD8 ( mAux[id].gpu, 0, mAux[id].size ), "cuMemsetD8", "PrepareAux" );
	}
}

void VolumeGVDB::InsertPoints ( int num_pnts, Vector3DF trans, bool bPrefix )
{
	if ( mbProfile ) PERF_PUSH ( "InsertPoints");
	
	PrepareVDB ();	

	// Prepare aux arrays
	if ( mbProfile ) PERF_PUSH ( "Prepare Aux");
	int bricks = mPool->getAtlas(0).num;
	PrepareAux ( AUX_PNODE, num_pnts, sizeof(int), false );			// node which each point falls into
	PrepareAux ( AUX_PNDX,  num_pnts, sizeof(int), false );			// index of the point inside that node
	PrepareAux ( AUX_GRIDCNT, bricks, sizeof(int), true );			// number of points in each brick cell
	if ( mbProfile ) PERF_POP();
	
	// Insert particles
	if ( mbProfile ) PERF_PUSH ( "Insert kernel");
	int threads = 512;		
	int pblks = int(num_pnts / threads)+1;
	void* args[8] = { &num_pnts, &mAux[AUX_PNTPOS].gpu, &mAux[AUX_PNTPOS].subdim.x, &mAux[AUX_PNTPOS].stride, &mAux[AUX_PNODE].gpu, &mAux[AUX_PNDX].gpu, &mAux[AUX_GRIDCNT].gpu, &trans.x };
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_INSERT_POINTS], pblks, 1, 1, threads, 1, 1, 0, NULL, args, NULL ), "cuLaunch(INSERT_POINTS)", "InsertPoints" );		
	if ( mbProfile ) PERF_POP();

	if ( bPrefix ) {
		// Prefix sum for brick offsets
		if ( mbProfile ) PERF_PUSH ( "  Prefix sum");
		PrepareAux ( AUX_GRIDOFF, bricks, sizeof(int), false );
		PrefixSum ( mAux[AUX_GRIDCNT].gpu, mAux[AUX_GRIDOFF].gpu, bricks );
		if ( mbProfile ) PERF_POP();

		if ( mbProfile ) PERF_PUSH ( "  Sort points");
		PrepareAux ( AUX_PNTSORT, num_pnts, sizeof(Vector3DF), false );
		void* args[12] = { &num_pnts, &mAux[AUX_PNTPOS].gpu, &mAux[AUX_PNTPOS].subdim.x, &mAux[AUX_PNTPOS].stride, &mAux[AUX_PNODE].gpu, &mAux[AUX_PNDX].gpu, 
									&bricks, &mAux[AUX_GRIDCNT].gpu, &mAux[AUX_GRIDOFF].gpu, &mAux[AUX_PNTSORT].gpu, &trans.x  };
		
		cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_SORT_POINTS], pblks, 1, 1, threads, 1, 1, 0, NULL, args, NULL ), "cuLaunch(SORT_POINTS)", "InsertPoints" );
		
		if ( mbProfile ) PERF_POP();
	}	


	if ( mbProfile ) PERF_POP();	// InsertParticles

}

void VolumeGVDB::InsertSupportPoints ( int num_pnts, float offset, Vector3DF trans, bool bPrefix )
{
	if ( mbProfile ) PERF_PUSH ( "InsertPoints");
	
	PrepareVDB ();	

	// Prepare aux arrays
	if ( mbProfile ) PERF_PUSH ( "Prepare Aux");
	int bricks = mPool->getAtlas(0).num;
	PrepareAux ( AUX_PNODE, num_pnts, sizeof(int), false );			// node which each point falls into
	PrepareAux ( AUX_PNDX,  num_pnts, sizeof(int), false );			// index of the point inside that node
	PrepareAux ( AUX_GRIDCNT, bricks, sizeof(int), true );			// number of points in each brick cell
	if ( mbProfile ) PERF_POP();
	
	// Insert particles
	if ( mbProfile ) PERF_PUSH ( "Insert kernel");
	int threads = 512;		
	int pblks = int(num_pnts / threads)+1;
	void* args[12] = { &num_pnts, &offset,
		&mAux[AUX_PNTPOS].gpu, &mAux[AUX_PNTPOS].subdim.x, &mAux[AUX_PNTPOS].stride, 
		&mAux[AUX_PNODE].gpu, &mAux[AUX_PNDX].gpu, &mAux[AUX_GRIDCNT].gpu, 
		&mAux[AUX_PNTDIR].gpu, &mAux[AUX_PNTDIR].subdim.x, &mAux[AUX_PNTDIR].stride,
		&trans.x };
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_INSERT_SUPPORT_POINTS], pblks, 1, 1, threads, 1, 1, 0, NULL, args, NULL ), "cuLaunch(INSERT_SUPPORT)", "InsertSupportPoints" );		
	if ( mbProfile ) PERF_POP();

	if ( mbProfile ) PERF_POP();	// InsertParticles

	/*---- debugging
	mPool->RetrieveMem ( mNCnt );
	mPool->RetrieveMem ( mNOff );
	for (int n=0; n < bricks; n++ ) {
		gprintf ( " %d: %u %u\n", n, ((uint*) mNCnt.cpu)[n], ((uint*) mNOff.cpu)[n] );
	}*/
}

void VolumeGVDB::GatherPointDensity ( int num_pnts, float radius, int chan )
{
	if ( mbProfile ) PERF_PUSH ("Gather Density");

	// Send VDB Info	
	PrepareVDB ();

	// Gather Point Density

	int subcell = 2;		// number of sub-cells in each brick (voxels in subcell define a warp)

	int subres = (mPool->getAtlasBrickres(chan) - mPool->getAtlas(chan).apron*2) / subcell;
	Vector3DI block ( subres, subres, subres );
	Vector3DI bcnt = mPool->getAtlasRes( chan ) / mPool->getAtlasBrickres( chan );
	Vector3DI res = bcnt * subres * subcell;
	
	Vector3DI grid ( int(res.x/block.x)+1, int(res.y/block.y)+1, int(res.z/block.z)+1 );	
	int bricks = mPool->getAtlas(0).num;	
	void* args[9] = { &res, &chan, &num_pnts, &radius, &mAux[AUX_PNTSORT].gpu,  
		                 &bricks, &mAux[AUX_GRIDCNT].gpu, &mAux[AUX_GRIDOFF].gpu, &subcell };	

	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_GATHER_DENSITY], grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args, NULL ), "cuLaunch(GATHER_DENSITY)", "GatherPointDensity" );

	if ( mbProfile ) PERF_POP ();
}

void VolumeGVDB::ScatterPointDensity ( int num_pnts, float radius, float amp, Vector3DF trans, bool expand, bool avgColor )
{
	uint num_voxels; 

	PrepareVDB ();	

	// Splat particles
	if ( mbProfile ) PERF_PUSH ( "ScatterPointDensity");
	int threads = 256;		
	int pblks = int(num_pnts / threads)+1;	
	
    if (mAux[AUX_PNTCLR].gpu != NULL && avgColor) {
		Vector3DI brickResVec = getRes3DI(0);
		num_voxels = brickResVec.x * brickResVec.y * brickResVec.z * getNumNodes(0);
		if (mbProfile) PERF_PUSH("Prepare Aux");
		PrepareAux(AUX_COLAVG, 4 * num_voxels, sizeof(uint), true);					// node which each point falls into
		if (mbProfile) PERF_POP();
    }
	
	void* args[13] = { &num_pnts, &radius, &amp, &mAux[AUX_PNTPOS].gpu, &mAux[AUX_PNTPOS].subdim.x, &mAux[AUX_PNTPOS].stride, &mAux[AUX_PNTCLR].gpu, &mAux[AUX_PNTCLR].subdim.x, &mAux[AUX_PNTCLR].stride, &mAux[AUX_PNODE].gpu, &trans.x, &expand, &mAux[AUX_COLAVG].gpu };
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_SCATTER_DENSITY], pblks, 1, 1, threads, 1, 1, 0, NULL, args, NULL ), "cuLaunch(SPLAT)", "SplatPoints" );

	if (mAux[AUX_PNTCLR].gpu != NULL && avgColor) {
		int threads_avgcol = 256;
		int pblks_avgcol = int(num_voxels / threads_avgcol) + 1;
		void* args_avgcol[2] = { &num_voxels, &mAux[AUX_COLAVG].gpu };
		cudaCheck( cuLaunchKernel (cuFunc[FUNC_SCATTER_AVG_COL], pblks_avgcol, 1, 1, threads_avgcol, 1, 1, 0, NULL, args_avgcol, NULL), "cuLaunch(SPLAT_POINT_AVG_COL)", "SplatPoints" );
	}

	if ( mbProfile ) PERF_POP ();	
}

void VolumeGVDB::AddSupportVoxel ( int num_pnts, float radius, float offset, float amp, Vector3DF trans, bool expand, bool avgColor )
{
	PrepareVDB ();	

	// Splat particles
	if ( mbProfile ) PERF_PUSH ( "AddSupportVoxel");
	int threads = 256;		
	int pblks = int(num_pnts / threads)+1;	
	
	void* args[14] = { &num_pnts, &radius, &offset, &amp, 
		&mAux[AUX_PNTPOS].gpu, &mAux[AUX_PNTPOS].subdim.x, &mAux[AUX_PNTPOS].stride, 
		&mAux[AUX_PNTDIR].gpu, &mAux[AUX_PNTDIR].subdim.x, &mAux[AUX_PNTDIR].stride,
		&mAux[AUX_PNODE].gpu, &trans.x, &expand, &mAux[AUX_COLAVG].gpu };
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_ADD_SUPPORT_VOXEL], pblks, 1, 1, threads, 1, 1, 0, NULL, args, NULL ), "cuAddSupportVoxel", "AddSupportVoxel" );

	if ( mbProfile ) PERF_POP ();	
}

// Node queries
int	VolumeGVDB::getNumNodes ( int lev )				{ return mPool->getPoolCnt(0, lev); }
Node* VolumeGVDB::getNodeAtLevel ( int n, int lev )	{ return (Node*) (mPool->PoolData( 0, lev, n )); }
Vector3DF VolumeGVDB::getWorldMin ( Node* node )	{ return Vector3DF(node->mPos) * mVoxsize; }
Vector3DF VolumeGVDB::getWorldMax ( Node* node )	{ return Vector3DF(node->mPos) * mVoxsize + getCover(node->mLev); }

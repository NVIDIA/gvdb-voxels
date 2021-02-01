//-----------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2016 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
// 
// Version 1.0: Rama Hoetzlein, 5/1/2017
// Version 1.1: Rama Hoetzlein, 3/25/2018
//-----------------------------------------------------------------------------

#include "gvdb_allocator.h"
#include "gvdb_volume_3D.h"
#include "gvdb_volume_gvdb.h"
#include "gvdb_render.h"
#include "gvdb_node.h"
#include "app_perf.h"
#include "string_helper.h"

#include "gvdb_cutils.cuh"

#include <algorithm>
#include <iostream>
#include <float.h>
#include <fstream>

#if !defined(_WIN32)
#	include <GL/glx.h>
#endif

using namespace nvdb;

#define MAJOR_VERSION		1
#define MINOR_VERSION		11

// Version history
// GVDB 1.0  - First release. GTC'2017
// GVDB 1.0 - Incremental fixes, Beta pre-release. 
// GVDB 1.1 - Second release. GTC'2018
// GVDB 1.1.1  - Bug Fixes


#define PUSH_CTX		cuCtxPushCurrent(mContext);
#define POP_CTX			CUcontext pctx; cuCtxPopCurrent(&pctx);
#define	MRES	2048

#ifndef GVDB_CPU_VERSION
#define GVDB_CPU_VERSION  0
#endif

#ifdef BUILD_OPENVDB
	#include <openvdb/tools/ValueTransformer.h>
#endif

Vector3DI VolumeGVDB::getVersion() {
	return Vector3DI(MAJOR_VERSION, MINOR_VERSION, 0);
}

void VolumeGVDB::TimerStart ()		{ PERF_START(); }
float VolumeGVDB::TimerStop ()		{ return PERF_STOP(); }

VolumeGVDB::VolumeGVDB ()
{
	TimeX start;
	
	mDevice = NULL;
	mContext = NULL;
	mStream = NULL;
	mPool = nullptr;
	mScene = nullptr;
	mV3D = 0x0;
	mAtlasResize.Set ( 0, 20, 0 );
	mEpsilon = 0.001f;				// default epsilon
	mMaxIter = 256;					// default max iter
	mApron = 1;						// default apron
	mbDebug = false;

	// identity transform
	SetTransform(Vector3DF(0, 0, 0), Vector3DF(1, 1, 1), Vector3DF(0, 0, 0), Vector3DF(0, 0, 0));

	mRoot = ID_UNDEFL;
	mbUseGLAtlas = false;

	mVDBInfo.update = true;	
	mVDBInfo.clr_chan = CHAN_UNDEF;

	mbProfile = false;
	mbVerbose = false;

	mDummyFrameBuffer = -1;

	for (int n=0; n < 5; n++ ) cuModule[n] = (CUmodule) -1;
	for (int n=0; n < MAX_FUNC; n++ ) cuFunc[n] = (CUfunction) -1;
	for (int n=0; n < MAX_CHANNEL; n++ ) { mVDBInfo.volIn[n] = ID_UNDEFL; mVDBInfo.volOut[n] = ID_UNDEFL; }
	for (int n=0; n < MAX_AUX; n++) { 
		mAux[n].lastEle = 0; mAux[n].usedNum = 0; mAux[n].max = 0; mAux[n].size = 0; mAux[n].stride = 0;
		mAux[n].cpu = 0; mAux[n].gpu = 0; 
	}

	mRendName[SHADE_VOXEL] = "SHADE_VOXEL";
	mRendName[SHADE_SECTION2D] = "SHADE_SECTION2D";
	mRendName[SHADE_SECTION3D] = "SHADE_SECTION3D";
	mRendName[SHADE_EMPTYSKIP] = "SHADE_EMPTYSKIP";
	mRendName[SHADE_TRILINEAR] = "SHADE_TRILINEAR";
	mRendName[SHADE_TRICUBIC] = "SHADE_TRICUBIC";
	mRendName[SHADE_LEVELSET] = "SHADE_LEVELSET";
	mRendName[SHADE_VOLUME] = "SHADE_VOLUME";

	mAuxName[AUX_PNTPOS] = "PNTPOS";
	mAuxName[AUX_PNTCLR] = "PNTCLR"; 
	mAuxName[AUX_PNODE] = "PNODE";
	mAuxName[AUX_PNDX] = "PNDX";
	mAuxName[AUX_GRIDCNT] = "GRIDCNT";
	mAuxName[AUX_GRIDOFF] = "GRIDOFF";
	mAuxName[AUX_ARRAY1] = "ARRAY1";
	mAuxName[AUX_SCAN1] = "SCAN1";
	mAuxName[AUX_ARRAY2] = "ARRAY2";
	mAuxName[AUX_SCAN2] = "SCAN2";
	mAuxName[AUX_COLAVG] = "COLAVG";
	mAuxName[AUX_VOXELIZE] = "VOXELIZE";
	mAuxName[AUX_VERTEX_BUF] = "VERTEX_BUF";
	mAuxName[AUX_ELEM_BUF] = "ELEM_BUF";
	mAuxName[AUX_TRI_BUF] = "TRI_BUF";
	mAuxName[AUX_PNTSORT] = "PNTSORT";
	mAuxName[AUX_PNTDIR] = "PNTDIR";
	mAuxName[AUX_DATA2D] = "DATA2D";
	mAuxName[AUX_DATA3D] = "DATA3D";
	mAuxName[AUX_MATRIX4F] = "MATRIX4F";

	mAuxName[AUX_PBRICKDX] = "PBRICKDX"; 
	mAuxName[AUX_ACTIVBRICKCNT] = "ACTIVEBRICKCNT";
	mAuxName[AUX_BRICK_LEVXYZ] = "BRICK_LEVXYZ";
	mAuxName[AUX_RANGE_RES] = "RANGE_RES";
	mAuxName[AUX_MARKER] = "MARKER";
	mAuxName[AUX_RADIX_PRESCAN] = "RADIX_PRESCAN";
	mAuxName[AUX_SORTED_LEVXYZ] = "SORTED_LEVXYZ";
	mAuxName[AUX_TMP] = "TMP";
	mAuxName[AUX_UNIQUE_CNT] = "UNIQUE_CNT";
	mAuxName[AUX_MARKER_PRESUM] = "MARKER_PRESUM";
	mAuxName[AUX_UNIQUE_LEVXYZ] = "UNIQUE_LEVXYZ";
	mAuxName[AUX_LEVEL_CNT] = "LEVEL_CNT";

	mAuxName[AUX_EXTRA_BRICK_CNT] = "EXTRA_BRICK_CNT";
	mAuxName[AUX_NODE_MARKER] = "NODE_MARKER";
	mAuxName[AUX_PNTVEL] = "PNTVEL";
	mAuxName[AUX_DIV] = "DIV";

	mAuxName[AUX_SUBCELL_CNT] = "SUBCELL_CNT";
	mAuxName[AUX_SUBCELL_PREFIXSUM] = "SUBCELL_PREFIXSUM";
	mAuxName[AUX_SUBCELL_PNTS] = "SUBCELL_PNTS";
	mAuxName[AUX_SUBCELL_POS] = "SUBCELL_POS";
	mAuxName[AUX_SUBCELL_PNT_POS] = "SUBCELL_PNT_POS";
	mAuxName[AUX_SUBCELL_PNT_VEL] = "SUBCELL_PNT_VEL";
	mAuxName[AUX_SUBCELL_PNT_CLR] = "SUBCELL_PNT_CLR";
	mAuxName[AUX_BOUNDING_BOX] = "BOUNDING_BOX";
	mAuxName[AUX_WORLD_POS_X] = "WORLD_POS_X";
	mAuxName[AUX_WORLD_POS_Y] = "WORLD_POS_Y";
	mAuxName[AUX_WORLD_POS_Z] = "WORLD_POS_Z";

	mAuxName[AUX_VOLUME] = "VOLUME";
	mAuxName[AUX_CG] = "CG";
	mAuxName[AUX_INNER_PRODUCT] = "INNER_PRODUCT";
	mAuxName[AUX_TEXTURE_MAX] = "TEXTURE_MAX";
	mAuxName[AUX_TEXTURE_MAX_TMP] = "TEXTURE_MAX_TMP";
	mAuxName[AUX_TEST] = "TEST";
	mAuxName[AUX_TEST_1] = "TEST_1";
	mAuxName[AUX_OUT1] = "OUT1";
	mAuxName[AUX_OUT2] = "OUT2";
	mAuxName[AUX_SUBCELL_MAPPING] = "SUBCELL_MAPPING";
	mAuxName[AUX_SUBCELL_FLAG] = "SUBCELL_FLAG";
	mAuxName[AUX_SUBCELL_NID] = "SUBCELL_NID";
	mAuxName[AUX_SUBCELL_OBS_NID] = "SUBCELL_OBS_ND";

}

VolumeGVDB::~VolumeGVDB()
{
	// Free the atlas
	DestroyChannels();

	// Free all auxiliary memory
	CleanAux();

	// Delete CUDA objects
	if (cuVDBInfo != 0) {
		cudaCheck(cuMemFree(cuVDBInfo), "VolumeGVDB", "~VolumeGVDB", "cuMemFree", "cuVDBInfo", mbDebug);
	}

	for (int n = 0; n < sizeof(cuModule)/sizeof(cuModule[0]); n++) {
		if (cuModule[n] != (CUmodule)(-1)) {
			cudaCheck(cuModuleUnload(cuModule[n]), "VolumeGVDB", "~VolumeGVDB", "cuModuleUnload", "cuModule[n]", mbDebug);
		}
	}

	if (mV3D != 0) {
		delete mV3D;
		mV3D = 0;
	}

	// Free mTransferPtr
	if (mPool != nullptr) {
		mPool->FreeMemLinear(mTransferPtr);
		// Tell mScene that its transfer function has been destroyed to avoid
		// free-after-free
		if (mScene != nullptr) {
			mScene->mTransferFunc = nullptr;
		}
	}

	if (mScene != nullptr) {
		delete mScene;
		mScene = 0;
	}

	// VolumeBase destructor called here
}

void VolumeGVDB::SetProfile ( bool bCPU, bool bGPU ) 
{
	mbProfile = bCPU; 	
	PERF_INIT ( 64, bCPU, bGPU, bCPU, 0, "" );		
}
void VolumeGVDB::SetDebug ( bool d )
{
	mbDebug = d;
	if (mPool != 0x0) mPool->SetDebug(d);
}

// Loads a CUDA function into memory from ptx file
void VolumeGVDB::LoadFunction ( int fid, std::string func, int mid, std::string ptx )
{
	char cptx[512];		strcpy ( cptx, ptx.c_str() );
	char cfn[512];		strcpy ( cfn, func.c_str() );

	if ( cuModule[mid] == (CUmodule) -1 ) 
		cudaCheck ( cuModuleLoad ( &cuModule[mid], cptx ), "VolumeGVDB", "LoadFunction", "cuModuleLoad", cptx, mbDebug);
	if ( cuFunc[fid] == (CUfunction) -1 )
		cudaCheck ( cuModuleGetFunction ( &cuFunc[fid], cuModule[mid], cfn ), "VolumeGVDB", "LoadFunction", "cuModuleGetFunction", cfn, mbDebug);
}

// Set the current CUDA device, load all GVDB kernels
void VolumeGVDB::SetCudaDevice ( int devid, CUcontext ctx )
{
	size_t len = 0;

	mDevSelect = devid;	
	StartCuda(devid, ctx, mDevice, mContext, &mStream, mbVerbose );

	PUSH_CTX

	//--- Load cuda kernels
	// Raytracing
	LoadFunction ( FUNC_RAYDEEP,			"gvdbRayDeep",					MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_RAYVOXEL,			"gvdbRaySurfaceVoxel",			MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_RAYTRILINEAR,		"gvdbRaySurfaceTrilinear",		MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_RAYTRICUBIC,		"gvdbRaySurfaceTricubic",		MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_RAYSURFACE_DEPTH,	"gvdbRaySurfaceDepth",			MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_RAYLEVELSET,		"gvdbRayLevelSet",				MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_EMPTYSKIP,			"gvdbRayEmptySkip",				MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_SECTION2D,			"gvdbSection2D",				MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_SECTION3D,			"gvdbSection3D",				MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );	
	LoadFunction ( FUNC_RAYTRACE,			"gvdbRaytrace",					MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	
	// Sorting / Points / Triangles
	LoadFunction ( FUNC_PREFIXSUM,			"prefixSum",					MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_PREFIXFIXUP,		"prefixFixup",					MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_INSERT_POINTS,		"gvdbInsertPoints",				MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_SORT_POINTS,		"gvdbSortPoints",				MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );	
	LoadFunction ( FUNC_SCATTER_DENSITY,	"gvdbScatterPointDensity",		MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_SCATTER_AVG_COL,	"gvdbScatterPointAvgCol",		MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_INSERT_TRIS,		"gvdbInsertTriangles",			MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_SORT_TRIS,			"gvdbSortTriangles",			MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_VOXELIZE,			"gvdbVoxelize",					MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );	
	LoadFunction ( FUNC_RESAMPLE,			"gvdbResample",					MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_REDUCTION,			"gvdbReduction",				MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_DOWNSAMPLE,			"gvdbDownsample",				MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_SCALE_PNT_POS,		"gvdbScalePntPos",				MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_CONV_AND_XFORM,		"gvdbConvAndTransform",			MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );

	LoadFunction ( FUNC_ADD_SUPPORT_VOXEL,	"gvdbAddSupportVoxel",			MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_INSERT_SUPPORT_POINTS, "gvdbInsertSupportPoints",	MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );

	// Topology
	LoadFunction ( FUNC_FIND_ACTIV_BRICKS,	"gvdbFindActivBricks",			MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );	
	LoadFunction ( FUNC_BITONIC_SORT,		"gvdbBitonicSort",				MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );	
	LoadFunction ( FUNC_CALC_BRICK_ID,		"gvdbCalcBrickId",				MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );	
	LoadFunction ( FUNC_FIND_UNIQUE,		"gvdbFindUnique",				MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_COMPACT_UNIQUE,		"gvdbCompactUnique",			MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_LINK_BRICKS,		"gvdbLinkBricks",				MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );

	// Incremental Topology
	LoadFunction ( FUNC_CALC_EXTRA_BRICK_ID,"gvdbCalcExtraBrickId",			MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );	

	LoadFunction ( FUNC_CALC_INCRE_BRICK_ID,"gvdbCalcIncreBrickId",			MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );	
	LoadFunction ( FUNC_CALC_INCRE_EXTRA_BRICK_ID,"gvdbCalcIncreExtraBrickId",			MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );	

	LoadFunction ( FUNC_DELINK_LEAF_BRICKS,	"gvdbDelinkLeafBricks",			MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_DELINK_BRICKS,		"gvdbDelinkBricks",				MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_MARK_LEAF_NODE,		"gvdbMarkLeafNode",				MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );

	// Gathering
	LoadFunction ( FUNC_COUNT_SUBCELL,		"gvdbCountSubcell",				MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );	
	LoadFunction ( FUNC_INSERT_SUBCELL,		"gvdbInsertSubcell",			MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );	
	LoadFunction ( FUNC_INSERT_SUBCELL_FP16,"gvdbInsertSubcell_fp16",		MODL_PRIMARY, CUDA_GVDB_MODULE_PTX);
	LoadFunction ( FUNC_GATHER_DENSITY,		"gvdbGatherDensity",			MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );	
	LoadFunction ( FUNC_GATHER_LEVELSET,	"gvdbGatherLevelSet",			MODL_PRIMARY, CUDA_GVDB_MODULE_PTX);
	LoadFunction ( FUNC_GATHER_LEVELSET_FP16, "gvdbGatherLevelSet_fp16",    MODL_PRIMARY, CUDA_GVDB_MODULE_PTX);
	
	LoadFunction ( FUNC_CALC_SUBCELL_POS,	"gvdbCalcSubcellPos",			MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );	
	LoadFunction ( FUNC_MAP_EXTRA_GVDB,		"gvdbMapExtraGVDB",			    MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );	
	LoadFunction ( FUNC_SPLIT_POS,			"gvdbSplitPos",					MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_SET_FLAG_SUBCELL,	"gvdbSetFlagSubcell",			MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );	

	LoadFunction ( FUNC_READ_GRID_VEL,		"gvdbReadGridVel",				MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );	
	LoadFunction ( FUNC_CHECK_VAL,			"gvdbCheckVal",					MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );	
	
	// Apron Updates
	LoadFunction ( FUNC_UPDATEAPRON_F,		"gvdbUpdateApronF",				MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_UPDATEAPRON_F4,		"gvdbUpdateApronF4",			MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_UPDATEAPRON_C,		"gvdbUpdateApronC",				MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_UPDATEAPRON_C4,		"gvdbUpdateApronC4",			MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_UPDATEAPRONFACES_F, "gvdbUpdateApronFacesF",		MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	
	// Operators
	LoadFunction ( FUNC_FILL_F,				"gvdbOpFillF",					MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_FILL_C,				"gvdbOpFillC",					MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_FILL_C4,			"gvdbOpFillC4",					MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_SMOOTH,				"gvdbOpSmooth",					MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );
	LoadFunction ( FUNC_NOISE,				"gvdbOpNoise",					MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );	
	LoadFunction ( FUNC_CLR_EXPAND,			"gvdbOpClrExpand",				MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );	
	LoadFunction ( FUNC_EXPANDC,			"gvdbOpExpandC",				MODL_PRIMARY, CUDA_GVDB_MODULE_PTX );	

	SetModule ( cuModule[MODL_PRIMARY] );	

	POP_CTX
}

// Reset to default module
void VolumeGVDB::SetModule ()
{
	SetModule ( cuModule[MODL_PRIMARY] );	
}

// Set to a user-defined module (application)
void VolumeGVDB::SetModule ( CUmodule module )
{
	PUSH_CTX
	
	cudaCheck ( cuCtxSynchronize (), "VolumeGVDB", "SetModule", "cuCtxSynchronize", "", mbDebug);

	ClearAtlasAccess ();

	size_t len = 0;
	cudaCheck ( cuModuleGetGlobal ( &cuScnInfo, &len,	module, "scn" ),	"VolumeGVDB", "SetModule", "cuModuleGetGlobal", "cuScnInfo", mbDebug);

	cudaCheck ( cuModuleGetGlobal ( &cuXform,  &len,	module, "cxform" ), "VolumeGVDB", "SetModule", "cuModuleGetGlobal", "cuXform", mbDebug);
	cudaCheck ( cuModuleGetGlobal ( &cuDebug,  &len,	module, "cdebug" ), "VolumeGVDB", "SetModule", "cuModuleGetGlobal", "cuDebug", mbDebug);

	SetupAtlasAccess ();

	POP_CTX
}

// Takes `inbuf`, a buffer of `cnt` Vector3DF objects (i.e. `3*cnt` floats),
// and writes the length of each vector into `outbuf`, a buffer of `cnt` floats.
// Sets vmin and vmax to the minimum and maximum lengths.
float* ConvertToScalar ( int cnt, float* inbuf, float* outbuf, float& vmin, float& vmax )
{
	float* outv = outbuf;
	Vector3DF* vec = (Vector3DF*) inbuf;	
	float val;	
	for ( int n=0; n < cnt; n++ ) {
		val = static_cast<float>(vec->Length());
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
	PUSH_CTX

	char buf[2048];
	strcpy (buf, fname.c_str() );
	FILE* fp = fopen ( buf, "rt" );
	std::string lin, word;
	bool bAscii = false;	
	float f;
	float* vox = 0x0;
	uint64 cnt = 0, max_cnt = 0;
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
			word = strSplit ( lin, " \n" ); strIsNum(word, f ); res.x = static_cast<int>(f-1);
			word = strSplit ( lin, " \n" ); strIsNum(word, f ); res.y = static_cast<int>(f-1);
			word = strSplit ( lin, " \n" ); strIsNum(word, f ); res.z = static_cast<int>(f-1);
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
			word = strSplit ( lin, " \n" ); strIsNum(word, f); max_cnt = static_cast<int>(f);
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
			cnt = static_cast<uint64>(vox - (float*) mAux[AUX_DATA3D].cpu);
			if ( cnt >= max_cnt) status = 'D';		// done
		}
	}
	verbosef ( "  Values Read: %d\n", cnt );

	// Commit data to GPU
	CommitData ( mAux[AUX_DATA3D] );

	POP_CTX

	return true;
}

void VolumeGVDB::ConvertBitmaskToNonBitmask( int levs )
{
	Node* node;
	uint64* clist;
	uint64 childid;
//	uint32 cnt;
	uint32 ndx;
	uint64 ccnt, cmax, btchk;

	// Activate all bricks (assuming earlier file did not use flags)
	for (int n = 0; n < getNumTotalNodes(0); n++) {
		node = getNode(0, 0, n);
		node->mFlags = 1;
	}

	// Convert upper nodes to non-bitmask
	for (int lv = 1; lv < levs; lv++) {
		for (int n = 0; n < getNumTotalNodes(lv); n++) {
			node = getNode(0, lv, n);			
			if (node->mChildList == ID_UNDEFL) {
				gprintf("ERROR: ConvertBitmaskToNonBitmask, child list is null.\n");
				gerror();
			}			
			// number of children
			ccnt = getNumChild( node );
			cmax = getVoxCnt( node->mLev );
			clist = mPool->PoolData64( node->mChildList );
			
			// pad remainder of child list with ID_UNDEFL			
			memset(clist + ccnt, 0xFF, sizeof(uint64)*(cmax - ccnt));

			// move children into index locations			
			for (int64_t j = static_cast<int64_t>(ccnt-1); j >=0; j--) {
				ndx = countToIndex(node, j);
				btchk = countOn(node, ndx);
				if (btchk != j) {
					gprintf("ERROR: countToIndex error.\n");
					gerror();
				}
				childid = *(clist + j);		// get child
				*(clist + ndx) = childid;	// put child
				*(clist + j) = ID_UNDEF64;
			}			
		}
	}
}



// Load a VBX file
bool VolumeGVDB::LoadVBX(const std::string fname, int force_maj, int force_min)
{
	// See GVDB_FILESPEC.txt for the specification of the VBX file format.
	PUSH_CTX

	FILE* fp = fopen(fname.c_str(), "rb");
	if (fp == 0x0) {
		gprintf("Error: Unable to open file %s\n", fname.c_str());
		return false;
	}

	PERF_PUSH("Read VBX");

	// Read VBX config
	uchar major, minor;
	int num_grids;
	const int grid_name_len = 256; // Length of grid_name
	char grid_name[grid_name_len];
	char grid_components;
	char grid_dtype;
	char grid_compress;
	char grid_topotype;
	int  grid_reuse;
	char grid_layout;

	int leafcnt;
	int num_chan;
	Vector3DI leafdim;
	int apron;
	Vector3DI axiscnt;
	Vector3DI axisres;
	Vector3DF voxelsize_deprecated;
	slong atlas_sz;

	int levels;
	uint64 root;
	Vector3DI range[MAXLEV];
	int ld[MAXLEV], res[MAXLEV];
	int cnt0[MAXLEV], cnt1[MAXLEV];
	int width0[MAXLEV]{}, width1[MAXLEV];

	std::vector<uint64> grid_offs;

	//--- VBX file header
	fread (&major, sizeof(uchar), 1, fp);					// major version
	fread (&minor, sizeof(uchar), 1, fp);					// minor version

	if (force_maj > 0 || force_min > 0) {
		major = force_maj;
		minor = force_min;
	}

	verbosef("LoadVBX: %s (ver %d.%d)\n", fname.c_str(), major, minor );
	verbosef("Sizes: char %d, int %d, u64 %d, float %d\n", sizeof(char), sizeof(int), sizeof(uint64), sizeof(float));

	if ((major == 1 && minor >= 11) || major > 1) {
		// GVDB >=1.11+ saves grid transforms (GVDB 1.1 and earlier do not)
		fread( &mPretrans, sizeof(float), 3, fp);
		fread( &mAngs, sizeof(float), 3, fp);
		fread( &mScale, sizeof(float), 3, fp);
		fread( &mTrans, sizeof(float), 3, fp);
		SetTransform(mPretrans, mScale, mAngs, mTrans);		// set grid transform
	}

	fread ( &num_grids, sizeof(int), 1, fp );				// number of grids

	// bitmask info
	uchar use_masks = 0; // Whether the read volume should use bitmasks
	#ifdef USE_BITMASKS
		use_masks = 1;
	#endif
	uchar read_masks = 0; // Whether the VBX file uses bitmasks
	if (major >= 2) {
		// GVDB >= 2, bitmasks optional 
		fread(&read_masks, sizeof(uchar), 1, fp);
	}
	else if (major == 1 && minor == 0) {
		// GVDB 1.0 always uses bitmasks
		read_masks = 1;
	}
	
	//--- grid offset table
	for (int n=0; n < num_grids; n++ ) {
		grid_offs.push_back(0);
		fread ( &grid_offs[n], sizeof(uint64), 1, fp );		// grid offsets
	}

	for (int n = 0; n < num_grids; n++) {

		//---- grid header
		fread(&grid_name, 256, 1, fp);					// grid name
		fread(&grid_dtype, sizeof(uchar), 1, fp);		// grid data type
		fread(&grid_components, sizeof(uchar), 1, fp);	// grid components
		fread(&grid_compress, sizeof(uchar), 1, fp);	// grid compression (0=none, 1=blosc, 2=..)
		fread(&voxelsize_deprecated, sizeof(float), 3, fp);	// voxel size
		fread(&leafcnt, sizeof(int), 1, fp);			// total brick count
		fread(&leafdim.x, sizeof(int), 3, fp);			// brick dimensions
		fread(&apron, sizeof(int), 1, fp);				// brick apron
		fread(&num_chan, sizeof(int), 1, fp);			// number of channels
		fread(&atlas_sz, sizeof(uint64), 1, fp);		// total atlas size (all channels)
		fread(&grid_topotype, sizeof(uchar), 1, fp);	// topology type? (0=none, 1=reuse, 2=gvdb, 3=..)
		fread(&grid_reuse, sizeof(int), 1, fp);			// topology reuse
		fread(&grid_layout, sizeof(uchar), 1, fp);		// brick layout? (0=atlas, 1=brick)
		fread(&axiscnt.x, sizeof(int), 3, fp);			// atlas brick count
		fread(&axisres.x, sizeof(int), 3, fp);			// atlas res

		//---- topology section
		fread(&levels, sizeof(int), 1, fp);				// num levels
		fread(&root, sizeof(uint64), 1, fp);			// root id
		for (int n = 0; n < levels; n++) {
			fread(&ld[n], sizeof(int), 1, fp);
			fread(&res[n], sizeof(int), 1, fp);
			fread(&range[n].x, sizeof(int), 1, fp);
			fread(&range[n].y, sizeof(int), 1, fp);
			fread(&range[n].z, sizeof(int), 1, fp);
			fread(&cnt0[n], sizeof(int), 1, fp);
			fread(&width0[n], sizeof(int), 1, fp);
			fread(&cnt1[n], sizeof(int), 1, fp);
			fread(&width1[n], sizeof(int), 1, fp);
		}
		if (width0[0] != sizeof(nvdb::Node)) {
			gprintf("ERROR: VBX file contains nodes incompatible with current gvdb_library.\n");
			gprintf("       Size in file: %d,  Size in library: %d\n", width0[0], sizeof(nvdb::Node));
			gerror();
		}

		// Initialize GVDB
		Configure ( levels, ld, cnt0, (read_masks==1) );

		mRoot = root;		// must be set after initialize

		// Read topology
		for (int n=0; n < levels; n++ ) {
			mPool->PoolRead(fp, 0, n, cnt0[n], width0[n]);
		}
		for (int n = 0; n < levels; n++) {
			mPool->PoolRead(fp, 1, n, cnt1[n], width1[n]);
		}

		if (read_masks==1 && use_masks==0) {
			// Convert bitmasks to non-bitmasks
			ConvertBitmaskToNonBitmask( levels );
		}

		FinishTopology ();

		// Atlas section
		DestroyChannels ();
		
		// Read atlas into GPU slice-by-slice to conserve CPU and GPU mem
		for (int chan = 0 ; chan < num_chan; chan++ ) {
			int chan_type, chan_stride;
			fread ( &chan_type, sizeof(int), 1, fp );
			fread ( &chan_stride, sizeof(int), 1, fp );

			AddChannel ( chan, chan_type, apron, F_LINEAR, F_BORDER, axiscnt );		// provide axiscnt

			mPool->AtlasSetNum ( chan, cnt0[0] );		// assumes atlas contains all bricks (all are resident)

			DataPtr slice;
			mPool->CreateMemLinear ( slice, 0x0, chan_stride, axisres.x*axisres.y, true );
			for (int z = 0; z < axisres.z; z++ ) {
				fread ( slice.cpu, slice.size, 1, fp );
				mPool->AtlasWriteSlice ( chan, z, static_cast<int>(slice.size),
					slice.gpu, (uchar*) slice.cpu );		// transfer from GPU, directly into CPU atlas				
			}
			mPool->FreeMemLinear ( slice );
		}
		UpdateAtlas ();
	}

	PERF_POP ();

	POP_CTX

	return true;
}

// Set the current color channel for rendering
void VolumeGVDB::SetColorChannel ( uchar chan )
{
	mVDBInfo.clr_chan = chan;
	mVDBInfo.update = true;	
}

void VolumeGVDB::AtlasRetrieveBrickXYZ(uchar channel, Vector3DI minimumCorner, DataPtr& dest) {
	mPool->AtlasRetrieveTexXYZ(channel, minimumCorner, dest);
}

// Clear device access to atlases
void VolumeGVDB::ClearAtlasAccess ()
{
	if ( mPool==0x0 ) return;

	PUSH_CTX

	int num_chan = mPool->getNumAtlas();
	for (int chan=0; chan < num_chan; chan++ ) {
		if ( mVDBInfo.volIn[chan] != ID_UNDEFL ) {
			cudaCheck ( cuTexObjectDestroy (mVDBInfo.volIn[chan] ), "VolumeGVDB", "ClearAtlasAccess", "cuTexObjectDestroy", "volIn", mbDebug);
			mVDBInfo.volIn[chan] = ID_UNDEFL;
		}
		if (mVDBInfo.volOut[chan] != ID_UNDEFL ) {
			cudaCheck ( cuSurfObjectDestroy (mVDBInfo.volOut[chan] ), "VolumeGVDB", "ClearAtlasAccess", "cuSurfObjectDestroy", "volOut", mbDebug);
			mVDBInfo.volIn[chan] = ID_UNDEFL;
		}
	}
	cudaCheck ( cuCtxSynchronize (), "VolumeGVDB", "ClearAtlasAccess", "cuCtxSynchronize", "", mbDebug);

	POP_CTX
}

// Setup device access to atlases
void VolumeGVDB::SetupAtlasAccess ()
{	
	if ( mPool == 0x0 ) return;
	if ( mPool->getNumAtlas() == 0 ) return;
	
	PUSH_CTX

	DataPtr atlas;

	//-- Texture Access using CUDA Bindless
	int num_chan = mPool->getNumAtlas();
	char chanmsg[256];
	for (int chan=0; chan < num_chan; chan++ ) {	
		
		atlas = mPool->getAtlas(chan );		
		sprintf( chanmsg, "chan %d", chan);

		CUDA_RESOURCE_DESC resDesc;
		memset ( &resDesc, 0, sizeof(resDesc) );
		resDesc.resType = CU_RESOURCE_TYPE_ARRAY;
		
		if ( atlas.grsc != 0x0 ) {
			cudaCheck ( cuGraphicsMapResources(1, &atlas.grsc, mStream), "VolumeGVDB", "SetupAtlasAccess", "cuGraphicsMapResource", chanmsg, mbDebug);
			cudaCheck ( cuGraphicsSubResourceGetMappedArray ( &atlas.garray, atlas.grsc, 0, 0 ), "VolumeGVDB", "SetupAtlasAccess", "cuGraphicsSubResourceGetMappedArray", chanmsg, mbDebug);
		}		

		resDesc.res.array.hArray = atlas.garray;
		resDesc.flags = 0;		

		CUDA_TEXTURE_DESC texDesc;
		memset ( &texDesc, 0, sizeof(texDesc) );				

		// filter mode
		switch ( atlas.filter ) {
		case F_POINT:	texDesc.filterMode = CU_TR_FILTER_MODE_POINT;break;
		case F_LINEAR:	texDesc.filterMode = CU_TR_FILTER_MODE_LINEAR; break;
		};
		// read mode
		switch ( atlas.type ) {
		case T_FLOAT: case T_FLOAT3: case T_FLOAT4:		
			texDesc.flags = 0;			
			break;
		case T_UCHAR: case T_UCHAR3: case T_UCHAR4:	case T_INT: case T_INT3: case T_INT4:			
			texDesc.flags = CU_TRSF_READ_AS_INTEGER;		
			break;
		};
		// border mode
		switch ( atlas.border ) {
		case F_BORDER:
			texDesc.addressMode[0] = CU_TR_ADDRESS_MODE_BORDER;
			texDesc.addressMode[1] = CU_TR_ADDRESS_MODE_BORDER;
			texDesc.addressMode[2] = CU_TR_ADDRESS_MODE_BORDER;		
			break;
		case F_CLAMP:
			texDesc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
			texDesc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
			texDesc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
			break;		
		case F_WRAP:
			texDesc.addressMode[0] = CU_TR_ADDRESS_MODE_WRAP;
			texDesc.addressMode[1] = CU_TR_ADDRESS_MODE_WRAP;
			texDesc.addressMode[2] = CU_TR_ADDRESS_MODE_WRAP;
			break;
		};
		if ( mVDBInfo.volIn[chan] != ID_UNDEFL ) {
			cudaCheck ( cuTexObjectDestroy ( mVDBInfo.volIn[chan] ), "VolumeGVDB", "SetupAtlasAccess", "cuTexObjectDestroy", chanmsg, mbDebug);
			mVDBInfo.volIn[ chan ] = ID_UNDEFL;
		}
		if ( mVDBInfo.volOut[chan] != ID_UNDEFL ) {
			cudaCheck ( cuSurfObjectDestroy ( mVDBInfo.volOut[chan] ), "VolumeGVDB", "SetupAtlasAccess", "cuSurfObjectDestroy", chanmsg, mbDebug);
			mVDBInfo.volOut[ chan ] = ID_UNDEFL;
		}
						
		cudaCheck ( cuTexObjectCreate ( &mVDBInfo.volIn[chan], &resDesc, &texDesc, NULL ), "VolumeGVDB", "SetupAtlasAccess", "cuTexObjectCreate", chanmsg, mbDebug);
		cudaCheck ( cuSurfObjectCreate ( &mVDBInfo.volOut[chan], &resDesc ), "VolumeGVDB", "SetupAtlasAccess", "cuSurfObjectCreate", chanmsg, mbDebug);

		if ( atlas.grsc != 0x0 )
			cudaCheck ( cuGraphicsUnmapResources(1, &atlas.grsc, mStream), "VolumeGVDB", "SetupAtlasAccess", "cuGraphicsUnmapResources", chanmsg, mbDebug);
	}	
	// Require VDBInfo update
	mVDBInfo.update = true;

	POP_CTX
}

//////////////////////////////////////////////////////////////////////////

void VolumeGVDB::ClearPoolCPU()
{
	PUSH_CTX
	mPool->PoolClearCPU();
	POP_CTX
}

void VolumeGVDB::FetchPoolCPU()
{
	PUSH_CTX
	mPool->PoolFetchAll();
	POP_CTX
}

int VolumeGVDB::DetermineDepth( Vector3DI& pos)
{
	// looking for certain level that one brick can contain the bounding box
	Vector3DI range, posMin, posMax;
	for (int lev = 0; lev < MAXLEV; lev++)
	{
		posMin = GetCoveringNode ( lev, mPosMin, range );
		posMax = GetCoveringNode ( lev, mPosMax, range );

		if (posMin.x == posMax.x && posMin.y == posMax.y && posMin.z == posMax.z) {
			pos = posMin;
			return lev;
		}
	}

	return -1;
}

bool pairCompare(std::pair<int, int> firstElem, std::pair<int, int> secondElem) {
	return firstElem.first < secondElem.first;
}

void VolumeGVDB::FindUniqueBrick(int pNumPnts, int pLevDepth, int& pNumUniqueBrick)
{
	PERF_PUSH ("Find unique");
	int numPnts = pNumPnts;
	int threads = 512;		
	int pblks = int(numPnts / threads)+1;

	PrepareAux ( AUX_MARKER, numPnts, sizeof(int), true);
	PrepareAux ( AUX_MARKER_PRESUM, numPnts, sizeof(int), true);
	PrepareAux ( AUX_UNIQUE_CNT, 1, sizeof(int), true);
	PrepareAux ( AUX_LEVEL_CNT, pLevDepth, sizeof(int), true);
	
	PUSH_CTX

	void* argsFindUnique[5] = { &numPnts, &mAux[AUX_SORTED_LEVXYZ].gpu, &mAux[AUX_MARKER].gpu, &mAux[AUX_UNIQUE_CNT].gpu, &mAux[AUX_LEVEL_CNT].gpu};
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_FIND_UNIQUE], 
		pblks, 1, 1, threads, 1, 1, 0, NULL, argsFindUnique, NULL ), "VolumeGVDB", "FindUniqueBrick", "cuLaunch", "FUNC_FIND_UNIQUE", mbDebug);

	cudaCheck ( cuMemcpyDtoH ( &pNumUniqueBrick, mAux[AUX_UNIQUE_CNT].gpu, sizeof(int) ), "VolumeGVDB", "FindUniqueBrick", "cuMemcpyDtoH", "AUX_UNIQUE_CNT", mbDebug);
	
	PrefixSum(mAux[AUX_MARKER].gpu, mAux[AUX_MARKER_PRESUM].gpu, numPnts);

	PrepareAux ( AUX_UNIQUE_LEVXYZ, pNumUniqueBrick, sizeof(long long), false);
	void* argsCompactUnique[5] = { &numPnts, &mAux[AUX_SORTED_LEVXYZ].gpu, &mAux[AUX_MARKER].gpu, &mAux[AUX_MARKER_PRESUM].gpu, &mAux[AUX_UNIQUE_LEVXYZ].gpu};
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_COMPACT_UNIQUE], 
		pblks, 1, 1, threads, 1, 1, 0, NULL, argsCompactUnique, NULL ), "VolumeGVDB", "FindUniqueBrick", "cuLaunch", "FUNC_COMPACT_UNIQUE", mbDebug );

	PERF_POP ();

	POP_CTX
}

static const int NUM_SMS = 16;
static const int NUM_THREADS_PER_SM = 192;
static const int NUM_THREADS_PER_BLOCK = 64;
//static const int NUM_THREADS = NUM_THREADS_PER_SM * NUM_SMS;
static const int NUM_BLOCKS = (NUM_THREADS_PER_SM / NUM_THREADS_PER_BLOCK) * NUM_SMS;
static const int RADIX = 8;                                                        // Number of bits per radix sort pass
static const int RADICES = 1 << RADIX;                                             // Number of radices
static const int RADIXMASK = RADICES - 1;                                          // Mask for each radix sort pass
static const int RADIXBITS = 64;                                                   // Number of bits to sort over
static const int RADIXTHREADS = 16;                                                // Number of threads sharing each radix counter
static const int RADIXGROUPS = NUM_THREADS_PER_BLOCK / RADIXTHREADS;               // Number of radix groups per CTA
static const int TOTALRADIXGROUPS = NUM_BLOCKS * RADIXGROUPS;                      // Number of radix groups for each radix
static const int SORTRADIXGROUPS = TOTALRADIXGROUPS * RADICES;                     // Total radix count
static const int GRFELEMENTS = (NUM_THREADS_PER_BLOCK / RADIXTHREADS) * RADICES; 
static const int GRFSIZE = GRFELEMENTS * sizeof(uint); 

// Prefix sum variables
static const int PREFIX_NUM_THREADS_PER_SM = NUM_THREADS_PER_SM;
static const int PREFIX_NUM_THREADS_PER_BLOCK = PREFIX_NUM_THREADS_PER_SM;
static const int PREFIX_NUM_BLOCKS = (PREFIX_NUM_THREADS_PER_SM / PREFIX_NUM_THREADS_PER_BLOCK) * NUM_SMS;
static const int PREFIX_BLOCKSIZE = SORTRADIXGROUPS / PREFIX_NUM_BLOCKS;
static const int PREFIX_GRFELEMENTS = PREFIX_BLOCKSIZE + 2 * PREFIX_NUM_THREADS_PER_BLOCK;
static const int PREFIX_GRFSIZE = PREFIX_GRFELEMENTS * sizeof(uint);

// Shuffle variables
static const int SHUFFLE_GRFOFFSET = RADIXGROUPS * RADICES;
static const int SHUFFLE_GRFELEMENTS = SHUFFLE_GRFOFFSET + PREFIX_NUM_BLOCKS; 
static const int SHUFFLE_GRFSIZE = SHUFFLE_GRFELEMENTS * sizeof(uint); 

void VolumeGVDB::RadixSortByByte(int pNumPnts, int pLevDepth)
{
	PUSH_CTX

	PERF_PUSH("Radix sort");

	int numPnts = pNumPnts;

	int threads = 512;		
	int pblks = int(numPnts / threads)+1;
	int length = pNumPnts * 4;

	PrepareAux ( AUX_SORTED_LEVXYZ, pNumPnts * 4, sizeof(unsigned short), false);

	cudaCheck ( cuMemcpyDtoD (mAux[AUX_SORTED_LEVXYZ].gpu, mAux[AUX_BRICK_LEVXYZ].gpu, length * sizeof(unsigned short) ),
		               "VolumeGVDB", "RadixSortByByte", "cuMemcpyDtoD", "AUX_SORTED_LEVXYZ", mbDebug);

	// gvdbDeviceRadixSort takes a pointer to memory interpreted as 64-bit
	// integers, and the number of 64-bit integers in the array.
	gvdbDeviceRadixSort(mAux[AUX_SORTED_LEVXYZ].gpu, pNumPnts);

	PERF_POP();

	POP_CTX
}


void VolumeGVDB::ActivateExtraBricksGPU(int pNumPnts, float pRadius, Vector3DF pOrig, int pRootLev, Vector3DI pRootPos)
{
	PERF_PUSH("Extra node");

	int threads = 512;		
	int pblks = int(pNumPnts / threads)+1;

	PrepareAux ( AUX_BRICK_LEVXYZ, pNumPnts * pRootLev * 4, sizeof(unsigned short), true );
	PrepareAux ( AUX_EXTRA_BRICK_CNT, 1, sizeof(int), true, true);

	PUSH_CTX

	void* argsCalcExtraLevXYZ[11] = { &cuVDBInfo, &pRadius,
		&pNumPnts, &pRootLev, &mAux[AUX_RANGE_RES].gpu,
		&mAux[AUX_PNTPOS].gpu, &mAux[AUX_PNTPOS].subdim.x, &mAux[AUX_PNTPOS].stride,
		&pOrig.x, &mAux[AUX_BRICK_LEVXYZ].gpu, &mAux[AUX_EXTRA_BRICK_CNT].gpu};

	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_CALC_EXTRA_BRICK_ID], 
		pblks, 1, 1, threads, 1, 1, 0, NULL, argsCalcExtraLevXYZ, NULL ), "VolumeGVDB", "ActivateExtraBricksGPU", "cuLaunch", "FUNC_CALC_EXTRA_BRICK_ID", mbDebug);

	RetrieveData(mAux[AUX_EXTRA_BRICK_CNT]);
	int* extraNodeNum = (int *)mAux[AUX_EXTRA_BRICK_CNT].cpu;

	int numUniqueExtraBrick = 0;	
	RadixSortByByte(extraNodeNum[0], pRootLev);
	FindUniqueBrick(extraNodeNum[0], pRootLev, numUniqueExtraBrick);

	PERF_PUSH ( "Create pool (CPU)");
	int* extraLevCnt = (int*) malloc ( pRootLev * sizeof(int) );
	cudaCheck ( cuMemcpyDtoH ( extraLevCnt, mAux[AUX_LEVEL_CNT].gpu, pRootLev * sizeof(int) ), "VolumeGVDB", "ActivateExtraBricksGPU", "cuMemcpyDtoH", "AUX_LEVEL_CNT", mbDebug);

	unsigned short* extraUniBricks = (unsigned short*) malloc ( numUniqueExtraBrick * 4 * sizeof(unsigned short) );
	cudaCheck ( cuMemcpyDtoH ( extraUniBricks, mAux[AUX_UNIQUE_LEVXYZ].gpu, numUniqueExtraBrick * 4 * sizeof(unsigned short) ), "VolumeGVDB", "ActivateExtraBricksGPU", "cuMemcpyDtoH", "AUX_UNIQUE_LEVXYZ", mbDebug);

	int extraLevPrefixSum = 0;
	for (int lev = 0; lev < pRootLev; lev++)	
	{
		int r = getRes(lev); 
		uint64 childlistLen = ((uint64) r*r*r);

		Vector3DI brickRange = getRange(lev);
		for (int n = extraLevPrefixSum; n < extraLevCnt[lev]+extraLevPrefixSum; n++)
		{
			uint64 nodeID = mPool->PoolAlloc ( 0, lev, true );
			SetupNode ( nodeID, lev, Vector3DI( extraUniBricks[n * 4 + 2] * brickRange.x, 
				extraUniBricks[n * 4 + 1] * brickRange.y, 
				extraUniBricks[n * 4 + 0] * brickRange.z));

			if (lev > 0) 
			{
				Node* nd = getNode(nodeID);
				nd->mChildList = mPool->PoolAlloc ( 1, lev, true );
				uint64* clist = mPool->PoolData64 ( nd->mChildList );			
				memset(clist, 0xFF, sizeof(uint64) * childlistLen);
			}
		}
		extraLevPrefixSum += extraLevCnt[lev];
	}
	PERF_POP();

	PERF_PUSH ( "Commit");
	mPool->PoolCommitAll();
	PERF_POP ();
	PERF_PUSH ( "PrepPartial");
	PrepareVDBPartially();
	PERF_POP ();

	PERF_PUSH ( "Link node");
	for (int lev = pRootLev-1; lev >= 0; lev--)	
	{
		//std::cout << "link " << lev << std::endl;
		pblks = int(mPool->getPoolTotalCnt(0,lev)/ threads)+1;
		void* argsLink[2] = { &cuVDBInfo, &lev};
		cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_LINK_BRICKS], 
			pblks, 1, 1, threads, 1, 1, 0, NULL, argsLink, NULL ), "VolumeGVDB", "ActivateExtraBricksGPU", "cuLaunch", "FUNC_LINK_BRICKS", mbDebug);
	}
	PERF_POP();

	PERF_PUSH ( "FetchAll");
	mPool->PoolFetchAll();
	PERF_POP();

	POP_CTX

	PERF_POP();
}
  
int Incre_ratio = 1;	// assume only 10% new points need to be sorted

void VolumeGVDB::ActivateIncreBricksGPU(int pNumPnts, float pRadius, Vector3DF pOrig, int pRootLev, Vector3DI pRootPos, bool bAccum)
{
	if (getNumTotalNodes(0) == 0) return;

	PUSH_CTX

	PERF_PUSH("Incremental node");	

	// mark all leaf node deactivated
	int totalLeafNodeCnt = static_cast<int>(getNumTotalNodes(0));
	PrepareAux ( AUX_NODE_MARKER, totalLeafNodeCnt, sizeof(int), false, true);
	
	//--- Accumulate topology (OPTIONAL)
	int* marker = (int*) mAux[AUX_NODE_MARKER].cpu;
	if ( bAccum ) {	
		// mark existing nodes active, preserves previous brick topology
		for (int ni=0; ni < totalLeafNodeCnt; ni++ ) {
			*marker++ = getNode(0,0, ni)->mFlags;
		}
	} else {
		// mark all deactivated (default behavior)
		memset ( marker, 0, totalLeafNodeCnt*sizeof(int) );
	}
	CommitData ( mAux[AUX_NODE_MARKER] );	
	//---	
	
	int threads = 512;		
	int pblks = int(pNumPnts / threads)+1;

	PrepareAux ( AUX_BRICK_LEVXYZ, pNumPnts * pRootLev * 4 / Incre_ratio, sizeof(unsigned short), true );
	PrepareAux ( AUX_EXTRA_BRICK_CNT, 1, sizeof(int), true, true);

	void* argsCalcExtraLevXYZ[12] = { &cuVDBInfo, &pRadius,
		&pNumPnts, &pRootLev, &mAux[AUX_RANGE_RES].gpu,
		&mAux[AUX_PNTPOS].gpu, &mAux[AUX_PNTPOS].subdim.x, &mAux[AUX_PNTPOS].stride,
		&pOrig.x, &mAux[AUX_BRICK_LEVXYZ].gpu, &mAux[AUX_EXTRA_BRICK_CNT].gpu,
		&mAux[AUX_NODE_MARKER].gpu,
	};

	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_CALC_INCRE_EXTRA_BRICK_ID], pblks, 1, 1, threads, 1, 1, 0, NULL, argsCalcExtraLevXYZ, NULL ), 
						"VolumeGVDB", "ActivateIncreBricksGPU", "cuLaunch", "FUNC_CALC_INCRE_EXTRA_BRICK_ID", mbDebug);
	
	RetrieveData(mAux[AUX_EXTRA_BRICK_CNT]);
	int* extraNodeNum = (int *)mAux[AUX_EXTRA_BRICK_CNT].cpu;


	if (extraNodeNum[0] == 0) { POP_CTX; PERF_POP(); return; }	// no new node, then return

	int numUniqueExtraBrick = 0;	
	RadixSortByByte(extraNodeNum[0], pRootLev);
	FindUniqueBrick(extraNodeNum[0], pRootLev, numUniqueExtraBrick);

	int* extraLevCnt = new int[pRootLev];
	cudaCheck ( cuMemcpyDtoH ( extraLevCnt, mAux[AUX_LEVEL_CNT].gpu, pRootLev * sizeof(int) ), "VolumeGVDB", "ActivateIncreBricksGPU", "cuMemcpyDtoH", "AUX_LEVEL_CNT", mbDebug);

	unsigned short* extraUniBricks = new unsigned short[numUniqueExtraBrick * 4];
	cudaCheck ( cuMemcpyDtoH ( extraUniBricks, mAux[AUX_UNIQUE_LEVXYZ].gpu, numUniqueExtraBrick * 4 * sizeof(unsigned short) ), "VolumeGVDB", "ActivateIncreBricksGPU", "cuMemcpyDtoH", "AUX_UNIQUE_LEVXYZ", mbDebug);

	PERF_PUSH ( "Update markers (CPU)");
		RetrieveData ( mAux[AUX_NODE_MARKER] );	

		marker = (int*) mAux[AUX_NODE_MARKER].cpu;
		for (int ni = 0; ni < totalLeafNodeCnt; ni++) {
			getNode(0,0,ni)->mFlags = *marker++;
		}
	PERF_POP ();

	PERF_PUSH ( "Allocate pool (CPU)");
	int extraLevPrefixSum = 0;
	for (int lev = 0; lev < pRootLev; lev++)	
	{
		int r = getRes(lev); 
		uint64 childlistLen = ((uint64) r*r*r);
		Vector3DI brickRange = getRange(lev);
		for (int n = extraLevPrefixSum; n < extraLevCnt[lev]+extraLevPrefixSum; n++)
		{
			// Allocate new node
			// - preserve existing on expand
			// - marker is set to 1 (active)
			uint64 nodeID = mPool->PoolAlloc ( 0, lev, true );		
			SetupNode ( nodeID, lev, Vector3DI( extraUniBricks[n * 4 + 2] * brickRange.x, extraUniBricks[n * 4 + 1] * brickRange.y, extraUniBricks[n * 4 + 0] * brickRange.z));
	
			if (lev > 0) // alloc childlist for nodes except leaf nodes
			{
				Node* nd = getNode(nodeID);
				nd->mChildList = mPool->PoolAlloc ( 1, lev, true );
				uint64* clist = mPool->PoolData64 ( nd->mChildList );			
				memset(clist, 0xFF, sizeof(uint64) * childlistLen);
			}
		}
		extraLevPrefixSum += extraLevCnt[lev];
	}
	PERF_POP ();

	mPool->PoolCommitAll();

	// Commit all existing and new nodes, including marker data
	PERF_PUSH ( "PrepPartial");
	PrepareVDBPartially();
	PERF_POP ();

	// link - top down
	pblks = int(getNumTotalNodes(0) / threads)+1;
	PERF_PUSH ( "Link node");
	for (int lev = pRootLev - 1; lev >= 0; lev--)		
	{
		void* argsLink[2] = { &cuVDBInfo, &lev};
		cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_LINK_BRICKS], pblks, 1, 1, threads, 1, 1, 0, NULL, argsLink, NULL ), "VolumeGVDB", "ActivateIncreBricksGPU", "cuLaunch", "FUNC_LINK_BRICKS", mbDebug);
	}
	PERF_POP();

	// delink - bottom up 
	PERF_PUSH ( "Delink node");	
	void* argsDelinkLeaf[1] = { &cuVDBInfo};
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_DELINK_LEAF_BRICKS], pblks, 1, 1, threads, 1, 1, 0, NULL, argsDelinkLeaf, NULL ), "VolumeGVDB", "ActivateIncreBricksGPU", "cuLaunch", "FUNC_DELINK_LEAF_BRICKS", mbDebug);
	for (int lev = 1; lev < pRootLev; lev++)				
	{
		void* argsDelink[2] = { &cuVDBInfo, &lev};
		cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_DELINK_BRICKS], pblks, 1, 1, threads, 1, 1, 0, NULL, argsDelink, NULL ), "VolumeGVDB", "ActivateIncreBricksGPU", "cuLaunch", "FUNC_DELINK_BRICKS", mbDebug);
	}
	PERF_POP();

	mPool->PoolFetchAll();

	PERF_PUSH ( "UpdateUsed");
	uint64 usedNum = mPool->getPoolTotalCnt(0,0);
	for (int ni = 0; ni < mPool->getPoolTotalCnt(0,0); ni++) if(!getNode(0,0,ni)->mFlags) usedNum--;
	mPool->SetPoolUsedCnt(0,0,usedNum);
	PERF_POP();

	delete[] extraLevCnt;
	delete[] extraUniBricks;

	PERF_POP();

	POP_CTX
}

void VolumeGVDB::AccumulateTopology(int pNumPnts, float pRadius, Vector3DF pOrig, int iDepth )
{
	GetBoundingBox(pNumPnts, pOrig);
	
	Vector3DI rootPos;
	int depth = DetermineDepth(rootPos);
	if ( depth < iDepth ) depth = iDepth;		// min depth
	if ( depth == 0 ) {
		gprintf("ERROR: Points extents smaller than 1 brick. Not yet supported.\n");
		return;
	}
	if (depth > 5) {
		gprintf("ERROR: Point extents exceed 5 levels.\n");
		return;
	}
	if (mRebuildTopo) {
		RebuildTopology ( pNumPnts, pRadius, pOrig );
		return;
	}
	ActivateIncreBricksGPU(pNumPnts, pRadius, pOrig, depth, rootPos, true );	
}


void VolumeGVDB::RebuildTopology(int pNumPnts, float pRadius, Vector3DF pOrig )
{
	GetBoundingBox(pNumPnts, pOrig);

	Vector3DI rootPos;
	int depth = DetermineDepth(rootPos);
	if (depth == 0) {
		gprintf("ERROR: Points extents smaller than 1 brick. Not yet supported.\n");
		return;
	}
	if (depth > 5) {
		gprintf("ERROR: Point extents exceed 5 levels.\n");
		return;
	}
	if (mRebuildTopo) {
		Clear();
	} else {
		if (depth != mCurrDepth) { // in case the depth is changing 
			gprintf("Warning: Depth change in rebuild.\n");
			Clear();
			mRebuildTopo = true;
			mCurrDepth = depth;
		}
	}
	if (mRebuildTopo) {
		ActivateBricksGPU(pNumPnts, pRadius, pOrig, depth, rootPos);
		if (pRadius >= 1)
			ActivateExtraBricksGPU(pNumPnts, pRadius, pOrig, depth, rootPos);	// for radius > 1 and staggered grid
		mRebuildTopo = false;

		CleanAux(AUX_BRICK_LEVXYZ);
		CleanAux(AUX_MARKER);
		CleanAux(AUX_SORTED_LEVXYZ);
		CleanAux(AUX_MARKER_PRESUM);
	} else {
		ActivateIncreBricksGPU(pNumPnts, pRadius, pOrig, depth, rootPos);
	}
}

void VolumeGVDB::RebuildTopologyCPU(int pNumPnts, Vector3DF pOrig, Vector3DF* pPos)
{
	Vector3DF p;
	for (int n=0; n < pNumPnts; n++) {		
		p = (*pPos++) + pOrig;					
		ActivateSpace ( p );					
	}	
}

void VolumeGVDB::ActivateBricksGPU(int pNumPnts, float pRadius, Vector3DF pOrig, int pRootLev, Vector3DI pRootPos)
{
	PUSH_CTX

	//////////////////////////////////////////////////////////////////////////
	// Prepare
	//////////////////////////////////////////////////////////////////////////
	PrepareAux ( AUX_RANGE_RES, pRootLev, sizeof(int), true, true );
	int* range_res = (int*) mAux[AUX_RANGE_RES].cpu;
	for (int lev = 0; lev < pRootLev; lev++) 
		range_res[lev] = getRange(lev).x;

	CommitData ( mAux[AUX_RANGE_RES] );	

	PrepareAux ( AUX_BRICK_LEVXYZ, pNumPnts * pRootLev * 4, sizeof(unsigned short), false );	// 4 - lev, x, y, z
	
	void* argsCalcLevXYZ[8] = { &pNumPnts, &pRootLev, &mAux[AUX_RANGE_RES].gpu,
		&mAux[AUX_PNTPOS].gpu, &mAux[AUX_PNTPOS].subdim.x, &mAux[AUX_PNTPOS].stride,
		&pOrig.x, &mAux[AUX_BRICK_LEVXYZ].gpu};

	int threads = 512;		
	int pblks = int(pNumPnts / threads)+1;

	//////////////////////////////////////////////////////////////////////////
	// calculate brick id for each level
	//////////////////////////////////////////////////////////////////////////
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_CALC_BRICK_ID], 
		pblks, 1, 1, threads, 1, 1, 0, NULL, argsCalcLevXYZ, NULL ), "VolumeGVDB", "ActivateBricksGPU", "cuLaunch", "FUNC_CALC_BRICK_ID", mbDebug);

#if 0
	//////////////////////////////////////////////////////////////////////////
	// checking data
	//////////////////////////////////////////////////////////////////////////
	int length = pNumPnts * pRootLev * 4;
	unsigned short* dat1 = (unsigned short*) malloc ( length * sizeof(unsigned short) );
	cudaCheck ( cuMemcpyDtoH ( dat1, mAux[AUX_BRICK_LEVXYZ].gpu, length * sizeof(unsigned short) ), "Memcpy dat1", "ActivateBricksGPU" );

	for (int i = 0; i < pNumPnts; i++)
	{
		std::cout << "Point " << i << std::endl;
		for (int lev = 0; lev < pRootLev; lev++)
		{
			std::cout << "	level " << lev;
			std::cout << ": " << dat1[ i * pRootLev * 4 + lev * 4 + 0];
			std::cout << " " << dat1[ i * pRootLev * 4 + lev * 4 + 1];
			std::cout << " " << dat1[ i * pRootLev * 4 + lev * 4 + 2];
			std::cout << " " << dat1[ i * pRootLev * 4 + lev * 4 + 3] << std::endl;
		}
	}
#endif

	int numUniqueBrick = 0;	
	RadixSortByByte(pNumPnts * pRootLev, pRootLev);
	FindUniqueBrick(pNumPnts * pRootLev, pRootLev, numUniqueBrick);

	//////////////////////////////////////////////////////////////////////////
	// create pool for each level
	// set lev, pos, and childlist
	//////////////////////////////////////////////////////////////////////////
	PERF_PUSH("Create pool (CPU)");
	{
		mRoot = mPool->PoolAlloc ( 0, pRootLev, true );
		uint64 childlistId = mPool->PoolAlloc ( 1, pRootLev, true );
		SetupNode ( mRoot, pRootLev, pRootPos);
		Node* rootnd = getNode(mRoot);
		rootnd->mFlags = true;
		rootnd->mChildList = childlistId;
		uint64* clist = mPool->PoolData64 ( rootnd->mChildList );
		int r = getRes(pRootLev); 
		uint64 childlistLen = ((uint64) r*r*r);
		memset(clist, 0xFF, sizeof(uint64) * childlistLen);
	}
	
	int* levCnt = new int[pRootLev];
	cudaCheck ( cuMemcpyDtoH ( levCnt, mAux[AUX_LEVEL_CNT].gpu, pRootLev * sizeof(int) ), "VolumeGVDB", "ActivateBricksGPU", "cuMemcpyDtoH", "AUX_LEVEL_CNT", mbDebug);

	unsigned short* uniBricks = new unsigned short[numUniqueBrick * 4];
	cudaCheck ( cuMemcpyDtoH ( uniBricks, mAux[AUX_UNIQUE_LEVXYZ].gpu, numUniqueBrick * 4 * sizeof(unsigned short) ), "VolumeGVDB", "ActivateBricksGPU", "cuMemcpyDtoH", "AUX_UNIQUE_LEVXYZ", mbDebug);

	int levPrefixSum = 0;
	for (int lev = 0; lev < pRootLev; lev++)	
	{
		int r = getRes(lev); 
		uint64 childlistLen = ((uint64) r*r*r);

		Vector3DI brickRange = getRange(lev);
		for (int n = levPrefixSum; n < levCnt[lev]+levPrefixSum; n++)
		{
			uint64 nodeID = mPool->PoolAlloc ( 0, lev, true );
			SetupNode ( nodeID, lev, Vector3DI( uniBricks[n * 4 + 2] * brickRange.x, 
				uniBricks[n * 4 + 1] * brickRange.y, 
				uniBricks[n * 4 + 0] * brickRange.z));

			if (lev > 0) 
			{
				Node* nd = getNode(nodeID);
				nd->mChildList = mPool->PoolAlloc ( 1, lev, true );
				uint64* clist = mPool->PoolData64 ( nd->mChildList );			
				memset(clist, 0xFF, sizeof(uint64) * childlistLen);
			}
		}
		levPrefixSum += levCnt[lev];
	}
	PERF_POP();
		
	//////////////////////////////////////////////////////////////////////////
	// set bit mask and link between parent and children
	//////////////////////////////////////////////////////////////////////////
		
	
#if GVDB_CPU_VERSION // CPU version 	
	Vector3DF posInNode;
	uint32 bitMaskPos;
	for (int lev = pRootLev-1; lev >= 0; lev--)	
	{
		Vector3DI res = getRes3DI(lev+1);
		Vector3DI range = getRange(lev+1);
		for (int n = 0; n < levCnt[lev]; n++)
		{
			uint64 nodeID = Elem(0, lev, n);
			Node* nd = getNode(nodeID);

			uint64 parentNodeID = FindParent(lev+1, nd->mPos) ;
			Node* parentNd = getNode(parentNodeID);

			posInNode = nd->mPos - parentNd->mPos;
			posInNode *= res;
			posInNode /= range;
			posInNode.x = floor(posInNode.x);	// IMPORTANT!!! truncate decimal 
			posInNode.y = floor(posInNode.y);
			posInNode.z = floor(posInNode.z);	
			bitMaskPos = (posInNode.z*res.x + posInNode.y)*res.x+ posInNode.x;

			// parent in node
			nd->mParent = parentNodeID;	// set parent of child
#ifdef USE_BITMASKS
			// determine child bit position	
			uint64 p = parentNd->countOn ( bitMaskPos );
			uint64 cnum = parentNd->getNumChild();		// existing children count
			parentNd->setOn ( bitMaskPos );

			// insert into child list in parent node
			uint64* clist = mPool->PoolData64 ( parentNd->mChildList );
			if ( p < cnum ) {
				memmove ( clist + p+1, clist + p, (cnum-p)*sizeof(uint64) );
				*(clist + p) = nodeID;
			} else {
				*(clist + cnum) = nodeID;		
			}
#else
			uint64* clist = mPool->PoolData64 ( parentNd->mChildList );
			*(clist + bitMaskPos) = nodeID;	
#endif
		}
		mPool->PoolCommitAll();
	}
#else	// GPU version

	PERF_PUSH ( "Commit");
	mPool->PoolCommitAll();
	PERF_POP ();
	PERF_PUSH ( "PrepPartial");
	PrepareVDBPartially();
	PERF_POP ();

	PERF_PUSH("Link Levels");
	for (int lev = pRootLev-1; lev >= 0; lev--)	
	{
		pblks = int(mPool->getPoolTotalCnt(0,lev) / threads)+1;
		void* argsLink[3] = { &cuVDBInfo, &lev};
		cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_LINK_BRICKS], 
			pblks, 1, 1, threads, 1, 1, 0, NULL, argsLink, NULL ), "VolumeGVDB", "ActivateBricksGPU", "cuLaunch", "FUNC_LINK_BRICKS", mbDebug);
	}
	PERF_POP();

	PERF_PUSH ( "FetchAll");
	mPool->PoolFetchAll();
	PERF_POP();
#endif	
	
	POP_CTX

	delete[] levCnt;
	delete[] uniBricks;
}

void VolumeGVDB::FindActivBricks(int pLev,  int pRootlev,  int pNumPnts, Vector3DF pOrig, Vector3DI pRootPos)
{
	PUSH_CTX

	std::cout << "======================" << pLev << "======================\n";
	//////////////////////////////////////////////////////////////////////////
	// create root node
	//////////////////////////////////////////////////////////////////////////
	if (pLev == pRootlev)
	{
		mRoot = mPool->PoolAlloc ( 0, pLev, true );
		uint64 childlistId = mPool->PoolAlloc ( 1, pLev, true );
		SetupNode ( mRoot, pLev, pRootPos);

		Node* rootnd = getNode(mRoot);
		rootnd->mChildList = childlistId;
	}

	//////////////////////////////////////////////////////////////////////////
	// find activated bricks per points
	//////////////////////////////////////////////////////////////////////////
	PERF_PUSH ("Find brick idx for points");
	int nxtLev = pLev - 1;
	int dim = getRange(pRootlev).x / getRange(nxtLev).x;
	int d2 = dim * dim;

	PrepareAux ( AUX_PBRICKDX, pNumPnts, sizeof(int), false );

	int threads = 512;		
	int pblks = int(pNumPnts / threads)+1;
	Vector3DI brickRange = getRange(nxtLev);
	Vector3DI orig_shift = GetCoveringNode ( nxtLev, pOrig, brickRange );

	void* args[11] = { &pNumPnts, &nxtLev, &brickRange, &dim,
		&mAux[AUX_PNTPOS].gpu, &mAux[AUX_PNTPOS].subdim.x, &mAux[AUX_PNTPOS].stride,
		&pOrig.x,  &orig_shift, &mAux[AUX_PBRICKDX].gpu};

	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_FIND_ACTIV_BRICKS], pblks, 1, 1, threads, 1, 1, 0, NULL, args, NULL ), 
					"VolumeGVDB", "ActivateBricksGPU", "cuLaunch", "FUNC_FIND_ACTIV_BRICKS", mbDebug);
	
	// retrieve data
	int* brickIdx = new int[pNumPnts];
	cudaCheck ( cuMemcpyDtoH ( brickIdx, mAux[AUX_PBRICKDX].gpu, pNumPnts*sizeof(int) ), "VolumeGVDB", "ActivateBricksGPU", "cuMemcpyDtoH", "AUX_PBRICKDX", mbDebug);

	Vector3DF* sortedPos = new Vector3DF[pNumPnts];
	Vector3DF* pos = new Vector3DF[pNumPnts];
	cudaCheck ( cuMemcpyDtoH ( pos, mAux[AUX_PNTPOS].gpu, pNumPnts*sizeof(Vector3DF) ), "VolumeGVDB", "ActivateBricksGPU", "cuMemcpyDtoH", "AUX_PNTPOS", mbDebug);

	PERF_POP ();
	//////////////////////////////////////////////////////////////////////////
	// sort points based on brick idx 
	// TODO: CUDA implementation
	//////////////////////////////////////////////////////////////////////////
	PERF_PUSH ("Sort points");
	std::vector< std::pair<int, int> > pntList;
	for (int n = 0; n < pNumPnts; n++) {
		std::pair<int, int> tmp(brickIdx[n], n);
		pntList.push_back(tmp);
	}
	std::sort(pntList.begin(), pntList.end(), pairCompare);

	// write back to pPos and pBrickIdx
	for (int n = 0; n < pNumPnts; n++) {
		brickIdx[n] = pntList[n].first;
		sortedPos[n] = pos[pntList[n].second];
	}
	PERF_POP ();
	//////////////////////////////////////////////////////////////////////////
	// determine activated bricks number
	// TODO: CUDA implementation
	//////////////////////////////////////////////////////////////////////////
	PERF_PUSH ("Determine brick number");
	std::vector<int> activatingBrickIdx;
	int numActivatingBrick = 1;
	int prevBrickIdx = brickIdx[0];
	activatingBrickIdx.push_back(brickIdx[0]);
	for (int n = 1; n < pNumPnts; n++)
	{
		if (brickIdx[n] != prevBrickIdx) {
			activatingBrickIdx.push_back(brickIdx[n]);
			numActivatingBrick++;
		}
		prevBrickIdx = brickIdx[n];
	}
	PERF_POP ();
	//////////////////////////////////////////////////////////////////////////
	// create pool 0 and pool 1 for next level
	//////////////////////////////////////////////////////////////////////////
	PERF_PUSH ("Create pool");
	for (int n = 0; n < numActivatingBrick; n++)
	{
		uint64 nodeID = mPool->PoolAlloc ( 0, pLev - 1, true );
		SetupNode ( nodeID, pLev - 1, Vector3DI( activatingBrickIdx[n] % dim * brickRange.x, 
			activatingBrickIdx[n] / dim % dim * brickRange.y, 
			activatingBrickIdx[n] / d2 % dim * brickRange.z));
		if ((pLev - 1) > 0) 
		{
			Node* nd = getNode(nodeID);
			nd->mChildList = mPool->PoolAlloc ( 1, pLev - 1, true );
		}
	}
	PERF_POP ();
	//////////////////////////////////////////////////////////////////////////
	// set bit mask and parent for current level
	// set childlist for parent node
	// TODO: CUDA implementation
	//////////////////////////////////////////////////////////////////////////
	PERF_PUSH ("Setup child and parent");
	Vector3DI res = getRes3DI ( nxtLev + 1 );
	Vector3DI range = getRange ( nxtLev + 1);
	Vector3DF posInNode;
	uint32 bitMaskPos;
	Vector3DI tmpRange;
	for (int n = 0; n < numActivatingBrick; n++)
	{
		uint64 nodeID = Elem(0, pLev - 1, n);
		Node* nd = getNode(nodeID);

		uint64 parentNodeID = FindParent(pLev, nd->mPos) ;
		Node* parentNd = getNode(parentNodeID);

		posInNode = nd->mPos - parentNd->mPos;
		posInNode *= res;
		posInNode /= range;
		int posInNodeX = int(floor(posInNode.x));	// IMPORTANT!!! truncate decimal 
		int posInNodeY = int(floor(posInNode.y));
		int posInNodeZ = int(floor(posInNode.z));
		bitMaskPos = (posInNodeZ*res.x + posInNodeY)*res.x+ posInNodeX;

		// parent in node
		nd->mParent = parentNodeID;	// set parent of child

		// determine child bit position	
#ifdef USE_BITMASKS
		uint64 p = parentNd->countOn ( bitMaskPos );
		uint64 cnum = parentNd->getNumChild();		// existing children count
		parentNd->setOn ( bitMaskPos );

		// insert into child list in parent node
		uint64* clist = mPool->PoolData64 ( parentNd->mChildList );
		if ( p < cnum ) {
			memmove ( clist + p+1, clist + p, (cnum-p)*sizeof(uint64) );
			*(clist + p) = nodeID;
		} else {
			*(clist + cnum) = nodeID;		
		}
#else
		uint64* clist = mPool->PoolData64 ( parentNd->mChildList );
		*(clist + bitMaskPos) = nodeID;
#endif
	}
	PERF_POP ();
	//////////////////////////////////////////////////////////////////////////
	// clear data
	//////////////////////////////////////////////////////////////////////////
	delete brickIdx;
	delete sortedPos;
	delete pos;

	POP_CTX
}

//////////////////////////////////////////////////////////////////////////

void VolumeGVDB::FinishTopology (bool pCommitPool, bool pComputeBounds)
{
	PUSH_CTX

	// compute bounds
	if (pComputeBounds) ComputeBounds ();

	// commit topology
	if (pCommitPool)	mPool->PoolCommitAll();

	// update VDB data on gpu 
	mVDBInfo.update = true;	

	POP_CTX
}

// Clear all channels
void VolumeGVDB::ClearChannel (uchar chan)
{
	// This launches a kernel to clear the CUarray.
	//   (there is no MemsetD8 for cuda arrays)
	PUSH_CTX
	mPool->AtlasFill(chan);	
	POP_CTX
}


uchar VolumeGVDB::GetChannelType(uchar channel) {
	return mPool->getAtlas(channel).type;
}


// Clear all channels
void VolumeGVDB::ClearAllChannels ()
{
	PUSH_CTX

	PERF_PUSH ( "Clear All" );
	for (int n = 0; n < mPool->getNumAtlas(); n++)
		ClearChannel(n);

	PERF_POP ();

	POP_CTX
}

// Save a VBX file
void VolumeGVDB::SaveVBX ( const std::string fname )
{
	// See GVDB_FILESPEC.txt for the specification of the VBX file format.
	PUSH_CTX
	
	FILE* fp = fopen ( fname.c_str(), "wb" );

	const uchar major = MAJOR_VERSION;
	const uchar minor = MINOR_VERSION;
	
	PERF_PUSH ( "Saving VBX" );

	verbosef ( "  Saving VBX (ver %d.%d)\n", major, minor );

	const int levels = mPool->getNumLevels();

	const int	num_grids = 1;
	const int	grid_name_len = 256; // Length of grid_name
	char		grid_name[grid_name_len];
	const char	grid_components = 1;						// one component
	const char	grid_dtype = 'f';							// float
	const char	grid_compress = 0;							// no compression
	const char	grid_topotype = 2;							// gvdb topology
	const int	grid_reuse = 0;
	const char	grid_layout = 0;							// atlas layout

	const int		leafcnt = static_cast<int>(mPool->getPoolTotalCnt(0,0));	// brick count
	const int		num_chan = mPool->getNumAtlas();		// number of channels
	const int		leafdimX = getRes(0);
	const Vector3DI leafdim = Vector3DI(leafdimX, leafdimX, leafdimX);	// brick resolution
	const int		apron	= mPool->getAtlas(0).apron;		// brick apron
	const Vector3DI axiscnt = mPool->getAtlas(0).subdim;	// atlas count
	const Vector3DI axisres = mPool->getAtlasRes(0);		// atlas res
	const Vector3DF voxelsize_deprecated(1, 1, 1);			// world units per voxel
	const uint64	atlas_sz = mPool->getAtlas(0).size;		// atlas size

	std::vector<uint64>	grid_offs(num_grids, 0); // All values are initially 0

	//--- VBX file header
	fwrite ( &major, sizeof(uchar), 1, fp );				// major version
	fwrite ( &minor, sizeof(uchar), 1, fp );				// minor version

	if ((major == 1 && minor >= 11) || major > 1) {
		// GVDB 1.11+ saves grid transforms (GVDB 1.1 and earlier do not)
		fwrite( &mPretrans.x, sizeof(float), 3, fp);
		fwrite( &mAngs.x, sizeof(float), 3, fp);
		fwrite( &mScale.x, sizeof(float), 3, fp);
		fwrite( &mTrans.x, sizeof(float), 3, fp);
	}
	
	fwrite ( &num_grids, sizeof(int), 1, fp );				// number of grids - future expansion
	
	// bitmask info
	uchar use_bitmask = 0;
	#ifdef USE_BITMASK
		use_bitmask = 1;
	#endif
	if (major >= 2) {
		fwrite( &use_bitmask, sizeof(uchar), 1, fp );		// bitmask usage (GVDB 2.0 or higher)
	}

	//--- grid offset table
	const long grid_table = ftell ( fp );					// position of grid table in file
	fwrite(grid_offs.data(), sizeof(uint64), grid_offs.size(), fp); // grid offsets (populated later)

	for (int n=0; n < num_grids; n++ ) {
		grid_offs[n] = ftell ( fp );						// record grid offset

		//---- grid header
		fwrite ( &grid_name, 256, 1, fp );					// grid name
		fwrite ( &grid_dtype, sizeof(uchar), 1, fp );		// grid data type
		fwrite ( &grid_components, sizeof(uchar), 1, fp );	// grid components
		fwrite ( &grid_compress, sizeof(uchar), 1, fp );	// grid compression (0=none, 1=blosc, 2=..)
		fwrite ( &voxelsize_deprecated.x, sizeof(float), 3, fp );	// voxel size
		fwrite ( &leafcnt, sizeof(int), 1, fp );			// total brick count
		fwrite ( &leafdim.x, sizeof(int), 3, fp );			// brick dimensions
		fwrite ( &apron, sizeof(int), 1, fp );				// brick apron
		fwrite ( &num_chan, sizeof(int), 1, fp );			// number of channels
		fwrite ( &atlas_sz, sizeof(uint64), 1, fp );		// total atlas size (all channels)
		fwrite ( &grid_topotype, sizeof(uchar), 1, fp );	// topology type? (0=none, 1=reuse, 2=gvdb, 3=..)
		fwrite ( &grid_reuse, sizeof(int), 1, fp);			// topology reuse
		fwrite ( &grid_layout, sizeof(uchar), 1, fp);		// brick layout? (0=atlas, 1=brick)
		fwrite ( &axiscnt.x, sizeof(int), 3, fp );			// atlas axis count
		fwrite ( &axisres.x, sizeof(int), 3, fp );			// atlas res

		//---- topology section
		fwrite ( &levels, sizeof(int), 1, fp );				// num levels
		fwrite ( &mRoot, sizeof(uint64), 1, fp );			// root id
		for (int n=0; n < levels; n++ ) {
			const int res = getRes(n);
			const Vector3DI range = getRange(n);
			const int width0 = static_cast<int>(mPool->getPoolWidth(0, n));
			const int width1 = static_cast<int>(mPool->getPoolWidth(1, n));
			const int cnt0 = static_cast<int>(mPool->getPoolTotalCnt(0, n));
			const int cnt1 = static_cast<int>(mPool->getPoolTotalCnt(1, n));
			fwrite ( &mLogDim[n], sizeof(int), 1, fp );
			fwrite ( &res, sizeof(int), 1, fp );
			fwrite ( &range.x, sizeof(int), 3, fp );
			fwrite ( &cnt0,   sizeof(int), 1, fp );
			fwrite ( &width0, sizeof(int), 1, fp );
			fwrite ( &cnt1,   sizeof(int), 1, fp );
			fwrite ( &width1, sizeof(int), 1, fp );
		}

		// Write topology
		for (int n = 0; n < levels; n++) {
			mPool->PoolWrite(fp, 0, n); // write pool 0
		}
		for (int n = 0; n < levels; n++) {
			mPool->PoolWrite(fp, 1, n); // write pool 1
		}

		//---- atlas section
		// readback slice-by-slice from gpu to conserve CPU and GPU mem

		for (int chan = 0 ; chan < num_chan; chan++ ) {
			DataPtr slice;
			const int chan_type = mPool->getAtlas(chan).type ;
			const int chan_stride = mPool->getSize ( chan_type );

			fwrite ( &chan_type, sizeof(int), 1, fp );
			fwrite ( &chan_stride, sizeof(int), 1, fp );
			mPool->CreateMemLinear ( slice, 0x0, chan_stride, axisres.x*axisres.y, true );

			for (int z = 0; z < axisres.z; z++ ) {
				mPool->AtlasRetrieveSlice ( chan, z, static_cast<int>(slice.size),
					slice.gpu, (uchar*) slice.cpu );		// transfer from GPU, directly into CPU atlas		
				fwrite ( slice.cpu, slice.size, 1, fp );
			}
			mPool->FreeMemLinear ( slice );
		}
	}
	// update grid offsets table
	fseek(fp, grid_table, SEEK_SET);
	fwrite(grid_offs.data(), sizeof(uint64), grid_offs.size(), fp); // grid offsets

	fclose ( fp );

	PERF_POP ();

	POP_CTX
}

void VolumeGVDB::SetBounds(Vector3DF pMin, Vector3DF pMax)
{
	Vector3DI range = getRange(0);

	mVoxMin.x = float(int(pMin.x / range.x) * range.x - range.x);
	mVoxMin.y = float(int(pMin.y / range.y) * range.y - range.y);
	mVoxMin.z = float(int(pMin.z / range.z) * range.z - range.z);

	mVoxMax.x = float(int(pMax.x / range.x) * range.x + 2 * range.x);
	mVoxMax.y = float(int(pMax.y / range.y) * range.y + 2 * range.y);
	mVoxMax.z = float(int(pMax.z / range.z) * range.z + 2 * range.z);

	mObjMin = mVoxMin;
	mObjMax = mVoxMax;
	mVoxRes = mVoxMax;  mVoxRes -= mVoxMin;

	if ( mVoxRes.x > mVoxResMax.x ) mVoxResMax.x = mVoxRes.x;
	if ( mVoxRes.y > mVoxResMax.y ) mVoxResMax.y = mVoxRes.y;
	if ( mVoxRes.z > mVoxResMax.z ) mVoxResMax.z = mVoxRes.z;
}

// Compute bounding box of entire volume.
// - This is done by finding the min/max of all bricks
void VolumeGVDB::ComputeBounds ()
{
	Vector3DI range = getRange(0);
	Node* curr = getNode ( 0, 0, 0 );	
	mVoxMin = curr->mPos;
	mVoxMax = mVoxMin;
	for (int n=0; n < mPool->getPoolTotalCnt(0,0); n++ ) {
		curr = getNode ( 0, 0, n );
		if (!curr->mFlags) continue;		// inactivated, skip
		
		if ( curr->mPos.x < mVoxMin.x ) mVoxMin.x = static_cast<float>(curr->mPos.x);
		if ( curr->mPos.y < mVoxMin.y ) mVoxMin.y = static_cast<float>(curr->mPos.y);
		if ( curr->mPos.z < mVoxMin.z ) mVoxMin.z = static_cast<float>(curr->mPos.z);	
		if ( curr->mPos.x + range.x > mVoxMax.x ) mVoxMax.x = static_cast<float>(curr->mPos.x + range.x);
		if ( curr->mPos.y + range.y > mVoxMax.y ) mVoxMax.y = static_cast<float>(curr->mPos.y + range.y);
		if ( curr->mPos.z + range.z > mVoxMax.z ) mVoxMax.z = static_cast<float>(curr->mPos.z + range.z);		
	}
	mObjMin = mVoxMin;
	mObjMax = mVoxMax;
	mVoxRes = mVoxMax;  mVoxRes -= mVoxMin;

	if ( mVoxRes.x > mVoxResMax.x ) mVoxResMax.x = mVoxRes.x;
	if ( mVoxRes.y > mVoxResMax.y ) mVoxResMax.y = mVoxRes.y;
	if ( mVoxRes.z > mVoxResMax.z ) mVoxResMax.z = mVoxRes.z;
}

#ifdef BUILD_OPENVDB

// Skips to the `leaf_start`th leaf in the file.
template<class GridType>
void vdbSkip(typename GridType::Ptr& grid, typename GridType::TreeType::LeafCIter& iter, int leaf_start) {
	iter = grid->tree().cbeginLeaf();
	for (int j = 0; iter && j < leaf_start; j++) {
		++iter;
	}
}

#endif

// Load a raw BRK file
bool VolumeGVDB::LoadBRK ( std::string fname )
{
	PUSH_CTX

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
	
	PERF_PUSH ( buf );	
	
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
	Configure ( mVCFG[0], mVCFG[1], mVCFG[2], mVCFG[3], mVCFG[4] );
	SetApron ( 1 );
	
	// Create atlas
	mPool->AtlasReleaseAll ();	
	PERF_PUSH ( "Create Atlas" );	
	
	int side = int(ceil(pow(brkcnt, 1 / 3.0f)));		// number of leaves along one axis	
	Vector3DI axiscnt (side, side, side);
	mPool->AtlasCreate ( 0, T_FLOAT, bres, axiscnt, mApron, sizeof(AtlasNode), false, mbUseGLAtlas );
	PERF_POP ();
	
	float vmin = FLT_MAX, vmax = -FLT_MAX;

	// Read all bricks
	Vector3DF t;

	PERF_PUSH ( "Load Bricks" );	
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
				mPool->AtlasCopyTex ( 0, brickpos, vtemp.getPtr() );
			}
			leaf_cnt++;
			t.z += PERF_STOP ();
		}
	}
	PERF_POP ();

	verbosef( "    Read Brk: %f ms\n", t.x );
	verbosef( "    Activate: %f ms\n", t.y );
	verbosef( "    To Atlas: %f ms\n", t.z );

	ComputeBounds ();

	// Commit all node pools to gpu
	PERF_PUSH ( "Commit" );		
	mPool->PoolCommitAll ();		
	PERF_POP ();

	PERF_POP ();

	FinishTopology ();
	UpdateApron ();

	POP_CTX

	return true;
}


#ifdef BUILD_OPENVDB
template<class GridType>
bool VolumeGVDB::LoadVDBInternal(openvdb::GridBase::Ptr& baseGrid) {
	// isFloat is true iff the OpenVDB grid contains floating-point values. If it
	// contains vector values instead, for instance, it will be false.
	const bool isFloat = std::is_same<GridType::ValueType, float>::value;
	// Sanity check
	if (!baseGrid->isType<GridType>()) {
		gprintf("ERROR: Base grid type did not match template in LoadVDBInternal.\n");
		return false;
	}
	GridType::Ptr grid = openvdb::gridPtrCast<GridType>(baseGrid);
	const Vector3DF voxelsize = Vector3DF(
		static_cast<float>(grid->voxelSize().x()),
		static_cast<float>(grid->voxelSize().y()),
		static_cast<float>(grid->voxelSize().z()));
	SetTransform(Vector3DF(0, 0, 0), voxelsize, Vector3DF(0, 0, 0), Vector3DF(0, 0, 0));
	SetApron(1);

	const float poolUsed = MeasurePools();
	verbosef("   Topology Used: %6.2f MB\n", poolUsed);

	int leaf_start = 0;
	int leaf_max = 0;
	Vector3DF vclipmin, vclipmax, voffset;
	vclipmin = getScene()->mVClipMin;
	vclipmax = getScene()->mVClipMax;

	// Determine volume bounds
	verbosef("   Compute volume bounds.\n");
	GridType::TreeType::LeafCIter iterator;
	vdbSkip<GridType>(grid, iterator, leaf_start);
	for (leaf_max = 0; iterator.test(); ) {
		openvdb::Coord origin;
		iterator->getOrigin(origin);
		Vector3DF p0 = Vector3DI(origin.x(), origin.y(), origin.z());
		// Only accept origins within the vclip bounding box
		if (p0.x > vclipmin.x && p0.y > vclipmin.y && p0.z > vclipmin.z && p0.x < vclipmax.x && p0.y < vclipmax.y && p0.z < vclipmax.z) {
			if (leaf_max == 0) {
				mVoxMin = p0; mVoxMax = p0;
			}
			else {
				if (p0.x < mVoxMin.x) mVoxMin.x = p0.x;
				if (p0.y < mVoxMin.y) mVoxMin.y = p0.y;
				if (p0.z < mVoxMin.z) mVoxMin.z = p0.z;
				if (p0.x > mVoxMax.x) mVoxMax.x = p0.x;
				if (p0.y > mVoxMax.y) mVoxMax.y = p0.y;
				if (p0.z > mVoxMax.z) mVoxMax.z = p0.z;
			}
			leaf_max++;
		}
		iterator.next();
	}
	voffset = mVoxMin * -1;		// offset to positive space (hack)	


	// Activate Space
	PERF_PUSH("Activate");
	verbosef("   Activating space.\n");

	vdbSkip<GridType>(grid, iterator, leaf_start);
	for (int n = leaf_max = 0; iterator.test(); n++) {

		// Read leaf position
		openvdb::Coord origin;
		iterator->getOrigin(origin);
		Vector3DF p0 = Vector3DI(origin.x(), origin.y(), origin.z());
		p0 += voffset;

		// only accept those in clip volume
		if (p0.x > vclipmin.x && p0.y > vclipmin.y && p0.z > vclipmin.z && p0.x < vclipmax.x && p0.y < vclipmax.y && p0.z < vclipmax.z) {		// accept condition
			
			bool bnew = false;
			slong leaf = ActivateSpace(mRoot, p0, bnew);
			leaf_ptr.push_back(leaf);
			leaf_pos.push_back(p0);
			if (leaf_max == 0) {
				mVoxMin = p0; mVoxMax = p0;
				verbosef("   First leaf: %d  (%f %f %f)\n", leaf_start + n, p0.x, p0.y, p0.z);
			}
			leaf_max++;
		}
		iterator.next();
	}

	// Finish Topology
	FinishTopology();

	PERF_POP();

	// Resize Atlas
	//verbosef ( "   Create Atlas. Free before: %6.2f MB\n", cudaGetFreeMem() );
	PERF_PUSH("Atlas");
	DestroyChannels();
	AddChannel(0, T_FLOAT, mApron, F_LINEAR);
	UpdateAtlas();
	PERF_POP();
	//verbosef ( "   Create Atlas. Free after:  %6.2f MB, # Leaf: %d\n", cudaGetFreeMem(), leaf_max );

	// Resize temp 3D texture to match leaf res
	int res0 = getRes(0);
	Vector3DI vres0(res0, res0, res0);		// leaf resolution
	Vector3DF vrange0 = getRange(0);

	Volume3D vtemp(mScene);
	vtemp.Resize(T_FLOAT, vres0, 0x0, false);

	// Read brick data
	PERF_PUSH("Read bricks");

	// Advance to starting leaf		
	vdbSkip<GridType>(grid, iterator, leaf_start);

	// Tempoary buffer for velocity fields
	const int64_t res0_64 = static_cast<int64_t>(res0);
	float* srcLengths = new float[res0_64 * res0_64 * res0_64];
	float mValMin, mValMax;
	mValMin = FLT_MAX; mValMax = FLT_MIN;

	// Fill atlas from leaf data
	int percentLast = 0, percent = 0;
	verbosef("   Loading bricks.\n");
	for (int leaf_cnt = 0; iterator.test(); ) {

		// Read leaf position
		openvdb::Coord origin;
		iterator->getOrigin(origin);
		Vector3DF p0 = Vector3DI(origin.x(), origin.y(), origin.z());
		p0 += voffset;

		// Only process nodes whose origins are within the bounding box
		if (p0.x > vclipmin.x && p0.y > vclipmin.y && p0.z > vclipmin.z && p0.x < vclipmax.x && p0.y < vclipmax.y && p0.z < vclipmax.z) {
			// get leaf	
			GridType::TreeType::LeafNodeType::Buffer leafBuffer = iterator->buffer();
			float* src;
			if (isFloat) {
				src = leafBuffer.data();
			}
			else {
				src = ConvertToScalar(res0 * res0 * res0, (float*)leafBuffer.data(), srcLengths, mValMin, mValMax);
			}

			// Copy data from CPU into temporary 3D texture
			vtemp.SetDomain(leaf_pos[leaf_cnt], leaf_pos[leaf_cnt] + vrange0);
			vtemp.CommitFromCPU(src);

			// Copy from 3D texture into Atlas brick
			Node* node = getNode(leaf_ptr[leaf_cnt]);
			mPool->AtlasCopyTexZYX(0, node->mValue, vtemp.getPtr());

			// Progress percent
			leaf_cnt++; percent = int(leaf_cnt * 100 / leaf_max);
			if (percent != percentLast) {
				verbosef("%d%%%% ", percent);
				percentLast = percent;
			}
		}
		iterator.next();
	}

	PERF_POP();
	if (mValMin != FLT_MAX || mValMax != FLT_MIN)
		verbosef("\n    Value Range: %f %f\n", mValMin, mValMax);

	UpdateApron();

	delete[] srcLengths;

	return true;
}
#endif // BUILD_OPENVDB


bool VolumeGVDB::LoadVDB ( std::string fname )
{
	
#ifdef BUILD_OPENVDB

	// Only initialize OpenVDB once
	static bool openVDBInitialized = false;
	if (!openVDBInitialized) {
		openvdb::initialize();
		FloatGrid34::registerGrid();
		// We don't need to register FloatGrid543, as it is part of OpenVDB
		openVDBInitialized = true;
	}

	PERF_PUSH ( "Load VDB" );	

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
	
	PERF_POP ();

	// Switch based on the type of the grid
	bool success = false;
	if (baseGrid->isType<FloatGrid543>()) {
		Configure(5, 5, 5, 4, 3);
		success = LoadVDBInternal<FloatGrid543>(baseGrid);
	}
	else if (baseGrid->isType<Vec3fGrid543>()) {
		Configure(5, 5, 5, 4, 3);
		success = LoadVDBInternal<FloatGrid543>(baseGrid);
	}
	else if (baseGrid->isType<FloatGrid34>()) {
		Configure(3, 3, 3, 3, 4);
		success = LoadVDBInternal<FloatGrid34>(baseGrid);
	}
	else if (baseGrid->isType<Vec3fGrid34>()) {
		Configure(3, 3, 3, 3, 4);
		success = LoadVDBInternal<FloatGrid34>(baseGrid);
	}

	vdbfile->close();
	delete vdbfile;

	return success;
#else

	gprintf ( "ERROR: Unable to load .vdb file. OpenVDB library not linked.\n");
	return false;

#endif

}

#ifdef BUILD_OPENVDB
// Activates all voxels whose value is nonzero.
// GridValueAllIter is e.g. FloatGrid34::ValueAllIter
struct Activator {
	template<class GridValueAllIter>
	static inline void op(const GridValueAllIter& iter) {
		if ( iter.getValue() != 0.0 )
			iter.setActiveState ( true );
	}
};

template<class TreeType>
void VolumeGVDB::SaveVDBInternal(std::string& fname) {
	// Raw pointer to tree
	TreeType* treePtr = new TreeType(0.0);
	TreeType::Ptr tree(treePtr);

	// `typename` here doesn't change the type - it just lets the
	// compiler know that these are types, and not values:
	using GridType = typename openvdb::Grid<TreeType>;
	using IterType = typename GridType::ValueAllIter;
	using RootType = typename TreeType::RootNodeType;
	using LeafType = typename TreeType::LeafNodeType;
	using ValueType = typename LeafType::ValueType;

	LeafType* leaf;
	const int res = getRes(0);
	const uint64 bytesPerLeaf = getVoxCnt(0) * sizeof(float); // = (res^3) floats

	DataPtr p;
	mPool->CreateMemLinear(p, 0x0, 1, bytesPerLeaf, true);

	const uint64 leafcnt = mPool->getPoolTotalCnt(0, 0); // Leaf count
	Node* node;
	float minValue = FLT_MAX;
	float maxValue = FLT_MIN;
	for (uint64 n = 0; n < leafcnt; n++) {
		node = getNode(0, 0, n);
		Vector3DI pos = node->mPos;
		// Get a pointer to this node
		leaf = tree->touchLeaf(openvdb::Coord(pos.x, pos.y, pos.z));
		leaf->setValuesOff();
		ValueType* leafBuffer = leaf->buffer().data();

		// Get the brick from the GPU
		mPool->AtlasRetrieveTexXYZ(0, node->mValue, p);

		// Set the leaf voxels
		memcpy(leafBuffer, p.cpu, bytesPerLeaf);
	}
	verbosef("  Leaf count: %d\n", tree->leafCount());

	// Now, create the grid from the tree and activate it
	PERF_PUSH("Creating grid");
	GridType::Ptr grid = GridType::create(tree);
	grid->setGridClass(openvdb::GRID_FOG_VOLUME);
	grid->setName("density");
	grid->setTransform(openvdb::math::Transform::createLinearTransform(1.0));
	grid->addStatsMetadata();
	// VS2019: Ignore warning about OpenVDB using const in a template argument
#pragma warning(push)
#pragma warning(disable:4180)
	openvdb::tools::foreach(grid->beginValueAll(), Activator::op<IterType>);
#pragma warning(pop)
	verbosef("  Leaf count: %d\n", grid->tree().leafCount());
	PERF_POP();

	PERF_PUSH("Writing grids");
	openvdb::io::File* vdbfile = new openvdb::io::File(fname);
	vdbfile->setGridStatsMetadataEnabled(true);
	vdbfile->setCompression(openvdb::io::COMPRESS_NONE);

	openvdb::GridPtrVec grids;
	grids.push_back(grid);
	vdbfile->write(grids);
	vdbfile->close();
	PERF_POP();
}
#endif

void VolumeGVDB::SaveVDB ( std::string fname )
{
#ifdef BUILD_OPENVDB
	// Verify that the atlas's datatype is T_FLOAT
	if (mPool->getAtlas(0).type != T_FLOAT) {
		gprintf("ERROR: Unable to save .vdb file. Atlas channel 0 must have "
			"type T_FLOAT in order to export correctly.\n");
		return;
	}

	const int logRes = getLD(0);
	if (logRes == 3) {
		SaveVDBInternal<TreeType543F>(fname);
	}
	else if (logRes == 4) {
		SaveVDBInternal<TreeType34F>(fname);
	}
	else {
		gprintf("ERROR: Unable to save .vdb file. No configurations available for given log res.\n");
	}
	return;
#else
	gprintf("ERROR: Unable to save .vdb file. OpenVDB library not linked.\n");
	return;
#endif
}

// Add a search path for assets
void VolumeGVDB::AddPath ( const char* path )
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
	PUSH_CTX

	// Create Pool Allocator
	verbosef("Starting GVDB Voxels. ver %d.%d\n", MAJOR_VERSION, MINOR_VERSION );
	verbosef(" Creating Allocator..\n");
	mPool = new Allocator;	
	mPool->SetStream(mStream);
	mPool->SetDebug(mbDebug);

	// Create Scene object
	verbosef(" Creating Scene..\n");
	mScene = new Scene;		

	// Create VDB object
	cudaCheck(cuMemAlloc(&cuVDBInfo, sizeof(VDBInfo)), "VolumeGVDB", "Initialize", "cuMemAlloc", "cuVDBInfo", mbDebug);

	// Default Camera & Light
	mScene->SetCamera ( new Camera3D );		// Default camera
	mScene->SetLight ( 0, new Light );		// Default light
	mScene->SetVolumeRange ( 0.1f, 0, 1 );	// Default transfer range
	mScene->LinearTransferFunc ( 0, 1, Vector4DF(0,0,0,0), Vector4DF(1,1,1,0.1f) );		// Default transfer function
	
	// Default transfer function
	CommitTransferFunc ();	

	// Default channel config
	SetChannelDefault ( 16, 16, 1 );

	// Default paths
	AddPath ( "../source/shared_assets/" );	
	AddPath ( "../shared_assets/" );
	
	
	mRebuildTopo = true;
	mCurrDepth = -1;

	mHasObstacle = false;

	POP_CTX
}

// Configure VDB tree (5-level)
void VolumeGVDB::Configure ( int q4, int q3, int q2, int q1, int q0 )
{
	int r[5], n[5];
	r[0] = q0; r[1] = q1; r[2] = q2; r[3] = q3; r[4] = q4;

	n[0] = 4;		// leaf max
	int cnt = 4;
	n[1] = cnt;		cnt >>= 1;
	n[2] = cnt;		cnt >>= 1;
	n[3] = cnt;		cnt >>= 1;
	n[4] = cnt;
	
	Configure ( 5, r, n );	
}

// Configure VDB tree (N-level)
void VolumeGVDB::Configure ( int levs, int* r, int* numcnt, bool use_masks )
{
	PUSH_CTX

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
	mClrDim[8] = Vector3DF(0.7f,0.7f,0.7f);  // grey

	// Initialize memory pools
	uint64 hdr = sizeof(Node);

	mPool->PoolReleaseAll();

	#ifdef USE_BITMASKS	 
		use_masks = true;
	#endif

	// node & mask list	
	uint64 nodesz;
	for (int n = 0; n < levs; n++) {
		nodesz = hdr;
		if ( use_masks ) nodesz += getMaskSize(n);
		mPool->PoolCreate(0, n, nodesz, maxcnt[n], true);
	}

	// child lists	
	mPool->PoolCreate ( 1, 0, 0, 0, true );								
	for (int n=1; n < levs; n++ ) 	
		mPool->PoolCreate ( 1, n, sizeof(uint64)*getVoxCnt(n), maxcnt[n], true );

	mVoxResMax.Set ( 0, 0, 0 );

	// Clear tree and create default root
	Clear ();

	POP_CTX
}

// Add a data channel (voxel attribute)
void VolumeGVDB::AddChannel ( uchar chan, int dt, int apron, int filter, int border, Vector3DI axiscnt )
{
	PUSH_CTX

	if (axiscnt.x==0 && axiscnt.y==0 && axiscnt.z==0) {
		if ( chan == 0 ) 	axiscnt = mDefaultAxiscnt;
		else				axiscnt = mPool->getAtlas(0).subdim;
	}
	mApron = apron;

	mPool->AtlasCreate ( chan, dt, getRes3DI(0), axiscnt, apron, sizeof(AtlasNode), false, mbUseGLAtlas );
	mPool->AtlasSetFilter ( chan, filter, border );

	SetupAtlasAccess ();	

	POP_CTX
}

// Fill data channel
void VolumeGVDB::FillChannel ( uchar chan, Vector4DF val )
{
	PUSH_CTX

	if (val.x == 0 && val.y==0 && val.z==0 && val.w==0) {
		ClearChannel(chan);
	} else {
		switch (mPool->getAtlas(chan).type) {
		case T_FLOAT:	Compute(FUNC_FILL_F, chan, 1, val, false, false);	break;
		case T_UCHAR:	Compute(FUNC_FILL_C, chan, 1, val, false, false);	break;
		case T_UCHAR4:	Compute(FUNC_FILL_C4, chan, 1, val, false, false);	break;
		};
	}

	POP_CTX
}

// Destroy all channels
void VolumeGVDB::DestroyChannels ()
{
	PUSH_CTX

	if (mPool) {
		mPool->AtlasReleaseAll();
	}
	SetColorChannel ( -1 );

	POP_CTX
}

// Clear GVDB without changing config
void VolumeGVDB::Clear ()
{
	PUSH_CTX

	// Empty VDB data (keep pools)
	mPool->PoolEmptyAll ();		// does not free pool mem
	mRoot = ID_UNDEFL;

	// Empty atlas & atlas map
	mPool->AtlasEmptyAll ();	// does not free atlas
	
	mPnt.Set ( 0, 0, 0 );

	mRebuildTopo = true;			// full rebuild required

	POP_CTX
}

// Allocate a new VDB node
slong VolumeGVDB::AllocateNode ( int lev )
{
	return mPool->PoolAlloc ( 0, lev, true );	
}

// Setup a VDB node
void VolumeGVDB::SetupNode ( slong nodeid, int lev, Vector3DF pos, bool marker)
{
	Node* node = getNode ( nodeid );
	node->mLev = lev;	
	node->mPos = pos;	
	node->mChildList = ID_UNDEFL;
	node->mParent = ID_UNDEFL;
	node->mValue = Vector3DI(-1,-1,-1);
	node->mFlags = marker;
#ifdef USE_BITMASKS
	if ( lev > 0 ) node->clearMask ();
#endif
}


void VolumeGVDB::UpdateNeighbors()
{
	PUSH_CTX

	uint64 brks = mPool->getPoolTotalCnt(0, 0);
	
	mPool->AllocateNeighbors( brks );

	Vector3DF ct;
	float d = static_cast<float>(mPool->getAtlasBrickwid(0));

	DataPtr* ntable = mPool->getNeighborTable();
	int* nbr = (int*) ntable->cpu;
	Node* node;

	for (uint64 n = 0; n < brks; n++) {
		node = getNode(0, 0, n);
		ct = node->mPos + Vector3DF(d / 2, d / 2, d / 2);
		*nbr++ = static_cast<int>(ElemNdx(getNodeAtPoint(mRoot, ct - Vector3DF(d, 0, 0))));
		*nbr++ = static_cast<int>(ElemNdx(getNodeAtPoint(mRoot, ct - Vector3DF(0, d, 0))));
		*nbr++ = static_cast<int>(ElemNdx(getNodeAtPoint(mRoot, ct - Vector3DF(0, 0, d))));
		*nbr++ = static_cast<int>(ElemNdx(getNodeAtPoint(mRoot, ct + Vector3DF(d, 0, 0))));
		*nbr++ = static_cast<int>(ElemNdx(getNodeAtPoint(mRoot, ct + Vector3DF(0, d, 0))));
		*nbr++ = static_cast<int>(ElemNdx(getNodeAtPoint(mRoot, ct + Vector3DF(0, 0, d))));
	}

	mPool->CommitNeighbors();

	POP_CTX
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
				an = (AtlasNode*) mPool->getAtlasMapNode ( 0, brickpos );
				an->mLeafNode = ID_UNDEFL;
				an->mPos.Set ( ID_UNDEFL, ID_UNDEFL, ID_UNDEFL );
			}
		}
	}
}

// Assign an atlas mapping
void VolumeGVDB::AssignMapping ( Vector3DI brickpos, Vector3DI pos, int leafid )
{
	AtlasNode* an = (AtlasNode*) mPool->getAtlasMapNode ( 0, brickpos );
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
		r = static_cast<float>(z.Length());
		if ( r > bail ) break;

		theta = asin(z.z/r) * pwr;
		phi = atan2(z.y, z.x) * pwr;
		zr = powf(r, pwr - 1.0f );
		dr = zr*pwr*dr + 1.0f;
		zr *= r;		

		z = Vector3DF( cos(theta)*cos(phi), cos(theta)*sin(phi), sin(theta) );
		z *= zr;
		z += s;
	}
	return -0.5f*log(r)*r / dr;
}

void VolumeGVDB::CopyChannel(int chanDst, int chanSrc)
{
	PUSH_CTX
	mPool->CopyChannel(chanDst, chanSrc);
	POP_CTX
}

// Update the atlas
// - Resize atlas if needed
// - Assign new nodes to atlas
// - Update atlas mapping and pools
void VolumeGVDB::UpdateAtlas ()
{
	PUSH_CTX

	PERF_PUSH ( "Update Atlas" );

	Vector3DI brickpos;
	Node* node;
	uint64 totalLeafcnt = mPool->getPoolTotalCnt(0,0);
	uint64 usedLeafcnt = mPool->getPoolUsedCnt(0,0);
	
	mPool->AtlasEmptyAll ();

	// Resize atlas
	uint64 amax = mPool->getAtlas(0).max;
	if ( usedLeafcnt > amax ) {
		PERF_PUSH ( "Resize Atlas" );
		for (int n=0; n < mPool->getNumAtlas(); n++ ) 
			mPool->AtlasResize ( n, usedLeafcnt );
		SetupAtlasAccess ();							// must reassign surf/obj access objects to new glids
		PERF_POP ();
	}

	// Assign new nodes to atlas
	PERF_PUSH ( "Assign Atlas" );
	for (uint64 n=0; n < totalLeafcnt; n++ ) {
		node = getNode ( 0, 0, n );
		if (!node->mFlags) continue;
			if ( mPool->AtlasAlloc ( 0, brickpos ) )	// assign to atlas brick
				node->mValue = brickpos;
		
	}
	PERF_POP ();

	// Reallocate Atlas Map (if needed)	
	// -- Note: subdim = full size of atlas including unused bricks
	PERF_PUSH ( "Allocate Atlas" );
	mPool->AllocateAtlasMap ( sizeof(AtlasNode), mPool->getAtlas(0).subdim );
	PERF_POP ();

	// ensure mapping for unused bricks
	PERF_PUSH ( "Clear mapping" );
	ClearMapping ();
	PERF_POP ();

	// Build Atlas Mapping	
	PERF_PUSH ( "Atlas Mapping" );
	//int brickcnt = totalLeafcnt;//mPool->getAtlas(0).lastEle;
	int brickres = mPool->getAtlasBrickres(0);
	Vector3DI atlasres = mPool->getAtlasRes(0);
	Vector3DI atlasmax = atlasres - brickres + mPool->getAtlas(0).apron; 
	for (uint64 n=0; n < totalLeafcnt; n++ ) {
		Node* node = getNode ( 0, 0, n );
		if (!node->mFlags) continue;
		if ( node->mValue.x > atlasmax.x || node->mValue.y > atlasmax.y || node->mValue.z > atlasmax.z ) {
			gprintf ( "ERROR: Node value exceeds atlas res. node: %d, val: %d %d %d, atlas: %d %d %d\n", n, node->mValue.x, node->mValue.y, node->mValue.z, atlasres.x, atlasres.y, atlasres.z );
			gerror ();
		}
		AssignMapping ( node->mValue, node->mPos, static_cast<int>(n) );
	}
	PERF_POP ();
	

	// Commit to GPU
	PERF_PUSH ( "Commit Atlas Map" );
	mPool->PoolCommitAtlasMap ();		// Commit Atlas Map (HtoD)
	
	mPool->PoolCommit ( 0, 0 );			// Commit brick nodes *with new values* (HtoD)
	
	PERF_POP ();

	PERF_POP ();

	POP_CTX
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
	Vector3DI brickpos;
	bool bnew = false;
	slong node_id = ActivateSpace ( mRoot, pos, bnew, ID_UNDEFL, 0 );
	if ( node_id == ID_UNDEFL ) return ID_UNDEFL;
	if ( !bnew ) return node_id; // exiting node. return
	return ID_UNDEFL;
}

// Activate region of space at 3D position down to a given level
slong VolumeGVDB::ActivateSpaceAtLevel ( int lev, Vector3DI pos )
{
	Vector3DI brickpos;
	bool bnew = false;
	slong node_id = ActivateSpace ( mRoot, pos, bnew, ID_UNDEFL, lev );		// specify level to stop
	if ( node_id == ID_UNDEFL ) return ID_UNDEFL;
	if ( !bnew ) return node_id; // exiting node. return
	return ID_UNDEFL;
}

bool VolumeGVDB::isOn (slong nodeid, uint32 b )
{
	#ifdef USE_BITMASKS
		Node* node = getNode(noideid);
		return node->isOn( b );
	#else
		uint64 cid = getChildNode ( nodeid, b );
		return (cid != ID_UNDEF64 );
	#endif
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
			if (pos.x == sp.x && pos.y == sp.y && pos.z == sp.z && !isOn(nodeid, b) && curr->mLev == stopn->mLev + 1) {
				// if same position as stopnode, bit is not on, 
				return InsertChild( nodeid, stopnode, b);
			}
		}
		// check stop level
		if ( curr->mLev == stoplev ) return nodeid;

		// point is inside this node, add children
		if ( !isOn (nodeid, b ) ) {		// not on yet - create new child
			childid = AddChildNode ( nodeid, curr->mPos, curr->mLev, b, pos );
			if ( curr->mLev==1 ) bNew = true;
		} else {						// already on - insert into existing child
			childid = getChildNode ( nodeid, b );	
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
	int ndx = static_cast<int>(ElemNdx(nodeid));
	
	if ( curr->mLev < 4 && curr->mParent != ID_UNDEFL ) {
		Vector3DF range = getRange ( curr->mLev+1 );
		Vector3DI p = curr->mPos; p -= getNode(curr->mParent)->mPos; p *= getRes3DI ( curr->mLev+1 ); p /= range;
		b = getBitPos ( curr->mLev+1, p );
	}
#ifdef USE_BITMASKS
	if ( curr->mLev > 0 ) {		
		gprintf ( "%*s L%d #%d, Bit: %d, Pos: %d %d %d, Mask: %s\n", (5-curr->mLev)*2, "", (int) curr->mLev, ndx, b, curr->mPos.x, curr->mPos.y, curr->mPos.z, binaryStr( *curr->getMask() ) );
		for (int n=0; n < curr->getNumChild(); n++ ) {
			slong childid = getChildNode ( nodeid, n );
			DebugNode ( childid );
		}
	} else {
		gprintf ( "%*s L%d #%d, Bit: %d, Pos: %d %d %d, Atlas: %d %d %d\n", (5-curr->mLev)*2, "", (int) curr->mLev, ndx, b, curr->mPos.x, curr->mPos.y, curr->mPos.z, curr->mValue.x, curr->mValue.y, curr->mValue.z);
	}
#endif
}

// Get the index-space position of the corner of a node covering the given 'pos'
Vector3DI VolumeGVDB::GetCoveringNode ( int lev, Vector3DI pos, Vector3DI& range )
{
	Vector3DI nodepos;
	range = getRange( lev );			// range of node

	if ( lev == MAXLEV-1 ) {		
		nodepos = Vector3DI(0,0,0);
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

slong VolumeGVDB::FindParent(int lev,  Vector3DI pos)
{
	if(mRoot == ID_UNDEFL) return ID_UNDEFL;

	Node* nd = getNode(mRoot);
	int l = nd->mLev;

	if (l == lev) return mRoot;

	Vector3DI range, res;
	Vector3DF posInNode;
	uint32 bitMaskPos;
	while (l > lev)
	{
		res = getRes3DI(l);
		range = getRange(l);
		posInNode = pos - nd->mPos;
		posInNode *= res;
		posInNode /= range;
		int posInNodeX = static_cast<int>(floor(posInNode.x)); // IMPORTANT!!! truncate decimal 
		int posInNodeY = static_cast<int>(floor(posInNode.y)); // IMPORTANT!!! truncate decimal 
		int posInNodeZ = static_cast<int>(floor(posInNode.z)); // IMPORTANT!!! truncate decimal 
		bitMaskPos = (posInNodeZ*res.x + posInNodeY)*res.x+ posInNodeX;
#ifdef USE_BITMASKS
		uint64 p = nd->countOn ( bitMaskPos );

		l--;

		uint64* clist = mPool->PoolData64(nd->mChildList);
		if (l == lev) return clist[p];

		nd = getNode(clist[p]);
#else
		l--;

		uint64* clist = mPool->PoolData64(nd->mChildList);
		if (l == lev) return clist[bitMaskPos];

		nd = getNode(clist[bitMaskPos]);
#endif
	}

	return ID_UNDEFL;
}


// Insert child into a child list
slong VolumeGVDB::InsertChild ( slong nodeid, slong childid, uint32 i )
{
	Node* curr = getNode ( nodeid );
	Node* child = getNode ( childid );
	child->mParent = nodeid;	// set parent of child

#ifdef USE_BITMASKS
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
#else
	if ( curr->mChildList == ID_UNDEFL ) {
		curr->mChildList = mPool->PoolAlloc ( 1, curr->mLev, true );	
		uint64* clist = mPool->PoolData64 ( curr->mChildList );			
		memset(clist, 0xFF, sizeof(uint64) * getVoxCnt(curr->mLev));
	}

	// insert into child list
	uint64* clist = mPool->PoolData64 ( curr->mChildList );
	clist[i] = childid;		
	
#endif
	return childid;
}

// Get child node at index
Node* VolumeGVDB::getChild(Node* curr, uint ndx)
{
	if (curr->mChildList == ID_UNDEFL) return 0x0;
	uint64* clist = mPool->PoolData64(curr->mChildList);
	uint64* cnext = (uint64*) mPool->PoolData ( ElemGrp(curr->mChildList), ElemLev(curr->mChildList), ElemNdx(curr->mChildList)+1 );
	uint64 ch;
	#ifdef USE_BITMASKS
		ch = *(clist + ndx);
	#else
		int vox = 0;
		uint sum = 0;
		ndx++;
		for (; sum < ndx && vox < getVoxCnt(curr->mLev); vox++ ) {
			if ( *(clist + vox) != ID_UNDEF64) sum++;
		}		
		if (vox >= getVoxCnt(curr->mLev)) return 0x0;		
		ch = *(clist + (vox-1) );		
	#endif
	return getNode(ch);
}

// Get child node at bit position
Node* VolumeGVDB::getChildAtBit (Node* curr, uint b)
{
	if (curr->mChildList == ID_UNDEFL) return 0x0;
	uint64* clist = mPool->PoolData64(curr->mChildList);
#ifdef USE_BITMASKS
	uint32 ndx = curr->countOn(b);
	uint64 ch = *(clist + ndx);	
#else		
	uint64 ch = *(clist + b);
#endif
	if (ch == ID_UNDEF64) return 0x0;
	return getNode(ch);
}

uint64 VolumeGVDB::getChildRefAtBit(Node* curr, uint b)
{
	if (curr->mChildList == ID_UNDEFL) return ID_UNDEF64;
	uint64* clist = mPool->PoolData64(curr->mChildList);
#ifdef USE_BITMASKS
	uint32 ndx = curr->countOn(b);
	return *(clist + ndx);
#else
	return *(clist + b);
#endif
}

// Get child node at bit position
uint64 VolumeGVDB::getChildNode ( slong nodeid, uint b )
{
	Node* curr = getNode ( nodeid );
	if (curr->mChildList == ID_UNDEFL) return ID_UNDEF64;	
	uint64* clist = mPool->PoolData64 ( curr->mChildList );

#ifdef USE_BITMASKS
	uint32 ndx = curr->countOn ( b );	
	return *(clist + ndx);
#else		
	return *(clist + b);
#endif

}

// Get child a given local 3D brick position
uint32 VolumeGVDB::getChildOffset ( slong nodeid, slong childid, Vector3DI& pos )
{
	Node* curr = getNode ( nodeid );
	Node* child = getNode ( childid );
	pos = child->mPos;
	pos -= curr->mPos;
	pos *= getRes3DI ( curr->mLev );
	pos /= getRange ( curr->mLev );
	return getBitPos ( curr->mLev, pos );
}


void VolumeGVDB::writeCube (  FILE* fp, unsigned char vpix[], slong& numfaces, int gv[], int*& vgToVert, slong vbase )
{
	// find absolute vert indices for this cube
	// * map from local grid index to local vertex id, then add base index for this leaf	
	slong v[8];
	vbase++;	// * .obj file indices are base 1;
	v[0] = vbase + vgToVert[ gv[0] ];
	v[1] = vbase + vgToVert[ gv[1] ];
	v[2] = vbase + vgToVert[ gv[2] ];
	v[3] = vbase + vgToVert[ gv[3] ];
	v[4] = vbase + vgToVert[ gv[4] ];
	v[5] = vbase + vgToVert[ gv[5] ];
	v[6] = vbase + vgToVert[ gv[6] ];
	v[7] = vbase + vgToVert[ gv[7] ];

	if ( vpix[1] == 0 ) {fprintf(fp, "f %lld//0 %lld//0 %lld//0 %lld//0\n", v[1], v[2], v[6], v[5] );  numfaces++; }	// x+	
	if ( vpix[2] == 0 ) {fprintf(fp, "f %lld//1 %lld//1 %lld//1 %lld//1\n", v[0], v[3], v[7], v[4] );  numfaces++; }	// x-
	if ( vpix[3] == 0 ) {fprintf(fp, "f %lld//2 %lld//2 %lld//2 %lld//2\n", v[2], v[3], v[7], v[6] );  numfaces++; }	// y+
	if ( vpix[4] == 0 ) {fprintf(fp, "f %lld//3 %lld//3 %lld//3 %lld//3\n", v[1], v[0], v[4], v[5] );  numfaces++; }	// y- 
	if ( vpix[5] == 0 ) {fprintf(fp, "f %lld//4 %lld//4 %lld//4 %lld//4\n", v[4], v[5], v[6], v[7] );  numfaces++; }	// z+
	if ( vpix[6] == 0 ) {fprintf(fp, "f %lld//5 %lld//5 %lld//5 %lld//5\n", v[0], v[1], v[2], v[3] );  numfaces++; }	// z-	
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
	//NOT IMPLEMENTED
	//TODO - IMPLEMENT OR RAISE EXCEPTION
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
		makeSimpleShaderGL (mScene, "simple.vert.glsl", "simple.frag.glsl");
		makeVoxelizeShader  ( mScene, "voxelize.vert.glsl", "voxelize.frag.glsl", "voxelize.geom.glsl" );
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
	PUSH_CTX

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
	POP_CTX
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
	e.imax = (node->mPos / range) + getRes3DI(e.lev) - Vector3DI(1,1,1);	
	e.vmin = Vector3DF(e.imin) * e.cover;				// world-space bounds of children
	e.vmax = Vector3DF(e.imax + Vector3DI(1,1,1)) * e.cover;	
	e.ires = e.imax; e.ires -= e.imin; e.ires += Vector3DI(1, 1, 1);
	e.icnt = e.ires.x * e.ires.y * e.ires.z;
	return e;		
}

// Map an OpenGL VBO to auxiliary geometry
void VolumeGVDB::AuxGeometryMap ( Model* model, int vertaux, int elemaux )
{
	PUSH_CTX

	DataPtr* vaux = &mAux[vertaux];
	DataPtr* eaux = &mAux[elemaux];
	size_t vsize, esize;

    cudaCheck(cuGraphicsGLRegisterBuffer( &vaux->grsc, model->vertBufferID, CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY), "VolumeGVDB", "AuxGeometryMap", "cuGraphicsGLRegisterBuffer", "vaux", mbDebug);
    cudaCheck(cuGraphicsMapResources(1, &vaux->grsc, 0), "VolumeGVDB", "AuxGeometryMap", "cuGraphicsMapResources", "vaux", mbDebug );
    cudaCheck(cuGraphicsResourceGetMappedPointer ( &vaux->gpu, &vsize, vaux->grsc ), "VolumeGVDB", "AuxGeometryMap", "cuGraphicsResourceGetMappedPointer ", "vaux", mbDebug );
	vaux->lastEle = model->vertCount;
	vaux->usedNum = model->vertCount;
	vaux->stride = model->vertStride;
	vaux->size = vsize;

    cudaCheck(cuGraphicsGLRegisterBuffer( &eaux->grsc, model->elemBufferID, CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY), "VolumeGVDB", "AuxGeometryMap", "cuGraphicsGLRegisterBuffer", "eaux", mbDebug);
    cudaCheck(cuGraphicsMapResources(1, &eaux->grsc, 0), "VolumeGVDB", "AuxGeometryMap", "cuGraphicsMapResources", "eaux", mbDebug );
    cudaCheck(cuGraphicsResourceGetMappedPointer ( &eaux->gpu, &esize, eaux->grsc ), "VolumeGVDB", "AuxGeometryMap", "cuGraphicsResourceGetMappedPointer", "eaux", mbDebug);
	eaux->lastEle = model->elemCount;
	eaux->usedNum = model->elemCount;
	eaux->stride = model->elemStride;
	eaux->size = esize;
	
	POP_CTX
}

// Unmap an OpenGL VBO
void VolumeGVDB::AuxGeometryUnmap ( Model* model, int vertaux, int elemaux )
{
	PUSH_CTX

	DataPtr* vaux = &mAux[vertaux];
	DataPtr* eaux = &mAux[elemaux];
	
    cudaCheck(cuGraphicsUnmapResources(1, &vaux->grsc, 0), "VolumeGVDB", "AuxGeometryUnmap", "cuGraphicsUnmapResources", "vaux", mbDebug);
    cudaCheck(cuGraphicsUnmapResources(1, &eaux->grsc, 0), "VolumeGVDB", "AuxGeometryUnmap", "cuGraphicsUnmapResources", "eaux", mbDebug);

	POP_CTX
}

 void VolumeGVDB::Synchronize() {
	PUSH_CTX
		cuCtxSynchronize();
	POP_CTX
}

// Activate a region of space
int VolumeGVDB::ActivateRegion ( int lev, Extents& e )
{
	Vector3DI pos;
	uint64 leaf;		
	int cnt = 0;
	assert ( lev == e.lev-1 );		// make sure extens match desired level
	for (int z=e.imin.z; z <= e.imax.z; z++ )
		for (int y=e.imin.y; y <= e.imax.y; y++ )
			for (int x=e.imin.x; x <= e.imax.x; x++ ) {
				pos.Set(x, y, z); pos *= e.cover;
				leaf = ActivateSpaceAtLevel ( e.lev-1, pos );
				cnt++;
			}

	return cnt;
}

int VolumeGVDB::ActivateHalo(Extents& e)
{
	Vector3DI pos, nbr;
	int cnt, actv;

	// tag buffer
	int sz = e.ires.x*e.ires.y*e.ires.z * sizeof(uchar);
	uchar* tagbuf = (uchar*) malloc( sz );
	memset(tagbuf, 0, sz);

	// Tag halo regions
	cnt = 0;
	for (int z = e.imin.z; z <= e.imax.z; z++)
		for (int y = e.imin.y; y <= e.imax.y; y++)
			for (int x = e.imin.x; x <= e.imax.x; x++) {
				pos.Set(x, y, z); pos *= e.cover;
				if (!isActive(pos)) {
					actv = 0;
					nbr = pos + Vector3DF(-1, 0, 0)*e.cover;	if (isActive(nbr)) actv++;
					nbr = pos + Vector3DF(1, 0, 0)*e.cover;		if (isActive(nbr)) actv++;
					nbr = pos + Vector3DF(0, -1, 0)*e.cover;	if (isActive(nbr)) actv++;
					nbr = pos + Vector3DF(0, 1, 0)*e.cover;		if (isActive(nbr)) actv++;
					nbr = pos + Vector3DF(0, 0, -1)*e.cover;	if (isActive(nbr)) actv++;
					nbr = pos + Vector3DF(0, 0, 1)*e.cover;		if (isActive(nbr)) actv++;
					if (actv > 0) {
						tagbuf[( (z - e.imin.z)*e.ires.y + (y - e.imin.y) )*e.ires.x + (x - e.imin.x)] = 1;
					} 
				}
			}

	// Activate space at tags
	for (int z = e.imin.z; z <= e.imax.z; z++)
		for (int y = e.imin.y; y <= e.imax.y; y++) 
			for (int x = e.imin.x; x <= e.imax.x; x++) {
				pos.Set(x, y, z); pos *= e.cover;
				if (tagbuf[((z - e.imin.z)*e.ires.y + (y - e.imin.y))*e.ires.x + (x - e.imin.x)] > 0) {
					ActivateSpaceAtLevel(e.lev - 1, pos);
					cnt++;
				}
			}



	free(tagbuf);

	return cnt;
}

// Activate a region of space from an auxiliary byte buffer
int VolumeGVDB::ActivateRegionFromAux (Extents& e, int auxid, uchar dt, float vthresh)
{
	Vector3DI pos;
	uint64 leaf;
	char* vdat = mAux[auxid].cpu;			// Get AUX data		
	int cnt = 0;

	switch (dt) {
	case T_UCHAR: {
		for (int z = e.imin.z; z <= e.imax.z; z++)
			for (int y = e.imin.y; y <= e.imax.y; y++)
				for (int x = e.imin.x; x <= e.imax.x; x++) {					
					uchar vset = *(vdat + ((z - e.imin.z)*e.ires.y + (y - e.imin.y))*e.ires.x + (x - e.imin.x));
					if (vset > vthresh) { 
						pos.Set(x, y, z); pos *= e.cover;
						leaf = ActivateSpaceAtLevel(e.lev - 1, pos); cnt++; 
					}
				}
	} break;
	case T_FLOAT: {
		for (int z = e.imin.z; z <= e.imax.z; z++)
			for (int y = e.imin.y; y <= e.imax.y; y++)
				for (int x = e.imin.x; x <= e.imax.x; x++) {					
					float vset = *(((float*)vdat) + ((z - e.imin.z)*e.ires.y + (y - e.imin.y))*e.ires.x + (x - e.imin.x));
					if (vset > vthresh) {
						pos.Set(x, y, z); pos *= e.cover;
						leaf = ActivateSpaceAtLevel(e.lev - 1, pos); cnt++;
					}
				}
	} break;
	};
	return cnt;
}

// Check data returned from GPU
void VolumeGVDB::CheckData ( std::string msg, CUdeviceptr ptr, int dt, int stride, int cnt )
{
	PUSH_CTX

	char* dat = (char*) malloc ( stride*cnt );
	cudaCheck ( cuMemcpyDtoH ( dat, ptr, stride*cnt ), "VolumeGVDB", "CheckData", "cuMemcpyDtoH", "", mbDebug);

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
	POP_CTX
}

// Voxelize a node from polygons
int VolumeGVDB::VoxelizeNode ( Node* node, uchar chan, Matrix4F* xform, float bdiv, float val_surf, float val_inside, float vthresh  )
{
	int cnt = 0;
	Extents e = ComputeExtents ( node );

	// CUDA voxelize
	uchar dt = mPool->getAtlas(chan).type;				
	PrepareAux ( AUX_VOXELIZE, e.icnt, mPool->getSize(dt), true, true );	//   Prepare buffer for data retrieve		

	Vector3DI block ( 8, 8, 8 );
	Vector3DI grid ( int(e.ires.x/block.x) + 1, int(e.ires.y/block.y) + 1, int(e.ires.z/block.z) + 1 );
	int bmax = static_cast<int>(mAux[AUX_GRIDOFF].lastEle);
	void* args[12] = {  &e.vmin, &e.vmax, &e.ires, &mAux[AUX_VOXELIZE].gpu, &dt, &val_surf, &val_inside, 
						&bdiv, &bmax, &mAux[AUX_GRIDCNT].gpu, &mAux[AUX_GRIDOFF].gpu, &mAux[AUX_TRI_BUF].gpu };

	cudaCheck(cuLaunchKernel(cuFunc[FUNC_VOXELIZE], grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args, NULL),
							"VolumeGVDB", "VoxelizeNode", "cuLaunch", "FUNC_VOXELIZE", mbDebug );

	if ( node->mLev==0 ) {
		mPool->AtlasCopyLinear ( chan, node->mValue, mAux[AUX_VOXELIZE].gpu );		
	} else {
		RetrieveData ( mAux[AUX_VOXELIZE] );				
		cnt = ActivateRegionFromAux ( e, AUX_VOXELIZE, dt, vthresh);		
	} 

	#ifdef VOX_GL
		// OpenGL voxelize	
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
		} 
	#endif	

	return cnt;
}

// SolidVoxelize - Voxelize a polygonal mesh to a sparse volume
void VolumeGVDB::SolidVoxelize ( uchar chan, Model* model, Matrix4F* xform, float val_surf, float val_inside, float vthresh )
{
	PUSH_CTX

	//TimerStart();
	
	AuxGeometryMap ( model, AUX_VERTEX_BUF, AUX_ELEM_BUF );					// Setup VBO for CUDA (interop)
	
	// Prepare model geometry for use by CUDA
	cudaCheck ( cuMemcpyHtoD ( cuXform, xform->GetDataF(), sizeof(float)*16), "VolumeGVDB", "SolidVoxelize", "cuMemcpyHtoD", "cuXform", mbDebug );	// Send transform
	
	// Identify model bounding box
	model->ComputeBounds ( *xform, 0.1f );
	mObjMin = model->objMin; mObjMax = model->objMax;
	mVoxMin = mObjMin;
	mVoxMax = mObjMax;
	mVoxRes = mVoxMax; mVoxRes -= mVoxMin;

	// VDB Hierarchical Rasterization	
	Clear ();									// creates a new root	

	// Voxelize all nodes in bounding box at starting level		
	int N = mPool->getNumLevels ();
	Extents e = ComputeExtents ( N, mObjMin, mObjMax );			// start - level N
	ActivateRegion ( N-1, e );									// activate - level N-1
	#ifdef VOX_GL
		PrepareV3D ( Vector3DI(8,8,8), 0 );
	#endif

	// Voxelize at each level	
	int node_cnt, cnt;
	Node* node;	
	for (int lev = N-1; lev >= 0; lev-- ) {						// scan - level N-1 down to 0
		node_cnt = static_cast<int>(getNumUsedNodes(lev));
		cnt = 0;		
		// Insert triangles into bins
		float ydiv = getCover(lev).y;							// use brick boundaries for triangle sorting
		Vector3DI tcnts = InsertTriangles ( model, xform, ydiv );

		// Voxelize each node at this level
		for (int n = 0; n < node_cnt; n++ ) {					// get each node at current
			node = getNodeAtLevel ( n, lev );
			cnt += VoxelizeNode ( node, chan, xform, ydiv, val_surf, val_inside, vthresh);	// Voxelize each node
		}
		if ( lev==1 ) {											// Finish and Update atlas before doing bricks			
			FinishTopology( true, true );
			UpdateAtlas();						
		}		
		verbosef("Voxelized.. lev: %d, nodes: %d, new: %d\n", lev, node_cnt, cnt );
	}	

	// Update apron
	UpdateApron ();		

	#ifdef VOX_GL
		PrepareV3D ( Vector3DI(0,0,0), 0 );
	#endif

	AuxGeometryUnmap ( model, AUX_VERTEX_BUF, AUX_ELEM_BUF );

	POP_CTX
	//float msec = TimerStop();
	//verbosef( "Voxelize Complete: %4.2f\n", msec );
}

// Insert triangles into auxiliary bins
Vector3DI VolumeGVDB::InsertTriangles ( Model* model, Matrix4F* xform, float& ydiv )
{
	PUSH_CTX

	// Identify model bounding box
	model->ComputeBounds ( *xform, 0.1f );	
	int ybins = int(model->objMax.y / ydiv)+1;						// y divisions align with lev0 brick boundaries	
	
	PrepareAux ( AUX_GRIDCNT, ybins, sizeof(uint), true, true );	// allow return to cpu
	PrepareAux ( AUX_GRIDOFF, ybins, sizeof(uint), false, false );
	int vcnt = static_cast<int>(mAux[AUX_VERTEX_BUF].lastEle);			// input
	int ecnt = static_cast<int>(mAux[AUX_ELEM_BUF].lastEle);
	int tri_cnt = 0;								// output
		
	Vector3DI block ( 512, 1, 1 );
	Vector3DI grid ( int(ecnt/block.x)+1, 1, 1 );	
	void* args[7] = { &ydiv, &ybins, &mAux[AUX_GRIDCNT].gpu, &vcnt, &ecnt, &mAux[AUX_VERTEX_BUF].gpu, &mAux[AUX_ELEM_BUF].gpu };
	cudaCheck( cuLaunchKernel(cuFunc[FUNC_INSERT_TRIS], grid.x, 1, 1, block.x, 1, 1, 0, NULL, args, NULL),
					"VolumeGVDB", "InsertTriangles", "cuLaunch", "FUNC_INSERT_TRIS", mbDebug);

	// Prefix sum for bin offsets	
	PrefixSum ( mAux[AUX_GRIDCNT].gpu, mAux[AUX_GRIDOFF].gpu, ybins );
	
	// Sort triangles into bins (deep copy)		
	RetrieveData ( mAux[AUX_GRIDCNT] );
	uint* cnt = (uint*) mAux[AUX_GRIDCNT].cpu;
	for (int n=0; n < ybins; n++ ) tri_cnt += cnt[n];				// get total - should return from prefixsum
		
	// Reset grid counts
	cudaCheck ( cuMemsetD8 ( mAux[AUX_GRIDCNT].gpu, 0, sizeof(uint)*ybins ), "VolumeGVDB", "InsertTriangles", "cuMemsetD8", "AUX_GRIDCNT", mbDebug);

	// Prepare output triangle buffer
	PrepareAux ( AUX_TRI_BUF, tri_cnt, sizeof(Vector3DF)*3, false, false );

	// Deep copy sorted tris into output buffer
	block.Set ( 512, 1, 1 );
	grid.Set ( int(tri_cnt/block.x)+1, 1, 1 );	
	void* args2[10] = { &ydiv, &ybins, &mAux[AUX_GRIDCNT].gpu, &mAux[AUX_GRIDOFF].gpu, &tri_cnt, &mAux[AUX_TRI_BUF].gpu, &vcnt, &ecnt, &mAux[AUX_VERTEX_BUF].gpu, &mAux[AUX_ELEM_BUF].gpu };
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_SORT_TRIS], grid.x, 1, 1, block.x, 1, 1, 0, NULL, args2, NULL ), 
					"VolumeGVDB", "InsertTriangles", "cuLaunch", "FUNC_SORT_TRIS", mbDebug );

	POP_CTX

	return Vector3DI( ybins, tri_cnt, ecnt);	// return: number of bins, total inserted tris, original tri count
}

// Voxelize a mesh as surface voxels using OpenGL hardware rasterizer
void VolumeGVDB::SurfaceVoxelizeGL ( uchar chan, Model* model, Matrix4F* xform )
{
	PUSH_CTX

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
	model->ComputeBounds ( *xform, 0.1f );
	mObjMin = model->objMin; mObjMax = model->objMax;
	mVoxMin = mObjMin;
	mVoxMax = mObjMax;
	mVoxRes = mVoxMax; mVoxRes -= mVoxMin;
	
	// Determine which leaf voxels to activate
	PERF_PUSH ( "activate" );	

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
				vtemp.SurfaceVoxelizeFastGL ( vmin1, vmax1, xform );				

				// Readback the voxels
				vtemp.RetrieveGL ( (char*) vdat );

				// Identify leaf nodes at level-1
				for ( int z0=0; z0 < res1; z0++ ) {
					for ( int y0=0; y0 < res1; y0++ ) {
						for ( int x0=0; x0 < res1; x0++ ) {
							vpix[0] = static_cast<uchar>(*(vdat + (z0*res1 + y0)*res1 + x0));
							if ( vpix[0] > 0 ) {
								vmin0.x = vmin1.x + x0 * vrange1.x / res1;
								vmin0.y = vmin1.y + y0 * vrange1.y / res1;
								vmin0.z = vmin1.z + z0 * vrange1.z / res1;										
								leaf = ActivateSpace ( mRoot, vmin0, bnew );	
								if ( leaf != ID_UNDEFL ) {
									leaf_pos.push_back ( vmin0 );
									leaf_ptr.push_back ( leaf );
								}
							} else {
								// correction when not using conservative raster									
								c[1]=c[2]=c[3]=c[4]=c[5]=c[6] = 0;
								vpix[1] = static_cast<uchar>((x0 == res1 - 1) ? 0 : *(vdat + (z0 * res1 + y0) * res1 + (x0 + 1)));
								vpix[2] = static_cast<uchar>((y0 == res1 - 1) ? 0 : *(vdat + (z0 * res1 + (y0 + 1)) * res1 + x0));
								vpix[3] = static_cast<uchar>((z0 == res1 - 1) ? 0 : *(vdat + ((z0 + 1) * res1 + y0) * res1 + x0));
								vpix[4] = static_cast<uchar>((x0 == 0) ? 0 : *(vdat + (z0 * res1 + y0) * res1 + (x0 - 1)));
								vpix[5] = static_cast<uchar>((y0 == 0) ? 0 : *(vdat + (z0 * res1 + (y0 - 1)) * res1 + x0));
								vpix[6] = static_cast<uchar>((z0 == 0) ? 0 : *(vdat + ((z0 - 1) * res1 + y0) * res1 + x0));
								vpix[7] = (vpix[1] > 0) + (vpix[2] > 0) + (vpix[3] > 0) + (vpix[4] > 0) + (vpix[5] > 0 ) + (vpix[6] > 0 );								
								
								if ( vpix[7] > 0 ) {
									vmin0.x = vmin1.x + x0 * vrange1.x / res1;
									vmin0.y = vmin1.y + y0 * vrange1.y / res1;
									vmin0.z = vmin1.z + z0 * vrange1.z / res1;									
									leaf = ActivateSpace ( mRoot, vmin0, bnew );
									if ( leaf != ID_UNDEFL ) {
										leaf_pos.push_back ( vmin0 );
										leaf_ptr.push_back ( leaf );
									}
								}
								c[1]=c[2]=c[3]=c[4]=c[5]=c[6] = 0;
								vpix[1] = static_cast<uchar>((x0 == res1 - 1) ? (++c[1] == 1) : *(vdat + (z0 * res1 + y0) * res1 + (x0 + 1)));
								vpix[2] = static_cast<uchar>((y0 == res1 - 1) ? (++c[2] == 1) : *(vdat + (z0 * res1 + (y0 + 1)) * res1 + x0));
								vpix[3] = static_cast<uchar>((z0 == res1 - 1) ? (++c[3] == 1) : *(vdat + ((z0 + 1) * res1 + y0) * res1 + x0));
								vpix[4] = static_cast<uchar>((x0 == 0) ? (++c[4] == 1) : *(vdat + (z0 * res1 + y0) * res1 + (x0 - 1)));
								vpix[5] = static_cast<uchar>((y0 == 0) ? (++c[5] == 1) : *(vdat + (z0 * res1 + (y0 - 1)) * res1 + x0));
								vpix[6] = static_cast<uchar>((z0 == 0) ? (++c[6] == 1) : *(vdat + ((z0 - 1) * res1 + y0) * res1 + x0));
								vpix[7] = (vpix[1] > 0) + (vpix[2] > 0) + (vpix[3] > 0) + (vpix[4] > 0) + (vpix[5] > 0 ) + (vpix[6] > 0 );
								c[7] = (c[1]+c[2]+c[3]+c[4]+c[5]+c[6] == 3) ? 1 : 0;
								if ( vpix[7] >= 3 + c[7] ) {
									vmin0.x = vmin1.x + x0 * vrange1.x / res1;
									vmin0.y = vmin1.y + y0 * vrange1.y / res1;
									vmin0.z = vmin1.z + z0 * vrange1.z / res1;									
									leaf = ActivateSpace ( mRoot, vmin0, bnew );
									if ( leaf != ID_UNDEFL ) {
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
	PERF_POP ();

	FinishTopology ();

	UpdateAtlas ();

	cudaCheck ( cuCtxSynchronize(), "VolumeGVDB", "SurfaceVoxelizeGL", "cuCtxSynchronize", "", mbDebug );

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
		PERF_PUSH ( "rasterize" );		
		vtemp.PrepareRasterGL ( true );

		for (int n=0; n < leaf_ptr.size(); n++ ) {

			// Rasterize into temporary 3D texture						
			vtemp.SurfaceVoxelizeFastGL ( leaf_pos[n], leaf_pos[n] + vrange0, xform );			

			// Copy 3D texture in VDB atlas
			Node* node = getNode ( leaf_ptr[n] );
			if ( node == 0x0 || node->mValue.x == -1 ) {
				gprintf ( "WARNING: Node or value is empty in SurfaceVoxelizeGL.\n" );
			} else {
				mPool->AtlasCopyTex ( chan, node->mValue, vtemp.getPtr() );			
			}
		}
		vtemp.PrepareRasterGL ( false );			

		PERF_POP ();
	}
	glFinish ();

	cudaCheck ( cuCtxSynchronize(), "VolumeGVDB", "SurfaceVoxelizeGL", "cuCtxSynchronize", "", mbDebug );

	free ( vdat ); 	

	POP_CTX

	#endif
}

Vector3DF VolumeGVDB::MemoryUsage(std::string name, std::vector<std::string>& outlist )
{
	PUSH_CTX

	float vol_total=0, vol_dense = 0;
	float MB = (1024.f*1024.f);
	char str[1024];	
	int bpv;
	float curr;
	Vector3DF mem = cudaGetMemUsage();
	for (int n = 0; n < mPool->getNumAtlas(); n++) {
		bpv = mPool->getSize(mPool->getAtlas(n).type);
		curr = float(mPool->getAtlas(n).size) / MB;
		sprintf ( str, "%s, Voxel Channel %d: %6.2f MB (%4.2f%%)\n", name.c_str(), n, curr, float(curr*100.0/mem.x) );		
		outlist.push_back(str);
		vol_total += curr;
		vol_dense += (mVoxResMax.x * mVoxResMax.y * mVoxResMax.z) * bpv;
	}
	sprintf(str, "%s, Volume Total: %6.2f MB (%4.2f%%)\n", name.c_str(), vol_total, float(vol_total*100.0 / mem.x));
	outlist.push_back(str);

	float aux_total = 0;
	for (int n = 0; n < MAX_AUX; n++) {
		if (mAux[n].size > 0) {
			curr = float(mAux[n].size) / MB;
			sprintf(str, "%s, Aux %02d: %6.2f MB (%4.2f%%) - %s\n", name.c_str(), n, curr, float(curr*100.0 / mem.x), mAuxName[n].c_str() );
			outlist.push_back(str);
			aux_total += curr;
		}
	}
	sprintf(str, "%s, Aux Total: %6.2f MB (%4.2f%%)\n", name.c_str(), aux_total, float(aux_total*100.0 / mem.x));
	outlist.push_back(str);

	POP_CTX

	return mem;
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

// getUsage - Quickly report memory info
// - ext	= Extents of volume in voxels
// - vox	= # Bricks, # Voxels(M), Occupancy (%)
// - used	= Topology total (MB), Atlas total (MB), Aux total (MB)
// - free	= GPU mem free (MB), GPU total mem (MB)
//
void VolumeGVDB::getUsage( Vector3DF& ext, Vector3DF& vox, Vector3DF& used, Vector3DF& free )
{
	float MB = 1024.0*1024.0;	// convert to MB
	
	// extents
	RetrieveVDB();	
	ext = mVoxResMax;							
	
	// brick/voxel count 
	if (mPool->getNumAtlas() > 0) {
		int leafdim = static_cast<int>(mPool->getAtlas(0).stride);
		Vector3DI vb = mVoxResMax; vb /= Vector3DI(leafdim, leafdim, leafdim);
		long vbrk = vb.x*vb.y*vb.z;					// number of bricks covering bounded world domain
		float abrk = static_cast<float>(mPool->getPoolUsedCnt(0, 0));	// active bricks
		vox = Vector3DF(abrk, abrk*leafdim*leafdim*leafdim / 1000000.0f, abrk*100.0f / vbrk);		// # bricks, # voxels(M), occupancy(%)
	} else {
		vox = Vector3DF(0, 0, 0);
	}

	// memory used
	float topo_total = MeasurePools();
	float vol_total = 0;
	for (int n = 0; n < mPool->getNumAtlas(); n++) {		
		vol_total += mPool->getAtlas(n).size / MB;
	}
	float aux_total = 0;
	for (int ai = 0; ai < MAX_AUX; ai++) {
		if (mAux[ai].gpu != 0x0) aux_total += float(mAux[ai].size) / MB;
	}
	used = Vector3DF(topo_total, vol_total, aux_total );

	// memory free
	free = Vector3DF(cudaGetMemUsage().y, cudaGetMemUsage().z, 0);
}


// Measure GVDB topology & data
void VolumeGVDB::Measure ( bool bPrint )
{	
	PUSH_CTX

	uint64 numnodes_at_lev, maxnodes_at_lev;
	uint64 numnodes_total = 0, maxnodes_total = 0;
	Vector3DI axisres, axiscnt;
	int leafdim;
	uint64 atlas_sz = 0;	

	int levs = mPool->getNumLevels ();	

	//--- Print	
	if (bPrint) {
		gprintf("  EXTENTS:\n");
		gprintf("   Volume Res: %.0f x %.0f x %.0f\n", mVoxRes.x, mVoxRes.y, mVoxRes.z);
		gprintf("   Volume Max: %.0f x %.0f x %.0f\n", mVoxResMax.x, mVoxResMax.y, mVoxResMax.z);
		gprintf("   Bound Min:  %f x %f x %f\n", mObjMin.x, mObjMin.y, mObjMin.z);
		gprintf("   Bound Max:  %f x %f x %f\n", mObjMax.x, mObjMax.y, mObjMax.z);
		gprintf("  VDB CONFIG: <%d, %d, %d, %d, %d>:\n", getLD(4), getLD(3), getLD(2), getLD(1), getLD(0));
		gprintf("             # Nodes / # Pool   Pool Size\n");
	
		
		for (int n=0; n < levs; n++ ) {		
			// node counts
			numnodes_at_lev = mPool->getPoolUsedCnt(0, n);
			maxnodes_at_lev = mPool->getPoolTotalCnt(0, n);			
			numnodes_total += numnodes_at_lev;
			maxnodes_total += maxnodes_at_lev;
			gprintf ( "   Level %d: %8d %8d  %8d MB\n", n, numnodes_at_lev, maxnodes_at_lev );		
		}

		float MB = 1024.0*1024.0;	// convert to MB
		gprintf ( "   Percent Pool Used: %4.2f%%%%\n", float(numnodes_total)*100.0f / maxnodes_total );		

		// Atlas info
		if ( mPool->getNumAtlas() > 0 ) {
			leafdim = static_cast<int>(mPool->getAtlas(0).stride);				// voxel res of one brick
			axiscnt = mPool->getAtlas(0).subdim;				// number of bricks in atlas			
			axisres = axiscnt * (leafdim + mApron*2);			// number of voxels in atlas
			atlas_sz = mPool->getAtlas(0).size;			

			gprintf ( "  ATLAS STORAGE\n" );
			gprintf ( "   Atlas Res:     %d x %d x %d  LeafCnt: %d x %d x %d  LeafDim: %d^3\n", axisres.x, axisres.y, axisres.z, axiscnt.x, axiscnt.y, axiscnt.z, leafdim );
			Vector3DI vb = mVoxResMax;
			vb /= Vector3DI(leafdim, leafdim, leafdim);
			long vbrk = vb.x*vb.y*vb.z;		// number of bricks covering bounded world domain
			int sbrk = axiscnt.x*axiscnt.y*axiscnt.z;	// number of bricks stored in atlas
			uint64 abrk = mPool->getPoolUsedCnt(0, 0);

			gprintf ( "   Vol Extents:   %d bricks,  %5.2f million voxels\n", vbrk, float(vbrk)*leafdim*leafdim*leafdim / 1000000.0f );
			gprintf ( "   Atlas Storage: %d bricks,  %5.2f million voxels\n", sbrk, float(sbrk)*leafdim*leafdim*leafdim / 1000000.0f );
			gprintf ( "   Atlas Active:  %d bricks,  %5.2f million voxels\n", abrk, float(abrk)*leafdim*leafdim*leafdim / 1000000.0f );
			gprintf ( "   Occupancy:     %6.2f%%%% \n", float(abrk)*100.0f / vbrk );
		}
		float pool_total = MeasurePools();
		float vol_total, vol_dense;		
		vol_total = 0; vol_dense = 0;		
		gprintf ( "  MEMORY USAGE:\n");		
		gprintf ( "   Topology Total:    %6.2f MB\n", pool_total);
		gprintf ( "   Atlas:\n" );		
		int bpv;
		for (int n=0; n < mPool->getNumAtlas(); n++ ) {
			bpv = mPool->getSize(mPool->getAtlas(n).type) ;
			gprintf ( "     Channel %d:       %6.2f MB (%d bytes/vox)\n", n, float(mPool->getAtlas(n).size) / MB, bpv );
			vol_total += mPool->getAtlas(n).size;
			vol_dense += (mVoxResMax.x * mVoxResMax.y * mVoxResMax.z) * bpv;
		}
		gprintf ( "   Atlas Total:       %6.2f MB (Dense: %10.2f MB)\n", vol_total / MB, vol_dense / MB );				

	}
	POP_CTX
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
			mVDBInfo.vdel[n]		= Vector3DF(getRange(n));	mVDBInfo.vdel[n] /= Vector3DF( getRes3DI(n) );
			mVDBInfo.noderange[n]	= getRange(n);		// integer (cannot send cover)
			mVDBInfo.nodecnt[n]		= static_cast<int>(mPool->getPoolTotalCnt(0, n));
			mVDBInfo.nodewid[n]		= static_cast<int>(mPool->getPoolWidth(0, n));
			mVDBInfo.childwid[n]	= static_cast<int>(mPool->getPoolWidth(1, n));
			mVDBInfo.nodelist[n]	= mPool->getPoolGPU(0, n);
			mVDBInfo.childlist[n]	= mPool->getPoolGPU(1, n);								
			if ( mVDBInfo.nodecnt[n] == 1 ) tlev = n;		// get top level for rendering
		}
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
		mVDBInfo.epsilon			= mEpsilon;
		mVDBInfo.max_iter			= mMaxIter;
		mVDBInfo.bmin				= mObjMin;
		mVDBInfo.bmax				= mObjMax;
		
		PUSH_CTX
		cudaCheck ( cuMemcpyHtoD ( cuVDBInfo, &mVDBInfo, sizeof(VDBInfo) ), "VolumeGVDB", "PrepareVDB", "cuMemcpyHtoD", "cuVDBInfo", mbDebug);
		POP_CTX
	}		
}
void VolumeGVDB::RetrieveVDB()
{
	PUSH_CTX
	cudaCheck(cuMemcpyDtoH( &mVDBInfo, cuVDBInfo, sizeof(VDBInfo)), "VolumeGVDB", "RetrieveVDB", "cuMemcpyDtoH", "cuVDBInfo", mbDebug);
	POP_CTX
}
void VolumeGVDB::PrepareVDBPartially ()
{
	mVDBInfo.update = false;
	int tlev = 1;
	int levs = mPool->getNumLevels ();
	for (int n = levs-1; n >= 0; n-- ) {				
		mVDBInfo.dim[n]			= mLogDim[n];
		mVDBInfo.res[n]			= getRes(n);
		mVDBInfo.vdel[n]		= Vector3DF(getRange(n));	mVDBInfo.vdel[n] /= Vector3DF( getRes3DI(n) );
		mVDBInfo.noderange[n]	= getRange(n);		// integer (cannot send cover)
		mVDBInfo.nodecnt[n]		= static_cast<int>(mPool->getPoolTotalCnt(0, n));
		mVDBInfo.nodewid[n]		= static_cast<int>(mPool->getPoolWidth(0, n));
		mVDBInfo.childwid[n]	= static_cast<int>(mPool->getPoolWidth(1, n));
		mVDBInfo.nodelist[n]	= mPool->getPoolGPU(0, n);
		mVDBInfo.childlist[n]	= mPool->getPoolGPU(1, n);								
		if ( mVDBInfo.nodecnt[n] == 1 ) tlev = n;		// get top level for rendering
	}
	mVDBInfo.top_lev			= tlev;		
	mVDBInfo.bmin				= mObjMin;
	mVDBInfo.bmax				= mObjMax;

	PUSH_CTX
	cudaCheck ( cuMemcpyHtoD ( cuVDBInfo, &mVDBInfo, sizeof(VDBInfo) ), "VolumeGVDB", "PrepareVDBPartially", "cuMemcpyHtoD", "cuVDBInfo", mbDebug);
	POP_CTX
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
	PUSH_CTX

	PERF_PUSH("WriteDepthGL");

	if (mDummyFrameBuffer == -1) {
		glGenFramebuffers(1, (GLuint*)&mDummyFrameBuffer);
	}

	// Set current framebuffer to input buffer:
	glBindFramebuffer(GL_FRAMEBUFFER, mDummyFrameBuffer);

	// Setup frame buffer so that we can copy depth info from the depth target bound
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, glid, 0);

	// Copy contents of depth target into _depthBuffer for use in CUDA rendering:
	glBindBuffer(GL_PIXEL_PACK_BUFFER, mRenderBuf[chan].glid);
	glReadPixels(0, 0,
		static_cast<GLsizei>(mRenderBuf[chan].stride),
		static_cast<GLsizei>(mRenderBuf[chan].max / mRenderBuf[chan].stride),
		GL_DEPTH_COMPONENT, GL_FLOAT, 0);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	glCheck();

	// Clear render target:
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glFinish();

	PERF_POP();

	POP_CTX
}

void VolumeGVDB::WriteRenderTexGL ( int chan, int glid )
{
	PUSH_CTX

  PERF_PUSH ( "ReadTexGL" );

  if (mRenderBuf[chan].garray == 0) {
    // Prepare render target
    mRenderBuf[chan].glid = glid;

    // Cuda-GL interop to get target CUarray				
    cudaCheck(cuGraphicsGLRegisterImage(&mRenderBuf[chan].grsc, glid, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST), "VolumeGVDB", "WriteRenderTexGL", "cuGraphicsGLRegisterImage", "mRenderBuf", mbDebug );
    cudaCheck(cuGraphicsMapResources(1, &mRenderBuf[chan].grsc, 0), "VolumeGVDB", "WriteRenderTexGL", "cuGraphicsMapResources", "mRenderBuf", mbDebug);
    cudaCheck(cuGraphicsSubResourceGetMappedArray(&mRenderBuf[chan].garray, mRenderBuf[chan].grsc, 0, 0), "VolumeGVDB", "WriteRenderTexGL", "cuGraphicsSubResourceGetMappedArray", "mRenderBuf", mbDebug);
    cudaCheck(cuGraphicsUnmapResources(1, &mRenderBuf[chan].grsc, 0), "VolumeGVDB", "WriteRenderTexGL", "cuGraphicsUnmapResources", "mRenderBuf", mbDebug);
  }
  size_t bpp = mRenderBuf[chan].size / mRenderBuf[chan].max;
  size_t width = mRenderBuf[chan].stride * bpp;
  size_t height = mRenderBuf[chan].max / mRenderBuf[chan].stride;

  // Cuda Memcpy2D to transfer cuda output buffer
  // into opengl texture as a CUarray
  CUDA_MEMCPY2D cp = { 0 };
  cp.dstMemoryType = CU_MEMORYTYPE_DEVICE;		// from cuda buffer
  cp.dstDevice = mRenderBuf[chan].gpu;
  cp.srcMemoryType = CU_MEMORYTYPE_ARRAY;			// to CUarray (opengl texture)
  cp.srcArray = mRenderBuf[chan].garray;
  cp.WidthInBytes = width;
  cp.Height = height;

  cudaCheck(cuMemcpy2D(&cp), "VolumeGVDB", "WriteRenderTexGL", "cuMemcpy2D", "", mbDebug );

  cuCtxSynchronize();

  PERF_POP ();

	POP_CTX	
}

void VolumeGVDB::ReadRenderTexGL ( int chan, int glid )
{
	PUSH_CTX

	PERF_PUSH ( "ReadTexGL" );

	if ( mRenderBuf[chan].garray == 0 ) {
		// Prepare render target
		mRenderBuf[chan].glid = glid;

		// Cuda-GL interop to get target CUarray				
		cudaCheck ( cuGraphicsGLRegisterImage ( &mRenderBuf[chan].grsc, glid, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST ), "VolumeGVDB", "ReadRenderTexGL", "cuGraphicsGLRegisterImage ", "mRenderBuf", mbDebug);
		cudaCheck ( cuGraphicsMapResources(1, &mRenderBuf[chan].grsc, 0), "VolumeGVDB", "ReadRenderTexGL", "cuGraphicsMapResources", "mRenderBuf", mbDebug);
		cudaCheck ( cuGraphicsSubResourceGetMappedArray ( &mRenderBuf[chan].garray, mRenderBuf[chan].grsc, 0, 0 ), "VolumeGVDB", "ReadRenderTexGL", "cuGraphicsSubResourcesGetMappedArray", "mRenderBuf", mbDebug);
		cudaCheck ( cuGraphicsUnmapResources(1, &mRenderBuf[chan].grsc, 0), "VolumeGVDB", "ReadRenderTexGL", "cuGraphicsUnmapResources", "mRenderBuf", mbDebug);
	}
	size_t bpp = mRenderBuf[chan].size / mRenderBuf[chan].max;
	size_t width = mRenderBuf[chan].stride * bpp;
	size_t height = mRenderBuf[chan].max / mRenderBuf[chan].stride;

	// Cuda Memcpy2D to transfer cuda output buffer
	// into opengl texture as a CUarray
	CUDA_MEMCPY2D cp = {0};
	cp.srcMemoryType = CU_MEMORYTYPE_DEVICE;		// from cuda buffer
	cp.srcDevice = mRenderBuf[chan].gpu;
	cp.dstMemoryType = CU_MEMORYTYPE_ARRAY;			// to CUarray (opengl texture)
	cp.dstArray = mRenderBuf[chan].garray;
	cp.WidthInBytes = width;
	cp.Height = height;
	
	cudaCheck ( cuMemcpy2D ( &cp ), "VolumeGVDB", "ReadRenderTexGL", "cuMemcpy2D", "", mbDebug );

	cuCtxSynchronize ();

	PERF_POP ();

	POP_CTX
}

// Add a depth buffer
void VolumeGVDB::AddDepthBuf(int chan, int width, int height)
{
	mRenderBuf.resize(chan + 1);
	
	mRenderBuf[chan].alloc = mPool;
	mRenderBuf[chan].cpu = 0x0;				// no cpu residence yet
	mRenderBuf[chan].lastEle = 0;
	mRenderBuf[chan].usedNum = 0;
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
	mRenderBuf[chan].usedNum = 0;
	mRenderBuf[chan].lastEle = 0;
	mRenderBuf[chan].garray = 0;
	mRenderBuf[chan].grsc = 0;
	mRenderBuf[chan].glid = -1;
	mRenderBuf[chan].gpu = 0x0;

	ResizeRenderBuf ( chan, width, height, byteperpix );
}

// Resize a depth buffer
void VolumeGVDB::ResizeDepthBuf( int chan, int width, int height )
{
	PUSH_CTX

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

	cudaCheck(cuGraphicsGLRegisterBuffer(&mRenderBuf[chan].grsc, mRenderBuf[chan].glid, 0), "VolumeGVDB", "ResizeDepthBuf", "cuGraphicsGLRegisterBuffer", "", mbDebug);
	cudaCheck(cuGraphicsMapResources(1, &mRenderBuf[chan].grsc, 0), "VolumeGVDB", "ResizeDepthBuf", "cuGraphicsMapResources", "", mbDebug);
	cudaCheck(cuGraphicsResourceGetMappedPointer(&mRenderBuf[chan].gpu, &mRenderBuf[chan].size, mRenderBuf[chan].grsc), "VolumeGVDB", "ResizeDepthBuf", "cuGraphicsResourcesGetMappedPointer", "", mbDebug);
	cudaCheck(cuGraphicsUnmapResources(1, &mRenderBuf[chan].grsc, 0), "VolumeGVDB", "ResizeDepthBuf", "cuGraphicsUnmapResources", "", mbDebug);

	POP_CTX
}

// Resize a render buffer
void VolumeGVDB::ResizeRenderBuf ( int chan, int width, int height, int byteperpix )
{
	PUSH_CTX

	if ( chan >= mRenderBuf.size() ) {
		gprintf ( "ERROR: Attempt to resize render buf that doesn't exist." );
	}
	mRenderBuf[chan].max = width * height; 
	mRenderBuf[chan].size = width * height * byteperpix; 
	mRenderBuf[chan].stride = width;
	if ( chan == 0 ) getScene()->SetRes ( width, height );
	
	size_t sz = mRenderBuf[chan].size;
	if ( mRenderBuf[chan].gpu != 0x0 ) { 
		cudaCheck ( cuMemFree ( mRenderBuf[chan].gpu ), "VolumeGVDB", "ResizeRenderBuf", "cuMemFree", "", mbDebug);
	}
	cudaCheck ( cuMemAlloc ( &mRenderBuf[chan].gpu, sz ), "VolumeGVDB", "ResizeRenderBuf", "cuMemAlloc", "", mbDebug);

	if ( mRenderBuf[chan].garray != 0x0 ) {		
		cudaCheck ( cuGraphicsUnregisterResource ( mRenderBuf[chan].grsc ), "VolumeGVDB", "ResizeRenderBuf", "cuGraphicsUnregisterResource", "", mbDebug);
		mRenderBuf[chan].grsc = 0x0;
		mRenderBuf[chan].garray = 0x0;
	}

	cuCtxSynchronize ();

	POP_CTX
}

// Read render buffer to CPU
void VolumeGVDB::ReadRenderBuf ( int chan, unsigned char* outptr )
{
	PUSH_CTX

	if ( mbVerbose ) PERF_PUSH ( "ReadBuf" );
	mRenderBuf[chan].cpu = (char*) outptr;
	mPool->RetrieveMem ( mRenderBuf[chan] );		// transfer dev to host
	if ( mbVerbose ) PERF_POP ();
	
	POP_CTX
}

// Prepare scene data for rendering kernel
void VolumeGVDB::PrepareRender ( int w, int h, char shading )
{
	PUSH_CTX

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
	mScnInfo.bias		= m_bias;
	// Transform the light position from application space to voxel space,
	// like getViewPos:
	nvdb::Vector4DF lightPos4 = getScene()->getLight()->getPos();
	lightPos4.w = 1.0f; // Since it's a position, not a direction
	lightPos4 *= mInvXform;
	mScnInfo.light_pos  = Vector3DF(lightPos4.x, lightPos4.y, lightPos4.z);
	Vector3DF crs 		= Vector3DF(0,0,0);		
	mScnInfo.slice_pnt  = getScene()->getSectionPnt();
	mScnInfo.slice_norm = getScene()->getSectionNorm();
	mScnInfo.shading	= shading;				// getScene()->getShading();	
	mScnInfo.filtering	= getScene()->getFilterMode();
	mScnInfo.frame		= getScene()->getFrame();
	mScnInfo.samples	= getScene()->getSample();
	mScnInfo.shadow_params = getScene()->getShadowParams();
	mScnInfo.backclr	= getScene()->getBackClr ();
	mScnInfo.extinct	= getScene()->getExtinct ();
	mScnInfo.steps		= getScene()->getSteps ();
	mScnInfo.cutoff		= getScene()->getCutoff ();
	mScnInfo.thresh		= getScene()->mVThreshold;
	// Grid transform
	memcpy(mScnInfo.xform, mXform.GetDataF(), sizeof(float)*16);
	memcpy(mScnInfo.invxform, mInvXform.GetDataF(), sizeof(float)*16);	
	memcpy(mScnInfo.invxrot, mInvXrot.GetDataF(), sizeof(float) * 16);
	// Transfer function
	mScnInfo.transfer	= getTransferFuncGPU();	
	if (mScnInfo.transfer == 0) {
		gprintf("Error: Transfer function not on GPU. Must call CommitTransferFunc.\n");
		gerror();
	}
	// Depth buffer
	mScnInfo.outbuf		= -1;			// NOT USED  (was mRenderBuf[0].gpu;)
	int dbuf = getScene()->getDepthBuf();
	mScnInfo.dbuf 		= (dbuf == 255 ? NULL : mRenderBuf[dbuf].gpu);	

	cudaCheck ( cuMemcpyHtoD ( cuScnInfo, &mScnInfo, sizeof(ScnInfo) ), "VolumeGVDB", "PrepareRender", "cuMemcpyHtoD", "cuScnInfo", mbDebug);

	POP_CTX
}

// Render using custom user kernel
void VolumeGVDB::RenderKernel ( CUfunction user_kernel, uchar chan, uchar rbuf )
{
	PUSH_CTX

	if (mbProfile) PERF_PUSH ( "Render" );
	
	int width = static_cast<int>(mRenderBuf[rbuf].stride);
	int height = static_cast<int>(mRenderBuf[rbuf].max / mRenderBuf[rbuf].stride);

	// Send Scene info (camera, lights)
	PrepareRender ( width, height, getScene()->getShading() );

	// Send VDB Info & Atlas
	PrepareVDB ();												

	// Run CUDA User-Kernel
	Vector3DI block ( 8, 8, 1 );
	Vector3DI grid ( (int(width) + block.x - 1)/block.x, (int(height) + block.y - 1)/block.y, 1);		
	void* args[3] = { &cuVDBInfo,  &chan, &mRenderBuf[rbuf].gpu };
	cudaCheck ( cuLaunchKernel ( user_kernel, grid.x, grid.y, 1, block.x, block.y, 1, 0, NULL, args, NULL ), "VolumeGVDB", "RenderKernel", "cuLaunch", "(user kernel)", mbDebug);
	
	if (mbProfile) PERF_POP ();

	POP_CTX
}

// Render using native kernel
void VolumeGVDB::Render ( char shading, uchar chan, uchar rbuf )
{
	int width = static_cast<int>(mRenderBuf[rbuf].stride);
	int height = static_cast<int>(mRenderBuf[rbuf].max / mRenderBuf[rbuf].stride);
	if ( shading==SHADE_OFF ) {
		PUSH_CTX
		cudaCheck ( cuMemsetD8 ( mRenderBuf[rbuf].gpu, 0, static_cast<uint64>(width)*static_cast<uint64>(height)*4 ),
			"VolumeGVDB", "Render", "cuMemsetD8", "(no shading)", mbDebug);
		POP_CTX
		return;
	}
	
	PUSH_CTX
	
	if (mbProfile) PERF_PUSH ( "Render" );

	// Send Scene info (camera, lights)	
	PrepareRender ( width, height, shading );

	// Send VDB Info & Atlas
	PrepareVDB ();												

	// Prepare kernel
	Vector3DI block ( 16, 16, 1);
	Vector3DI grid((int(width) + block.x - 1) / block.x, (int(height) + block.y - 1) / block.y, 1);
	void* args[3] = { &cuVDBInfo, &chan, &mRenderBuf[rbuf].gpu };
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
	cudaCheck(cuLaunchKernel(cuFunc[kern], grid.x, grid.y, 1, block.x, block.y, 1, 0, NULL, args, NULL),
				"VolumeGVDB", "Render", "cuLaunch", mRendName[shading], mbDebug);

	if (mbProfile) PERF_POP ();

	POP_CTX
}

// Explicit raytracing
void VolumeGVDB::Raytrace ( DataPtr rays, uchar chan, char shading, int frame, float bias )
{
	PUSH_CTX

	if (mbProfile) PERF_PUSH ( "Raytrace" );

	// Send scene data
	PrepareRender ( 1, 1, 0 );

	// Send VDB Info & Atlas
	PrepareVDB ();												

	// Run CUDA GVDB Raytracer
	int cnt = static_cast<int>(rays.lastEle);
	Vector3DI block ( 64, 1, 1);
	Vector3DI grid ( (cnt + block.x - 1)/block.x, 1, 1);		
    void* args[5] = { &cuVDBInfo, &chan, &cnt, &rays.gpu, &bias };
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_RAYTRACE], grid.x, 1, 1, block.x, 1, 1, 0, NULL, args, NULL ), 
				"VolumeGVDB", "Raytrace", "cuLaunch", "FUNC_RAYTRACE", mbDebug );	 	

	if (mbProfile) PERF_POP ();

	POP_CTX

}

// Update apron (for all channels)
void VolumeGVDB::UpdateApron ()
{
	for (int n=0; n < mPool->getNumAtlas(); n++ )
		UpdateApron ( n );	// only update marker channel
}

// Update apron (one channel)
void VolumeGVDB::UpdateApron ( uchar chan, float boundval, bool changeCtx)
{ 	
	if ( mApron == 0 ) return;	
	
	// Send VDB Info	
	PrepareVDB ();			

	if(changeCtx) PUSH_CTX

	// Channel type
	int kern;
	char typ[3]; typ[2] = '\0';

	switch ( mPool->getAtlas(chan).type ) {
	case T_FLOAT:	kern = FUNC_UPDATEAPRON_F;	 typ[0] = 'F'; typ[1] = '1';	break;
	case T_FLOAT3:  kern = FUNC_UPDATEAPRON_F4;	 typ[0] = 'F'; typ[1] = '3';	break;		// F3 is implemented as F4 
	case T_FLOAT4:  kern = FUNC_UPDATEAPRON_F4;  typ[0] = 'F'; typ[1] = '4';	break;
	case T_UCHAR:	kern = FUNC_UPDATEAPRON_C;   typ[0] = 'U'; typ[1] = '1';	break;
	case T_UCHAR4:	kern = FUNC_UPDATEAPRON_C4;	 typ[0] = 'U'; typ[1] = '4'; 	break;
	}

	// Dimensions
	int bricks = static_cast<int>(mPool->getPoolTotalCnt(0,0));	// number of bricks
	int brickres = mPool->getAtlasBrickres(chan);				// dimension of brick (including apron)
	int brickwid = mPool->getAtlasBrickwid(chan);				// dimension of brick (without apron)

	if (bricks == 0) return;

	Vector3DI threadcnt(bricks, brickres, brickres);		// First component is passed directly to grid
	Vector3DI block(6 * mApron, 8, 8);
	Vector3DI grid(threadcnt.x, (threadcnt.y + block.y - 1) / block.y, (threadcnt.y + block.z - 1) / block.z);
	
	// Note: Kernels currently assume apron size of 1.
	void* args[6] = { &cuVDBInfo, &chan, &bricks, &brickres, &brickwid, &boundval };
	cudaCheck(cuLaunchKernel(cuFunc[kern], grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args, NULL), 
				"VolumeGVDB", "UpdateApron", "cuLaunch", typ, mbDebug);


	if (changeCtx) { POP_CTX }
}


// Update apron (one channel)
void VolumeGVDB::UpdateApronFaces (uchar chan)
{
	if (mApron == 0) return;

	if (mbProfile) PERF_PUSH("UpdateApron");

	// Send VDB Info	
	PrepareVDB();

	// Channel type
	int kern;
	switch (mPool->getAtlas(chan).type) {
	case T_FLOAT:	kern = FUNC_UPDATEAPRONFACES_F;		break;
	}

	PUSH_CTX

	// Dimensions
	int bricks = static_cast<int>(mPool->getPoolTotalCnt(0, 0));				// number of bricks
	int brickres = mPool->getAtlasBrickres(chan);			// dimension of brick (including apron)
	int brickwid = mPool->getAtlasBrickwid(chan);			// dimension of brick (without apron)

	Vector3DI threadcnt(bricks, brickwid, brickwid);		// First component is passed directly to grid
	Vector3DI block(3 * mApron, 8, 8);
	Vector3DI grid(threadcnt.x, (threadcnt.y + block.y - 1) / block.y, (threadcnt.z + block.z - 1) / block.z);

	DataPtr* ntable = mPool->getNeighborTable();			// get neighbor table

	void* args[6] = { &cuVDBInfo, &chan, &bricks, &brickres, &brickwid, &ntable->gpu };
	cudaCheck(cuLaunchKernel(cuFunc[kern], grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args, NULL),
					"VolumeGVDB", "UpdateApronFaces", "cuLaunch", "F1 (only)", mbDebug);

	POP_CTX

	if (mbProfile) PERF_POP();
}


void VolumeGVDB::ComputeKernel (CUmodule user_module, CUfunction user_kernel, uchar channel, bool bUpdateApron, bool skipOverAprons)
{
	PERF_PUSH ("ComputeKernel");

	SetModule ( user_module );

	// Send VDB Info (*to user module*)
	PrepareVDB ();

	PUSH_CTX

	// Determine grid and block dims (must match atlas bricks)	
	Vector3DI block ( 8, 8, 8 );
	Vector3DI atlasRes = mPool->getAtlasRes(channel);
	Vector3DI threadCount = (skipOverAprons ? mPool->getAtlasPackres(channel) : atlasRes);
	Vector3DI grid = (threadCount + block - 1) / block;

	void* args[3] = { &cuVDBInfo, &atlasRes, &channel };
	cudaCheck ( cuLaunchKernel ( user_kernel, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args, NULL ), 
					"VolumeGVDB", "ComputeKernel", "cuLaunch", "(user kernel)", mbDebug);
	
	
	if ( bUpdateApron ) {
		cudaCheck ( cuCtxSynchronize(), "VolumeGVDB", "ComputeKernel", "cuCtxSyncronize", "", mbDebug);
		SetModule ();			// Update apron must be called from default module
		UpdateApron ();				
	}

	POP_CTX

	PERF_POP();
}

void VolumeGVDB::Compute (int effect, uchar channel, int num_iterations, Vector3DF parameters, bool bUpdateApron, bool skipOverAprons, float boundval)
{ 
	PERF_PUSH ("Compute");

	// Send VDB Info	
	PrepareVDB ();
	
	PUSH_CTX

	// Determine grid and block dims (must match atlas bricks)	
	Vector3DI block ( 8, 8, 8 );
	Vector3DI atlasRes = mPool->getAtlasRes(channel);
	Vector3DI threadCount = (skipOverAprons ? mPool->getAtlasPackres(channel) : atlasRes);
	Vector3DI grid = (threadCount + block - 1) / block;

	void* args[6] = { &cuVDBInfo, &atlasRes, &channel, &parameters.x, &parameters.y, &parameters.z };
	
	for (int n=0; n < num_iterations; n++ ) {
		cudaCheck ( cuLaunchKernel ( cuFunc[effect], grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, mStream, args, NULL ),
						"VolumeGVDB", "Compute", "cuLaunch", "", mbDebug);
			
		if (bUpdateApron) UpdateApron(channel, boundval); // update the apron
	}
	POP_CTX
		
	PERF_POP();
}

void VolumeGVDB::DownsampleCPU(Matrix4F xform, Vector3DI in_res, char in_aux, Vector3DI out_res, Vector3DF out_max, char out_aux, Vector3DF inr, Vector3DF outr)
{
	PUSH_CTX

	PrepareAux(out_aux, out_res.x*out_res.y*out_res.z, sizeof(float), true, true);

	// Determine grid and block dims
	Vector3DI block(8, 8, 8);
	Vector3DI grid(int(out_res.x / block.x) + 1, int(out_res.y / block.y) + 1, int(out_res.z / block.z) + 1);

	// Send transform matrix to cuda
	PrepareAux(AUX_MATRIX4F, 16, sizeof(float), true, true);
	memcpy(mAux[AUX_MATRIX4F].cpu, xform.GetDataF(), 16 * sizeof(float));
	CommitData(mAux[AUX_MATRIX4F]);

	void* args[8] = { &in_res, &mAux[in_aux].gpu, &out_res, &out_max, &mAux[out_aux].gpu, &mAux[AUX_MATRIX4F].gpu, &inr, &outr };
	cudaCheck(cuLaunchKernel(cuFunc[FUNC_DOWNSAMPLE], grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args, NULL), 
					"VolumeGVDB", "DownsampleCPU", "cuLaunch", "FUNC_DOWNSAMPLE", mbDebug);

	// Retrieve data back to CPU
	RetrieveData(mAux[out_aux]);

	POP_CTX
}

Vector3DF VolumeGVDB::Reduction(uchar chan)
{
	if (mbProfile) PERF_PUSH("Reduction");

	PrepareVDB();

	PUSH_CTX

	// Launch 2D grid of voxels 
	//  (x/y plane, not including apron)	
	Vector3DI packres = mPool->getAtlasPackres(chan);		// exclude apron voxels
	Vector3DI res = mPool->getAtlasRes(chan);
	Vector3DI block(8, 8, 1);
	Vector3DI grid(int(packres.x / block.x) + 1, int(packres.y / block.y) + 1, 1);

	// Prepare accumulation buffer
	PrepareAux(AUX_DATA2D, packres.x*packres.y, sizeof(float), false, true);

	void* args[5] = { &cuVDBInfo, &res, &chan, &packres, &mAux[AUX_DATA2D].gpu };
	cudaCheck(cuLaunchKernel(cuFunc[FUNC_REDUCTION], grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args, NULL), 
				"VolumeGVDB", "Reduction", "cuLaunch", "FUNC_REDUCTION", mbDebug);
		
	RetrieveData(mAux[AUX_DATA2D]);

	float sum = 0.0;
	float* dat = (float*)mAux[AUX_DATA2D].cpu;
	for (int y = 0; y < packres.y; y++) {
		for (int x = 0; x < packres.x; x++) {
			sum += *dat;
			dat++;
		}
	}
	POP_CTX

	if (mbProfile) PERF_POP();

	return Vector3DF(sum, 0, 0);
}


void VolumeGVDB::Resample ( uchar chan, Matrix4F xform, Vector3DI in_res, char in_aux, Vector3DF inr, Vector3DF outr )
{
	PrepareVDB ();

	PUSH_CTX

	// Determine grid and block dims (must match atlas bricks)	
	Vector3DI block ( 8, 8, 8 );
	Vector3DI res = mPool->getAtlasRes( chan );
	Vector3DI grid = (res + block - 1) / block;

	// Send transform matrix to cuda
	PrepareAux ( AUX_MATRIX4F, 16, sizeof(float), true, true );
	memcpy ( mAux[AUX_MATRIX4F].cpu, xform.GetDataF(), 16*sizeof(float) );
	CommitData ( mAux[AUX_MATRIX4F] );

	void* args[8] = { &cuVDBInfo, &res, &chan, &in_res, &mAux[in_aux].gpu, &mAux[AUX_MATRIX4F].gpu, &inr, &outr };

	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_RESAMPLE], grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args, NULL ), 
					"VolumeGVDB", "Resample", "cuLaunch", "FUNC_RESAMPLE", mbDebug);

	POP_CTX
}


uint64 VolumeGVDB::getNodeAtPoint (uint64 nodeid, Vector3DF pos)
{
	// Recurse to find node	
	Node* curr = getNode( nodeid );
	Vector3DF p = pos; p -= curr->mPos;
	Vector3DF range = getRange(curr->mLev);
	
	if (p.x >= 0 && p.y >= 0 && p.z >= 0 && p.x < range.x && p.y < range.y && p.z < range.z) {

		// point is inside this node
		p *= getRes(curr->mLev);
		p /= range;

		if (isLeaf(nodeid)) {
			return nodeid;
		} else {
#ifdef USE_BITMASKS			
			uint32 i = getBitPos(curr->mLev, p);
			uint32 b = curr->countOn(i);
			slong childid;
			if (curr->isOn(i)) {
				childid = getChildNode(nodeid, b);
				return getNodeAtPoint( childid, pos);
			}
#else
			uint32 i = getBitPos(curr->mLev, p);
			slong childid;
			if (isOn(nodeid, i)) {
				childid = getChildNode(nodeid, i);
				return getNodeAtPoint( childid, pos);
			}
#endif
		}
	}	
	return ID_UNDEFL;
}


bool  VolumeGVDB::isActive(Vector3DI wpos)
{
	return isActive(wpos, mRoot);
}

bool  VolumeGVDB::isActive( Vector3DI pos, slong nodeid)
{
	// Recurse to leaf	
	unsigned int b;

	if (getPosInNode(nodeid, pos, b)) {
	
		// point is inside
		if (isLeaf(nodeid)) {
			return true;		// found leaf. is active.
		} else {
			slong childid;
			if (isOn(nodeid, b)) {
				childid = getChildNode(nodeid, b);
				return isActive(pos, childid);
			}
		}
	}
	return false;
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
#ifdef USE_BITMASKS			
			uint32 i = getBitPos ( curr->mLev, p );
			uint32 b = curr->countOn ( i );
			slong childid;			
			if ( curr->isOn ( i ) ) {
				childid = getChildNode ( nodeid, b );
				return getValue ( childid, pos, atlas );
			}		
#else
			uint32 i = getBitPos ( curr->mLev, p );
			slong childid;			
			if ( isOn (nodeid, i ) ) {
				childid = getChildNode ( nodeid, i );
				return getValue ( childid, pos, atlas );
			}
#endif
		}
	} 
	return 0.0;	
}

#define SCAN_BLOCKSIZE		512				// <--- must match cuda_gvdb_particles.cuh header
#define ONE_LEVEL			0

void VolumeGVDB::PrefixSum ( CUdeviceptr inArray, CUdeviceptr outArray, int numElem )
{
	int naux = SCAN_BLOCKSIZE << 1;				// 1024
	int len1 = numElem / naux;					// (1024^2+1024) = 1,049,600, len1 = 1025
	int blks1 = int( numElem / naux )+1;		// 1026
	int blks2 = int( blks1 / naux )+1;			// 2
	int threads = SCAN_BLOCKSIZE;
	int zon = 1;	
	CUdeviceptr nptr = {0};

	PUSH_CTX

	if ( blks1 <= 1024 ) {
		PrepareAux ( AUX_ARRAY1, SCAN_BLOCKSIZE << 1, sizeof(uint), false);		
		PrepareAux ( AUX_SCAN1,  SCAN_BLOCKSIZE << 1, sizeof(uint), false);
	} else {		
		PrepareAux ( AUX_ARRAY1, blks2*(SCAN_BLOCKSIZE << 1), sizeof(uint), false);		
		PrepareAux ( AUX_SCAN1,  blks2*(SCAN_BLOCKSIZE << 1), sizeof(uint), false);
	}
	PrepareAux ( AUX_ARRAY2, SCAN_BLOCKSIZE << 1, sizeof(uint), false );
	PrepareAux ( AUX_SCAN2,  SCAN_BLOCKSIZE << 1, sizeof(uint), false );
	
	void* argsA[5] = {&inArray, &outArray, &mAux[AUX_ARRAY1].gpu, &numElem, &zon };
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_PREFIXSUM], blks1, 1, 1, threads, 1, 1, 0, NULL, argsA, NULL ), 
					"VolumeGVDB", "PrefixSum", "cuLaunch", "FUNC_PREFIXSUM (in,out->A1)", mbDebug);
		
#if ONE_LEVEL
	void* argsB[5] = { &mAux[AUX_ARRAY1].gpu, &mAux[AUX_SCAN1].gpu, &nptr, &blks1, &zon };
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_PREFIXSUM], blks2, 1, 1, threads, 1, 1, 0, NULL, argsB, NULL ), "cuPrefixSumB", "PrefixSum" );

	RetrieveData ( mAux[AUX_SCAN1] );
	uint* dat = (uint*) getDataPtr ( 0, mAux[AUX_SCAN1] );
	gprintf ( "----\n" );
	for (int n=0; n < 10; n++ ) {
		gprintf ( "%d: %d\n", n, *dat++ );
	}
	gprintf ( "----\n" );	

#else
	//int blen1 = blks1 * naux;
	void* argsB[5] = { &mAux[AUX_ARRAY1].gpu, &mAux[AUX_SCAN1].gpu, &mAux[AUX_ARRAY2].gpu, &blks1, &zon };
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_PREFIXSUM], blks2, 1, 1, threads, 1, 1, 0, NULL, argsB, NULL ), 
					"VolumeGVDB", "PrefixSum", "cuLaunch", "FUNC_PREFIXSUM (A1,S1->A2)", mbDebug);		
	
	//int blen2 = blks2 * naux;
	void* argsC[5] = { &mAux[AUX_ARRAY2].gpu, &mAux[AUX_SCAN2].gpu, &nptr, &blks2, &zon };
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_PREFIXSUM], 1, 1, 1, threads, 1, 1, 0, NULL, argsC, NULL ), 
					"VolumeGVDB", "PrefixSum", "cuLaunch", "FUNC_PREFIXSUM (A2,S2->null)", mbDebug);

	void* argsD[3] = { &mAux[AUX_SCAN1].gpu, &mAux[AUX_SCAN2].gpu, &blks1 };
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_PREFIXFIXUP], blks2, 1, 1, threads, 1, 1, 0, NULL, argsD, NULL ), 
					"VolumeGVDB", "PrefixSum", "cuLaunch", "FUNC_PREFIXFIXUP (S2->S1)", mbDebug);		
#endif

	void* argsE[3] = { &outArray, &mAux[AUX_SCAN1].gpu, &numElem };
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_PREFIXFIXUP], blks1, 1, 1, threads, 1, 1, 0, NULL, argsE, NULL ), 
					"VolumeGVDB", "PrefixSum", "cuLaunch", "FUNC_PREFIXFIXUP (S1->out)", mbDebug);

	POP_CTX
}

void VolumeGVDB::AllocData ( DataPtr& ptr, int cnt, int stride, bool bCPU )
{
	PUSH_CTX
	if ( (ptr.cpu == 0 && bCPU) || ptr.gpu==0 || cnt > ptr.max || stride != ptr.stride )
		mPool->CreateMemLinear ( ptr, 0x0, stride, cnt, bCPU );		// always reallocates	
	POP_CTX
}
void VolumeGVDB::FreeData( DataPtr& ptr )
{
	PUSH_CTX
	mPool->FreeMemLinear(ptr);
	POP_CTX
}
void VolumeGVDB::CommitData ( DataPtr ptr )
{
	PUSH_CTX
	PERF_PUSH ( "Commit Data" );	
	mPool->CommitMem ( ptr );
	PERF_POP ();
	POP_CTX
}
void VolumeGVDB::CommitData ( DataPtr& dat, int cnt, char* cpubuf, int offs, int stride )
{
	dat.cpu = cpubuf;
	dat.usedNum = cnt;
	dat.lastEle = cnt;
	dat.max = cnt;
	dat.stride = stride;
	dat.subdim.Set ( offs, stride, 0 );

	if (dat.gpu == 0x0) {
		gprintf( "ERROR: Buffer not allocated on GPU. May be missing call to AllocData.\n");
		gerror();
		return; 
	}

	PUSH_CTX
	PERF_PUSH ( "Commit Data" );	
	mPool->CommitMem ( dat );
	PERF_POP ();
	POP_CTX
}
void VolumeGVDB::SetDataCPU ( DataPtr& dat, int cnt, char* cptr, int offs, int stride )
{
	dat.usedNum = cnt;
	dat.lastEle = cnt;
	dat.max = cnt;
	dat.cpu = cptr;	
	dat.stride = stride;
	dat.subdim.Set ( offs, stride, 0 );
}
void VolumeGVDB::SetDataGPU ( DataPtr& dat, int cnt, CUdeviceptr gptr, int offs, int stride )
{
	dat.lastEle = cnt;
	dat.usedNum = cnt;
	dat.max = cnt;
	dat.gpu = gptr;
	dat.stride = stride;
	dat.subdim.Set ( offs, stride, 0 );
}

void VolumeGVDB::RetrieveData ( DataPtr ptr )
{	
	PUSH_CTX
	mPool->RetrieveMem ( ptr );
	POP_CTX
}
char* VolumeGVDB::getDataCPU ( DataPtr ptr, int n, int stride )
{
	return (ptr.cpu + n*stride);
}
void VolumeGVDB::CommitTransferFunc ()
{
	PUSH_CTX
	mPool->CreateMemLinear ( mTransferPtr, (char*) mScene->getTransferFunc(), 16384*sizeof(Vector4DF) );
	mVDBInfo.update = true;		// tranfer func pointer may have changed for next render
	POP_CTX
}

void VolumeGVDB::SetPoints ( DataPtr& pntpos, DataPtr& pntvel, DataPtr& pntclr )
{
	mAux[AUX_PNTPOS] = pntpos;
	mAux[AUX_PNTVEL] = pntvel;
	mAux[AUX_PNTCLR] = pntclr;

	if ( pntvel.gpu==0 ) CleanAux ( AUX_SUBCELL_PNT_VEL );
	if ( pntclr.gpu==0 ) CleanAux ( AUX_SUBCELL_PNT_CLR );
}

void VolumeGVDB::SetDiv ( DataPtr div )
{
	mAux[AUX_DIV] = div;
}

void VolumeGVDB::SetSupportPoints ( DataPtr& pntpos, DataPtr& dirpos )
{
	mAux[AUX_PNTPOS] = pntpos;
	mAux[AUX_PNTDIR] = dirpos;
}

void VolumeGVDB::CleanAux()
{
	PUSH_CTX

	for (int id = 0; id < MAX_AUX; id++) {
		mAux[id].lastEle = 0; mAux[id].usedNum = 0;
		mAux[id].max = 0; mAux[id].size = 0; mAux[id].stride = 0;
		if (mAux[id].cpu != 0x0) { free(mAux[id].cpu); mAux[id].cpu = 0x0;}
		if (mAux[id].gpu != 0x0) {	(cuMemFree(mAux[id].gpu), "cuMemFree", "CleanAux");	mAux[id].gpu = 0x0; }
	}

	POP_CTX
}

void VolumeGVDB::CleanAux(int id)
{
	PUSH_CTX
	//std::cout << "clean aux " << id << std::endl;
	mAux[id].lastEle = 0;
	mAux[id].usedNum = 0;
	mAux[id].max = 0;
	mAux[id].size = 0;
	mAux[id].stride = 0;
	if (mAux[id].cpu != 0x0)
	{
		free(mAux[id].cpu);
		mAux[id].cpu = 0x0;
	}
	if (mAux[id].gpu != 0x0) 
	{
		(cuMemFree(mAux[id].gpu), "cuMemFree", "CleanAux");
		mAux[id].gpu = 0x0;
	}
	POP_CTX
}

void VolumeGVDB::PrepareAux ( int id, int cnt, int stride, bool bZero, bool bCPU )
{
	PUSH_CTX
	if ( mAux[id].lastEle < cnt || mAux[id].stride != stride ) {
		mPool->CreateMemLinear ( mAux[id], 0x0, stride, cnt, bCPU );
	}
	if ( bZero ) {
		cudaCheck ( cuMemsetD8 ( mAux[id].gpu, 0, mAux[id].size ), "VolumeGVDB", "PrepareAux", "cuMemsetD8", "", mbDebug);
	}
	POP_CTX
}



void VolumeGVDB::InsertPointsSubcell( int subcell_size, int num_pnts, float pRadius, Vector3DF trans,  int& pSCPntsLength)
{
	int numBrick = static_cast<int>(mPool->getPoolUsedCnt(0, 0));
	if (numBrick == 0) {
		gprintf("Warning: InsertPointsSubcell yet no bricks exist.\n");
		return;
	}

	PrepareVDB();

	PUSH_CTX

	PERF_PUSH("Insert pnt sc");

	int numSCellPerBrick = static_cast<int>(pow(getRes(0) / subcell_size, 3));	
	int numSCell = numSCellPerBrick * numBrick;
	int SCDim = getRes(0) / subcell_size;

	Vector3DI range = getRange(0) / (getRes(0) / subcell_size);

	int numSCellMapping = static_cast<int>(mPool->getPoolTotalCnt(0, 0));

	int threads = 512;	
	int pblks = int(numSCellMapping / threads)+1;

	PrepareAux( AUX_SUBCELL_FLAG, numSCellMapping, sizeof(int), false, true);
	PrepareAux( AUX_SUBCELL_MAPPING, numSCellMapping, sizeof(int), false);

	void* argsSetFlag[3] = { &cuVDBInfo, &numSCellMapping, &mAux[AUX_SUBCELL_FLAG].gpu};
	cudaCheck(cuLaunchKernel(cuFunc[FUNC_SET_FLAG_SUBCELL], pblks, 1, 1, threads, 1, 1, 0, NULL, argsSetFlag, NULL),
					"VolumeGVDB", "InsertPointsSubcell", "cuLaunch", "FUNC_SET_FLAG_SUBCELL", mbDebug);


	PrefixSum ( mAux[AUX_SUBCELL_FLAG].gpu, mAux[AUX_SUBCELL_MAPPING].gpu, numSCellMapping);


	//////////////////////////////////////////////////////////////////////////
	PERF_PUSH("SC counting");

	// subcell count	
	pblks = int(num_pnts / threads)+1;
	PrepareAux ( AUX_SUBCELL_CNT, numSCell, sizeof(int), true );
	
	void* argsCounting[14] = { &cuVDBInfo, &subcell_size, &numSCellPerBrick, 
					&num_pnts, &mAux[AUX_PNTPOS].gpu, &mAux[AUX_PNTPOS].subdim.x, &mAux[AUX_PNTPOS].stride, 
					&mAux[AUX_SUBCELL_CNT].gpu, &range.x, &trans.x, &SCDim, &pRadius, &numSCell, &mAux[AUX_SUBCELL_MAPPING].gpu};
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_COUNT_SUBCELL], pblks, 1, 1, threads, 1, 1, 0, NULL, argsCounting, NULL ),
					"VolumeGVDB", "InsertPointsSubcell", "cuLaunch", "FUNC_COUNT_SUBCELL", mbDebug);		

	PERF_POP();

#if 0
	int* sc_cnt = (int*) malloc(numSCell * sizeof(int));
	cudaCheck ( cuMemcpyDtoH ( sc_cnt, mAux[AUX_SUBCELL_CNT].gpu, numSCell * sizeof(int) ), "InsertPointsSubcell","","","",false );
	int maxSCCnt = 0;
	int minSCCnt = 1000000;
	int totalCnt = 0;
	for (int i = 0; i < numSCell; i++) 
	{
		if(sc_cnt[i] > maxSCCnt)  maxSCCnt = sc_cnt[i];
		if(sc_cnt[i] < minSCCnt)  minSCCnt = sc_cnt[i];
		totalCnt += sc_cnt[i];
	}
	std::cout << "# Subcells: " << numSCell << std::endl;
	std::cout << "Pnt/Subcell (Max): " << maxSCCnt << std::endl;
	std::cout << "Pnt/Subcell (Min): " << minSCCnt << std::endl;
	std::cout << "Pnt/Subcell (Ave): " << totalCnt/float(numSCell) << std::endl;	
	std::cout << "Total Subcell Pnt: " << totalCnt << std::endl;
	delete sc_cnt;
#endif
	
	//////////////////////////////////////////////////////////////////////////
	// prefixsum
	
	PERF_PUSH("PrefixSum");
	PrepareAux ( AUX_SUBCELL_PREFIXSUM, numSCell, sizeof(int), true );
	PrefixSum ( mAux[AUX_SUBCELL_CNT].gpu, mAux[AUX_SUBCELL_PREFIXSUM].gpu, numSCell);
	PERF_POP();

	//////////////////////////////////////////////////////////////////////////
	PERF_PUSH("Fetching");
	int offset_last, cnt_last;
	cudaCheck ( cuMemcpyDtoH ( &offset_last, mAux[AUX_SUBCELL_PREFIXSUM].gpu + (numSCell - 1) * sizeof(int), sizeof(int) ),
					"VolumeGVDB", "InsertPointsSubcell", "cuMemcpyDtoH", "AUX_SUBCELL_PREFIXSUM", mbDebug);	
	cudaCheck ( cuMemcpyDtoH ( &cnt_last, mAux[AUX_SUBCELL_CNT].gpu + (numSCell - 1) * sizeof(int), sizeof(int) ), 
					"VolumeGVDB", "InsertPointsSubcell", "cuMemcpyDtoH", "AUX_SUBCELL_CNT", mbDebug);	
	pSCPntsLength = offset_last + cnt_last;		// total number of subcell points
	if (pSCPntsLength == 0 ) return;

	PrepareAux ( AUX_SUBCELL_CNT, numSCell, sizeof(int), true );
	PrepareAux ( AUX_SUBCELL_PNT_POS, pSCPntsLength, sizeof(float3), false );
	if (mAux[AUX_PNTVEL].gpu != 0x0) PrepareAux ( AUX_SUBCELL_PNT_VEL, pSCPntsLength, sizeof(float3), false );
	if (mAux[AUX_PNTCLR].gpu != 0x0) PrepareAux ( AUX_SUBCELL_PNT_CLR, pSCPntsLength, sizeof(uint), false);
	PERF_POP();

	//////////////////////////////////////////////////////////////////////////
	PERF_PUSH("SC insert");
	void* argsInsert[23] = { &cuVDBInfo, &subcell_size, &numSCellPerBrick, &num_pnts, 		
		&mAux[AUX_SUBCELL_CNT].gpu, &mAux[AUX_SUBCELL_PREFIXSUM].gpu, &mAux[AUX_SUBCELL_MAPPING].gpu,
		&range.x, &trans.x, &SCDim, &pRadius, 
		&mAux[AUX_PNTPOS].gpu, &mAux[AUX_PNTPOS].subdim.x, &mAux[AUX_PNTPOS].stride, &mAux[AUX_SUBCELL_PNT_POS].gpu,		
		&mAux[AUX_PNTVEL].gpu, &mAux[AUX_PNTVEL].subdim.x, &mAux[AUX_PNTVEL].stride, &mAux[AUX_SUBCELL_PNT_VEL].gpu,
		&mAux[AUX_PNTCLR].gpu, &mAux[AUX_PNTCLR].subdim.x, &mAux[AUX_PNTCLR].stride, &mAux[AUX_SUBCELL_PNT_CLR].gpu
	};
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_INSERT_SUBCELL], pblks, 1, 1, threads, 1, 1, 0, NULL, argsInsert, NULL ), 
					"VolumeGVDB", "InsertPointsSubcell", "cuLaunch", "FUNC_INSERT_SUBCELL", mbDebug);		
	PERF_POP();

	//////////////////////////////////////////////////////////////////////////
	PERF_PUSH("SC pos");
	PrepareAux ( AUX_SUBCELL_POS, numSCell, sizeof(int3), false );
	PrepareAux ( AUX_SUBCELL_NID, numSCell, sizeof(int), false );
	void* argsSCPos[10] = { &cuVDBInfo, &mAux[AUX_SUBCELL_NID].gpu, &mAux[AUX_SUBCELL_POS].gpu, &numSCellMapping, &SCDim, &numSCellPerBrick, &subcell_size, &mAux[AUX_SUBCELL_MAPPING].gpu};
	pblks = int(numSCellMapping / threads)+1;
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_CALC_SUBCELL_POS], pblks, 1, 1, threads, 1, 1, 0, NULL, argsSCPos, NULL ), 
					"VolumeGVDB", "InsertPointsSubcell", "cuLaunch", "FUNC_CALC_SUBCELL_POS", mbDebug);		
	PERF_POP();

	PERF_POP();

	POP_CTX
}

void VolumeGVDB::InsertPointsSubcell_FP16(int subcell_size, int num_pnts, float pRadius, Vector3DF trans,  int& pSCPntsLength)
{
	PrepareVDB();

	PUSH_CTX

		PERF_PUSH("Insert pnt sc");

	int numSCellPerBrick = static_cast<int>(pow(getRes(0) / subcell_size, 3));
	int numBrick = static_cast<int>(mPool->getPoolUsedCnt(0, 0));
	int numSCell = numSCellPerBrick * numBrick;
	int SCDim = getRes(0) / subcell_size;

	Vector3DI range = getRange(0) / (getRes(0) / subcell_size);

	int numSCellMapping = static_cast<int>(mPool->getPoolTotalCnt(0, 0));

	int threads = 512;
	int pblks = int(numSCellMapping / threads) + 1;

	PrepareAux(AUX_SUBCELL_FLAG, numSCellMapping, sizeof(int), false);
	PrepareAux(AUX_SUBCELL_MAPPING, numSCellMapping, sizeof(int), false);

	void* argsSetFlag[3] = { &cuVDBInfo, &numSCellMapping, &mAux[AUX_SUBCELL_FLAG].gpu };
	cudaCheck(cuLaunchKernel(cuFunc[FUNC_SET_FLAG_SUBCELL], pblks, 1, 1, threads, 1, 1, 0, NULL, argsSetFlag, NULL), 
					"VolumeGVDB", "InsertPointsSubcell_FP16", "cuLaunch", "FUNC_SET_FLAG_SUBCELL", mbDebug);		

	PrefixSum(mAux[AUX_SUBCELL_FLAG].gpu, mAux[AUX_SUBCELL_MAPPING].gpu, numSCellMapping);

	//////////////////////////////////////////////////////////////////////////
	PERF_PUSH("SC counting");

	// subcell count	
	pblks = int(num_pnts / threads) + 1;
	PrepareAux(AUX_SUBCELL_CNT, numSCell, sizeof(int), true);

	void* argsCounting[14] = { &cuVDBInfo, &subcell_size, &numSCellPerBrick, &num_pnts, &mAux[AUX_PNTPOS].gpu, &mAux[AUX_PNTPOS].subdim.x, &mAux[AUX_PNTPOS].stride,
		&mAux[AUX_SUBCELL_CNT].gpu, &range.x, &trans.x, &SCDim, &pRadius, &numSCell, &mAux[AUX_SUBCELL_MAPPING].gpu };
	cudaCheck(cuLaunchKernel(cuFunc[FUNC_COUNT_SUBCELL], pblks, 1, 1, threads, 1, 1, 0, NULL, argsCounting, NULL), 
				"VolumeGVDB", "InsertPointsSubcell_FP16", "cuLaunch", "FUNC_COUNT_SUBCELL", mbDebug);		

	PERF_POP();


	//////////////////////////////////////////////////////////////////////////
	// prefixsum

	PERF_PUSH("PrefixSum");
	PrepareAux(AUX_SUBCELL_PREFIXSUM, numSCell, sizeof(int), true);
	PrefixSum(mAux[AUX_SUBCELL_CNT].gpu, mAux[AUX_SUBCELL_PREFIXSUM].gpu, numSCell);
	PERF_POP();

	//////////////////////////////////////////////////////////////////////////
	PERF_PUSH("Fetching");
	int tmpa, tmpb;
	cudaCheck(cuMemcpyDtoH(&tmpa, mAux[AUX_SUBCELL_PREFIXSUM].gpu + (numSCell - 1) * sizeof(int), sizeof(int)), 
						"VolumeGVDB", "InsertPointsSubcell_FP16", "cuMemcpyDtoH", "AUX_SUBCELL_PREFIXSUM", mbDebug);		
	cudaCheck(cuMemcpyDtoH(&tmpb, mAux[AUX_SUBCELL_CNT].gpu + (numSCell - 1) * sizeof(int), sizeof(int)), 
						"VolumeGVDB", "InsertPointsSubcell_FP16", "cuMemcpyDtoH", "AUX_SUBCELL_CNT", mbDebug);
	pSCPntsLength = tmpa + tmpb;
	PrepareAux(AUX_SUBCELL_CNT, numSCell, sizeof(int), true);
	PrepareAux(AUX_SUBCELL_PNT_POS, pSCPntsLength, sizeof(ushort3), false);
	if (mAux[AUX_PNTVEL].gpu != 0x0) PrepareAux(AUX_SUBCELL_PNT_VEL, pSCPntsLength, sizeof(ushort3), false);
	if (mAux[AUX_PNTCLR].gpu != 0x0) PrepareAux(AUX_SUBCELL_PNT_CLR, pSCPntsLength, sizeof(uint), false);
	PERF_POP();

	//////////////////////////////////////////////////////////////////////////
	PERF_PUSH("SC insert");
	void* argsInsert[27] = { &cuVDBInfo, &subcell_size, &numSCellPerBrick, &num_pnts,
		&mAux[AUX_SUBCELL_CNT].gpu, &mAux[AUX_SUBCELL_PREFIXSUM].gpu, &mAux[AUX_SUBCELL_MAPPING].gpu,
		&mPosMin.x, &mPosRange.x, &mVelMin.x, &mVelRange.x,
		&range.x, &trans.x, &SCDim, &pRadius,
		&mAux[AUX_PNTPOS].gpu, &mAux[AUX_PNTPOS].subdim.x, &mAux[AUX_PNTPOS].stride, &mAux[AUX_SUBCELL_PNT_POS].gpu,
		&mAux[AUX_PNTVEL].gpu, &mAux[AUX_PNTVEL].subdim.x, &mAux[AUX_PNTVEL].stride, &mAux[AUX_SUBCELL_PNT_VEL].gpu,
		&mAux[AUX_PNTCLR].gpu, &mAux[AUX_PNTCLR].subdim.x, &mAux[AUX_PNTCLR].stride, &mAux[AUX_SUBCELL_PNT_CLR].gpu
	};

	cudaCheck(cuLaunchKernel(cuFunc[FUNC_INSERT_SUBCELL_FP16], pblks, 1, 1, threads, 1, 1, 0, NULL, argsInsert, NULL),
			"VolumeGVDB", "InsertPointsSubcell_FP16", "cuLaunch", "FUNC_INSERT_SUBCELL_FP16", mbDebug);		
	PERF_POP();

	//////////////////////////////////////////////////////////////////////////
	PERF_PUSH("SC pos");
	PrepareAux(AUX_SUBCELL_POS, numSCell, sizeof(int3), false);
	PrepareAux(AUX_SUBCELL_NID, numSCell, sizeof(int), false);
	void* argsSCPos[10] = { &cuVDBInfo, &mAux[AUX_SUBCELL_NID].gpu, &mAux[AUX_SUBCELL_POS].gpu, &numSCellMapping, &SCDim, &numSCellPerBrick, &subcell_size, &mAux[AUX_SUBCELL_MAPPING].gpu };
	pblks = int(numSCellMapping / threads) + 1;
	cudaCheck(cuLaunchKernel(cuFunc[FUNC_CALC_SUBCELL_POS], pblks, 1, 1, threads, 1, 1, 0, NULL, argsSCPos, NULL), 
				"VolumeGVDB", "InsertPointsSubcell_FP16", "cuLaunch", "FUNC_CALC_SUBCELL_POS", mbDebug);

	PERF_POP();

	PERF_POP();

	POP_CTX
}

void VolumeGVDB::MapExtraGVDB (int subcell_size)
{
	if (!mHasObstacle) return;

	PUSH_CTX
	PERF_PUSH("SC extra");
	
	int numSCellPerBrick = static_cast<int>(pow(getRes(0) / subcell_size, 3));
	int numBrick = static_cast<int>(mPool->getPoolUsedCnt(0, 0));
	int numSCell = numSCellPerBrick * numBrick;
	int SCDim = getRes(0) / subcell_size;

	Vector3DI range = getRange(0) / (getRes(0) / subcell_size);

	int numSCellMapping = static_cast<int>(mPool->getPoolTotalCnt(0, 0));

	int threads = 512;	

	PrepareAux(AUX_SUBCELL_OBS_NID, numSCell, sizeof(int), false);
	void* argsSCPos[10] = { &cuVDBInfo, &numSCellMapping, &SCDim, &numSCellPerBrick, &subcell_size,  &mAux[AUX_SUBCELL_MAPPING].gpu, &cuOBSVDBInfo, &mAux[AUX_SUBCELL_OBS_NID].gpu };
	int pblks = int(numSCellMapping / threads) + 1;
	cudaCheck(cuLaunchKernel(cuFunc[FUNC_MAP_EXTRA_GVDB], pblks, 1, 1, threads, 1, 1, 0, NULL, argsSCPos, NULL), 
				"VolumeGVDB", "MapExtraGVDB", "cuLaunch", "FUNC_MAP_EXTRA_GVDB", mbDebug);		
	
	PERF_POP();
	POP_CTX
}

void VolumeGVDB::InsertPoints ( int num_pnts, Vector3DF trans, bool bPrefix )
{
	PrepareVDB ();	

	PUSH_CTX

	PERF_PUSH("InsertPoints");

	// Prepare aux arrays
	PERF_PUSH ( "Prepare Aux");
	int bricks = static_cast<int>(mPool->getAtlas(0).lastEle);
	PrepareAux ( AUX_PNODE, num_pnts, sizeof(int), false );			// node which each point falls into
	PrepareAux ( AUX_PNDX,  num_pnts, sizeof(int), false );			// index of the point inside that node
	PrepareAux ( AUX_GRIDCNT, bricks, sizeof(int), true );			// number of points in each brick cell
	PERF_POP();
	
	// Insert particles
	PERF_PUSH ( "Insert kernel");
	int threads = 512;		
	int pblks = int(num_pnts / threads)+1;
	void* args[9] = { &cuVDBInfo, &num_pnts, &mAux[AUX_PNTPOS].gpu, &mAux[AUX_PNTPOS].subdim.x, &mAux[AUX_PNTPOS].stride, &mAux[AUX_PNODE].gpu, &mAux[AUX_PNDX].gpu, &mAux[AUX_GRIDCNT].gpu, &trans.x };
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_INSERT_POINTS], pblks, 1, 1, threads, 1, 1, 0, NULL, args, NULL ), 
				"VolumeGVDB", "InsertPoints", "cuLaunch", "FUNC_INSERT_POINTS", mbDebug);

	PERF_POP();

	if ( bPrefix ) {
		// Prefix sum for brick offsets
		PERF_PUSH ( "  Prefix sum");
		PrepareAux ( AUX_GRIDOFF, bricks, sizeof(int), false );
		PrefixSum ( mAux[AUX_GRIDCNT].gpu, mAux[AUX_GRIDOFF].gpu, bricks );
		PERF_POP();

		PERF_PUSH ( "  Sort points");
		PrepareAux ( AUX_PNTSORT, num_pnts, sizeof(Vector3DF), false );
  	    void* args[13] = { &cuVDBInfo, &num_pnts, &mAux[AUX_PNTPOS].gpu, &mAux[AUX_PNTPOS].subdim.x, &mAux[AUX_PNTPOS].stride, &mAux[AUX_PNODE].gpu, &mAux[AUX_PNDX].gpu, 
									&bricks, &mAux[AUX_GRIDCNT].gpu, &mAux[AUX_GRIDOFF].gpu, &mAux[AUX_PNTSORT].gpu, &trans.x  };
				
		cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_SORT_POINTS], pblks, 1, 1, threads, 1, 1, 0, NULL, args, NULL ), 
						"VolumeGVDB", "InsertPoints", "cuLaunch", "FUNC_SORT_POINTS", mbDebug);			
		
		PERF_POP();
	}	

	PERF_POP();	// InsertParticles

	POP_CTX

}

void VolumeGVDB::InsertSupportPoints ( int num_pnts, float offset, Vector3DF trans, bool bPrefix )
{	
	PrepareVDB ();	

	PUSH_CTX

	PERF_PUSH("InsertPoints");	

	// Prepare aux arrays
	PERF_PUSH ( "Prepare Aux");
	int bricks = static_cast<int>(mPool->getAtlas(0).usedNum);
	PrepareAux ( AUX_PNODE, num_pnts, sizeof(int), false );			// node which each point falls into
	PrepareAux ( AUX_PNDX,  num_pnts, sizeof(int), false );			// index of the point inside that node
	PrepareAux ( AUX_GRIDCNT, bricks, sizeof(int), true );			// number of points in each brick cell
	PERF_POP();
	
	// Insert particles
	PERF_PUSH ( "Insert kernel");
	int threads = 512;		
	int pblks = int(num_pnts / threads)+1;
	void* args[12] = { &num_pnts, &offset,
		&mAux[AUX_PNTPOS].gpu, &mAux[AUX_PNTPOS].subdim.x, &mAux[AUX_PNTPOS].stride, 
		&mAux[AUX_PNODE].gpu, &mAux[AUX_PNDX].gpu, &mAux[AUX_GRIDCNT].gpu, 
		&mAux[AUX_PNTDIR].gpu, &mAux[AUX_PNTDIR].subdim.x, &mAux[AUX_PNTDIR].stride,
		&trans.x };
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_INSERT_SUPPORT_POINTS], pblks, 1, 1, threads, 1, 1, 0, NULL, args, NULL ), 
					"VolumeGVDB", "InsertSupportPoints", "cuLaunch", "FUNC_INSERT_SUPPORT_POINTS", mbDebug);		
	PERF_POP();

	PERF_POP();	// InsertParticles

	POP_CTX

}


void VolumeGVDB::ScalePntPos (int num_pnts, float scale)
{
	PUSH_CTX

	int threads = 512;		
	int pblks = int(num_pnts / threads)+1;
	void* args[5] = { &num_pnts, &mAux[AUX_PNTPOS].gpu, &mAux[AUX_PNTPOS].subdim.x, &mAux[AUX_PNTPOS].stride, &scale };
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_SCALE_PNT_POS], pblks, 1, 1, threads, 1, 1, 0, NULL, args, NULL ),
					"VolumeGVDB", "ScalePntPos", "cuLaunch", "FUNC_SCALE_PNT_POS", mbDebug); 

	POP_CTX
}

void VolumeGVDB::GatherDensity ( int subcell_size, int num_pnts, float radius, Vector3DF trans, int& pSCPntsLength, int chanDensity, int chanClr, bool bAccum )
{
	int num_brick = static_cast<int>(mPool->getPoolUsedCnt(0, 0));
	if (pSCPntsLength==0) return;
	if (num_brick == 0) return;

	PrepareVDB();

	PERF_PUSH("Gather Density");

	int numSCell = static_cast<int>(pow(getRes(0) / subcell_size, 3)) *  num_brick;

	void* args[14] = 
	{  
		&cuVDBInfo, &num_pnts, &numSCell, &radius,
		&mAux[AUX_SUBCELL_NID].gpu, &mAux[AUX_SUBCELL_CNT].gpu, &mAux[AUX_SUBCELL_PREFIXSUM].gpu, &mAux[AUX_SUBCELL_POS].gpu, 
		&mAux[AUX_SUBCELL_PNT_POS].gpu, &mAux[AUX_SUBCELL_PNT_VEL].gpu, &mAux[AUX_SUBCELL_PNT_CLR].gpu,
		&chanDensity, &chanClr, &bAccum
	};

	PUSH_CTX
		cudaCheck(cuLaunchKernel(cuFunc[FUNC_GATHER_DENSITY], numSCell, 1, 1, subcell_size, subcell_size, subcell_size, 0, NULL, args, NULL), 
					"VolumeGVDB", "GatherDensity", "cuLaunch", "FUNC_GATHER_DENSITY", mbDebug);			
	POP_CTX

	PERF_POP();
}



void VolumeGVDB::GatherLevelSet (int subcell_size, int num_pnts, float radius, Vector3DF trans, int& pSCPntsLength, int chanDensity, int chanClr, bool bAccum )
{
	int num_brick = static_cast<int>(mPool->getPoolUsedCnt(0, 0));
	if (num_brick == 0) return;

	PrepareVDB();

	PERF_PUSH("Gather LevelSet");

	int numSCell = static_cast<int>(pow(getRes(0) / subcell_size, 3)) *  num_brick;

	void* args[14] = 
	{  
		&cuVDBInfo, &num_pnts, &numSCell, &radius,
		&mAux[AUX_SUBCELL_NID].gpu, &mAux[AUX_SUBCELL_CNT].gpu, &mAux[AUX_SUBCELL_PREFIXSUM].gpu, &mAux[AUX_SUBCELL_POS].gpu, 
		&mAux[AUX_SUBCELL_PNT_POS].gpu, &mAux[AUX_SUBCELL_PNT_VEL].gpu, &mAux[AUX_SUBCELL_PNT_CLR].gpu,
		&chanDensity, &chanClr, &bAccum
	};

	PUSH_CTX
		cudaCheck(cuLaunchKernel(cuFunc[FUNC_GATHER_LEVELSET], numSCell, 1, 1, subcell_size, subcell_size, subcell_size, 0, NULL, args, NULL), 
					"VolumeGVDB", "GatherLevelSet", "cuLaunch", "FUNC_GATHER_LEVELSET", mbDebug);			
	POP_CTX

	PERF_POP();
}

void VolumeGVDB::GatherLevelSet_FP16(int subcell_size, int num_pnts, float radius, Vector3DF trans, int& pSCPntsLength, int chanDensity, int chanClr)
{
	int num_brick = static_cast<int>(mPool->getPoolUsedCnt(0, 0));
	if (num_brick == 0) return;

	PrepareVDB();

	PERF_PUSH("Gather LevelSet");

	int numSCell = static_cast<int>(pow(getRes(0) / subcell_size, 3)) * num_brick;
	
	void* args[23] =
	{
		&cuVDBInfo, &num_pnts, &numSCell, &radius,
		&mPosMin.x, &mPosRange.x, &mVelMin.x, &mVelRange.x,
		&mAux[AUX_SUBCELL_NID].gpu, &mAux[AUX_SUBCELL_CNT].gpu, &mAux[AUX_SUBCELL_PREFIXSUM].gpu, &mAux[AUX_SUBCELL_POS].gpu, 
		&mAux[AUX_SUBCELL_PNT_POS].gpu, &mAux[AUX_SUBCELL_PNT_VEL].gpu, &mAux[AUX_SUBCELL_PNT_CLR].gpu,
		&chanDensity, &chanClr
	};

	PUSH_CTX
		cudaCheck(cuLaunchKernel(cuFunc[FUNC_GATHER_LEVELSET_FP16], numSCell, 1, 1, subcell_size, subcell_size, subcell_size, 0, NULL, args, NULL), 
					"VolumeGVDB", "GatherLevelSet_FP16", "cuLaunch", "FUNC_GATHER_LEVELSET_FP16", mbDebug);			
	POP_CTX

		PERF_POP();
}


void VolumeGVDB::GetBoundingBox(int num_pnts, Vector3DF pTrans)
{
	PUSH_CTX

		PERF_PUSH("GPU bounding box");

	PERF_PUSH("Split pos");
	PrepareAux(AUX_WORLD_POS_X, num_pnts, sizeof(float), false);
	PrepareAux(AUX_WORLD_POS_Y, num_pnts, sizeof(float), false);
	PrepareAux(AUX_WORLD_POS_Z, num_pnts, sizeof(float), false);

	int threads = 256;
	int pblks = int(num_pnts / threads) + 1;
	void* args[8] =
	{
		&num_pnts, &pTrans.x,
		&mAux[AUX_PNTPOS].gpu, &mAux[AUX_PNTPOS].subdim.x, &mAux[AUX_PNTPOS].stride,
		&mAux[AUX_WORLD_POS_X].gpu, &mAux[AUX_WORLD_POS_Y].gpu, &mAux[AUX_WORLD_POS_Z].gpu
	};

	cudaCheck(cuLaunchKernel(cuFunc[FUNC_SPLIT_POS], pblks, 1, 1, threads, 1, 1, 0, NULL, args, NULL),
		"VolumeGVDB", "GetBoundingBox", "cuLaunch", "FUNC_SPLIT_POS", mbDebug);
	PERF_POP();

	PERF_PUSH("Reduce");
	PrepareAux(AUX_BOUNDING_BOX, 6, sizeof(float), false, true);

	const int axes[] = {
		AUX_WORLD_POS_X,
		AUX_WORLD_POS_Y,
		AUX_WORLD_POS_Z
	};

	for (uint32_t i = 0; i < 6; ++i) {
		if (i < 3) {
			gvdbDeviceMinElementF((mAux[AUX_BOUNDING_BOX].gpu + i * sizeof(float)),mAux[axes[i%3]].gpu,num_pnts);
		}
		else {
			gvdbDeviceMaxElementF((mAux[AUX_BOUNDING_BOX].gpu + i * sizeof(float)), mAux[axes[i % 3]].gpu, num_pnts);
		}
	}

	PERF_POP();

	RetrieveData(mAux[AUX_BOUNDING_BOX]);

	float* dat = (float*)getDataPtr(0, mAux[AUX_BOUNDING_BOX]);

	mPosMin.x = dat[0] - 1; mPosMin.y = dat[1] - 1; mPosMin.z = dat[2] - 1;
	mPosMax.x = dat[3] + 2; mPosMax.y = dat[4] + 2; mPosMax.z = dat[5] + 2;

	mPosMin.x = 0 > mPosMin.x ? 0 : mPosMin.x;
	mPosMin.y = 0 > mPosMin.y ? 0 : mPosMin.y;
	mPosMin.z = 0 > mPosMin.z ? 0 : mPosMin.z;
	mPosMax.x = 0 > mPosMax.x ? 0 : mPosMax.x;
	mPosMax.y = 0 > mPosMax.y ? 0 : mPosMax.y;
	mPosMax.z = 0 > mPosMax.z ? 0 : mPosMax.z;

	mPosRange = mPosMax - mPosMin;

	PERF_POP();

	POP_CTX
}

void VolumeGVDB::GetMinMaxVel(int num_pnts)
{
	Vector3DF pTrans(0, 0, 0);
	PUSH_CTX

		PERF_PUSH("Vel bounding box");

	PERF_PUSH("Split pos");
	PrepareAux(AUX_WORLD_VEL_X, num_pnts, sizeof(float), false);
	PrepareAux(AUX_WORLD_VEL_Y, num_pnts, sizeof(float), false);
	PrepareAux(AUX_WORLD_VEL_Z, num_pnts, sizeof(float), false);

	int threads = 256;
	int pblks = int(num_pnts / threads) + 1;
	void* args[8] =
	{
		&num_pnts, &pTrans.x,
		&mAux[AUX_PNTVEL].gpu, &mAux[AUX_PNTVEL].subdim.x, &mAux[AUX_PNTVEL].stride,
		&mAux[AUX_WORLD_VEL_X].gpu, &mAux[AUX_WORLD_VEL_Y].gpu, &mAux[AUX_WORLD_VEL_Z].gpu
	};

	cudaCheck(cuLaunchKernel(cuFunc[FUNC_SPLIT_POS], pblks, 1, 1, threads, 1, 1, 0, NULL, args, NULL), "VolumeGVDB", "GetMinMaxVel", "cuLaunch", "FUNC_SPLIT_POS", mbDebug);
	PERF_POP();

	PERF_PUSH("Reduce");
	PrepareAux(AUX_VEL_BOUNDING_BOX, 6, sizeof(float), false, true);

	const int axes[] = {
		AUX_WORLD_VEL_X,
		AUX_WORLD_VEL_Y,
		AUX_WORLD_VEL_Z
	};

	for (uint32_t i = 0; i < 6; ++i) {
		if (i < 3) {
			gvdbDeviceMinElementF((mAux[AUX_VEL_BOUNDING_BOX].gpu + i * sizeof(float)), mAux[axes[i % 3]].gpu, num_pnts);
		}
		else {
			gvdbDeviceMaxElementF((mAux[AUX_VEL_BOUNDING_BOX].gpu + i * sizeof(float)), mAux[axes[i % 3]].gpu, num_pnts);
		}
	}

	PERF_POP();

	RetrieveData(mAux[AUX_VEL_BOUNDING_BOX]);

	float* dat = (float*)getDataPtr(0, mAux[AUX_VEL_BOUNDING_BOX]);

	mVelMin.x = dat[0]; mVelMin.y = dat[1]; mVelMin.z = dat[2];
	mVelMax.x = dat[3]; mVelMax.y = dat[4]; mVelMax.z = dat[5];

	mVelRange = mVelMax - mVelMin;

	PERF_POP();

	POP_CTX
}

char* VolumeGVDB::GetTestPtr()
{
	return mAux[AUX_TEST].cpu;
}

void VolumeGVDB::ReadGridVel(int N)
{
	PUSH_CTX

	int cellNum = N * N * N;
	PrepareAux(AUX_TEST, cellNum, sizeof(float), true, true);		// cell_vel
	PrepareAux(AUX_TEST_1, cellNum, sizeof(int) * 3, true, true);	// cell_pos

	int3* cell_pos = (int3*)mAux[AUX_TEST_1].cpu;

	int index = 0;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				cell_pos[index].x = i;
				cell_pos[index].y = j;
				cell_pos[index++].z = k;
			}
		}
	}

	CommitData(mAux[AUX_TEST_1]);

	int threads = 256;
	int pblks = int(cellNum / threads) + 1;

	void* args[4] = { &cuVDBInfo, &cellNum, &mAux[AUX_TEST_1].gpu, &mAux[AUX_TEST].gpu };

	cudaCheck(cuLaunchKernel(cuFunc[FUNC_READ_GRID_VEL], pblks, 1, 1, threads, 1, 1, 0, NULL, args, NULL), "VolumeGVDB", "ReadGridVel", "cuLaunch", "FUNC_READ_GRID_VEL", mbDebug);

	RetrieveData(mAux[AUX_TEST]);
	float* cell_vel = (float *)mAux[AUX_TEST].cpu;
	index = 0;
	std::ofstream myfile;
	myfile.open("G_matrix_free.txt");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			myfile << "C " << i << " " << j << " : ";
			for (int k = 0; k < N; k++) {
				myfile << cell_vel[index] << " ";
				index++;
			}
			myfile << std::endl;
		}
	}
	//std::cout << "----------------\n";
	myfile.close();

	POP_CTX
}


void VolumeGVDB::CheckVal(float slice, int chanVx, int chanVy, int chanVz, int chanVxOld, int chanVyOld, int chanVzOld)
{
	PUSH_CTX

		Vector3DI res = Vector3DI(30, 30, 30);

	Vector3DI block(8, 1, 8);
	Vector3DI grid(int(res.x / block.x) + 1, 1, int(res.z / block.z) + 1);

	PrepareAux(AUX_OUT1, res.x*res.z, sizeof(float), true, true);
	PrepareAux(AUX_OUT2, res.x*res.z, sizeof(float), true, true);

	float* out1 = (float*)mAux[AUX_OUT1].cpu;
	float* out2 = (float*)mAux[AUX_OUT2].cpu;
	for (int z = 0; z < res.z; z++)
		for (int x = 0; x < res.x; x++) {
			out1[z*res.x + x] = -1.0;
			out2[z*res.x + x] = -1.0;
		}
	CommitData(mAux[AUX_OUT1]);
	CommitData(mAux[AUX_OUT2]);


	void* args[11] = { &cuVDBInfo, &slice, &res, &chanVx, &chanVy, &chanVz, &chanVxOld, &chanVyOld, &chanVzOld, &mAux[AUX_OUT1].gpu, &mAux[AUX_OUT2].gpu };
	cudaCheck(cuLaunchKernel(cuFunc[FUNC_CHECK_VAL], grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args, NULL), "VolumeGVDB", "CheckVal", "cuLaunch", "FUNC_CHECK_VAL", mbDebug);

	RetrieveData(mAux[AUX_OUT1]);
	RetrieveData(mAux[AUX_OUT2]);

	cudaCheck(cuCtxSynchronize(), "VolumeGVDB", "CheckVal", "cuCtxSynchronize", "", mbDebug);

	for (int z = 0; z < 10; z++) {
		for (int x = 0; x < 10; x++) {
			gprintf("%4.3f/%4.3f ", out1[z*res.x + x], out2[z*res.x + x]);
		}
		gprintf("\n");
	}
	POP_CTX
}

void VolumeGVDB::ScatterDensity ( int num_pnts, float radius, float amp, Vector3DF trans, bool expand, bool avgColor )
{
	uint num_voxels; 

	PrepareVDB ();	

	PUSH_CTX

	// Splat particles
	PERF_PUSH ( "ScatterPointDensity");
	int threads = 256;		
	int pblks = int(num_pnts / threads)+1;	
	
    if (mAux[AUX_PNTCLR].gpu != NULL && avgColor) {
		Vector3DI brickResVec = getRes3DI(0);
		num_voxels = brickResVec.x * brickResVec.y * brickResVec.z * static_cast<uint>(getNumUsedNodes(0));
		if (mbProfile) PERF_PUSH("Prepare Aux");
		PrepareAux(AUX_COLAVG, 4 * num_voxels, sizeof(uint), true);					// node which each point falls into
		if (mbProfile) PERF_POP();
    }
	
	void* args[13] = { &num_pnts, &radius, &amp, &mAux[AUX_PNTPOS].gpu, &mAux[AUX_PNTPOS].subdim.x, &mAux[AUX_PNTPOS].stride, &mAux[AUX_PNTCLR].gpu, &mAux[AUX_PNTCLR].subdim.x, &mAux[AUX_PNTCLR].stride, &mAux[AUX_PNODE].gpu, &trans.x, &expand, &mAux[AUX_COLAVG].gpu };
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_SCATTER_DENSITY], pblks, 1, 1, threads, 1, 1, 0, NULL, args, NULL ), 
				"VolumeGVDB", "ScatterPointDensity", "cuLaunch", "FUNC_SCATTER_DENSITY", mbDebug);		

	if (mAux[AUX_PNTCLR].gpu != NULL && avgColor) {
		int threads_avgcol = 256;
		int pblks_avgcol = int(num_voxels / threads_avgcol) + 1;
		void* args_avgcol[2] = { &num_voxels, &mAux[AUX_COLAVG].gpu };
		cudaCheck( cuLaunchKernel (cuFunc[FUNC_SCATTER_AVG_COL], pblks_avgcol, 1, 1, threads_avgcol, 1, 1, 0, NULL, args_avgcol, NULL), 
				"VolumeGVDB", "ScatterPointDensity", "cuLaunch", "FUNC_SCATTER_AVG_COL", mbDebug);			
	}

	PERF_POP ();	

	POP_CTX
}

// psrcbits / pdestbits are:
//   1 =   byte components,  3 bytes/point 
//   2 = ushort components   6 bytes/point
//   4 =  float components, 12 bytes/point
//   8 = double components, 24 bytes/point
//
void VolumeGVDB::ConvertAndTransform(DataPtr& psrc, char psrcbits, DataPtr& pdest, char pdestbits, int num_pnts, Vector3DF wMin, Vector3DF wDelta, Vector3DF trans, Vector3DF scal)
{
	PUSH_CTX

		int threads = 512;
	int pblks = int(num_pnts / threads) + 1;
	void* args[9] = { &num_pnts,
		&psrc.gpu, &psrcbits, &pdest.gpu, &pdestbits,
		&wMin, &wDelta, &trans, &scal };

	cudaCheck(cuLaunchKernel(cuFunc[FUNC_CONV_AND_XFORM], pblks, 1, 1, threads, 1, 1, 0, NULL, args, NULL),
		"VolumeGVDB", "ConvertAndTransform", "cuLaunch", "FUNC_CONV_AND_XFORM", mbDebug);

	POP_CTX
}


void VolumeGVDB::AddSupportVoxel ( int num_pnts, float radius, float offset, float amp, Vector3DF trans, bool expand, bool avgColor )
{
	PrepareVDB ();	

	PUSH_CTX

	// Splat particles
	PERF_PUSH ( "AddSupportVoxel");
	int threads = 256;		
	int pblks = int(num_pnts / threads)+1;	
	
	void* args[14] = { &num_pnts, &radius, &offset, &amp, 
		&mAux[AUX_PNTPOS].gpu, &mAux[AUX_PNTPOS].subdim.x, &mAux[AUX_PNTPOS].stride, 
		&mAux[AUX_PNTDIR].gpu, &mAux[AUX_PNTDIR].subdim.x, &mAux[AUX_PNTDIR].stride,
		&mAux[AUX_PNODE].gpu, &trans.x, &expand, &mAux[AUX_COLAVG].gpu };
	cudaCheck ( cuLaunchKernel ( cuFunc[FUNC_ADD_SUPPORT_VOXEL], pblks, 1, 1, threads, 1, 1, 0, NULL, args, NULL ),
				"VolumeGVDB", "AddSupportVoxel", "cuLaunch", "FUNC_ADD_SUPPORT_VOXEL", mbDebug);		

	PERF_POP ();	

	POP_CTX
}



void VolumeGVDB::PrintPool(uchar grp, uchar lev)
{
	DataPtr* p = mPool->getPool(grp, lev);

	std::cout << "Pool " << (int)grp << " at level " << (int)lev << std::endl;
	std::cout << "   num: " << p->lastEle << std::endl;

	for (int i = 0; i < p->lastEle; i++)
	{
		char* pool = p->cpu;
		nvdb::Node* nd = (nvdb::Node*) (pool + p->stride * i);
		std::cout << "   Node " << i << ":\n";
		
#ifdef USE_BITMASKS	
		if ((int)lev > 0) {
			std::cout << "      mMask: " << nd->countOn() << std::endl;
			uint64* w1 = (uint64*) &nd->mMask;
			uint64* we = (uint64*) &nd->mMask + nd->getMaskWords();
			
		}
		std::cout << "   Parent ID:" << nd->mParent << std::endl;
		std::cout << "   Childlist address:" << (int)nd->mChildList << std::endl;
		std::cout << "   Child ID:" << std::endl;
		if ((int)nd->mChildList > 0) {
			int numChild = nd->countOn();
			uint64* clist = mPool->PoolData64 ( nd->mChildList );
			std::cout << "   ";
			for (int i = 0; i < numChild; i++)
			{
				std::cout << clist[i] << " ";
			}
			std::cout << std::endl;
		}	
		std::cout << "------------------------------------\n";
#else
	
		std::cout << "   Parent ID:" << nd->mParent << std::endl;
		std::cout << "   Childlist address:" << (int)nd->mChildList << std::endl;
		int r = getRes(lev); 
		int childlistLen = ((uint64) r*r*r);
		std::cout << "   Childlist length:" << childlistLen << std::endl;
		std::cout << "   Child ID:" << std::endl;
		if ((int)nd->mChildList > 0) {
			uint64* clist = mPool->PoolData64 ( nd->mChildList );
			for (int i = 0; i < childlistLen; i++)
			{
				if ( clist[i] != ID_UNDEF64)
				{
					std::cout << i << " " << clist[i] << " ";
				}		
			}
			std::cout << std::endl;
		}	
		std::cout << "------------------------------------\n";
#endif
	}
	std::cout << "====================================\n";
}


void VolumeGVDB::SetTransform(Vector3DF pretrans, Vector3DF scal, Vector3DF angs, Vector3DF trans ) 
{
	mPretrans = pretrans;
	mScale = scal;
	mAngs = angs;
	mTrans = trans;

	// Compute the rotation-only part of the matrix, mXrot = R
	Matrix4F mXrot;
	mXrot.RotateZYX(mAngs);

	// Compute mInvXrot = (S R)^-1 = R^-1 S^-1
	mInvXrot.Identity();
	mInvXrot.InvScaleInPlace(mScale); // scale here supports non-uniform voxels
	mInvXrot.InvLeftMultiplyInPlace(mXrot);

	// Compute mXform = T R S PT (pretranslate, then scale, then rotate, then translate)
	mXform.Identity();
	mXform.RotateTZYXS(mAngs, mTrans, mScale); // T R S
	mXform.PreTranslate(mPretrans); // T R S PT

	// Compute mInvXform = mXform^-1
	mInvXform = mXform;
	mInvXform.InvertTRS();
}

// Node queries
uint64 VolumeGVDB::getNumUsedNodes ( int lev )		{ return mPool->getPoolUsedCnt(0, lev); }
uint64 VolumeGVDB::getNumTotalNodes ( int lev )		{ return mPool->getPoolTotalCnt(0, lev); }
Node* VolumeGVDB::getNodeAtLevel ( int n, int lev )	{ return (Node*) (mPool->PoolData( 0, lev, n )); }

Vector3DF VolumeGVDB::getWorldMin ( Node* node )	{ return Vector3DF(node->mPos); }
Vector3DF VolumeGVDB::getWorldMax ( Node* node )	{ return Vector3DF(node->mPos) + getCover(node->mLev); }

Vector3DF VolumeGVDB::getWorldMin() {
	return mObjMin;
}
Vector3DF VolumeGVDB::getWorldMax()
{
	return mObjMax;
}

//-----------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2016 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
// 
// Version 1.0: Rama Hoetzlein, 5/1/2017
// Version 1.1: Rama Hoetzlein, 3/25/2018
//-----------------------------------------------------------------------------

#ifndef DEF_VOL_BASE
	#define DEF_VOL_BASE
	
	#include "gvdb_allocator.h"
	#include "gvdb_types.h"
	#include "gvdb_vec.h"
	#include <stdint.h>		
	#include <stdarg.h>

	#pragma warning( disable : 4251 )

	namespace nvdb {

	class Scene;
	class Model;
	class Allocator;

	class VolStats {
		float		mem_used;			// in MB
		float		mem_effective;		// in MB
		Vector3DF	voxel_size;			// in microns (us)
		Vector3DI	voxel_res;			// in voxels
		float		load_time;			// in millisec
		float		raster_time;		// in millisec
		float		render_time;		// in millisec
	};

	class GVDB_API VolumeBase {
	public:
		// Destructor
		~VolumeBase();

		// Commit model geometry to GPU
		void CommitGeometry ( int model_id );
		void CommitGeometry ( Model* m );
		void ClearGeometry ( Model* m );
		
		// Query functions
		void			getDimensions ( Vector3DF& objmin, Vector3DF& objmax, Vector3DF& voxmin, Vector3DF& voxmax, Vector3DF& voxres );
		void			getTiming ( float& render_time );
		// Gets the transfer function as a DataPtr
		DataPtr			getTransferPtr ()		{ return mTransferPtr; }
		// Gets the transfer function as a CUdeviceptr
		CUdeviceptr		getTransferFuncGPU ()	{ return mTransferPtr.gpu; }
		// Gets the GVDB scene object
		Scene*			getScene()				{ return mScene; }
		// Get the minimum of the bounding box of the entire volume in voxels.
		Vector3DF		getVolMin ()			{ return mObjMin; }
		// Get the maximum of the bounding box of the entire volume in voxels.
		Vector3DF		getVolMax ()			{ return mObjMax; }

		void			SetProfiling ( bool tf )		{ mbProfile = tf; }
		void			SetVerbose ( bool tf )			{ mbVerbose = tf; }

	public:
		Vector3DF		mObjMin, mObjMax;			// world space
		Vector3DF		mVoxMin, mVoxMax, mVoxRes;	// coordinate space
		Vector3DF		mVoxResMax;

		Vector3DF		mRenderTime;
		bool			mbProfile;
		bool			mbVerbose;

		DataPtr			mTransferPtr;				// Transfer function
		std::vector<DataPtr	>	mRenderBuf;			// Non-owning list of render buffers (since apps can add their own render buffers)

		Allocator*		mPool = nullptr;			// Allocator

		Scene*			mScene = nullptr;			// Scene (non-owning pointer)
	};

	}

#endif

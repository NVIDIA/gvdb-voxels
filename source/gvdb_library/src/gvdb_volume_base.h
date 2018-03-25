//--------------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2016-2018, NVIDIA Corporation. 
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
// Version 1.1: Rama Hoetzlein, 3/25/2018
//----------------------------------------------------------------------------------

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

		// Commit model geometry to GPU
		void CommitGeometry ( int model_id );
		void CommitGeometry ( Model* m );
		void ClearGeometry ( Model* m );
		
		// Query functions
		void			getDimensions ( Vector3DF& objmin, Vector3DF& objmax, Vector3DF& voxmin, Vector3DF& voxmax, Vector3DF& voxsize, Vector3DF& voxres );
		void			getTiming ( float& render_time );
		DataPtr			getTransferPtr ()		{ return mTransferPtr; }
		CUdeviceptr		getTransferFuncGPU ()	{ return mTransferPtr.gpu; }
		Scene*			getScene()				{ return mScene; }
		Vector3DF		getVolMin ()			{ return mObjMin; }
		Vector3DF		getVolMax ()			{ return mObjMax; }

		void			SetProfiling ( bool tf )		{ mbProfile = tf; }
		void			SetVerbose ( bool tf )			{ mbVerbose = tf; }

	public:
		Vector3DF		mVoxsize;					// size of voxels (in microns)
		Vector3DF		mObjMin, mObjMax;			// world space
		Vector3DF		mVoxMin, mVoxMax, mVoxRes;	// coordinate space
		Vector3DF		mVoxResMax;

		Vector3DF		mRenderTime;
		bool			mbProfile;
		bool			mbVerbose;

		DataPtr			mTransferPtr;				// Transfer function
		std::vector<DataPtr	>	mRenderBuf;		// Render buffers

		Allocator*		mPool;						// Allocator

		Scene*			mScene;		
	};

	}

#endif

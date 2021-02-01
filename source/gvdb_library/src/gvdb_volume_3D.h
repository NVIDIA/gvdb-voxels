//-----------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2016 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
// 
// Version 1.0: Rama Hoetzlein, 5/1/2017
// Version 1.1: Rama Hoetzlein, 3/25/2018
//-----------------------------------------------------------------------------


#ifndef DEF_VOL_3D
	#define DEF_VOL_3D

	#include "gvdb_volume_base.h"
	#include "gvdb_allocator.h"

	namespace nvdb {

	class Volume3D : public VolumeBase {
	public:
		Volume3D ( Scene* scn );
		~Volume3D ();
		
		void Clear ();
		void Resize ( char typ, Vector3DI res, Matrix4F* xform, bool bGL );
		void CommitFromCPU ( float* src );
		void SetDomain ( Vector3DF vmin, Vector3DF vmax );
		void Empty ();
		void getMemory ( float& voxels, float& overhead, float& effective );
		
		// OpenGL Poly-to-Voxels
		void PrepareRasterGL ( bool start );
		void SurfaceVoxelizeGL ( uchar  chan, Model* model, Matrix4F* xform );	
		void SurfaceVoxelizeFastGL ( Vector3DF vmin, Vector3DF vmax, Matrix4F* model );
		void RetrieveGL ( char* dest );	

		DataPtr	getPtr ()	{ return mPool->getAtlas(0); }
		Vector3DI getSize ()	{ return mPool->getAtlas(0).subdim; }
		bool hasSize ( Vector3DI i, uchar dt )	{ 
			Vector3DI sz = mPool->getAtlas(0).subdim;
			return (sz.x==i.x && sz.y==i.y && sz.z==i.z && mPool->getAtlas(0).type==dt);
		}
	
	public:				
		int				mMargin;		

		// Allocator
		Allocator*		mPool;

		static int		mVFBO[2];		
		static int		mVCLEAR;
	};

	}

#endif

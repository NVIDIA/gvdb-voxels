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

#include "gvdb_vec.h"
using namespace nvdb;

#ifndef DEF_PIVOTX
	#define DEF_PIVOTX

	namespace nvdb {
	
	class GVDB_API PivotX {
	public:
		PivotX()	{ from_pos.Set(0,0,0); to_pos.Set(0,0,0); ang_euler.Set(0,0,0); scale.Set(1,1,1); trans.Identity(); }
		PivotX( Vector3DF& f, Vector3DF& t, Vector3DF& s, Vector3DF& a) { from_pos=f; to_pos=t; scale=s; ang_euler=a; }

		void setPivot ( float x, float y, float z, float rx, float ry, float rz );
		void setPivot ( Vector3DF& pos, Vector3DF& ang ) { from_pos = pos; ang_euler = ang; }
		void setPivot ( PivotX  piv )	{ from_pos = piv.from_pos; to_pos = piv.to_pos; ang_euler = piv.ang_euler; updateTform(); }		
		void setPivot ( PivotX& piv )	{ from_pos = piv.from_pos; to_pos = piv.to_pos; ang_euler = piv.ang_euler; updateTform(); }

		void setIdentity ()		{ from_pos.Set(0,0,0); to_pos.Set(0,0,0); ang_euler.Set(0,0,0); scale.Set(1,1,1); trans.Identity(); }

		void setAng ( float rx, float ry, float rz )	{ ang_euler.Set(rx,ry,rz);	updateTform(); }
		void setAng ( Vector3DF& a )					{ ang_euler = a;			updateTform(); }

		void setPos ( float x, float y, float z )		{ from_pos.Set(x,y,z);		updateTform(); }
		void setPos ( Vector3DF& p )					{ from_pos = p;				updateTform(); }

		void setToPos ( float x, float y, float z )		{ to_pos.Set(x,y,z);		updateTform(); }
		
		void updateTform ();
		void setTform ( Matrix4F& t )		{ trans = t; }
		inline Matrix4F& getTform ()		{ return trans; }
		inline float* getTformData ()		{ return trans.GetDataF(); }

		// Pivot		
		PivotX getPivot ()	{ return PivotX(from_pos, to_pos, scale, ang_euler); }
		Vector3DF& getPos ()			{ return from_pos; }
		Vector3DF& getToPos ()			{ return to_pos; }
		Vector3DF& getAng ()			{ return ang_euler; }
		Vector3DF getDir ()			{ 
			return to_pos - from_pos; 
		}

		Vector3DF	from_pos;
		Vector3DF	to_pos;
		Vector3DF	scale;
		Vector3DF	ang_euler;
		Matrix4F	trans;
		
		//Quatern	ang_quat;
		//Quatern	dang_quat;
	};

	}

#endif

#ifndef DEF_CAMERA_3D
	#define	DEF_CAMERA_3D
	
	#define DEG_TO_RAD			(3.141592/180.0)

	namespace nvdb {

	class GVDB_API Camera3D : public PivotX {
	public:
		enum eProjection {
			Perspective = 0,
			Parallel = 1
		};
		Camera3D ();
		void Copy ( Camera3D& op );

		void draw_gl();

		// Camera settings
		void setAspect ( float asp )					{ mAspect = asp;			updateMatricies(); }
		void setPos ( float x, float y, float z )		{ from_pos.Set(x,y,z);		updateMatricies(); }
		void setToPos ( float x, float y, float z )		{ to_pos.Set(x,y,z);		updateMatricies(); }
		void setFov (float fov)							{ mFov = fov;				updateMatricies(); }
		void setNearFar (float n, float f )				{ mNear = n; mFar = f;		updateMatricies(); }
		void setDolly(float d)							{ mDolly = d;				updateMatricies();  }
		void setDist ( float d )						{ mOrbitDist = d;			updateMatricies(); }
		void setTile ( float x1, float y1, float x2, float y2 )		{ mTile.Set ( x1, y1, x2, y2 );		updateMatricies(); }
		void setProjection (eProjection proj_type);
		void setModelMatrix ( float* mtx );
		void setViewMatrix ( float* mtx, float* invmtx );
		void setProjMatrix ( float* mtx, float* invmtx );
		virtual void setMatrices ( const float* view_mtx, const float* proj_mtx, Vector3DF model_pos );
		
		// Camera motion
		void setOrbit  ( float ax, float ay, float az, Vector3DF tp, float dist, float dolly );
		void setOrbit  ( Vector3DF angs, Vector3DF tp, float dist, float dolly );
		void setAngles ( float ax, float ay, float az );
		void moveOrbit ( float ax, float ay, float az, float dist );		
		void moveToPos ( float tx, float ty, float tz );		
		void moveRelative ( float dx, float dy, float dz );

		// Frustum testing
		bool pointInFrustum ( float x, float y, float z );
		bool boxInFrustum ( Vector3DF bmin, Vector3DF bmax);
		float calculateLOD ( Vector3DF pnt, float minlod, float maxlod, float maxdist );

		// Utility functions
		virtual void updateMatricies ();				// Updates camera axes and projection matricies
		void updateFrustum ();						// Updates frustum planes
		Vector3DF inverseRay ( float x, float y, float z );
		Vector3DF inverseRayProj ( float x, float y, float z );
		Vector4DF project ( Vector3DF& p );
		Vector4DF project ( Vector3DF& p, Matrix4F& vm );		// Project point - override view matrix

		void getVectors ( Vector3DF& dir, Vector3DF& up, Vector3DF& side )	{ dir = dir_vec; up = up_vec; side = side_vec; }
		void getBounds ( float dst, Vector3DF& min, Vector3DF& max );
		float getNear ()				{ return mNear; }
		float getFar ()					{ return mFar; }
		float getFov ()					{ return mFov; }
		float getDolly()				{ return mDolly; }	
		float getOrbitDist()			{ return mOrbitDist; }
		Vector3DF& getUpDir ()			{ return up_dir; }
		Vector4DF& getTile ()			{ return mTile; }
		Matrix4F& getInvViewProjMatrix () { return invviewproj_matrix; }
		Matrix4F& getViewMatrix ()		{ return view_matrix; }
		Matrix4F& getInvView ()			{ return invrot_matrix; }
		Matrix4F& getRotateMatrix ()	{ return rotate_matrix; }
		Matrix4F& getProjMatrix ()		{ return tileproj_matrix; }	
		Matrix4F& getFullProjMatrix ()	{ return proj_matrix; }
		Matrix4F& getModelMatrix()		{ return model_matrix; }
		Matrix4F& getMVMatrix()			{ return mv_matrix; }
		float getAspect ()				{ return mAspect; }
		Vector3DF getU ();
		Vector3DF getV ();
		Vector3DF getW ();
		float getDu ();
		float getDv ();


	public:
		eProjection		mProjType;								// Projection type

		// Camera Parameters									// NOTE: Pivot maintains camera from and orientation
		float			mDolly;									// Camera to distance
		float			mOrbitDist;
		float			mFov, mAspect;							// Camera field-of-view
		float			mNear, mFar;							// Camera frustum planes
		Vector3DF		dir_vec, side_vec, up_vec;				// Camera aux vectors (W, V, and U)
		Vector3DF		up_dir;
		Vector4DF		mTile;
		
		// Transform Matricies
		Matrix4F		invviewproj_matrix;
		Matrix4F		rotate_matrix;							// Vr matrix (rotation only)
		Matrix4F		view_matrix;							// V matrix	(rotation + translation)
		Matrix4F		proj_matrix;							// P matrix
		Matrix4F		invrot_matrix;							// Vr^-1 matrix
		Matrix4F		invproj_matrix;
		Matrix4F		tileproj_matrix;						// tiled projection matrix
		Matrix4F		model_matrix;
		Matrix4F		mv_matrix;
		float			frustum[6][4];							// frustum plane equations

		bool			mOps[8];
		bool			orbit_set_;
		int				mWire;
				
		Vector3DF		origRayWorld;
		Vector4DF		tlRayWorld;
    	Vector4DF		trRayWorld;
    	Vector4DF		blRayWorld;
    	Vector4DF		brRayWorld;
	};

	}

	typedef Camera3D		Light;

#endif

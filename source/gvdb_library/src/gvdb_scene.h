//-----------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2016 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
// 
// Version 1.0: Rama Hoetzlein, 5/1/2017
// Version 1.1: Rama Hoetzlein, 3/25/2018
//-----------------------------------------------------------------------------


#ifndef DEF_SCENE_H
	#define DEF_SCENE_H


	#include <vector>
	#include "gvdb_model.h"	
	#include "gvdb_camera.h"	
	#include "gvdb_allocator.h"

	#pragma warning (disable : 4251 )

	#define MAX_PATHS	32

	#define GLS_SIMPLE		0	// GL Shader Programs
	#define GLS_OUTLINE		1
	#define GLS_SLICE		2
	#define GLS_VOXELIZE	3
	#define GLS_RAYCAST		4
	#define GLS_INSTANCE	5
	#define GLS_SCREENTEX	6

	#define MAX_PROG		8

	class CallbackParser;

	namespace nvdb {

	struct ParamList {
		int p[128];
	};
	struct Key {
		char			obj;		// object type
		int				objid;		// object ID
		unsigned long	varid;		// variable ID
		float			f1, f2;		// frame start/end
		Vector3DF		val1, val2;	// value start/end		
	};
	struct Mat {
		int				id;		
		Vector3DF		mAmb, mDiff, mSpec;		
		Vector3DF		mParam[64];
	};	

	class GVDB_API Scene {
	public:
		Scene();
		~Scene();
		static Scene*			gScene;
		static CallbackParser*	gParse;

		void		Clear ();
		void		LoadFile ( std::string filename );				
		void		AddPath ( std::string path );
		bool		FindFile ( std::string fname, char* path );
		// Destroys the current camera if it exists, and sets an owning pointer to the current camera.
		Camera3D*	SetCamera ( Camera3D* cam );
		// Destroys light n if it exists, and sets light n to an owning pointer to the given light.
		Light*		SetLight ( int n, Light* light );
		Model*		AddModel ();
		size_t		AddModel ( std::string filename, float scale, float tx, float ty, float tz );
		size_t		AddVolume ( std::string filename, Vector3DI res, char vtype, float scale=1.0 );
		size_t		AddGround ( float hgt, float scale=1.0 );
		void		SetAspect ( int w, int h );
		void		ClearModel ( Model* m );
		void		LoadModel ( Model* m, std::string filestr, float scale, float tx, float ty, float tz );
		
		// Shaders
		int			AddShader ( int prog_id, const char* vertfile, const char* fragfile );
		int			AddShader ( int prog_id, const char* vertfile, const char* fragfile, const char* geomfile );
		int			AddParam (  int prog_id, int id, const char* name );
		int			AddAttrib ( int prog_id, int id, const char* name );
		int			getProgram(int prog_id) { return mProgram[prog_id]; }

		// Materials
		int			AddMaterial ();
		void		SetMaterial ( int model, Vector4DF amb, Vector4DF diff, Vector4DF spec );		
		void		SetMaterialParam ( int id, int p, Vector3DF val );
		void		SetOverrideMaterial ( Vector4DF amb, Vector4DF diff, Vector4DF spec );
		void		SetVolumeRange ( float viso, float vmin, float vmax );

		// Animation
		void		AddKey ( std::string obj, std::string var, int f1, int f2, Vector3DF val1, Vector3DF val2 );
		bool		DoAnimation ( int frame );
		void		UpdateValue (  char obj, int objid, long varid, Vector3DF val );
		void		RecordKeypoint ( int w, int h );
		int			getFrameSamples ()	{ return mFrameSamples; }
		void		setFrameSamples ( int n )	{ mFrameSamples = n; }
		
		// Transfer function		
		void		LinearTransferFunc ( float t0, float t1, Vector4DF a, Vector4DF b );		

		int			getShaderProgram(int i);			 
		int			getNumModels ()		{ return (int) mModels.size(); }
		Model*		getModel ( int n )	{ if (n < mModels.size()) return mModels[n]; else return 0x0; }
		Camera3D*	getCamera ()		{ return mCamera; }
		Light*		getLight ()			{ return mLights[0]; }
		int			getSlot ( int prog_id );
		int			getParam ( int prog_id, int id )	{ return mParams[ mProgToSlot[ mProgram[prog_id] ] ].p[id]; }
		bool		useOverride ()		{ return clrOverride; }
		Vector3DF	getShadowParams ()	{ return mShadowParams; }		
		Vector3DI	getFrameRange ()	{ return mVFrames; }
		int			getNumLights()		{ return (int) mLights.size(); }
		
		// Loading scenes
		static void	LoadPath ();		
		static void	LoadModel ();
		static void	LoadVolume ();
		static void	LoadGround ();
		static void LoadCamera ();
		static void LoadLight ();
		static void LoadAnimation ();
		static void LoadShadow ();		
		static void VolumeThresh ();
		static void VolumeTransfer ();
		static void VolumeClip ();

		void SetRes ( int x, int y )	{ mXres=x; mYres=y; SetAspect(x,y); }
		Vector3DI getRes ()				{ return Vector3DI(mXres,mYres,0); }
		Vector4DF* getTransferFunc ()	{ return mTransferFunc; }

		Vector4DF getBackClr()			{ return mBackgroundClr; }
		Vector3DF getExtinct()			{ return mExtinct; }
		Vector3DF getCutoff()			{ return mCutoff; }
		Vector3DF getSteps()			{ return mSteps; }
		Vector3DF getSectionPnt()		{ return mSectionPnt; }
		Vector3DF getSectionNorm()		{ return mSectionNorm; }
		int getShading()				{ return mShading; }		
		int getSample()					{ return mSample; }
		int getFrame()					{ return mFrame; }
		int getFilterMode()				{ return mFilterMode; }
		int getDepthBuf()				{ return mDepthBuf; }
		void SetExtinct ( float a, float b, float c )	{ mExtinct.Set(a,b,c); }
		// Sets how far each iteration marches in voxel space when raymarching.
		void SetSteps(float directStep, float shadowStep, float fineStep) { mSteps.Set(directStep, shadowStep, fineStep); }
		void SetCutoff ( float a, float b, float c )	{ mCutoff.Set(a,b,c); }
		void SetBackgroundClr ( float r, float g, float b, float a )	{ mBackgroundClr.Set(r,g,b,a); }
		void SetCrossSection ( Vector3DF pos, Vector3DF norm )	{ mSectionPnt = pos; mSectionNorm = norm; }		
		void SetShading ( uchar s )		{ mShading = s; }
		void SetShadowParams ( float x, float y, float z )	{mShadowParams.Set(x,y,z); }
		void SetSample ( int s )		{ mSample = s; }
		void SetFrame ( int f )			{ mFrame = f; }
		void SetFilterMode ( int f )	{ mFilterMode = f; }
		void SetDepthBuf ( int db )		{ mDepthBuf = db; }

	public:
		int						mXres, mYres;
		Camera3D*				mCamera;				
		std::vector<Model*>		mModels;
		std::vector<Light*>		mLights;
		std::vector<int>		mShaders;
		std::vector<ParamList>	mParams;
		std::vector<Key>		mKeys;
		std::vector<Mat>		mMaterials;
		int						mFrameSamples;

		bool					clrOverride;
		Vector4DF				clrAmb, clrDiff, clrSpec;	

		std::vector<std::string> mSearchPaths;		
		
		int						mProgram[MAX_PROG];
		int						mProgToSlot[ 512 ];
		
		// Animation recording
		std::string				mOutFile;
		int						mOutFrame;
		Camera3D*				mOutCam;
		Light*					mOutLight;
		std::string				mOutModel;

		// Shadow parameters (independent of method used)
		Vector3DF				mShadowParams;

		// Volume import settings
		Vector3DF				mVClipMin, mVClipMax;
		Vector3DF				mVLeaf;
		Vector3DF				mVFrames;
		std::string				mVName;

		// Transfer function				
		Vector3DF				mTransferVec;			// x=alpha, y=gain
		std::string				mTransferName;			// transfer function filename
		Vector4DF*				mTransferFunc;

		// Volume settings
		Vector3DF				mVThreshold;
		Vector3DF				mExtinct;
		Vector3DF				mSteps; // Direct step, shadow step, fine step
		Vector3DF				mCutoff;
		Vector4DF				mBackgroundClr;

		// Cross sections
		Vector3DF				mSectionPnt;
		Vector3DF				mSectionNorm;

		// Rendering settings
		uchar					mShading;
		uchar					mFilterMode;
		int						mFrame;
		int						mSample;		
		int						mDepthBuf;		
							
		std::string		mLastShader;
	};

	}

#endif

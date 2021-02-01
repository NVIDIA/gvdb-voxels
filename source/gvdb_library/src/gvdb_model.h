//-----------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2016 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
// 
// Version 1.0: Rama Hoetzlein, 5/1/2017
// Version 1.1: Rama Hoetzlein, 3/25/2018
//-----------------------------------------------------------------------------

#ifndef DEF_MODEL_H
	#define DEF_MODEL_H

    #pragma warning (disable : 4251 )

	#include "gvdb_types.h"
	#include "gvdb_vec.h"

	namespace nvdb {

	struct GVDB_API Vert {
		Vector3DF pos;
		Vector3DF norm;
	};
	struct GVDB_API Face {
		Vector3DI vert;
	};

	class GVDB_API Model {
	public:
		Model();
		~Model();

		bool	isPolygonal ()	{ return modelType==0; }		
		bool	isVolume ()		{ return modelType==1; }
		void	Transform ( Vector3DF move, Vector3DF scale );
		void	UniqueNormals ();
		void	ComputeBounds ( Matrix4F& xform, float margin );

		int		getNumVert ()		{ return vertCount; }
		int		getNumElem ()		{ return elemCount; }
		Vert*	getVert ( int n )	{ return ((Vert*) vertBuffer) + n; }
		Face*   getFace ( int n )	{ return ((Face*) elemBuffer) + n; }

		Vector3DF	objMin, objMax;

	public:
		char	modelType;
		
		// Polygonal models
		int		elemDataType;
		int		elemCount;		
		int		elemStride;
		int		elemArrayOffset;
		
		int		vertCount;
		int		vertDataType;
		int		vertComponents;		
		int		vertStride;
		int		vertOffset;

		int		normDataType;
		int		normComponents;
		int		normOffset;

		int		vertArrayID, vertBufferID, elemBufferID;
		
		float*			vertBuffer;
		unsigned int*	elemBuffer;

		Vector4DF		clrAmb;
		Vector4DF		clrDiff;
		Vector4DF		clrSpec;

		// Volumetric models
		std::string		volFile;		// points to either .raw or .vdb
		Vector3DI		volRes;
		char			volType;
	};

	}

#endif
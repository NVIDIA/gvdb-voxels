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
		void	UpdateVBO ();
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
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
// ObjarReader.cpp                                           
// Chris Wyman (9/2/2014)   
//

#ifndef OBJAR_READER
	#define OBJAR_READER

	#include <stdio.h>
	#include <stdlib.h>
	#include <GL/glew.h>

	#include "gvdb_model.h"
	using namespace nvdb;

	// Header for the IGLU library's .objar binary object file
	//    -> Not real important for the purposes of this sample (other than we have to read this to read the model)
	typedef struct {
			unsigned int vboVersionID;       // Magic number / file version header
			unsigned int numVerts;           // Number of vertices in the VBO
			unsigned int numElems;           // Number of elements (i.e., indices in the VBO)
			unsigned int elemType;           // E.g. GL_TRIANGLES
			unsigned int vertStride;         // Number of bytes between subsequent vertices
			unsigned int vertBitfield;       // Binary bitfield describing valid vertex components 
			unsigned int matlOffset;         // In vertex, offset to material ID
			unsigned int objOffset;          // In vertex, offset to object ID
			unsigned int vertOffset;         // In vertex, offset to vertex 
			unsigned int normOffset;         // In vertex, offset to normal 
			unsigned int texOffset;          // In vertex, offset to texture coordinate
			char matlFileName[84];           // Filename containing material information
			float bboxMin[3];                // Minimum point on an axis-aligned bounding box
			float bboxMax[3];                // Maximum point on an axis-aligned bounding box
			char pad[104];                   // padding to be used later!
	} OBJARHeader;


	class OBJARReader {
	public:
		int LoadHeader( FILE *fp, OBJARHeader *hdr );		
		bool LoadFile ( Model* model, char *filename, std::vector<std::string>& paths);
		static bool isMyFile ( const char* filename );		

		friend Model;

	};
#endif


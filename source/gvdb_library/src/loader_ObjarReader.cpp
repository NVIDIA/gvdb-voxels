//-----------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2016 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
// 
// ObjarReader.cpp by Chris Wyman (September 2nd, 2014)
//
// Version 1.0: Rama Hoetzlein, 5/1/2017
// Version 1.1: Rama Hoetzlein, 3/25/2018
//-----------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>

#include "string_helper.h"
#include "loader_Parser.h"
#include "loader_ObjarReader.h"


bool OBJARReader::isMyFile( const char* filename )
{
	return getExtension( filename ) == "objar";
}

int OBJARReader::LoadHeader( FILE *fp, OBJARHeader *hdr )
{
	int fileVersion;

	// Read the type of the file, then reset the file pointer to the start
	fread( &fileVersion, sizeof(unsigned int), 1, fp );
	fseek( fp, 0, SEEK_SET );

	if (fileVersion == 2)  // currently supported file type.
		fread( hdr, sizeof(OBJARHeader), 1, fp );
	else
		return -1;

	return fileVersion;
}


// Do a really simple load of a binary model format into a shadow library usable model format.
bool OBJARReader::LoadFile ( Model* model, char *filename, std::vector<std::string>& paths )
{
	char filePath[512];

	// Open the file.  Check all the specified search directories
	if ( !getFileLocation ( filename, filePath, paths ) ) 
		return false;

	FILE* fp = fopen ( filePath, "rb" );

	// Read the OBJAR header.  If the header isn't version 2, exit now.
	OBJARHeader hdr;
	if ( LoadHeader( fp, &hdr ) != 2 ) 
	{
		fclose( fp );
		return NULL;
	}

	// We require both vertices and normals for our shadow algorithm.  If the model
	//    file does not contain these, exit now.
	if ( (hdr.vertBitfield & 17u) != 17u ) 
	{
		fclose( fp );
		return NULL;
	}

	// Create a new structure to hold our model	
	model->elemDataType     = hdr.elemType;           // What type of GL-renderable primitive does this model contain (e.g., GL_TRIANGLES)
	model->elemCount        = hdr.numElems/3;           // How many of the above primitives are there in the model?
	model->elemArrayOffset  = 0;                      // In the element index array, what's the byte offset to the index of the first element to use?  	
	model->elemStride		= 3*sizeof(unsigned int);

	model->vertCount		= hdr.numVerts;
	model->vertDataType      = GL_FLOAT;               // What is the GL data type of each component in the vertex positions?
	model->vertComponents 	= 3;                      // How many components are in each vertex position attribute?
	model->vertOffset        = hdr.vertOffset;         // What's the byte offset to the start of the first vertex position in the vertex buffer?
	model->vertStride		= hdr.vertStride;         // How many bytes are subsequent verts separated by in the interleaved vertex array created below?

	model->normDataType     = GL_FLOAT;               // What is the GL data type of each component in the vertex normals?
	model->normComponents	= 3;                      // How many components are in each vertex normal attribute?
	model->normOffset       = hdr.normOffset;         // What's the byte offset to the start of the first vertex normal in the vertex buffer?

	// Allocate space for our array of vertex data
	model->vertBuffer = (float*) malloc ( hdr.numVerts * hdr.vertStride );
	void *vertPtr = model->vertBuffer;
	fread ( vertPtr, hdr.numVerts * hdr.vertStride, 1, fp );

	// Allocate space for our array of element data
	model->elemBuffer = (unsigned int*) malloc( hdr.numElems * sizeof( int ) );
	void *elemPtr = model->elemBuffer;
	fread( elemPtr, hdr.numElems * sizeof( int ), 1, fp );

	// *NOTE*: Must still call gvdb.CommitGeometry for GPU buffers

	// We're done! Close file.
	fclose( fp );

	return true;
}


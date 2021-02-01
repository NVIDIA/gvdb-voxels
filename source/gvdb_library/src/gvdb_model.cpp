//-----------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2016 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
// 
// Version 1.0: Rama Hoetzlein, 5/1/2017
// Version 1.1: Rama Hoetzlein, 3/25/2018
//-----------------------------------------------------------------------------

#include "gvdb_model.h"

#include "loader_OBJReader.h"

Model::Model() :
	vertBuffer(0), elemBuffer(0),
	vertArrayID(-1), vertBufferID(-1), elemBufferID(-1)
{
}

Model::~Model ()
{
}

void Model::Transform ( Vector3DF move, Vector3DF scale )
{
	float* pos = (float*) ( (char*) vertBuffer + vertOffset);	

	// Scale object
	for (int n=0; n < vertCount; n++ ) {
		*(pos)	 = *(pos) * scale.x + move.x;
		*(pos+1) = *(pos+1) * scale.y + move.y;
		*(pos+2) = *(pos+2) * scale.z + move.z;
		pos += vertStride/sizeof(float);
	}		
	Matrix4F ident;
	ComputeBounds ( ident, 0 );	
}

void Model::ComputeBounds ( Matrix4F& xform, float margin )
{
	if ( modelType == 1 ) {
		// Volume bounds
		objMin.Set ( 0, 0 ,0 );
		objMax = volRes;
		return;
	}

	// Compute polygon bounds
	Vector3DF pnt;
	float* pos = (float*) ( (char*) vertBuffer + vertOffset);	
	pnt = *(Vector3DF*) pos; pnt *= xform;
	objMin = pnt;
	objMax = pnt;
	for (int n=0; n < vertCount; n++ ) {
		pnt = *(Vector3DF*) pos; pnt *= xform;
		if ( pnt.x < objMin.x ) objMin.x = pnt.x;
		if ( pnt.y < objMin.y ) objMin.y = pnt.y;
		if ( pnt.z < objMin.z ) objMin.z = pnt.z;
		if ( pnt.x > objMax.x ) objMax.x = pnt.x;
		if ( pnt.y > objMax.y ) objMax.y = pnt.y;
		if ( pnt.z > objMax.z ) objMax.z = pnt.z;
		pos += vertStride/sizeof(float);
	}
	// Add margin
	Vector3DF m = objMax; m -= objMin; m *= margin;
	objMin -= m;
	objMax += m;	
}

void Model::UniqueNormals ()
{
	// Build unique triangles to get flat normals
	int v1, v2, v3;
	Vector3DF a, b, c, n;

	// Create new vert/normal buffers
	Vector3DF* vert_buf = (Vector3DF*) malloc ( (9*elemCount) * 2*3*sizeof(float) );	
	Vector3DF* vert_dest = vert_buf;		

	unsigned int* indx_buf = (unsigned int*) malloc ( elemCount * 3*sizeof(unsigned int) );
	unsigned int* indx_dest = indx_buf;
	Vector3DF* vert_src = (Vector3DF*) (vertBuffer + vertOffset);
	int vm = vertStride / sizeof(Vector3DF);		// stride as multiple of Vec3F

	for (int n=0; n < elemCount; n++ ) {

		// Get vertices 
		v1 = elemBuffer[ n*3 ]; v2 = elemBuffer[ n*3+1 ]; v3 = elemBuffer[ n*3+2 ];		

		// Compute face normal
		a = vert_src[v1*vm]; b = vert_src[v2*vm];	c = vert_src[v3*vm];		
		a -= c;	b -= c; a.Cross ( b );
		a.Normalize ();
		
		// Output vertices and normals
		*vert_dest++ = vert_src[v1*vm];		*vert_dest++ = a;
		*vert_dest++ = vert_src[v2*vm];		*vert_dest++ = a;
		*vert_dest++ = vert_src[v3*vm];		*vert_dest++ = a;		

		// Output new indices
		*indx_dest++ = v1;
		*indx_dest++ = v2;
		*indx_dest++ = v3;
	}

	// Update model data
	free ( vertBuffer );
	free ( elemBuffer );
	
	vertBuffer = (float*) vert_buf;	
	vertStride = 2*3*sizeof(float);
	vertCount = elemCount*3;
	vertComponents = 6;

	elemBuffer = indx_buf;
	elemStride = 3*sizeof(unsigned int);
}

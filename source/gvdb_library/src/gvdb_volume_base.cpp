//-----------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2016 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
// 
// Version 1.0: Rama Hoetzlein, 5/1/2017
// Version 1.1: Rama Hoetzlein, 3/25/2018
//-----------------------------------------------------------------------------

#include "gvdb_volume_base.h"
#include "gvdb_render.h"
#include "gvdb_allocator.h"
#include "gvdb_scene.h"

using namespace nvdb;

VolumeBase::~VolumeBase() {
	// Delete pool
	if (mPool != nullptr) {
		delete mPool;
		mPool = nullptr;
	}
}

void VolumeBase::getDimensions ( Vector3DF& objmin, Vector3DF& objmax, Vector3DF& voxmin, Vector3DF& voxmax, Vector3DF& voxres )
{
	objmin = mObjMin;
	objmax = mObjMax;
	voxmin = mVoxMin;
	voxmax = mVoxMax;
	voxres = mVoxRes;
}

void VolumeBase::getTiming ( float& render_time )
{
	render_time = mRenderTime.x;
}

void VolumeBase::ClearGeometry ( Model* m )
{
	if ( m->vertArrayID != -1 ) glDeleteVertexArrays ( 1,  (GLuint*) &m->vertArrayID );
	if ( m->vertBufferID != -1 ) glDeleteBuffers ( 1,  (GLuint*) &m->vertBufferID );
	if ( m->elemBufferID != -1 ) glDeleteBuffers ( 1,  (GLuint*) &m->elemBufferID );
}

void VolumeBase::CommitGeometry ( int model_id )
{
	Model* m = mScene->getModel( model_id );
	CommitGeometry ( m );
}

void VolumeBase::CommitGeometry ( Model* m )
{
	#ifdef BUILD_OPENGL
		// Create VAO
		if ( m->vertArrayID == -1 )  glGenVertexArrays ( 1, (GLuint*) &m->vertArrayID );
		glBindVertexArray ( m->vertArrayID );

		// Update Vertex VBO
		if ( m->vertBufferID == -1 ) glGenBuffers( 1, (GLuint*) &m->vertBufferID );	
		
		glNamedBufferDataEXT( m->vertBufferID, m->vertCount * m->vertStride, m->vertBuffer, GL_STATIC_DRAW );
		glEnableVertexAttribArray ( 0 );
		glBindVertexBuffer ( 0, m->vertBufferID, 0, m->vertStride );
		glVertexAttribFormat ( 0, m->vertComponents, GL_FLOAT, false, m->vertOffset );
		glVertexAttribBinding ( 0, 0 );
		glEnableVertexAttribArray ( 1 );
		glVertexAttribFormat ( 1, m->normComponents, GL_FLOAT, false, m->normOffset );
		glVertexAttribBinding ( 1, 0 );
	
		// Update Element VBO
		if ( m->elemBufferID == -1 ) glGenBuffers( 1, (GLuint*) &m->elemBufferID );
		glNamedBufferDataEXT( m->elemBufferID, m->elemCount * m->elemStride, m->elemBuffer, GL_STATIC_DRAW );	

		glBindVertexArray ( 0 );
	
		glBindBuffer ( GL_ARRAY_BUFFER, 0 );
		glBindBuffer ( GL_ELEMENT_ARRAY_BUFFER, 0 );

	#endif
}
	
	

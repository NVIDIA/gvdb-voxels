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
	
	

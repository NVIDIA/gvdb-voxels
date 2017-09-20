//--------------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2017, NVIDIA Corporation. 
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
//----------------------------------------------------------------------------------
#include "gvdb_render.h"

int GLS_SIMPLE = -1;	// GL Shader Programs
int GLS_OUTLINE = -1;
int GLS_SLICE = -1;
int GLS_VOXELIZE = -1;
int GLS_RAYCAST = -1;
int GLS_INSTANCE = -1;
int GLS_SCREENTEX = -1;

#ifdef BUILD_OPENGL


	int getShaderID (int i)
	{
		switch (i) {
		case 0:	return GLS_SIMPLE;
		case 1: return GLS_OUTLINE;
		case 2: return GLS_SLICE;
		case 3: return GLS_VOXELIZE;
		case 4: return GLS_RAYCAST;
		case 5: return GLS_INSTANCE;
		case 6: return GLS_SCREENTEX;
		};
	}

	void checkGL( char* msg )
	{
		GLenum errCode;
		//const GLubyte* errString;
		errCode = glGetError();
		if (errCode != GL_NO_ERROR) {
			//printf ( "%s, ERROR: %s\n", msg, gluErrorString(errCode) );
			gprintf ("%s, ERROR: 0x%x\n", msg, errCode );
		}
	}

	void makeSimpleShader ( Scene* scene, char* vertname, char* fragname )
	{
		GLS_SIMPLE = scene->AddShader ( vertname, fragname );
		scene->AddParam ( GLS_SIMPLE, UVIEW, "uView" );
		scene->AddParam ( GLS_SIMPLE, UPROJ, "uProj" );
		scene->AddParam ( GLS_SIMPLE, UMODEL, "uModel" );	
		scene->AddParam ( GLS_SIMPLE, ULIGHTPOS, "uLightPos" );
		scene->AddParam ( GLS_SIMPLE, UCLRAMB, "uClrAmb" );
		scene->AddParam ( GLS_SIMPLE, UCLRDIFF, "uClrDiff" );		
	}
	void makeOutlineShader ( Scene* scene, char* vertname, char* fragname )
	{
		GLS_OUTLINE = scene->AddShader ( vertname, fragname );
		scene->AddParam ( GLS_OUTLINE, UVIEW, "uView" );
		scene->AddParam ( GLS_OUTLINE, UPROJ, "uProj" );
		scene->AddParam ( GLS_OUTLINE, UMODEL, "uModel" );	
		scene->AddParam ( GLS_OUTLINE, UCLRAMB, "uClrAmb" );
	}

	void makeSliceShader ( Scene* scene, char* vertname, char* fragname )
	{
		GLS_SLICE = scene->AddShader ( vertname, fragname );
		scene->AddParam ( GLS_SLICE, UVIEW, "uView" );
		scene->AddParam ( GLS_SLICE, UPROJ, "uProj" );
		scene->AddParam ( GLS_SLICE, ULIGHTPOS, "uLightPos" );
		scene->AddParam ( GLS_SLICE, UTEX, "uTex" );	
	}
	void makeVoxelizeShader ( Scene* scene, char* vertname, char* fragname, char* geomname )
	{
		GLS_VOXELIZE = scene->AddShader ( vertname, fragname, geomname );
		scene->AddParam ( GLS_VOXELIZE, UW, "uW" );
		scene->AddParam ( GLS_VOXELIZE, UTEXRES, "uTexRes" );
		scene->AddParam ( GLS_VOXELIZE, USAMPLES, "uSamples" );
	}
	void makeRaycastShader ( Scene* scene, char* vertname, char* fragname )
	{
		GLS_RAYCAST = scene->AddShader ( vertname, fragname );	
		scene->AddParam ( GLS_RAYCAST, UINVVIEW, "uInvView" );
		scene->AddParam ( GLS_RAYCAST, UCAMPOS,  "uCamPos" );
		scene->AddParam ( GLS_RAYCAST, UCAMDIMS, "uCamDims" );
		scene->AddParam ( GLS_RAYCAST, UVOLMIN,  "uVolMin" );
		scene->AddParam ( GLS_RAYCAST, UVOLMAX,  "uVolMax" );
		scene->AddParam ( GLS_RAYCAST, UTEX,     "uTex" );
		scene->AddParam ( GLS_RAYCAST, UTEXRES,  "uTexRes" );
		scene->AddParam ( GLS_RAYCAST, ULIGHTPOS, "uLightPos" );
		//scene->AddParam ( GLS_RAYCAST, USAMPLES, "uSamples" );
	}
	void makeInstanceShader ( Scene* scene, char* vertname, char* fragname )
	{
		GLS_INSTANCE = scene->AddShader ( vertname, fragname );	
		scene->AddParam ( GLS_INSTANCE, UVIEW, "uView" );
		scene->AddParam ( GLS_INSTANCE, UPROJ, "uProj" );	
	
		/*scene->AddAttrib ( GLS_INSTANCE, UVOLMIN, "instVMin" );		// instance attributes
		scene->AddAttrib ( GLS_INSTANCE, UVOLMAX, "instVMax" );
		scene->AddAttrib ( GLS_INSTANCE, UTEXMIN, "instTMin" );
		scene->AddAttrib ( GLS_INSTANCE, UTEXMAX, "instTMax" );
		scene->AddAttrib ( GLS_INSTANCE, UCLRAMB, "instClr" );*/
	}
	void makeScreenShader ( Scene* scene, char* vertname, char* fragname )
	{
		GLS_SCREENTEX = scene->AddShader ( vertname, fragname );	
		scene->AddParam ( GLS_SCREENTEX, UTEX,     "uTex" );	
	}

	void renderCamSetupGL ( Scene* scene, int prog, Matrix4F* model )
	{
		int attr_model = scene->getParam(prog, UMODEL);
		int attr_cam = scene->getParam(prog, UCAMPOS);

		// Set model, view, projection matrices
		if ( attr_model != -1 ) {		
			if ( model == 0x0 ) {
				Matrix4F ident; ident.Identity ();
				glProgramUniformMatrix4fv( prog, attr_model, 1, GL_FALSE, ident.GetDataF() );
			} else  {
				glProgramUniformMatrix4fv( prog, attr_model, 1, GL_FALSE, model->GetDataF() );
			}
		}
		Camera3D* cam = scene->getCamera ();		
		glProgramUniformMatrix4fv( prog, scene->getParam(prog, UVIEW), 1, GL_FALSE, cam->getViewMatrix().GetDataF() ); 
		glProgramUniformMatrix4fv( prog, scene->getParam(prog, UPROJ), 1, GL_FALSE, cam->getProjMatrix().GetDataF() );
		if ( attr_cam != -1 ) glProgramUniform3fv ( prog, attr_cam, 1, &cam->getPos().x );	
	}

	void renderLightSetupGL ( Scene* scene, int prog ) 
	{
		Light* light = scene->getLight ();	
		glProgramUniform3fv  ( prog, scene->getParam(prog, ULIGHTPOS), 1, &light->getPos().x );	
	}
	void renderSetMaterialGL ( Scene* scene, int prog, Vector4DF amb, Vector4DF diff, Vector4DF spec )
	{
		glProgramUniform4fv  ( prog, scene->getParam(prog, UCLRAMB),  1, &amb.x );
		glProgramUniform4fv  ( prog, scene->getParam(prog, UCLRDIFF), 1, &diff.x );
		glProgramUniform4fv  ( prog, scene->getParam(prog, UCLRSPEC), 1, &spec.x );
	}
	void renderSetTex3D ( Scene* scene, int prog, int tex, Vector3DF res )
	{
		if ( scene->getParam(prog, UTEXRES) != -1 )
			glProgramUniform3fv ( prog, scene->getParam(prog, UTEXRES), 1, &res.x );

		glProgramUniform1i ( prog, scene->getParam(prog, UTEX), 0 );	
		glActiveTexture ( GL_TEXTURE0 );
		glBindTexture ( GL_TEXTURE_3D, tex );
	}
	void renderSetTex2D ( Scene* scene, int prog, int tex )
	{
		glProgramUniform1i ( prog, scene->getParam(prog, UTEX), 0 );	
		glActiveTexture ( GL_TEXTURE0 );
		glBindTexture ( GL_TEXTURE_2D, tex );
	}
	void renderSetUW ( Scene* scene, int prog, Matrix4F* model, Vector3DF res )
	{
		glProgramUniformMatrix4fv( prog, scene->getParam(prog, UW ), 1, GL_FALSE, model->GetDataF() );
		glProgramUniform3fv ( prog, scene->getParam(prog, UTEXRES ), 1, &res.x );
	}

	void renderSceneGL ( Scene* scene, int prog )
	{
		glEnable ( GL_CULL_FACE );
		glEnable ( GL_DEPTH_TEST );		
		renderSceneGL ( scene, prog, true );
	}

	void renderSceneGL ( Scene* scene, int prog, bool bMat )
	{	
		// Render each model
		Model* model;
		if ( scene->useOverride() && bMat )
			renderSetMaterialGL ( scene, prog, scene->clrAmb, scene->clrDiff, scene->clrSpec );

		for (int n = 0; n < scene->getNumModels(); n++ ) {
			model = scene->getModel( n );		
			if ( !scene->useOverride() && bMat ) renderSetMaterialGL ( scene, prog, model->clrAmb, model->clrDiff, model->clrSpec );
			glBindVertexArray ( model->vertArrayID );
			glBindBuffer ( GL_ELEMENT_ARRAY_BUFFER, model->elemBufferID );
			glDrawElements ( GL_TRIANGLES, model->elemCount * 3, GL_UNSIGNED_INT, 0 );
		}
		glBindBuffer ( GL_ELEMENT_ARRAY_BUFFER, 0 );
		glBindVertexArray ( 0 );
		glDisable ( GL_DEPTH_TEST );
		glDisable ( GL_CULL_FACE );	

		// OpenGL 3.3
		/* glEnableClientState ( GL_VERTEX_ARRAY );
		glBindBuffer ( GL_ARRAY_BUFFER, model->vertBufferID );	
		glVertexPointer ( model->vertComponents, GL_FLOAT, model->vertStride, (char*) model->vertOffset );
		glNormalPointer ( GL_FLOAT, model->vertStride, (char*) model->normOffset );
		glBindBuffer ( GL_ELEMENT_ARRAY_BUFFER, model->elemBufferID );	 
		glDrawElements ( model->elemDataType, model->elemCount*3, GL_UNSIGNED_INT, 0 );	*/
	}

	void renderScreenspaceGL ( Scene* scene, int prog )
	{
		Camera3D* cam = scene->getCamera ();		

		if ( scene->getParam(prog,UINVVIEW) != -1 ) 
			glProgramUniformMatrix4fv( prog, scene->getParam(prog, UINVVIEW ), 1, GL_FALSE, cam->getInvView().GetDataF() );	
	
		if ( scene->getParam(prog,UCAMPOS) != -1 ) {
			glProgramUniform3fv ( prog, scene->getParam(prog, UCAMPOS), 1, &cam->getPos().x );	
			Vector3DF cd;
			cd.x = tan ( cam->getFov() * 0.5 * 3.141592/180.0f );
			cd.y = cd.x / cam->getAspect();
			cd.z = cam->getNear();
			glProgramUniform3fv	( prog, scene->getParam(prog, UCAMDIMS), 1, &cd.x );	
		}

		//scene->getScreenquad().SelectVBO ();
		//scene->getScreenquad().Draw(1);
	}

#endif
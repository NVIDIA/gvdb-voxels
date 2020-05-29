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
#include "gvdb_render.h"


#ifdef BUILD_OPENGL

	void gchkGL ( const char* msg )
	{
		GLenum errCode;
		//const GLubyte* errString;
		errCode = glGetError();
		if (errCode != GL_NO_ERROR) {
			//printf ( "%s, ERROR: %s\n", msg, gluErrorString(errCode) );
			gprintf ("%s, ERROR: 0x%x\n", msg, errCode );
		}
	}

	void makeSimpleShaderGL ( Scene* scene, const char* vertname, const char* fragname )
	{
		scene->AddShader ( GLS_SIMPLE, vertname, fragname );
		scene->AddParam ( GLS_SIMPLE, UVIEW, "uView" );
		scene->AddParam ( GLS_SIMPLE, UPROJ, "uProj" );
		scene->AddParam ( GLS_SIMPLE, UMODEL, "uModel" );	
		scene->AddParam ( GLS_SIMPLE, ULIGHTPOS, "uLightPos" );
		scene->AddParam ( GLS_SIMPLE, UCLRAMB, "uClrAmb" );
		scene->AddParam ( GLS_SIMPLE, UCLRDIFF, "uClrDiff" );		
	}
	void makeOutlineShader ( Scene* scene, const char* vertname, const char* fragname )
	{
		scene->AddShader (GLS_OUTLINE, vertname, fragname );
		scene->AddParam ( GLS_OUTLINE, UVIEW, "uView" );
		scene->AddParam ( GLS_OUTLINE, UPROJ, "uProj" );
		scene->AddParam ( GLS_OUTLINE, UMODEL, "uModel" );	
		scene->AddParam ( GLS_OUTLINE, UCLRAMB, "uClrAmb" );
	}

	void makeSliceShader ( Scene* scene, const char* vertname, const char* fragname )
	{
		scene->AddShader ( GLS_SLICE, vertname, fragname );
		scene->AddParam ( GLS_SLICE, UVIEW, "uView" );
		scene->AddParam ( GLS_SLICE, UPROJ, "uProj" );
		scene->AddParam ( GLS_SLICE, ULIGHTPOS, "uLightPos" );
		scene->AddParam ( GLS_SLICE, UTEX, "uTex" );	
	}
	void makeVoxelizeShader ( Scene* scene, const char* vertname, const char* fragname, const char* geomname )
	{
		scene->AddShader ( GLS_VOXELIZE, vertname, fragname, geomname );
		scene->AddParam ( GLS_VOXELIZE, UW, "uW" );
		scene->AddParam ( GLS_VOXELIZE, UTEXRES, "uTexRes" );
		scene->AddParam ( GLS_VOXELIZE, USAMPLES, "uSamples" );
	}
	void makeRaycastShader ( Scene* scene, const char* vertname, const char* fragname )
	{
		scene->AddShader ( GLS_RAYCAST, vertname, fragname );
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
	void makeInstanceShader ( Scene* scene, const char* vertname, const char* fragname )
	{
		scene->AddShader ( GLS_INSTANCE, vertname, fragname );
		scene->AddParam ( GLS_INSTANCE, UVIEW, "uView" );
		scene->AddParam ( GLS_INSTANCE, UPROJ, "uProj" );	
	}
	void makeScreenShader ( Scene* scene, const char* vertname, const char* fragname )
	{
		scene->AddShader ( GLS_SCREENTEX, vertname, fragname );
		scene->AddParam ( GLS_SCREENTEX, UTEX,     "uTex" );	
	}

	
	void renderCamSetupGL ( Scene* scene, int prog_id, Matrix4F* model )
	{
		int prog = scene->getProgram(prog_id);
		int attr_model = scene->getParam(prog_id, UMODEL);
		int attr_cam = scene->getParam(prog_id, UCAMPOS);

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
		glProgramUniformMatrix4fv( prog, scene->getParam(prog_id, UVIEW), 1, GL_FALSE, cam->getViewMatrix().GetDataF() );
		glProgramUniformMatrix4fv( prog, scene->getParam(prog_id, UPROJ), 1, GL_FALSE, cam->getProjMatrix().GetDataF() );
		if ( attr_cam != -1 ) glProgramUniform3fv ( prog, attr_cam, 1, &cam->getPos().x );	
	}

    void renderLightSetupGL ( Scene* scene, int prog_id ) 
	{
		Light* light = scene->getLight ();	
		glProgramUniform3fv  (scene->getProgram(prog_id), scene->getParam(prog_id, ULIGHTPOS), 1, &light->getPos().x );
	}
	void renderSetMaterialGL(Scene* scene, int prog_id, Vector4DF amb, Vector4DF diff, Vector4DF spec)
	{
		int prog = scene->getProgram(prog_id);
		glProgramUniform4fv(prog, scene->getParam(prog_id, UCLRAMB), 1, &amb.x);
		glProgramUniform4fv(prog, scene->getParam(prog_id, UCLRDIFF), 1, &diff.x);
		glProgramUniform4fv(prog, scene->getParam(prog_id, UCLRSPEC), 1, &spec.x);
	}
	void renderSetTex3D ( Scene* scene, int prog_id, int tex, Vector3DF res )
	{
		int prog = scene->getProgram(prog_id);
		if ( scene->getParam(prog_id, UTEXRES) != -1 )
			glProgramUniform3fv ( prog, scene->getParam(prog_id, UTEXRES), 1, &res.x );

		glProgramUniform1i ( prog, scene->getParam(prog_id, UTEX), 0 );	
		glActiveTexture ( GL_TEXTURE0 );
		glBindTexture ( GL_TEXTURE_3D, tex );
	}
	void renderSetTex2D ( Scene* scene, int prog_id, int tex )
	{
		glProgramUniform1i (scene->getProgram(prog_id), scene->getParam(prog_id, UTEX), 0 );
		glActiveTexture ( GL_TEXTURE0 );
		glBindTexture ( GL_TEXTURE_2D, tex );
	}
	void renderSetUW ( Scene* scene, int prog_id, Matrix4F* model, Vector3DF res )
	{
		glProgramUniformMatrix4fv( scene->getProgram(prog_id), scene->getParam(prog_id, UW ), 1, GL_FALSE, model->GetDataF() );
		glProgramUniform3fv ( scene->getProgram(prog_id), scene->getParam(prog_id, UTEXRES ), 1, &res.x );
	}

	void renderSceneGL ( Scene* scene, int prog_id )
	{
		glEnable ( GL_CULL_FACE );
		glEnable ( GL_DEPTH_TEST );		
		renderSceneGL ( scene, prog_id, true );
	}

	void renderSceneGL ( Scene* scene, int prog_id, bool bMat )
	{	
		// Render each model
		Model* model;
		if ( scene->useOverride() && bMat )
			renderSetMaterialGL ( scene, prog_id, scene->clrAmb, scene->clrDiff, scene->clrSpec );

		for (int n = 0; n < scene->getNumModels(); n++ ) {
			model = scene->getModel( n );		
			if ( !scene->useOverride() && bMat ) 
				renderSetMaterialGL ( scene, prog_id, model->clrAmb, model->clrDiff, model->clrSpec );
			
			glBindVertexArray ( model->vertArrayID );
			glBindBuffer ( GL_ELEMENT_ARRAY_BUFFER, model->elemBufferID );
			glDrawElements ( GL_TRIANGLES, model->elemCount * 3, GL_UNSIGNED_INT, 0 );
		}
		glBindBuffer ( GL_ELEMENT_ARRAY_BUFFER, 0 );
		glBindVertexArray ( 0 );
		glDisable ( GL_DEPTH_TEST );
		glDisable ( GL_CULL_FACE );	

	}

	void renderScreenspaceGL ( Scene* scene, int prog_id )
	{
		int prog = scene->getProgram(prog_id);
		Camera3D* cam = scene->getCamera ();		

		if ( scene->getParam(prog_id, UINVVIEW) != -1 ) 
			glProgramUniformMatrix4fv( prog, scene->getParam(prog_id, UINVVIEW ), 1, GL_FALSE, cam->getInvView().GetDataF() );	
	
		if ( scene->getParam(prog_id, UCAMPOS) != -1 ) {
			glProgramUniform3fv ( prog, scene->getParam(prog_id, UCAMPOS), 1, &cam->getPos().x );	
			Vector3DF cd;
			cd.x = tanf( cam->getFov() * 0.5f * 3.141592f/180.0f );
			cd.y = cd.x / cam->getAspect();
			cd.z = cam->getNear();
			glProgramUniform3fv	( prog, scene->getParam(prog_id, UCAMDIMS), 1, &cd.x );	
		}

	}

#endif
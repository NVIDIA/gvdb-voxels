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

#include "gvdb_vec.h"
#include "gvdb_scene.h"
using namespace nvdb;

#define UVIEW		0
#define UPROJ		1
#define UMODEL		2
#define UINVVIEW	3
#define ULIGHTPOS	4
#define UCAMPOS		5
#define UCAMDIMS	6
#define UCLRAMB		7
#define UCLRDIFF	8
#define UCLRSPEC	9
#define UTEX		10
#define UTEXRES		11
#define UW			12
#define USHADOWMASK 13
#define USHADOWSIZE	14
#define UEYELIGHT	15
#define UOVERTEX	16
#define UOVERSIZE	17
#define UVOLMIN		18
#define UVOLMAX		19
#define USAMPLES	20
#define UTEXMIN		21
#define UTEXMAX		22

// OpenGL Render
#ifdef BUILD_OPENGL

	#include <GL/glew.h>			// OpenGL extensions
	#include <cudaGL.h>				// Cuda-GL interop
	#include <cuda_gl_interop.h>
	
	extern int GLS_SIMPLE;
	extern int GLS_OUTLINE;
	extern int GLS_SLICE;
	extern int GLS_VOXELIZE;
	extern int GLS_RAYCAST;
	extern int GLS_INSTANCE;
	extern int GLS_SCREENTEX;

	void checkGL( char* msg );

	void renderAddShaderGL ( Scene* scene, char* vertfile, char* fragfile );
	void renderCamSetupGL ( Scene* scene, int prog, Matrix4F* model );
	void renderLightSetupGL ( Scene* scene, int prog );
	void renderSceneGL ( Scene* scene, int prog );
	void renderSceneGL ( Scene* scene, int prog, bool bMat );
	void renderSetTex3D ( Scene* scene, int prog, int tex, Vector3DF res );
	void renderSetTex2D ( Scene* scene, int prog, int tex );
	void renderSetUW ( Scene* scene, int prog, Matrix4F* model, Vector3DF res );
	void renderScreenspaceGL ( Scene* scene, int prog );

	void makeSliceShader ( Scene* scene, char* vertname, char* fragname );
	void makeOutlineShader ( Scene* scene, char* vertname, char* fragname );
	void makeVoxelizeShader ( Scene* scene, char* vertname, char* fragname, char* geomname );
	void makeRaycastShader ( Scene* scene, char* vertname, char* fragname );
	void makeInstanceShader ( Scene* scene, char* vertname, char* fragname );
	void makeScreenShader ( Scene* scene, char* vertname, char* fragname );

#endif

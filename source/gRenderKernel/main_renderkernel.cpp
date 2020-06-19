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

#include "gvdb.h"
#include "file_png.h"

#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>

CUmodule		cuCustom;
CUfunction		cuRaycastKernel;

bool cudaCheck ( CUresult status, char* msg )
{
	if ( status != CUDA_SUCCESS ) {
		const char* stat = "";
		cuGetErrorString ( status, &stat );
		printf ( "CUDA ERROR: %s (in %s)\n", stat, msg  );	
		exit(-1);
		return false;
	} 
	return true;
}

int main (int argc, char* argv)
{
	int w = 1024, h = 768;

	VolumeGVDB gvdb;

	// Initialize GVDB
	printf ( "Starting GVDB.\n" );	
	int devid = -1;
	gvdb.SetCudaDevice ( devid );
	gvdb.Initialize ();				
	gvdb.AddPath ( "../source/shared_assets/" );
	gvdb.AddPath ( "../shared_assets/" );
	gvdb.AddPath ( ASSET_PATH );

	// Load VBX
	char scnpath[1024];
	if ( !gvdb.FindFile ( "explosion.vbx", scnpath ) ) {
		printf ( "Cannot find vbx files.\n" );
		exit(-1);
	}
	printf ( "Loading VBX. %s\n", scnpath );
	gvdb.LoadVBX ( scnpath );	
	
	gvdb.getScene()->SetVolumeRange ( 0.01f, 1.0f, 0.0f );

	// Create Camera and Light
	Camera3D* cam = new Camera3D;						
	cam->setOrbit ( Vector3DF(20,30,0), Vector3DF(125,160,125), 800, 1.0f );		
	gvdb.getScene()->SetCamera( cam );	
	gvdb.getScene()->SetRes ( w, h );
	
	Light* lgt = new Light;	
	lgt->setOrbit ( Vector3DF(50,65,0), Vector3DF(125,140,125), 200, 1.0f );
	gvdb.getScene()->SetLight ( 0, lgt );		
	
	// Add render buffer 
	printf ( "Creating screen buffer. %d x %d\n", w, h );
	gvdb.AddRenderBuf ( 0, w, h, 4 );					

	// Load custom module and kernel
	printf ( "Loading module: render_custom.ptx\n");
	cudaCheck ( cuModuleLoad ( &cuCustom, "render_custom.ptx" ), "cuModuleLoad (render_custom)" );
	cudaCheck ( cuModuleGetFunction ( &cuRaycastKernel, cuCustom, "raycast_kernel" ), "cuModuleGetFunction (raycast_kernel)" );	

	// Set GVDB to custom module 
	gvdb.SetModule ( cuCustom );

	// Render with user-defined kernel
	printf ( "Render custom kernel.\n" );
	gvdb.getScene()->SetSteps ( 0.25f, 16, 0.25f );
	gvdb.getScene()->SetVolumeRange ( 0.1f, 0.0, 1.0f );
	gvdb.RenderKernel ( cuRaycastKernel, 0, 0 );	

	// Read render buffer
	unsigned char* buf = (unsigned char*) malloc ( w*h*4 );
	gvdb.ReadRenderBuf ( 0, buf );						

	// Save as png
	printf ( "Saving out_rendkernel.png\n");
	save_png ( "out_rendkernel.png", buf, w, h, 4 );

	free ( buf );

	printf ( "Done.\n" );

 	return 1;
}

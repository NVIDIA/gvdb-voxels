//-----------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2017 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
// 
// Version 1.0: Rama Hoetzlein, 5/1/2017
//-----------------------------------------------------------------------------

#include "gvdb.h"
using namespace nvdb;

#include <stdlib.h>
#include <stdio.h>

#include "file_png.h"		// sample utils

int main (int argc, char* argv)
{
	int w = 1024, h = 768;

	VolumeGVDB gvdb;

	// Initialize GVDB
	printf ( "Starting GVDB.\n" );	
	gvdb.SetVerbose ( true );		// enable/disable console output from gvdb
	gvdb.SetCudaDevice ( GVDB_DEV_FIRST );
	gvdb.Initialize ();			
	gvdb.AddPath ( "../source/shared_assets/" );
	gvdb.AddPath ( "../shared_assets/" );
	gvdb.AddPath ( ASSET_PATH );


	// Load VBX
	char scnpath[1024];		
	if ( !gvdb.FindFile ( "explosion.vbx", scnpath ) ) {
		printf ( "Cannot find vbx file.\n" );
		exit (-1);
	}
	printf ( "Loading VBX. %s\n", scnpath );
	gvdb.LoadVBX ( scnpath );							// Load VBX

	// Set volume params
	gvdb.getScene()->SetSteps ( .25f, 16, .25f );			// Set raycasting steps
	gvdb.getScene()->SetExtinct ( -1.0f, 1.5f, 0.0f );		// Set volume extinction
	gvdb.getScene()->SetVolumeRange ( 0.1f, 0.0f, .1f );	// Set volume value range
	gvdb.getScene()->SetCutoff ( 0.005f, 0.01f, 0.0f );
	gvdb.getScene()->SetBackgroundClr ( 0.1f, 0.2f, 0.4f, 1.0 );
	gvdb.getScene()->LinearTransferFunc ( 0.00f, 0.25f, Vector4DF(0,0,0,0), Vector4DF(1,1,0,0.1f) );
	gvdb.getScene()->LinearTransferFunc ( 0.25f, 0.50f, Vector4DF(1,1,0,0.4f), Vector4DF(1,0,0,0.3f) );
	gvdb.getScene()->LinearTransferFunc ( 0.50f, 0.75f, Vector4DF(1,0,0,0.3f), Vector4DF(.2f,.2f,0.2f,0.1f) );
	gvdb.getScene()->LinearTransferFunc ( 0.75f, 1.00f, Vector4DF(.2f,.2f,0.2f,0.1f), Vector4DF(0,0,0,0.0) );
	gvdb.CommitTransferFunc ();

	Camera3D* cam = new Camera3D;						// Create Camera 
	cam->setFov ( 50.0 );
	cam->setOrbit ( Vector3DF(20,30,0), Vector3DF(125,160,125), 500, 1.0f );	
	gvdb.getScene()->SetCamera( cam );	
	gvdb.getScene()->SetRes ( w, h );
	
	Light* lgt = new Light;								// Create Light
	lgt->setOrbit ( Vector3DF(299,57.3f,0), Vector3DF(132,-20,50), 200, 1.0f );
	gvdb.getScene()->SetLight ( 0, lgt );		
	
	printf ( "Creating screen buffer. %d x %d\n", w, h );
	gvdb.AddRenderBuf ( 0, w, h, 4 );					// Add render buffer 

	gvdb.TimerStart ();
	gvdb.Render ( SHADE_VOLUME, 0, 0 );					// Render as volume (in channel 0, out buffer 0)
	float rtime = gvdb.TimerStop();
	printf ( "Render volume. %6.3f ms\n", rtime );

	printf ( "Writing out_rendfile.png\n" );
	unsigned char* buf = (unsigned char*) malloc ( w*h*4 );
	gvdb.ReadRenderBuf ( 0, buf );						// Read render buffer

	save_png ( "out_rendtofile.png", buf, w, h, 4 );				// Save as png

	free ( buf );

	printf ( "Done.\n" );	

    return 1;
}

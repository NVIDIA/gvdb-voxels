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

VolumeGVDB gvdb;

int main (int argc, char* argv)
{
	int w = 1024, h = 768;

	// Initialize GVDB
	printf ( "Starting GVDB.\n" );		
#ifdef _DEBUG
	gvdb.SetDebug(true);
#endif // #ifdef _DEBUG
	gvdb.SetVerbose ( true );		// enable/disable console output from gvdb
	gvdb.SetCudaDevice ( GVDB_DEV_FIRST );
	gvdb.Initialize ();			
	gvdb.AddPath ( "../source/shared_assets/" );
	gvdb.AddPath ( ASSET_PATH );

	// Load VDB
	char scnpath[1024];		
	if ( !gvdb.getScene()->FindFile ( "bunny.vdb", scnpath ) ) {
		gprintf ( "Cannot find vdb file.\n" );
		gerror();
	}
	printf ( "Loading VDB. %s\n", scnpath );
	gvdb.SetChannelDefault ( 16, 16, 1 );
	if ( !gvdb.LoadVDB ( scnpath ) ) {                	// Load OpenVDB format			
		gerror();
	}	

	gvdb.SaveVBX ( "bunny.vbx" );						// Save VBX format

	// At this point, data was converted from vdb to vbx format.
	// The vbx file can now be used by gvdb (e.g. render using InteractiveGL)
	
	// Remainder of this sample demonstrates level set rendering to file.

	// Set volume params
	gvdb.getScene()->SetSteps ( 0.25f, 16, 0.25f );			// Set raycasting steps per voxel
	gvdb.getScene()->SetExtinct ( -1.0f, 1.5f, 0.0f );		// Set volume extinction
	gvdb.getScene()->SetVolumeRange ( 0.0f, 1.0f, -1.0f );	// Set volume value range (for a level set)
	gvdb.getScene()->SetCutoff ( 0.005f, 0.01f, 0.0f );
	gvdb.getScene()->LinearTransferFunc ( 0.00f, 0.25f, Vector4DF(1,1,0,0.05f), Vector4DF(1,1,0,0.03f) );
	gvdb.getScene()->LinearTransferFunc ( 0.25f, 0.50f, Vector4DF(1,1,1,0.03f), Vector4DF(1,0,0,0.02f) );
	gvdb.getScene()->LinearTransferFunc ( 0.50f, 0.75f, Vector4DF(1,0,0,0.02f), Vector4DF(1,.5f,0,0.01f) );
	gvdb.getScene()->LinearTransferFunc ( 0.75f, 1.00f, Vector4DF(1,.5f,0,0.01f), Vector4DF(0,0,0,0.005f) );
	gvdb.getScene()->SetBackgroundClr ( 0, 0, 0, 1 );
	gvdb.SetEpsilon(0.01f, 256);
	gvdb.CommitTransferFunc ();


	Camera3D* cam = new Camera3D;						// Create Camera 
	cam->setFov ( 30.0 );
	cam->setOrbit ( Vector3DF(-10,30,0), Vector3DF(14.2f,15.3f,18.0f), 130, 1.0f );	
	gvdb.getScene()->SetCamera( cam );	
	gvdb.getScene()->SetRes ( w, h );
	
	Light* lgt = new Light;								// Create Light
	lgt->setOrbit ( Vector3DF(30,50,0), Vector3DF(15,15,15), 200, 1.0 );
	gvdb.getScene()->SetLight ( 0, lgt );		
	
	printf ( "Creating screen buffer. %d x %d\n", w, h );
	gvdb.AddRenderBuf ( 0, w, h, 4 );					// Add render buffer 

	gvdb.TimerStart ();
	gvdb.Render ( SHADE_LEVELSET, 0, 0 );			// Render as volume
	float rtime = gvdb.TimerStop();
	printf ( "Render volume. %6.3f ms\n", rtime );

	printf ( "Writing img_importvdb.png\n" );
	unsigned char* buf = (unsigned char*) malloc ( w*h*4 );
	gvdb.ReadRenderBuf ( 0, buf );						// Read render buffer	

	save_png ( "img_importvdb.png", buf, w, h, 4 );		// Save image as png

	free ( buf );

	gprintf ( "Done.\n" );	
}

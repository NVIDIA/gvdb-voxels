//-----------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2017 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

// GVDB library
#include "gvdb.h"			
using namespace nvdb;

// Sample utils
#include "main.h"			// window system 
#include <GL/glew.h>

VolumeGVDB	gvdb;

class Sample : public NVPWindow {
public:
	virtual bool init();
	virtual void display();
	virtual void reshape(int w, int h);
	virtual void motion(int x, int y, int dx, int dy);
	virtual void mouse (NVPWindow::MouseButton button, NVPWindow::ButtonAction state, int mods, int x, int y);

	int			gl_screen_tex;
	int			mouse_down;

	Vector3DF	m_pretrans, m_scale, m_angs, m_trans;
};


bool Sample::init() 
{
	int w = getWidth(), h = getHeight();			// window width & height
	mouse_down = -1;
	gl_screen_tex = -1;	
	m_pretrans.Set(-125, -160, -125);
	m_scale.Set(1, 1, 1);
	m_angs.Set(0, 0, 0);
	m_trans.Set(0, 0, 0);	

	// Initialize GVDB	
	gvdb.SetVerbose ( true );
	gvdb.SetCudaDevice ( GVDB_DEV_FIRST );
	gvdb.Initialize ();
	gvdb.AddPath("../../source/shared_assets/");
	gvdb.AddPath("../source/shared_assets/");
	gvdb.AddPath ( "../shared_assets/" );
	gvdb.AddPath ( ASSET_PATH );

	// Load VBX
	char scnpath[1024];		
	if ( !gvdb.getScene()->FindFile ( "explosion.vbx", scnpath ) ) {
		nvprintf ( "Cannot find vbx file.\n" );
		nverror();
	}
	printf ( "Loading VBX. %s\n", scnpath );
	gvdb.SetChannelDefault ( 16, 16, 16 );
	gvdb.LoadVBX ( scnpath );	

	// Set volume params
	gvdb.SetTransform(m_pretrans, m_scale, m_angs, m_trans);
	gvdb.getScene()->SetSteps ( .25f, 16, .25f );			// Set raycasting steps
	gvdb.getScene()->SetExtinct ( -1.0f, 1.0f, 0.0f );		// Set volume extinction
	gvdb.getScene()->SetVolumeRange ( 0.1f, 0.0f, .5f );	// Set volume value range
	gvdb.getScene()->SetCutoff ( 0.005f, 0.005f, 0.0f );
	gvdb.getScene()->SetBackgroundClr ( 0.1f, 0.2f, 0.4f, 1.0 );
	gvdb.getScene()->LinearTransferFunc(0.00f, 0.25f, Vector4DF(0, 0, 0, 0), Vector4DF(1, 0, 0, 0.05f));
	gvdb.getScene()->LinearTransferFunc(0.25f, 0.50f, Vector4DF(1, 0, 0, 0.05f), Vector4DF(1, .5f, 0, 0.1f));
	gvdb.getScene()->LinearTransferFunc(0.50f, 0.75f, Vector4DF(1, .5f, 0, 0.1f), Vector4DF(1, 1, 0, 0.15f));
	gvdb.getScene()->LinearTransferFunc(0.75f, 1.00f, Vector4DF(1, 1, 0, 0.15f), Vector4DF(1, 1, 1, 0.2f));
	gvdb.CommitTransferFunc ();


	// Create Camera 
	Camera3D* cam = new Camera3D;						
	cam->setFov ( 50.0 );
	cam->setOrbit ( Vector3DF(20,30,0), Vector3DF(0,0,0), 700, 1.0 );	
	gvdb.getScene()->SetCamera( cam );
	
	// Create Light
	Light* lgt = new Light;								
	lgt->setOrbit ( Vector3DF(299,57.3f,0), Vector3DF(132,-20,50), 200, 1.0 );
	gvdb.getScene()->SetLight ( 0, lgt );	

	// Add render buffer
	nvprintf ( "Creating screen buffer. %d x %d\n", w, h );
	gvdb.AddRenderBuf ( 0, w, h, 4 );	

	// Create opengl texture for display
	// This is a helper func in sample utils (not part of gvdb),
	// which creates or resizes an opengl 2D texture.
	createScreenQuadGL ( &gl_screen_tex, w, h );

	return true; 
}

void Sample::reshape (int w, int h)
{
	// Resize the opengl screen texture
	glViewport(0, 0, w, h);
	createScreenQuadGL ( &gl_screen_tex, w, h );

	// Resize the GVDB render buffers
	gvdb.ResizeRenderBuf ( 0, w, h, 4 );

	postRedisplay();
}

void Sample::display() 
{
	m_angs.y += 0.05f;		
	gvdb.SetTransform(m_pretrans, m_scale, m_angs, m_trans);

	// Render volume
	gvdb.TimerStart ();
	gvdb.Render ( SHADE_VOLUME, 0, 0 );
	float rtime = gvdb.TimerStop();
	nvprintf ( "Render volume. %6.3f ms\n", rtime );

	// Copy render buffer into opengl texture
	// This function does a gpu-gpu device copy from the gvdb cuda output buffer
	// into the opengl texture, avoiding the cpu readback found in ReadRenderBuf
	gvdb.ReadRenderTexGL ( 0, gl_screen_tex );

	// Render screen-space quad with texture
	// This is a helper func in sample utils (not part of gvdb),
	// which renders an opengl 2D texture to the screen.
	renderScreenQuadGL ( gl_screen_tex );

	postRedisplay();
}

void Sample::motion(int x, int y, int dx, int dy) 
{
	// Get camera for GVDB Scene
	Camera3D* cam = gvdb.getScene()->getCamera();	

	switch ( mouse_down ) {	
	case NVPWindow::MOUSE_BUTTON_LEFT: {
		// Adjust orbit angles
		Vector3DF angs = cam->getAng();
		angs.x += dx*0.2f;
		angs.y -= dy*0.2f;
		cam->setOrbit ( angs, cam->getToPos(), cam->getOrbitDist(), cam->getDolly() );				
		postRedisplay();	// Update display
		} break;
	
	case NVPWindow::MOUSE_BUTTON_MIDDLE: {
		// Adjust target pos		
		cam->moveRelative ( float(dx) * cam->getOrbitDist()/1000, float(-dy) * cam->getOrbitDist()/1000, 0 );	
		postRedisplay();	// Update display
		} break;
	
	case NVPWindow::MOUSE_BUTTON_RIGHT: {	
		// Adjust dist
		float dist = cam->getOrbitDist();
		dist -= dy;
		cam->setOrbit ( cam->getAng(), cam->getToPos(), dist, cam->getDolly() );		
		postRedisplay();	// Update display
		} break;
	}
}

void Sample::mouse ( NVPWindow::MouseButton button, NVPWindow::ButtonAction state, int mods, int x, int y)
{
	// Track when we are in a mouse drag
	mouse_down = (state == NVPWindow::BUTTON_PRESS) ? button : -1;	
}

int sample_main ( int argc, const char** argv ) 
{
	Sample sample_obj;
	return sample_obj.run ( "NVIDIA(R) GVDB Voxels - gInteractveGL", "intergl", argc, argv, 1024, 768, 4, 5 );
}

void sample_print( int argc, char const *argv)
{
}


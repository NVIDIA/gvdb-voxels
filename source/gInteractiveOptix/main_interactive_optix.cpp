//--------------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2017, NVIDIA Corporation
//
// Redistribution and use in source and binary forms, with or without modification, 
// are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this 
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this 
//    list of conditions and the following disclaimer in the documentation and/or 
//    other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors may 
//    be used to endorse or promote products derived from this software without specific 
//   prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
// SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT 
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) 
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR 
// TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// Version 1.0: Rama Hoetzlein, 5/1/2017
//----------------------------------------------------------------------------------

// GVDB library
#include "gvdb.h"			
using namespace nvdb;

// OptiX scene
#include "optix_scene.h"	

// Sample utils
#include "main.h"			// window system 
#include <GL/glew.h>

VolumeGVDB	gvdb;

OptixScene  optx;

class Sample : public NVPWindow {
public:
	virtual bool init();
	virtual void display();
	virtual void reshape(int w, int h);
	virtual void motion(int x, int y, int dx, int dy);
	virtual void mouse (NVPWindow::MouseButton button, NVPWindow::ButtonAction state, int mods, int x, int y);

	void RebuildOptixGraph ();

	int			gl_screen_tex;			// screen texture
	int			mouse_down;				// mouse down status
	Vector3DF	delta;					// mouse delta
	int			frame;					// current frame
	int			sample;					// current sample
	int			max_samples;			// sample convergence

	int			mat_surf1;				// material id for surface objects
	int			mat_deep;				// material id for volumetric objects
};

void Sample::RebuildOptixGraph ()
{
	optx.ClearGraph ();	
	
	nvprintf ( "Adding OptiX materials.\n" );
	
	/// Add surface material
	mat_surf1 = optx.AddMaterial ( "optix_trace_surface", "trace_surface", "trace_shadow" );
	MaterialParams* matp = optx.getMaterialParams ( mat_surf1 );
	matp->light_width = 1.5;
	matp->shadow_width = 0.5;
	matp->diff_color = Vector3DF(.3f, .34f, .3f);
	matp->spec_color = Vector3DF(.3f, .3f, .3f );
	matp->spec_power = 40.0;
	matp->env_color  = Vector3DF(.1f, .1f, .1f );
	matp->refl_width = 5.0f;
	matp->refl_color = Vector3DF(3, 3, 3 );
	matp->refr_width = 0.5f;
	matp->refr_color = Vector3DF(.35f, .4f, .4f );
	matp->refr_ior = 1.2f;
	matp->refr_amount = 1.0f;
	matp->refr_offset = 15.0f;
	optx.SetMaterialParams ( mat_surf1 );

	/// Add deep volume material
	mat_deep = optx.AddMaterial ( "optix_trace_deep", "trace_deep", "trace_shadow" );
		
	// Add GVDB volume to the OptiX scene
	nvprintf ( "Adding GVDB Volume to OptiX graph.\n" );
	bool isDeep = true;
	bool isLevelSet = false;
	Vector3DF volmin = gvdb.getVolMin ();
	Vector3DF volmax = gvdb.getVolMax ();
	Matrix4F xform;
	xform.Identity ();
	int atlas_glid = gvdb.getAtlasGLID ( 0 );
	optx.AddVolume ( atlas_glid, volmin, volmax, (isDeep ? mat_deep : mat_surf1), xform, isDeep, isLevelSet );

	// Add polygonal model to the OptiX scene
	Model* m = gvdb.getScene()->getModel ( 0 );
	xform.Identity ();
	optx.AddPolygons ( m, 0, xform );
	
	// Set Transfer Function (once before validate)
	Vector4DF* src = gvdb.getScene()->getTransferFunc();
	optx.SetTransferFunc ( src );

	// Validate OptiX graph
	nvprintf ( "Validating OptiX.\n" );
	optx.ValidateGraph ();	

	// Assign GVDB data to OptiX	
	gvdb.PrepareVDB ();
	int vsz = gvdb.getVDBSize ();
	char* vdat = gvdb.getVDBInfo ();
	optx.AssignGVDB ( vsz, vdat );
}

bool Sample::init() 
{
	int w = getWidth(), h = getHeight();			// window width & height
	mouse_down = -1;
	gl_screen_tex = -1;
	frame = 0;
	sample = 0;
	max_samples = 1024;

	// Initialize Optix Scene
	optx.InitializeOptix ( w, h );

	// Initialize GVDB
	int devid = -1;
	gvdb.SetVerbose ( true );
        gvdb.SetProfile ( false );
	gvdb.SetCudaDevice ( devid );
	gvdb.Initialize ();	
	gvdb.AddPath ( std::string("../source/shared_assets/") );
	gvdb.AddPath ( std::string(ASSET_PATH) );

	// Must set GVDB to create OpenGL atlases, since OptiX uses 
	// opengl to access textures in optix intersection programs.
	gvdb.UseOpenGLAtlas ( true );

	// Load polygons
	// This loads an obj file into scene memory on cpu.
	char scnpath[1024];		
	printf ( "Loading polygon model.\n" );
	gvdb.getScene()->AddModel ( "lucy.obj", 300.0, 0, 0, 0 );
	gvdb.CommitGeometry( 0 );					// Send the polygons to GPU as OpenGL VBO	

	// Load VBX
	// This loads volumetric data
	if ( !gvdb.getScene()->FindFile ( "explosion.vbx", scnpath ) ) {
		nvprintf ( "Cannot find vbx file.\n" );
		nverror();
	}
	printf ( "Loading VDB. %s\n", scnpath );
	gvdb.LoadVBX ( scnpath );	

	// Set volume params
	gvdb.getScene()->SetSteps ( 0.25f, 16, 0.25f );				// Set raycasting steps
	gvdb.getScene()->SetExtinct ( -1.0f, 1.0f, 0.0f );		// Set volume extinction
	gvdb.getScene()->SetVolumeRange ( 0.1f, 0.0f, 1.0f );	// Set volume value range
	gvdb.getScene()->SetCutoff ( 0.005f, 0.01f, 0.0f );		
	gvdb.getScene()->LinearTransferFunc ( 0.00f, 0.25f, Vector4DF(0.f, 0.f,  0.f, 0.0f), Vector4DF(0.0f, 0.5f, 1.f, 0.02f) );
	gvdb.getScene()->LinearTransferFunc ( 0.25f, 0.50f, Vector4DF(0.f, 0.5f, 1.f, 0.02f), Vector4DF(0.0f, 0.f,  1.f, 0.02f) );
	gvdb.getScene()->LinearTransferFunc ( 0.50f, 0.75f, Vector4DF(0.f, 0.f,  1.f, 0.02f), Vector4DF(0.0f, 1.f,  1.f, 0.03f) );
	gvdb.getScene()->LinearTransferFunc ( 0.75f, 1.00f, Vector4DF(0.f, 1.f,  1.f, 0.03f), Vector4DF(0.0f, 1.f,  1.f, 0.05f) );
	gvdb.getScene()->SetBackgroundClr ( 0.1f, 0.2f, 0.4f, 1 );
	gvdb.CommitTransferFunc ();

	// Create Camera 
	Camera3D* cam = new Camera3D;						
	cam->setFov ( 30.0 );	
	cam->setOrbit ( Vector3DF(-45,30,0), Vector3DF(60,60,60), 600, 1.0 );	
	gvdb.getScene()->SetCamera( cam );
	
	// Create Light
	Light* lgt = new Light;								
	lgt->setOrbit ( Vector3DF(45,45,0), Vector3DF(50,50,50), 700, 1.0 );
	gvdb.getScene()->SetLight ( 0, lgt );	

	// Add render buffer
	nvprintf ( "Creating screen buffer. %d x %d\n", w, h );
	gvdb.AddRenderBuf ( 0, w, h, 4 );	

	// Create opengl texture for display
	// This is a helper func in sample utils
	// which creates or resizes an opengl 2D texture.
	createScreenQuadGL ( &gl_screen_tex, w, h );

	// Rebuild the Optix scene graph with GVDB
	RebuildOptixGraph ();

	return true; 
}

void Sample::reshape (int w, int h)
{
	// Resize the opengl screen texture
	createScreenQuadGL ( &gl_screen_tex, w, h );

	// Resize the GVDB render buffers
	gvdb.ResizeRenderBuf ( 0, w, h, 4 );

	postRedisplay();
}

void Sample::display() 
{
	// Update sample convergence
	optx.SetSample ( frame, sample );
	if ( ++sample < max_samples ) {		
		postRedisplay();
	} else {
		++frame;
		sample = 0;
	}

	// Set OptiX Camera 
	Camera3D* cam = gvdb.getScene()->getCamera();	
	optx.SetCamera ( cam->getPos(), cam->getU(), cam->getV(), cam->getW(), cam->getAspect() );		

	// Set OptiX Light	
	Light* light = gvdb.getScene()->getLight();
	Vector3DF lpos = light->getPos();
	optx.SetLight ( lpos );

	// Set Shading 
	optx.SetShading ( SHADE_TRILINEAR );	
	Vector3DF steps ( 1.0, 16, 0 );
	Vector3DF extinct ( -1.0f, 1.1f, 0.0f );	
	Vector3DF cutoff ( 0.005f, 0.01f, 0.0f );
	optx.SetVolumeParams ( steps, extinct, cutoff );

	// Render 	
	// gvdb.TimerStart ();
	if (1) {
		optx.Launch ();
		optx.ReadOutputTex ( gl_screen_tex );
	} else {		
		gvdb.getScene()->SetCrossSection ( Vector3DF(220,250,180), Vector3DF(0,0,-1) );
		gvdb.Render ( 0, SHADE_VOLUME, 0, 0, 1, 1, 1.0 );
		gvdb.ReadRenderTexGL ( 0, gl_screen_tex );
	}	
	// float rtime = gvdb.TimerStop();
	// nvprintf ( "Render. %6.3f ms\n", rtime );

	// Render screen-space quad with texture
	// This is a helper func in sample utils which 
	// renders an opengl 2D texture to the screen.
	renderScreenQuadGL ( gl_screen_tex );

	postRedisplay ();
}

void Sample::motion(int x, int y, int dx, int dy) 
{
	// Get camera for GVDB Scene
	Camera3D* cam = gvdb.getScene()->getCamera();	

	switch ( mouse_down ) {	
	case NVPWindow::MOUSE_BUTTON_LEFT: {
		// Adjust orbit angles
		Vector3DF angs = cam->getAng();
		delta.Set ( dx*0.2f, -dy*0.2f, 0 );
		angs += delta;
		cam->setOrbit ( angs, cam->getToPos(), cam->getOrbitDist(), cam->getDolly() );				
		sample = 0;
		postRedisplay();	// Update display
		} break;
	
	case NVPWindow::MOUSE_BUTTON_MIDDLE: {
		// Adjust target pos		
		cam->moveRelative ( float(dx) * cam->getOrbitDist()/1000, float(-dy) * cam->getOrbitDist()/1000, 0 );	
		sample = 0;
		postRedisplay();	// Update display
		} break;
	
	case NVPWindow::MOUSE_BUTTON_RIGHT: {	
		// Adjust dist
		float dist = cam->getOrbitDist();
		dist -= dy;
		cam->setOrbit ( cam->getAng(), cam->getToPos(), dist, cam->getDolly() );		
		sample = 0;
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
	return sample_obj.run ( "NVIDIA(R) GVDB Voxels - gInteractveOptix", argc, argv, 1024, 768, 4, 5 );
}

void sample_print( int argc, char const *argv)
{
}


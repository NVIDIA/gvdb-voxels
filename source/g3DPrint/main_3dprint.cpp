
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
// Version 1.1: Rama Hoetzlein, 2/20/2018
//----------------------------------------------------------------------------------

// GVDB library
#include "gvdb.h"			
using namespace nvdb;

// Sample utils
#include "main.h"			// window system 
#include "nv_gui.h"			// gui system
#include <GL/glew.h>
#include <algorithm>

VolumeGVDB	gvdb1;
VolumeGVDB	gvdb2;

//#define USE_GVDB2

class Sample : public NVPWindow {
public:
	virtual bool init();
	virtual void display();
	virtual void reshape(int w, int h);
	virtual void motion(int x, int y, int dx, int dy);
	virtual void keyboardchar(unsigned char key, int mods, int x, int y);
	virtual void mouse (NVPWindow::MouseButton button, NVPWindow::ButtonAction state, int mods, int x, int y);
	
	void		draw_topology (VolumeGVDB* gvdb);	// draw gvdb topology		
	void		render_section ();
	void		start_guis ( int w, int h );

	int			gl_screen_tex;
	int			gl_section_tex;
	int			mouse_down;
	bool		m_show_topo;	
	int			m_shade_style;
	int			m_chan;
};


void Sample::start_guis (int w, int h)
{
	clearGuis();
	setview2D ( w, h );
	guiSetCallback ( 0x0 );		
	addGui ( 10, h-30, 130, 20, "Topology", GUI_CHECK, GUI_BOOL, &m_show_topo, 0.f, 1.f );	
	addGui ( 150, h-30, 130, 20, "Shading",  GUI_COMBO, GUI_INT, &m_shade_style, 0.f, 5.f );		
		addItem ( "Voxel" );
		addItem ( "Surface" );
		addItem ( "Section" );
		addItem ( "Volume" );
}

bool Sample::init ()
{
	int w = getWidth(), h = getHeight();			// window width & height
	mouse_down = -1;
	gl_screen_tex = -1;	
	m_show_topo = false;
	m_shade_style = 0;
	m_chan = 0;
	srand ( 6572 );

	init2D ( "arial" );
	setview2D ( w, h );

	// Initialize GVDB
	printf ( "Starting GVDB.\n" );		
	gvdb1.SetDebug ( true );
	gvdb1.SetVerbose ( true );		// enable/disable console output from gvdb
	gvdb1.SetCudaDevice ( GVDB_DEV_FIRST );
	gvdb1.Initialize ();								
	gvdb1.StartRasterGL ();			// Start GVDB Rasterizer. Requires an OpenGL context.
	gvdb1.AddPath ( "../source/shared_assets/" );
	gvdb1.AddPath ( ASSET_PATH );

#ifdef USE_GVDB2
	// GVDB #2
	gvdb2.SetVerbose(true);		// enable/disable console output from gvdb
	gvdb2.SetCudaDevice(devid);
	gvdb2.Initialize();
	gvdb2.StartRasterGL();			// Start GVDB Rasterizer. Requires an OpenGL context.
	gvdb2.AddPath("../source/shared_assets/");
	gvdb2.AddPath(ASSET_PATH);
#endif

	
	// Load polygons
	// This loads an obj file into scene memory on cpu.
	printf ( "Loading polygon model.\n" );
	gvdb1.getScene()->AddModel ( "lucy.obj", 1.0, 0, 0, 0 );
	gvdb1.CommitGeometry( 0 );					// Send the polygons to GPU as OpenGL VBO

#ifdef USE_GVDB2
	gvdb2.getScene()->AddModel("lucy.obj", 1.1, 0, 0, 0);
	gvdb2.CommitGeometry(0);			
#endif

	// Configure the GVDB tree to the desired topology. We choose a 
	// topology with small upper nodes (3=8^3) and large bricks (5=32^3) for performance.
	// An apron of 1 is used for correct smoothing and trilinear surface rendering.
	printf ( "Configure.\n" );
	gvdb1.Configure ( 3, 3, 3, 3, 5 );	
	gvdb1.SetChannelDefault ( 16, 16, 1 );
	gvdb1.AddChannel ( 0, T_FLOAT, 1 );

#ifdef USE_GVDB2
	gvdb2.Configure ( 3, 3, 3, 3, 4);
	gvdb2.SetChannelDefault( 16, 16, 1 );
	gvdb2.AddChannel ( 0, T_FLOAT, 1 );
#endif

	// Create a transform	
	// The input polygonal model has been normalized with 1 unit height, so we 
	// set the desired part size by scaling in millimeters (mm). 
	// Translation has been added to position the part at (50,55,50).
	Matrix4F xform;	
	float part_size = 100.0;					// Part size is set to 100 mm height.
	xform.SRT ( Vector3DF(1,0,0), Vector3DF(0,1,0), Vector3DF(0,0,1), Vector3DF(50,55,50), part_size );
	
	// The part can be oriented arbitrarily inside the target GVDB volume
	// by applying a rotation, translation, or scale to the transform.
	Matrix4F rot;
	rot.RotateZYX( Vector3DF( 0, -10, 0 ) );
	xform *= rot;									// Post-multiply to rotate part

	// Set the voxel size
	// We can specify the voxel size directly to GVDB. This is the size of a single voxel in world units.
	// The voxel resolution of a rasterized part is the maximum number of voxels along each axis, 
	// and is found by dividing the part size by the voxel size.
	// To limit the resolution, one can invert the equation and find the voxel size for a given resolution.	
	Vector3DF voxelsize ( 0.2f, 0.2f, 0.2f );	// Voxel size (mm)

	// Poly-to-Voxels
	// Converts polygons-to-voxels using the GPU graphics pipeline.		
	Model* m = gvdb1.getScene()->getModel(0);
	
	gvdb1.SetVoxelSize ( voxelsize.x, voxelsize.y, voxelsize.z );	

	gvdb1.SolidVoxelize ( 0, m, &xform, 1.0, 0.5 );

#ifdef USE_GVDB2
	printf ( "SurfaceVoxelizeGL 2.\n" );
	gvdb2.SetVoxelSize(voxelsize.x, voxelsize.y, voxelsize.z);	
	gvdb2.SurfaceVoxelizeGL ( 0, m, &xform );
#endif

	// Set volume params
	printf ( "Volume params.\n" );
	gvdb1.getScene()->SetSteps ( 0.25f, 16.f, 0.25f );			// Set raycasting steps
	gvdb1.getScene()->SetVolumeRange ( 0.25f, 0.0f, 1.0f );		// Set volume value range
	gvdb1.getScene()->SetExtinct ( -1.0f, 1.1f, 0.f );			// Set volume extinction	
	gvdb1.getScene()->SetCutoff ( 0.005f, 0.005f, 0.f );
	gvdb1.getScene()->SetShadowParams ( 0, 0, 0 );
	gvdb1.getScene()->LinearTransferFunc ( 0.0f, 0.5f, Vector4DF(0,0,0,0), Vector4DF(1.f,1.f,1.f,0.5f) );
	gvdb1.getScene()->LinearTransferFunc ( 0.5f, 1.0f, Vector4DF(1.f,1.f,1.f,0.5f), Vector4DF(1,1,1,0.8f) );	
	gvdb1.CommitTransferFunc ();
	gvdb1.getScene()->SetBackgroundClr ( 0.1f, 0.2f, 0.4f, 1.0f );	

#ifdef USE_GVDB2
	gvdb2.getScene()->SetSteps(0.25f, 16.f, 0.25f);			// Set raycasting steps
	gvdb2.getScene()->SetVolumeRange(0.5f, 0.0f, 1.0f);	// Set volume value range	
	gvdb2.getScene()->SetExtinct(-1.0f, 1.5f, 0.f);		// Set volume extinction	
	gvdb2.getScene()->SetCutoff(0.005f, 0.01f, 0.f);
	gvdb2.getScene()->LinearTransferFunc(0.0f, 0.1f, Vector4DF(0, 0, 0, 0), Vector4DF(1.f, 1.f, 1.f, 0.5f));
	gvdb2.getScene()->LinearTransferFunc(0.1f, 1.0f, Vector4DF(1.f, 1.f, 1.f, 0.5f), Vector4DF(1, 1, 1, 1.f));
	gvdb2.CommitTransferFunc();
	gvdb2.getScene()->SetBackgroundClr(0.1f, 0.2f, 0.4f, 1.0f);
#endif
	
	// Create Camera 
	Camera3D* cam = new Camera3D;						
	cam->setFov ( 50.0 );
	cam->setOrbit ( Vector3DF(-45.f, 30.f, 0.f ), Vector3DF(50,55,50), 300.f, 1.0f );	
	gvdb1.getScene()->SetCamera( cam );		

	// Create Light
	Light* lgt = new Light;								
	lgt->setOrbit ( Vector3DF(299.0f, 57.3f, 0.f), Vector3DF(132.0f, -20.0f, 50.f), 200.f, 1.0f );
	gvdb1.getScene()->SetLight ( 0, lgt );	

	// Add render buffer 
	printf ( "Creating screen buffer. %d x %d\n", w, h );	
	gvdb1.AddRenderBuf ( 0, w, h, 4 );	
	gvdb1.AddRenderBuf ( 1, 256, 256, 4 );

#ifdef USE_GVDB2
	gvdb2.getScene()->SetCamera(cam);
	gvdb2.getScene()->SetLight(0, lgt);
	gvdb2.AddRenderBuf ( 0, w, h, 4);
	gvdb2.AddRenderBuf ( 1, 256, 256, 4);
#endif
	
	// Screen textures
	glViewport ( 0, 0, w, h );
	createScreenQuadGL ( &gl_screen_tex, w, h );			// screen render
	createScreenQuadGL ( &gl_section_tex, 256, 256 );		// cross section inset

	start_guis ( w, h );

	return true;
}


void Sample::render_section ()
{
	// Render cross-section
	float h = (float) getHeight();
	gvdb1.getScene()->SetCrossSection ( Vector3DF(50.0f, 100.0f-(getCurY()*100.0f/h), 50.0f), Vector3DF(30.0f, 1.f, 30.0f) );

	gvdb1.Render ( SHADE_SECTION2D, 0, 1 );		

	gvdb1.ReadRenderTexGL ( 1, gl_section_tex );
	
	renderScreenQuadGL ( gl_section_tex, -1, 0, 0, getWidth()/4, getHeight()/4, 0  );
}

void Sample::display()
{
	VolumeGVDB* gvdb;

	clearScreenGL();

	float h = (float)getHeight();
	float yslice = 100.0f - (getCurY()*100.0f / h);		// mouse to select section 
	if ( isFirstFrame() ) yslice = 50.0;				// first frame

	gvdb1.getScene()->SetCrossSection(Vector3DF(0.0f, yslice, 0.0f), Vector3DF(0.f, 1.f, 0.f));
	
	int sh;
	switch (m_shade_style) {
	case 0: sh = SHADE_VOXEL;		break;
	case 1: sh = SHADE_TRILINEAR;	break;
	case 2: sh = SHADE_SECTION3D;	break;
	case 3: sh = SHADE_VOLUME;		break;
	};
	if (m_chan == 0) gvdb = &gvdb1;
	if (m_chan == 1) gvdb = &gvdb2;
	
	gvdb->Render( sh, m_chan, 0 );	// Render voxels
	
	gvdb->ReadRenderTexGL(0, gl_screen_tex);		// Copy internal buffer into opengl texture
	
	renderScreenQuadGL ( gl_screen_tex );			// Render screen-space quad with texture 

	render_section ();	

	if ( m_show_topo ) draw_topology ( gvdb );			// Draw GVDB topology

	draw3D ();										// Render the 3D drawing groups

	drawGui (0);									// Render the GUI

	draw2D ();

	postRedisplay();								// Post redisplay since simulation is continuous

}


void Sample::draw_topology ( VolumeGVDB* gvdb )
{
	Vector3DF clrs[10];
	clrs[0] = Vector3DF(0,0,1);			// blue
	clrs[1] = Vector3DF(0,1,0);			// green
	clrs[2] = Vector3DF(1,0,0);			// red
	clrs[3] = Vector3DF(1,1,0);			// yellow
	clrs[4] = Vector3DF(1,0,1);			// purple
	clrs[5] = Vector3DF(0,1,1);			// aqua
	clrs[6] = Vector3DF(1,0.5,0);		// orange
	clrs[7] = Vector3DF(0,0.5,1);		// green-blue
	clrs[8] = Vector3DF(0.7f,0.7f,0.7f);	// grey

	Camera3D* cam = gvdb->getScene()->getCamera();
	
	start3D ( gvdb->getScene()->getCamera() );		// start 3D drawing
	
	Vector3DF bmin, bmax;
	Node* node;
	for (int lev=0; lev < 5; lev++ ) {				// draw all levels
		int node_cnt = gvdb->getNumNodes(lev);
		for (int n=0; n < node_cnt; n++) {			// draw all nodes at this level
			node = gvdb->getNodeAtLevel ( n, lev );
			bmin = gvdb->getWorldMin ( node );		// get node bounding box
			bmax = gvdb->getWorldMax ( node );		// draw node as a box
			drawBox3D ( bmin.x, bmin.y, bmin.z, bmax.x, bmax.y, bmax.z, clrs[lev].x, clrs[lev].y, clrs[lev].z, 1 );			
		}		
	}
	end3D();										// end 3D drawing
}

void Sample::motion(int x, int y, int dx, int dy) 
{
	// Get camera for GVDB Scene
	Camera3D* cam = gvdb1.getScene()->getCamera();	
	Light* lgt = gvdb1.getScene()->getLight();
	bool shift = (getMods() & NVPWindow::KMOD_SHIFT);		// Shift-key to modify light

	switch ( mouse_down ) {	
	case NVPWindow::MOUSE_BUTTON_LEFT: {
		// Adjust orbit angles
		Vector3DF angs = (shift ? lgt->getAng() : cam->getAng() );
		angs.x += dx*0.2f;
		angs.y -= dy*0.2f;		
		if ( shift )	lgt->setOrbit ( angs, lgt->getToPos(), lgt->getOrbitDist(), lgt->getDolly() );				
		else			cam->setOrbit ( angs, cam->getToPos(), cam->getOrbitDist(), cam->getDolly() );				
		postRedisplay();	// Update display
		} break;
	
	case NVPWindow::MOUSE_BUTTON_MIDDLE: {
		// Adjust target pos		
		cam->moveRelative ( float(dx) * cam->getOrbitDist()/1000, float(-dy) * cam->getOrbitDist()/1000, 0 );	
		postRedisplay();	// Update display
		} break;
	
	case NVPWindow::MOUSE_BUTTON_RIGHT: {	
		// Adjust dist
		float dist = (shift ? lgt->getOrbitDist() : cam->getOrbitDist());
		dist -= dy;
		if ( shift )	lgt->setOrbit ( lgt->getAng(), lgt->getToPos(), dist, cam->getDolly() );
		else			cam->setOrbit ( cam->getAng(), cam->getToPos(), dist, cam->getDolly() );		
		postRedisplay();	// Update display
		} break;
	}
}

void Sample::mouse ( NVPWindow::MouseButton button, NVPWindow::ButtonAction state, int mods, int x, int y)
{
	if ( guiHandler ( button, state, x, y ) ) return;

	// Track when we are in a mouse drag
	mouse_down = (state == NVPWindow::BUTTON_PRESS) ? button : -1;	
}

void Sample::keyboardchar(unsigned char key, int mods, int x, int y)
{
	switch ( key ) {
	case '1':  m_show_topo = !m_show_topo; break;	
	case '2':  m_shade_style = ( m_shade_style==3 ) ? 0 : m_shade_style+1; break;
	case ' ':  m_chan = 1 - m_chan; break;
	};
}


void Sample::reshape (int w, int h)
{	
	// Resize the opengl screen texture
	glViewport(0, 0, w, h);
	createScreenQuadGL ( &gl_screen_tex, w, h );

	// Resize the GVDB render buffers
	gvdb1.ResizeRenderBuf ( 0, w, h, 4 );

	// Resize 2D UI
	start_guis(w, h);

	postRedisplay();
}

int sample_main ( int argc, const char** argv ) 
{
	Sample sample_obj;
	return sample_obj.run ( "NVIDIA(R) GVDB Voxels - g3DPrint", "3dprint", argc, argv, 1024, 768, 4, 4 );
}

void sample_print( int argc, char const *argv)
{
}


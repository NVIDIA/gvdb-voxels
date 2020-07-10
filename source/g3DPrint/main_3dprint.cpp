
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
	
	void		revoxelize();
	void		draw_topology (VolumeGVDB* gvdb);	// draw gvdb topology		
	void		render_section ();
	void		start_guis ( int w, int h );

	int			gl_screen_tex;
	int			gl_section_tex;
	int			mouse_down;

	Vector3DF	m_pivot;
	float		m_part_size;
	float		m_voxel_size;
	int			m_voxelsize_select;

	bool		m_show_topo;	
	int			m_shade_style;
	int			m_chan;
};

Sample sample_obj;

void handle_gui(int gui, float val)
{
	switch (gui) {
	case 2: {				// Voxel size gui changed
		sample_obj.revoxelize();
	} break;
	}
}


void Sample::start_guis (int w, int h)
{
	clearGuis();
	setview2D ( w, h );
	guiSetCallback ( handle_gui );		
	addGui ( 10, h-30, 130, 20, "Topology", GUI_CHECK, GUI_BOOL, &m_show_topo, 0.f, 1.f );	
	addGui ( 150, h-30, 130, 20, "Shading",  GUI_COMBO, GUI_INT, &m_shade_style, 0.f, 5.f );		
		addItem ( "Voxel" );
		addItem ( "Surface" );
		addItem ( "Section" );
		addItem ( "Volume" );
	addGui( 300, h - 30, 200, 20, "Voxel Size", GUI_COMBO, GUI_INT, &m_voxelsize_select, 0.f, 5.f);
		addItem ( "0.5 mm, 10 MB");
		addItem ( "0.4 mm, 10 MB");
		addItem ( "0.3 mm, 20 MB");
		addItem ( "0.2 mm, 40 MB");
}

void Sample::revoxelize()
{
	gvdb1.DestroyChannels();
	gvdb1.AddChannel(0, T_FLOAT, 1);

	// Setup part dimensions
	m_part_size = 100.0;					// Part size = 100 mm (default)

	// Voxel size
	// *NOTE*: Voxelsize has been deprecated inside GVDB 1.1 (7/14/2019)
	// so we bring it out into the sample itself. For flexibility, applications now handle arbitrary
	// transforms to/from world coordinates and the voxel grid. The new function SetTransform 
	// allows the application to setup rendering to convert from unit voxel grid to world coordinates.
	// To apply voxel size for 3D printing, we include it in the xform matrix for SolidVoxelize.
	switch (m_voxelsize_select) {
	case 0:	m_voxel_size = 0.5f; break;
	case 1:	m_voxel_size = 0.4f; break;
	case 2:	m_voxel_size = 0.3f; break;
	case 3:	m_voxel_size = 0.2f; break;
	};

	// Create a transform	
	Matrix4F xform, m;
	xform.Identity();

	// Complete poly-to-voxel transform: 
	//    X = S(partsize) S(1/voxelsize) Torigin R
	//  (remember, matrices are multiplied left-to-right but applied conceptually right-to-left)
	m.Scale(m_part_size, m_part_size, m_part_size);
	xform *= m;									// 4th) Apply part size 
	m.Scale(1 / m_voxel_size, 1 / m_voxel_size, 1 / m_voxel_size);
	xform *= m;									// 3rd) Apply voxel size (scale by inverse of this)
	m.Translate(m_pivot.x, m_pivot.y, m_pivot.z);	// 2nd) Move part so origin is at bottom corner 
	xform *= m;
	m.RotateZYX(Vector3DF(0, -10, 0));			// 1st) Rotate part about the geometric center
	xform *= m;

	// Set transform for rendering
	// Scale the GVDB grid by voxelsize to render the model in our desired world coordinates
	gvdb1.SetTransform(Vector3DF(0, 0, 0), Vector3DF(m_voxel_size, m_voxel_size, m_voxel_size), Vector3DF(0, 0, 0), Vector3DF(0, 0, 0));

	// Poly-to-Voxels
	// Converts polygons-to-voxels using the GPU	
	Model* model = gvdb1.getScene()->getModel(0);

	gvdb1.SolidVoxelize(0, model, &xform, 1.0, 0.5);

	gvdb1.Measure(true);

#ifdef USE_GVDB2
	printf("SurfaceVoxelizeGL 2.\n");
	gvdb2.SetVoxelSize(voxelsize.x, voxelsize.y, voxelsize.z);
	gvdb2.SurfaceVoxelizeGL(0, m, &xform);
#endif
}

bool Sample::init ()
{
	int w = getWidth(), h = getHeight();			// window width & height
	mouse_down = -1;
	gl_screen_tex = -1;	
	m_show_topo = false;
	m_shade_style = 0;
	m_chan = 0;
	m_voxelsize_select = 0;
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
	gvdb1.CommitGeometry( 0 );		// Send the polygons to GPU as OpenGL VBO

	m_pivot.Set(0.3f, 0.45f, 0.3f); // This is the center of the polygon model.

#ifdef USE_GVDB2
	gvdb2.getScene()->AddModel("lucy.obj", 1.1, 0, 0, 0);
	gvdb2.CommitGeometry(0);			
#endif

	// Configure the GVDB tree to the desired topology. We choose a 
	// topology with small upper nodes (3=8^3) and large bricks (5=32^3) for performance.
	// An apron of 1 is used for correct smoothing and trilinear surface rendering.
	printf("Configure.\n");
	gvdb1.Configure(3, 3, 3, 3, 5);
	gvdb1.SetChannelDefault(16, 16, 1);

#ifdef USE_GVDB2
	gvdb2.Configure ( 3, 3, 3, 3, 4);
	gvdb2.SetChannelDefault( 16, 16, 1 );
	gvdb2.AddChannel ( 0, T_FLOAT, 1 );
#endif

	// Revoxelize the model into GVDB
	revoxelize();

	// Set volume params
	printf ( "Volume params.\n" );
	gvdb1.getScene()->SetSteps ( 0.5f, 16.f, 0.5f );			// Set raycasting steps
	gvdb1.getScene()->SetVolumeRange ( 0.25f, 0.0f, 1.0f );		// Set volume value range
	gvdb1.getScene()->SetExtinct ( -1.0f, 1.1f, 0.f );			// Set volume extinction	
	gvdb1.getScene()->SetCutoff ( 0.005f, 0.005f, 0.f );
	gvdb1.getScene()->SetShadowParams ( 0, 0, 0 );
	gvdb1.getScene()->LinearTransferFunc ( 0.0f, 0.5f, Vector4DF(0,0,0,0), Vector4DF(1.f,1.f,1.f,0.5f) );
	gvdb1.getScene()->LinearTransferFunc ( 0.5f, 1.0f, Vector4DF(1.f,1.f,1.f,0.5f), Vector4DF(1,1,1,0.8f) );	
	gvdb1.CommitTransferFunc ();
	gvdb1.getScene()->SetBackgroundClr ( 0.1f, 0.2f, 0.4f, 1.0f );	

#ifdef USE_GVDB2
	gvdb2.getScene()->SetSteps(0.5f, 16.f, 0.5f);		// Set raycasting steps
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
	cam->setOrbit ( Vector3DF(-45.f, 30.f, 0.f ), m_pivot * m_part_size, 300.f, 1.0f );	
	gvdb1.getScene()->SetCamera( cam );		

	// Create Light
	Light* lgt = new Light;								
	lgt->setOrbit ( Vector3DF(299.0f, 57.3f, 0.f), m_pivot * m_part_size * Vector3DF(1.3f, 1.8f, 1.1f), 200.f, 1.0f );
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
	Vector3DF world;
	world = m_pivot * m_part_size / m_voxel_size;	// GVDB voxel grid to world coordinates
	world.y *= 1.0f - getCurY() / h;				// Select cross-section on world y-axis 
	gvdb1.getScene()->SetCrossSection ( world, Vector3DF(world.x, 1.f, world.z) );

	gvdb1.Render ( SHADE_SECTION2D, 0, 1 );		

	gvdb1.ReadRenderTexGL ( 1, gl_section_tex );
	
	renderScreenQuadGL ( gl_section_tex, -1, 0, 0,
		static_cast<float>(getWidth())/4, static_cast<float>(getHeight())/4, 0 );
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
	start3D ( gvdb->getScene()->getCamera() );		// start 3D drawing

	for (int lev=0; lev < 5; lev++ ) {				// draw all levels
		int node_cnt = static_cast<int>(gvdb->getNumNodes(lev));
		const Vector3DF& color = gvdb->getClrDim(lev);
		const Matrix4F& xform = gvdb->getTransform();

		for (int n=0; n < node_cnt; n++) {			// draw all nodes at this level
			Node* node = gvdb->getNodeAtLevel ( n, lev );
			Vector3DF bmin = gvdb->getWorldMin(node); // get node bounding box
			Vector3DF bmax = gvdb->getWorldMax(node); // draw node as a box
			drawBox3DXform(bmin, bmax, color, xform);
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
	return sample_obj.run ( "NVIDIA(R) GVDB Voxels - g3DPrint", "3dprint", argc, argv, 1024, 768, 4, 4 );
}

void sample_print( int argc, char const *argv)
{
}


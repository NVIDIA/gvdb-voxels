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

// GVDB library
#include "gvdb.h"			
using namespace nvdb;

// Sample utils
#include "main.h"			// window system 
#include "nv_gui.h"			// gui system
#include <GL/glew.h>
#include <algorithm>

VolumeGVDB	gvdb;

class Sample : public NVPWindow {
public:
	virtual bool init();
	virtual void display();
	virtual void reshape(int w, int h);
	virtual void motion(int x, int y, int dx, int dy);
	virtual void keyboardchar(unsigned char key, int mods, int x, int y);
	virtual void mouse (NVPWindow::MouseButton button, NVPWindow::ButtonAction state, int mods, int x, int y);
	
	void		draw_topology ();	// draw gvdb topology
	void		draw_rays ();		// draw deposition rays
	void		simulate ();		// simulation material deposition
	void		start_guis ( int w, int h );

	int			gl_screen_tex;
	int			mouse_down;
	int			m_numrays;			// number of deposition rays
	DataPtr		m_rays;				// bundle of rays
	float		m_time;				// simulation time	
	bool		m_show_topo;
	bool		m_show_rays;
	bool		m_simulate;
	int			m_shade_style;
	int			m_wand_style;
};

#define WAND_ROTATE		0
#define WAND_SWEEP		1
#define WAND_WAVE		2

void handle_gui ( int gui, float val )
{
	if ( gui==3 ) {				// If shading gui has changed..
		if ( val==3 ) {			// Set cross section style, orange border 
			gvdb.getScene()->LinearTransferFunc ( 0.0f, 0.2f,  Vector4DF(0,0,0,0.0), Vector4DF(1,0.5,0,0.5) );	
			gvdb.getScene()->LinearTransferFunc ( 0.2f, 0.3f,   Vector4DF(1,0.5,0,0.5), Vector4DF(0,0,0,0.0) );	
			gvdb.getScene()->LinearTransferFunc ( 0.3f, 1.0f,   Vector4DF(0,0,0,0.0), Vector4DF(0,0,0,0.0) );	
			gvdb.CommitTransferFunc (); 
		} else {				// Or set volumetric style, x-ray white
			gvdb.getScene()->LinearTransferFunc ( 0.00f, 0.1f,  Vector4DF(0,0,0,0), Vector4DF(0,0,0, 0) );
			gvdb.getScene()->LinearTransferFunc ( 0.1f, 0.25f,  Vector4DF(0,0,0,0), Vector4DF(1,1,1, 0.8f) );
			gvdb.getScene()->LinearTransferFunc ( 0.25f, 0.5f,  Vector4DF(1,1,1,0.8f), Vector4DF(0,0,0,0) );
			gvdb.getScene()->LinearTransferFunc ( 0.5f, 1.0f,   Vector4DF(0,0,0,0.0), Vector4DF(0,0,0,0) );
			gvdb.CommitTransferFunc ();		
		}
	}
}

void Sample::start_guis (int w, int h)
{
	clearGuis();
	setview2D (w, h);
	guiSetCallback ( handle_gui );	
	addGui (  10, h-30, 130, 20, "Simulate", GUI_CHECK, GUI_BOOL, &m_simulate, 0, 1 );
	addGui ( 150, h-30, 130, 20, "Topology", GUI_CHECK, GUI_BOOL, &m_show_topo, 0, 1 );	
	addGui ( 300, h-30, 130, 20, "Rays",	 GUI_CHECK, GUI_BOOL, &m_show_rays, 0, 1 );
	addGui ( 450, h-30, 130, 20, "Shading",  GUI_COMBO, GUI_INT,  &m_shade_style, 0, 5 );
		addItem ( "Off" );
		addItem ( "Voxel" );
		addItem ( "Surface" );
		addItem ( "Section" );
		addItem ( "Volume" );
	addGui ( 600, h-30, 130, 20, "Wand Style",  GUI_COMBO, GUI_INT,  &m_wand_style, 0, 5 );
		addItem ( "Rotate" );
		addItem ( "Sweep" );
		addItem ( "Wave" );
}

bool Sample::init() 
{
	int w = getWidth(), h = getHeight();			// window width & height
	mouse_down = -1;
	gl_screen_tex = -1;
	m_time = 0;
	m_simulate = true;
	m_show_topo = false;
	m_show_rays = true;
	m_shade_style = 2;
	m_wand_style = WAND_ROTATE;
	srand ( 6572 );

	init2D ( "arial" );
	setview2D ( w, h );

	// Initialize GVDB
	int devid = -1;
	gvdb.SetDebug ( true );
	gvdb.SetVerbose ( true );
	gvdb.SetProfile ( false, true );
	gvdb.SetCudaDevice ( devid );
	gvdb.Initialize ();
	gvdb.AddPath ( "../source/shared_assets/" );
	gvdb.AddPath ( "../shared_assets/" );
	gvdb.AddPath ( ASSET_PATH );
	gvdb.StartRasterGL ();

	// Load polygons
	// This loads an obj file into scene memory on cpu.
	char scnpath[1024];		
	nvprintf ( "Loading polygon model.\n" );
	if ( !gvdb.FindFile ( "metal.obj", scnpath ) ) {
		nvprintf ( "Cannot find obj file.\n" );
		nverror();
	}
	gvdb.getScene()->AddModel ( scnpath, 1.0, 0, 0, 0 );
	gvdb.CommitGeometry( 0 );					// Send the polygons to GPU as OpenGL VBO

	// Set volume params
	gvdb.getScene()->SetSteps ( 0.25f, 16, 0.25f );				// Set raycasting steps
	gvdb.getScene()->SetExtinct ( -1.0f, 1.2f, 0.0f );		// Set volume extinction
	gvdb.getScene()->SetVolumeRange ( 0.3f, 0.0f, 1.0f );	// Set volume value range
	gvdb.getScene()->SetCutoff ( 0.005f, 0.01f, 0.0f );
	gvdb.getScene()->SetBackgroundClr ( 0.1f, 0.2f, 0.4f, 1.0f );
	gvdb.getScene()->LinearTransferFunc ( 0.0f, 0.2f,  Vector4DF(0,0,0,0.0), Vector4DF(1,1,1,0.1f) );	
	gvdb.getScene()->LinearTransferFunc ( 0.2f, 0.3f,   Vector4DF(1,1,1,0.05f), Vector4DF(1,1,1,0.05f) );	
	gvdb.getScene()->LinearTransferFunc ( 0.3f, 1.0f,   Vector4DF(1,1,1,0.05f), Vector4DF(0,0,0,0.0) );	
	gvdb.CommitTransferFunc (); 
	gvdb.SetEpsilon(0.01f, 256); // Use a larger epsilon than default to avoid artifacts between bricks

	// Configure a new GVDB volume
	gvdb.Configure ( 3, 3, 3, 3, 5 );

	// Atlas memory expansion will be supported in the Fall 2016 release, 
	// allowing the number of bricks to change dynamically. 
	// For this GVDB Beta, the last argument to AddChannel specifies the
	// maximum number of bricks. Keep this as low as possible for performance reasons.
	// AddChanell ( channel_id, channel_type, apron, max bricks )

	// Create two channels (density & color)
	gvdb.AddChannel ( 0, T_FLOAT, 1 );
	gvdb.AddChannel ( 1, T_UCHAR4, 1 );
	gvdb.SetColorChannel ( 1 );					// Let GVDB know channel 1 can be used for color	

	// Create Camera 
	Camera3D* cam = new Camera3D;						
	cam->setFov ( 50.0 );
	cam->setOrbit ( Vector3DF(45,40,0), Vector3DF(200,200,200), 1000, 1.0 );	
	gvdb.getScene()->SetCamera( cam );
	
	// Create Light
	Light* lgt = new Light;								
	lgt->setOrbit ( Vector3DF(80,50,0), Vector3DF(200,200,200), 800, 1.0 );
	gvdb.getScene()->SetLight ( 0, lgt );	

	// Add render buffer
	nvprintf ( "Creating screen buffer. %d x %d\n", w, h );
	gvdb.AddRenderBuf ( 0, w, h, 4 );	

	// Create rays
	m_numrays = 1000;
	gvdb.AllocData ( m_rays, m_numrays, sizeof(ScnRay) );

	// Rasterize the polygonal part to voxels
	Matrix4F xform;	
	float part_size = 200.0;							// Part size is set to 100 mm height.
	xform.SRT ( Vector3DF(1,0,0), Vector3DF(0,1,0), Vector3DF(0,0,1), Vector3DF(200,200,200), part_size );
	Model* m = gvdb.getScene()->getModel(0);

	gvdb.SolidVoxelize ( 0, m, &xform, 1, 1 );		// polygons to voxels

	// Fill color channel	
	gvdb.FillChannel ( 1, Vector4DF(0.7f, 0.7f, 0.7f, 1) );
	gvdb.Compute ( FUNC_SMOOTH, 0, 2, Vector3DF(4,0,0), true, true );
	gvdb.UpdateApron ();

	// Create opengl texture for display
	glViewport ( 0, 0, w, h );
	createScreenQuadGL ( &gl_screen_tex, w, h );

	start_guis ( w, h );

	return true; 
}

void Sample::reshape (int w, int h)
{
	// Resize the opengl screen texture
	glViewport(0, 0, w, h);
	createScreenQuadGL ( &gl_screen_tex, w, h );

	// Resize the GVDB render buffers
	gvdb.ResizeRenderBuf ( 0, w, h, 4 );

	// Resize 2D UI
	start_guis(w, h);

	postRedisplay();
}

void Sample::keyboardchar(unsigned char key, int mods, int x, int y)
{
	switch ( key ) {
	case '1': case ' ': m_simulate = !m_simulate; break;
	case '2':  m_show_topo = !m_show_topo; break;	
	case '3':  m_show_rays = !m_show_rays; break;
	case '4':  m_shade_style = ( m_shade_style==4 ) ? 0 : m_shade_style+1; handle_gui(3, (float) m_shade_style); break;	
	case '5':  m_wand_style = ( m_wand_style==2 ) ? 0 : m_wand_style+1;   break;	
	};
}


void Sample::simulate()
{
	m_time += 1.0;	

	// Metal Deposition simulation
	Vector3DF dir, rnd;
	ScnRay* ray = (ScnRay*) gvdb.getDataPtr( 0, m_rays );	
	float x, y, st, lt;
	int ndiv = (int) sqrt( (float) m_numrays);
	Vector3DF effect;

	switch ( m_wand_style ) {
	case WAND_ROTATE:	effect.Set( 1.0f, 0.0, 0.0 );	break;
	case WAND_SWEEP:	effect.Set( 0.0, 50, 0.0 );	break;
	case WAND_WAVE:		effect.Set( 0.0, 0.0, 0.5f );	break;
	};

	Matrix4F rot;
	rot.RotateY ( effect.x * m_time );		// wand rotation

	Vector3DF clr;							// color variation
	clr.x = sin( 2.0f*m_time * DEGtoRAD)*0.5f + 0.5f;
	clr.y = sin( 1.0f*m_time * DEGtoRAD)*0.5f + 0.5f;
	clr.z = sin( 0.5f*m_time * DEGtoRAD)*0.5f + 0.5f;

	st = sin( m_time * DEGtoRAD);			// sin time
	lt = (int(m_time) % 100)/100.0f - 0.5f;		// linear time

	// Initial ray origins and directions
	for (int n=0; n < m_numrays; n++ ) {
		rnd.Random ( -1, 1, -1, 1, -1, 1);
		
		// set ray origin
		x = float(n % ndiv)/ndiv - 0.5f + rnd.x*0.1f;		// random variation in rays
		y = float(n / ndiv)/ndiv - 0.5f + rnd.y*0.1f;
		x = std::max<float>(-0.5f, std::min<float>(0.5f, x));
		y = std::max<float>(-0.5f, std::min<float>(0.5f, y));
		ray->orig.Set ( x*25.0f, 0, y*0.5f + lt*effect.y );	// wand sweeping
		ray->orig *= rot;									// rotating wand over time
		ray->orig += Vector3DF( 200, 400, 200 );			// position of wand
		
		// set ray direction
		dir.Random ( -0.1f, 0.1f, 0, 0.0, 0, 0 );		
		dir += Vector3DF ( x, -0.5, y*0.04f + st*effect.z);	// wand angle (direction of rays)
		dir.Normalize ();		
		ray->dir = dir;
		ray->dir *= rot;									// rotate direction also
		
		// set ray color
		ray->clr = COLORA( clr.x, clr.y, clr.z, 0.2 );
		ray++;
	}

	// Transfer rays to the GPU	
	gvdb.CommitData ( m_rays );

	// Trace rays
	// Returns the hit position and normal of each ray
	gvdb.Raytrace ( m_rays, 0, SHADE_TRILINEAR, 0, -0.0001f);

	// Insert Points into the GVDB grid
	// This identifies a grid cell for each point. The SetPointStruct function accepts an arbitrary 
	// structure, which must contain a vec3f position input, and uint node offset and index outputs.
	// We can use the ray hit points as input directly from the ScnRay data structure.
	DataPtr pntpos, pntclr; 
	gvdb.SetDataGPU ( pntpos, m_numrays, m_rays.gpu, 0, sizeof(ScnRay) );
	gvdb.SetDataGPU ( pntclr, m_numrays, m_rays.gpu, 48, sizeof(ScnRay) );
	DataPtr data;
	gvdb.SetPoints ( pntpos, data, pntclr );  

	int scPntLen = 0;
	int subcell_size = 4;
	float radius=1.0;
	gvdb.InsertPointsSubcell (subcell_size, m_numrays, radius, Vector3DF(0,0,0), scPntLen);
	gvdb.GatherDensity (subcell_size, m_numrays, radius, Vector3DF(0,0,0), scPntLen, 0, 1, true ); // true = accumulate

	gvdb.UpdateApron ();
		
	// Smooth the volume
	// A smoothing effect simulates gradual erosion
	if ( int(m_time) % 20 == 0 ) {
		gvdb.Compute(FUNC_SMOOTH, 0, 1, Vector3DF(4, 0, 0), true, false);
	}
	
}

void Sample::draw_rays ()
{
	// Retrieve ray results from GPU
	gvdb.RetrieveData ( m_rays );

	// Draw rays
	Camera3D* cam = gvdb.getScene()->getCamera();
	start3D ( gvdb.getScene()->getCamera() );	
	Vector3DF hit, p1, p2;
	Vector4DF clr;	

	for (int n=0; n < m_numrays; n++ ) {
		ScnRay* ray = (ScnRay*) gvdb.getDataPtr ( n, m_rays );			
		clr.Set ( ray->clr );
		if (ray->hit.z != NOHIT ) {
			p2 = ray->hit;
			p1 = ray->orig; //dir; p1 *= -5.0; p1 += p2;
			drawLine3D ( p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, clr.x, clr.y, clr.z, clr.w );
		}
	}
	end3D();
}

void Sample::draw_topology ()
{
	start3D(gvdb.getScene()->getCamera());		// start 3D drawing

	for (int lev = 0; lev < 5; lev++) {				// draw all levels
		int node_cnt = static_cast<int>(gvdb.getNumNodes(lev));
		const Vector3DF& color = gvdb.getClrDim(lev);
		const Matrix4F& xform = gvdb.getTransform();

		for (int n = 0; n < node_cnt; n++) {			// draw all nodes at this level
			Node* node = gvdb.getNodeAtLevel(n, lev);
			Vector3DF bmin = gvdb.getWorldMin(node); // get node bounding box
			Vector3DF bmax = gvdb.getWorldMax(node); // draw node as a box
			drawBox3DXform(bmin, bmax, color, xform);
		}
	}

	end3D();										// end 3D drawing
}


// Primary display loop
void Sample::display() 
{
	clearScreenGL ();

	if ( m_simulate ) simulate();						// Simulation step

	gvdb.getScene()->SetCrossSection ( Vector3DF(50,50,50), Vector3DF(-1,0,0) );

	int sh;
	switch ( m_shade_style ) {
	case 0: sh = SHADE_OFF;			break;
	case 1: sh = SHADE_VOXEL;		break;
	case 2: sh = SHADE_TRILINEAR;	break;
	case 3: sh = SHADE_SECTION3D;	break;
	case 4: sh = SHADE_VOLUME;		break;
	};
	gvdb.Render ( sh, 0, 0 );						// Render volume to output buffer

	gvdb.ReadRenderTexGL ( 0, gl_screen_tex );		// Copy internal buffer into opengl texture

	renderScreenQuadGL ( gl_screen_tex );			// Render screen-space quad with texture 

	if ( m_show_rays && m_simulate ) draw_rays ();	// Draw deposition rays with OpenGL in 3D

	if ( m_show_topo ) draw_topology ();			// Draw GVDB topology
	
	draw3D ();										// Render the 3D drawing groups

	drawGui (0);									// Render the GUI

	draw2D ();

	postRedisplay();								// Post redisplay since simulation is continuous
}

void Sample::motion(int x, int y, int dx, int dy) 
{
	// Get camera for GVDB Scene
	Camera3D* cam = gvdb.getScene()->getCamera();	
	Light* lgt = gvdb.getScene()->getLight();
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

int sample_main ( int argc, const char** argv ) 
{
	Sample sample_obj;
	return sample_obj.run ( "NVIDIA(R) GVDB Voxels - gSprayDeposit", "spraydep", argc, argv, 1024, 768, 4, 5, 100 );
}

void sample_print( int argc, char const *argv)
{
}


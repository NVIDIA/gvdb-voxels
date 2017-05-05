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

// Fluid System
#include "fluid_system.h"

// Sample utils
#include "main.h"			// window system 
#include "nv_gui.h"			// gui system
#include <GL/glew.h>

VolumeGVDB	gvdb;

FluidSystem fluid;

struct TPnt {				// offset	size
	Vector3DF pos;			// 0		12
	uint	  pnode;		// 12		4
	uint      poff;			// 16		4
	uint      clr;			// 20		4	
};

class Sample : public NVPWindow {
public:
	virtual bool init();
	virtual void display();
	virtual void reshape(int w, int h);
	virtual void motion(int x, int y, int dx, int dy);
	virtual void keyboardchar(unsigned char key, int mods, int x, int y);
	virtual void mouse (NVPWindow::MouseButton button, NVPWindow::ButtonAction state, int mods, int x, int y);

	void		info(int id);
	void		draw_fluid ();		// draw fluid system
	void		draw_topology ();	// draw gvdb topology
	void		simulate ();		// simulation material deposition
	void		start_guis (int w, int h);	
	void		reconfigure ();

	Vector3DF	m_origin;
	int			m_numpnts;
	DataPtr		m_pntpos;
	DataPtr		m_pntclr;
	int			gl_screen_tex;
	int			mouse_down;	
	float		m_time;				// simulation time	
	bool		m_show_gui;
	bool		m_show_fluid;	
	bool		m_show_topo;
	bool		m_use_color;
	bool		m_simulate;
	int			m_surface;
	int			m_shade_style;
	int			m_id;
};
Sample sample_obj;

void Sample::reconfigure ()
{
	// Configure new GVDB volume
	gvdb.Configure ( 3, 3, 3, 3, 4 );	
	
	// VoxelSize. Determines the effective resolution of the voxel grid. 
	gvdb.SetVoxelSize ( 1.0f, 1.0f, 1.0f );
	gvdb.DestroyChannels ();	

	// Atlas memory expansion will be supported in the Fall 2016 release, 
	// allowing the number of bricks to change dynamically. 
	// For this GVDB Beta, the last argument to AddChannel specifies the
	// maximum number of bricks. Keep this as low as possible for performance reasons.
	// AddChanell ( channel_id, channel_type, apron, max bricks )
	gvdb.SetChannelDefault ( 8, 8, 8 );
	gvdb.AddChannel ( 0, T_FLOAT, 1 );	
	if ( m_use_color ) {
		gvdb.AddChannel ( 1, T_UCHAR4, 1 );
		gvdb.SetColorChannel ( 1 );
	}
}

void handle_gui ( int gui, float val )
{
	switch ( gui ) {
	case 3:	{				// Shading gui changed
		float alpha = (val==4) ? 0.03f : 0.8f;		// when in volume mode (#4), make volume very transparent
		gvdb.getScene()->LinearTransferFunc ( 0.00f, 0.05f,  Vector4DF(0, 0, 0, 0), Vector4DF(1, .5f, 0, 0.1f) );
		gvdb.getScene()->LinearTransferFunc ( 0.05f, 0.18f,  Vector4DF(1, .5f, 0, .1f), Vector4DF(.9f, .9f, .9f, alpha) );
		gvdb.getScene()->LinearTransferFunc ( 0.18f, 1.0f,   Vector4DF(.9f, .9f, .9f, alpha), Vector4DF(0,0,0,0) );
		gvdb.CommitTransferFunc ();		
		} break;
	case 4:					// Color gui changed
		sample_obj.reconfigure ();			// Reconfigure GVDB volume to add/remove a color channel
		break;
	}
}

void Sample::start_guis (int w, int h)
{
	setview2D (w, h);
	guiSetCallback ( handle_gui );	
	addGui (  10, h-30, 130, 20, "Simulate", GUI_CHECK, GUI_BOOL, &m_simulate, 0, 1 );
	addGui ( 150, h-30, 130, 20, "Topology", GUI_CHECK, GUI_BOOL, &m_show_topo, 0, 1 );
	addGui ( 300, h-30, 130, 20, "Fluid",    GUI_CHECK, GUI_BOOL, &m_show_fluid, 0, 1 );
	addGui ( 450, h-30, 130, 20, "Shading",  GUI_COMBO, GUI_INT, &m_shade_style, 0, 5 );
		addItem ( "Off" );
		addItem ( "Voxel" );
		addItem ( "Surface" );
		addItem ( "Section" );
		addItem ( "Volume" );
	addGui ( 600, h-30, 130, 20, "Color",    GUI_CHECK, GUI_BOOL, &m_use_color, 0, 1 );
	addGui ( 750, h-30, 130, 20, "Surface",  GUI_COMBO, GUI_INT,  &m_surface, 0, 1 );
		addItem ( "Scatter" );
		addItem ( "Gather" );
}

bool Sample::init() 
{
	int w = getWidth(), h = getHeight();			// window width & height
	mouse_down = -1;
	gl_screen_tex = -1;
	m_time = 0;
	m_simulate = true;
	m_show_gui = true;
	m_show_fluid = false;
	m_show_topo = false;
	m_use_color = false;
	m_surface = 0;
	m_shade_style = 2;
	m_id = 0;
	srand ( 6572 );

	init2D ( "arial" );

	// Initialize GVDB
	int devid = -1;
	gvdb.SetVerbose ( false );
	gvdb.SetProfile ( false );
	gvdb.SetCudaDevice ( devid );
	gvdb.Initialize ();
	gvdb.AddPath ( std::string ( "../source/shared_assets/" ) );
	gvdb.AddPath ( std::string (ASSET_PATH) );
	
	// Set volume params
	gvdb.getScene()->SetSteps ( 0.25, 16, 0.25 );				// Set raycasting steps
	gvdb.getScene()->SetExtinct ( -1.0f, 1.5f, 0.0f );		// Set volume extinction
	gvdb.getScene()->SetVolumeRange ( 0.5f, 0.0f, 1.0f );	// Set volume value range
	gvdb.getScene()->SetCutoff ( 0.005f, 0.01f, 0.0f );
	gvdb.getScene()->SetBackgroundClr ( 0.1f, 0.2f, 0.4f, 1.0f );
	
	// Configure volume
	reconfigure ();

	// Create opengl texture for display
	glViewport ( 0, 0, w, h );
	createScreenQuadGL ( &gl_screen_tex, w, h );

	// Initialize Fluid System
	nvprintf ( "Starting Fluid System.\n" );
	m_numpnts = 500000;
	m_origin = Vector3DF(450, 100, 450);
	fluid.Initialize ();
	fluid.Start ( m_numpnts );

	// Create Camera 
	Camera3D* cam = new Camera3D;						
	cam->setFov ( 50.0 );
	cam->setOrbit ( Vector3DF(50,30,0), m_origin, 700, 1.0 );	
	gvdb.getScene()->SetCamera( cam );
	
	// Create Light
	Light* lgt = new Light;								
	lgt->setOrbit ( Vector3DF(0,40,0), m_origin, 500, 1.0 );
	gvdb.getScene()->SetLight ( 0, lgt );	

	// Add render buffer
	nvprintf ( "Creating screen buffer. %d x %d\n", w, h );
	gvdb.AddRenderBuf ( 0, w, h, 4 );	

	// Initialize GUIs
	start_guis ( w, h );

	nvprintf ( "Running..\n" );
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

void Sample::simulate()
{
	m_time += 1.0;

	// Run fluid simulation
	PERF_PUSH ( "Simulate" );
	fluid.Run ();
	PERF_POP ();

	if ( m_shade_style == 0 ) return;		// Do not create volume if surface shading is off

	// Create a GVDB topology from fluid particles
   PERF_PUSH ( "Topology" );

	PERF_PUSH ( "Clear" );
	gvdb.Clear ();
	PERF_POP ();
	Vector3DF*	fpos = fluid.getPos(0);				// fluid positions
	uint*		fclr = fluid.getClr(0);					// fluid colors
	Vector3DF p1;	

	PERF_PUSH ( "Activate" );
	for (int n=0; n < m_numpnts; n++) {		
		p1 = (*fpos++) + m_origin;	// get fluid sim pos		
		if ( n % 2 == 0 ) gvdb.ActivateSpace ( p1 );					// Activate GVDB topology
	}
	PERF_POP ();	

	PERF_PUSH ( "Finish" );
	gvdb.FinishTopology ();	
	PERF_POP ();
	PERF_POP ();

	// Update and Clear Atlas
	gvdb.UpdateAtlas ();
	gvdb.ClearAtlas ();	

	// Insert and splat fluid particles into volume	
	
	//-- transfer fluid from cpu
	//gvdb.CommitData ( m_pntpos, m_numpnts, (char*) fluid.getPos(0), 0, sizeof(Vector3DF) );
	//gvdb.CommitData ( m_pntclr, m_numpnts, (char*) fluid.getClr(0), 0, sizeof(uint) );
	
	//-- use data already on gpu
	gvdb.SetDataGPU ( m_pntpos, m_numpnts, fluid.getBufferGPU(FPOS), 0, sizeof(Vector3DF) );	
	gvdb.SetDataGPU ( m_pntclr, m_numpnts, fluid.getBufferGPU(FCLR), 0, sizeof(uint) );
	gvdb.SetPoints ( m_pntpos, m_use_color ? m_pntclr : DataPtr() );
	

	if ( m_surface == 0 ) {
		// Scatter & Smooth
		gvdb.InsertPoints ( m_numpnts, m_origin, false );
		gvdb.ScatterPointDensity ( m_numpnts, 4.0, 1.0, m_origin );
		gvdb.Compute ( FUNC_SMOOTH,		0, 2, Vector3DF(2, 0.02f, 0), true );	
		if ( m_use_color )
		  gvdb.Compute ( FUNC_CLR_EXPAND,	1, 5, Vector3DF(1, 1, 0), true );
	
	} else {
		// Gather
		gvdb.InsertPoints ( m_numpnts, m_origin, true );			// true = prefix sum into node bins
		gvdb.GatherPointDensity ( m_numpnts, 2.0, 0 );
		gvdb.UpdateApron ();
	}

}

void Sample::draw_fluid ()
{
	Camera3D* cam = gvdb.getScene()->getCamera();	

	start3D ( gvdb.getScene()->getCamera() );		// start 3D drawing

	Vector3DF*	fpos = fluid.getPos(0);
	Vector3DF*	fvel = fluid.getVel(0);
	uint*		fclr = fluid.getClr(0);
	Vector3DF p1, p2;
	uint c;
	for (int n=0; n < fluid.NumPoints(); n++ ) {
		p1 = *fpos++; p1 += m_origin;
		p2 = *fvel++; p2 += p1; p2 += Vector3DF(.1f,.1f,.1f);
		c =  *fclr++;
		drawLine3D ( p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, RED(c), GRN(c), BLUE(c), 1 );		
	}
	end3D ();
}


void Sample::draw_topology ()
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

	Camera3D* cam = gvdb.getScene()->getCamera();		
	
	start3D ( gvdb.getScene()->getCamera() );		// start 3D drawing
	Vector3DF bmin, bmax;
	Node* node;
	for (int lev=0; lev < 5; lev++ ) {				// draw all levels
		int node_cnt = gvdb.getNumNodes(lev);				
		for (int n=0; n < node_cnt; n++) {			// draw all nodes at this level
			node = gvdb.getNodeAtLevel ( n, lev );
			bmin = gvdb.getWorldMin ( node );		// get node bounding box
			bmax = gvdb.getWorldMax ( node );		// draw node as a box
			drawBox3D ( bmin.x, bmin.y, bmin.z, bmax.x, bmax.y, bmax.z, clrs[lev].x, clrs[lev].y, clrs[lev].z, 1 );			
		}		
	}

	end3D();										// end 3D drawing
}


// Primary display loop	
void Sample::display() 
{
	clearScreenGL ();

	if ( m_simulate ) simulate();					// Simulation step

	gvdb.getScene()->SetCrossSection ( m_origin, Vector3DF(0,0,-1) );

	int sh;
	switch ( m_shade_style ) {
	case 0: sh = SHADE_OFF;			break;
	case 1: sh = SHADE_VOXEL;		break;
	case 2: sh = SHADE_TRILINEAR;	break;
	case 3: sh = SHADE_SECTION3D;	break;
	case 4: sh = SHADE_VOLUME;		break;
	};
	gvdb.Render ( 0, sh, 0, 0, 1, 1, 0.6f );	// Render volume to internal cuda buffer
	
	gvdb.ReadRenderTexGL ( 0, gl_screen_tex );		// Copy internal buffer into opengl texture*/

	renderScreenQuadGL ( gl_screen_tex );			// Render screen-space quad with texture 

	if ( m_show_fluid ) draw_fluid ();				// Draw fluid system

	if ( m_show_topo ) draw_topology ();			// Draw GVDB topology 
	
	if ( m_show_gui ) {

		draw3D ();									// Render the 3D drawing groups

		drawGui (0);

		draw2D ();
	}

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
		} break;
	
	case NVPWindow::MOUSE_BUTTON_MIDDLE: {
		// Adjust target pos		
		cam->moveRelative ( float(dx) * cam->getOrbitDist()/1000, float(-dy) * cam->getOrbitDist()/1000, 0 );	
		} break;
	
	case NVPWindow::MOUSE_BUTTON_RIGHT: {	
		// Adjust dist
		float dist = (shift ? lgt->getOrbitDist() : cam->getOrbitDist());
		dist -= dy;
		if ( shift )	lgt->setOrbit ( lgt->getAng(), lgt->getToPos(), dist, cam->getDolly() );
		else			cam->setOrbit ( cam->getAng(), cam->getToPos(), dist, cam->getDolly() );		
		} break;
	}
}


void Sample::info (int id)
{
	int lev = 0;
	nvdb::Node* node = gvdb.getNodeAtLevel( id, lev );
	
	Vector3DI val = node->mValue;
	nvprintf("%d: %d,%d,%d\n", id, val.x, val.y, val.z);
}

void Sample::keyboardchar(unsigned char key, int mods, int x, int y)
{
	switch ( key ) {
	case '`':  m_show_gui = !m_show_gui; break;
	case '1':  m_show_topo = !m_show_topo; break;
	case '2':  m_show_fluid = !m_show_fluid; break;	
	case '3':  m_shade_style = ( m_shade_style==4 ) ? 0 : m_shade_style+1; handle_gui(3, (float) m_shade_style); break;
	case '4':  m_use_color = !m_use_color; handle_gui(4, m_use_color); break;
	case ' ':  m_simulate = !m_simulate; break;
	case ',': m_id--; info(m_id);  break;
	case '.': m_id++; info(m_id);  break;
	};
}

void Sample::mouse ( NVPWindow::MouseButton button, NVPWindow::ButtonAction state, int mods, int x, int y)
{
	if ( guiHandler ( button, state, x, y ) ) return;

	// Track when we are in a mouse drag
	mouse_down = (state == NVPWindow::BUTTON_PRESS) ? button : -1;	
}

int sample_main ( int argc, const char** argv ) 
{
	return sample_obj.run ( "NVIDIA(R) GVDB Voxels - gFluidSim Sample", argc, argv, 1024, 768, 4, 5 );
}

void sample_print( int argc, char const *argv)
{
}


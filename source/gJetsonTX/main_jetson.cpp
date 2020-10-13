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

// Sample utils
#include "main.h"			// window system 
#include "nv_gui.h"			// gui system
#include <GL/glew.h>
#include <algorithm>

#include <fcntl.h>
#ifdef WIN32
	#include <io.h>
	#define O_NONBLOCK  0x0004
	#define O_NOCTTY	0x0400
    #define	O_NDELAY	O_NONBLOCK
#else
	#include <unistd.h>
#endif


VolumeGVDB	gvdb;

// 3D Printer states
#define STATE_READY  1			// Ready - printer is ready
#define STATE_SHOW   2			// Show - printer is exposing resin with image
#define STATE_COOL   3			// Cool - printer is letting layer cool
#define STATE_CYCLE  4			// Cycle - printer is cycling the part release motor (Y-axis)
#define STATE_ADV    5			// Advance - printer is advancing to next layer (X-axis)
#define STATE_RESET  6			// Reset - printer is being reset
#define STATE_DEMO_START  7		// DemoStart - printer is starting demo mode
#define STATE_DEMO_FWD  8		// DemoFwd - printer is in demo mode, running forward
#define STATE_DEMO_REV  9		// DemoRev - printer is in demo mode, running backward

class Sample : public NVPWindow {
public:
	virtual bool init();
	virtual void display();
	virtual void reshape(int w, int h);
	virtual void motion(int x, int y, int dx, int dy);
	virtual void keyboardchar(unsigned char key, int mods, int x, int y);
	virtual void mouse (NVPWindow::MouseButton button, NVPWindow::ButtonAction state, int mods, int x, int y);
	
	void		ExposeLayer ( int y, int oy  );		// Expose layer on projector 
    void		SendPrinter ( char* cmd );			// Send G-code command to printer
    void		StartPrinter ();					// Make connection to the printer 
    void		RunPrinter ();						// Run the printer
    void		ResetPrinter ();					// Reset the printer
    void		DemoPrinter ();						// Enter printer demo mode

	void		draw_topology ();	// draw gvdb topology	    
	void		start_guis ( int w, int h );
	int			gl_screen_tex;
	int			gl_section_tex;
	int			mouse_down;
	bool		m_show_topo;	
	int			m_shade_style;

	// 3D Printer settings
    float       m_partsize;		// height of part in mm
    float       m_layerhgt;		// height of each layer in mm
    int         m_prn;			// handle to the printer device
    int         m_state;		// state of printer 
    int         m_next;			// next state 
    int         m_count;		// current counter for waiting
    int         m_wait;			// time to wait until next state
    int         m_curry;		// current y-slice of printer
    int         m_maxy;			// maximum y-slice
    int         m_offy;			// optionally avoid starting at 0
};

void Sample::start_guis (int w, int h)
{
	setview2D (w, h);
	guiSetCallback ( 0x0 );		
	addGui ( 10, h-30, 130, 20, "Topology", GUI_CHECK, GUI_BOOL, &m_show_topo, 0, 1 );	
	addGui ( 150, h-30, 130, 20, "Shading",  GUI_COMBO, GUI_INT, &m_shade_style, 0, 5 );		
		addItem ( "Voxel" );
		addItem ( "Surface" );
		addItem ( "Section" );
		addItem ( "Volume" );
}

void Sample::SendPrinter ( char* cmd )
{
  nvprintf ( "  Sent: %s", cmd );
  write (m_prn, cmd, (int) strlen(cmd) );
}

void Sample::StartPrinter ()
{
  
    m_prn = open("/dev/ttyACM0", O_RDWR | O_NOCTTY | O_NDELAY );
    if ( m_prn == -1 ) {
      nvprintf ( "Error: Cannot connect to printer.\n");
      nverror();	
    } else {
		#ifndef WIN32
			fcntl ( m_prn, F_SETFL, 0 );
		#endif
    }
    m_next = STATE_RESET;
    m_count = 1;
    m_curry = 0;
}

void Sample::ResetPrinter ()
{
   m_count = 1;
   m_next = STATE_RESET;
}
void Sample::DemoPrinter ()
{
  m_count = 1;
   m_next = STATE_DEMO_START;
}


void Sample::RunPrinter ()
{
	// Very simple printer driver for the VOX3, 
	// JetsonTX1 DLP/SLA 3D printer.

	char cmd[1024];

    m_count--;
    if ( m_count > 0 )  return;
    if ( m_count == 0 )  m_state = m_next;
      
    switch ( m_state ) {
    case STATE_READY:      
        nvprintf ( "Ready.\n" );
		SendPrinter ( "F100\n" );			// set default feed rate
        m_next = STATE_SHOW;
		m_wait = 100;
	 break;
    case STATE_SHOW:
		// during show state, cross-section will be rendered
        nvprintf ( "Show: %d\n", m_curry );	
        m_wait = 200;						// exposure delay
        m_next = STATE_COOL;	 
        break;
    case STATE_COOL:
        nvprintf ( "Cool\n" );
        m_wait = 200;						// cooling delay
        m_next = STATE_CYCLE;	 
        break;
    case STATE_CYCLE:
		// cycle uses the Y-axis for tray release
      	nvprintf ( "Cycle\n" );				
		SendPrinter ( "G0 Y-1\n" );			// move to release position	
        SendPrinter ( "G0 Y0\n" );			// move to build position
		m_wait = 200;						// wait for motion to finish (no feedback)
        m_next = STATE_ADV;	
        break;
    case STATE_ADV:
        m_curry++;							// advance the current y-slice
		nvprintf ( "Advanced: %d, %f mm\n", m_curry, float(m_curry)*m_layerhgt );
		// advance uses the X-axis for part holder
		sprintf ( cmd, "G0 X%f\n", float(m_curry)*m_layerhgt );		// move to new y (in mm)
        SendPrinter ( cmd );
		m_wait = 100;						// wait for motion to finish
		m_next = STATE_SHOW;
	  break;
    case STATE_RESET:
        SendPrinter ( "G0 X0 Y0\n" );		// reset all axes to 0,0
		m_wait = 1000;						// wait for finish
		m_next = STATE_READY;
		m_curry = 0;
	 break;
    case STATE_DEMO_START:					// demo mode. 
        SendPrinter ( "G0 X0 Y0\n" );
        m_wait = 1000;
		m_next = STATE_DEMO_FWD;
		m_curry = 0;
        break;
    case STATE_DEMO_FWD:
		// demo mode: no exposures, continuous advance, 
		// occassionally run the release motor
        m_curry++;							// advance y
		nvprintf ( "Advanced: %d/%d, %f mm\n", m_curry, m_maxy, float(m_curry)*m_layerhgt );
		sprintf ( cmd, "G0 X%f\n", float(m_curry)*m_layerhgt );
        SendPrinter ( cmd );
		if ( m_curry % 20 == 0 ) {          // demo release axis
			 SendPrinter ( "G0 Y-1\n" );
			 SendPrinter ( "G0 Y0\n" );
		}
		m_wait = 5;	 
		m_next = STATE_DEMO_FWD;
		if ( m_curry > m_maxy )  m_next = STATE_DEMO_REV;	 
		break;
    case STATE_DEMO_REV:
		// demo mode: reverse direction, bring part holder back to origin
        m_curry--;                
		nvprintf ( "Reverse: %d/%d, %f mm\n", m_curry, m_maxy, float(m_curry)*m_layerhgt );
		sprintf ( cmd, "G0 X%f\n", float(m_curry)*m_layerhgt );
        SendPrinter ( cmd );
		m_wait = 5;	 
		m_next = STATE_DEMO_REV;
		if ( m_curry <= 0 )
			m_next = STATE_DEMO_START;
        break;
    };
    m_count = m_wait;

}

bool Sample::init ()
{
	int w = getWidth(), h = getHeight();			// window width & height
	mouse_down = -1;
	gl_screen_tex = -1;	
	m_show_topo = false;
	m_shade_style = 0;
	srand ( 6572 );

	init2D ( "arial" );
	setview2D ( w, h );

    m_offy = 20;
	m_partsize = 100.0f;	 // Part is 100 mm high (~4 inches)
	m_layerhgt = 0.25f;      // 0.25 mm/layer
	m_maxy = int(m_partsize / m_layerhgt);
	
	StartPrinter ();

	// Initialize GVDB
	printf ( "Starting GVDB.\n" );	
	int devid = -1;	
	gvdb.SetVerbose ( true );		// enable/disable console output from gvdb
	gvdb.SetCudaDevice ( devid );
	gvdb.Initialize ();								
	gvdb.StartRasterGL ();			// Start GVDB Rasterizer. Requires an OpenGL context.
	gvdb.AddPath ( ASSET_PATH );

	// Load polygons
	// This loads an obj file into scene memory on cpu.
	printf ( "Loading polygon model.\n" );
	gvdb.getScene()->AddModel ( "lucy.obj", 1.0, 0, 0, 0 );
	gvdb.CommitGeometry( 0 );					// Send the polygons to GPU as OpenGL VBO
	
	// Configure the GVDB tree to the desired topology. We choose a 
	// topology with small upper nodes (3=8^3) and large bricks (5=32^3) for performance.
	// An apron of 1 is used for correct smoothing and trilinear surface rendering.
	gvdb.Configure ( 3, 3, 3, 3, 5 );
    gvdb.SetChannelDefault ( 8, 8, 8 );
	gvdb.AddChannel ( 0, T_FLOAT, 1 );

	// Create a transform	
	// The input polygonal model has been normalized with 1 unit height, so we 
	// set the desired part size by scaling in millimeters (mm). 
	// Translation has been added to position the part at (50,55,50).
	Matrix4F xform;		
	xform.SRT ( Vector3DF(4.0f,0,0), Vector3DF(0,4.0f,0), Vector3DF(0,0,4.0f), Vector3DF(50,55,50), m_partsize );
	
	// The part can be oriented arbitrarily inside the target GVDB volume
	// by applying a rotation, translation, or scale to the transform.
	Matrix4F rot;
	rot.RotateZYX( Vector3DF( 0, -10, 0 ) );
	xform *= rot;								// Post-multiply to rotate part

	// Poly-to-Voxels
	// Converts polygons-to-voxels using the GPU graphics pipeline.	
	Model* m = gvdb.getScene()->getModel(0);
	gvdb.SurfaceVoxelizeGL ( 0, m, &xform );
	gvdb.UpdateApron ();

	// Set volume params
	gvdb.getScene()->SetVolumeRange ( 0.25f, 16, 0.25f );	// Set volume value range
	gvdb.getScene()->SetSteps ( 0.5f, 16, 0 );			// Set raycasting steps
	gvdb.getScene()->SetExtinct ( -1.0, 1.5f, 0 );		// Set volume extinction	
	gvdb.getScene()->SetCutoff ( 0.005f, 0.01f, 0 );
	gvdb.getScene()->LinearTransferFunc ( 0,    0.1f, Vector4DF(0,0,0,0), Vector4DF(1.f,1.f,1.f,0.5f) );
	gvdb.getScene()->LinearTransferFunc ( 0.1f, 1.0f, Vector4DF(1.f,1.f,1.f,0.5f), Vector4DF(1,1,1,1.f) );	
	gvdb.CommitTransferFunc ();
	gvdb.getScene()->SetBackgroundClr ( 0.1f, 0.2f, 0.4f, 1.0f );

	// Create Camera 
	Camera3D* cam = new Camera3D;						
	cam->setFov ( 50.0 );
	cam->setOrbit ( Vector3DF(-45,30,0), Vector3DF(50,55,50), 300, 1.0f );	
	gvdb.getScene()->SetCamera( cam );		
	
	// Create Light
	Light* lgt = new Light;								
	lgt->setOrbit ( Vector3DF(299,57.3f,0), Vector3DF(132,-20,50), 200, 1.0f );
	gvdb.getScene()->SetLight ( 0, lgt );	

	// Add render buffer 
	printf ( "Creating screen buffer. %d x %d\n", w, h );
	glViewport ( 0, 0, w, h );
	gvdb.AddRenderBuf ( 0, w, h, 4 );	
	gvdb.AddRenderBuf ( 1, 1920, 1080, 4 );

	createScreenQuadGL ( &gl_screen_tex, w, h );			// screen render

	createScreenQuadGL ( &gl_section_tex, 1920, 1080 );		// cross section inset

	start_guis ( w, h );

	return true;
}


void Sample::ExposeLayer ( int y, int oy )
{
	// Expose slice for printing
	
	// Determine layer to print (in mm)
	float ysec = float(y+oy) * m_layerhgt;
	gvdb.getScene()->SetCrossSection ( Vector3DF(50, ysec, 50), Vector3DF(50.0, 1, 50.0) );

	// Render slice with GVDB
	gvdb.Render ( SHADE_SECTION2D, 0, 0 );

	// Read back slice into OpenGL texture
	gvdb.ReadRenderTexGL ( 1, gl_section_tex );
	
	// Display slice full-screen 
	//renderScreenQuadGL ( gl_section_tex, 0, 0, 1920, 1080 );
	renderScreenQuadGL ( gl_section_tex );
}

void Sample::display ()
{
	// Run the printer driver
    RunPrinter ();
  
	clearScreenGL ();
			
	// If in show state, expose layer for printing
	if ( m_state == STATE_SHOW || m_state == STATE_DEMO_FWD || m_state == STATE_DEMO_REV ) 
	  ExposeLayer ( m_curry, m_offy );	
	
	draw2D ();

	postRedisplay();								// Post redisplay since simulation is continuous

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

void Sample::keyboardchar(unsigned char key, int mods, int x, int y)
{
  nvprintf ( "Key: %c\n", key );
	switch ( key ) {
	case '1':  m_show_topo = !m_show_topo; break;	
	case '2':  m_shade_style = ( m_shade_style==3 ) ? 0 : m_shade_style+1; break;
	case 'r': case 'R':	ResetPrinter (); break;
	case 'd': case 'D':     DemoPrinter (); break;
	};
}


void Sample::reshape (int w, int h)
{
	// Resize the opengl screen texture
	createScreenQuadGL ( &gl_screen_tex, w, h );

	// Resize the GVDB render buffers
	gvdb.ResizeRenderBuf ( 0, w, h, 4 );

	postRedisplay();
}

int sample_main ( int argc, const char** argv ) 
{
	Sample sample_obj;
	return sample_obj.run ( "NVIDIA(R) GVDB Voxels - gJetsonTX", "jetsontx", argc, argv, 1920, 1080, 4, 4 );
}

void sample_print( int argc, char const *argv)
{
}


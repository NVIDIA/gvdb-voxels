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
#include "gvdb_vec.h"
#include <GL/glew.h>

#ifdef USE_OPTIX
	// OptiX scene
	#include "optix_scene.h"
#endif

//#define USE_CPU_COPY		// by default use data already on GPU (no CPU copy)

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
	virtual void shutdown() override;

	void		info(int id);
	void		draw_fluid ();		// draw fluid system
	void		draw_topology ();	// draw gvdb topology
	void		simulate ();		// simulation material deposition
	void		start_guis (int w, int h);	
	void		reconfigure ();
	void		RebuildOptixGraph ();

	VolumeGVDB	gvdb;
	FluidSystem fluid;
#ifdef USE_OPTIX
	OptixScene  optx;
#endif

	Vector3DF	m_origin;
	float		m_radius;
	bool		m_rebuild_gpu;
	int			m_numpnts;
	DataPtr		m_pntpos;
	DataPtr		m_pntclr;
	int			gl_screen_tex;
	int			mouse_down;	
	float		m_time;				// simulation time	
	bool		m_show_gui;
	bool		m_render_optix;
	bool		m_show_fluid;	
	bool		m_show_topo;
	bool		m_use_color;
	bool		m_simulate;	
	int			m_shade_style;
	int			m_id;

	int			m_mat_surf1;
	int			m_frame;
	int			m_sample;
};
Sample sample_obj;

void Sample::reconfigure ()
{
	// Configure new GVDB volume
	gvdb.Configure ( 3, 3, 3, 3, 5 );	

	gvdb.DestroyChannels ();	

	gvdb.SetChannelDefault ( 16, 16, 1 );
	gvdb.AddChannel ( 0, T_FLOAT, 1 );	
	if ( m_use_color ) {
		gvdb.AddChannel ( 1, T_UCHAR4, 1 );
		gvdb.SetColorChannel ( 1 );
	}

}

void handle_gui ( int gui, float val )
{
	switch ( gui ) {	
	case 4:								// Color GUI changed
		sample_obj.reconfigure ();		// Reconfigure GVDB volume to add/remove a color channel		
		break;
	}
}

void Sample::start_guis (int w, int h)
{
	clearGuis();
	setview2D (w, h);
	guiSetCallback ( handle_gui );	
	addGui (  10, h-30, 130, 20, "Simulate", GUI_CHECK, GUI_BOOL, &m_simulate, 0, 1 );
	addGui ( 150, h-30, 130, 20, "Topology", GUI_CHECK, GUI_BOOL, &m_show_topo, 0, 1 );
	addGui ( 300, h-30, 130, 20, "Fluid",    GUI_CHECK, GUI_BOOL, &m_show_fluid, 0, 1 );	
	addGui ( 450, h-30, 130, 20, "Color",    GUI_CHECK, GUI_BOOL, &m_use_color, 0, 1 );	
}


void Sample::RebuildOptixGraph ()
{
	char filepath[1024];

	optx.ClearGraph();

	if ( gvdb.FindFile ( "sky.png", filepath) )
		optx.CreateEnvmap ( filepath );

	m_mat_surf1 = optx.AddMaterial("optix_trace_surface", "trace_surface", "trace_shadow");		
	MaterialParams* matp = optx.getMaterialParams( m_mat_surf1 );
	matp->light_width = 1.2f;
	matp->shadow_width = 0.1f;
	matp->shadow_bias = 0.5f;
	matp->amb_color = Vector3DF(.05f, .05f, .05f);
	matp->diff_color = Vector3DF(.7f, .7f, .7f);
	matp->spec_color = Vector3DF(1.f, 1.f, 1.f);
	matp->spec_power = 400.0;
	matp->env_color = Vector3DF(0.f, 0.f, 0.f);
	matp->refl_width = 0.5f;
	matp->refl_bias = 0.5f;
	matp->refl_color = Vector3DF(0.4f, 0.4f, 0.4f);
	
	matp->refr_width = 0.0f;
	matp->refr_color = Vector3DF(0.1f, .1f, .1f);
	matp->refr_ior = 1.1f;
	matp->refr_amount = 0.5f;
	matp->refr_offset = 50.0f;
	matp->refr_bias = 0.5f;
	optx.SetMaterialParams( m_mat_surf1, matp );

	// Add GVDB volume to the OptiX scene
	nvprintf("Adding GVDB Volume to OptiX graph.\n");
	// Get the dimensions of the volume by looking at how the fluid system was
	// initialized (since at this moment, gvdb.getVolMin and gvdb.getVolMax
	// are both 0).
	Vector3DF volmin = fluid.GetGridMin();
	Vector3DF volmax = fluid.GetGridMax();
	Matrix4F xform = gvdb.getTransform();
	int atlas_glid = gvdb.getAtlasGLID(0);
	optx.AddVolume( atlas_glid, volmin, volmax, xform, m_mat_surf1, 'L' );		

	// Ground polygons
	if ( gvdb.FindFile ( "ground.obj", filepath) ) {
		Model* m;
		gvdb.getScene()->AddModel ( filepath, 1.0, 0, 0, 0 );	
		m = gvdb.getScene()->getModel ( 0 );	
		xform.RotateZ ( 5 );
		xform.PreTranslate ( Vector3DF(0, 10, 0) );		
		optx.AddPolygons ( m, m_mat_surf1, xform );
	}

	// Set Transfer Function (once before validate)
	Vector4DF* src = gvdb.getScene()->getTransferFunc();
	optx.SetTransferFunc(src);

	// Validate OptiX graph
	nvprintf("Validating OptiX.\n");
	optx.ValidateGraph();

	// Assign GVDB data to OptiX	
	nvprintf("Update GVDB Volume.\n");
	optx.UpdateVolume(&gvdb);

	nvprintf("Rebuild Optix.. Done.\n");
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
	m_use_color = true;
	m_render_optix = true;
	m_shade_style = 1;
	m_frame = 0;
	m_sample = 0;
	m_id = 0;
	m_origin = Vector3DF(0, 0, 0);
	m_radius = 1.0;
	m_rebuild_gpu = true;
	srand ( 6572 );

	init2D ( "arial" );

	// Initialize OptiX
	if (m_render_optix) {
		optx.InitializeOptix(w, h);
	}
	// Initialize 
	fluid.SetDebug ( false );

	gvdb.SetDebug ( false );
	gvdb.SetVerbose ( false );
	gvdb.SetProfile ( false, true );	
	gvdb.SetCudaDevice( m_render_optix ? GVDB_DEV_CURRENT : GVDB_DEV_FIRST );
	gvdb.Initialize ();
	gvdb.AddPath ( ASSET_PATH );
	
	// Set volume params
	gvdb.getScene()->SetSteps ( 0.25f, 16, 0.25f );			// Set raycasting steps
	gvdb.getScene()->SetExtinct ( -1.0f, 1.5f, 0.0f );		// Set volume extinction
	gvdb.getScene()->SetVolumeRange ( 0.0f, 3.0f, -1.0f );	// Set volume value range
	gvdb.getScene()->SetCutoff ( 0.005f, 0.01f, 0.0f );
	gvdb.getScene()->SetBackgroundClr ( 0.8f, 0.8f, 0.8f, 1.0f );
	
	// Configure volume
	reconfigure ();

	// Create opengl texture for display
	glViewport ( 0, 0, w, h );
	createScreenQuadGL ( &gl_screen_tex, w, h );

	// Initialize Fluid System
	nvprintf ( "Starting Fluid System.\n" );
	m_numpnts = 1500000;
	
	#ifdef USE_CPU_COPY
		// Do not assume there is a GPU buffer. Create some.
		gvdb.AllocData(m_pntpos, m_numpnts, sizeof(Vector3DF), false );
		gvdb.AllocData(m_pntclr, m_numpnts, sizeof(uint), false);
	#endif

	fluid.Initialize ();
	fluid.Start ( m_numpnts );

	Vector3DF ctr = (fluid.GetGridMax() + fluid.GetGridMin()) * Vector3DF(0.5,0.5,0.5);

	// Create Camera 
	Camera3D* cam = new Camera3D;
	cam->setFov ( 50.0 );
	cam->setOrbit ( Vector3DF(50,30,0), ctr, 1200, 1.0 );	
	gvdb.getScene()->SetCamera( cam );
	
	// Create Light
	Light* lgt = new Light;								
	lgt->setOrbit ( Vector3DF(20,60,0), ctr, 1000, 1.0 );
	gvdb.getScene()->SetLight ( 0, lgt );	

	// Add render buffer
	nvprintf ( "Creating screen buffer. %d x %d\n", w, h );
	gvdb.AddRenderBuf ( 0, w, h, 4 );	

	// Initialize GUIs
	start_guis ( w, h );

	if ( m_render_optix ) {
		RebuildOptixGraph ();
		optx.UpdateVolume ( &gvdb );
	}

	nvprintf ( "Running..\n" );
	return true; 
}

void Sample::reshape (int w, int h)
{
	// Resize the opengl screen texture
	glViewport(0, 0, w, h);
	createScreenQuadGL ( &gl_screen_tex, w, h );

	// Resize the GVDB render buffers
	gvdb.ResizeRenderBuf ( 0, w, h, 4 );

	// Resize OptiX output
	if ( m_render_optix )
		optx.ResizeOutput(w, h);

	// Resize 2D UI
	start_guis(w, h);

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

	// Setup GPU points for GVDB
	#ifdef USE_CPU_COPY
												//-- transfer point data from cpu
												// common use case: reading data from disk, import from a device
		gvdb.CommitData(m_pntpos, m_numpnts, (char*)fluid.getPos(0), 0, sizeof(Vector3DF));
		gvdb.CommitData(m_pntclr, m_numpnts, (char*)fluid.getClr(0), 0, sizeof(uint));
	#else 
												//-- assign data already on gpu
												// common use case: simulation already on gpu
		gvdb.SetDataGPU(m_pntpos, m_numpnts, fluid.getBufferGPU(FPOS), 0, sizeof(Vector3DF));
		gvdb.SetDataGPU(m_pntclr, m_numpnts, fluid.getBufferGPU(FCLR), 0, sizeof(uint));
	#endif

	DataPtr pntvel, clrpos;
	gvdb.SetPoints(m_pntpos, pntvel, m_use_color ? m_pntclr : clrpos);

	// Rebuild Topology
	PERF_PUSH("Topology");	

	if (m_rebuild_gpu) {
		// GPU rebuild
		gvdb.RebuildTopology( m_numpnts, m_radius*2.0f, m_origin);
		gvdb.FinishTopology( false, true );

	} else {
		// CPU rebuild
		Vector3DF*	fpos = fluid.getPos(0);				// fluid positions
		uint*		fclr = fluid.getClr(0);					// fluid colors
		Vector3DF p1;			
		for (int n=0; n < m_numpnts; n++) {		
			p1 = (*fpos++) + m_origin;	// get fluid sim pos		
			if ( n % 2 == 0 ) gvdb.ActivateSpace ( p1 );					// Activate GVDB topology
		}	
		gvdb.FinishTopology ();			
	}
	PERF_POP ();

	// Update and Clear Atlas	
	gvdb.UpdateAtlas ();	

	// Insert and Gather Points-to-Voxels
	int scPntLen = 0, subcell_size = 4;
	gvdb.InsertPointsSubcell (subcell_size, m_numpnts, m_radius*2.0f, m_origin, scPntLen );
	gvdb.GatherLevelSet (subcell_size, m_numpnts, m_radius, m_origin, scPntLen, 0, 1 );
	gvdb.UpdateApron(0, 3.0f);
	if (m_use_color) gvdb.UpdateApron(1, 0.0f);

	// Smooth voxels
	//gvdb.Compute ( FUNC_SMOOTH,		0, 2, Vector3DF(1, -0.05f, 0), true );	

	// Update OptiX
	PERF_PUSH("Update OptiX");
		if (m_render_optix) optx.UpdateVolume(&gvdb);			// GVDB topology has changed
	PERF_POP();
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

	if ( m_simulate ) {
		simulate();					// Simulation step				
		m_frame++;
		m_sample = 0;
	}

	if (m_render_optix) {
		// OptiX render
		optx.SetSample( m_frame, m_sample++ );
		PERF_PUSH("Raytrace");
		optx.Render( &gvdb, SHADE_LEVELSET, 0);
		PERF_POP();
		PERF_PUSH("ReadToGL");
		optx.ReadOutputTex(gl_screen_tex);
		PERF_POP();
	} else {
		// CUDA render
		PERF_PUSH("Raytrace");
		gvdb.Render( SHADE_LEVELSET, 0, 0);
		PERF_POP();
		PERF_PUSH("ReadToGL");
		gvdb.ReadRenderTexGL(0, gl_screen_tex);
		PERF_POP();
	}
	renderScreenQuadGL ( gl_screen_tex );			// Render screen-space quad with texture 
	
	if ( m_show_gui ) {

		if (m_show_fluid) draw_fluid();				// Draw fluid system

		if (m_show_topo) draw_topology();			// Draw GVDB topology 

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
		m_sample = 0;
		} break;
	
	case NVPWindow::MOUSE_BUTTON_MIDDLE: {
		// Adjust target pos		
		cam->moveRelative ( float(dx) * cam->getOrbitDist()/1000, float(-dy) * cam->getOrbitDist()/1000, 0 );	
		m_sample = 0;
		} break;
	
	case NVPWindow::MOUSE_BUTTON_RIGHT: {	
		// Adjust dist
		float dist = (shift ? lgt->getOrbitDist() : cam->getOrbitDist());
		dist -= dy;
		if ( shift )	lgt->setOrbit ( lgt->getAng(), lgt->getToPos(), dist, cam->getDolly() );
		else			cam->setOrbit ( cam->getAng(), cam->getToPos(), dist, cam->getDolly() );		
		m_sample = 0;
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

void Sample::shutdown() {
	// Because of the way NVPWindow works, we have to explicitly destroy its objects:
	gvdb.~VolumeGVDB();
	fluid.~FluidSystem();
	optx.~OptixScene();
}

int sample_main ( int argc, const char** argv ) 
{
	return sample_obj.run ( "NVIDIA(R) GVDB Voxels - gFluidSurface Sample", "fluidsurf", argc, argv, 1024, 768, 4, 5 );
}

void sample_print( int argc, char const *argv)
{
}




// GVDB library
#include "gvdb.h"			
#include "gvdb_render.h"	// OpenGL rendering
using namespace nvdb;

// Sample utils
#include "main.h"			// window system 
#include "nv_gui.h"			// gui system
#include <GL/glew.h>

VolumeGVDB	gvdb;

class Sample : public NVPWindow {
public:
	virtual bool init();
	virtual void display();
	virtual void reshape(int w, int h);
	virtual void motion(int x, int y, int dx, int dy);
	virtual void keyboardchar(unsigned char key, int mods, int x, int y);
	virtual void mouse (NVPWindow::MouseButton button, NVPWindow::ButtonAction state, int mods, int x, int y);

	bool		LoadRAW(char* fname, Vector3DI res, int bpp);
	bool		ConvertToFloat(Vector3DI res, uchar* dat);
	void		Rebuild() { Rebuild(m_VolMax, m_sparse, m_halo); }
	void		Rebuild( Vector3DF vmax, bool bSparse, bool bHalo);
	
	void		start_guis(int w, int h);
	void		draw_topology();

	Vector3DI	m_DataRes;
	int			m_DataBpp;			// 1=byte, 2=ushort, 4=float
	char*		m_DataBuf;
	Vector3DF	m_VolMax;

	int			m_gvdb_tex;	
	int			mouse_down;
	bool		m_show_topo;
	bool		m_sparse;
	bool		m_halo;
};
Sample sample_obj;


void handle_gui(int gui, float val)
{
	if (gui >= 1) {
		sample_obj.Rebuild();
	}
	sample_obj.postRedisplay();
}

void Sample::start_guis(int w, int h)
{
	clearGuis();
	setview2D(w, h);
	guiSetCallback(handle_gui);
	addGui(20, h - 30, 130, 20, "Topology", GUI_CHECK, GUI_BOOL, &m_show_topo, 0, 1);
	addGui(160, h - 30, 130, 20, "Halo", GUI_CHECK, GUI_BOOL, &m_halo, 0, 1);
	addGui(300, h - 30, 130, 20, "Sparse", GUI_CHECK, GUI_BOOL, &m_sparse, 0, 1);	
}

bool Sample::ConvertToFloat(Vector3DI res, uchar* dat)
{
	float* vnew = (float*) malloc(res.x*res.y*res.z * sizeof(float) );
	float* vdest = vnew;
	uchar* vsrc = dat;

	for (int n = 0; n < res.x*res.y*res.z; n++)
		*vdest++ = float(*vsrc++) / 256.0f;
	
	if (m_DataBuf != 0) free(m_DataBuf);
	m_DataBuf = (char*) vnew;
	m_DataRes = res;
	m_DataBpp = 4;
	
	return true;
}

bool Sample::LoadRAW (char* fname, Vector3DI res, int bpp)
{
	// Load RAW file into memory
	FILE* fp = fopen(fname, "rb");
	if (fp == 0) {
		nvprintf("Cannot read %s.\n", fname);
		return false;
	}
	m_DataRes = res;
	m_DataBpp = bpp;
	int sz = res.x*res.y*res.z*bpp;	

	if (m_DataBuf != 0x0) free(m_DataBuf);
	m_DataBuf = (char*) malloc(sz);

	fread(m_DataBuf, 1, sz, fp);
	
	fclose(fp);
	return true;
}

void Sample::Rebuild(Vector3DF vmax, bool bSparse, bool bHalo )
{
	gvdb.Clear();
	gvdb.DestroyChannels();

	// Configure VDB 
	printf("Configure GVDB.\n");	
	gvdb.Configure(3, 3, 3, 3, 4);
	gvdb.SetChannelDefault(16, 16, 16);
	gvdb.AddChannel(0, T_FLOAT, 1);

	// Set volume transform
	Matrix4F xform;
	// NOTE: 
	// Transform represents the mapping from output space to input space (GVDB space to source data).
	// SRT = Scale, Rotate, Translate.  p' = S R T p
	// Rotate: Source data uses a different coordinate system, so we swap the Y and Z axes.
	// Scale:  Source has Z+/- inverted, so we translate and invert the Y input (which gets mapped to Z)
	//         Also, source X-axis is 128, so we scale the output res of 256 by 0.5. Scale=(.5,-1,1)
	// Transl: Source has Z+/- inverted, so we translate to adjust the inverted Y. Trans=(0,0,256)
	//
	xform.SRT(Vector3DF(1, 0, 0), Vector3DF(0, 0, 1), Vector3DF(0, 1, 0), Vector3DF(0, 0, 252), Vector3DF(.5, -1, 1));

	// Activate volume
	printf("Activate GVDB volume.\n");
	Extents e;
	e = gvdb.ComputeExtents(1, Vector3DF(0, 0, 0), vmax);

	if (!bSparse) {
		// Dense - Activate all bricks in the data volume
		// To load data densely, we simply active every brick in the volume extents.
		gvdb.ActivateRegion(0, e);

	} else {
		// Sparse - Only activate bricks above threshold value
		// To load data sparsely, we downsample the volume to get the average value at each brick.		
		// DownsampleCPU will downsample the input AUX volume and retrieve the values back to CPU.
		gvdb.DownsampleCPU (xform, m_DataRes, AUX_DATA3D, e.ires, vmax, AUX_DOWNSAMPLED, Vector3DF(0, 1, 0), Vector3DF(0, 1, 0));
	
		// Then we only activate bricks whose average is above some threshold.
		gvdb.ActivateRegionFromAux (e, AUX_DOWNSAMPLED, T_FLOAT, 0.05f);

		if ( bHalo ) gvdb.ActivateHalo(e);
	}
	gvdb.FinishTopology();
	gvdb.UpdateAtlas();
	gvdb.ClearAllChannels();

	// Resample data. 
	// The two vectors here represent the input and output value ranges.
	// During ConvertToFloat we already divided the source uchar by 256, so input data is already [0,1]. 
	// Output data, stored in GVDB, will also be [0,1]. Third value is reserved for future (ignored).
	printf("Resample.\n");
	gvdb.Resample(0, xform, m_DataRes, AUX_DATA3D, Vector3DF(0, 1, 0), Vector3DF(0, 1, 0));
}

bool Sample::init() 
{
	int w = getWidth(), h = getHeight();			// window width & height
	m_gvdb_tex = -1;
	mouse_down = -1;
	m_show_topo = false;
	m_halo = true;
	m_sparse = false;	
	m_DataBuf = 0;

	init2D("arial");

	// Initialize GVDB
	int devid = -1;
	gvdb.SetVerbose ( true );
	gvdb.SetCudaDevice ( devid );
	gvdb.Initialize ();
	gvdb.AddPath ( "../source/shared_assets/" );
	gvdb.AddPath ( ASSET_PATH );

	gvdb.StartRasterGL();

	// Load RAW in CPU memory
	char scnpath[1024];
	printf("Loading volume data.\n");	
	if ( !gvdb.getScene()->FindFile ( "head.raw", scnpath ) ) {
		nvprintf ( "Cannot find pvm file.\n" );
		nverror();
	}
	printf ( "Loading RAW. %s\n", scnpath );			
	LoadRAW ( scnpath, Vector3DI(128,256,256), 1 );		// This sets m_DataRes and m_DataBuf

	// Convert data to float
	printf("Convert to float.\n");
	ConvertToFloat( m_DataRes, (uchar*) m_DataBuf );

	// Transfer source data to GPU
	printf("Transfer data to GPU.\n");
	gvdb.PrepareAux( AUX_DATA3D, m_DataRes.x*m_DataRes.y*m_DataRes.z, sizeof(float), false, false );
	DataPtr& aux3D = gvdb.getAux(AUX_DATA3D);
	gvdb.SetDataCPU( aux3D, m_DataRes.x*m_DataRes.y*m_DataRes.z, m_DataBuf, 0, sizeof(float) );
	gvdb.CommitData(aux3D);

	// Rebuild the data in GVDB	
	m_VolMax = Vector3DF(256, 256, 256);
	Rebuild ( m_VolMax, m_sparse, m_halo );

	// Set volume params
	gvdb.getScene()->SetSteps ( .5f, 16, .5f );				// Set raycasting steps
	gvdb.getScene()->SetExtinct ( -1.0f, 1.5f, 0.0f );		// Set volume extinction
	gvdb.getScene()->SetVolumeRange ( 0.05f, 0.0f, 1.f );	// Set volume value range
	gvdb.getScene()->SetCutoff ( 0.001f, 0.001f, 0.0f );
	gvdb.getScene()->SetBackgroundClr ( 0.1f, 0.2f, 0.4f, 1.0 );
	gvdb.getScene()->LinearTransferFunc ( 0.00f, 0.15f, Vector4DF(0,0,0,0), Vector4DF(0, 0, 0, 0.0f));		
	gvdb.getScene()->LinearTransferFunc ( 0.15f, 0.25f, Vector4DF(0,0,0,0), Vector4DF(0,0,1,0.01f) );			// skin range, blue
	gvdb.getScene()->LinearTransferFunc ( 0.25f, 0.50f, Vector4DF(0,0,1,0.01f), Vector4DF(1,0,0,0.02f) );		// bone range, red
	gvdb.getScene()->LinearTransferFunc ( 0.50f, 0.75f, Vector4DF(1,0,0,0.02f), Vector4DF(.2f,.2f,0.2f,0.02f) );
	gvdb.getScene()->LinearTransferFunc ( 0.75f, 1.00f, Vector4DF(.2f,.2f,0.2f,0.02f), Vector4DF(.1f,.1f,.1f,.1f) );
	gvdb.CommitTransferFunc ();
	gvdb.SetEpsilon(0.01f, 256); // Use a larger value of epsilon than the default (0.001) to avoid artifacts between bricks

	// Create Camera 
	Camera3D* cam = new Camera3D;						
	cam->setFov ( 50.0 );
	cam->setNearFar(.1f, 5000);
	cam->setOrbit ( Vector3DF(90,20,0), Vector3DF(128,100,128), 1000, 1.0 );	
	gvdb.getScene()->SetCamera( cam );
	
	// Create Light
	Light* lgt = new Light;								
	lgt->setOrbit ( Vector3DF(299,57.3f,0), Vector3DF(0,0,0), 1400, 1.0 );
	gvdb.getScene()->SetLight ( 0, lgt );	

	// Add render buffer
	nvprintf ( "Creating screen buffer. %d x %d\n", w, h );
	gvdb.AddRenderBuf ( 0, w, h, 4 );

	// Create opengl texture for display
	// This is a helper func in sample utils (not part of gvdb),
	// which creates or resizes an opengl 2D texture.
	createScreenQuadGL ( &m_gvdb_tex, w, h );

	start_guis(w, h);

	return true; 
}

void Sample::reshape (int w, int h)
{
	// Resize the opengl screen texture
	glViewport(0, 0, w, h);
	createScreenQuadGL ( &m_gvdb_tex, w, h );

	// Resize the GVDB render buffers
	gvdb.ResizeRenderBuf ( 0, w, h, 4 );

	// Resize 2D UIs
	start_guis(w, h);

	postRedisplay();
}


void Sample::draw_topology()
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


void Sample::display() 
{
	int w = getWidth(), h = getHeight();			// window width & height
	
	clearScreenGL();

	// Render volume
	gvdb.TimerStart ();
	gvdb.Render( SHADE_VOLUME, 0, 0 );    // last value indicates render buffer for depth input
	//gvdb.Render( SHADE_VOLUME, 0, 0 );    // last value indicates render buffer for depth input
	float rtime = gvdb.TimerStop();
	nvprintf ( "Render volume. %6.3f ms\n", rtime );

	// Copy GVDB output buffer into OpenGL texture	
	gvdb.ReadRenderTexGL ( 0, m_gvdb_tex );

	// Render to screen
	renderScreenQuadGL( m_gvdb_tex );

	if (m_show_topo) draw_topology();			// Draw GVDB topology 

	draw3D();									// Render the 3D drawing groups
	drawGui(0);
	draw2D();
}

void Sample::keyboardchar(unsigned char key, int mods, int x, int y)
{
	switch (key) {
	case '1':	m_show_topo = !m_show_topo;	break;
	case '2':	m_sparse = !m_sparse; Rebuild(); break;
	case '3':	m_halo = !m_halo; Rebuild(); break;
	};
}


void Sample::motion(int x, int y, int dx, int dy) 
{
	// Get camera for GVDB Scene
	Camera3D* cam = gvdb.getScene()->getCamera();
	Light* lgt = gvdb.getScene()->getLight();
	bool shift = (getMods() & NVPWindow::KMOD_SHIFT);		// Shift-key to modify light

	switch (mouse_down) {
	case NVPWindow::MOUSE_BUTTON_LEFT: {
		// Adjust orbit angles
		Vector3DF angs = (shift ? lgt->getAng() : cam->getAng());
		angs.x += dx*0.2f;
		angs.y -= dy*0.2f;
		if (shift)	lgt->setOrbit(angs, lgt->getToPos(), lgt->getOrbitDist(), lgt->getDolly());
		else		cam->setOrbit(angs, cam->getToPos(), cam->getOrbitDist(), cam->getDolly());
		postRedisplay();
	} break;

	case NVPWindow::MOUSE_BUTTON_MIDDLE: {
		// Adjust target pos		
		cam->moveRelative(float(dx) * cam->getOrbitDist() / 1000, float(-dy) * cam->getOrbitDist() / 1000, 0);
		postRedisplay();
	} break;

	case NVPWindow::MOUSE_BUTTON_RIGHT: {
		// Adjust dist
		float dist = (shift ? lgt->getOrbitDist() : cam->getOrbitDist());
		dist -= dy;
		if (shift)	lgt->setOrbit(lgt->getAng(), lgt->getToPos(), dist, cam->getDolly());
		else		cam->setOrbit(cam->getAng(), cam->getToPos(), dist, cam->getDolly());
		postRedisplay();
	} break;
	}
}

void Sample::mouse(NVPWindow::MouseButton button, NVPWindow::ButtonAction state, int mods, int x, int y)
{
	if (guiHandler(button, state, x, y)) return;

	// Track when we are in a mouse drag
	mouse_down = (state == NVPWindow::BUTTON_PRESS) ? button : -1;	
}

int sample_main ( int argc, const char** argv ) 
{
	return sample_obj.run ( "NVIDIA(R) GVDB Voxels - gResample", "resample", argc, argv, 1024, 768, 4, 5 );
}

void sample_print( int argc, char const *argv)
{
}



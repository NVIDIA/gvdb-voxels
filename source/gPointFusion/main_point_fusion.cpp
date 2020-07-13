//#define USE_DENSE

// GVDB library
#include "gvdb.h"			
using namespace nvdb;

// Sample utils
#include "main.h"			// window system 
#include "nv_gui.h"			// gui system
#include <GL/glew.h>
#include <fstream> 

#include "string_helper.h"

#define GRID_X		10
#define GRID_Y		10
#define GRID_CNT	(GRID_X*GRID_Y)
#define GRID_BMAX	(GRID_CNT*100)
#define GRID_SCALE	5.0f

VolumeGVDB	gvdb;

#ifdef USE_OPTIX
	// OptiX scene
	#include "optix_scene.h"
	OptixScene  optx;
#endif

struct ALIGN(16) Obj {
	Vector3DF	pos;
	Vector3DF	size;
	Vector3DF	loc;
	uint		clr;
};
struct ALIGN(16) ScanInfo {
	CUdeviceptr	objGrid;
	CUdeviceptr	objCnts;
	CUdeviceptr	objList;
	CUdeviceptr pntList;
	CUdeviceptr pntClrs;
	Vector3DI	gridRes;
	Vector3DF	gridSize;		
	Vector3DF	cams;
	Vector3DF	camu;
	Vector3DF	camv;
	CUdeviceptr rnd_seeds;
};

class Sample : public NVPWindow {
public:
	Sample();
	virtual bool init();
	virtual void display();
	virtual void reshape(int w, int h);
	virtual void motion(int x, int y, int dx, int dy);
	virtual void keyboardchar(unsigned char key, int mods, int x, int y);
	virtual void mouse (NVPWindow::MouseButton button, NVPWindow::ButtonAction state, int mods, int x, int y);

	void		SetupGVDB();
	void		GenerateBuilding ( Vector3DF limits, Vector3DF& bloc, Vector3DF& bpos, Vector3DF& bsz, float block_max, float dctr );
	Vector3DF	LimitBuilding ( int first_obj, Vector3DF bloc );
	Vector3DF	getBuildingPos ( Vector3DF bloc, float block_max, Vector3DF bdim, Vector3DF& bsz);
	Vector3DF	getBuildingPos ( Vector3DF bloc, float block_max );
	void		GenerateCity ();	
	void		ScanBuildings ();
	void		SwitchCamera();
	void		LoadKernel ( int fid, std::string func );	

	void		render_update();
	void		render_frame();	
	void		draw_objects ();	
	void		draw_topology ();	
	void		draw_points ();	
	void		draw_camera ();
	void		start_guis (int w, int h);		
	void		RebuildOptixGraph();
	void		ReportMemory();
	
	float		m_speed;
	float		m_gridsz;
	Vector3DF	m_city_ctr;
	Vector3DI	m_scanres;

	int			m_maxpnts;					// Point list
	int			m_numpnts;		
	DataPtr		m_pnts;	
	DataPtr		m_clrs;

	DataPtr		m_objlist;					// Object list
	DataPtr		m_objgrid;					// Object grid 
	DataPtr		m_objcnts;					// Object grid cnts	
	int			m_objnum;

	CUfunction	m_Func;
	CUmodule	m_Module;
	CUdeviceptr	m_cuScanInfo;
	CUdeviceptr m_cuPntout;
	ScanInfo	m_ScanInfo;	
	DataPtr		m_seeds;

	int			m_w, m_h;
	int			m_radius;
	Vector3DF	m_origin;
	float		m_renderscale;
	int			m_shade_style;
	int			m_frame, m_sample, m_max_samples;
	int			gl_screen_tex;
	int			mouse_down;	
	bool		m_generate;
	bool		m_render_optix;
	bool		m_show_topo;
	bool		m_show_objs;
	bool		m_show_points;
	bool		m_show_pov;
	bool		m_use_color;
	std::string m_mem, m_vox, m_ext, m_pt;
	long		m_totalpnts;

	Camera3D	m_carcam;
};

Sample sample_obj;

Sample::Sample()
{
	m_frame = -1;
}

void handle_gui ( int gui, float val )
{
	switch ( gui ) {
	case 8:					// Camera changed
		sample_obj.SwitchCamera();		
		break;
	case 9:					// Color mode changed
		sample_obj.SetupGVDB();
		break;
	};
}
void Sample::SwitchCamera ()
{
	if (!m_show_pov) {
		Camera3D* cam = gvdb.getScene()->getCamera();
		cam->setOrbit(Vector3DF(190, 30, 0), cam->getPos(), cam->getOrbitDist(), cam->getDolly());
	}
}

void Sample::start_guis (int w, int h)
{
	clearGuis();
	setview2D (w, h);
	guiSetCallback ( handle_gui );	
	
	addGui(   10, 10, 240, 18, "Memory", GUI_PRINT, GUI_STR, &m_mem, 0, 1.0f);
	addGui(   10, 30, 240, 18, "Extents", GUI_PRINT, GUI_STR, &m_ext, 0, 1.0f);
	addGui(   10, 50, 240, 18, "Voxels", GUI_PRINT, GUI_STR, &m_vox, 0, 1.0f);	
	addGui(   10, 70, 240, 18, "Points", GUI_PRINT, GUI_STR, &m_pt, 0, 1.0f);

	addGui(   10, h-30, 130, 20, "Scan", GUI_CHECK, GUI_BOOL, &m_generate, 0, 1.0f);
	addGui ( 150, h-30, 130, 20, "Topology", GUI_CHECK, GUI_BOOL, &m_show_topo, 0, 1.0f);
	addGui ( 300, h-30, 130, 20, "Objects", GUI_CHECK, GUI_BOOL, &m_show_objs,	  0, 1.0f );
	addGui ( 450, h-30, 130, 20, "Points", GUI_CHECK, GUI_BOOL, &m_show_points,   0, 1.0f );
	addGui(  600, h-30, 130, 20, "FPV Cam", GUI_CHECK, GUI_BOOL, &m_show_pov, 0, 1.0f);
	addGui(  750, h-30, 130, 20, "Color", GUI_CHECK, GUI_BOOL, &m_use_color, 0, 1.0f);
	
}

void Sample::RebuildOptixGraph ()
{
	char filepath[1024];

	optx.ClearGraph();

	if ( gvdb.FindFile ( "sky.png", filepath) )
		optx.CreateEnvmap ( filepath );

	int m_mat_surf1 = optx.AddMaterial("optix_trace_surface", "trace_surface", "trace_shadow");		
	MaterialParams* matp = optx.getMaterialParams( m_mat_surf1 );
	matp->light_width = 1.2f;
	matp->shadow_width = 0.1f;
	matp->shadow_bias = 0.5f;
	matp->amb_color = Vector3DF(.1f, .1f, .1f);
	matp->diff_color = Vector3DF(.9f, .9f, .9f);
	matp->spec_color = Vector3DF(.3f, .3f, .3f);
	matp->spec_power = 5.0;
	matp->env_color = Vector3DF(0.f, 0.f, 0.f);
	matp->refl_width = 0.3f;
	matp->refl_bias = 0.5f;
	matp->refl_color = Vector3DF(0.5f, 0.5f, 0.5f);
	
	matp->refr_width = 0.0f;
	matp->refr_color = Vector3DF(0.1f, .1f, .1f);
	matp->refr_ior = 1.1f;
	matp->refr_amount = 0.5f;
	matp->refr_offset = 50.0f;
	matp->refr_bias = 0.5f;
	optx.SetMaterialParams( m_mat_surf1, matp );

	// Add GVDB volume to the OptiX scene
	nvprintf("Adding GVDB Volume to OptiX graph.\n");
	Vector3DF volmin = gvdb.getVolMin();
	Vector3DF volmax = gvdb.getVolMax();
	Matrix4F xform = gvdb.getTransform();
	int atlas_glid = gvdb.getAtlasGLID(0);
	optx.AddVolume( atlas_glid, volmin, volmax, xform, m_mat_surf1, 'S' );		

	// Ground polygons
	if ( gvdb.FindFile ( "ground.obj", filepath) ) {
		Model* m;
		gvdb.getScene()->AddModel ( filepath, 10.0, 0, 0, 0 );	
		m = gvdb.getScene()->getModel ( 0 );			
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

Vector3DF Sample::getBuildingPos ( Vector3DF bloc, float block_max )
{
	Vector3DF tmp;
	return getBuildingPos ( bloc, block_max , Vector3DF(0,0,0), tmp);
}

Vector3DF Sample::getBuildingPos ( Vector3DF bloc, float block_max, Vector3DF bdim, Vector3DF& bsz )
{
	Vector3DF p; 

	// place building along edge of city block
	switch ( (int) bloc.y ) {
	case 0:		p.Set ( bloc.x*block_max, 0, 0 );						bsz.Set( bdim.x, bdim.z, bdim.y );		break;
	case 1:		p.Set ( block_max - bdim.y, 0, bloc.x*block_max );		bsz.Set( bdim.y, bdim.z, bdim.x );		break;
	case 2:		p.Set ( (1.0f-bloc.x)*block_max - bdim.x, 0, block_max - bdim.y );  bsz.Set( bdim.x, bdim.z, bdim.y );		break;
	case 3:		p.Set ( 0, 0, (1.0f-bloc.x)*block_max - bdim.x );		bsz.Set( bdim.y, bdim.z, bdim.x );		break;
	};
	return p;
}


void Sample::GenerateBuilding ( Vector3DF limits, Vector3DF& bloc, Vector3DF& bpos, Vector3DF& bsz, float block_max, float dctr ) 
{
	float hgt = 3.0f + 200 * expf ( dctr / -200.0f );

	Vector3DF bdim;
	bdim.Random ( 5, 20, 10, 40, 3, hgt );			// x=along street, y=away from street	
	if ( bloc.x + bdim.x/block_max > 1.0 ) {
		bdim.x = (1 - bloc.x)*block_max;		
	}

	// place building along edge of city block
	bpos = getBuildingPos ( bloc, block_max, bdim, bsz );

	// next building location
	bloc.x += bdim.x / block_max;
	if ( bloc.x >= 1 ) {
		bloc.x = bdim.y / block_max;
		bloc.y++;
	}
}


Vector3DF Sample::LimitBuilding ( int first_obj, Vector3DF bloc ) 
{
	return Vector3DF(0,0,0);
//	for (int n=first_obj; n < m_objs.size(); n++ ) {
//}
}


void Sample::GenerateCity ()
{
	Obj bldg; 
	Vector3DF bloc, bpos, bsize, bclr, blimit;		// building parameters
	Vector3DF block_pos;	
	int bcnt;
	float dist_ctr;
	int first_obj;
	
	float block_sz = 100;					// size of city block in meters	
	float lane_sz = 7.4f;					// width of street (meters)
	float curb_sz = 3.7f;					// width of curb
	m_gridsz = (block_sz + lane_sz*2 + curb_sz*2);

	m_city_ctr = Vector3DF( GRID_X*m_gridsz*0.5f, 0, GRID_Y*m_gridsz*0.5f );

	gvdb.AllocData ( m_objgrid, GRID_CNT, sizeof(int), true );
	gvdb.AllocData ( m_objcnts, GRID_CNT, sizeof(int), true );
	gvdb.AllocData ( m_objlist, GRID_BMAX, sizeof(Obj), true );

	Obj* objlist = (Obj*) m_objlist.cpu;
	m_objnum = 0;
	
	for (int y=0; y < GRID_Y; y++ ) {
		for (int x=0; x < GRID_X; x++ ) {

			block_pos.Set ( x*m_gridsz + lane_sz+curb_sz, 0, y*m_gridsz + lane_sz+curb_sz );			
			bloc.Set ( 0, 0, 0 );
			first_obj = m_objnum;	
			bcnt = 0;

			// Set first object in city block			
			*(int*) (m_objgrid.cpu + (y*GRID_X+x)*sizeof(int)) = first_obj;

			// Generate buildings in block
			while ( bloc.y < 4 && bcnt < 100 ) {

				// generate building
				blimit = LimitBuilding ( first_obj, bloc );				
				dist_ctr = static_cast<float>(m_city_ctr.Dist(block_pos + getBuildingPos(bloc, block_sz)));
				
				GenerateBuilding ( blimit, bloc, bpos, bsize, block_sz, dist_ctr );
				
				// add building
				bclr.Random ( 0, 0.9f, 0, .5f, 0, 0 );				
				bldg.clr = COLORA ( bclr.x, bclr.y, bclr.z, 1 );
				bldg.loc = bloc;
				bldg.pos = (block_pos + bpos) * GRID_SCALE;
				bldg.size = bsize * GRID_SCALE;
				*objlist++ = bldg;
				bcnt++;
			}			
			// generate curb
			bldg.clr = COLORA ( .3, .3, .3, 1 );
			bldg.loc = Vector3DF(0,0,0);
			bldg.pos = (block_pos - Vector3DF(curb_sz, 0, curb_sz )) * GRID_SCALE;
			bldg.size = Vector3DF( block_sz+curb_sz*2, 0.25, block_sz+curb_sz*2 ) * GRID_SCALE;
			*objlist++ = bldg;
			bcnt++;
			
			// generate street
			bldg.clr = COLORA ( .5, .5, .5, 1 );
			bldg.loc = Vector3DF(0,0,0);
			bldg.pos = (block_pos - Vector3DF(lane_sz+curb_sz, 0, lane_sz+curb_sz )) * GRID_SCALE;
			bldg.size = Vector3DF( block_sz+(lane_sz+curb_sz)*2, 0.01f, block_sz+(lane_sz+curb_sz)*2 ) * GRID_SCALE;
			*objlist++ = bldg;
			bcnt++;			

			// Set number of objects in city block			
			*(int*) (m_objcnts.cpu + (y*GRID_X+x)*sizeof(int)) = bcnt;

			m_objnum += bcnt;
		}
	}

	// Commit building data to GPU
	gvdb.CommitData ( m_objgrid );
	gvdb.CommitData ( m_objcnts );
	gvdb.CommitData ( m_objlist );
}


void Sample::ScanBuildings ()
{
	// Set scan info	
	Vector3DF pos = m_carcam.getPos();
	m_ScanInfo.cams = m_carcam.tlRayWorld;
	m_ScanInfo.camu = m_carcam.trRayWorld; m_ScanInfo.camu -= m_ScanInfo.cams;
	m_ScanInfo.camv = m_carcam.blRayWorld; m_ScanInfo.camv -= m_ScanInfo.cams;
	m_ScanInfo.gridRes = Vector3DI(GRID_X, GRID_Y, 0);	
	m_ScanInfo.gridSize = Vector3DF(GRID_X*m_gridsz, GRID_Y*m_gridsz, 0) * GRID_SCALE;
	m_ScanInfo.objGrid = m_objgrid.gpu;
	m_ScanInfo.objCnts = m_objcnts.gpu;
	m_ScanInfo.objList = m_objlist.gpu;
	m_ScanInfo.pntList = m_pnts.gpu;
	m_ScanInfo.pntClrs = m_clrs.gpu;
	m_ScanInfo.rnd_seeds = m_seeds.gpu;
	
	cudaCheck ( cuMemcpyHtoD ( m_cuScanInfo, &m_ScanInfo, sizeof(ScanInfo)), "PointFusion", "ScanBuildings", "cuMemcpyHtoD", "m_ScanInfo", true );

	Vector3DI block ( 16, 16, 1 );
	Vector3DI grid ( int(m_scanres.x/block.x)+1, int(m_scanres.y/block.y)+1, 1 );
	
	// Max number of points per frame
	m_numpnts = m_scanres.x * m_scanres.y;		
	
	float tmax = m_gridsz * 4* GRID_SCALE;

	void* args[4] = { &pos, &m_scanres, &m_objnum, &tmax };

	cudaCheck(cuMemsetD8(m_cuPntout, 0, sizeof(int)), "PointFusion", "ScanBuildings", "cuMemsetD8", "pntout", false);

	cudaCheck( cuLaunchKernel( m_Func, grid.x, grid.y, 1, block.x, block.y, 1, 0, NULL, args, NULL),
		"PointFusion", "ScanBuildings", "cuLaunch", "scanBuildings", true );

	// Count hit points 
	int pntout;
	cudaCheck(cuMemcpyDtoH( &pntout, m_cuPntout, sizeof(int)), "PointFusion", "ScanBuildings", "cuMemsetDtoH", "pntout", false);
	m_totalpnts += pntout;

	if ( m_show_points ) {
		m_pnts.usedNum = m_numpnts; m_pnts.size = sizeof(Vector3DF)*m_numpnts;
		m_clrs.usedNum = m_numpnts; m_clrs.size = sizeof(uint)*m_numpnts;
		gvdb.RetrieveData ( m_pnts );
		gvdb.RetrieveData ( m_clrs );
	}
}

void Sample::LoadKernel ( int fid, std::string func )
{
	char cfn[512];		strcpy ( cfn, func.c_str() );

	cudaCheck ( cuModuleGetFunction ( &m_Func, m_Module, cfn ), "LoadKernel", "cuModuleGetFunction", cfn, "", true );	
}

void Sample::SetupGVDB()
{
	// Configure
	gvdb.Configure(3, 3, 3, 3, 5);
	gvdb.SetChannelDefault(16, 16, 8);
	gvdb.AddChannel(0, T_FLOAT, 1, F_LINEAR);
	gvdb.FillChannel(0, Vector4DF(3, 0, 0, 0));
	if (m_use_color) {
		gvdb.AddChannel(1, T_UCHAR4, 1, F_POINT);
		gvdb.SetColorChannel(1);
		gvdb.FillChannel(1, Vector4DF(0, 0, 0, 0));
	}
	DataPtr a, b;
	gvdb.SetPoints(m_pnts, a, (m_use_color ? m_clrs : b));
}

bool Sample::init() 
{
	m_w = getWidth();			// window width & height
	m_h = getHeight();			
	mouse_down = -1;
	gl_screen_tex = -1;	
	m_show_topo = false;
	m_show_points = false;
	m_show_objs = false;
	m_radius = 1;		
	m_origin = Vector3DF(0,0,0);
	m_shade_style = 5;
	m_generate = true;
	m_show_pov = true;
	m_use_color = false;

	m_speed = 0;
	m_scanres = Vector3DI(240, 180, 0 );	

	m_max_samples = 1;
	m_sample = 0;
	m_frame = 0;
	m_render_optix = false;

	init2D ( "arial" );

	// Initialize Optix Scene
	if (m_render_optix) {
		optx.InitializeOptix(m_w, m_h);
	}

	// Initialize GVDB
	gvdb.SetDebug(true);
	gvdb.SetVerbose(false);
	gvdb.SetProfile(false, true);	
	gvdb.SetCudaDevice( m_render_optix ? GVDB_DEV_CURRENT : GVDB_DEV_FIRST );
	gvdb.Initialize();
	gvdb.StartRasterGL();
	gvdb.AddPath ( ASSET_PATH );

	// Load custom CUDA function
	cudaCheck ( cuModuleLoad ( &m_Module, "point_fusion_cuda.ptx" ), "PointFusion", "LoadKernel", "cuModuleLoad", "point_fusion_cuda.ptx", true );
	LoadKernel ( 0, "scanBuildings" );
	size_t len = 0;
	cudaCheck ( cuModuleGetGlobal ( &m_cuScanInfo, &len, m_Module, "scan" ), "PointFusion", "LoadKernel", "cuModuleGetGlobal", "cuScanInfo", true );
	cudaCheck ( cuModuleGetGlobal ( &m_cuPntout, &len, m_Module, "pntout"), "PointFusion", "LoadKernel", "cuModuleGetGlobal", "pntout", true);
	
	// Default Camera 
	Camera3D* cam = new Camera3D;
	cam->setFov(65.0);
	cam->setNearFar(1, 100000);
	cam->setOrbit ( Vector3DF( 190, 30, 0 ), Vector3DF(0,0,0), 4200, 1.0 );
	gvdb.getScene()->SetCamera(cam);

	// Default Light 
	Light* lgt = new Light;
	lgt->setNearFar(1, 100000);
	lgt->setOrbit(Vector3DF(50, 30, 0), m_city_ctr, 5000, 1.0);
	gvdb.getScene()->SetLight(0, lgt);

	// Default volume params
	gvdb.getScene()->SetSteps(0.5f, 16, 0.5f);			// Set raycasting steps
	gvdb.getScene()->SetExtinct(-1.0f, 1.1f, 0.0f);			// Set volume extinction	
	gvdb.getScene()->SetVolumeRange(0.0f, 3.0f, -1.0f);		// Set volume value range
	gvdb.getScene()->SetCutoff(0.005f, 0.001f, 0.0f);
	gvdb.getScene()->SetShadowParams ( 0.8f, 1, 0 );	
	gvdb.getScene()->SetBackgroundClr(0.1f, 0.2f, 0.3f, 1.0f);

	// Add render buffer
	nvprintf("Output buffer: %d x %d\n", m_w, m_h);	
	gvdb.AddRenderBuf(0, m_w, m_h, 4);

	// Resize window
	resize_window ( m_w, m_h );	

	// Create opengl texture for display
	glViewport ( 0, 0, m_w, m_h );
	createScreenQuadGL ( &gl_screen_tex, m_w, m_h );

	// Allocate points
	m_maxpnts = 1000000;
	m_numpnts = 0;
	gvdb.AllocData ( m_pnts, m_maxpnts, sizeof(Vector3DF), true );
	gvdb.AllocData ( m_clrs, m_maxpnts, sizeof(uint), true );	

	// Setup GVDB topology & channels
	SetupGVDB();

	// Random seeds (pixel scan sampling)
	gvdb.AllocData ( m_seeds, 128*128, sizeof(uint), true );
	int* sd = (int*) m_seeds.cpu;
	srand ( 1043 );
	for (int n=0; n < 128*128; n++)
		*sd++ = rand()/RAND_MAX;
	gvdb.CommitData ( m_seeds );

	// Generate city
	GenerateCity ();

	// Setup car camera
	m_carcam.setNearFar(1, 100000);
	m_carcam.setFov ( 90 );
	m_carcam.setDist ( 1400 );		
	m_carcam.setPos ( 2*m_gridsz*GRID_SCALE, 4*GRID_SCALE, 2*m_gridsz*GRID_SCALE );	
	m_carcam.setAngles ( 180, 0, 0);
	
	lgt->setOrbit(Vector3DF(42, 40, 0), m_carcam.getPos(), 2000*GRID_SCALE, 1.0);

	// Initialize GUIs
	start_guis ( m_w, m_h );

	// Rebuild the Optix scene graph with GVDB
	if (m_render_optix)	
		RebuildOptixGraph();
		
	return true; 
}

void Sample::reshape (int w, int h)
{
	// Resize the opengl screen texture
	glViewport(0, 0, w, h);
	createScreenQuadGL ( &gl_screen_tex, w, h );

	// Resize the GVDB render buffers
	gvdb.ResizeRenderBuf ( 0, w, h, 4 );

	if (m_render_optix) optx.ResizeOutput ( w, h );

	// Resize 2D UI
	start_guis(w, h);

	postRedisplay();
}
void Sample::render_update()
{
	// Rebuild GVDB Render topology
	
	PERF_PUSH("Dynamic Topology");	
	gvdb.AccumulateTopology( m_numpnts, m_radius*2.0f, m_origin );
	gvdb.FinishTopology(false, true);	// false. no commit pool	false. no compute bounds
	gvdb.UpdateAtlas();
	PERF_POP();

	// Gather points to level set
	PERF_PUSH("Points-to-Voxels");
	

	int scPntLen = 0;
	int subcell_size = 4;
	gvdb.InsertPointsSubcell (subcell_size, m_numpnts, m_radius*2.0f, m_origin, scPntLen);
	gvdb.GatherLevelSet (subcell_size, m_numpnts, float(m_radius), m_origin, scPntLen, 0, 1, true );		// true = accumulate
	gvdb.UpdateApron(0, 0.0f);	
	PERF_POP();

	if (m_render_optix) {
		PERF_PUSH("Update OptiX");
		optx.UpdateVolume(&gvdb);			// GVDB topology has changed
		PERF_POP();
	}

	char buf[1024];
	Vector3DF ext, vox, used, free;
	gvdb.getUsage(ext, vox, used, free);
	sprintf(buf, "%6.0f / %6.0f MB (%4.3f%%)", free.y-free.x, free.y, (free.y-free.x)*100.0 / free.y); m_mem = buf;	
	sprintf(buf, "%d x %d x %d", (int)ext.x, (int)ext.y, (int)ext.z); m_ext = buf;
	sprintf(buf, "%d brk, %dM vox, %4.3f%%", (int)vox.x, (int)vox.y, vox.z); m_vox = buf;
	sprintf(buf, "%6.2fM pnts", float(m_totalpnts) / 1000000.0f ); m_pt = buf;
	
}

void Sample::render_frame()
{
	// Render frame
	gvdb.getScene()->SetCrossSection(m_origin, Vector3DF(0, 0, -1));

	int sh;
	switch (m_shade_style) {
	case 0: sh = SHADE_OFF;			break;
	case 1: sh = SHADE_VOXEL;		break;
	case 2: sh = SHADE_TRILINEAR;	break;
	case 3: sh = SHADE_SECTION3D;	break;
	case 4: sh = SHADE_VOLUME;		break;
	case 5: sh = SHADE_LEVELSET;	break;
	};
	
	if (m_render_optix) {
		// OptiX render
		PERF_PUSH("Raytrace");
		optx.Render( &gvdb, sh, 0);
		PERF_POP();
		PERF_PUSH("ReadToGL");
		optx.ReadOutputTex(gl_screen_tex);
		PERF_POP();
	}
	else {
		// CUDA render
		PERF_PUSH("Raytrace");
		gvdb.Render(sh, 0, 0);
		PERF_POP();
		PERF_PUSH("ReadToGL");
		gvdb.ReadRenderTexGL(0, gl_screen_tex);
		PERF_POP();
	}

	renderScreenQuadGL(gl_screen_tex);		// Render screen-space quad with texture 	
}

void Sample::draw_camera ()
{
	Camera3D* cam = gvdb.getScene()->getCamera();	
	start3D ( cam );		// start 3D drawing

	Vector3DF p[8];
	p[0] = m_carcam.from_pos;
	p[1] = p[0] + Vector3DF(m_carcam.tlRayWorld)*0.005f;
	p[2] = p[0] + Vector3DF(m_carcam.trRayWorld)*0.005f;
	p[3] = p[0] + Vector3DF(m_carcam.blRayWorld)*0.005f;
	p[4] = p[0] + Vector3DF(m_carcam.brRayWorld)*0.005f;

	drawLine3D ( p[0].x, 0, p[0].z, p[0].x, p[0].y, p[0].z, .5, .5, .5, 1 );
	drawLine3D ( p[0].x,p[0].y,p[0].z, p[1].x, p[1].y, p[1].z, 1, 1, 0, 1 );
	drawLine3D ( p[0].x,p[0].y,p[0].z, p[2].x, p[2].y, p[2].z, 1, 1, 0, 1 );
	drawLine3D ( p[0].x,p[0].y,p[0].z, p[3].x, p[3].y, p[3].z, 1, 1, 0, 1 );
	drawLine3D ( p[0].x,p[0].y,p[0].z, p[4].x, p[4].y, p[4].z, 1, 1, 0, 1 );

	end3D ();
}

void Sample::draw_objects ()
{
	Camera3D* cam = gvdb.getScene()->getCamera();		
	Vector3DF p1, p2;
	uint c;
	Obj* objlist = (Obj*) m_objlist.cpu;

	start3D ( cam );		// start 3D drawing
	
	for (int n=0; n < m_objnum; n++ ) {		
		p1 = objlist->pos;
		p2 = p1 + objlist->size;
		c = objlist->clr;
		
		drawBox3D ( p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, RED(c), GRN(c), BLUE(c), 1 );		

		objlist++;
	}
	end3D ();
}


void Sample::draw_points ()
{
	Camera3D* cam = gvdb.getScene()->getCamera();	
	Vector3DF*	fpos = (Vector3DF*) m_pnts.cpu;	
	uint*		fclr = (uint*) m_clrs.cpu;
	Vector3DF p1, p2;
	uint c;
	
	start3D ( gvdb.getScene()->getCamera() );		// start 3D drawing
	for (int n=0; n < m_numpnts; n++ ) {
		p1 = *fpos++; p2 = p1 + Vector3DF(.05f,.05f,.05f);
		c =  *fclr++;				
		drawBox3D ( p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, RED(c), GRN(c), BLUE(c), 1 );
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
			if (node->mFlags == 0) continue;

			Vector3DF bmin = gvdb.getWorldMin(node); // get node bounding box
			Vector3DF bmax = gvdb.getWorldMax(node); // draw node as a box
			drawBox3DXform(bmin, bmax, color, xform);
		}
	}

	end3D();										// end 3D drawing
}


void Sample::display() 
{
	// Update sample convergence
	if (m_render_optix) optx.SetSample(m_frame, m_sample);
	clearScreenGL();

	// Render frame
	render_frame();

	// Move car		
	m_carcam.moveRelative(0, 0, float(-m_speed));
	Camera3D* cam = gvdb.getScene()->getCamera();
	Vector3DF pos = m_carcam.getPos();
	if (m_show_pov) {
		cam->setPos(pos.x, pos.y, pos.z);
		cam->setAngles(m_carcam.getAng().x, m_carcam.getAng().y, m_carcam.getAng().z);
	}

	if (m_generate) {		

		// Scan buildings
		ScanBuildings ();

		render_update ();

		m_frame++;		
		m_sample = 1;

	} else {		
		
		m_sample++;	
		
	}
	
	glDisable(GL_DEPTH_TEST);
	glClearDepth(1.0);
	glClear(GL_DEPTH_BUFFER_BIT);

	if ( m_show_objs)	draw_objects ();
	if ( m_show_points) draw_points ();
	if ( m_show_topo )	draw_topology ();
	if ( !m_show_pov )	draw_camera ();

	draw3D();
	drawGui(0);
	draw2D();

	postRedisplay();								// Post redisplay since simulation is continuous	
}

void Sample::motion(int x, int y, int dx, int dy) 
{
	// Get camera for GVDB Scene
	Camera3D* cam = gvdb.getScene()->getCamera();	
	Light* lgt = gvdb.getScene()->getLight();
	bool shift = (getMods() & NVPWindow::KMOD_SHIFT);		// Shift-key to modify light
	Vector3DF angs = (shift ? m_carcam.getAng() : cam->getAng());

	switch ( mouse_down ) {	
	case NVPWindow::MOUSE_BUTTON_LEFT: {
		// Adjust orbit angles		
		angs.x += dx*0.2f;
		angs.y -= dy*0.2f;

		if ( shift || m_show_pov )	{			
			m_carcam.setAngles(angs.x, angs.y, angs.z);							
		} else {
			cam->setOrbit ( angs, cam->getToPos(), cam->getOrbitDist(), cam->getDolly() );				
		}
		m_sample = 0;
		} break;
	
	case NVPWindow::MOUSE_BUTTON_MIDDLE: {	
		if (shift || m_show_pov) {
			Vector3DF p = m_carcam.getPos();
			p.y += dy;
			if (p.y < 0) p.y = 0;
			if (p.y > 100 * GRID_SCALE) p.y = 100 * GRID_SCALE;
			m_carcam.setPos(p.x, p.y, p.z);
			m_carcam.setAngles(angs.x, angs.y, angs.z);
			m_speed = 0;
		} else {
			cam->moveRelative(float(dx) * cam->getOrbitDist() / 1000, float(-dy) * cam->getOrbitDist() / 1000, 0);
		}
		m_sample = 0;
		} break;
	
	case NVPWindow::MOUSE_BUTTON_RIGHT: {		
		if ( shift || m_show_pov ) {
			m_speed += dy * 0.1f;
			if (m_speed < 0) m_speed = 0;			
		} else {
			float dist = cam->getOrbitDist() - dy;
			cam->setOrbit ( cam->getAng(), cam->getToPos(), dist, cam->getDolly() );			
		}
		m_sample = 0;
		} break;
	}
	if (m_sample == 0) {
		nvprintf("cam ang: %f %f %f\n", cam->getAng().x, cam->getAng().y, cam->getAng().z);
		nvprintf("cam dst: %f\n", cam->getOrbitDist() );
		nvprintf("cam to:  %f %f %f\n", cam->getToPos().x, cam->getToPos().y, cam->getToPos().z );
		nvprintf("lgt ang: %f %f %f\n\n", lgt->getAng().x, lgt->getAng().y, lgt->getAng().z);		
	}
}


void Sample::keyboardchar(unsigned char key, int mods, int x, int y)
{
	switch ( key ) {	
	case '1':	m_show_points   = !m_show_points;	break;	
	case '2':	m_show_topo		= !m_show_topo;		break;	
	case ' ':	
		m_generate		= !m_generate;		
		if (!m_generate) m_speed = 0;
		break;
	};
}

void Sample::mouse ( NVPWindow::MouseButton button, NVPWindow::ButtonAction state, int mods, int x, int y)
{
	if ( guiHandler ( button, state, x, y ) ) return;
	mouse_down = (state == NVPWindow::BUTTON_PRESS) ? button : -1;	
}

int sample_main ( int argc, const char** argv ) 
{
	return sample_obj.run ( "GVDB Sparse Volumes - gPointFusion Sample", "pointfusion", argc, argv, 1024, 768, 4, 5, 120 );
}

void sample_print( int argc, char const *argv)
{
}


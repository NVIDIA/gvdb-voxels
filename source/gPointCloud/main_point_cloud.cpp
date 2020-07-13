// GVDB library
#include "gvdb.h"			
using namespace nvdb;

// Sample utils
#include "main.h"			// window system 
#include "nv_gui.h"			// gui system
#include <GL/glew.h>
#include <fstream> 

#include "string_helper.h"

VolumeGVDB	gvdb;

#ifdef USE_OPTIX
	// OptiX scene
	#include "optix_scene.h"
	OptixScene  optx;
#endif

struct PolyModel {
	char		fpath[1024];
	char		fname[1024];
	int			mat;
	float		scal;
	Vector3DF	offs;
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
	virtual void on_arg(std::string arg, std::string val);

	void		parse_scene ( std::string fname );
	void		parse_value ( int mode, std::string tag, std::string val );
	void		add_material ( bool bDeep );
	void		add_model ();
	void		load_points ( std::string pntpath, std::string pntfile, int frame );
	void		load_polys ( std::string polypath, std::string polyfile, int frame, float pscale, Vector3DF poffs, int pmat );	
	void		clear_gvdb();
	void		render_update();
	void		render_frame();	
	void		draw_points ();
	void		draw_topology ();	// draw gvdb topology
	void		start_guis (int w, int h);	
	void		ClearOptix () ;
	void		RebuildOptixGraph(int shading);
	void		ReportMemory();


	int			m_radius;
	Vector3DF	m_origin;
	float		m_renderscale;

	int			m_w, m_h;
	int			m_numpnts;	
	DataPtr		m_pnt1;
	DataPtr		m_pnts;	
	int			m_frame;
	int         m_fstep;
	int			m_sample;
	int			m_max_samples;
	int			m_shade_style;
	int			gl_screen_tex;
	int			mouse_down;	
	bool		m_render_optix;
	bool		m_show_points;
	bool		m_show_topo;
	bool		m_save_png;	

	int			m_smooth;
	Vector3DF	m_smoothp;

	bool		m_pnton;		// point time series
	std::string	m_pntpath;
	std::string m_pntfile;
	int			m_pntmat;

	bool		m_polyon;		// polygon time series
	std::string m_polypath;
	std::string m_polyfile;
	int			m_polymat;
	int			m_pframe, m_pfstep;
	float		m_pscale;
	Vector3DF	m_poffset;

	std::string m_infile;
	std::string m_envfile;
	std::string m_outpath;
	std::string m_outfile;
	
	std::vector<MaterialParams>	mat_list;
	std::vector<PolyModel>		model_list;

	bool		m_key;
	bool		m_info;
};

Sample sample_obj;

void handle_gui ( int gui, float val )
{
	switch ( gui ) {
	case 3:	{				// Shading gui changed
		float alpha = (val==4) ? 0.03f : 0.8f;		// when in volume mode (#4), make volume very transparent
		gvdb.getScene()->LinearTransferFunc ( 0.00f, 0.50f,  Vector4DF(0,0,1,0), Vector4DF(0.0f,1,0, 0.1f) );
		gvdb.getScene()->LinearTransferFunc ( 0.50f, 1.0f,  Vector4DF(0.0f,1,0, 0.1f), Vector4DF(1.0f,.0f,0, 0.1f) );
		gvdb.CommitTransferFunc ();		
		} break;
	}
}

void Sample::start_guis (int w, int h)
{
	clearGuis();
	setview2D (w, h);
	guiSetCallback ( handle_gui );	
	addGui (  10, h-30, 130, 20, "Points", GUI_CHECK, GUI_BOOL, &m_show_points,	  0, 1.0f );
	addGui ( 150, h-30, 130, 20, "Topology", GUI_CHECK, GUI_BOOL, &m_show_topo,   0, 1.0f );
}

void Sample::add_material ( bool bDeep )
{
	MaterialParams p;
	mat_list.push_back ( p );
}
void Sample::add_model ()
{
	PolyModel p;
	strcpy( p.fpath, "" );
	strcpy( p.fname, "" );
	p.offs = Vector3DF(0,0,0);
	p.scal = 1;
	model_list.push_back (p);
}

void Sample::ClearOptix () 
{
	optx.ClearGraph();
}

void Sample::RebuildOptixGraph(int shading)
{
	char filepath[1024];

	optx.ClearGraph();

	for (int n = 0; n < mat_list.size(); n++) {
		MaterialParams* p = &mat_list[n];
		int id = optx.AddMaterial("optix_trace_surface", "trace_surface", "trace_shadow");		
		optx.SetMaterialParams( id, p );		
	}

	optx.CreateEnvmap("");
	if ( !m_envfile.empty() ) {
		char fname[1024]; strcpy(fname, m_envfile.c_str());
		if ( gvdb.FindFile ( fname, filepath )) {
			nvprintf("Loading env map %s.\n", filepath );
			optx.CreateEnvmap( filepath );
		}
	}

	if ( mat_list.size() == 0 ) {
		nvprintf ( "Error: No materials have been specified in scene.\n" );
		nverror ();	
	}

	/// Add deep volume material
	//mat_surf[1] = optx.AddMaterial("optix_trace_deep", "trace_deep", "trace_shadow");

	// Add GVDB volume to the OptiX scene
	nvprintf("Adding GVDB Volume to OptiX graph.\n");
	char isect;
	switch (shading) {
	case SHADE_TRILINEAR:	isect = 'S';	break;
	case SHADE_VOLUME:		isect = 'D';	break;
	case SHADE_LEVELSET:	isect = 'L';	break;
	case SHADE_EMPTYSKIP:	isect = 'E';	break;
	}
	Vector3DF volmin = gvdb.getVolMin();
	Vector3DF volmax = gvdb.getVolMax();
	Matrix4F xform = gvdb.getTransform();
	int atlas_glid = gvdb.getAtlasGLID(0);
	optx.AddVolume( atlas_glid, volmin, volmax, xform, mat_list[ m_pntmat ].id, isect );		
	
	Model* m;	

	// Add poly time series (optional)	
	if ( m_polyon ) {
		m = gvdb.getScene()->getModel ( 0 );
		nvprintf ( "Adding Polygon time series data.\n" );
		xform.SRT ( Vector3DF(1,0,0), Vector3DF(0,1,0), Vector3DF(0,0,1), Vector3DF(0,0,0), m_renderscale );
		optx.AddPolygons ( m, m_polymat, xform );
	}

	// Add polygonal models
	int id;
	for (int n=0; n < model_list.size(); n++ ) {		
		if ( strlen(model_list[n].fpath)==0)  {
			gvdb.FindFile ( model_list[n].fname, filepath );
		} else {
			sprintf ( filepath, "%s%s", model_list[n].fpath, model_list[n].fname );
		}
		nvprintf ( "Load model %s...", filepath );
		id = static_cast<int>(gvdb.getScene()->AddModel(filepath,
			model_list[n].scal, model_list[n].offs.x, model_list[n].offs.y, model_list[n].offs.z));
		gvdb.CommitGeometry( id );

		m = gvdb.getScene()->getModel ( id );
		xform.Identity ();
		xform.SRT ( Vector3DF(1,0,0), Vector3DF(0,1,0), Vector3DF(0,0,1), Vector3DF(0,0,0), m_renderscale );
		optx.AddPolygons ( m, model_list[n].mat, xform );
		nvprintf ( " Done.\n" );
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


#define M_GLOBAL	0
#define M_RENDER	1
#define M_LIGHT		2
#define M_CAMERA	3
#define M_MODEL		4
#define M_POINTS	5
#define M_POLYS		6
#define M_VOLUME	7
#define M_MATERIAL	8

Sample::Sample()
{
	m_frame = -1;
	m_key = false;
	m_renderscale = 0.0;
	m_infile = "teapot.scn";
}

void Sample::parse_value ( int mode, std::string tag, std::string val )
{
	MaterialParams* matp; 
	Vector3DF vec;

	switch ( mode ) {
	case M_POINTS:
		if (strEq(tag,"path")) m_pntpath = strTrim(val);
		if (strEq(tag,"file")) m_pntfile = strTrim(val);
		if (strEq(tag,"mat")) m_pntmat= static_cast<int>(strToNum(val));
		if (strEq(tag,"frame")) m_frame = static_cast<int>(strToNum(val));
		if (strEq(tag, "fstep")) m_fstep = static_cast<int>(strToNum(val));
		break;
	case M_POLYS:
		if (strEq(tag,"path")) m_polypath = val;
		if (strEq(tag,"file")) m_polyfile = val;
		if (strEq(tag, "mat")) m_polymat = static_cast<int>(strToNum(val));
		if (strEq(tag, "frame")) m_pframe = static_cast<int>(strToNum(val));
		if (strEq(tag, "fstep")) m_pfstep = static_cast<int>(strToNum(val));
		break;
	case M_MATERIAL: {
		int i = static_cast<int>(mat_list.size()-1);
		matp = &mat_list[i];
		if (strEq(tag,"lightwid"))	matp->light_width = strToNum(val);	
		if (strEq(tag,"shwid"))		matp->shadow_width = strToNum(val);
		if (strEq(tag,"shbias"))	matp->shadow_bias = strToNum(val);
		if (strEq(tag,"ambient"))	strToVec3( val, "<",",",">", &matp->amb_color.x );
		if (strEq(tag,"diffuse"))	strToVec3( val, "<",",",">", &matp->diff_color.x );
		if (strEq(tag,"spec"))		strToVec3( val, "<",",",">", &matp->spec_color.x );
		if (strEq(tag,"spow"))		matp->spec_power = strToNum(val);
		if (strEq(tag,"env"))		strToVec3( val, "<",",",">", &matp->env_color.x );
		if (strEq(tag,"reflwid"))	matp->refl_width = strToNum(val);
		if (strEq(tag,"reflbias"))	matp->refl_bias = strToNum(val);
		if (strEq(tag,"reflcolor"))	strToVec3( val, "<",",",">", &matp->refl_color.x );
		if (strEq(tag,"refrwid"))	matp->refr_width = strToNum(val);
		if (strEq(tag,"refrbias"))	matp->refr_bias = strToNum(val);
		if (strEq(tag,"refrcolor"))	strToVec3( val, "<",",",">", &matp->refr_color.x );
		if (strEq(tag,"refroffs"))	matp->refr_offset = strToNum(val);
		if (strEq(tag,"refrior"))	matp->refr_ior = strToNum(val);
		if (strEq(tag,"reframt"))	matp->refr_amount = strToNum(val);		
		} break;
	case M_RENDER: 
		if (strEq(tag,"width"))		m_w = static_cast<int>(strToNum(val));
		if (strEq(tag,"height"))	m_h = static_cast<int>(strToNum(val));
		if (strEq(tag,"samples"))	m_max_samples = static_cast<int>(strToNum(val));		
		if (strEq(tag,"backclr"))	{ strToVec3( val, "<",",",">", &vec.x); gvdb.getScene()->SetBackgroundClr(vec.x,vec.y,vec.z, 1.0); }
		if (strEq(tag,"envmap"))	m_envfile = val;
		if (strEq(tag,"outpath"))	m_outpath = val;
		if (strEq(tag,"outfile"))	m_outfile = val;

		break;
	case M_VOLUME: {
		nvdb::Scene* scn = gvdb.getScene();
		if (strEq(tag,"scale") && m_renderscale==0)  m_renderscale = strToNum(val);
		if (strEq(tag,"steps"))		{ strToVec3( val, "<",",",">", &vec.x); scn->SetSteps(vec.x,vec.y,vec.z); }
		if (strEq(tag,"extinct"))	{ strToVec3( val, "<",",",">", &vec.x); scn->SetExtinct(vec.x,vec.y,vec.z); }
		if (strEq(tag,"range"))		{ strToVec3( val, "<",",",">", &vec.x); scn->SetVolumeRange(vec.x,vec.y,vec.z); }
		if (strEq(tag,"cutoff"))	{ strToVec3( val, "<",",",">", &vec.x); scn->SetCutoff(vec.x,vec.y,vec.z); }
		if (strEq(tag,"smooth"))	m_smooth = static_cast<int>(strToNum(val));
		if (strEq(tag,"smoothp")) { strToVec3(val, "<", ",", ">", &vec.x); m_smoothp = vec; }		
		} break;
	case M_CAMERA: {
		Camera3D* cam = gvdb.getScene()->getCamera();	
		if (strEq(tag,"angs"))		{ strToVec3( val, "<",",",">", &vec.x); cam->setAng(vec); }
		if (strEq(tag, "target")) { strToVec3(val, "<", ",", ">", &vec.x); vec *= m_renderscale;  cam->setToPos(vec.x, vec.y, vec.z); }
		if (strEq(tag,"dist"))		{ cam->setDist ( strToNum(val)*m_renderscale ); }
		if (strEq(tag,"fov"))		{ cam->setFov ( strToNum(val) ); }
		cam->setOrbit ( cam->getAng(), cam->getToPos(), cam->getOrbitDist(), cam->getDolly() );
		} break;
	case M_LIGHT: {
		Light* lgt = gvdb.getScene()->getLight();
		if (strEq(tag,"angs"))		{ strToVec3( val, "<",",",">", &vec.x); lgt->setAng(vec); }
		if (strEq(tag, "target")) { strToVec3(val, "<", ",", ">", &vec.x); vec *= m_renderscale;  lgt->setToPos(vec.x, vec.y, vec.z); }
		if (strEq(tag,"dist"))		{ lgt->setDist ( strToNum(val)*m_renderscale); }
		if (strEq(tag,"fov"))		{ lgt->setFov ( strToNum(val) ); }
		lgt->setOrbit ( lgt->getAng(), lgt->getToPos(), lgt->getOrbitDist(), lgt->getDolly() );
		} break;
	case M_MODEL: {
		int id = static_cast<int>(model_list.size()-1);
		if (strEq(tag,"path"))		strncpy( model_list[id].fpath, val.c_str(), 1024);
		if (strEq(tag,"file"))		strncpy( model_list[id].fname, val.c_str(), 1024);
		if (strEq(tag,"mat"))		model_list[id].mat = static_cast<int>(strToNum(val));
		if (strEq(tag,"scale"))		model_list[id].scal = strToNum(val);
		if (strEq(tag,"offset"))	{ strToVec3( val, "<",",",">", &vec.x);  model_list[id].offs = vec; }
		} break;
	};
}

void Sample::parse_scene ( std::string fname ) 
{
	int mode = M_GLOBAL;

	char fn[1024]; strcpy(fn, fname.c_str());
	char fpath[1024];	
	if (!gvdb.FindFile ( fn, fpath)) {
		printf ( "Error: Cannot find scene file %s\n", fname.c_str() );
	}
	FILE* fp = fopen ( fpath, "rt" );
	char buf[2048];
	std::string lin, tag;
	Vector3DF vec;

	while (!feof(fp)) {
		fgets ( buf, 2048, fp );
		lin = buf;
		
		if ( lin.find("points")==0 )	{ m_pnton = true; mode = M_POINTS; }
		if ( lin.find("polys")==0 )		{ m_polyon = true; mode = M_POLYS; }
		if ( lin.find("light")==0 )		mode = M_LIGHT;
		if ( lin.find("camera")==0 )	mode = M_CAMERA;
		if ( lin.find("global")==0 )	mode = M_GLOBAL;
		if ( lin.find("render")==0 )	mode = M_RENDER;
		if ( lin.find("volume") == 0)	mode = M_VOLUME;
		if ( lin.find("material")==0 )	{ mode = M_MATERIAL; add_material (false); }
		if ( lin.find("model")==0 )		{ mode = M_MODEL; add_model(); }	

		tag = strTrim( strSplit ( lin, ":" ) );
		lin = strTrim( lin );
		if ( tag.length() > 0 && lin.length() > 0 )
			parse_value ( mode, tag, lin );
	}
	fclose ( fp );
}

void Sample::on_arg(std::string arg, std::string val)
{
	if (arg.compare("-in") == 0) {	
		m_infile = val;
		nvprintf("input: %s\n", m_infile.c_str());
	}

	if (arg.compare("-frame") == 0) {		
		m_frame = static_cast<int>(strToNum(val));
		nvprintf("frame: %d\n", m_frame);	
	}

	if (arg.compare("-key") == 0)
		m_key = true;

	if (arg.compare("-info") == 0) {
		nvprintf("print gvdb info\n" );
		m_info = true;
	}

	if (arg.compare("-scale") == 0) {		
		m_renderscale = strToNum(val);
		nvprintf("render scale: %f\n", m_renderscale);
	}
}

bool Sample::init() 
{	
	m_w = getWidth();			// window width & height
	m_h = getHeight();			
	mouse_down = -1;
	gl_screen_tex = -1;	
	m_show_topo = false;
	m_radius = 1;		
	m_origin = Vector3DF(0,0,0);
	m_shade_style = 5;	
	
	m_max_samples = 1;
	m_envfile = "";
	m_outpath = "";
	m_outfile = "img%04d.png";

	m_sample = 0;
	m_save_png = true;
	m_render_optix = true;
	m_smooth = 0;
	m_smoothp.Set(0, 0, 0);

	m_pnton = false;				// point time series
	m_pntmat = 0;
	m_fstep = 0;
	
	m_polyon = false;					// polygonal time series
	m_pframe = 0;
	m_pfstep = 1;
	m_pscale = 1.0;
	m_poffset = Vector3DF(0,0,0);
	m_polymat = 0;

	init2D ( "arial" );

	// Initialize Optix Scene
	if (m_render_optix) {
		optx.InitializeOptix(m_w, m_h);
	}

	gvdb.SetDebug(false);
	gvdb.SetVerbose(false);
	gvdb.SetProfile(false, true);	
	gvdb.SetCudaDevice( m_render_optix ? GVDB_DEV_CURRENT : GVDB_DEV_FIRST );
	gvdb.Initialize();
	gvdb.StartRasterGL();
	gvdb.AddPath ( ASSET_PATH );

	// Default Camera 
	Camera3D* cam = new Camera3D;
	cam->setFov(50.0);
	cam->setNearFar(1, 10000);
	cam->setOrbit(Vector3DF(50, 30, 0), Vector3DF(128, 128, 128), 1400, 1.0);
	gvdb.getScene()->SetCamera(cam);
	
	// Default Light 
	Light* lgt = new Light;
	lgt->setOrbit(Vector3DF(0, 40, 0), Vector3DF(128, 128, 128), 2000, 1.0);
	gvdb.getScene()->SetLight(0, lgt);

	// Default volume params
	gvdb.getScene()->SetSteps(0.25f, 16, 0.25f);			// Set raycasting steps
	gvdb.getScene()->SetExtinct(-1.0f, 1.1f, 0.0f);			// Set volume extinction
	gvdb.getScene()->SetVolumeRange(0.0f, -1.0f, 3.0f);		// Set volume value range
	gvdb.getScene()->SetCutoff(0.005f, 0.001f, 0.0f);
	gvdb.getScene()->SetBackgroundClr(0.1f, 0.2f, 0.4f, 1.0);

	// Parse scene file
	if ( m_render_optix) ClearOptix ();
	parse_scene ( m_infile );

	// Add render buffer
	nvprintf("Output buffer: %d x %d\n", m_w, m_h);	
	gvdb.AddRenderBuf(0, m_w, m_h, 4);

	// Resize window
	resize_window ( m_w, m_h );	

	// Create opengl texture for display
	glViewport ( 0, 0, m_w, m_h );
	createScreenQuadGL ( &gl_screen_tex, m_w, m_h );

	// Configure
	gvdb.Configure(3, 3, 3, 3, 4);
	gvdb.SetChannelDefault(32, 32, 1);
	gvdb.AddChannel( 0, T_FLOAT, 1, F_LINEAR );	
	gvdb.FillChannel( 0, Vector4DF(0, 0, 0, 0) );

	// Initialize GUIs
	start_guis ( m_w, m_h );

	clear_gvdb();

	// Load input data
	if ( m_pnton ) 
		load_points ( m_pntpath, m_pntfile, m_frame );
	if ( m_polyon )
		load_polys ( m_polypath, m_polyfile, m_pframe, m_pscale, m_poffset, m_polymat );

	render_update();

	// Rebuild the Optix scene graph with GVDB
	if (m_render_optix)	
		RebuildOptixGraph( SHADE_LEVELSET );
	
	return true; 
}

void Sample::reshape (int w, int h)
{
	// Resize the opengl screen texture
	glViewport(0, 0, w, h);
	createScreenQuadGL ( &gl_screen_tex, w, h );

	// Resize the GVDB render buffers
	gvdb.ResizeRenderBuf ( 0, w, h, 4 );

	// Resize OptiX buffers
	if (m_render_optix) optx.ResizeOutput ( w, h );

	// Resize 2D UI
	start_guis(w, h);

	postRedisplay();
}

void Sample::load_points ( std::string pntpath, std::string pntfile, int frame )
{
	// Load points	
	char filepath[1024];
	char srcfile[1024];
	char pntfmt[1024];
	sprintf ( pntfmt, "%s%s", m_pntpath.c_str(), m_pntfile.c_str() );
	sprintf ( srcfile, pntfmt, frame );
	
	if ( pntpath.empty() )  {
		gvdb.FindFile ( srcfile, filepath );
	} else {
		sprintf ( filepath, "%s", srcfile );
	}

	nvprintf ( "Load points from %s...", filepath );

	// Read # of points
	m_numpnts = 0;
	Vector3DF wMin, wMax;

	// Read header
	PERF_PUSH ( "  Open file" );
	{
		// Standard read for header info whne using C_IO
		FILE* fph = fopen(filepath, "rb");
		if (fph == 0) {
			printf("Cannot open file: %s\n", filepath);
			exit(-1);
		}
		fread(&m_numpnts, sizeof(int), 1, fph);		// 7*4 = 28 byte header
		fread(&wMin.x, sizeof(float), 1, fph);
		fread(&wMin.y, sizeof(float), 1, fph);
		fread(&wMin.z, sizeof(float), 1, fph);
		fread(&wMax.x, sizeof(float), 1, fph);
		fread(&wMax.y, sizeof(float), 1, fph);
		fread(&wMax.z, sizeof(float), 1, fph);
		fclose(fph);
	}
	PERF_POP ();

	// Read data from disk to CPU	
	{
		// C-style IO
		PERF_PUSH("Read");
		FILE* fph = fopen(filepath, "rb");
		if (fph == 0) {
			printf("Cannot open file: %s\n", filepath);
			exit(-1);
		}
		fread(&m_numpnts, sizeof(int), 1, fph);		// 7*4 = 28 byte header
		fread(&wMin.x, sizeof(float), 1, fph);
		fread(&wMin.y, sizeof(float), 1, fph);
		fread(&wMin.z, sizeof(float), 1, fph);
		fread(&wMax.x, sizeof(float), 1, fph);
		fread(&wMax.y, sizeof(float), 1, fph);
		fread(&wMax.z, sizeof(float), 1, fph);	

		// Allocate memory for points
		PERF_PUSH("  Buffer alloc");
		gvdb.AllocData(m_pnt1, m_numpnts, sizeof(ushort) * 3, true);
		gvdb.AllocData(m_pnts, m_numpnts, sizeof(Vector3DF), true);
		PERF_POP();

		//fseek(fp, 28, SEEK_SET);
		fread (m_pnt1.cpu, 3 * sizeof(ushort), m_numpnts, fph);
		PERF_POP();
		PERF_PUSH("Commit");
		gvdb.CommitData(m_pnt1);		// Commit to GPU
		PERF_POP();
	} 	

	// Convert format and transform
	PERF_PUSH ( "  Convert" );
	Vector3DF wdelta ( (wMax.x - wMin.x)/65535.0f, (wMax.y - wMin.y)/65535.0f, (wMax.z - wMin.z)/65535.0f );
	gvdb.ConvertAndTransform ( m_pnt1, 2, m_pnts, 4, m_numpnts, wMin, wdelta, Vector3DF(0,0,0), Vector3DF(m_renderscale,m_renderscale,m_renderscale) );
	gvdb.RetrieveData(m_pnts); // Copy back to the CPU so that we can locally view it
	PERF_POP ();
	
	// Set points for GVDB	
	DataPtr temp;
	gvdb.SetPoints( m_pnts, temp, temp);
	printf("m_numpnts = %d\n", m_numpnts);
	nvprintf ( "  Done.\n" );
}

void Sample::load_polys ( std::string polypath, std::string polyfile, int frame, float pscale, Vector3DF poffs, int pmat )
{
	bool bFirst = false;
	char fmt[1024], fpath[1024];
	
	Model* m = gvdb.getScene()->getModel(0);
	
	// get filename
	sprintf ( fmt, "%s%s", polypath.c_str(), polyfile.c_str() );
	sprintf ( fpath, fmt, frame );
	
	nvprintf ( "Load polydata from %s...", fpath );

	// create new model if needed
	if ( m == 0x0 ) {
		bFirst = true;
		m = gvdb.getScene()->AddModel ();
	}

	// load model
	gvdb.getScene()->LoadModel ( m, fpath, pscale, poffs.x, poffs.y, poffs.z );
	gvdb.CommitGeometry (0);	

	nvprintf ( " Done.\n" );
}

void Sample::ReportMemory()
{
	std::vector<std::string> outlist;
	gvdb.MemoryUsage("gvdb", outlist);
	for (int n = 0; n < outlist.size(); n++)
		nvprintf("%s", outlist[n].c_str());
}

void Sample::clear_gvdb ()
{
	// Clear
	DataPtr temp;
	gvdb.SetPoints(temp, temp, temp);
	gvdb.CleanAux();
}

void Sample::render_update()
{
	if (!m_pnton) return;	

	// Rebuild GVDB Render topology
	PERF_PUSH("Dynamic Topology");
	//gvdb.RequestFullRebuild ( true );
	gvdb.RebuildTopology(m_numpnts, m_radius*2.0f, m_origin);
	gvdb.FinishTopology(false, true);	// false. no commit pool	false. no compute bounds
	gvdb.UpdateAtlas();
	PERF_POP();

	// Gather points to level set
	PERF_PUSH("Points-to-Voxels");
	gvdb.ClearChannel(0);

	int scPntLen = 0;
	int subcell_size = 4;
	gvdb.InsertPointsSubcell_FP16 (subcell_size, m_numpnts, static_cast<float>(m_radius), m_origin, scPntLen);
	gvdb.GatherLevelSet_FP16 (subcell_size, m_numpnts, static_cast<float>(m_radius), m_origin, scPntLen, 0, 0);
	gvdb.UpdateApron(0, 3.0f);
	PERF_POP();

	if (m_smooth > 0) {
		PERF_PUSH("Smooth");
		nvprintf("Smooth: %d, %f %f %f\n", m_smooth, m_smoothp.x, m_smoothp.y, m_smoothp.z);
		gvdb.Compute( FUNC_SMOOTH, 0, m_smooth, m_smoothp, true, true, 3.0f);		// 8x smooth iterations	
		PERF_POP();
	}

	if (m_render_optix) {
		PERF_PUSH("Update OptiX");
		optx.UpdateVolume(&gvdb);			// GVDB topology has changed
		PERF_POP();
	}

	if (m_info) {
		ReportMemory();
		gvdb.Measure(true);
	}
}

void Sample::render_frame()
{
	// Render frame
	gvdb.getScene()->SetCrossSection(m_origin, Vector3DF(0, 0, -1));

	int sh;
	switch (m_shade_style) {
	case 0: sh = SHADE_OFF;			break;
	case 1: sh = SHADE_VOXEL;		break;
	case 2: sh = SHADE_EMPTYSKIP;	break;
	case 3: sh = SHADE_SECTION3D;	break;
	case 4: sh = SHADE_VOLUME;		break;
	case 5: sh = SHADE_LEVELSET;	break;
	};
	
	if (m_render_optix) {
		// OptiX render
		PERF_PUSH("Raytrace");
		optx.Render( &gvdb, SHADE_LEVELSET, 0);
		PERF_POP();
		PERF_PUSH("ReadToGL");
		optx.ReadOutputTex(gl_screen_tex);
		PERF_POP();
	}
	else {
		// CUDA render
		PERF_PUSH("Raytrace");
		gvdb.Render( sh, 0, 0);
		PERF_POP();
		PERF_PUSH("ReadToGL");
		gvdb.ReadRenderTexGL(0, gl_screen_tex);
		PERF_POP();
	}
	renderScreenQuadGL(gl_screen_tex);		// Render screen-space quad with texture 	
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


void Sample::draw_points ()
{
	Vector3DF*	fpos = (Vector3DF*) m_pnts.cpu;

	Vector3DF p1, p2;
	Vector3DF c;

	Camera3D* cam = gvdb.getScene()->getCamera();
	start3D(cam);
	for (int n=0; n < m_numpnts; n++ ) {
		p1 = *fpos++; 
		p2 = p1+Vector3DF(0.01f,0.01f,0.01f);		
		c =  p1 / Vector3DF(256.0,256,256);
		drawLine3D ( p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, c.x, c.y, c.z, 1);		
	}
	end3D();
}

Vector3DF interp(Vector3DF a, Vector3DF b, float t)
{
	return Vector3DF(a.x + t*(b.x - a.x), a.y + t*(b.y - a.y), a.z + t*(b.z - a.z));
}

void Sample::display() 
{
	// Update sample convergence
	if (m_render_optix) optx.SetSample(m_frame, m_sample);

	if (m_key && m_frame >= 1100 && m_frame <= 1340 ) {
		float u = float(m_frame - 1100) / (1340 - 1100);
		Vector3DF a0, t0, d0; 
		a0 = interp(Vector3DF(0,35,0), Vector3DF(0, 24.6f,0), u);
		t0 = interp(Vector3DF(240, 0, 0), Vector3DF(362, -201, 20), u);
		d0 = interp(Vector3DF(1600, 0, 0), Vector3DF(2132, 0, 0), u);
		Camera3D* cam = gvdb.getScene()->getCamera();
		cam->setOrbit(a0, t0*m_renderscale, d0.x*m_renderscale, cam->getDolly());
	}

	clearScreenGL();

	// Render frame
	render_frame();

	if (m_sample % 8 == 0 && m_sample > 0) {
		int pct = (m_sample * 100) / m_max_samples;
		nvprintf("%d%%%% ", pct);
	}

	if (++m_sample >= m_max_samples) {
		m_sample = 0;

		nvprintf("100%%%%\n");

		if (m_save_png && m_render_optix) {
			// Save current frame to PNG
			char png_name[1024];
			char pfmt[1024];
			sprintf(pfmt, "%s%s", m_outpath.c_str(), m_outfile.c_str());
			sprintf(png_name, pfmt, m_frame);
			std::cout << "Save png to " << png_name << " ...";
			optx.SaveOutput(png_name);
			std::cout << " Done\n";
		}

		m_frame += m_fstep;

		if ( m_pnton )  
			load_points ( m_pntpath, m_pntfile, m_frame );
		
		if ( m_polyon ) {
			m_pframe += m_pfstep;
			load_polys ( m_polypath, m_polyfile, m_pframe, m_pscale, m_poffset, m_polymat );			
			if (m_render_optix) optx.UpdatePolygons ();			
		}
		render_update();		
	}
	
	glDisable(GL_DEPTH_TEST);
	glClearDepth(1.0);
	glClear(GL_DEPTH_BUFFER_BIT);

	if ( m_show_points) draw_points ();
	if ( m_show_topo ) draw_topology ();

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
	};
}

void Sample::mouse ( NVPWindow::MouseButton button, NVPWindow::ButtonAction state, int mods, int x, int y)
{
	if ( guiHandler ( button, state, x, y ) ) return;
	mouse_down = (state == NVPWindow::BUTTON_PRESS) ? button : -1;	
}

int sample_main ( int argc, const char** argv ) 
{
	return sample_obj.run ( "GVDB Sparse Volumes - gPointCloud Sample", "pointcloud", argc, argv, 1280, 760, 4, 5, 30 );
}

void sample_print( int argc, char const *argv)
{
}


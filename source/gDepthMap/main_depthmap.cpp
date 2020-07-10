

// GVDB library
#include "gvdb.h"			
#include "gvdb_render.h"	// OpenGL rendering
using namespace nvdb;

// Sample utils
#include "main.h"			// window system 
#include <GL/glew.h>

VolumeGVDB	gvdb;

class Sample : public NVPWindow {
public:
	virtual bool init();
	virtual void display();
	virtual void reshape(int w, int h);
	virtual void motion(int x, int y, int dx, int dy);
	virtual void mouse (NVPWindow::MouseButton button, NVPWindow::ButtonAction state, int mods, int x, int y);

	void	prepare_depth(int w, int h);
	void	resize_depth(int w, int h );
	void	render_depth(int w, int h);
	void	update_camera_gl(int w, int h);

	int		mFBO;				// Framebuffer object
	int		mDepthInputID;		// GVDB render buffer being used for depth input

	int		m_depth_tex;		// Texture holding GL depth buffer for polygons
	int		m_polys_tex;		// Texture holding GL color buffer for polygons
	int     m_gvdb_tex;			// Texture holding GL color /w alpha for GVDB volume 

	Vector3DF	m_translate;
	Vector3DF	m_rotate;

	bool	m_use_opengl_cam;
	int		mouse_down;

	float	m_bias;
};



bool Sample::init() 
{
	int w = getWidth(), h = getHeight();			// window width & height
	mFBO = -1;
	m_bias = 1.0;
	m_depth_tex = -1;
	m_polys_tex = -1;
	m_gvdb_tex = -1;
	mouse_down = -1;
	m_use_opengl_cam = false;
	m_translate = Vector3DF(0, 0, -25);
	m_rotate = Vector3DF(0, 0, 0);

	// Initialize GVDB	
	gvdb.SetVerbose ( true );
	gvdb.SetCudaDevice ( GVDB_DEV_FIRST );
	gvdb.Initialize ();	
	gvdb.AddPath ( ASSET_PATH );

	gvdb.StartRasterGL();

	// Load polygons
	// This loads an obj file into scene memory on cpu.
	char scnpath[1024];
	printf("Loading polygon model.\n");
	gvdb.getScene()->AddModel("lucy.obj", 100.0, 0, 0, 0);
	gvdb.CommitGeometry(0);					// Send the polygons to GPU as OpenGL VBO	

	// Load VBX
	printf("Loading volume data.\n");
	if ( !gvdb.getScene()->FindFile ( "explosion.vbx", scnpath ) ) {
		nvprintf ( "Cannot find vbx file.\n" );
		nverror();
	}
	printf ( "Loading VBX. %s\n", scnpath );
	gvdb.SetChannelDefault ( 16, 16, 16 );
	gvdb.LoadVBX ( scnpath );	
	gvdb.UpdateApron();

	// Set volume params
	gvdb.getScene()->SetSteps ( .5f, 16, .5f );				// Set raycasting steps
	gvdb.getScene()->SetExtinct ( -1.0f, 1.0f, 0.0f );		// Set volume extinction
	gvdb.getScene()->SetVolumeRange ( 0.1f, 0.0f, 0.5f );	// Set volume value range
	gvdb.getScene()->SetCutoff ( 0.005f, 0.005f, 0.0f );
	gvdb.getScene()->SetBackgroundClr ( 0.1f, 0.2f, 0.4f, 1.0 );		
	gvdb.getScene()->LinearTransferFunc ( 0.00f, 0.10f, Vector4DF(0,0,0,0), Vector4DF(0,0,0,0.0f) );	
	gvdb.getScene()->LinearTransferFunc ( 0.10f, 0.50f, Vector4DF(1,1,0,0.1f), Vector4DF(1,0,0,0.3f) );
	gvdb.getScene()->LinearTransferFunc ( 0.50f, 0.75f, Vector4DF(1,0,0,0.3f), Vector4DF(.2f,.2f,0.2f,0.1f) );
	gvdb.getScene()->LinearTransferFunc ( 0.75f, 1.00f, Vector4DF(.2f,.2f,0.2f,0.1f), Vector4DF(0.1f,0.1f,.1f,0.3f) ); 
	gvdb.CommitTransferFunc ();
	gvdb.SetTransform(Vector3DF(-125, -160, -125), Vector3DF(.5, .5, .5), m_rotate, m_translate );

	// Default Camera 
	Camera3D* cam = new Camera3D;						
	cam->setFov ( 50.0 );
	cam->setNearFar(.1f, 1000);
	cam->setOrbit ( Vector3DF(-40,20,0), Vector3DF(0,0,0), 400, 1.0 );	
	gvdb.getScene()->SetCamera( cam );

	// Default Light
	Light* lgt = new Light;								
	lgt->setOrbit ( Vector3DF(299,57.3f,0), Vector3DF(0,0,0), 500, 1.0 );
	gvdb.getScene()->SetLight ( 0, lgt );	

	// Add render buffer
	nvprintf ( "Creating screen buffer. %d x %d\n", w, h );
	gvdb.AddRenderBuf ( 0, w, h, 4 );
	mDepthInputID = 1;
	gvdb.AddDepthBuf (mDepthInputID, w, h );
	gvdb.getScene()->SetDepthBuf( mDepthInputID );

	// Create opengl texture for display
	// This is a helper func in sample utils (not part of gvdb),
	// which creates or resizes an opengl 2D texture.
	createScreenQuadGL ( &m_gvdb_tex, w, h );

	// Prepare OpenGL depth buffer 
	prepare_depth(w, h);

	return true; 
}

void Sample::prepare_depth(int w, int h )
{
	// Create OpenGL render texture
	glGenTextures(1, (GLuint*) &m_polys_tex);
	glBindTexture(GL_TEXTURE_2D, m_polys_tex );
	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glBindTexture(GL_TEXTURE_2D, 0);

	// Create OpenGL depth texture
	glGenTextures(1, (GLuint*) &m_depth_tex);
	glBindTexture(GL_TEXTURE_2D, m_depth_tex);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, w, h, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
	glBindTexture(GL_TEXTURE_2D, 0);

	// Create framebuffer object
	glGenFramebuffers(1, (GLuint*) &mFBO);
	glBindFramebuffer(GL_FRAMEBUFFER, mFBO );
	glViewport(0, 0, w, h);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, m_polys_tex, 0);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, m_depth_tex, 0);
	checkGL("glBindFramebuffer, prepare_depth");
	GLenum stat = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	printf("Framebuffer status: %d\n", stat);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	checkGL("prepare_depth");
}

void Sample::resize_depth(int w, int h )
{
	// Resize OpenGL textures
	glBindTexture(GL_TEXTURE_2D, m_polys_tex );
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glBindTexture(GL_TEXTURE_2D, 0);

	glBindTexture(GL_TEXTURE_2D, m_depth_tex);	
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, w, h, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
}


void Sample::update_camera_gl (int w, int h )
{
	// Manually setup an OpenGL camera
	// - Normally, GVDB will automatically use the camera specified by gvdb.getScene()->SetCamera.
	// - This code demonstrates use of an OpenGL camera with GVDB. 
	// - Here, the Camera3D is only used to hold camera position and target.
	// - The matrices are created by OpenGL, then loaded into the GVDB camera.
	// - *Note* This introduces a pipeline stall as it performs readback from OpenGL of the matrices to CPU.
	//   Better way is to explicitly construct the matrices and then send to GVDB camera.
		
	Camera3D* cam = gvdb.getScene()->getCamera();
	Vector3DF pos = cam->getPos();	
	double znear = 0.1;
	double zfar = 5000.0;
	double fov = 50.0;
	double aspect = float(w)/float(h);
	GLfloat proj_mtx[16];
	GLfloat view_mtx[16];

	// construct & read back projection matrix 
	glMatrixMode ( GL_PROJECTION );
	glLoadIdentity();
	double xmax = 0.5f * znear * tan( fov * DEGtoRAD/2.0f );
	glFrustum( -xmax, xmax, -xmax/aspect, xmax/aspect, znear, zfar );	
	glGetFloatv ( GL_PROJECTION_MATRIX, proj_mtx );    // (introduces pipeline stall)	
	
	// construct & read back modelview matrix 
	Matrix4F rotation;
	Vector3DF side, up, dir;
	dir = cam->getToPos(); dir -= cam->getPos(); dir.Normalize();		// direction vector
	side = dir; side.Cross ( Vector3DF(0,1,0) ); side.Normalize();		// side vector
	up = side; up.Cross ( dir ); up.Normalize(); dir *= -1;				// up vector
	rotation.Basis ( side, up, dir );									// create a viewing matrix 

	glMatrixMode ( GL_MODELVIEW );
	glLoadIdentity ();
	glMultMatrixf ( rotation.GetDataF() );	
	glTranslatef ( -pos.x, -pos.y, -pos.z );
	glGetFloatv ( GL_MODELVIEW_MATRIX, view_mtx );		// (introduces pipeline stall)

	// volume position
	Vector3DF vol_pos ( 0, 0, 0 );

	// Assign OpenGL matrices to GVDB camera
	cam->setMatrices ( view_mtx, proj_mtx, vol_pos );
}

void Sample::render_depth (int w, int h)
{
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);		
	glEnable (GL_TEXTURE_2D);	
	glEnable (GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
	glDepthFunc(GL_LEQUAL);
	glDisable(GL_MULTISAMPLE);
	glDisable(GL_STENCIL_TEST);

	// Bind shader
	int simple_id = 0;
	int simple_prog = gvdb.getScene()->getShaderProgram( simple_id );
	glUseProgram(simple_prog);
	checkGL("glUseProgram(SIMPLE), render_depth");

	glViewport(0, 0, w, h);

	// Bind framebuffer to output color and depth
	glBindFramebuffer(GL_FRAMEBUFFER, mFBO);	
	checkGL( "glBindFramebuffer, render_depth");

	// Clear both textures
	glClearDepth(1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	checkGL ( "clear depth" );

	// Send shader params
	Matrix4F model_mtx;	model_mtx.Identity();
	renderCamSetupGL(gvdb.getScene(), simple_id, &model_mtx);	
	renderLightSetupGL(gvdb.getScene(), simple_id);
	checkGL ( "lights" );
	renderSetMaterialGL(gvdb.getScene(), simple_id, Vector4DF(.1f, .1f, .1f, 1), Vector4DF(.5f, .5f, .5f, 1), Vector4DF(1, 1, 1, 1));
	checkGL ( "mats" );

	// Render polygons for color and depth
	renderSceneGL(gvdb.getScene(), simple_id, false);
	checkGL ( "draw depth" );

	glUseProgram(0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glDisable(GL_DEPTH_TEST);

	// Write depth texture into GVDB depth texture for CUDA access
	gvdb.WriteDepthTexGL( mDepthInputID, m_depth_tex );	
	
}

void Sample::reshape (int w, int h)
{
	// Resize the OpenGL textures
	resize_depth ( w, h );

	// Resize the GVDB render buffers
	glViewport(0, 0, w, h);
	gvdb.ResizeRenderBuf ( 0, w, h, 4 );
	gvdb.ResizeDepthBuf ( 1, w, h );

	// Resize the GVDB to OpenGL texture
	createScreenQuadGL ( &m_gvdb_tex, w, h );	

	Camera3D* cam = gvdb.getScene()->getCamera();
	cam->setAspect ( float(w)/float(h) );

	postRedisplay();
}

void Sample::display() 
{
	int w = getWidth(), h = getHeight();	// window width & height

	gvdb.SetBias(m_bias);

	if ( m_use_opengl_cam )
		update_camera_gl (w, h);				// Use OpenGL camera matrices

	render_depth(w, h);
	
	clearScreenGL();

	// Render volume
	gvdb.TimerStart ();
	gvdb.Render( SHADE_VOLUME, 0, 0 );   // in_chan=0, out_rbuf=0
	float rtime = gvdb.TimerStop();
	nvprintf ( "Render volume. %6.3f ms\n", rtime );

	// Copy GVDB output buffer into OpenGL texture	
	gvdb.ReadRenderTexGL ( 0, m_gvdb_tex );

	// Composite the OpenGL render with the GVDB render.
	// The GL is treated as background for polygons, and the 
	// depth-limited GVDB volume is composited over it with alpha.
	compositeScreenQuadGL( m_polys_tex, m_gvdb_tex, 1, 0 );
}

void Sample::motion(int x, int y, int dx, int dy) 
{
	// Get camera for GVDB Scene
	Camera3D* cam = gvdb.getScene()->getCamera();
	Light* lgt = gvdb.getScene()->getLight();
	bool shift = (getMods() & NVPWindow::KMOD_SHIFT);		// Shift-key to move volume
	bool ctrl = (getMods() & NVPWindow::KMOD_CONTROL);		// Ctrl-key to rotate volume
	bool alt = (getMods() & NVPWindow::KMOD_ALT);			// Alt-key to move light	

	switch (mouse_down) {
	case NVPWindow::MOUSE_BUTTON_LEFT: {
		if (shift) {
			// Move volume
			m_translate.x -= dx;
			m_translate.z -= dy;
			gvdb.SetTransform(Vector3DF(-125, -160, -125), Vector3DF(.5, .5, .5), m_rotate, m_translate);
		} else if (ctrl) {
			m_rotate.y += dx*0.1f;
			m_rotate.x += dy*0.1f;
			gvdb.SetTransform(Vector3DF(-125, -160, -125), Vector3DF(.5, .5, .5), m_rotate, m_translate);
		} else {
			// Adjust orbit angles
			Vector3DF angs = (alt ? lgt->getAng() : cam->getAng());
			angs.x += dx*0.2f;
			angs.y -= dy*0.2f;
			if (alt)	lgt->setOrbit(angs, lgt->getToPos(), lgt->getOrbitDist(), lgt->getDolly());
			else		cam->setOrbit(angs, cam->getToPos(), cam->getOrbitDist(), cam->getDolly());
		}
		postRedisplay();
	} break;

	case NVPWindow::MOUSE_BUTTON_MIDDLE: {
		// Adjust target pos		
		//m_bias += dy *0.01;
		cam->moveRelative(float(dx) * cam->getOrbitDist() / 1000, float(-dy) * cam->getOrbitDist() / 1000, 0);
		postRedisplay();
	} break;

	case NVPWindow::MOUSE_BUTTON_RIGHT: {
		if (shift) {
			// Move volume
			m_translate.y += dy;
			gvdb.SetTransform(Vector3DF(-125, -160, -125), Vector3DF(.5, .5, .5), m_rotate, m_translate);
		} 	else {
			// Adjust dist
			float dist = (shift ? lgt->getOrbitDist() : cam->getOrbitDist());
			dist -= dy;
			if (ctrl)	lgt->setOrbit(lgt->getAng(), lgt->getToPos(), dist, cam->getDolly());
			else		cam->setOrbit(cam->getAng(), cam->getToPos(), dist, cam->getDolly());
		}
		postRedisplay();
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
	return sample_obj.run ( "NVIDIA(R) GVDB Voxels - gDepthMap", "depthmap", argc, argv, 1024, 768, 4, 5 );
}

void sample_print( int argc, char const *argv)
{
}



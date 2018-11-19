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

#include <GL/glew.h>
#include "main.h"
#include "file_png.h"
#include "optix_scene.h"
#include "nv_gui.h"

OptixScene::OptixScene ()
{
	m_OptixContext = 0;
	m_OptixMainGroup = 0;
	m_OptixVolSampler = 0;
	m_OptixVolIntersectSurfProg = 0;
	m_OptixVolIntersectLevelSetProg = 0;
	m_OptixVolIntersectDeepProg = 0;
	m_OptixVolBBoxProg = 0;
	m_OptixMeshIntersectProg = 0;
	m_OptixMeshBBoxProg = 0;
	m_OptixSeeds = 0;
	m_OptixEnvmap = 0;
}

// InitializeOptiX
// Creates the optix context, loads all mesh and volume intersection programs,
// and prepares the scene graph root.
void OptixScene::InitializeOptix ( int w, int h )
{
	// Create OptiX context
	nvprintf ( "Creating OptiX context.\n" );
	m_OptixContext = Context::create();
	int devlist[2];
	devlist[0] = 0;		// assign context to optix
	rtContextSetDevices(m_OptixContext->get(), 1, devlist);
	m_OptixContext->setEntryPointCount ( 1 );
	m_OptixContext->setRayTypeCount( 2 );
	m_OptixContext->setStackSize( 1024 );
	
	/*m_OptixContext->setPrintEnabled ( true );
	m_OptixContext->setPrintBufferSize ( 1024 );
	m_OptixContext->setExceptionEnabled ( RT_EXCEPTION_ALL, true );*/

	m_OptixContext["scene_epsilon"]->setFloat( 1.0e-6f );	

	// Create Output buffer	
	Variable outbuf = m_OptixContext["output_buffer"];
	m_OptixBuffer = CreateOutputOptix( RT_FORMAT_FLOAT3, w, h );
	outbuf->set ( m_OptixBuffer );

	// Camera ray gen and exception program  
	nvprintf ( "Setting Ray Generation program.\n" );
	m_OptixContext->setRayGenerationProgram( 0, CreateProgramOptix( "optix_trace_primary.ptx", "trace_primary" ) );
	m_OptixContext->setExceptionProgram(     0, CreateProgramOptix( "optix_trace_primary.ptx", "exception" ) );

	// Used by both exception programs
	m_OptixContext["bad_color"]->setFloat( 0.0f, 0.0f, 0.0f );

	// Assign miss program
	nvprintf ( "Setting Miss program.\n" );
	m_OptixContext->setMissProgram( 0, CreateProgramOptix( "optix_trace_miss.ptx", "miss" ) );

	// Declare variables 
	// These will be filled in main loop, but must be declared before optix validation	
	
	SetLight ( Vector3DF(0,0,0) );	
	SetCamera(0x0);
	VDBInfo empty_info;
	memset ( &empty_info, 0, sizeof(VDBInfo) );
	m_OptixContext[ "gvdbObj" ]->setUserData ( sizeof(VDBInfo), (char*) &empty_info );
	m_OptixContext[ "gvdbChan" ]->setInt(0);
	
	// Random seed buffer
	nvprintf ( "Creating random number buffer.\n" );
	int sw=128, sh=128;
	m_OptixSeeds = m_OptixContext->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_INT, sw, sh );
	
	SetSample(0, 0);		// prepares OptixSeeds

	// Initialize mesh intersection programs
	nvprintf ( "Creating Mesh intersection programs.\n" );
	m_OptixMeshIntersectProg =	CreateProgramOptix ( "optix_mesh_intersect.ptx", "mesh_intersect" );
	m_OptixMeshBBoxProg	=		CreateProgramOptix ( "optix_mesh_intersect.ptx", "mesh_bounds" );
	if (m_OptixMeshIntersectProg==0)	{ nvprintf  ( "Error: Unable to load mesh_intersect program.\n" ); nverror (); }
	if (m_OptixMeshBBoxProg==0)			{ nvprintf  ( "Error: Unable to load mesh_bounds program.\n" ); nverror (); }	

	// Initialize volume intersection programs
	nvprintf ( "Creating Volume intersection programs.\n" );
	m_OptixVolIntersectSurfProg =	CreateProgramOptix ( "optix_vol_intersect.ptx", "vol_intersect" );
	m_OptixVolIntersectLevelSetProg = CreateProgramOptix ( "optix_vol_intersect.ptx", "vol_levelset" );
	m_OptixVolIntersectDeepProg =	CreateProgramOptix ( "optix_vol_intersect.ptx", "vol_deep" );
	m_OptixVolBBoxProg	=		CreateProgramOptix ( "optix_vol_intersect.ptx", "vol_bounds" );	
	if (m_OptixVolIntersectSurfProg==0)		{ nvprintf  ( "Error: Unable to load vol_intersect program.\n" ); nverror (); }
	if (m_OptixVolIntersectLevelSetProg==0)		{ nvprintf  ( "Error: Unable to load vol_levelset program.\n" ); nverror (); }
	if (m_OptixVolIntersectDeepProg==0)		{ nvprintf  ( "Error: Unable to load vol_deep program.\n" ); nverror (); }
	if (m_OptixVolBBoxProg==0)			{ nvprintf  ( "Error: Unable to load vol_bounds program.\n" ); nverror (); }

	// Create main group (no geometry yet)
	nvprintf ( "Creating OptiX Main Group and BVH.\n" );
	m_OptixMainGroup = m_OptixContext->createGroup ();
	m_OptixMainGroup->setChildCount ( 0 );
	m_OptixMainGroup->setAcceleration( m_OptixContext->createAcceleration("NoAccel","NoAccel") );	
	//m_OptixMainGroup->setAcceleration( m_OptixContext->createAcceleration("Bvh","Bvh") );	
	m_OptixContext["top_object"]->set( m_OptixMainGroup );
}

void OptixScene::ResizeOutput ( int w, int h )
{
	// Recreate output buffer
	m_OptixBuffer->destroy();
	nvprintf ( "OptiX buffer size: %d, %d\n", w, h );
	Variable outbuf = m_OptixContext["output_buffer"];
	m_OptixBuffer = CreateOutputOptix( RT_FORMAT_FLOAT3, w, h );
	outbuf->set ( m_OptixBuffer );
}


// CreateOutputOptiX
// Creates an optix output buffer from an opengl buffer
Buffer OptixScene::CreateOutputOptix ( RTformat format, unsigned int width, unsigned int height )
{
	// Create OpenGL buffer 
	Buffer buffer;
	GLuint vbo = 0;
	glGenBuffers (1, &vbo );
	glBindBuffer ( GL_ARRAY_BUFFER, vbo );
	size_t element_size;
	m_OptixContext->checkError( rtuGetSizeForRTformat(format, &element_size));
	glBufferData(GL_ARRAY_BUFFER, element_size * width * height, 0, GL_STREAM_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Create OptiX output buffer from OpenGL buffer
	buffer = m_OptixContext->createBufferFromGLBO( RT_BUFFER_OUTPUT , vbo);
	buffer->setFormat(format);
	buffer->setSize( width, height );

	//buffer = m_OptixContext->createBuffer ( RT_BUFFER_INPUT_OUTPUT , format, width, height );

	return buffer;
}

optix::Program OptixScene::CreateProgramOptix ( std::string name, std::string prog_func )
{
	optix::Program program;

	nvprintf  ( "  Loading: %s, %s\n", name.c_str(), prog_func.c_str() );
	try { 
		program = m_OptixContext->createProgramFromPTXFile ( name, prog_func );
	} catch (Exception e) {
		nvprintf  ( "  OPTIX ERROR: %s \n", m_OptixContext->getErrorString( e.getErrorCode() ).c_str() );
		nverror ();		
	}
	return program;
}

// ClearGraph
// Clears the OptiX scene graph
void OptixScene::ClearGraph ()
{
	for (int n=0; n < m_OptixModels.size(); n++ ) {
		optix::GeometryGroup	geomgroup	= (optix::GeometryGroup) m_OptixModels[n]->m_tform->getChild<optix::GeometryGroup> ();
		optix::GeometryInstance geominst	= (optix::GeometryInstance) geomgroup->getChild(0);
		optix::Geometry			geom		= (optix::Geometry) geominst->getGeometry ();
		geom->destroy();
		geominst->destroy();
		geomgroup->destroy();		
	}
	if ( m_OptixModels.size() > 0 ) m_OptixModels.clear ();

	for (int n=0; n < m_OptixVolumes.size(); n++ ) {
		optix::GeometryGroup	geomgroup	= (optix::GeometryGroup) m_OptixVolumes[n]->getChild<optix::GeometryGroup> ();
		optix::GeometryInstance geominst	= (optix::GeometryInstance) geomgroup->getChild(0);
		optix::Geometry			geom		= (optix::Geometry) geominst->getGeometry ();
		geom->destroy();
		geominst->destroy();
		geomgroup->destroy();
		m_OptixVolumes[n]->destroy();
	}
	if ( m_OptixVolumes.size() > 0 ) m_OptixVolumes.clear ();

	for (int n=0; n < m_OptixMats.size(); n++ )
		m_OptixMats[n]->destroy();

	if ( m_OptixMats.size() > 0 ) m_OptixMats.clear ();	

	CreateEnvmap ( "" );
}

// AddMaterial
// Add a material to the Optix graph
int OptixScene::AddMaterial ( std::string fname, std::string cast_prog, std::string shadow_prog )
{
	// Create OptiX material
	optix::Material omat = m_OptixContext->createMaterial();

	// Load material shaders
	std::string ptx_file = fname + ".ptx";
	Program ch_program = CreateProgramOptix ( ptx_file, cast_prog );		
	Program ah_program = CreateProgramOptix ( ptx_file, shadow_prog );
	omat->setClosestHitProgram ( 0, ch_program );
	omat->setAnyHitProgram ( 1, ah_program );			
	m_OptixMats.push_back ( omat );	

	// Add material params
	MaterialParams matp;
	memset ( &matp, 0, sizeof(MaterialParams) );
	m_OptixMatParams.push_back ( matp );
	omat["mat"]->setUserData( sizeof(MaterialParams), &matp );

	return (int) m_OptixMats.size()-1 ;
}

void OptixScene::SetMaterialParams ( int n, MaterialParams* p )
{
	// Get the material
	optix::Material omat = m_OptixMats[n]; 

	// Set the material param variable to user data
	//  (See optix_trace_surface.cu where this is used)
	p->id = n;
	memcpy ( &m_OptixMatParams[n], p, sizeof(MaterialParams) );	
	omat["mat"]->setUserData( sizeof(MaterialParams), p );
}


OptixModel* OptixScene::AddPolygons ( Model* model, int mat_id, Matrix4F& xform )
{
	OptixModel* om = new OptixModel;
	m_OptixModels.push_back ( om );	
	
	om->m = model;
	om->m_matid = mat_id;
	om->m_xform = xform;

	// Model definition
	//    Transform 
	//        |
	//   GeometryGroup -- Acceleration 
	//        |
	//  GeometryInstance
	//        |
	//     Geometry -- Intersect Prog/BBox Prog

	// Geometry	
	UpdatePolygons ( om, model, mat_id, xform );

	// Geometry Instance node
	Material mat;
	mat = m_OptixMats[ mat_id ];
	GeometryInstance geominst = m_OptixContext->createGeometryInstance ( om->m_geom, &mat, &mat+1 );		// <-- geom is specified as child here

	// Geometry Group node
	GeometryGroup geomgroup;
	geomgroup = m_OptixContext->createGeometryGroup ();		
	const char* Builder = "Sbvh";
	const char* Traverser = "Bvh";
	const char* Refine = "0";
	const char* Refit = "0";
	optix::Acceleration acceleration = m_OptixContext->createAcceleration( Builder, Traverser );
	acceleration->setProperty( "refine", Refine );
	acceleration->setProperty( "refit", Refit );
	acceleration->setProperty( "vertex_buffer_name", "vertex_buffer" );
    acceleration->setProperty( "index_buffer_name", "vindex_buffer" );	
	acceleration->markDirty();
	geomgroup->setAcceleration( acceleration );	
	geomgroup->setChildCount ( 1 );
	geomgroup->setChild( 0, geominst );

	// Transform Node
	if ( om->m_tform==0 ) 
		om->m_tform = m_OptixContext->createTransform ();
	om->m_tform->setMatrix ( true, xform.GetDataF(), 0x0 );	
	om->m_tform->setChild ( geomgroup );

	// Add model root (Transform) to the Main Group
	int id = m_OptixModels.size() - 1 + m_OptixVolumes.size();
	m_OptixMainGroup->setChildCount ( id+1 );
	m_OptixMainGroup->setChild ( id, om->m_tform );

	return om;
}

void OptixScene::UpdatePolygons ( OptixModel* om, Model* model, int mat_id, Matrix4F& xform )
{
	om->m_numvert = model->vertCount;
	om->m_numtri = model->elemCount;
	om->m_numnorm = om->m_numvert;

	// Create new buffers
	if ( om->m_vbuf !=0 ) om->m_vbuf->destroy();
	om->m_vbuf = m_OptixContext->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, om->m_numvert );
	
	if ( om->m_nbuf !=0 ) om->m_nbuf->destroy();
	om->m_nbuf = m_OptixContext->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, om->m_numnorm );

	if ( om->m_tbuf !=0 ) om->m_tbuf->destroy();
	om->m_tbuf = m_OptixContext->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, om->m_numvert  );

	if ( om->m_vibuf !=0 ) om->m_vibuf->destroy();
	om->m_vibuf = m_OptixContext->createBuffer ( RT_BUFFER_INPUT, RT_FORMAT_INT3, om->m_numtri );

	if ( om->m_nibuf !=0 ) om->m_nibuf->destroy();
	om->m_nibuf = m_OptixContext->createBuffer ( RT_BUFFER_INPUT, RT_FORMAT_INT3, om->m_numtri );

	if ( om->m_mibuf !=0 ) om->m_mibuf->destroy();
	om->m_mibuf = m_OptixContext->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, om->m_numtri );

	// Map buffers
	float3* vbuffer_data = static_cast<float3*>( om->m_vbuf->map() );
	float3* nbuffer_data = static_cast<float3*>( om->m_nbuf->map() );
	float2* tbuffer_data = static_cast<float2*>( om->m_tbuf->map() );
	int3* vindex_data = static_cast<int3*>( om->m_vibuf->map() );
	int3* nindex_data = static_cast<int3*>( om->m_nibuf->map() );
    unsigned int* mindex_data = static_cast<unsigned int*>( om->m_mibuf->map() );

	// Copy vertex data
	float2 tc;
	tc.x = 0; tc.y = 0;
	char* vdat = (char*) model->vertBuffer;
	float3* v3;
	Vector4DF vert;
	for ( int i=0; i < (int) om->m_numvert ; i++ ) {
		v3 = (float3*) (vdat + model->vertOffset);
		vbuffer_data[i] = *v3;
		v3 = (float3*) (vdat + model->normOffset);
		nbuffer_data[i] = *v3;
		tbuffer_data[i] = tc;
		vdat += model->vertStride;
	}	

	// Copy element data (indices)
	for (int i=0; i < (int) om->m_numtri; i++ ) {
		int3 tri_verts;		// vertices in trangle
		tri_verts.x = model->elemBuffer[ i*3   ];
		tri_verts.y = model->elemBuffer[ i*3+1 ];
		tri_verts.z = model->elemBuffer[ i*3+2 ];
		vindex_data [ i ] = tri_verts;
		nindex_data [ i ] = tri_verts;
		mindex_data [ i ] = 0;
	}

	// Geometry node
	if ( om->m_geom==0 ) 
		om->m_geom = m_OptixContext->createGeometry ();

	om->m_geom->setPrimitiveCount ( om->m_numtri );
	om->m_geom->setIntersectionProgram ( m_OptixMeshIntersectProg );
	om->m_geom->setBoundingBoxProgram ( m_OptixMeshBBoxProg );	
	om->m_geom[ "vertex_buffer" ]->setBuffer( om->m_vbuf );		
    om->m_geom[ "normal_buffer" ]->setBuffer( om->m_nbuf );	
	om->m_geom[ "texcoord_buffer" ]->setBuffer( om->m_tbuf );
	om->m_geom[ "vindex_buffer" ]->setBuffer( om->m_vibuf );	
    om->m_geom[ "nindex_buffer" ]->setBuffer( om->m_nibuf );
	om->m_geom[ "tindex_buffer" ]->setBuffer( om->m_nibuf );
    om->m_geom[ "mindex_buffer" ]->setBuffer( om->m_mibuf );

	// Unmap buffers
	om->m_vbuf->unmap();	
	om->m_nbuf->unmap();
	om->m_tbuf->unmap();
	om->m_vibuf->unmap();
	om->m_nibuf->unmap();
	om->m_mibuf->unmap();

	om->m_geom->markDirty();		// mark geometry dirty
    
	// Update Transform node
	if ( om->m_tform!=0 ) {
		om->m_tform->setMatrix ( true, xform.GetDataF(), 0x0 );	
		
		// mark acceleration dirty
		optix::GeometryGroup geomgroup = (optix::GeometryGroup) om->m_tform->getChild<optix::GeometryGroup> ();
		geomgroup->getAcceleration()->markDirty();
	}
}


// AddVolume
// Adds one GVDB Volume to the Optix scene graph.
// Multiple volumes and meshes can exist in the Optix graph.
// GVDB Volumes are described to the OptiX scene graph as bounding box (brick buffer). 
// Once a ray traverses the BVH, the GVDB Intersection programs traverse the volume data.
// For OptiX to access the GVDB data an optix-specific volume sampler is created
// based on the GVDB Texture Atlas.
void OptixScene::AddVolume ( int atlas_glid, Vector3DF vmin, Vector3DF vmax, Matrix4F& xform, int mat_id, char isect )
{
	int id = (int) m_OptixModels.size() + (int) m_OptixVolumes.size();

	// Model definition
	//    Transform 
	//        |
	//   GeometryGroup <-- Acceleration struct
	//        |
	//  GeometryInstance <-- Material IDs
	//        |
	//     Geometry <-- Intersect Progs/BBox Prog
	//  (brick buffer)

	// Geometry node
	optix::Geometry geom;
	geom = m_OptixContext->createGeometry ();
	geom->setPrimitiveCount ( 1 );	
	switch (isect) {
	case 'D':	geom->setIntersectionProgram(m_OptixVolIntersectDeepProg); break;
	case 'L':	geom->setIntersectionProgram(m_OptixVolIntersectLevelSetProg);	break;
	case 'S':	geom->setIntersectionProgram(m_OptixVolIntersectSurfProg);	break;
//	case 'C':	geom->setIntersectionProgram(m_OptixVolIntersectSectionProg);	break;
	};
	geom->setBoundingBoxProgram ( m_OptixVolBBoxProg );

	// Brick buffer	
	int num_bricks = 1;
	Buffer brick_buffer = m_OptixContext->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, num_bricks*2 );
	
	float3* brick_data = static_cast<float3*>( brick_buffer->map() );		
	brick_data[0] = * (float3*) &vmin;		// Cast Vector3DF to float3. Assumes both are 3x floats
	brick_data[1] = * (float3*) &vmax;	

	geom[ "brick_buffer" ]->setBuffer( brick_buffer );
	
	brick_buffer->unmap();	
	
	geom[ "mat_id"]->setUint( mat_id );	

	// Geometry Instance node
	Material mat;
	mat = m_OptixMats[ mat_id ];
	GeometryInstance geominst = m_OptixContext->createGeometryInstance ( geom, &mat, &mat+1 );		// <-- geom is specified as child here

	// Geometry Group node
	GeometryGroup geomgroup;
	geomgroup = m_OptixContext->createGeometryGroup ();		

	const char* Builder = "NoAccel";		// or "Trbvh"
	const char* Traverser = "NoAccel";
	const char* Refine = "0";
	const char* Refit = "1";
	optix::Acceleration acceleration = m_OptixContext->createAcceleration( Builder, Traverser );
	acceleration->setProperty( "refine", Refine );
	acceleration->setProperty( "refit", Refit );
	acceleration->setProperty( "vertex_buffer_name", "vertex_buffer" );
    acceleration->setProperty( "index_buffer_name", "vindex_buffer" );	
	acceleration->markDirty();
	
	geomgroup->setAcceleration( acceleration );		
	geomgroup->setChildCount ( 1 );
	geomgroup->setChild( 0, geominst );

	// Transform node
	Transform tform = m_OptixContext->createTransform ();
	tform->setMatrix ( true, xform.GetDataF(), 0x0 );	
	tform->setChild ( geomgroup );
	
	// Add model root (Transform) to the Main Group
	m_OptixMainGroup->setChildCount ( id+1 );
	m_OptixMainGroup->setChild ( id, tform );

	m_OptixVolumes.push_back ( tform );

	// Create a volume texture sampler
	if ( m_OptixVolSampler != 0x0 )
		m_OptixVolSampler->destroy();
}

// ValidateGraph
// Validate the OptiX scene graph, 
// last step before rendering.
void OptixScene::ValidateGraph ()
{
	try {
		m_OptixContext->validate ();
	} catch (const Exception& e) {		
		std::string msg = m_OptixContext->getErrorString ( e.getErrorCode() );		
		nvprintf  ( "OPTIX ERROR:\n%s\n", msg.c_str() );
		nverror ();		
	}
	try {
		m_OptixContext->compile ();
	} catch (const Exception& e) {		
		std::string msg = m_OptixContext->getErrorString ( e.getErrorCode() );		
		nvprintf  ( "OPTIX ERROR:\n%s\n", msg.c_str() );
		nverror ();		
	}
}
void OptixScene::SetSample(int frame, int sample)
{
	m_OptixContext["frame_number"]->setUint(frame);
	m_OptixContext["sample"]->setUint(sample);

	RTsize bw, bh;
	m_OptixSeeds->getSize(bw, bh);
	unsigned int* seeds = (unsigned int*) m_OptixSeeds->map();
	srand( sample*17 + 4732 + frame );
	for (int i = 0; i < bw*bh; i++) {
		seeds[i] = rand() * 0xffffL / RAND_MAX;
	}
	m_OptixSeeds->unmap();
	m_OptixContext["rnd_seeds"]->set(m_OptixSeeds);
}

void OptixScene::SetCamera ( Camera3D* cam )
{
	if (cam == 0x0) {				
		m_OptixContext["cam_pos"]->setFloat(0, 0, 0);				
		m_OptixContext["cam_U"]->setFloat(1, 0, 0);
		m_OptixContext["cam_V"]->setFloat(0, 1, 0);
		m_OptixContext["cam_S"]->setFloat(0, 0, 0);
		return;
	}
	m_OptixContext["cam_pos"]->setFloat(cam->getPos().x, cam->getPos().y, cam->getPos().z);	
	Vector3DF cams = cam->tlRayWorld;
	Vector3DF camu = cam->trRayWorld; camu -= cams;
	Vector3DF camv = cam->blRayWorld; camv -= cams;	
	m_OptixContext["cam_U"]->setFloat(camu.x, camu.y, camu.z);
	m_OptixContext["cam_V"]->setFloat(camv.x, camv.y, camv.z);
	m_OptixContext["cam_S"]->setFloat(cams.x, cams.y, cams.z);
}
void OptixScene::SetLight ( Vector3DF pos )
{
	m_OptixContext["light_pos"]->setFloat ( pos.x, pos.y, pos.z );
}

void OptixScene::SetTransferFunc ( Vector4DF* src )
{
	m_OptixTransferFunc = m_OptixContext->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, 16384 );
	
	float4* buf_data = static_cast<float4*>( m_OptixTransferFunc->map() );
	memcpy ( buf_data, src, 16384*sizeof(float4) );
	m_OptixTransferFunc->unmap();	
		
	m_OptixContext[ "scn_transfer_func" ]->set( m_OptixTransferFunc );
}
void OptixScene::CreateEnvmap (char* fpath)
{
	// Load env map
	nvImg img;
	if ( strlen(fpath) > 0 ) {
		img.LoadPng(fpath);
	} else {
		img.Create ( 64, 64, IMG_RGBA );
		img.Fill ( 1,1,1,1 );
	}
	// Create tex sampler and populate with default values
	if ( m_OptixEnvmap != 0 ) m_OptixEnvmap->destroy();
	m_OptixEnvmap = m_OptixContext->createTextureSampler();
	m_OptixEnvmap->setWrapMode(0, RT_WRAP_REPEAT);
	m_OptixEnvmap->setWrapMode(1, RT_WRAP_REPEAT);
	m_OptixEnvmap->setWrapMode(2, RT_WRAP_REPEAT);
	m_OptixEnvmap->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
	m_OptixEnvmap->setMaxAnisotropy(1.0f);
	m_OptixEnvmap->setMipLevelCount(1u);
	m_OptixEnvmap->setArraySize(1u);

	const unsigned int nx = img.getWidth();
	const unsigned int ny = img.getHeight();
	unsigned char* ipx = img.getData();

	// Create buffer and fill from image
	optix::Buffer buffer = m_OptixContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, nx, ny);
	float* buf_data = static_cast<float*>(buffer->map());
	float* buf_pix = buf_data;
	for (int y = 0; y < ny; y++)
		for (int x = 0; x < nx; x++) {
			*buf_pix++ = float(*ipx++) / 255.0f;
			*buf_pix++ = float(*ipx++) / 255.0f;
			*buf_pix++ = float(*ipx++) / 255.0f;
			*buf_pix++ = float(*ipx++) / 255.0f;
		}
	buffer->unmap();

	m_OptixEnvmap->setBuffer(0u, 0u, buffer);
	m_OptixEnvmap->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);

	m_OptixContext["envmap"]->set( m_OptixEnvmap );
}



void OptixScene::UpdateScene ( nvdb::Scene* scene )
{
	// Camera
	Camera3D* cam = scene->getCamera();
	SetCamera(cam);

	// Light	
	Light* light = scene->getLight();
	Vector3DF lpos = light->getPos();
	SetLight(lpos);
}

void OptixScene::UpdateVolume ( nvdb::VolumeGVDB* gvdb )
{
	// PrepareVDB
	gvdb->PrepareVDB ();

	// Update VDBInfo in OptiX
	size_t sz = gvdb->getVDBSize ();
	VDBInfo* vdbinfo = (VDBInfo*) gvdb->getVDBInfo ();
	
	m_OptixContext[ "gvdbObj" ]->setUserData ( sz, vdbinfo );	
}

void OptixScene::UpdatePolygons ()
{
	for (int n=0; n < m_OptixModels.size(); n++ ) {
		UpdatePolygons ( m_OptixModels[n], m_OptixModels[n]->m, m_OptixModels[n]->m_matid, m_OptixModels[n]->m_xform );
	}
}

void OptixScene::Render ( VolumeGVDB* gvdb, char shading, char chan )
{
	// Get buffer dims	
	RTsize bw, bh;	
	m_OptixBuffer->getSize ( bw, bh );		

	// Prepare ScnInfo for GVDB
	gvdb->PrepareRender(bw, bh, shading);

	// Transfer ScnInfo to OptiX variable
	size_t sz = gvdb->getScnSize();
	ScnInfo* scninfo = (ScnInfo*)gvdb->getScnInfo();
	m_OptixContext["scn"]->setUserData(sz, scninfo);
	m_OptixContext["gvdbChan"]->setInt( (int) chan );

	// Update Camera parameters directly
	UpdateScene(gvdb->getScene());
	
	try {
		m_OptixContext->launch ( 0, (int) bw, (int) bh );	
	} catch (const Exception& e) {		
		std::string msg = m_OptixContext->getErrorString ( e.getErrorCode() );		
		nvprintf  ( "OPTIX ERROR:\n%s\n", msg.c_str() );
		//nverror ();		
	}
	cuCtxSynchronize ();

}
void OptixScene::SaveOutput(char* fname)
{
	// map optix buffer
	RTsize bw, bh;
	m_OptixBuffer->getSize(bw, bh);	
	float* dat_buf = (float*) m_OptixBuffer->map();
	float* dat = dat_buf;

	// assign maximum pixel value	
	float v;
	float vmax = 1.2f;	
	
	// remap to 8-bit RGB
	unsigned char* pix_buf = (unsigned char*) malloc(bw*bh * 3);
	unsigned char* pix = pix_buf;	
	for (int y=0; y < bh; y++)
		for (int x = 0; x < bw; x++) {
			v = (*dat++)*255.0f / vmax; *pix++ = (unsigned char) ((v > 255) ? 255 : v);
			v = (*dat++)*255.0f / vmax; *pix++ = (unsigned char) ((v > 255) ? 255 : v);
			v = (*dat++)*255.0f / vmax; *pix++ = (unsigned char) ((v > 255) ? 255 : v);
		}
			
	// save to png file
	save_png(fname, pix_buf, bw, bh, 3);

	m_OptixBuffer->unmap();

	free(pix_buf);
}

void OptixScene::ReadOutputTex ( int out_tex )
{
	// Target output to OpenGL texture
	glBindTexture( GL_TEXTURE_2D, out_tex );	

	// Get OptiX buffer 	
	RTsize bw, bh;	
	m_OptixBuffer->getSize ( bw, bh );	
	int vboid = m_OptixBuffer->getGLBOId ();
	glBindBuffer ( GL_PIXEL_UNPACK_BUFFER, vboid );		// Bind to the optix buffer
		
	RTsize elementSize = m_OptixBuffer->getElementSize();
	if      ((elementSize % 8) == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
	else if ((elementSize % 4) == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
	else if ((elementSize % 2) == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
	else                             glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	// Copy the OptiX results into a GL texture
	//  (device-to-device transfer using bound gpu buffer)
	RTformat buffer_format = m_OptixBuffer->getFormat();
	GLsizei tw = (GLsizei) bw;
	GLsizei th = (GLsizei) bh;
	switch (buffer_format) {
	case RT_FORMAT_UNSIGNED_BYTE4:	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,			tw, th, 0, GL_BGRA, GL_UNSIGNED_BYTE, 0 );	break;
	case RT_FORMAT_FLOAT4:			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB,		tw, th, 0, GL_RGBA, GL_FLOAT, 0);	break;
	case RT_FORMAT_FLOAT3:			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB,		tw, th, 0, GL_RGB, GL_FLOAT, 0);		break;
	case RT_FORMAT_FLOAT:			glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, tw, th, 0, GL_LUMINANCE, GL_FLOAT, 0);	break;	
	}
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );	
	glBindTexture( GL_TEXTURE_2D, 0);
}

// int msz = sizeof(PMaterial);
// PMaterial* mdat = gView.getMaterial();
// m_OptixContext[ "mat" ]->setUserData ( msz, mdat );
// g_OptixContext[ "gvdb" ]->setUserData ( vdb_sz, vdb_dat );




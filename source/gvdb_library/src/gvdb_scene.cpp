//--------------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2016-2018, NVIDIA Corporation. 
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
// Version 1.1: Rama Hoetzlein, 3/25/2018
//----------------------------------------------------------------------------------

#include "gvdb_allocator.h"
#include "gvdb_scene.h"
#include "loader_ObjarReader.h" 
#include "loader_OBJReader.h" 
#include "string_helper.h"
using namespace nvdb;

Scene* Scene::gScene = 0x0;
CallbackParser* Scene::gParse = 0x0;

Scene::Scene ()
{
	mCamera = 0x0;	
	mOutFile = "out.scn";
	mOutModel = "";
	mOutFrame = 0;
	mOutCam = new Camera3D;
	mOutLight = new Light;
	mShadowParams.Set ( 0.8f, 1.0f, 0 );
	mTransferFunc = nullptr;
	mFrameSamples = 8;
	mVClipMin.Set (  -1000000, -1000000, -1000000 );
	mVClipMax.Set (   1000000,  1000000,  1000000 );
	mVThreshold.Set ( 0.2f, 1, 0 );
	mVLeaf.Set ( 0, 0, 0 );
	mVFrames.Set ( 0, 0, 0 );
	mVName = "density";
	mShading = SHADE_TRILINEAR;
	mFilterMode = 0;
	mFrame = 0;
	mSample = 0;	
	mDepthBuf = 255;   // no depth buf
	
	// Scene and Parse must be global singletons because 
	// the callback parser accepts pointers-to-member function
	// which must be static. These static callbacks can only 
	// access the scene/parser through global variables. See: LoadModel
	gScene = this;
	gParse = new CallbackParser;	

	SetSteps ( 1.0f, 16.0f, 0.1f );
	SetExtinct ( -1.1f, 1.5f, 0.0f );
	SetVolumeRange ( 0.1f, 0.0f, 1.0f );
	SetCutoff ( 0.005f, 0.01f, 0.0f );
}

Vector4DF lerp4 ( Vector4DF a, Vector4DF b, float u )
{
	return Vector4DF( a.x+u*(b.x-a.x), a.y+u*(b.y-a.y), a.z+u*(b.z-a.z), a.w+u*(b.w-a.w) );
}

int Scene::getShaderProgram (int i)
{
	return mProgram[ i ];
}

void Scene::LinearTransferFunc ( float t0, float t1, Vector4DF a, Vector4DF b )
{
	Vector4DF clr;

	const int sz = 16384;
	
	int n0 = static_cast<int>(t0 * static_cast<float>(sz));
	int n1 = static_cast<int>(t1 * static_cast<float>(sz));
	
	if ( mTransferFunc == nullptr ) {
		// Initialize mTransferFunc and set all elements to 0
		// This has to be done using malloc/free, since VolumeGVDB currently recreates it
		// using CreateMemLinear. Hopefully we'll change everything to use new/delete in
		// the future.
		mTransferFunc = (Vector4DF*)calloc(sz, sizeof(Vector4DF));
	}
	
	for (int n=n0; n < n1; n++ ) {
		float x = float(n-n0) / float(n1-n0);
		clr = lerp4 ( a, b, x );
		mTransferFunc[n].Set ( clr.x, clr.y, clr.z, clr.w );
	}
}

Scene::~Scene ()
{
	// Delete all pointers to objects that we own
	if (mCamera != nullptr) {
		delete mCamera;
	}

	for (int i = 0; i < mModels.size(); i++) {
		if (mModels[i] != nullptr) {
			delete mModels[i];
		}
	}

	for (int i = 0; i < mLights.size(); i++) {
		if (mLights[i] != nullptr) {
			delete mLights[i];
		}
	}

	if (mOutCam != nullptr) {
		delete mOutCam;
	}

	if (mTransferFunc != nullptr) {
		free(mTransferFunc); // See notes on mTransferFunc above
	}

	if (mOutLight != nullptr) {
		delete mOutLight;
	}
}


void Scene::AddPath ( std::string path )
{
	mSearchPaths.push_back ( path );	
}
void Scene::ClearModel ( Model* m )
{
	if ( m->vertBuffer != 0x0 ) free ( m->vertBuffer );
	if ( m->elemBuffer != 0x0 ) free ( m->elemBuffer );
}

void Scene::LoadModel ( Model* m, std::string filestr, float scale, float tx, float ty, float tz )
{
	char filename[1024];
	strncpy ( filename, filestr.c_str(), 1024 );

	// polygonal model
	if ( OBJARReader::isMyFile ( filename ) ) {
		// OBJAR File
		OBJARReader load_objar;
		load_objar.LoadFile ( m, filename, mSearchPaths );
	
	} else if ( OBJReader::isMyFile ( filename ) ) {
		// OBJ FIle
		OBJReader load_obj;
		load_obj.LoadFile ( m, filename, mSearchPaths );
	}
	// Rescale if desired
	m->Transform ( Vector3DF(tx,ty,tz), Vector3DF(scale,scale,scale) );
	
	// Save name of model (for recording)
	char buf[32];
	sprintf ( buf, "%f", scale );
	mOutModel = " model " + std::string(filename) + " " + std::string(buf);
}

Model* Scene::AddModel ()
{
	Model* m = new Model;
	m->modelType = 0;	// polygonal model
	mModels.push_back ( m );
	return m;
}

// backward compatibility function
size_t Scene::AddModel ( std::string filestr, float scale, float tx, float ty, float tz)
{
	Model* m = AddModel ();	
	LoadModel ( m, filestr, scale, tx, ty, tz );
	return mModels.size()-1;
}

size_t Scene::AddVolume ( std::string filestr, Vector3DI res, char vtype, float scale)
{
	// volume model
	Model* m = new Model;
	m->modelType = 1;	
	
	m->volFile = filestr;
	m->volRes = res;
	m->volType = vtype;
	mModels.push_back ( m );	

	// Save name of model (for recording)
	char buf[32];
	sprintf ( buf, "%f", scale );
	mOutModel = " model " + filestr + " " + std::string(buf);

	return mModels.size()-1;
}

size_t Scene::AddGround ( float hgt, float scale )
{
	Model* m = new Model;
	mModels.push_back ( m );

	if ( OBJReader::isMyFile ( "ground.obj" ) ) {		
		OBJReader load_obj;
		load_obj.LoadFile ( m, "ground.obj", mSearchPaths );
	}
	// Rescale if desired
	m->Transform ( Vector3DF(0,hgt,0), Vector3DF(scale,scale,scale) );

	return mModels.size()-1;
}

Camera3D* Scene::SetCamera ( Camera3D* cam )
{
	if ( mCamera != 0x0 ) delete mCamera;
	mCamera = cam;	
	return cam;
}

Light* Scene::SetLight ( int n, Light* light )
{
	mLights.resize ( n+1, 0x0 );
	if ( mLights[n] != 0x0 ) delete mLights[n];
	mLights[ n ] = light;
	return light;
}

// Read a shader file into a character string
char *ReadShaderSource( char *fileName )
{
	FILE *fp = fopen( fileName, "rb" );
	if (!fp) return NULL;
	fseek( fp, 0L, SEEK_END );
	long fileSz = ftell( fp );
	fseek( fp, 0L, SEEK_SET );
	char *buf = (char *) malloc( fileSz+1 );
	if (!buf) { fclose(fp); return NULL; }
	fread( buf, 1, fileSz, fp );
	buf[fileSz] = '\0';
	fclose( fp );
	return buf;
}

// Create a GLSL program object from vertex and fragment shader files
int Scene::AddShader ( int prog_id, const char* vertfile, const char* fragfile )
{
	return AddShader ( prog_id, vertfile, fragfile, 0x0 );
}

bool Scene::FindFile ( std::string fname, char* path )
{
	return getFileLocation ( fname.c_str(), path, mSearchPaths );	
}

int Scene::AddShader ( int prog_id, const char* vertfile, const char* fragfile, const char* geomfile )
{
	mLastShader = vertfile;
	int maxLog = 65536, lenLog;
	char log[65536];

	// Search paths
	char vertpath[1024];
	char geompath[1024];
	char fragpath[1024];
	if ( !FindFile ( vertfile, vertpath ) ) {
		gprintf ( "ERROR: Unable to open '%s'\n", vertfile ); 
		gerror ();
	}
	if ( !FindFile ( fragfile, fragpath ) ) {
		gprintf ( "ERROR: Unable to open '%s'\n", fragfile ); 
		gerror ();
	}
	if ( geomfile != 0x0 ) { 
	if ( !FindFile ( geomfile, geompath ) ) {
		gprintf ( "ERROR: Unable to open '%s'\n", geomfile ); 
		gerror ();
	}
	}

	// Read shader source	
	GLuint program = glCreateProgram();
    char *vertSource = ReadShaderSource( vertpath );
	if ( !vertSource ) {
		gprintf ( "ERROR: Unable to read source '%s'\n", vertfile ); 
		gerror();
	}
	char *fragSource = ReadShaderSource( fragpath );
	if ( !fragSource) { 
		gprintf ( "ERROR: Unable to read source '%s'\n", fragfile ); 
		gerror();
	}
	char *geomSource = 0x0;
	if ( geomfile != 0x0 ) { 
		geomSource = ReadShaderSource( geompath );
		if ( !geomSource) { 
			gprintf ( "ERROR: Unable to read source '%s'\n", geomfile ); 
			gerror();
		}
	}

	int statusOK;
	GLuint vShader = glCreateShader( GL_VERTEX_SHADER );
	glShaderSource( vShader, 1, (const GLchar**) &vertSource, NULL );
	glCompileShader( vShader );
	glGetShaderiv( vShader, GL_COMPILE_STATUS, &statusOK );
	if (!statusOK) { 
		glGetShaderInfoLog ( vShader, maxLog, &lenLog, log );		
		gprintf ("***Compile Error in '%s'!\n", vertfile); 
		gprintf ("  %s\n", log );		
		gerror ();
	}
	free( vertSource );

	GLuint fShader = glCreateShader( GL_FRAGMENT_SHADER );
	glShaderSource( fShader, 1, (const GLchar**) &fragSource, NULL );
	glCompileShader( fShader );
	glGetShaderiv( fShader, GL_COMPILE_STATUS, &statusOK );
	if (!statusOK) { 
		glGetShaderInfoLog ( fShader, maxLog, &lenLog, log );		
		gprintf ("***Compile Error in '%s'!\n", fragfile); 
		gprintf ("  %s\n", log );	
		gerror ();
	}	
	free( fragSource );

	GLuint gShader;
	if ( geomfile != 0x0 ) { 
		gShader = glCreateShader( GL_GEOMETRY_SHADER );
		glShaderSource( gShader, 1, (const GLchar**) &geomSource, NULL );
		glCompileShader( gShader );
		glGetShaderiv( gShader, GL_COMPILE_STATUS, &statusOK );
		if (!statusOK) { 
			glGetShaderInfoLog ( gShader, maxLog, &lenLog, log );		
			gprintf ("***Compile Error in '%s'!\n", geomfile); 
			gprintf ("  %s\n", log );	
			gerror ();
		}	
		free( geomSource );
	}

	glAttachShader( program, vShader );	
	glAttachShader( program, fShader );
	if ( geomfile != 0x0 ) glAttachShader( program, gShader );
    glLinkProgram( program );
    glGetProgramiv( program, GL_LINK_STATUS, &statusOK );
    if ( !statusOK ) { 
		printf("***Error! Failed to link '%s' and '%s'!\n", vertfile, fragfile ); 
		gerror ();
	}

	mShaders.push_back ( program );
	ParamList params;	
	for (int n=0; n < 128; n++ ) 
		params.p[n] = -1;
	mParams.push_back ( params );
	
	mProgToSlot[ program ] = static_cast<int>(mParams.size()-1);
	mProgram[ prog_id ] = program;

    return program;
}

int	Scene::getSlot ( int prog_id )
{
	// Get abstract slot from a program ID
	for (int n=0; n < mShaders.size(); n++ ) {
		if ( mShaders[n] == mProgram[prog_id] )
			return n;
	}
	return -1;
}

int	Scene::AddParam ( int prog_id, int id, const char* name )
{
	int prog = mProgram[prog_id];
	int active = 0;
	glGetProgramiv ( prog, GL_ACTIVE_UNIFORMS, &active );
	int ndx = glGetProgramResourceIndex ( prog, GL_UNIFORM, name );	
	int slot = getSlot ( prog_id );
	if ( slot == -1 || ndx == -1 ) {
		gprintf ( "ERROR: Unable to access %s in %s. Active uniforms = %d\n", name, mLastShader.c_str(), active );
		gerror ();
	}
	mParams[slot].p[id] = ndx;
	return ndx;
}

int	Scene::AddAttrib ( int prog_id, int id, const char* name )
{
	int prog = mProgram[prog_id];
	//int ndx = glGetProgramResourceIndex ( prog, GL_BUFFER_VARIABLE, name );		
	int active = 0;
	glGetProgramiv ( prog, GL_ACTIVE_ATTRIBUTES, &active );
	int ndx = glGetAttribLocation ( prog, name );	
	int slot = getSlot ( prog_id );
	if ( slot == -1 || ndx == -1 ) {
		gprintf ( "ERROR: Unable to access %s in %s. Active attribs = %d\n", name, mLastShader.c_str(), active );
		gerror ();
	}
	mParams[slot].p[id] = ndx;
	return ndx;
}

void Scene::SetAspect ( int w, int h )
{
	if ( mCamera != 0x0 ) {
		mCamera->setAspect ( (float) w / (float) h );
	}
}

int Scene::AddMaterial ()
{
	Mat m;
	m.id = static_cast<int>(mMaterials.size());
	mMaterials.push_back ( m );
	return m.id;
}

void Scene::SetMaterialParam ( int id, int p, Vector3DF val )
{
	if ( id < mMaterials.size() ) {
		mMaterials[id].mParam[p] = val;
	}
}


void Scene::SetMaterial ( int model, Vector4DF amb, Vector4DF diff, Vector4DF spec )
{
	clrOverride = false;
	if ( model == -1 ) {
		for (int n=0; n < mModels.size(); n++) {
			mModels[n]->clrAmb = amb;
			mModels[n]->clrDiff = diff;
			mModels[n]->clrSpec = spec;		
		}
	}
	if ( model >= 0 && model < mModels.size() ) {
		mModels[model]->clrAmb = amb;
		mModels[model]->clrDiff = diff;
		mModels[model]->clrSpec = spec;		
	}
}

void Scene::SetOverrideMaterial ( Vector4DF amb, Vector4DF diff, Vector4DF spec )
{
	clrOverride = true;
	clrAmb = amb;
	clrDiff = diff;
	clrSpec = spec;
}

void Scene::LoadPath () 
{
	char path[512];   
	gParse->GetToken( path );
	gScene->AddPath ( path );	
}

void Scene::LoadModel ()  
{
	char modelFile[512];   
	char val[512];
	float scale;
	Vector3DF trans;
	gParse->GetToken( modelFile );
	gParse->GetToken( val );	scale = strtof(val, nullptr);
	gParse->GetToken( val );	trans.x = strtof(val, nullptr);
	gParse->GetToken( val );	trans.y = strtof(val, nullptr);
	gParse->GetToken( val );	trans.z = strtof(val, nullptr);
	gScene->AddModel ( modelFile, scale, trans.x, trans.y, trans.z );
}

void Scene::LoadVolume () 
{
	char volFile[512];   	
	char val[512];
	Vector3DF threshold;
	gParse->GetToken( volFile );									// volume filename
	gParse->GetToken( val );	gScene->mVName = val;				// grid name
	gParse->GetToken( val );	gScene->mVFrames.x = strtof(val, nullptr);
	gParse->GetToken( val );	gScene->mVFrames.y = strtof(val, nullptr);
	gParse->GetToken( val );	gScene->mVFrames.z = strtof(val, nullptr);
	if ( gScene->mVFrames.z == 0 ) gScene->mVFrames.z = 1;
 	gScene->AddVolume ( volFile, Vector3DI(1,1,1), 0, 1.0 );
}

void Scene::VolumeThresh ()
{
	char val[512];
	Vector3DF vec;
	gParse->GetToken( val );	vec.x = strtof(val, nullptr);
	gParse->GetToken( val );	vec.y = strtof(val, nullptr);
	gParse->GetToken( val );	vec.z = strtof(val, nullptr);
	gScene->mVThreshold = vec;
}

void Scene::SetVolumeRange ( float viso, float vmin, float vmax )
{
	gScene->mVThreshold = Vector3DF( viso, vmin, vmax );
}


void Scene::VolumeClip ()
{
	char val[512];
	Vector3DF vec;
	gParse->GetToken( val );	vec.x = strtof(val, nullptr);
	gParse->GetToken( val );	vec.y = strtof(val, nullptr);
	gParse->GetToken( val );	vec.z = strtof(val, nullptr);
	gScene->mVClipMin = vec;
	gParse->GetToken( val );	vec.x = strtof(val, nullptr);
	gParse->GetToken( val );	vec.y = strtof(val, nullptr);
	gParse->GetToken( val );	vec.z = strtof(val, nullptr);
	gScene->mVClipMax = vec;
}

void Scene::LoadGround () 
{
	char hgt[64];	
	char scale[64];	
	gParse->GetToken( hgt );	
	gParse->GetToken( scale );	
	gScene->AddGround ( strToNum( hgt ), strToNum( scale ) );
}

void Scene::LoadCamera () 
{
	std::string line = gParse->ReadNextLine( false );
	std::string var, value;

	gScene->SetCamera ( new Camera3D );

	size_t pos = line.find_first_of ( ':' );
	while ( pos != std::string::npos ) {
		var = line.substr ( 0, pos );
		value = line.substr ( pos+1 );		
		Vector3DF vec; strToVec3( value, "", " ", "", vec.Data() );		
		gScene->UpdateValue ( 'C', 0, strToID(var), vec );		
		line = gParse->ReadNextLine(false);
		pos = line.find_first_of ( ':' );
	}	
}

void Scene::LoadLight () 
{
	std::string line = gParse->ReadNextLine( false );
	std::string var, value;

	Light* light = new Light;
	gScene->SetLight ( gScene->getNumLights(), light );

	size_t pos = line.find_first_of ( ':' );
	while ( pos != std::string::npos ) {
		var = line.substr ( 0, pos );
		value = line.substr ( pos+1 );
		Vector3DF vec; strToVec3( value, "", " ", "", vec.Data() );		
		gScene->UpdateValue ( 'L', 0, strToID(var), vec );			
		line = gParse->ReadNextLine(false);
		pos = line.find_first_of ( ':' );
	}	
}

void Scene::LoadShadow ()
{
	std::string line = gParse->ReadNextLine( false );
	std::string var, value;
	size_t pos = line.find_first_of ( ':' );
	while ( pos != std::string::npos ) {
		var = line.substr ( 0, pos );
		value = line.substr ( pos+1 );
		Vector3DF vec; strToVec3( value, "", " ", "", vec.Data() );
		gScene->UpdateValue ( 'S', 0, strToID(var), vec );			
		line = gParse->ReadNextLine(false);
		pos = line.find_first_of ( ':' );
	}	
}

void Scene::LoadAnimation ()
{
	// Token contains start and end frames
	char buf[512];
	gParse->GetToken ( buf );
	Vector3DF frames;

	std::string line = gParse->ReadNextLine ( false );
	std::string var, obj, str1, str2;
	Vector3DF val1, val2;

	size_t pos = line.find_first_of ( ':' );
	while ( pos != std::string::npos ) {		
		var = line.substr( 0, pos );
		if ( var=="frames" ) {
			str1 = line.substr ( pos+1 );			
			strToVec3 ( str1, "", " ", "", frames.Data() );	
			gScene->setFrameSamples ( static_cast<int>(frames.z) );
		} else {
			obj = strParse ( var, "(", ")" );	// read variable and object
			if ( obj != "" ) {
				str2 = line.substr ( pos+1 );		
				str1 = strSplit ( str2, "," );		// read start and end values
				strToVec3 ( str1, "", " ", "", val1.Data() );
				strToVec3 ( str2, "", " ", "", val2.Data() );			// convert values to vec3
				// gprintf ( "%s: %f %f %f -> %f %f %f\n", obj.c_str(), val1.x, val1.y, val1.z, val2.x, val2.y, val2.z );
				gScene->AddKey ( obj, var,
					static_cast<int>(frames.x), static_cast<int>(frames.y),
					val1, val2 );
			}
		}

		line = gParse->ReadNextLine(false);
		pos = line.find_first_of ( ':' );
	}
}

void Scene::AddKey ( std::string obj, std::string var, int f1, int f2, Vector3DF val1, Vector3DF val2 )
{
	Key k;
	k.f1 = static_cast<float>(f1);
	k.f2 = static_cast<float>(f2);
	k.val1 = val1;
	k.val2 = val2;
	k.obj = obj.at(0);
	k.objid = 0;
	k.varid = strToID ( var );

	mKeys.push_back ( k );
}

bool Scene::DoAnimation ( int frame )
{
	Vector3DF val;
	float u;
	bool anim = false;
	
	if ( mKeys.size() == 0 ) return true;		// No animation.

	for (int n=0; n < mKeys.size(); n++ ) {	
		if ( frame >= mKeys[n].f1 && frame <= mKeys[n].f2 ) {
			u = (frame-mKeys[n].f1) / (mKeys[n].f2 - mKeys[n].f1);
			val = mKeys[n].val2; val -= mKeys[n].val1; val *= u;
			val += mKeys[n].val1;
			UpdateValue ( mKeys[n].obj, mKeys[n].objid, mKeys[n].varid, val );
			anim = true;
		}
	}
	return anim;
}

void Scene::UpdateValue ( char obj, int objid, long varid, Vector3DF val )
{
	if ( obj=='C' ) {
		// Camera			
		Camera3D* cam = getCamera();
		switch ( varid ) {
		case 'look': cam->setToPos ( val.x, val.y, val.z );		break;
		case 'eye ': cam->setPos ( val.x, val.y, val.z );		break;		
		case 'near': cam->setNearFar ( val.x, cam->getFar());	break;
		case 'far ': cam->setNearFar ( cam->getNear(), val.x );	break;				
		case 'fov ': cam->setFov ( val.x );						break;
		case 'dist': cam->setDist ( val.x );					break;
		case 'angs': cam->setOrbit ( val, cam->getToPos(), cam->getOrbitDist(), cam->getOrbitDist() );	break;		
		};
	} else if ( obj=='L' ) {
		// Light
		Light* light = getLight();
		switch ( varid ) {
		case 'look': light->setToPos ( val.x, val.y, val.z );	break;
		case 'pos ': light->setPos ( val.x, val.y, val.z );		break;
		case 'near': light->setNearFar ( val.x, light->getFar());	break;
		case 'far ': light->setNearFar ( light->getNear(), val.x );	break;		
		case 'fov ': light->setFov ( val.x );					break;
		case 'dist': light->setDist ( val.x );					break;
		case 'angs': light->setOrbit ( val, light->getToPos(), light->getOrbitDist(), light->getOrbitDist() );	break;		
		};			
	} else if ( obj=='S' ) {
		// Shadows
		switch ( varid ) {
		case 'x': mShadowParams.x = val.x;	break;
		case 'y': mShadowParams.y = val.x;	break;
		case 'z': mShadowParams.z = val.x;	break;		
		};
	}
}

void Scene::Clear ()
{
	if ( mCamera != 0x0 ) delete mCamera;
	mCamera = 0x0;
	mModels.clear ();
	mLights.clear ();	
	mKeys.clear();
	mMaterials.clear ();
}

void Scene::LoadFile ( std::string filestr )
{
	// Create new parse
	if ( gParse != 0x0 ) delete gParse;
	gParse = new CallbackParser;

	char filepath[1024];
	if ( !FindFile ( filestr, filepath ) )
		return;	

	// Load model(s)

	// Set keywords & corresponding callbacks to process the data
	gParse->SetCallback( "path",            &Scene::LoadPath );
	gParse->SetCallback( "volume",          &Scene::LoadVolume );
	gParse->SetCallback( "vthreshold",      &Scene::VolumeThresh );	
	gParse->SetCallback( "vclip",           &Scene::VolumeClip );	

	gParse->SetCallback( "model",           &Scene::LoadModel );
	gParse->SetCallback( "ground",			&Scene::LoadGround );
	gParse->SetCallback( "camera",			&Scene::LoadCamera );	
	gParse->SetCallback( "light",			&Scene::LoadLight );
	gParse->SetCallback( "animate",			&Scene::LoadAnimation );
	gParse->SetCallback( "shadow",			&Scene::LoadShadow );	

	// Go ahead and parse the file	
	gParse->ParseFile ( filepath, mSearchPaths );	
}

void Scene::RecordKeypoint ( int w, int h )
{
	int frame_delta = 500;

	char fname[512];
	strcpy ( fname, "record.scn" );	
	Camera3D* cam = getCamera ();
	Light* light = getLight ();

	if ( mOutFrame == 0 ) {
		// First keypoint - Write camera & light			
		FILE* fp = fopen ( fname, "wt" );
		fprintf ( fp, "\n" );
		fprintf ( fp, "size\n" );
		fprintf ( fp, "  width: %d\n", w );
		fprintf ( fp, "  height: %d\n", h );
		fprintf ( fp, "\n" );
		fprintf ( fp, " %s\n", mOutModel.c_str() );
		fprintf ( fp, "\n" );
		fprintf ( fp, "camera\n" );
		fprintf ( fp, "  look: %4.3f %4.3f %4.3f\n", cam->getToPos().x, cam->getToPos().y, cam->getToPos().z );
		fprintf ( fp, "  dist: %4.3f\n", cam->getOrbitDist() );
		fprintf ( fp, "  angs: %4.3f %4.3f %4.3f\n", cam->getAng().x, cam->getAng().y, cam->getAng().z );
		fprintf ( fp, "  fov:  %4.3f\n", cam->getFov() );
		fprintf ( fp, "  near: %5.5f\n", cam->getNear() );
		fprintf ( fp, "  far:  %5.5f\n", cam->getFar() );
		fprintf ( fp, "\n\n");
		fprintf ( fp, "light\n" );
		fprintf ( fp, "  look: %4.3f %4.3f %4.3f\n", light->getToPos().x, light->getToPos().y, light->getToPos().z );		
		fprintf ( fp, "  dist: %4.3f\n", light->getOrbitDist() );
		fprintf ( fp, "  angs: %4.3f %4.3f %4.3f\n", light->getAng().x, light->getAng().y, light->getAng().z );
		fprintf ( fp, "  fov:  %4.3f\n", light->getFov() );
		fprintf ( fp, "  near: %5.5f\n", light->getNear() );
		fprintf ( fp, "  far:  %5.5f\n", light->getFar() );
		fprintf ( fp, "\n\n");
	} else {
		// Later keypoints - Write animation
		FILE* fp = fopen ( fname, "a+t" );
		fprintf ( fp, "animate\n" );
		fprintf ( fp, "  frames: %d %d\n", mOutFrame, mOutFrame + frame_delta );
		fprintf ( fp, "  look (C): %4.3f %4.3f %4.3f, %4.3f %4.3f %4.3f\n", mOutCam->getToPos().x, mOutCam->getToPos().y, mOutCam->getToPos().z, cam->getToPos().x, cam->getToPos().y, cam->getToPos().z );
		fprintf ( fp, "  dist (C): %4.3f, %4.3f\n", mOutCam->getOrbitDist(), cam->getOrbitDist() );
		fprintf ( fp, "  angs (C): %4.3f %4.3f %4.3f, %4.3f %4.3f %4.3f\n", mOutCam->getAng().x, mOutCam->getAng().y,mOutCam->getAng().z, cam->getAng().x, cam->getAng().y, cam->getAng().z );
		fprintf ( fp, "  fov (C):  %4.3f, %4.3f\n", mOutCam->getFov(), cam->getFov() );				
		fprintf ( fp, "  look (L): %4.3f %4.3f %4.3f, %4.3f %4.3f %4.3f\n", mOutLight->getToPos().x, mOutLight->getToPos().y, mOutLight->getToPos().z,light->getToPos().x, light->getToPos().y, light->getToPos().z );		
		fprintf ( fp, "  dist (L): %4.3f, %4.3f\n", mOutLight->getOrbitDist(), light->getOrbitDist() );
		fprintf ( fp, "  angs (L): %4.3f %4.3f %4.3f, %4.3f %4.3f %4.3f\n", mOutLight->getAng().x, mOutLight->getAng().y,mOutLight->getAng().z, light->getAng().x, light->getAng().y, light->getAng().z );
		fprintf ( fp, "  fov (L):  %4.3f, %4.3f\n", mOutLight->getFov(), light->getFov() );		
		fprintf ( fp, "\n\n");	
		
		mOutFrame += frame_delta;
	}
	mOutCam->Copy ( *cam );
	mOutLight->Copy ( *light );	
}

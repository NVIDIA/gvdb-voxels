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
// Chris Wyman (9/2/2014)                                    
//

#include "loader_OBJReader.h"

#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <float.h>
#include <string.h>

struct OBJTri
{
	OBJTri() {}
	int vIdx[3], nIdx[3];
};

namespace {
	inline int Max( int x, int y )         { return (x>y)?x:y; }
	inline int Min( int x, int y )         { return (x<y)?x:y; }
	inline float Max( float x, float y )   { return (x>y)?x:y; }
	inline float Min( float x, float y )   { return (x<y)?x:y; }
}

// prototype from simpleParser.cpp
char *StripLeadingTokenToBuffer( char *string, char *buf );

OBJReader::OBJReader() :
	m_hasNormals(false), m_hasVertices(false),                // These flags are set to true when we encounter a facet line in the OBJ that uses normals or verts
	m_resize( false ), m_center( false ), m_guessNorms( true ), // These are usually parameters, but for this demo we have good defaults to hardcode.
	m_numVertices(0), m_numNormals(0), m_numTris(0),          // We start having read in 0 normals, vertices, and triangles
	m_triangles(0), m_normals(0), m_vertices(0),              // Set our arrays to NULL
	m_allocVert(0), m_allocNorm(0), m_allocTri(0)	
{
}

OBJReader::~OBJReader()
{
	Cleanup();
}

bool OBJReader::Cleanup ()
{
	if ( m_normals != 0 ) 	delete[] m_normals;
	if ( m_vertices != 0 )	delete[] m_vertices;
	if ( m_triangles != 0 )	free( m_triangles );
	return true;
}

bool OBJReader::isMyFile( const char* filename )
{
	return getExtension( filename ) == "obj";
}

bool OBJReader::LoadFile ( Model* model, const char *filename, std::vector<std::string>& paths )
{
	ParseFile ( filename, paths );

	// For clarity (later), we have a pointer to the Read_???_Token() method we will
	//    use when reading a facet.  This will be set when we first see a 'f' line.
	FnParserPtr fnFacetParsePtr = NULL;

	// OK, we have a parser.  Now parse through the file
	int lineCount = 0;
	OBJTri *curTri = 0;
	char *linePtr = 0, *mtlFilePtr = 0;
	char keyword[64], vertToken[128];

	while ( (linePtr = this->ReadNextLine()) != NULL )
	{
		// Each OBJ line starts with a keyword/keyletter
		this->GetLowerCaseToken( keyword );
		
		switch( keyword[0] )
		{
		case 'v': 
			if (keyword[1] == 'n')        // We found a normal!
				AddNormal( this->GetVec3() ); // m_objNorms.Add( this->GetVec3() );
			else if (keyword[1] == 't')   // Ignore texture coordinates in this demo
				; 
			else if (keyword[1] == 0 )    // We found a vertex!
			{
				AddVertex( this->GetVec3() ); // m_objVerts.Add( this->GetVec3() );  
			}
			break;
		case 'm':  // For this simple demo... We're ignoring materials, so do nothing.
		case 'o':  // For this simple demo... We're ignoring object designations, so do nothing.
		case 'g':  // For this simple demo... We're ignoring group designations, so do nothing.
		case 'u':  // For this simple demo... We're ignoring materials, so do nothing.
		case 's':  // There's a smoothing command.  We're ignoring these.
			break;
		case 'f': // We found a facet!
			if (keyword[1] != 0)  // There's some odd models out there with 'fo' or other
				continue;         //    additional facet types.  Mostly these are junk!
			
			//Ensure the 'f' line has at least three entries. 
			if(!IsValidFLine(&fnFacetParsePtr))
				continue;

			curTri = new OBJTri();

			// There are multiple different formats for 'f' lines.  Decide which it is,
			//    and select the appropriate class member we'll use to read this line.
			SelectReadMethod( &fnFacetParsePtr );

			// Read first three set of indices on this line
			((*this).*(fnFacetParsePtr))( curTri, 0, NULL );
			((*this).*(fnFacetParsePtr))( curTri, 1, NULL );
			((*this).*(fnFacetParsePtr))( curTri, 2, NULL );
			AddTriangle( curTri );  // m_objTris.Add( curTri );

			// Do we have more vertices in this facet?  Check to see if the 
			//    next token is non-empty (if empty the [0]'th char == 0)
			//    If we do, we'll have to triangulate the facet, so make a new tri
			this->GetToken( vertToken );
			while ( vertToken[0] ) 
			{
				curTri = new OBJTri();
				CopyForTriangleFan( curTri );
				((*this).*(fnFacetParsePtr))( curTri, 2, vertToken );
				AddTriangle( curTri ); // m_objTris.Add( curTri );
				this->GetToken( vertToken );
			}
			break;

		default:  // We have no clue what to do with this line....
			this->WarningMessage("Found corrupt line in OBJ.  Unknown keyword '%s'", keyword);
		}
	}

	// No need to keep the file hanging around open.
	CloseFile();
	
	//AddVertex(Vector4DF( 47.2224* 0.01, 0.0117588* 0.01, 46.8401* 0.01,  0.0f ) );
	//AddVertex(Vector4DF(47.2224* 0.01, 10.0784* 0.01, 46.8401* 0.01,   0.0f ));
	//AddVertex(Vector4DF( 47.2224* 0.01, 10.0784* 0.01, 47.8401* 0.01, 0.0f ));
	//AddVertex(Vector4DF( 47.2224* 0.01, 0.0117588* 0.01, 47.8401* 0.01, 0.0f ));

	//curTri = new OBJTri();
	//curTri->vIdx[0] = m_numVertices - 1; 
	//curTri->vIdx[1] = m_numVertices - 2;
	//curTri->vIdx[2] = m_numVertices - 3;
	//AddTriangle( curTri );

	//curTri = new OBJTri();
	//curTri->vIdx[0] = m_numVertices - 4; 
	//curTri->vIdx[1] = m_numVertices - 2;
	//curTri->vIdx[2] = m_numVertices - 3;
	//AddTriangle( curTri );

	// If we already have surface normals, there's no need to use facet normals
	if (m_hasNormals) m_guessNorms = false;

	// Create the GPU buffers for this object, so we have them laying around later.
	GetCompactArrayBuffer( model );


	// OK.  Now we've created our vertex array.  Enable the attributes so that we're sending the correct
	//    data to all our shaders.

	//m_vertArr->EnableAttribute( IGLU_ATTRIB_VERTEX, 3, GL_FLOAT, m_vertStride, BUFFER_OFFSET(m_vertOff));
	//if (HasNormals())  {
	//	m_vertArr->EnableAttribute( IGLU_ATTRIB_NORMAL, 3, GL_FLOAT, m_vertStride, BUFFER_OFFSET(m_normOff));
	//}

	// gprintf( " Model reading completed successfully! (%d verts, %d tris)\n", m_numVertices, m_numTris );

	return true;
}

unsigned int OBJReader::GetVertexIndex( int relativeIdx )
{
	if (relativeIdx == 0)
	{
		this->WarningMessage("Unexpected OBJ vertex index of 0!");
		relativeIdx = 1;  // 0 is undefined, and will crash.  Give at least a non-segfaultable value;
	}
	return relativeIdx > 0 ? relativeIdx-1 : m_numVertices+relativeIdx;
}

unsigned int OBJReader::GetNormalIndex( int relativeIdx )
{
	if (relativeIdx == 0)
		this->WarningMessage("Unexpected OBJ normal index of 0!");
	// Note the >=, which is different than the > in the vertex index.  An index
	//    of 0 is invalid.  Our code treats a -1 return value as "no normal".  That' fine
	//    for nomrals and texture coords
	return relativeIdx >= 0 ? relativeIdx-1 : m_numNormals+relativeIdx;
}

void OBJReader::Read_V_Token( OBJTri *tri, int idx, char *token )
{
	char vertToken[128], *tPtr = vertToken;
	int vIdx;

	// If the user didn't pass in a token, read one
	if (!token)
		this->GetToken( vertToken );
	else 
		tPtr = token;

	// Parse a vertex & normal from the token
	sscanf( tPtr, "%d", &vIdx );

	// Resolve these into indicies in our data structure (not the OBJ file)
	tri->vIdx[idx] = GetVertexIndex( vIdx );
	tri->nIdx[idx] = -1;
	//tri->tIdx[idx] = -1;
}

void OBJReader::Read_VN_Token( OBJTri *tri, int idx, char *token )
{
	char vertToken[128], *tPtr = vertToken;
	int vIdx, nIdx;

	// If the user didn't pass in a token, read one
	if (!token)
		this->GetToken( vertToken );
	else 
		tPtr = token;

	// Parse a vertex & normal from the token
	sscanf( tPtr, "%d//%d", &vIdx, &nIdx );

	// Resolve these into indicies in our data structure (not the OBJ file)
	tri->vIdx[idx] = GetVertexIndex( vIdx );
	tri->nIdx[idx] = GetNormalIndex( nIdx );
	//tri->tIdx[idx] = -1;
}

void OBJReader::Read_VT_Token( OBJTri *tri, int idx, char *token )
{
	char vertToken[128], *tPtr = vertToken;
	int vIdx, tIdx;

	// If the user didn't pass in a token, read one
	if (!token)
		this->GetToken( vertToken );
	else 
		tPtr = token;

	// Parse a vertex & normal from the token
	sscanf( tPtr, "%d/%d", &vIdx, &tIdx );

	// Resolve these into indicies in our data structure (not the OBJ file)
	tri->vIdx[idx] = GetVertexIndex( vIdx );
	tri->nIdx[idx] = -1;
	// tri->tIdx[idx] = GetTextureIndex( tIdx ); // Ignore texture coordinates
}

void OBJReader::Read_VTN_Token( OBJTri *tri, int idx, char *token )
{
	char vertToken[128], *tPtr = vertToken;
	int vIdx, tIdx, nIdx;

	// If the user didn't pass in a token, read one
	if (!token)
		this->GetToken( vertToken );
	else 
		tPtr = token;

	// Parse a vertex & normal from the token
	sscanf( tPtr, "%d/%d/%d", &vIdx, &tIdx, &nIdx );

	// Resolve these into indicies in our data structure (not the OBJ file)
	tri->vIdx[idx] = GetVertexIndex( vIdx );
	tri->nIdx[idx] = GetNormalIndex( nIdx );
	// tri->tIdx[idx] = GetTextureIndex( tIdx ); // Ignore texture coordinates
}

void OBJReader::CopyForTriangleFan( OBJTri *newTri )
{
	newTri->vIdx[0] = m_triangles[m_numTris-1]->vIdx[0]; 
	newTri->nIdx[0] = m_triangles[m_numTris-1]->nIdx[0];
	//newTri->tIdx[0] = m_triangles[m_numTris-1]->tIdx[0];
	newTri->vIdx[1] = m_triangles[m_numTris-1]->vIdx[2];
	newTri->nIdx[1] = m_triangles[m_numTris-1]->nIdx[2];
	//newTri->tIdx[1] = m_triangles[m_numTris-1]->tIdx[2];
}

void OBJReader::SelectReadMethod( FnParserPtr *pPtr )
{
	int v, t, n; // garbage vars
	char vertToken[128];
	char *lPtr = this->GetCurrentLinePointer();

	// Non-destructively parse the next token
	StripLeadingTokenToBuffer( lPtr, vertToken );

	if (strstr(vertToken, "//"))                               // Then it has the v//n format
	{
		*pPtr = &OBJReader::Read_VN_Token;
		m_hasVertices = m_hasNormals = true;
	}

	else if ( sscanf(vertToken, "%d/%d/%d", &v, &t, &n) == 3 ) // Then it has the v/t/n format
	{
		*pPtr = &OBJReader::Read_VTN_Token;
		m_hasVertices = m_hasNormals = m_hasTexCoords = true;
	}

	else if ( sscanf(vertToken, "%d/%d", &v, &t) == 2 )        // Then it has the v/t format
	{
		*pPtr = &OBJReader::Read_VT_Token;
		m_hasVertices = m_hasTexCoords = true;
	}else                                                     // Then it has the v format
		*pPtr = &OBJReader::Read_V_Token;
		m_hasVertices = true;
	}
	
bool OBJReader::IsValidFLine(FnParserPtr *pPtr)
{

	//Garbage variables. This will break if it comes accross an entry on a line
	//longer than 128 characters. 
	char c0[128];
	char c1[128];
	char c2[128];
	char vertToken[128]; 

	// Non-destructively parse the next token
	char *lPtr = this->GetCurrentLinePointer();
	StripLeadingTokenToBuffer( lPtr, vertToken );

	//Make sure we have at least three entries in this vertex
	if(sscanf(lPtr, "%s %s %s", c0, c1, c2) >= 3){
		return true;
	}else{
		this->WarningMessage("Corrupt 'f': %s", lPtr);
		return false; 
	}
}

void OBJReader::AddDataToArray( float *arr, int startIdx, Vector4DF *vert, Vector4DF *norm )
{	
	int i = startIdx;

	// Add the vertex
	arr[i++] = vert->X();
	arr[i++] = vert->Y();
	arr[i++] = vert->Z();

	// If this vertex has a normal, add it.
	if (norm)
	{
		arr[i++] = norm->X();
		arr[i++] = norm->Y();
		arr[i++] = norm->Z();
	}
}

void OBJReader::CenterAndResize( float *arr, int numVerts )
{
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

	int startIdx   = m_vertOff / sizeof(float);
	int vertStride = m_vertStride / sizeof(float);
	float minX = FLT_MAX, minY = FLT_MAX, minZ = FLT_MAX;
	float maxX = -FLT_MAX, maxY = -FLT_MAX, maxZ = -FLT_MAX;

	// Walk through vertex array, find max & min (x, y, z)
	for (int i=0; i<numVerts; i++)
	{
		int idx = startIdx + i*vertStride;
		minX = Min( minX, arr[idx+0] );
		maxX = Max( maxX, arr[idx+0] );
		minY = Min( minY, arr[idx+1] );
		maxY = Max( maxY, arr[idx+1] );
		minZ = Min( minZ, arr[idx+2] );
		maxZ = Max( maxZ, arr[idx+2] );
	}

	// Find the center of the object & the width of the object
	float ctrX = (m_resize || m_center ? 0.5f*(minX+maxX) : 0.0f );
	float ctrY = (m_resize || m_center ? 0.5f*(minY+maxY) : 0.0f ); 
	float ctrZ = (m_resize || m_center ? 0.5f*(minZ+maxZ) : 0.0f );
	float deltaX = 0.5f*(maxX-minX), deltaY = 0.5f*(maxY-minY), deltaZ = 0.5f*(maxZ-minZ);
	float delta = ( m_resize ? 
		            Max( deltaX, Max( deltaY, deltaZ ) ) :
	                1.0f );

	// Walk through the vertex array and update the positions
	for (int i=0; i<numVerts; i++)
	{
		int idx = startIdx + i*vertStride;
		arr[idx+0] = (arr[idx+0] - ctrX ) / delta;
		arr[idx+1] = (arr[idx+1] - ctrY ) / delta;
		arr[idx+2] = (arr[idx+2] - ctrZ ) / delta;
	}
}

void OBJReader::GetCompactArrayBuffer( Model* model )
{

	// Create an OBJ vert ID -> element array vert ID map.  Init all entries to 0xFFFFFFFF.
	//    Also, create mapping vertID -> last normal ID used for this vertex
	//          create mapping vertID -> last tex coord used for this vertex
	unsigned int *vertMapping = (unsigned int *)malloc( m_numVertices * sizeof( unsigned int ) );
	unsigned int *normMapping = (unsigned int *)malloc( m_numVertices * sizeof( unsigned int ) );
	memset( vertMapping, 0xFF, m_numVertices * sizeof( unsigned int ) );
	memset( normMapping, 0xFF, m_numVertices * sizeof( unsigned int ) );

	//   We'll have 1 float for a material ID
	//   We'll have 3 floats (x,y,z) for each of the 3 verts of each triangle 
	//   We'll have 3 floats (x,y,z) for each of the 3 normals of each triangle
	//   We'll have 2 floats (u,v) for each of the 3 texture coordinates of each triangle
	unsigned int  numComponents = 3 + (m_hasNormals||m_guessNorms ? 3 : 0);

	int bufSz  = numComponents * sizeof( float ) * (3 * m_numTris);
	m_vertStride = numComponents * sizeof( float );
	m_vertOff    = 0 * sizeof( float );
	m_normOff    = (m_hasNormals||m_guessNorms? 3 : 0) * sizeof( float );
	// gprintf ("    (*) Stride: %d, Offsets: v %d, m %d, o %d, n %d t %d\n", m_vertStride, m_vertOff, 0, 0, m_normOff, 0);

	// Add a vertex buffer to the model
	if ( model->vertBuffer != 0x0 )		free ( model->vertBuffer );
	model->vertBuffer = (float*)		malloc ( bufSz );
	if (!model->vertBuffer) this->ErrorMessage("Memory allocation error during .obj read!");
	float* tmpBuf = (float*) model->vertBuffer;							// cast to float* to populate it	
	
	// Add an element buffer to the model
	unsigned int elembufSz = 3 * sizeof( unsigned int ) * m_numTris;
	if ( model->elemBuffer != 0x0 )		free ( model->elemBuffer );
	model->elemBuffer = (unsigned int*)	malloc( elembufSz );
	unsigned int *tmpElemBuf = (unsigned int *) model->elemBuffer;		// cast to unsigned int* to populate it

	// We'll need to know the size of our two arrays
	int numArrayElements = 3 * m_numTris;         // Known in advance
	int numArrayVerts    = 0;                     // Depends on how many verts are reused.  We'll compute

	// A location to store normal guesses
	Vector4DF tmpNorm;
	Vector4DF *normGuess = (m_guessNorms ? &tmpNorm : 0);


	for (unsigned int i=0, triNum=0; triNum < m_numTris; i+=3,triNum++ )
	{
		int i0 = m_triangles[triNum]->vIdx[0];
		int i1 = m_triangles[triNum]->vIdx[1];
		int i2 = m_triangles[triNum]->vIdx[2];

		int *nIdx = m_triangles[triNum]->nIdx;
		//int *tIdx = m_triangles[triNum]->tIdx;

		if ( i0 < 0 || i0 > int(m_numVertices) ) continue;
		if ( i1 < 0 || i1 > int(m_numVertices) ) continue;
		if ( i2 < 0 || i2 > int(m_numVertices) ) continue;

		tmpNorm = ((m_vertices[i1]-m_vertices[i0]).Cross(m_vertices[i2]-m_vertices[i0])).Normalize();

		if (    (vertMapping[i0] == 0xFFFFFFFF) // We haven't seen this vertex yet.  Add to list
		     || (m_hasNormals && normMapping[i0] != m_triangles[triNum]->nIdx[0]) )    // We saw this vertex...  but w/different normal
		{
			AddDataToArray( tmpBuf, numArrayVerts*numComponents, &m_vertices[m_triangles[triNum]->vIdx[0]], (m_hasNormals && (nIdx[0]>=0)) ? &m_normals[nIdx[0]] : normGuess );
			tmpElemBuf[i]   = numArrayVerts;
			vertMapping[i0] = numArrayVerts++;
			normMapping[i0] = nIdx[0];
		}
		else                               // We've already seen vertex; reuse it.
			tmpElemBuf[i] = vertMapping[i0];

		if (    (vertMapping[i1] == 0xFFFFFFFF) // We haven't seen this vertex yet.  Add to list
		     || (m_hasNormals && normMapping[i1] != m_triangles[triNum]->nIdx[1]) )    // We saw this vertex...  but w/different normal
		{
			AddDataToArray( tmpBuf, numArrayVerts*numComponents, &m_vertices[m_triangles[triNum]->vIdx[1]], (m_hasNormals && (nIdx[1]>=0)) ? &m_normals[nIdx[1]] : normGuess );
			tmpElemBuf[i+1]   = numArrayVerts;
			vertMapping[i1] = numArrayVerts++;
			normMapping[i1] = nIdx[1];
		}
		else                               // We've already seen vertex; reuse it.
			tmpElemBuf[i+1] = vertMapping[i1];

		if (    (vertMapping[i2] == 0xFFFFFFFF) // We haven't seen this vertex yet.  Add to list
		     || (m_hasNormals && normMapping[i2] != m_triangles[triNum]->nIdx[2]) )    // We saw this vertex...  but w/different normal
		{
			AddDataToArray( tmpBuf, numArrayVerts*numComponents, &m_vertices[m_triangles[triNum]->vIdx[2]], (m_hasNormals && (nIdx[2]>=0)) ? &m_normals[nIdx[2]] : normGuess );
			tmpElemBuf[i+2]   = numArrayVerts;
			vertMapping[i2] = numArrayVerts++;
			normMapping[i2] = nIdx[2];
		}
		else                               // We've already seen vertex; reuse it.
			tmpElemBuf[i+2] = vertMapping[i2];
	}

	// If the user asked us to resize & center the object, do that.
	
	// NOTE: Should be moved out of OBJReader, as it is really
	// a post-process applied to mesh and load. Also, messes up animations.
	/* if (m_resize || m_center)
		CenterAndResize( tmpBuf, numArrayVerts ); */

	// Copy our arrays into model GPU buffers
	model->elemDataType      = GL_TRIANGLES;           // What type of GL-renderable primitive does this model contain (e.g., GL_TRIANGLES)
	model->elemCount	     = m_numTris;          // How many of the above primitives are there in the model?
	model->elemArrayOffset   = 0;                      // In the element index array, what's the byte offset to the index of the first element to use?  
	model->elemStride		 = 3 * sizeof(unsigned int);

	model->vertCount		 = numArrayVerts;
	model->vertDataType       = GL_FLOAT;               // What is the GL data type of each component in the vertex positions?
	model->vertComponents	 = 3;                      // How many components are in each vertex position attribute?
	model->vertOffset         = m_vertOff;              // What's the byte offset to the start of the first vertex position in the vertex buffer?
	model->vertStride         = m_vertStride;           // How many bytes are subsequent verts separated by in the interleaved vertex array created below?
	
	model->normDataType      = GL_FLOAT;               // What is the GL data type of each component in the vertex normals?
	model->normComponents	 = 3;                      // How many components are in each vertex normal attribute?
	model->normOffset        = m_normOff;              // What's the byte offset to the start of the first vertex normal in the vertex buffer?

	// Free our temporary copy of the data
	free( vertMapping );
	free( normMapping );

	// If we asked to guess normals, we've done it.  Treat everything hereon as if we had norms:
	if (m_guessNorms) m_hasNormals = true;
}



void OBJReader::AddVertex( const Vector4DF &vert )
{
	if ( m_numVertices >= m_allocVert )
	{
		unsigned int oldAlloc = m_allocVert;
		if (oldAlloc <= 0)
		{
			m_allocVert = 1000000;
			m_vertices = new Vector4DF[ m_allocVert ];
		}
		else
		{
			m_allocVert = oldAlloc + 1000000;
			Vector4DF *ptr = m_vertices;
			m_vertices = new Vector4DF[ m_allocVert ];
			for (unsigned int i=0; i<oldAlloc; i++)
				m_vertices[i] = ptr[i];
			delete[] ptr;
		}
	}
	m_vertices[ m_numVertices ] = vert;
	m_numVertices++;
}

void OBJReader::AddNormal( const Vector4DF &norm )
{
	if ( m_numNormals >= m_allocNorm )
	{
		unsigned int oldAlloc = m_allocNorm;
		if (oldAlloc <= 0)
		{
			m_allocNorm = 1000000;
			m_normals = new Vector4DF[ m_allocNorm ];
		}
		else
		{
			m_allocNorm = oldAlloc + 1000000;
			Vector4DF *ptr = m_normals;
			m_normals = new Vector4DF[ m_allocNorm ];
			for (unsigned int i=0; i<oldAlloc; i++)
				m_normals[i] = ptr[i];
			delete[] ptr;
		}
	}
	m_normals[ m_numNormals ] = norm;
	m_numNormals++;
}

void OBJReader::AddTriangle( OBJTri *tri )
{
	if ( m_numTris >= m_allocTri )
	{
		unsigned int oldAlloc = m_allocTri;
		if (oldAlloc <= 0)
		{
			m_allocTri = 1000000;
			m_triangles = (OBJTri **)malloc( m_allocTri * sizeof (OBJTri *) );
		}
		else
		{
			m_allocTri = oldAlloc + 1000000;
			OBJTri **ptr = m_triangles;
			m_triangles = (OBJTri **)malloc( m_allocTri * sizeof (OBJTri *) );
			for (unsigned int i=0; i<oldAlloc; i++)
				m_triangles[i] = ptr[i];
			free( ptr );
		}
	}
	m_triangles[ m_numTris ] = tri;
	m_numTris++;
}

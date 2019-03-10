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

#include "loader_Parser.h"
#include "string_helper.h"

#include <cstring>
#include <cctype>

#pragma warning( disable : 4996 )


// Anonymous namespace containing a bunch of simple, stupid helper functions. 
//     Probably could use some standard library, but after these have followed
//     me around so long, it's easier to copy than to find the 'right' way to do this
namespace {
	// takes an entire string, and makes it all lowercase 
	void MakeLower( char *buf )
	{
	  char *tmp = buf;
	  if (!buf) return;

	  while ( tmp[0] != 0 )
		{
		  *tmp = (char)tolower( *tmp );
		  tmp++;
		}
	}

	// takes an entire string, and makes it all uppercase 
	void MakeUpper( char *buf )
	{
	  char *tmp = buf;
	  if (!buf) return;

	  while ( tmp[0] != 0 )
		{
		  *tmp = (char)toupper( *tmp );
		  tmp++;
		}
	}

	// Returns a ptr to the first non-whitespace character in a string 
	char *StripLeadingWhiteSpace( char *string )
	{
	  char *tmp = string;
	  if (!string) return 0;

	  while ( (tmp[0] == ' ') ||
		  (tmp[0] == '\t') ||
		  (tmp[0] == '\n') ||
		  (tmp[0] == '\r') )
		tmp++;

	  return tmp;
	}

	// Exactly the same as strip leading white space, but also considers 2 extra characters
	// as whitespace to be removed.
	char *StripLeadingSpecialWhiteSpace( char *string, char special, char special2 )
	{
	  char *tmp = string;
	  if (!string) 
		  return 0;

	  while ( (tmp[0] == ' ') ||
		  (tmp[0] == '\t') ||
		  (tmp[0] == '\n') ||
		  (tmp[0] == '\r') ||
		  (tmp[0] == special) ||
		  (tmp[0] == special2) )
		tmp++;
  
	  return tmp;
	}
} // end anonymous namespace

std::string getExtension( const std::string& filename )
{
// Get the filename extension
std::string::size_type extension_index = filename.find_last_of( "." );
return extension_index != std::string::npos ?
		filename.substr( extension_index+1 ) :
		std::string();
}

// Returns the first contiguous string of non-whitespace characters in the buffer
// and returns a pointer to the next non-whitespace character (if any) in the string 
char *StripLeadingTokenToBuffer( char *string, char *buf )
{
	char *tmp = string;
	char *out = buf;
	if (!string) 
		return 0;

	while ( (tmp[0] != ' ') &&
		(tmp[0] != '\t') &&
		(tmp[0] != '\n') &&
		(tmp[0] != '\r') &&
		(tmp[0] != 0) )
	{
		*(out++) = *(tmp++); 
	}
	*out = 0;

	return StripLeadingWhiteSpace( tmp );
}

namespace {
	// Returs the first contiguous string of decimal, numerical characters in the buffer and returns
	// that character string (which atof() can be applied to).  Handles scientific notation.  Also 
	// performs some minor cleaning of commas and parentheses prior to the number.
	char *StripLeadingNumericalToken( char *string, char *result )
	{
	  char *tmp = string;
	  char buf[80];
	  char *ptr = buf;
	  char *ptr2;
	  if (!string) 
		  return 0;

	  // If there are any commas or ( before the next number, remove them.
	  tmp = StripLeadingSpecialWhiteSpace( tmp, ',', '(' );
	  tmp = StripLeadingTokenToBuffer( tmp, buf );
  
	  /* find the beginning of the number */
	  while( (ptr[0] != '-') &&
		 (ptr[0] != '.') &&
		 ((ptr[0]-'0' < 0) ||
		  (ptr[0]-'9' > 0)) )
		ptr++;

	  /* find the end of the number */
	  ptr2 = ptr;
	  while( (ptr2[0] == '-') ||
		 (ptr2[0] == '.') ||
		 ((ptr2[0]-'0' >= 0) && (ptr2[0]-'9' <= 0)) ||
		 (ptr2[0] == 'e') || (ptr2[0] == 'E'))
		ptr2++;

	  /* put a null at the end of the number */
	  ptr2[0] = 0;

	  /* copy the numerical portion of the token into the result */
	  ptr2 = ptr;
	  ptr = result;
	  while (ptr2[0])
		*(ptr++) = *(ptr2++);
	  ptr[0] = 0;

	  return tmp;
	}

};  // end anonymous namespace



Parser::Parser() :
	f(NULL), lineNum(0), fileName(0), m_closed(true), fileSize(-1)
{

}

void Parser::ParseFile ( const char *fname, std::vector<std::string>& paths ) 	
{
	// Check for this file in the current path, then all the search paths specified
	char fileName[1024];

	// Locate the file
	if ( !getFileLocation ( fname, fileName, paths ) ) {
		gprintf ("Error: Parser unable to find '%s'\n", fname );
		gerror ();
	}
	
	// Open file
	f = fopen ( fileName, "rb" );	
	if ( f == 0x0 ) {
		gprintf ("Error: Unable to open file '%s'\n", fileName );
		gerror ();	
	}
	m_closed = false;

	char *fptr = strrchr( fileName, '/' );
	char *rptr = strrchr( fileName, '\\' );
	if (!fptr && !rptr)
		unqualifiedFileName = fileName;
	else if (!fptr && rptr)
		unqualifiedFileName = rptr+1;
	else if (fptr && !rptr)
		unqualifiedFileName = fptr+1;
	else
		unqualifiedFileName = (fptr>rptr ? fptr+1 : rptr+1);

	fileDirectory = (char *)malloc( (unqualifiedFileName-fileName+1)*sizeof(char)); 
	strncpy( fileDirectory, fileName, unqualifiedFileName-fileName );
	fileDirectory[unqualifiedFileName-fileName]=0;

	// Get a filesize and remember where it is.  (XXX) This is, evidently, non-portable,
	//    though I've never seen a system where it didn't work.
	if (fseek(f, 0, SEEK_END)==0)  // If we went to the end of the file
	{
		fflush(f);
		fileSize = ftell(f);
		rewind(f);
	}
}

Parser::~Parser()
{
	if (!m_closed) fclose( f );
	if (fileName != 0x0 ) free( fileName );
}

void Parser::CloseFile( void )
{
	fclose( f );
	m_closed = true;
}


char *Parser::ReadNextLine( bool discardBlanks )
{
	char *haveLine;

	// Get the next line from the file, and increment our line counter
	while ( (haveLine = __ReadLine()) && discardBlanks )
	{
		char *ptr = StripLeadingWhiteSpace( internalBuf );

		// If we start with a comment or have no non-blank characters read a new line.
		if (ptr[0] == '#' || ptr[0] == 0) 
			continue;
		else
			break;
	}
	if (!haveLine) return NULL;

	// Replace '\n' at the end of the line with 0.
	char *tmp = strrchr( internalBuf, '\n' );
	if (tmp) tmp[0] = 0;

	// When we start looking through the line, we'll want the first non-whitespace
	internalBufPtr = StripLeadingWhiteSpace( internalBuf );
	return internalBufPtr;
}

char *Parser::__ReadLine( void )
{
	if (m_closed) 
		return 0;
	char *ptr = fgets(internalBuf, 1023, f);
	if (ptr) lineNum++;
	return ptr;
}

char *Parser::GetToken( char *token )
{
	internalBufPtr = StripLeadingTokenToBuffer( internalBufPtr, token );
	return internalBufPtr;
}

char *Parser::GetLowerCaseToken( char *token )
{
	internalBufPtr = StripLeadingTokenToBuffer( internalBufPtr, token );
	MakeLower( token );
	return internalBufPtr;
}

char *Parser::GetUpperCaseToken( char *token )
{
	internalBufPtr = StripLeadingTokenToBuffer( internalBufPtr, token );
	MakeUpper( token );
	return internalBufPtr;
}

int		 Parser::GetInteger( void )
{
	char token[ 128 ];
	internalBufPtr = StripLeadingNumericalToken( internalBufPtr, token );
	return (int)atoi( token );
}

unsigned Parser::GetUnsigned( void )
{
	char token[ 128 ];
	internalBufPtr = StripLeadingNumericalToken( internalBufPtr, token );
	return (unsigned)atoi( token );
}

float	 Parser::GetFloat( void )
{
	char token[ 128 ];
	internalBufPtr = StripLeadingNumericalToken( internalBufPtr, token );
	return (float)atof( token );
}

double	 Parser::GetDouble( void )
{
	char token[ 128 ];
	internalBufPtr = StripLeadingNumericalToken( internalBufPtr, token );
	return atof( token );
}


Vector4DF	 Parser:: GetVec4( void )
{
	float x = GetFloat();
	float y = GetFloat();
	float z = GetFloat();
	float w = GetFloat();
	return Vector4DF( x, y, z, w );
}

Vector4DF	 Parser:: GetVec3( void )
{
	float x = GetFloat();
	float y = GetFloat();
	float z = GetFloat();
	return Vector4DF( x, y, z, 0.0f );
}

Matrix4F Parser::Get4x4Matrix( void )
{
	float mat[16];
	for (int i=0; i<16; i++)
		mat[i] = GetFloat();
	return Matrix4F( mat );
}

char *	 Parser::GetInteger( int *result )
{
	char token[ 128 ];
	internalBufPtr = StripLeadingTokenToBuffer( internalBufPtr, token );
	*result = (int)atoi( token );
	return internalBufPtr;
}

char *	 Parser::GetUnsigned( unsigned *result )
{
	char token[ 128 ];
	internalBufPtr = StripLeadingTokenToBuffer( internalBufPtr, token );
	*result = (unsigned)atoi( token );
	return internalBufPtr;
}

char *	 Parser::GetFloat( float *result )
{
	char token[ 128 ];
	internalBufPtr = StripLeadingTokenToBuffer( internalBufPtr, token );
	*result = (float)atof( token );
	return internalBufPtr;
}

char *	 Parser::GetDouble( double *result )
{
	char token[ 128 ];
	internalBufPtr = StripLeadingTokenToBuffer( internalBufPtr, token );
	*result = atof( token );
	return internalBufPtr;
}

void Parser::ResetProcessingForCurrentLine( void )
{
	internalBufPtr = StripLeadingWhiteSpace( internalBuf );
}

void Parser::WarningMessage( const char *msg )
{
	gprintf ( "Warning: %s (%s, line %d)\n", msg, GetFileName(), GetLineNumber() );
}

void Parser::WarningMessage( const char *msg, const char *paramStr )
{
	char buf[ 256 ];
	sprintf( buf, msg, paramStr );
	gprintf ( "Warning: %s (%s, line %d)\n", buf, GetFileName(), GetLineNumber() );
}

void Parser::ErrorMessage( const char *msg )
{
	gprintf ( "Error: %s (%s, line %d)\n", msg, GetFileName(), GetLineNumber() );
	exit(-1);
}

void Parser::ErrorMessage( const char *msg, const char *paramStr )
{
	char buf[ 256 ];
	sprintf( buf, msg, paramStr );
	printf( "Error: %s (%s, line %d)\n", buf, GetFileName(), GetLineNumber() );
	exit(-1);
}



#pragma warning( disable : 4996 )


CallbackParser::CallbackParser () : Parser(), numCallbacks(0)
{
}


void CallbackParser::ParseFile ( const char *filename, std::vector<std::string>& paths ) 
{
	Parser::ParseFile ( filename, paths );

	Parse ();
}

CallbackParser::~CallbackParser()
{
	for (int i=0;i < numCallbacks;i++)
		free( callbackTokens[i] );
}

void CallbackParser::SetCallback( const char *token, CallbackFunction func )
{
	if (numCallbacks >= 32) return;
	
	char* tk = (char*) malloc ( strlen(token) + 1);
	strcpy ( tk, token );
	callbackTokens[numCallbacks] = tk;

	MakeLower( callbackTokens[numCallbacks] );
	callbacks[numCallbacks]      = func;
	numCallbacks++;
}

void CallbackParser::Parse( void )
{
	char *linePtr = 0;
	char keyword[512];

	// For each line in the file... (Except those processed by callbacks inside the loop)
	while ( (linePtr = this->ReadNextLine()) != NULL )
	{
		// Each line starts with a keyword.  Get this token
		this->GetLowerCaseToken( keyword );

		// Check all of our callbacks to see if this line matches
		for (int i=0; i<numCallbacks; i++)
			if ( !strcmp(keyword, callbackTokens[i]) )  // We found a matching token!
				(*callbacks[i])();                //    Execute the callback.

		// If no matching token/callback pairs were found, ignore the line.
	}

	// Done parsing the file.  Close it.
	CloseFile();
}

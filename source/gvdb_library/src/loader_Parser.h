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
// A class that opens a file and does some parsing				  
// Chris Wyman (06/09/2010)                                       
//

#ifndef __PARSER_H
	#define __PARSER_H

	#include "gvdb_vec.h"
	using namespace nvdb;

	#include <cstdio>
	#include <cstdlib>
	#include <string>
	#include <vector>

	// Utility function to get the file extension
	extern std::string getExtension( const std::string& filename );

	class Parser {
	public:
		Parser();
		virtual ~Parser();

		void ParseFile ( const char *filename, std::vector<std::string>& paths );	

		// Read the next line into an internal buffer
		char *ReadNextLine( bool discardBlanks=true );  

		// Get a pointer to the as-yet-unprocessed part of the current line
		char *GetCurrentLinePointer( void )						{ return internalBufPtr; }

		// Get a pointer to the beginning of the current line.
		char *GetUnprocessedLinePointer( void )					{ return internalBuf; }

		// Get the next token on the line (i.e., white-space delimited)
		char *GetToken( char *token );

		// Get the next token on the line, but force it all to be lower case first
		char *GetLowerCaseToken( char *token );

		// Get the next token on the line, but force it all to be upper case first
		char *GetUpperCaseToken( char *token );

		// Reads a number from the line.  Versions returning a (char *) return a pointer
		//    to the remainder of the current line and store the value in the parameter,
		//    the others return the value directly.
		int		 GetInteger( void );
		unsigned GetUnsigned( void );	
		float	 GetFloat( void );
		Vector4DF	 GetVec4( void );
		Vector4DF	 GetVec3( void );    // Hack for stand-alone parser
		double	 GetDouble( void );
		char *	 GetInteger( int *result );
		char *	 GetUnsigned( unsigned *result );
		char *	 GetFloat( float *result );
		char *	 GetDouble( double *result );

		// Reads a 4x4 matrix from the line.  WARNING: Somewhat fragile.
		//   -> Reads 16 numbers on one line, in OpenGL order (4 numbers 
		//      in 1st column, then 4 numbers in 2nd column, ...)
		Matrix4F Get4x4Matrix( void );

		// File line number accessor
		int GetLineNumber( void ) const                         { return lineNum; }

		// Accessors to the underlying file handle
		bool IsFileValid( void ) const							{ return f != NULL; }
		FILE *GetFileHandle( void ) const						{ return f; }

		// Get information about the file
		char *GetFileDirectory( void )							{ return fileDirectory; }
		char *GetFileName( void )								{ return unqualifiedFileName; }
		char *GetQualifiedFileName( void )						{ return fileName; }

		// Get the file's size.  Apparently the approach used to get size could be non-portable to
		//    certain systems, in which case the size will be -1.  This rarely happens.  -1 could
		//    also happen if the file size exceeds a 32-bit limit (currently used in IGLU).
		long GetFileSize( void ) const                          { return fileSize; }

		// Print error messages to standard output.
		void WarningMessage( const char *msg );
		void ErrorMessage( const char *msg );

		// Error messages where the first (msg) parameter includes a %s that
		//    is replaced by the second parameter
		void WarningMessage( const char *msg, const char *paramStr );
		void ErrorMessage( const char *msg, const char *paramStr );

		// Sometimes a line from the file is processed but then we find out that we were
		//    not the correct code to process this line.  Before passing the parser off
		//    to someone else, we should "reset" the line so the other code can start
		//    processing from the beginning of the line.
		void ResetProcessingForCurrentLine( void );

	protected:
		char *__ReadLine( void );		// A simple call to fgets, storing data internally, and increment our line counter
				
		void CloseFile( void );			// Derived classes may want to go ahead and close the file when they're ready

		FILE *f;
		bool m_closed;
		int lineNum;
		char *fileName, *unqualifiedFileName, *fileDirectory;
		char internalBuf[ 1024 ], *internalBufPtr;
		long fileSize;

	};

	
	class CallbackParser : public Parser 
	{
	public:
		typedef void (*CallbackFunction)();

		// Start parsing by creating a new object with the desired file
		CallbackParser ();
		virtual ~CallbackParser();

		void ParseFile ( const char *filename, std::vector<std::string>& paths );

		// Add a callback to the list of token / function pairs
		//   -> Matching with 'token' is not case sensitive.
		void SetCallback( const char *token, void(*func) () );

		// Once the callbacks are setup, parse the file.
		//  -> This also closes the file when done parsing
		void Parse( void );

	protected:
		int              numCallbacks;
		char *           callbackTokens[32];   // Hardcoded to a limit of 32 callbacks.  Silently fails if using more.
		CallbackFunction callbacks[32];
	};



#endif

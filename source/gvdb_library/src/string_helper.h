//-----------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2017 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
// 
// Version 1.0: Rama Hoetzlein, 5/1/2017
//-----------------------------------------------------------------------------

#ifndef DEF_STRING_HELPER
	#define DEF_STRING_HELPER

	#include "gvdb.h"
	#include <string>
	#include <vector>

	GVDB_API std::string strFilebase ( std::string str );	// basename of a file (minus ext)
	GVDB_API std::string strFilepath ( std::string str );	// path of a file

	GVDB_API int strToI (std::string s);
	GVDB_API float strToF (std::string s);
	GVDB_API std::string strParse ( std::string& str, std::string lsep, std::string rsep );
	GVDB_API bool strGet ( std::string str, std::string& result, std::string lsep, std::string rsep );	
	GVDB_API std::string strSplit ( std::string& str, std::string sep );
	GVDB_API bool strSub ( std::string str, int first, int cnt, std::string cmp );
	GVDB_API std::string strReplace ( std::string str, std::string delim, std::string ins );
	GVDB_API std::string strTrim ( std::string str );
	GVDB_API std::string strLeft ( std::string str, int n );
	GVDB_API std::string strRight ( std::string str, int n );	
	GVDB_API int strExtract ( std::string& str, std::vector<std::string>& list );
	GVDB_API unsigned long strToID ( std::string str );	
	GVDB_API bool strIsNum ( std::string str, float& f );	
	GVDB_API float strToNum ( std::string str );	
	GVDB_API bool strToVec ( std::string& str, std::string lsep, std::string insep, std::string rsep, float* vec, int cpt=3);	
	GVDB_API bool strToVec3 ( std::string& str, std::string lsep, std::string insep, std::string rsep, float* vec );	
	GVDB_API bool strToVec4 ( std::string& str, std::string lsep, std::string insep, std::string rsep, float* vec );	

	GVDB_API bool strEq ( std::string str, std::string str2 );

	// File helpers
	GVDB_API unsigned long getFileSize ( const char* fname );
	GVDB_API unsigned long getFilePos ( FILE* fp );
	GVDB_API bool getFileLocation ( const char* filename, char* outpath, std::vector<std::string>& paths );

#endif

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

#ifndef DEF_STRING_HELPER
	#define DEF_STRING_HELPER

	#include <string>
	#include <vector>

	std::string strFilebase ( std::string str );	// basename of a file (minus ext)
	std::string strFilepath ( std::string str );	// path of a file

	int strToI (std::string s);
	float strToF (std::string s);
	std::string strParse ( std::string& str, std::string lsep, std::string rsep );
	bool strGet ( std::string str, std::string& result, std::string lsep, std::string rsep );	
	std::string strSplit ( std::string& str, std::string sep );
	bool strSub ( std::string str, int first, int cnt, std::string cmp );
	std::string strReplace ( std::string str, std::string delim, std::string ins );
	std::string strLTrim ( std::string str );
	std::string strRTrim ( std::string str );
	std::string strTrim ( std::string str );
	std::string strLeft ( std::string str, int n );
	std::string strRight ( std::string str, int n );	
	int strExtract ( std::string& str, std::vector<std::string>& list );
	unsigned long strToID ( std::string str );	
	bool strIsNum ( std::string str, float& f );	
	float strToNum ( std::string str );	
	void strToVec3 ( std::string str, float* vec );	


	// File helpers
	unsigned long getFileSize ( char* fname );
	unsigned long getFilePos ( FILE* fp );
	bool getFileLocation ( char* filename, char* outpath, char** searchPaths, int numPaths );

#endif

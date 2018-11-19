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


#ifndef DEF_GVDB_NODE
	#define DEF_GVDB_NODE

	#pragma message ( "gvdb_node" )

	#include "gvdb_types.h"
	#include "gvdb_vec.h"
	#include <assert.h>

	#define imax(a,b)		((a) > (b) ? (a) : (b) )

	namespace nvdb {

	class VolumeGVDB;

	// GVDB Node
	// This is the primary element in a GVDB tree.
	// Nodes are stores in memory pools managed by VolumeGVDB and created in the allocator class.
	struct ALIGN(16) GVDB_API Node {
	public:							//						Size:	Range:
		uchar		mLev;			// Tree Level			1 byte	Max = 0 to 255
		uchar		mFlags;			// Flags				1 byte	true - used, false - discard
		uchar		mPriority;		// Priority				1 byte
		uchar		pad;			//						1 byte
		Vector3DI	mPos;			// Pos in Index-space	12 byte
		Vector3DI	mValue;			// Value in Atlas		12 byte
		Vector3DF	mVRange;		// Value min, max, ave	12 byte
		uint64		mParent;		// Parent ID			8 byte	Pool0 reference
		uint64		mChildList;		// Child List			8 byte	Pool1 reference									?#ifdef USE_BITMASKS
		uint64		mMask;			// Start of BITMASK.	8 byte  
									// HEADER TOTAL			64 bytes
	};

	};

	
	
#endif
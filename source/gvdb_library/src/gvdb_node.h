//-----------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2016 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
// 
// Version 1.0: Rama Hoetzlein, 5/1/2017
// Version 1.1: Rama Hoetzlein, 3/25/2018
//-----------------------------------------------------------------------------


#ifndef DEF_GVDB_NODE
	#define DEF_GVDB_NODE

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

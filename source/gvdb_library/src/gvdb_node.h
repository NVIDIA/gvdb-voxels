//--------------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2017, NVIDIA Corporation. 
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
	extern VolumeGVDB* gVDB; 

	// GVDB Node
	// This is the primary element in a GVDB tree.
	// Nodes are stores in memory pools managed by VolumeGVDB and created in the allocator class.
	struct ALIGN(16) GVDB_API Node {
	public:							//						Size:	Range:
		uchar		mLev;			// Tree Level			1 byte	Max = 0 to 255
		uchar		mFlags;			// Flags				1 byte	
		uchar		mPriority;		// Priority				1 byte
		uchar		pad;
		Vector3DI	mPos;			// Pos in Index-space	12 byte
		Vector3DI	mValue;			// Value in Atlas		12 byte
		Vector3DF	mVRange;		// Value min, max, ave	12 byte
		uint64		mParent;		// Parent ID			8 byte	Pool0 reference
		uint64		mChildList;		// Child List			8 byte	Pool1 reference					
		uint64		mMask;			// Start of BITMASK.	1 byte  +BYTES: (2^4)^3 = 4096 bits / 8 = +512 bytes
									// HEADER TOTAL			56 bytes

	public:

		// Mask size helpers
		int		getMaskBits();
		int		getMaskBytes();
		uint64	getMaskWords();
		uint64*  getMask();
		int		getNumChild();

		// Bit operations:
		// Based on Bithacks (Sean Eron Anderson, http://graphics.stanford.edu/~seander/bithacks.html)
		// 
		inline uint64 numBitsOn ( byte v )
		{
			static const byte numBits[256] = {
				#define B2(n)  n,     n+1,     n+1,     n+2
				#define B4(n)  B2(n), B2(n+1), B2(n+1), B2(n+2)
				#define B6(n)  B4(n), B4(n+1), B4(n+1), B4(n+2)
				B6(0), B6(1), B6(1),   B6(2)
			};
			return numBits[v];    
		}

		inline uint64 numBitsOff (byte v) { return numBitsOn( (byte) ~v); }

		inline uint64 numBitsOn( uint32 v)
		{
			v = v - ((v >> 1) & 0x55555555U);
			v = (v & 0x33333333U) + ((v >> 2) & 0x33333333U);
			return ((v + (v >> 4) & 0xF0F0F0FU) * 0x1010101U) >> 24;
		}

		inline uint64 numBitsOff ( uint32 v) { return numBitsOn(~v); }

		inline uint64 numBitsOn ( uint64 v)
		{
			v = v - ((v >> 1) & UINT64_C(0x5555555555555555));
			v = (v & UINT64_C(0x3333333333333333)) + ((v >> 2) & UINT64_C(0x3333333333333333));
			return ((v + (v >> 4) & UINT64_C(0xF0F0F0F0F0F0F0F)) * UINT64_C(0x101010101010101)) >> 56;
		}

		inline uint64 numBitsOff ( uint64 v) { return numBitsOn( (uint64) ~v); }

		inline uint64 firstBitOn ( byte v )
		{
			assert(v);		// make sure not 0
			static const byte DeBruijn[8] = {0, 1, 6, 2, 7, 5, 4, 3};
			return DeBruijn[ byte((v & -v) * 0x1DU) >> 5 ];
		}

		inline uint64 firstBitOn ( uint32 v)
		{
			assert(v);    
			static const byte DeBruijnBitPos[32] = {
				0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
				31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
			};
			return DeBruijnBitPos[ uint32((int(v) & -int(v)) * 0x077CB531U) >> 27 ];
		}

		inline uint64 firstBitOn ( uint64 v)
		{
			assert(v);    
			static const byte DeBruijn[64] = {
				0,   1,  2, 53,  3,  7, 54, 27, 4,  38, 41,  8, 34, 55, 48, 28,
				62,  5, 39, 46, 44, 42, 22,  9, 24, 35, 59, 56, 49, 18, 29, 11,
				63, 52,  6, 26, 37, 40, 33, 47, 61, 45, 43, 21, 23, 58, 17, 10,
				51, 25, 36, 32, 60, 20, 57, 16, 50, 31, 19, 15, 30, 14, 13, 12,
			};
			return DeBruijn[ uint64((sint64(v) & -sint64(v)) * UINT64_C(0x022FDD63CC95386D)) >> 58];
		}

		inline uint64 lastBitOn ( uint32 v)
		{
			static const byte DeBruijn[32] = {
				0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30,
				8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31
			};
			v |= v >> 1; // first round down to one less than a power of 2
			v |= v >> 2;
			v |= v >> 4;
			v |= v >> 8;
			v |= v >> 16;
			return DeBruijn[ uint32(v * 0x07C4ACDDU) >> 27];
		}

		void clearMask () 
		{						
			mMask = 0;
			memset ( &mMask, 0, getMaskBytes() );		
		}

		// set operator
		void operator = (Node &op2) {			
			uint64* w1 = (uint64*) &mMask;
			uint64* we = (uint64*) &mMask + getMaskWords();
			uint64* w2 = op2.getMask();
			for ( ; w1 != we; ) 
				*w1++ = *w2++;
		}
		// compare operator
		bool operator == (Node &op2 ) 
		{
			int sz = (int) getMaskWords();
			uint64* w1 = (uint64*) &mMask;
			uint64* we = (uint64*) &mMask + sz;
			uint64* w2 = op2.getMask();
			for ( ; w1 != we && (*w1++ == *w2++); );
			return w1 == we;
		}
		// not operator
		bool operator != (Node &op2)
		{
			return !(*this == op2); 
		}

		// count on
		uint32 countOn()
		{
			uint32 sum = 0;
			uint64* w1 = (uint64*) &mMask;
			uint64* we = (uint64*) &mMask + getMaskWords();
			for ( ; w1 != we ; ) sum += (int) numBitsOn( *w1++ );
			return sum;
		}	
		uint64 countOn ( uint32 b )
		{
			uint64 sum = 0;
			uint64* w1 = (uint64*) &mMask;
			uint64* we = (uint64*) &mMask + (b >> 6);
			for (; w1 != we; ) sum += (int) numBitsOn (*w1++ );
			uint64 w2 = *w1;
			w2 = w2 & (( uint64(1) << (b & 63))-1);
			sum += numBitsOn ( w2 );
			return sum;
		}

		uint64 countOff() { return getMaskBits() - countOn(); }

		void setOn ( uint32 n) {        
			(&mMask)[n >> 6] |= uint64(1) << (n & 63);
		}    
		void setOff (uint32 n ) {        
			(&mMask)[n >> 6] &=  ~(uint64(1) << (n & 63));
		}    
		void setAll (bool on)
		{
			const uint64 val = on ? ~uint64(0) : uint64(0);
			uint64* w1 = (uint64*) &mMask;
			uint64* we = (uint64*) &mMask + getMaskWords();
			for ( ; w1 != we; ) *w1++ = val;
		}    

		bool isOn ( uint64 n )
		{
			return ( ((uint64*) &mMask)[n >> 6] & (uint64(1) << (n & 63))) != 0;
		}    
		bool isOff ( uint64 n) 
		{
			return ( ((uint64*) &mMask)[n >> 6] & (uint64(1) << (n & 63))) == 0;
		}
	};

	}
	
#endif
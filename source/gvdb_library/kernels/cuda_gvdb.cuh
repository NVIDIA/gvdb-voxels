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
// Shading consts
#define SHADE_VOXEL		0	
#define SHADE_SECTION2D	1	
#define SHADE_SECTION3D	2
#define SHADE_EMPTYSKIP	3
#define SHADE_TRILINEAR	4
#define SHADE_TRICUBIC	5
#define SHADE_LEVELSET	6
#define SHADE_VOLUME	7	

#define ALIGN(x)	__align__(x)

struct ALIGN(16) VDBNode {
	uchar		mLev;			// Level		Max = 255			1 byte
	uchar		mFlags;
	uchar		pad[2];
	int3		mPos;			// Pos			Max = +/- 4 mil (linear space/range)	12 bytes
	int3		mValue;			// Value		Max = +8 mil		4 bytes
	uint64		mParent;		// Parent ID						8 bytes
	uint64		mChildList;		// Child List						8 bytes
#ifdef USE_BITMASKS
	uint64		mMask;			// Bitmask starts
#endif
};
struct ALIGN(16) VDBAtlasNode {
	int3		mPos;
	int			mLeafID;
};
struct ALIGN(16) VDBInfo {
	int			dim[10];
	int			res[10];	
	float3		vdel[10];
	float3		voxelsize;	
	int3		noderange[10];
	int			nodecnt[10];
	int			nodewid[10];
	int			childwid[10];
	char*		nodelist[10];
	char*		childlist[10];
	VDBAtlasNode*  atlas_map;
	int			atlas_stride;
	int			atlas_apron;
	int			atlas_res;
	int			apron_table[8];
	int			top_lev;
	bool		update;
	float3		bmin;
	float3		bmax;
	float3		thresh;
	float4*		transfer;
};

//------- BIT COUNTING
#ifdef USE_BITMASKS
inline __device__ uint64 numBitsOn ( uint64 v)
{
	v = v - ((v >> 1) & uint64(0x5555555555555555LLU));
	v = (v & uint64(0x3333333333333333LLU)) + ((v >> 2) & uint64(0x3333333333333333LLU));
	return ((v + (v >> 4) & uint64(0xF0F0F0F0F0F0F0FLLU)) * uint64(0x101010101010101LLU)) >> 56;
}

inline __device__ int countOn ( VDBNode* node, int n )
{
	uint64 sum = 0;
	uint64* w1 = (&node->mMask);
	uint64* we = w1 + (n >> 6);
	for (; w1 != we; ) sum += numBitsOn (*w1++ );
	uint64 w2 = *w1;
	w2 = *w1 & (( uint64(1) << (n & 63))-1);
	sum += numBitsOn ( w2 );
	return sum;
}

inline __device__ bool isBitOn ( VDBNode* node, int b )
{
	return ( (&node->mMask)[ b >> 6 ] & (uint64(1) << (b & 63))) != 0;
}
#endif
//------- BASIC GEOMETRY

inline __device__ float3 getRayPoint ( float3 pos, float3 dir, float t )
{
	return pos + t*dir;
}

inline __device__ float3 getViewRay ( float x, float y )
{
  float3 v = x*scn.camu + y*scn.camv + scn.cams;
  return mmult(normalize(v), SCN_INVXFORM);
}


inline __device__ float rayPlaneIntersect ( float3 rpos, float3 rdir, float3 pnorm, float3 ppnt )
{
	float t = ( (ppnt.x-rpos.x)*pnorm.x + (ppnt.y-rpos.y)*pnorm.y + (ppnt.z-rpos.z)*pnorm.z ) / (rdir.x*pnorm.x + rdir.y*pnorm.y + rdir.z*pnorm.z);		
	return (t > 0 ? t : NOHIT);
}

inline __device__ float raySphereIntersect ( float3 rpos, float3 rdir, float3 spos, float srad )
{
	float a = rdir.x*rdir.x + rdir.y*rdir.y + rdir.z*rdir.z;
	float b = 2*rdir.x*(rpos.x-spos.x) + 2*rdir.y*(rpos.y-spos.y) + 2*rdir.z*(rpos.z-spos.z);
	float c = spos.x*spos.x + spos.y*spos.y + spos.z*spos.z + rpos.x*rpos.x + rpos.y*rpos.y + rpos.z*rpos.z - 2*(spos.x*rpos.x+spos.y*rpos.y+spos.z*rpos.z) - srad*srad;
	float dem = b*b - 4*a*c;
	if ( dem < 0 ) return NOHIT;
	float t1 = (-b - sqrt( dem )) / 2*a;
	float t2 = (-b - sqrt( dem )) / 2*a;
	return (t1 < t2) ? t1 : t2;
}

inline __device__ float3 rayBoxIntersect ( float3 rpos, float3 rdir, float3 vmin, float3 vmax )
{
	register float ht[8];
	ht[0] = (vmin.x - rpos.x)/rdir.x;
	ht[1] = (vmax.x - rpos.x)/rdir.x;
	ht[2] = (vmin.y - rpos.y)/rdir.y;
	ht[3] = (vmax.y - rpos.y)/rdir.y;
	ht[4] = (vmin.z - rpos.z)/rdir.z;
	ht[5] = (vmax.z - rpos.z)/rdir.z;
	ht[6] = fmax(fmax(fmin(ht[0], ht[1]), fmin(ht[2], ht[3])), fmin(ht[4], ht[5]));
	ht[7] = fmin(fmin(fmax(ht[0], ht[1]), fmax(ht[2], ht[3])), fmax(ht[4], ht[5]));	
	ht[6] = (ht[6] < 0 ) ? 0.0 : ht[6];
	return make_float3( ht[6], ht[7], (ht[7]<ht[6] || ht[7]<0) ? NOHIT : 0 );
}

inline __device__ float3 frac3(float3 x) { return x - floor3(x); }


//------- RANDOM NUMBER GEN

inline __device__ uint hash( uint x ) 
{
    x += ( x << 10u );    x ^= ( x >>  6u );
    x += ( x <<  3u );    x ^= ( x >> 11u );
    x += ( x << 15u );
    return x;
}
// Compound versions of the hashing algorithm I whipped together.
inline __device__ uint hash( int2 v ) { return hash( v.x ^ hash(v.y)                         ); }
inline __device__ uint hash( int3 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z)             ); }

// Construct a float with half-open range [0:1] using low 23 bits.
// All zeroes yields 0.0, all ones yields the next smallest representable value below 1.0.
inline __device__ float floatConstruct( uint m ) 
{
    const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32
    m &= ieeeMantissa;                     // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                          // Add fractional part to 1.0
    float  f = __int_as_float( m );       // Range [1:2]
    return f - 1.0;                        // Range [0:1]
}
// Pseudo-random value in half-open range [0:1].
inline __device__ float random( float x ) { return floatConstruct(hash(__float_as_int(x))); }
inline __device__ float random( float2 v ) { return floatConstruct(hash( make_int2(__float_as_int(v.x), __float_as_int(v.y) ))); }
inline __device__ float random( float3 v ) { return floatConstruct(hash( make_int3(__float_as_int(v.x), __float_as_int(v.y), __float_as_int(v.z)))); }


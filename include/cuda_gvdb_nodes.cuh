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
// File: cuda_gvdb_node.cuh
//
// CUDA GVDB Node header
// - Node structure
// - Node variables (CUDA or OptiX)
// - Node traversal
// - Brick helpers
//-----------------------------------------------

#define ALIGN(x)	__align__(x)
#define ID_UNDEFL	0xFFFFFFFF
#define CHAN_UNDEF	255

struct ALIGN(16) VDBNode {
	uchar		mLev;			// Level		Max = 255			1 byte
	uchar		mFlags;
	uchar		mPriority;
	uchar		pad;
	int3		mPos;			// Pos			Max = +/- 4 mil (linear space/range)	12 bytes
	int3		mValue;			// Value		Max = +8 mil		4 bytes
	float3		mVRange;
	uint64		mParent;		// Parent ID						8 bytes
	uint64		mChildList;		// Child List						8 bytes		
	uint64		mMask;			// Bitmask starts
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
	int3		atlas_cnt;
	int3		atlas_res;
	int			atlas_apron;
	int			brick_res;
	int			apron_table[8];
	int			top_lev;
	bool		update;
	uchar		clr_chan;
	float3		bmin;
	float3		bmax;
	float3		thresh;
	float4*		transfer;
};

__device__ float								cdebug[256]; 
__device__ float								cxform[16];

#ifdef CUDA_PATHWAY
	__constant__ VDBInfo						gvdb;			// GVDB Topology
	texture<float, 3, cudaReadModeElementType>	volTexIn;		// GVDB Atlas
	surface<void, 3>							volTexOut;

	__device__ cudaTextureObject_t				volIn[10];
	__device__ cudaSurfaceObject_t				volOut[10];

	__device__	 float3							deep_depth;		// GVDB Node helpers
#endif
#ifdef OPTIX_PATHWAY
	rtDeclareVariable( VDBInfo,		gvdb, , );					// GVDB Topology
	rtTextureSampler<float, 3>		volTexIn;					// GVDB Atlas
#endif


inline __device__ uint64 numBitsOn ( uint64 v)
{
	v = v - ((v >> 1) & uint64(0x5555555555555555LLU));
	v = (v & uint64(0x3333333333333333LLU)) + ((v >> 2) & uint64(0x3333333333333333LLU));
	return ((v + (v >> 4) & uint64(0xF0F0F0F0F0F0F0FLLU)) * uint64(0x101010101010101LLU)) >> 56;
}
inline __device__ int countOn ( VDBNode* node, int n )
{
	uint64 sum = 0;
	uint64* w1 = (uint64*) (&node->mMask);
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

inline __device__ VDBAtlasNode* getAtlasNode ( float3 brickpos )
{
	int3 i = make_int3(brickpos.x/gvdb.brick_res, brickpos.y/gvdb.brick_res, brickpos.z/gvdb.brick_res);		// brick index
	int id = (i.z*gvdb.atlas_cnt.y + i.y )*gvdb.atlas_cnt.x + i.x;		// brick id
	return gvdb.atlas_map + id;											// brick node
}

inline __device__ VDBAtlasNode* getAtlasNodeFromIndex ( int3 i )
{
	int id = (i.z*gvdb.atlas_cnt.y + i.y )*gvdb.atlas_cnt.x + i.x;		// brick id
	return gvdb.atlas_map + id;											// brick node
}
inline __device__ bool getAtlasToWorld ( uint3 vox, float3& wpos )
{
	int3 bndx = make_int3(vox.x/gvdb.brick_res, vox.y/gvdb.brick_res, vox.z/gvdb.brick_res );
	int brickid = (bndx.z*gvdb.atlas_cnt.y + bndx.y )*gvdb.atlas_cnt.x + bndx.x;		// brick id			
	if ((gvdb.atlas_map + brickid)->mLeafID == ID_UNDEFL) return false;
	int3 poffset = make_int3(vox.x % gvdb.brick_res, vox.y % gvdb.brick_res, vox.z % gvdb.brick_res ) - make_int3(gvdb.atlas_apron);
	wpos = ( make_float3((gvdb.atlas_map + brickid)->mPos) + make_float3(poffset) + make_float3(0.5,0.5,0.5) ) * gvdb.voxelsize;
	return true;
}
inline __device__ bool getAtlasToWorldID ( uint3 vox, float3& wpos, int& leafid )
{
	int3 bndx = make_int3(vox.x/gvdb.brick_res, vox.y/gvdb.brick_res, vox.z/gvdb.brick_res );
	int brickid = (bndx.z*gvdb.atlas_cnt.y + bndx.y )*gvdb.atlas_cnt.x + bndx.x;		// brick id			
	leafid = (gvdb.atlas_map + brickid)->mLeafID;
	if ( leafid == ID_UNDEFL) return false;
	int3 poffset = make_int3(vox.x % gvdb.brick_res, vox.y % gvdb.brick_res, vox.z % gvdb.brick_res ) - make_int3(gvdb.atlas_apron);
	wpos = ( make_float3((gvdb.atlas_map + brickid)->mPos) + make_float3(poffset) + make_float3(0.5,0.5,0.5) ) * gvdb.voxelsize;
	return true;
}

inline __device__ int getChild ( VDBNode* node, int b )
{	
	int n = countOn ( node, b );
	uint64 listid = node->mChildList;
	uchar clev = uchar( (listid >> 8) & 0xFF );
	int cndx = listid >> 16;
	uint64* clist = (uint64*) (gvdb.childlist[clev] + cndx*gvdb.childwid[clev]);
	int c = (*(clist + n)) >> 16;
	return c;
}


inline __device__ int3 getAtlasPos ( uint64 id )
{
	int3 ap;	
	int a2 = gvdb.atlas_cnt.x * gvdb.atlas_cnt.y;
	// get block val (xyz)
	ap.z = int( id / a2 );					id -= uint64(ap.z)*a2;
	ap.y = int( id / gvdb.atlas_cnt.x );	id -= uint64(ap.y)*gvdb.atlas_cnt.x;
	ap.x = int( id );
	// get corner voxel
	ap *= gvdb.brick_res;  // + gvdb.atlas_apron;
	return ap;
}
inline __device__ int getBitPos ( int lev, int3 pos )
{
	int res = gvdb.res[ lev ];
	return (pos.z*res + pos.y)*res+ pos.x;
}

// get node at a specific level and pool index
inline __device__ VDBNode* getNode ( int lev, int n, float3* vmin )
{
	VDBNode* node = (VDBNode*) (gvdb.nodelist[lev] + n*gvdb.nodewid[lev]);
	*vmin = node->mPos * gvdb.voxelsize;	
	return node;
}

// iteratively find the leaf node at the given position
inline __device__ VDBNode* getNode ( int lev, int start_id, float3 pos, uint64* node_id )
{
	float3 vmin, vmax;
	int3 p;
	int b;
	*node_id = ID_UNDEFL;

	VDBNode* node = getNode ( lev, start_id, &vmin );		// get starting node
	while ( lev > 0 && node != 0x0 ) {			
		// is point inside node? if no, exit
		vmax = vmin + make_float3(gvdb.noderange[lev]) * gvdb.voxelsize; 
		if ( pos.x < vmin.x || pos.y < vmin.y || pos.z < vmin.z || pos.x >= vmax.x || pos.y >= vmax.y || pos.z >= vmax.z ) {
			*node_id = ID_UNDEFL; 
			return 0x0;		
		}
		p = make_int3 ( (pos-vmin)/gvdb.vdel[lev] );		// check child bit
		b = (( (int(p.z) << gvdb.dim[lev]) + int(p.y)) << gvdb.dim[lev]) + int(p.x);
		lev--;
		if ( isBitOn ( node, b ) ) {						// child exists, recurse down tree
			*node_id = getChild ( node, b );				// get next node_id
			node = getNode ( lev, *node_id, &vmin );
		} else {
			*node_id = ID_UNDEFL;
			return 0x0;										// no child, exit
		}		
	}
	return node;
}

inline __device__ VDBNode* getNodeAtPoint ( float3 pos, float3* offs, float3* vmin, float3* vdel, uint64* node_id )
{
	// iteratively get node at world point
	VDBNode* node = getNode ( gvdb.top_lev, 0, pos, node_id );	 
	if ( node == 0x0 ) return 0x0;
	
	// compute node bounding box
	*vmin = node->mPos * gvdb.voxelsize;	
	*vdel = gvdb.vdel[ node->mLev ];  //make_float3(gvdb.noderange[0]) * gvdb.voxelsize / gvdb.res[0];
	*offs = make_float3( node->mValue );
	return node;
}

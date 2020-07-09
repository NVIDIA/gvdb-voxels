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
#define ID_UNDEFI	0xFFFF
#define ID_UNDEFL	0xFFFFFFFF
#define ID_UNDEF64	0xFFFFFFFFFFFFFFFF
#define CHAN_UNDEF	255
#define MAX_CHANNEL  32

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
	uint64		mMask;			// Bitmask starts - Must keep here, even if not USE_BITMASKS
};


struct ALIGN(16) VDBAtlasNode {
	int3		mPos;
	int			mLeafID;
};
struct ALIGN(16) VDBInfo {
	int			dim[10]; // Log base 2 of lateral resolution of each node per level
	int			res[10]; // Lateral resolution of each node per level
	float3		vdel[10]; // How many voxels on a side a child of each level covers
	int3		noderange[10]; // How many voxels on a side a node of each level covers
	int			nodecnt[10]; // Total number of allocated nodes per level
	int			nodewid[10]; // Size of a node at each level in bytes
	int			childwid[10]; // Size of the child list per node at each level in bytes
	char*		nodelist[10]; // Pointer to each level's pool group 0 (nodes)
	char*		childlist[10]; // Pointer to each level's pool group 1 (child lists)
	VDBAtlasNode*  atlas_map; // Pointer to the atlas map (which maps from atlas to world space)
	int3		atlas_cnt; // Number of bricks on each axis of the atlas
	int3		atlas_res; // Total resolution in voxels of the atlas
	int			atlas_apron; // Apron size
	int			brick_res; // Resolution of a single brick
	int			apron_table[8]; // Unused
	int			top_lev; // Top level (i.e. tree spans from voxels to level 0 to level top_lev)
	int			max_iter; // Unused
	float		epsilon; // Epsilon used for voxel ray tracing
	bool		update; // Whether this information needs to be updated from the latest volume data
	uchar		clr_chan; // Index of the color channel for rendering color information
	float3		bmin; // Inclusive minimum of axis-aligned bounding box in voxels
	float3		bmax; // Inclusive maximum of axis-aligned bounding box in voxels
	cudaTextureObject_t	volIn[MAX_CHANNEL]; // Texture reference (read plus interpolation) to atlas per channel
	cudaSurfaceObject_t	volOut[MAX_CHANNEL]; // Surface reference (read and write) to atlas per channel
};

__device__ float								cdebug[256]; 
__device__ float								cxform[16];

#ifdef CUDA_PATHWAY
	
	__device__	 float3							deep_depth;		// GVDB Node helpers

#endif

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

inline __device__ int getChild ( VDBNode* node, int b )
{	
	int n = countOn ( node, b );
	uint64 listid = node->mChildList;
	if (listid == ID_UNDEFL) return ID_UNDEF64;
	uchar clev = uchar( (listid >> 8) & 0xFF );
	int cndx = listid >> 16;
	uint64* clist = (uint64*) (gvdb.childlist[clev] + cndx*gvdb.childwid[clev]);
	int c = (*(clist + n)) >> 16;
	return c;
}

#else 

inline __device__ int getChild ( VDBInfo* gvdb, VDBNode* node, int b )
{	
	uint64 listid = node->mChildList;
	if (listid == ID_UNDEFL) return ID_UNDEF64;
	uchar clev = uchar( (listid >> 8) & 0xFF );
	int cndx = listid >> 16;
	uint64* clist = (uint64*) (gvdb->childlist[clev] + cndx*gvdb->childwid[clev]);
	int c = (*(clist + b)) >> 16;
	return c;
}

inline __device__ bool isBitOn ( VDBInfo* gvdb, VDBNode* node, int b )
{
	return getChild(gvdb, node, b) != ID_UNDEF64;
}

#endif

inline __device__ VDBAtlasNode* getAtlasNode ( VDBInfo* gvdb, float3 brickpos )
{
	int3 i = make_int3(brickpos.x/gvdb->brick_res, brickpos.y/gvdb->brick_res, brickpos.z/gvdb->brick_res);		// brick index
	int id = (i.z*gvdb->atlas_cnt.y + i.y )*gvdb->atlas_cnt.x + i.x;		// brick id
	return gvdb->atlas_map + id;											// brick node
}

inline __device__ VDBAtlasNode* getAtlasNodeFromIndex ( VDBInfo* gvdb, int3 i )
{
	int id = (i.z*gvdb->atlas_cnt.y + i.y )*gvdb->atlas_cnt.x + i.x;		// brick id
	return gvdb->atlas_map + id;											// brick node
}
inline __device__ bool getAtlasToWorld ( VDBInfo* gvdb, uint3 vox, float3& wpos )
{
	int3 bndx = make_int3(vox.x/gvdb->brick_res, vox.y/gvdb->brick_res, vox.z/gvdb->brick_res );
	int brickid = (bndx.z*gvdb->atlas_cnt.y + bndx.y )*gvdb->atlas_cnt.x + bndx.x;		// brick id			
	if ((gvdb->atlas_map + brickid)->mLeafID == ID_UNDEFL) return false;
	int3 poffset = make_int3(vox.x % gvdb->brick_res, vox.y % gvdb->brick_res, vox.z % gvdb->brick_res ) - make_int3(gvdb->atlas_apron);
	wpos = ( make_float3((gvdb->atlas_map + brickid)->mPos) + make_float3(poffset) + make_float3(0.5,0.5,0.5) );
	return true;
}
inline __device__ bool getAtlasToWorldID ( VDBInfo* gvdb, uint3 vox, float3& wpos, int& leafid )
{
	int3 bndx = make_int3(vox.x/gvdb->brick_res, vox.y/gvdb->brick_res, vox.z/gvdb->brick_res );
	int brickid = (bndx.z*gvdb->atlas_cnt.y + bndx.y )*gvdb->atlas_cnt.x + bndx.x;		// brick id			
	leafid = (gvdb->atlas_map + brickid)->mLeafID;
	if ( leafid == ID_UNDEFL) return false;
	int3 poffset = make_int3(vox.x % gvdb->brick_res, vox.y % gvdb->brick_res, vox.z % gvdb->brick_res ) - make_int3(gvdb->atlas_apron);
	wpos = ( make_float3((gvdb->atlas_map + brickid)->mPos) + make_float3(poffset) + make_float3(0.5,0.5,0.5) );
	return true;
}

inline __device__ int3 getAtlasPos ( VDBInfo* gvdb, uint64 id )
{
	int3 ap;	
	int a2 = gvdb->atlas_cnt.x * gvdb->atlas_cnt.y;
	// get block val (xyz)
	ap.z = int( id / a2 );					id -= uint64(ap.z)*a2;
	ap.y = int( id / gvdb->atlas_cnt.x );	id -= uint64(ap.y)*gvdb->atlas_cnt.x;
	ap.x = int( id );
	// get corner voxel
	ap *= gvdb->brick_res;  // + gvdb->atlas_apron;
	return ap;
}
inline __device__ int getBitPos ( VDBInfo* gvdb, int lev, int3 pos )
{
	int res = gvdb->res[ lev ];
	return (pos.z*res + pos.y)*res+ pos.x;
}

// get node at a specific level and pool index
inline __device__ VDBNode* getNode ( VDBInfo* gvdb, int lev, int n )
{
	VDBNode* node = (VDBNode*) (gvdb->nodelist[lev] + n*gvdb->nodewid[lev]);
	return node;
}

// get node at a specific level and pool index
inline __device__ VDBNode* getNode ( VDBInfo* gvdb, int lev, int n, float3* vmin )
{
	VDBNode* node = (VDBNode*) (gvdb->nodelist[lev] + n*gvdb->nodewid[lev]);
	*vmin = make_float3(node->mPos);
	return node;
}

// iteratively find the leaf node at the given position
inline __device__ VDBNode* getNode ( VDBInfo* gvdb, int lev, int start_id, float3 pos, uint64* node_id )
{
	float3 vmin, vmax;
	int3 p;
	int b;
	*node_id = ID_UNDEFL;

	VDBNode* node = getNode ( gvdb, lev, start_id, &vmin );		// get starting node
	while ( lev > 0 && node != 0x0 ) {			
		// is point inside node? if no, exit
		vmax = vmin + make_float3(gvdb->noderange[lev]);
		if ( pos.x < vmin.x || pos.y < vmin.y || pos.z < vmin.z || pos.x >= vmax.x || pos.y >= vmax.y || pos.z >= vmax.z ) {
			*node_id = ID_UNDEFL; 
			return 0x0;		
		}
		p = make_int3 ( (pos-vmin)/gvdb->vdel[lev] );		// check child bit
		b = (( (int(p.z) << gvdb->dim[lev]) + int(p.y)) << gvdb->dim[lev]) + int(p.x);
		lev--;
		if ( isBitOn ( gvdb, node, b ) ) {						// child exists, recurse down tree
			*node_id = getChild ( gvdb, node, b );				// get next node_id
			node = getNode ( gvdb, lev, *node_id, &vmin );
		} else {
			*node_id = ID_UNDEFL;
			return 0x0;										// no child, exit
		}		
	}
	return node;
}

inline __device__ VDBNode* getNodeIdAtPoint ( VDBInfo* gvdb, float3 pos, uint64* node_id )
{
	// iteratively get node at world point
	VDBNode* node = getNode ( gvdb, gvdb->top_lev, 0, pos, node_id );	 
	if ( node == 0x0 ) return 0x0;
	
	return node;
}

// Gets the node pointer, atlas-space AABB minimum (not including apron), index-space AABB minimum, and voxel size of
// the brick (leaf node) containing the index-space position `pos`.
// `offs` is the brick's atlas-space AABB minimum (i.e. mValue. This does not include apron.)
// `vmin` is the brick's index-space AABB minimum
// Note that this function used to return `vdel`, the size of a brick's children in voxels. Since issue #74 on
// 2019-07-14, voxelsize has been deprecated, so this was always (1, 1, 1).
inline __device__ VDBNode* getNodeAtPoint ( VDBInfo* gvdb, float3 pos, float3* offs, float3* vmin, uint64* node_id )
{
	// iteratively get node at world point
	VDBNode* node = getNode ( gvdb, gvdb->top_lev, 0, pos, node_id );	 
	if ( node == 0x0 ) return 0x0;
	
	// compute node bounding box
	*vmin = make_float3(node->mPos);
	*offs = make_float3(node->mValue);
	return node;
}

inline __device__ int getPosLeafParent( VDBInfo* gvdb, int3 pos)
{
	VDBNode* nd = (VDBNode*) (gvdb->nodelist[gvdb->top_lev]);	// get root

	int l = nd->mLev;
	if (l == 0) return 0;	// root id

	int3 range, posInNode;
	int bitMaskPos;
	int childId;
	while (true)
	{
		int res = gvdb->res[l];
		range = gvdb->noderange[l];
		posInNode = pos - nd->mPos;
		posInNode *= res;
		posInNode.x /= range.x;
		posInNode.y /= range.y;
		posInNode.z /= range.z;

		if (posInNode.x > res || posInNode.y > res || posInNode.z > res) return ID_UNDEFL;

		bitMaskPos = (posInNode.z*res + posInNode.y)*res + posInNode.x;	

		childId = getChild( gvdb, nd, bitMaskPos);

		if (childId == ID_UNDEFL) return ID_UNDEFL;

		l--;
		if(l == 0) return childId;

		nd = (VDBNode*) (gvdb->nodelist[l] + childId*gvdb->nodewid[l]);//getNode(l, childId);
	}
}

inline __device__ VDBNode* getleafNodeAtPoint ( VDBInfo* gvdb, float3 pos, float3* vmin, float3* vdel)
{
	// iteratively get node at world point
	int pnodeId = getPosLeafParent(gvdb, make_int3(pos));
	if (pnodeId == ID_UNDEFL) return 0x0;
	VDBNode* node = (VDBNode*) (gvdb->nodelist[0] + pnodeId*gvdb->nodewid[0]);
	
	// compute node bounding box
	*vmin = make_float3(node->mPos);
	*vdel = gvdb->vdel[ node->mLev ];  
	return node;
}

inline __device__ uint64 getPosParent( VDBInfo* gvdb, int3 pos, int stopLev)
{
	VDBNode* nd = (VDBNode*) (gvdb->nodelist[gvdb->top_lev]);	// get root

	int l = nd->mLev;
	if (l == 0) return 0;	// root id

	int3 range, posInNode;
	int bitMaskPos;
	int childId;
	while (l > 0)
	{
		int res = gvdb->res[l];
		range = gvdb->noderange[l];
		posInNode = pos - nd->mPos;
		posInNode *= res;
		posInNode.x /= range.x;
		posInNode.y /= range.y;
		posInNode.z /= range.z;
		bitMaskPos = (posInNode.z*res + posInNode.y)*res + posInNode.x;
		//bitMaskPos = ((pos.z - nd->mPos.z)*gvdb->res[l]/gvdb->noderange[l].z*gvdb->res[l] + (pos.y - nd->mPos.y)*gvdb->res[l]/gvdb->noderange[l].y)*gvdb->res[l] + (pos.x - nd->mPos.x)*gvdb->res[l]/gvdb->noderange[l].x;
#ifdef USE_BITMASKS
		uint64 p = nd->countOn ( bitMaskPos );

		l--;

		uint64* clist = mPool->PoolData64(nd->mChildList);
		if (l == lev) return clist[p];

		nd = getNode(clist[p]);
#else
		
		childId = getChild( gvdb, nd, bitMaskPos);

		l--;

		if (childId == ID_UNDEFL) return ID_UNDEFL;

		if(l == stopLev) return childId;

		nd = (VDBNode*) (gvdb->nodelist[l] + childId*gvdb->nodewid[l]);//getNode(l, childId);
#endif
	}

	return ID_UNDEFL;
}

inline __device__ uint64 getParent( VDBInfo* gvdb, int stopLev, int3 pos)
{
	VDBNode* nd = getNode ( gvdb, gvdb->top_lev, 0);	// get root

	if (!nd->mFlags) return 0;

	int l = nd->mLev;
	if (l < 0) return 0;
	if (l == stopLev) return 0;	// root id

	int3 range, posInNode;
	int bitMaskPos;
	uint64 childId;
	while (l > stopLev)
	{
		int res = gvdb->res[l];
		range = gvdb->noderange[l];
		posInNode = pos - nd->mPos;//sortedPos[n] + pOrig - nd->mPos; 
		posInNode *= res;
		posInNode.x /= range.x;
		posInNode.y /= range.y;
		posInNode.z /= range.z;
		bitMaskPos = (posInNode.z*res + posInNode.y)*res + posInNode.x;

#ifdef USE_BITMASKS
		uint64 p = nd->countOn ( bitMaskPos );

		l--;

		uint64* clist = mPool->PoolData64(nd->mChildList);
		if (l == lev) return clist[p];

		nd = getNode( gvdb, clist[p] );
#else
		l--;

		childId = getChild( gvdb, nd, bitMaskPos);

		if (childId == ID_UNDEFL) return ID_UNDEFL;

		if(l == stopLev) return childId;

		nd = getNode( gvdb, l, childId);
#endif
	}

	return ID_UNDEFL;
}

// delink node

extern "C" __global__ void gvdbDelinkLeafBricks ( VDBInfo* gvdb )
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= gvdb->nodecnt[ 0 ] ) return;

	VDBNode* node = getNode ( gvdb, 0, i);

	if (node->mFlags) return;					// in use brick
	if (node->mParent == ID_UNDEF64) return;	// deactivated brick

	uint64 pnodeId = getParent( gvdb, 1, node->mPos);
	VDBNode* pnode = getNode ( gvdb, 1, pnodeId);

	int res = gvdb->res[1];
	int3 range = gvdb->noderange[1];

	int3 posInNode = node->mPos - pnode->mPos;
	posInNode *= res;
	posInNode.x /= range.x;
	posInNode.y /= range.y;
	posInNode.z /= range.z;
	int bitMaskPos = (posInNode.z*res + posInNode.y)*res+ posInNode.x;
	
	node->mParent = ID_UNDEF64;					// no parent = deactivated
	uint64 listid = pnode->mChildList;
	int cndx = listid >> 16;
	uint64* clist = (uint64*) (gvdb->childlist[1] + cndx*gvdb->childwid[1]);
	*(clist + bitMaskPos) = ID_UNDEF64;
}

extern "C" __global__ void gvdbDelinkBricks ( VDBInfo* gvdb, int lev )
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= gvdb->nodecnt[ lev ] ) return;

	VDBNode* node = getNode ( gvdb, lev, i);

	if (node->mParent == ID_UNDEF64) { node->mFlags = false; return;	}// deactivated brick

	for (int ci = 0; ci < gvdb->res[lev]*gvdb->res[lev]*gvdb->res[lev]; ci++)
	{
		uint64* clist = (uint64*) (gvdb->childlist[lev] + i*gvdb->childwid[lev]);
		if (*(clist + ci) != ID_UNDEF64) { node->mFlags = true; return; }
	}

	node->mFlags = false;

	uint64 pnodeId = getParent( gvdb, lev+1, node->mPos);

	VDBNode* pnode = getNode ( gvdb, lev+1, pnodeId);

	int res = gvdb->res[lev+1];
	int3 range = gvdb->noderange[lev+1];

	int3 posInNode = node->mPos - pnode->mPos;
	posInNode *= res;
	posInNode.x /= range.x;
	posInNode.y /= range.y;
	posInNode.z /= range.z;
	int bitMaskPos = (posInNode.z*res + posInNode.y)*res+ posInNode.x;
	
	node->mParent = ID_UNDEF64;					// no parent = deactivated
	uint64 listid = pnode->mChildList;
	int cndx = listid >> 16;
	uint64* clist = (uint64*) (gvdb->childlist[lev+1] + cndx*gvdb->childwid[lev+1]);
	*(clist + bitMaskPos) = ID_UNDEF64;
}

// link node
extern "C" __global__ void gvdbLinkBricks ( VDBInfo* gvdb, int lev)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= gvdb->nodecnt[ lev ] ) return;

	VDBNode* node = getNode ( gvdb, lev, i);

	if (!node->mFlags) return;
	uint64 pnodeId = getParent( gvdb, lev+1, node->mPos);
	if (pnodeId == ID_UNDEFL) return;

	VDBNode* pnode = getNode ( gvdb, lev+1, pnodeId);

	int res = gvdb->res[lev+1];
	int3 range = gvdb->noderange[lev+1];

	int3 posInNode = node->mPos - pnode->mPos;
	posInNode *= res;
	posInNode.x /= range.x;
	posInNode.y /= range.y;
	posInNode.z /= range.z;
	int bitMaskPos = (posInNode.z*res + posInNode.y)*res+ posInNode.x;
	
	if (posInNode.x > res || posInNode.x < 0 || posInNode.y > res || posInNode.y < 0 || posInNode.z > res || posInNode.z < 0) return;

	// set mParent in node
	node->mParent = ((pnodeId << 16) | ((lev+1) << 8));	// set parent of child

	uint64 listid = pnode->mChildList;
	uint64 cndx = listid >> 16;
	if (cndx >= gvdb->nodecnt[ lev+1 ]) return;
	uint64* clist = (uint64*) (gvdb->childlist[lev+1] + cndx*gvdb->childwid[lev+1]);

	*(clist + bitMaskPos) = ((uint64(i) << 16) | (uint64(lev) << 8));

}

extern "C" __global__ void gvdbSplitPos ( int num_pnts, float3 trans, char* ppos,  int pos_off, int pos_stride, float* xout, float* yout, float* zout)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_pnts) return;

	float3 wpos = (*(float3*) (ppos + i*pos_stride + pos_off)) + trans;

	xout[i] = wpos.x;
	yout[i] = wpos.y;
	zout[i] = wpos.z;
}

extern "C" __global__ void gvdbMarkLeafNode ( VDBInfo* gvdb, int num_nodes, int* node_markers)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= num_nodes ) return;

	VDBNode* nd = getNode ( gvdb, 0, i );
	nd->mFlags = node_markers[i];
}
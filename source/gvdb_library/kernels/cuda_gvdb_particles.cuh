
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
// GVDB Points
// - ClearNodeCounts	- clear brick particle counts
// - InsertPoints		- insert points into bricks
// - SplatPoints		- splat points into bricks

#define RGBA2INT(r,g,b,a)	(				uint((r)*255.0f) +		(uint((g)*255.0f)<<8) +			(uint((b)*255.0f)<<16) +		(uint((a)*255.0f)<<24) )
#define CLR2INT(c)			(				uint((c.x)*255.0f) +	(uint((c.y)*255.0f)<<8)	+		(uint((c.z)*255.0f)<<16) +		(uint((c.w)*255.0f)<<24 ) )
#define INT2CLR(c)			( make_float4( float(c & 0xFF)/255.0f,	float((c>>8) & 0xFF)/255.0f,	float((c>>16) & 0xFF)/255.0f,	float((c>>24) & 0xFF)/255.0f ))
#define CLR2CHAR(c)			( make_uchar4( uchar(c.x*255.0f),	uchar(c.y*255.0f),	uchar(c.z*255.0f),	uchar(c.w*255.0f) ))
#define CHAR2CLR(c)			( make_float4( float(c.x)/255.0f,	float(c.y)/255.0f,	float(c.z)/255.0f,	float(c.w)/255.0f ))

extern "C" __global__ void gvdbInsertPoints ( VDBInfo* gvdb, int num_pnts, char* ppos, int pos_off, int pos_stride, int* pnode, int* poff, int* gcnt, float3 ptrans )
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= num_pnts ) return;

	float3 wpos = (*(float3*) (ppos + i*pos_stride + pos_off)); // NOTE: +ptrans is below. Allows check for wpos.z==NOHIT 

	if ( wpos.z == NOHIT ) { pnode[i] = ID_UNDEFL; return; }		// If position invalid, return. 
	float3 offs, vmin, vdel;										// Get GVDB node at the particle point
	uint64 nid;
	VDBNode* node = getNodeAtPoint ( gvdb, wpos + ptrans, &offs, &vmin, &vdel, &nid );
	if ( node == 0x0 ) { pnode[i] = ID_UNDEFL; return; }			// If no brick at location, return.	

	__syncthreads();

	pnode[i] = nid;													// Place point in brick
	poff[i] = atomicAdd ( &gcnt[nid], (uint) 1 );					// Increment brick pcount, and retrieve this point index at the same time
}


extern "C" __global__ void gvdbInsertSupportPoints ( VDBInfo* gvdb, int num_pnts, float offset, char* ppos, int pos_off, int pos_stride, int* pnode, int* poff, int* gcnt, char* pdir, int dir_off, int dir_stride, float3 ptrans )
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= num_pnts ) return;

	float3 wpos = (*(float3*) (ppos + i*pos_stride + pos_off)); // NOTE: +ptrans is below. Allows check for wpos.z==NOHIT 
	float3 wdir = (*(float3*) (pdir + i*dir_stride + dir_off));

	if ( wpos.z == NOHIT ) { pnode[i] = ID_UNDEFL; return; }		// If position invalid, return. 
	float3 offs, vmin, vdel;										// Get GVDB node at the particle point
	uint64 nid;
	VDBNode* node = getNodeAtPoint ( gvdb, wpos + ptrans + wdir * offset, &offs, &vmin, &vdel, &nid );
	if ( node == 0x0 ) { pnode[i] = ID_UNDEFL; return; }			// If no brick at location, return.	

	__syncthreads();

	pnode[i] = nid;													// Place point in brick
	poff[i] = atomicAdd ( &gcnt[nid], (uint) 1 );					// Increment brick pcount, and retrieve this point index at the same time
}

extern "C" __global__ void gvdbSortPoints ( int num_pnts, char* ppos, int pos_off, int pos_stride, int* pnode, int* poff,
										   int num_nodes, int* gcnt, int* goff, float3* pout, float3 ptrans )
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= num_pnts ) return;

	uint64 nid = pnode[i];
	if ( nid > num_nodes ) return;

	int ndx = goff[nid] + poff[i];		// sorted index = brick offset (for particle's nid) + particle offset in brick

	float3 wpos = (*(float3*) (ppos + i*pos_stride + pos_off)) + ptrans ;
	pout[ndx] = wpos;
}

inline __device__ int3 GetCoveringNode (float3 pos, int3 range)
{
	int3 nodepos;

	nodepos.x = ceil(pos.x / range.x) * range.x;
	nodepos.y = ceil(pos.y / range.y) * range.y;
	nodepos.z = ceil(pos.z / range.z) * range.z;
	if ( pos.x < nodepos.x ) nodepos.x -= range.x;
	if ( pos.y < nodepos.y ) nodepos.y -= range.y;
	if ( pos.z < nodepos.z ) nodepos.z -= range.z;

	return nodepos;
}

inline __device__ int3 GetCoveringNode (float3 pos, int range)
{
	int3 nodepos;

	nodepos.x = ceil(pos.x / range) * range;
	nodepos.y = ceil(pos.y / range) * range;
	nodepos.z = ceil(pos.z / range) * range;
	if ( pos.x < nodepos.x ) nodepos.x -= range;
	if ( pos.y < nodepos.y ) nodepos.y -= range;
	if ( pos.z < nodepos.z ) nodepos.z -= range;

	return nodepos;
}

inline __device__ bool IsBoxIntersection (int3 amin, int3 amax, int3 bmin, int3 bmax)
{
	return (amin.x <= bmax.x && amax.x >= bmin.x) &&
         (amin.y <= bmax.y && amax.y >= bmin.y) &&
         (amin.z <= bmax.z && amax.z >= bmin.z);
}

extern "C" __global__ void gvdbSetFlagSubcell(VDBInfo* gvdb, int num_sc, int* sc_flag)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_sc) return;

	VDBNode* node = getNode(gvdb, 0, i);

	sc_flag[i] = (node->mParent == ID_UNDEF64) ? 0 : 1;
}

extern "C" __global__ void gvdbConvAndTransform ( int num_pnts, char* psrc, char psrcbits, char* pdest, char pdestbits,
									float3 wmin, float3 wdelta, float3 trans, float3 scal )
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= num_pnts ) return;

	float3 pnt;

	// unpack input format
	switch ( psrcbits ) {
	case 2: { 
		ushort* pbuf = (ushort*) (psrc+i*6);
		pnt = make_float3( *pbuf, *(pbuf+1), *(pbuf+2) ); 
		} break;
	case 4: {
		float* pbuf = (float*) (psrc+i*12);
		pnt = make_float3( *pbuf, *(pbuf+1), *(pbuf+2) );
		} break;
	};
	// scale and transform
	pnt = (wmin + pnt * wdelta) * scal + trans;

	*(float3*) (pdest + i*12) = pnt;
}

extern "C" __global__ void gvdbScalePntPos (int num_pnts, char* ppos, int pos_off, int pos_stride, float scale)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= num_pnts ) return;

	(*(float3*) (ppos + i*pos_stride + pos_off)) *= scale;
}


extern "C" __global__ void gvdbInsertSubcell_fp16 (
	VDBInfo* gvdb, int subcell_size, int sc_per_brick, int num_pnts,
	int* sc_cnt, int* sc_offset, int* sc_mapping,
	float3 pos_min, float3 pos_range, float3 vel_min, float3 vel_range,
	int3 sc_range, float3 ptrans, int res, float radius,	
	char* ppos, int pos_off, int pos_stride, ushort3* sc_pnt_pos,
	char* pvel, int vel_off, int vel_stride, ushort3* sc_pnt_vel,
	char* pclr, int clr_off, int clr_stride, uint* sc_pnt_clr	
)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_pnts) return;

	float3 wpos = (*(float3*)(ppos + i*pos_stride + pos_off)) + ptrans; // NOTE: +ptrans is below. Allows check for wpos.z==NOHIT 

	if (wpos.z == NOHIT) return;							// If position invalid, return. 
	if (wpos.x < 0 || wpos.y < 0 || wpos.z < 0) return;	// robust test

	float3 wvel = (pvel == 0x0) ? make_float3(0, 0, 0) : (*(float3*)(pvel + i*vel_stride + vel_off));
	uint wclr = (pclr == 0x0) ? 0 : (*(uint*)(pclr + i*clr_stride + clr_off));

	int scminx = (int(wpos.x - radius) / subcell_size) * subcell_size;
	int scminy = (int(wpos.y - radius) / subcell_size) * subcell_size;
	int scminz = (int(wpos.z - radius) / subcell_size) * subcell_size;
	int scmaxx = (int(wpos.x + 1 + radius) / subcell_size) * subcell_size;
	int scmaxy = (int(wpos.y + 1 + radius) / subcell_size) * subcell_size;
	int scmaxz = (int(wpos.z + 1 + radius) / subcell_size) * subcell_size;

	VDBNode* pnode;
	int3 scPos, posInNode;
	int pnodeId, localOffs, offset, sc_idx;
	for (scPos.x = scminx; scPos.x <= scmaxx; scPos.x += subcell_size)
	{
		for (scPos.y = scminy; scPos.y <= scmaxy; scPos.y += subcell_size)
		{
			for (scPos.z = scminz; scPos.z <= scmaxz; scPos.z += subcell_size)
			{
				pnodeId = getPosLeafParent(gvdb, scPos);
				if (pnodeId == ID_UNDEFL) continue;
				//getNode ( 0, pnodeId);
				pnode = (VDBNode*)(gvdb->nodelist[0] + pnodeId*gvdb->nodewid[0]);

				posInNode = scPos - pnode->mPos;
				posInNode.x /= subcell_size;
				posInNode.y /= subcell_size;
				posInNode.z /= subcell_size;
				localOffs = (posInNode.z*res + posInNode.y)*res + posInNode.x;

				sc_idx = sc_mapping[pnodeId] * sc_per_brick + localOffs;
				offset = atomicAdd(&sc_cnt[sc_idx], (uint)1);
				sc_pnt_pos[sc_offset[sc_idx] + offset] = (make_ushort3)(
					(wpos.x - pos_min.x) / pos_range.x * 65535,
					(wpos.y - pos_min.y) / pos_range.y * 65535,
					(wpos.z - pos_min.z) / pos_range.z * 65535);
				if (pvel != 0x0) sc_pnt_vel[sc_offset[sc_idx] + offset] = (make_ushort3)(
					(wvel.x - vel_min.x) / vel_range.x * 65535,
					(wvel.y - vel_min.y) / vel_range.y * 65535,
					(wvel.z - vel_min.z) / vel_range.z * 65535);
				if (pclr != 0x0) sc_pnt_clr[sc_offset[sc_idx] + offset] = wclr;
			}
		}
	}
}

extern "C" __global__ void gvdbInsertSubcell (
	VDBInfo* gvdb, int subcell_size, int sc_per_brick, int num_pnts,
	int* sc_cnt, int* sc_offset, int* sc_mapping,	
	int3 sc_range, float3 ptrans, int res, float radius,
	char* ppos, int pos_off, int pos_stride, float3* sc_pnt_pos,
	char* pvel, int vel_off, int vel_stride, float3* sc_pnt_vel,
	char* pclr, int clr_off, int clr_stride, uint* sc_pnt_clr
)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= num_pnts ) return;

	float3 wpos = (*(float3*) (ppos + i*pos_stride + pos_off));
	if ( wpos.z == NOHIT ) return;							// If position invalid, return. 
	if ( wpos.x < 0 || wpos.y < 0 || wpos.z < 0) return;	// robust test
	wpos += ptrans;			// add ptrans here to allow check for NOHIT

	float3 wvel = (pvel==0x0) ? make_float3(0,0,0) : (*(float3*) (pvel + i*vel_stride + vel_off));
	uint wclr = (pclr== 0x0) ? 0 : (*(uint*)(pclr + i*clr_stride + clr_off));

	int scminx = (int(wpos.x - radius) / subcell_size) * subcell_size;
	int scminy = (int(wpos.y - radius) / subcell_size) * subcell_size;
	int scminz = (int(wpos.z - radius) / subcell_size) * subcell_size;
	int scmaxx = (int(wpos.x + 1 + radius) / subcell_size) * subcell_size;
	int scmaxy = (int(wpos.y + 1 + radius) / subcell_size) * subcell_size;
	int scmaxz = (int(wpos.z + 1 + radius) / subcell_size) * subcell_size;

	VDBNode* pnode;
	int3 scPos, posInNode;
	int pnodeId, localOffs, offset, sc_idx;
	for (scPos.x = scminx; scPos.x <= scmaxx; scPos.x += subcell_size)
	{
		for (scPos.y = scminy; scPos.y <= scmaxy; scPos.y += subcell_size)
		{
			for (scPos.z = scminz; scPos.z <= scmaxz; scPos.z += subcell_size)
			{
				pnodeId = getPosLeafParent( gvdb, scPos);
				if (pnodeId == ID_UNDEFL) continue;
				//getNode ( 0, pnodeId);
				pnode = (VDBNode*) (gvdb->nodelist[0] + pnodeId*gvdb->nodewid[0]);

				posInNode = scPos - pnode->mPos;
				posInNode.x /= subcell_size;
				posInNode.y /= subcell_size;
				posInNode.z /= subcell_size;
				localOffs = (posInNode.z*res + posInNode.y)*res+ posInNode.x;

				sc_idx = sc_mapping[pnodeId] * sc_per_brick + localOffs;
				offset = atomicAdd( &sc_cnt[sc_idx], (uint) 1);
				sc_pnt_pos[sc_offset[sc_idx] + offset] = wpos;
				if ( pvel != 0x0 ) sc_pnt_vel[sc_offset[sc_idx] + offset] = wvel;
				if ( pclr != 0x0 ) sc_pnt_clr[sc_offset[sc_idx] + offset] = wclr;
			}
		}
	}
}

extern "C" __global__ void gvdbCountSubcell (VDBInfo* gvdb, int subcell_size, int sc_per_brick, int num_pnts, char* ppos, int pos_off, int pos_stride, 
											 int* sc_cnt, int3 sc_range, float3 ptrans, int res, float radius, int num_sc, int* sc_mapping)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= num_pnts ) return;

	float3 wpos = (*(float3*) (ppos + i*pos_stride + pos_off));
	if ( wpos.z == NOHIT ) return;							// If position invalid, return. 
	wpos += ptrans;											// add here to allow check above for NOHIT
	if ( wpos.x < 0 || wpos.y < 0 || wpos.z < 0) return;	// robust test

	int scminx = (int(wpos.x - radius) / subcell_size) * subcell_size;
	int scminy = (int(wpos.y - radius) / subcell_size) * subcell_size;
	int scminz = (int(wpos.z - radius) / subcell_size) * subcell_size;
	int scmaxx = (int(wpos.x + 1 + radius) / subcell_size) * subcell_size;
	int scmaxy = (int(wpos.y + 1 + radius) / subcell_size) * subcell_size;
	int scmaxz = (int(wpos.z + 1 + radius) / subcell_size) * subcell_size;

	VDBNode* pnode;
	int3 scPos, posInNode;
	int pnodeId, localOffs;
	for (scPos.x = scminx; scPos.x <= scmaxx; scPos.x += subcell_size)
	{
		for (scPos.y = scminy; scPos.y <= scmaxy; scPos.y += subcell_size)
		{
			for (scPos.z = scminz; scPos.z <= scmaxz; scPos.z += subcell_size)
			{
				pnodeId = getPosLeafParent(gvdb, scPos);
				if (pnodeId == ID_UNDEFL) continue;
				pnode = (VDBNode*) (gvdb->nodelist[0] + pnodeId*gvdb->nodewid[0]);
				posInNode = scPos - pnode->mPos;
				posInNode.x /= subcell_size;
				posInNode.y /= subcell_size;
				posInNode.z /= subcell_size;
				localOffs = (posInNode.z*res + posInNode.y)*res+ posInNode.x;

				atomicAdd( &sc_cnt[sc_mapping[pnodeId] * sc_per_brick + localOffs], (uint) 1);
			}
		}
	}
}

extern "C" __global__ void gvdbFindActivBricks (int num_pnts, int lev, int3 brick_range, int dim, char* ppos,  int pos_off, int pos_stride, float3 orig, int3 orig_shift, int* pout)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= num_pnts ) return;

	float3 wpos = (*(float3*) (ppos + i*pos_stride + pos_off)) + orig;
	
	int3 brickPos = GetCoveringNode(wpos, brick_range);																			

	int3 brickIdx3 = make_int3(brickPos.x / brick_range.x, brickPos.y / brick_range.y, brickPos.z / brick_range.z);	// brick idx in global

	int brickIdx = brickIdx3.x + brickIdx3.y * dim + brickIdx3.z * dim * dim;

	//if ( brickIdx >= num_bricks) return;

	//poff[i] = atomicAdd(&pout[brickIdx], 1);
	pout[i] = brickIdx;
}

extern "C" __global__ void gvdbFindUnique ( int num_pnts, long long* pin, int* marker, int* uniqueCnt, int* levCnt)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= num_pnts ) return;

	if ((i == 0) || (pin[i] != pin[i-1])) 
	{
		marker[i] = 1; atomicAdd(&uniqueCnt[0],1);

		int lev = ((pin[i] >> 48) & 0xFF);
		if (lev == 255) return;

		atomicAdd( &levCnt[lev], 1);
	}
}

extern "C" __global__ void gvdbCalcBrickId (  int num_pnts, int lev_depth, int* range_res, 
											char* ppos,  int pos_off, int pos_stride, 
											float3 orig, unsigned short* pout)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= num_pnts ) return;

	float3 wpos = (*(float3*) (ppos + i*pos_stride + pos_off)) + orig;
	
	if (wpos.x < 0 || wpos.y < 0 || wpos.z < 0)
	{
		for (unsigned short lev = 0; lev < lev_depth; lev++) 
		{
			pout[i * lev_depth * 4 + lev * 4 + 3] = 255;
			pout[i * lev_depth * 4 + lev * 4 + 2] = 0;
			pout[i * lev_depth * 4 + lev * 4 + 1] = 0;
			pout[i * lev_depth * 4 + lev * 4 + 0] = 0;
		}
		return;
	}

	for (unsigned short lev = 0; lev < lev_depth; lev++) 
	{
		int3 brickPos = GetCoveringNode( wpos, range_res[lev]);	

		pout[i * lev_depth * 4 + lev * 4 + 3] = lev; 
		pout[i * lev_depth * 4 + lev * 4 + 2] = (unsigned short) (brickPos.x / range_res[lev]); 
		pout[i * lev_depth * 4 + lev * 4 + 1] = (unsigned short) (brickPos.y / range_res[lev]); 
		pout[i * lev_depth * 4 + lev * 4 + 0] = (unsigned short) (brickPos.z / range_res[lev]); 
	}
}

extern "C" __global__ void gvdbCalcIncreBrickId (VDBInfo* gvdb, float radius, int num_pnts, int lev_depth, int* range_res,
												 char* ppos,  int pos_off, int pos_stride, 
												 float3 orig, unsigned short* pout, int* exBrick_cnt,
												 int* node_markers
												 )
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= num_pnts ) return;

	float3 wpos = (*(float3*) (ppos + i*pos_stride + pos_off)) + orig;
	
	if (wpos.x < 0 || wpos.y < 0 || wpos.z < 0) return;

	int pnodeId, idx;
	int3 brickPos;

	for (unsigned short lev = 0; lev < lev_depth; lev++) 
	{
		brickPos = GetCoveringNode(wpos, range_res[lev]);	
		pnodeId = getPosParent(gvdb, brickPos, lev);

		if (pnodeId == ID_UNDEFL) 
		{
			idx = atomicAdd ( &exBrick_cnt[0], (uint) 1 );
			pout[idx * 4 + 0 ] = (unsigned short) (brickPos.z / range_res[lev]); 
			pout[idx * 4 + 1 ] = (unsigned short) (brickPos.y / range_res[lev]); 
			pout[idx * 4 + 2 ] = (unsigned short) (brickPos.x / range_res[lev]); 
			pout[idx * 4 + 3 ] = lev; 
		}
		else 
		{
			if(lev == 0) atomicOr ( &node_markers[pnodeId], true );
		}
	}
}

extern "C" __global__ void gvdbCalcIncreExtraBrickId (VDBInfo* gvdb, float radius, int num_pnts, int lev_depth, int* range_res,
														char* ppos,  int pos_off, int pos_stride, 
														float3 orig, unsigned short* pout, int* exBrick_cnt,
														int* node_markers
														)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= num_pnts ) return;

	float3 wpos = (*(float3*) (ppos + i*pos_stride + pos_off)) + orig;
	
	if (wpos.x < 0 || wpos.y < 0 || wpos.z < 0) return;

	int3 bmin = make_int3(int(wpos.x) - radius, int(wpos.y) - radius, int(wpos.z) - radius); 
	int3 bmax = make_int3(int(wpos.x) + radius, int(wpos.y) + radius, int(wpos.z) + radius); 

	int3 ndPos;
	int pnodeId, idx;
	VDBNode* node;
	//unsigned short levxyx[40];
	//int ndCnt = 0;

	for (unsigned short lev = 0; lev < lev_depth; lev++) 
	{
		int rres = range_res[lev];
		for (ndPos.x = bmin.x / rres * rres; ndPos.x <= bmax.x / rres * rres; ndPos.x += rres)
		{
			for (ndPos.y = bmin.y / rres * rres; ndPos.y <= bmax.y / rres * rres; ndPos.y += rres)
			{
				for (ndPos.z = bmin.z / rres * rres; ndPos.z <= bmax.z / rres * rres; ndPos.z += rres)
				{
					//ndPos = make_int3(ndx, ndy, ndz);
					pnodeId = getPosParent(gvdb, ndPos, lev);
					

					if (pnodeId == ID_UNDEFL) 
					{
						idx = atomicAdd ( &exBrick_cnt[0], (uint) 1 );
						pout[idx * 4 + 0 ] = (unsigned short) (ndPos.z / rres); 
						pout[idx * 4 + 1 ] = (unsigned short) (ndPos.y / rres); 
						pout[idx * 4 + 2 ] = (unsigned short) (ndPos.x / rres); 
						pout[idx * 4 + 3 ] = lev; 
						//levxyx[ndCnt * 4 + 0 ] = (ndPos.z / rres); 
						//levxyx[ndCnt * 4 + 1 ] = (ndPos.y / rres); 
						//levxyx[ndCnt * 4 + 2 ] = (ndPos.x / rres); 
						//levxyx[ndCnt * 4 + 3 ] = lev; 
						//ndCnt++;
					} else {						
						if(lev == 0) atomicOr ( &node_markers[pnodeId], true );
					}
				}
			}
		}
	}

	//idx = atomicAdd ( &exBrick_cnt[0], (uint) ndCnt);
	//for (int pi = 0; pi < ndCnt * 4; pi++) pout[idx * 4 + pi] = levxyx[pi]; 
	//ndCnt = 0;
	//for (unsigned short lev = 0; lev < lev_depth; lev++) 
	//{
	//	int rres = range_res[lev];
	//	for (ndPos.x = bmin.x / rres * rres; ndPos.x <= bmax.x / rres * rres; ndPos.x += rres)
	//	{
	//		for (ndPos.y = bmin.y / rres * rres; ndPos.y <= bmax.y / rres * rres; ndPos.y += rres)
	//		{
	//			for (ndPos.z = bmin.z / rres * rres; ndPos.z <= bmax.z / rres * rres; ndPos.z += rres)
	//			{
	//				//ndPos = make_int3(ndx, ndy, ndz);
	//				pnodeId = getPosParent(ndPos, lev);
	//				if (pnodeId == ID_UNDEFL) 
	//				{
	//					//idx = atomicAdd ( &exBrick_cnt[0], (uint) 1 );
	//					pout[idx * 4 + 0 + ndCnt ] = (unsigned short) (ndPos.z / rres); 
	//					pout[idx * 4 + 1 + ndCnt ] = (unsigned short) (ndPos.y / rres); 
	//					pout[idx * 4 + 2 + ndCnt ] = (unsigned short) (ndPos.x / rres); 
	//					pout[idx * 4 + 3 + ndCnt ] = lev; 
	//					ndCnt += 4; 
	//				}
	//			}
	//		}
	//	}
	//}
}

extern "C" __global__ void gvdbCalcExtraBrickId (VDBInfo* gvdb, float radius, int num_pnts, int lev_depth, int* range_res,
														char* ppos,  int pos_off, int pos_stride, 
														float3 orig, unsigned short* pout, int* exBrick_cnt)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= num_pnts ) return;

	float3 wpos = (*(float3*) (ppos + i*pos_stride + pos_off)) + orig;
	
	if (wpos.x < 0 || wpos.y < 0 || wpos.z < 0) return;

	//int3 bmin = make_int3(wpos.x - radius, wpos.y - radius, wpos.z - radius); 
	//int3 bmax = make_int3(wpos.x + radius, wpos.y + radius, wpos.z + radius); 

	int3 bmin = make_int3(int(wpos.x) - radius, int(wpos.y) - radius, int(wpos.z) - radius); 
	int3 bmax = make_int3(int(wpos.x) + 2 * radius, int(wpos.y) + 2 * radius, int(wpos.z) + 2 * radius); 

	//int3 ndmin, ndmax;
	int3 ndPos;
	int pnodeId, idx;
	for (unsigned short lev = 0; lev < lev_depth; lev++) 
	{
		//ndmin = make_int3(int(wpos.x - radius) / range_res[lev] * range_res[lev], int(wpos.y - radius) / range_res[lev] * range_res[lev], int(wpos.z - radius) / range_res[lev] * range_res[lev]);
		//ndmax = make_int3(int(wpos.x + radius) / range_res[lev] * range_res[lev], int(wpos.y + radius) / range_res[lev] * range_res[lev], int(wpos.z + radius) / range_res[lev] * range_res[lev]);
		int rres = range_res[lev];
		for (ndPos.x = bmin.x / rres * rres; ndPos.x <= bmax.x / rres * rres; ndPos.x += rres)
		{
			for (ndPos.y = bmin.y / rres * rres; ndPos.y <= bmax.y / rres * rres; ndPos.y += rres)
			{
				for (ndPos.z = bmin.z / rres * rres; ndPos.z <= bmax.z / rres * rres; ndPos.z += rres)
				{
					//ndPos = make_int3(ndx, ndy, ndz);
					pnodeId = getPosParent(gvdb, ndPos, lev);
					if (pnodeId == ID_UNDEFL) 
					{
						idx = atomicAdd ( &exBrick_cnt[0], (uint) 1 );
						pout[idx * 4 + 0 ] = (unsigned short) (ndPos.z / rres); 
						pout[idx * 4 + 1 ] = (unsigned short) (ndPos.y / rres); 
						pout[idx * 4 + 2 ] = (unsigned short) (ndPos.x / rres); 
						pout[idx * 4 + 3 ] = lev; 
					}
				}
			}
		}
	}
}

 

inline __device__ float distFunc ( float3 a, float bx, float by, float bz, float r )
{
	bx -= a.x; by -= a.y; bz -= a.z;	
	float c = (bx*bx+by*by+bz*bz) / (r*r);
	return 1.0 + c*(-3 + c*(3-c));	

	//return (r - sqrt(bx*bx+by*by+bz*bz)) / r;
}

extern "C" __global__ void gvdbScatterPointDensity (VDBInfo* gvdb, int num_pnts, float radius, float amp, char* ppos, int pos_off, int pos_stride, char* pclr, int clr_off, int clr_stride, int* pnode, float3 ptrans, bool expand, uint* colorBuf)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= num_pnts ) return;
	if ( pnode[i] == ID_UNDEFL ) return;		// make sure point is inside a brick
	
	// Get particle position in brick	
	float3 wpos = (*(float3*) (ppos + i*pos_stride + pos_off)) + ptrans;	
	float3 vmin;
	float w;
	VDBNode* node = getNode ( gvdb, 0, pnode[i], &vmin );			// Get node		
	float3 p = (wpos-vmin)/gvdb->vdel[0];
	float3 pi = make_float3(int(p.x), int(p.y), int(p.z));

	// range of pi.x,pi.y,pi.z = [0, gvdb->res0-1]
	if ( pi.x < 0 || pi.y < 0 || pi.z < 0 || pi.x >= gvdb->res[0] || pi.y >= gvdb->res[0] || pi.z >= gvdb->res[0] ) return;
	uint3 q = make_uint3(pi.x,pi.y,pi.z) + make_uint3( node->mValue );	

	w = tex3D<float>( gvdb->volIn[0], q.x,q.y,q.z ) + distFunc(p, pi.x, pi.y,pi.z, radius) ;				surf3Dwrite ( w, gvdb->volOut[0], q.x*sizeof(float), q.y, q.z );

	if ( expand ) {		
		w = tex3D<float> (gvdb->volIn[0], q.x-1,q.y,q.z) + distFunc(p, pi.x-1, pi.y, pi.z, radius);		surf3Dwrite ( w, gvdb->volOut[0], (q.x-1)*sizeof(float), q.y, q.z );
		w = tex3D<float> (gvdb->volIn[0], q.x+1,q.y,q.z) + distFunc(p, pi.x+1, pi.y, pi.z, radius);		surf3Dwrite ( w, gvdb->volOut[0], (q.x+1)*sizeof(float), q.y, q.z );
		w = tex3D<float> (gvdb->volIn[0], q.x,q.y-1,q.z) + distFunc(p, pi.x, pi.y-1, pi.z, radius);		surf3Dwrite ( w, gvdb->volOut[0], q.x*sizeof(float), (q.y-1), q.z );
		w = tex3D<float> (gvdb->volIn[0], q.x,q.y+1,q.z) + distFunc(p, pi.x, pi.y+1, pi.z, radius); 		surf3Dwrite ( w, gvdb->volOut[0], q.x*sizeof(float), (q.y+1), q.z );
		w = tex3D<float> (gvdb->volIn[0], q.x,q.y,q.z-1) + distFunc(p, pi.x, pi.y, pi.z-1, radius);		surf3Dwrite ( w, gvdb->volOut[0], q.x*sizeof(float), q.y, (q.z-1) );
		w = tex3D<float> (gvdb->volIn[0], q.x,q.y,q.z+1) + distFunc(p, pi.x, pi.y, pi.z+1, radius);		surf3Dwrite ( w, gvdb->volOut[0], q.x*sizeof(float), q.y, (q.z+1) );
	}

	if ( pclr != 0 ) {
		uchar4 wclr = *(uchar4*) (pclr + i*clr_stride + clr_off );

		if ( colorBuf != 0 ) {	
			// Increment index
			uint brickres = gvdb->res[0];
			uint vid = (brickres * brickres * brickres * pnode[i]) + (brickres * brickres * (uint)pi.z) + (brickres * (uint)pi.y) + (uint)pi.x;
			uint colorIdx = vid * 4;
		
			// Store in color in the colorbuf
			atomicAdd(&colorBuf[colorIdx + 0], 1);
			atomicAdd(&colorBuf[colorIdx + 1], wclr.x);
			atomicAdd(&colorBuf[colorIdx + 2], wclr.y);
			atomicAdd(&colorBuf[colorIdx + 3], wclr.z);
		}
		else {
		 	surf3Dwrite(wclr, gvdb->volOut[3], q.x*sizeof(uchar4), q.y, q.z);
		}
	}
}

extern "C" __global__ void gvdbAddSupportVoxel (VDBInfo* gvdb, int num_pnts,  float radius, float offset, float amp,
												char* ppos, int pos_off, int pos_stride, 
												char* pdir, int dir_off, int dir_stride, 
												int* pnode, float3 ptrans, bool expand, uint* colorBuf)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= num_pnts ) return;

	// Get brick ID	
	uint  nid = pnode[i];
	if ( nid == ID_UNDEFL ) return;
	
	// Get particle position in brick	
	float3 wpos = (*(float3*) (ppos + i*pos_stride + pos_off)) + ptrans + (*(float3*) (pdir + i*dir_stride + dir_off)) * offset;	
	//wpos.y -=5.0;//threadIdx.y;
	float3 vmin;
	float w;
	VDBNode* node = getNode ( gvdb, 0, pnode[i], &vmin );			// Get node	
	float3 p = (wpos-vmin)/gvdb->vdel[0];
	float3 pi = make_float3(int(p.x), int(p.y), int(p.z));

	// -- should be ok that pi.x,pi.y,pi.z = 0 
	if ( pi.x <= -1 || pi.y <= -1 || pi.z <= -1 || pi.x >= gvdb->res[0] || pi.y >= gvdb->res[0] || pi.z >= gvdb->res[0] ) return;
	uint3 q = make_uint3(pi.x,pi.y,pi.z) + make_uint3( node->mValue );	

	w = tex3D<float>(gvdb->volIn[0], q.x, q.y, q.z ) + distFunc(p, pi.x, pi.y,pi.z, radius);
	surf3Dwrite ( w, gvdb->volOut[0], q.x*sizeof(float), q.y, q.z );
	surf3Dwrite ( (uchar)1, gvdb->volOut[1], q.x*sizeof(uchar), q.y, q.z );
	//surf3Dwrite ( 1.0f, volOut[2], q.x*sizeof(float), q.y, q.z );	

#if 1
	// expand to 3x3 square, write to both volume and material channels
	w = tex3D<float> (gvdb->volIn[0], q.x-1,q.y,q.z) + distFunc(p, pi.x-1, pi.y, pi.z, radius);
	surf3Dwrite ( w, gvdb->volOut[0], (q.x-1)*sizeof(float), q.y, q.z );
	surf3Dwrite ( (uchar)1, gvdb->volOut[1], (q.x-1)*sizeof(uchar), q.y, q.z );
	//surf3Dwrite ( 1.0f, volOut[2], (q.x-1)*sizeof(float), q.y, q.z );

	w = tex3D<float> (gvdb->volIn[0], q.x+1,q.y,q.z) + distFunc(p, pi.x+1, pi.y, pi.z, radius);
	surf3Dwrite ( w, gvdb->volOut[0], (q.x+1)*sizeof(float), q.y, q.z );
	surf3Dwrite ( (uchar)1, gvdb->volOut[1], (q.x+1)*sizeof(uchar), q.y, q.z );
	//surf3Dwrite ( 1.0f, volOut[2], (q.x+1)*sizeof(float), q.y, q.z );

	w = tex3D<float> (gvdb->volIn[0], q.x,q.y,q.z-1) + distFunc(p, pi.x, pi.y, pi.z-1, radius);
	surf3Dwrite ( w, gvdb->volOut[0], q.x*sizeof(float), q.y, (q.z-1) );
	surf3Dwrite ( (uchar)1, gvdb->volOut[1], q.x*sizeof(uchar), q.y, (q.z-1) );
	//surf3Dwrite ( 1.0f, volOut[2], q.x*sizeof(float), q.y, (q.z-1) );

	w = tex3D<float> (gvdb->volIn[0], q.x,q.y,q.z+1) + distFunc(p, pi.x, pi.y, pi.z+1, radius);
	surf3Dwrite ( w, gvdb->volOut[0], q.x*sizeof(float), q.y, (q.z+1) );
	surf3Dwrite ( (uchar)1, gvdb->volOut[1], q.x*sizeof(uchar), q.y, (q.z+1) );
	//surf3Dwrite ( 1.0f, volOut[2], q.x*sizeof(float), q.y, (q.z+1) );

	w = tex3D<float> (gvdb->volIn[0], q.x-1,q.y,q.z-1) + distFunc(p, pi.x-1, pi.y, pi.z-1, radius);
	surf3Dwrite ( w, gvdb->volOut[0], (q.x-1)*sizeof(float), q.y, (q.z-1) );
	surf3Dwrite ( (uchar)1, gvdb->volOut[1], (q.x-1)*sizeof(uchar), q.y, (q.z-1) );
	//surf3Dwrite ( 1.0f, volOut[2], (q.x-1)*sizeof(float), q.y, (q.z-1) );
	w = tex3D<float> (gvdb->volIn[0], q.x+1,q.y,q.z+1) + distFunc(p, pi.x+1, pi.y, pi.z+1, radius);
	surf3Dwrite ( w, gvdb->volOut[0], (q.x+1)*sizeof(float), q.y, (q.z+1) );
	surf3Dwrite ( (uchar)1, gvdb->volOut[1], (q.x+1)*sizeof(uchar), q.y, (q.z+1) );
	//surf3Dwrite ( 1.0f, volOut[2], (q.x+1)*sizeof(float), q.y, (q.z+1) );
	w = tex3D<float> (gvdb->volIn[0], q.x+1,q.y,q.z-1) + distFunc(p, pi.x+1, pi.y, pi.z-1, radius);
	surf3Dwrite ( w, gvdb->volOut[0], (q.x+1)*sizeof(float), q.y, (q.z-1) );
	surf3Dwrite ( (uchar)1, gvdb->volOut[1], (q.x+1)*sizeof(uchar), q.y, (q.z-1) );
	//surf3Dwrite ( 1.0f, volOut[2], (q.x+1)*sizeof(float), q.y, (q.z-1) );
	w = tex3D<float> (gvdb->volIn[0], q.x-1,q.y,q.z+1) + distFunc(p, pi.x-1, pi.y, pi.z+1, radius);
	surf3Dwrite ( w, gvdb->volOut[0], (q.x-1)*sizeof(float), q.y, (q.z+1) );
	surf3Dwrite ( (uchar)1, gvdb->volOut[1], (q.x-1)*sizeof(uchar), q.y, (q.z+1) );
	//surf3Dwrite ( 1.0f, volOut[2], (q.x-1)*sizeof(float), q.y, (q.z+1) );
#endif
}

extern "C" __global__ void gvdbScatterPointAvgCol (VDBInfo* gvdb, int num_voxels, uint* colorBuf)
{
  uint vid = blockIdx.x * blockDim.x + threadIdx.x;
  if (vid >= num_voxels) return;

  uint colorIdx = vid * 4;
  uint count = colorBuf[colorIdx + 0];
  if (count > 0)
  {
    // Average color dividing by count
    uint colx = colorBuf[colorIdx + 1] / count;
    uint coly = colorBuf[colorIdx + 2] / count;
    uint colz = colorBuf[colorIdx + 3] / count;
    uchar4 pclr = make_uchar4(colx, coly, colz, 255);

    // Get node
    uint brickres = gvdb->res[0];
    uint nid = vid / (brickres * brickres * brickres);
    float3 vmin;
    VDBNode* node = getNode(gvdb, 0, nid, &vmin);

    // Get local 3d indices
    uint3 pi;
    pi.x = vid % (brickres);
    pi.y = vid % (brickres * brickres) / (brickres);
    pi.z = vid % (brickres * brickres * brickres) / (brickres * brickres);
    
    // Get global atlas index
    uint3 q = make_uint3(pi.x, pi.y, pi.z) + make_uint3(node->mValue);
    
    surf3Dwrite(pclr, gvdb->volOut[1], q.x*sizeof(uchar4), q.y, q.z);
  }
}

extern "C" __global__ void gvdbReadGridVel (VDBInfo* gvdb, int cell_num, int3* cell_pos, float* cell_vel)
{
	uint cid = blockIdx.x * blockDim.x + threadIdx.x;
	if (cid >= cell_num) return;

	float3 wpos = make_float3(cell_pos[cid].x, cell_pos[cid].y, cell_pos[cid].z);

	float3 vmin, vdel;										
	VDBNode* node = getleafNodeAtPoint ( gvdb, wpos, &vmin, &vdel);	
	if ( node == 0x0 ) { cell_vel[cid] = 0.0f; return; }

	//cell_vel[cid] = -1.0f;

	int3 vox = node->mValue + make_int3((wpos.x - vmin.x)/vdel.x, (wpos.y - vmin.y)/vdel.y, (wpos.z - vmin.z)/vdel.z);

	cell_vel[cid] = (tex3D<float> ( gvdb->volIn[3], vox.x + 0.5f, vox.y + 0.5f, vox.z + 0.5f));
	//cell_vel[cid] = (tex3D<float> ( volIn[9], vox.x, vox.y, vox.z));
}

extern "C" __global__ void gvdbMapExtraGVDB (VDBInfo* gvdb, int numBricks, int sc_dim, int sc_per_brick, int subcell_size, int* sc_mapping, VDBInfo* obs, int* sc_obs_nid)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= numBricks) return;

	VDBNode* pnode = getNode ( gvdb, 0, i );
	if (pnode->mParent == ID_UNDEF64) return;
	int3 pos = pnode->mPos;

	int obs_nid = getPosLeafParent(obs, pos);

	for (int sc = 0; sc < sc_per_brick; sc++)
	{
		sc_obs_nid[sc_mapping[i] * sc_per_brick + sc] = obs_nid;
	}
}

extern "C" __global__ void gvdbCalcSubcellPos (VDBInfo* gvdb, int* sc_nid, int3* sc_pos, int numBricks, int sc_dim, int sc_per_brick, int subcell_size, int* sc_mapping)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= numBricks) return;
	//if (sc_mapping[i] < 0) return;

	VDBNode* pnode = getNode ( gvdb, 0, i );
	if (pnode->mParent == ID_UNDEF64) return;
	int3 pos = pnode->mPos;

	for (int sc = 0; sc < sc_per_brick; sc++)
	{
		sc_pos[sc_mapping[i] * sc_per_brick + sc].x = pos.x + sc % sc_dim * subcell_size;
		sc_pos[sc_mapping[i] * sc_per_brick + sc].y = pos.y + (sc / sc_dim) % sc_dim * subcell_size;
		sc_pos[sc_mapping[i] * sc_per_brick + sc].z = pos.z + (sc / sc_dim / sc_dim) % sc_dim* subcell_size;
		sc_nid[sc_mapping[i] * sc_per_brick + sc] = i;
	}
}

extern "C" __global__ void gvdbGatherDensity (VDBInfo* gvdb, int num_pnts, int num_sc, float radius,
	int* sc_nid, int* sc_cnt, int* sc_off, int3* sc_pos,
	float3* sc_pnt_pos, float3* sc_pnt_vel, uchar4* sc_pnt_clr,
	int chanDensity, int chanClr, bool bAccumulate )
{
	int sc_id = blockIdx.x;				// current subcell ID
	if (sc_id >= num_sc) return;

	int3 wpos; 
	wpos.x = sc_pos[sc_id].x + int(threadIdx.x);	wpos.y = sc_pos[sc_id].y + int(threadIdx.y);	wpos.z = sc_pos[sc_id].z + int(threadIdx.z);
	
	VDBNode* node = getNode(gvdb, 0, sc_nid[sc_id]);
	float3 vmin = node->mPos * gvdb->voxelsize;
	float3 vdel = gvdb->vdel[0];
	int3 vox = node->mValue + make_int3((wpos.x - vmin.x) / vdel.x, (wpos.y - vmin.y) / vdel.y, (wpos.z - vmin.z) / vdel.z);

	float3 jpos;
	float4 clr = make_float4(0,0,0,1);
	float val = 0, c = 0.0f;

	if ( bAccumulate ) {
		val = tex3D<float> ( gvdb->volIn[chanDensity], vox.x + 0.5f, vox.y + 0.5f, vox.z + 0.5f );
		if ( sc_pnt_clr != 0x0 ) clr = CHAR2CLR(tex3D<uchar4>(gvdb->volIn[chanClr], vox.x+ 0.5f, vox.y + 0.5f, vox.z + 0.5f) );
	}
	for (int j = 0; j < sc_cnt[sc_id]; j++) {
		jpos = sc_pnt_pos[sc_off[sc_id] + j] - make_float3(wpos);
		c = sqrtf(jpos.x*jpos.x + jpos.y*jpos.y + jpos.z*jpos.z);
		val = max(val, radius - c);
		if (sc_pnt_clr != 0x0) clr += CHAR2CLR(sc_pnt_clr[sc_off[sc_id] + j]);
	}
	surf3Dwrite( val, gvdb->volOut[chanDensity], vox.x * sizeof(float), vox.y, vox.z);

	if (sc_pnt_clr != 0x0) {	
		clr /= float(sc_cnt[sc_id] + (bAccumulate ? 1 : 0) );		
		surf3Dwrite( CLR2CHAR(clr), gvdb->volOut[chanClr], vox.x * sizeof(uchar4), vox.y, vox.z);
	}
}


extern "C" __global__ void gvdbGatherLevelSet (VDBInfo* gvdb, int num_pnts, int num_sc, float radius,
	int* sc_nid, int* sc_cnt, int* sc_off, int3* sc_pos,
	float3* sc_pnt_pos, float3* sc_pnt_vel, uchar4* sc_pnt_clr,
	int chanLevelset, int chanClr, bool bAccumulate)
{
	int sc_id = blockIdx.x;				// current subcell ID
	if (sc_id >= num_sc) return;

	int3 wpos; 
	wpos.x = sc_pos[sc_id].x + int(threadIdx.x);	wpos.y = sc_pos[sc_id].y + int(threadIdx.y);	wpos.z = sc_pos[sc_id].z + int(threadIdx.z);
	
	VDBNode* node = getNode(gvdb, 0, sc_nid[sc_id]);
	float3 vmin = node->mPos * gvdb->voxelsize;
	float3 vdel = gvdb->vdel[0];
	int3 vox = node->mValue + make_int3((wpos.x - vmin.x) / vdel.x, (wpos.y - vmin.y) / vdel.y, (wpos.z - vmin.z) / vdel.z);

	float3 jpos;
	float4 clr = make_float4(0,0,0,1);
	float dist = 3.0f, c = 0.0f;

	if ( bAccumulate ) {
		dist = tex3D<float> ( gvdb->volIn[chanLevelset ], vox.x + 0.5f, vox.y + 0.5f, vox.z + 0.5f );
		if ( sc_pnt_clr != 0x0 ) clr = CHAR2CLR(tex3D<uchar4>(gvdb->volIn[chanClr], vox.x + 0.5f, vox.y + 0.5f, vox.z + 0.5f) );
	}
	for (int j = 0; j < sc_cnt[sc_id]; j++) {
		jpos = sc_pnt_pos[sc_off[sc_id] + j] - make_float3(wpos);
		c = sqrtf(jpos.x*jpos.x + jpos.y*jpos.y + jpos.z*jpos.z);
		dist = min(dist, c - radius);
		if (sc_pnt_clr != 0x0) clr += CHAR2CLR(sc_pnt_clr[sc_off[sc_id] + j]);
	}
	surf3Dwrite( dist, gvdb->volOut[chanLevelset], vox.x * sizeof(float), vox.y, vox.z);

	if (sc_pnt_clr != 0x0) {	
		clr /= float(sc_cnt[sc_id] + (bAccumulate ? 1 : 0) );		
		surf3Dwrite( CLR2CHAR(clr), gvdb->volOut[chanClr], vox.x * sizeof(uint), vox.y, vox.z);
	}
}

extern "C" __global__ void gvdbGatherLevelSet_fp16(VDBInfo* gvdb, int num_pnts, int num_sc, float radius,
	float3 pos_min, float3 pos_range, float3 vel_min, float3 vel_range,
	int* sc_nid, int* sc_cnt, int* sc_off, int3* sc_pos,
	ushort3* sc_pnt_pos, ushort3* sc_pnt_vel, uint* sc_pnt_clr,
	int chanLevelset, int chanClr)
{
	int sc_id = blockIdx.x;				// current subcell ID
	if (sc_id >= num_sc) return;

	int3 wpos; wpos.x = sc_pos[sc_id].x + int(threadIdx.x);	wpos.y = sc_pos[sc_id].y + int(threadIdx.y);	wpos.z = sc_pos[sc_id].z + int(threadIdx.z);
	
	VDBNode* node = getNode(gvdb, 0, sc_nid[sc_id]);
	float3 vmin = node->mPos * gvdb->voxelsize;
	float3 vdel = gvdb->vdel[node->mLev];
	int3 vox = node->mValue + make_int3((wpos.x - vmin.x) / vdel.x, (wpos.y - vmin.y) / vdel.y, (wpos.z - vmin.z) / vdel.z);

	float3 jpos, tmppos;
	float4 clr;
	float c = 0.0f;
	float dist = 3.0f;
	
	for (int j = 0; j < sc_cnt[sc_id]; j++) {
		tmppos.x = sc_pnt_pos[sc_off[sc_id] + j].x / 65535.0f * pos_range.x + pos_min.x;
		tmppos.y = sc_pnt_pos[sc_off[sc_id] + j].y / 65535.0f * pos_range.y + pos_min.y;
		tmppos.z = sc_pnt_pos[sc_off[sc_id] + j].z / 65535.0f * pos_range.z + pos_min.z;

		jpos = tmppos - make_float3(wpos);
		c = sqrtf(jpos.x*jpos.x + jpos.y*jpos.y + jpos.z*jpos.z);
		dist = min(dist, c - radius);
		if (sc_pnt_clr != 0x0) clr += INT2CLR(sc_pnt_clr[sc_off[sc_id] + j]) / c;
	}
	
	surf3Dwrite( dist, gvdb->volOut[chanLevelset], vox.x * sizeof(float), vox.y, vox.z);
	
	if ( sc_pnt_clr != 0x0 )
		surf3Dwrite( CLR2INT(clr), gvdb->volOut[chanClr], vox.x * sizeof(float), vox.y, vox.z);
}



extern "C" __global__ void gvdbCheckVal (VDBInfo* gvdb, float slice, int3 res, int chanVx, int chanVy, int chanVz, int chanVxOld, int chanVyOld, int chanVzOld, float* outbuf1, float* outbuf2 )
												 
{
	uint3 vox = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + threadIdx;
	if ( vox.y >= 1  ) return;
	if ( vox.x > res.x || vox.z > res.z ) return;

	float3 wpos = make_float3(vox) * 0.5;	
	wpos.y = slice;

	float3 vmin, vdel;										
	VDBNode* node = getleafNodeAtPoint ( gvdb, wpos, &vmin, &vdel);
	float3 p = (wpos - vmin) / vdel;
	if ( node == 0x0 ) return; 

	//int pi(wpos.x), fj(wpos.y - 0.5f), pk(wpos.z);
	int fi(wpos.x - 0.5f), pj(wpos.y), fk(wpos.z - 0.5f);

	float vel_y, vel_y2;

	vel_y = 0.0f;
	for (int ii = 0; ii < 2; ii++) {
		for (int jj = 0; jj < 2; jj++) {
			for (int kk = 0; kk < 2; kk++) {
				int3 vox_pos = make_int3(fi+ii, pj+jj, fk+kk);
				int3 tmp_vox = node->mValue + make_int3((vox_pos.x - vmin.x)/vdel.x, (vox_pos.y - vmin.y)/vdel.y, (vox_pos.z - vmin.z)/vdel.z);
				float tmp_vel = tex3D<float> ( gvdb->volIn[chanVy], tmp_vox.x, tmp_vox.y, tmp_vox.z);
				vel_y += tmp_vel * (1.0f - fabs(fi + ii + 0.5f - wpos.x)) * (1.0f - fabs(pj + jj - wpos.y)) * (1.0f - fabs(fk + kk + 0.5f - wpos.z));	
			}
		}
	} 

	float3 sp	= make_float3(node->mValue) + p + make_float3(-0.5f, 0.0f, -0.5f);
	vel_y2		= tex3D<float> ( gvdb->volIn[chanVy], sp.x, sp.y, sp.z );

	outbuf1[ vox.z*res.x + vox.x ] = vel_y;
	outbuf2[ vox.z*res.x + vox.x ] = vel_y2;
}



#define SCAN_BLOCKSIZE		512

extern "C" __global__ void prefixFixup ( uint *input, uint *aux, int len) 
{
    unsigned int t = threadIdx.x;
	unsigned int start = t + 2 * blockIdx.x * SCAN_BLOCKSIZE; 	
	if (start < len)					input[start] += aux[blockIdx.x] ;
	if (start + SCAN_BLOCKSIZE < len)   input[start + SCAN_BLOCKSIZE] += aux[blockIdx.x];
}

extern "C" __global__ void prefixSum ( uint* input, uint* output, uint* aux, int len, int zeroff )
{
    __shared__ uint scan_array[SCAN_BLOCKSIZE << 1];    
	unsigned int t1 = threadIdx.x + 2 * blockIdx.x * SCAN_BLOCKSIZE;
	unsigned int t2 = t1 + SCAN_BLOCKSIZE;
    
	// Pre-load into shared memory
    scan_array[threadIdx.x] = (t1<len) ? input[t1] : 0.0f;
	scan_array[threadIdx.x + SCAN_BLOCKSIZE] = (t2<len) ? input[t2] : 0.0f;
    __syncthreads();

    // Reduction
    int stride;
    for (stride = 1; stride <= SCAN_BLOCKSIZE; stride <<= 1) {
       int index = (threadIdx.x + 1) * stride * 2 - 1;
       if (index < 2 * SCAN_BLOCKSIZE)
          scan_array[index] += scan_array[index - stride];
       __syncthreads();
    }

    // Post reduction
    for (stride = SCAN_BLOCKSIZE >> 1; stride > 0; stride >>= 1) {
       int index = (threadIdx.x + 1) * stride * 2 - 1;
       if (index + stride < 2 * SCAN_BLOCKSIZE)
          scan_array[index + stride] += scan_array[index];
       __syncthreads();
    }
	__syncthreads();
	
	// Output values & aux
	if (t1+zeroff < len)	output[t1+zeroff] = scan_array[threadIdx.x];
	if (t2+zeroff < len)	output[t2+zeroff] = (threadIdx.x==SCAN_BLOCKSIZE-1 && zeroff) ? 0 : scan_array[threadIdx.x + SCAN_BLOCKSIZE];	
	if ( threadIdx.x == 0 ) {
		if ( zeroff ) output[0] = 0;
		if (aux) aux[blockIdx.x] = scan_array[2 * SCAN_BLOCKSIZE - 1];				
	}    	
}

extern "C" __global__ void gvdbInsertTriangles ( float bdiv, int bmax, int* bcnt, int vcnt, int ecnt, float3* vbuf, int* ebuf )
{
	uint n = blockIdx.x * blockDim.x + threadIdx.x;
	if ( n >= ecnt ) return;

	// get transformed triangle
	float3 v0, v1, v2;
	int3 f = make_int3( ebuf[n*3], ebuf[n*3+1], ebuf[n*3+2] );
	v0 = vbuf[f.x << 1]; v0 = mul4x ( v0, cxform );
	v1 = vbuf[f.y << 1]; v1 = mul4x ( v1, cxform );
	v2 = vbuf[f.z << 1]; v2 = mul4x ( v2, cxform );

	// compute bounds on y-axis	
	float p0, p1;
	fminmax3( v0.y, v1.y, v2.y, p0, p1 );
	p0 = int(p0/bdiv);	p1 = int(p1/bdiv);							// y-min and y-max bins
	
	// scan bins covered by triangle	
	for (int y=p0; y <= p1; y++) {
		atomicAdd ( &bcnt[y], (uint) 1 );							// histogram bin counts
	}	
}

extern "C" __global__ void gvdbCompactUnique(int num_pnts, long long* pin, int* marker, int* offset, long long* pout)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= num_pnts ) return;

	if ( marker[i] > 0 )
	{
		pout[offset[i]] = pin[i]; 
	}
}

#define PRESCAN_BLOCKSIZE		256
extern "C" __global__ void gvdbRadixPrescan(int len, int* input, int* output)
{
	__shared__ int scan_array[PRESCAN_BLOCKSIZE];    
	int ti = threadIdx.x;
	int offset = 1; 

	// Pre-load into shared memory
    scan_array[ti] = input[ti];
	scan_array[ti + 128] = input[ti + 128];
    __syncthreads();

    for (int d = len>>1; d > 0; d >>= 1)
	{
		if (ti < d)
		{
			int ai = offset*(2*ti+1)-1;
			int bi = offset*(2*ti+2)-1;
			scan_array[bi] += scan_array[ai];
		}
		offset *= 2;
		__syncthreads();
	}
	if (ti == 0) { scan_array[len - 1] = 0; } // clear the last element
	__syncthreads();
	for (int d = 1; d < len; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;	
		if (ti < d)
		{
			int ai = offset*(2*ti+1)-1;
			int bi = offset*(2*ti+2)-1;
			int t = scan_array[ai];
			scan_array[ai] = scan_array[bi];
			scan_array[bi] += t;
		}
		__syncthreads();
	} 

	output[ti] = scan_array[ti];
	output[ti + 128] = scan_array[ti + 128];
}

extern "C" __global__ void gvdbRadixBincount (int num_pnts, long long* pin, int* pout, int bpos)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= num_pnts) return;

	long long value = pin[i];
	for (int b = 0; b < bpos; b++)	value >>= 8;

	atomicAdd( &pout[(value & 0xFF)], 1);

	//for (int b = 1; b < 7; b++)
	//{
	//	value >>= 8;	atomicAdd( &pout[(value & 0xFF) + 256 * b], 1);
	//}
}

extern "C" __global__ void gvdbRadixShuffle(int num_pnts, int* binPrefixSum, long long* pin, long long* pout, int bpos)
{
	__shared__ int offset[PRESCAN_BLOCKSIZE]; 

	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= num_pnts) return;

	long long value = pin[i];
	for (int b = 0; b < bpos; b++)	value >>= 8;

	//atomicSub( &binPrefixSum[(value & 0xFF) + 1], 1);

	int offs = atomicAdd( &offset[(value & 0xFF)], 1);
	__syncthreads();

	pout[binPrefixSum[(value & 0xFF)] + offs] = pin[i];
}

// Sort triangles
// Give a list of bins and known offsets (prefixes), and a list of vertices and faces,
// performs a deep copy of triangles into bins, where some may be duplicated.
// This may be used generically by others kernel that need a bin-sorted mesh.
// Input: 
//   bdiv, bmax - input: bins division and maximum number
//   bcnt       - input: number of triangles in each bin
//   boff       - input: starting offset of each bin in triangle buffer
//   vcnt, vbuf - input: vertex buffer (VBO) and number of verts
//   ecnt, ebuf - input: element buffer and number of faces
//   tricnt     - output: total number of triangles when sorted into bins
//   tbuf       - output: triangle buffer: list of bins and their triangles (can be more than vcnt due to overlaps)
extern "C" __global__ void gvdbSortTriangles ( float bdiv, int bmax, int* bcnt, int* boff, int tricnt, float3* tbuf,
													int vcnt, int ecnt, float3* vbuf, int* ebuf )
{
	uint n = blockIdx.x * blockDim.x + threadIdx.x;
	if ( n >= ecnt ) return;

	// get transformed triangle
	float3 v0, v1, v2;
	int3 f = make_int3( ebuf[n*3], ebuf[n*3+1], ebuf[n*3+2] );
	v0 = vbuf[f.x << 1]; v0 = mul4x ( v0, cxform );
	v1 = vbuf[f.y << 1]; v1 = mul4x ( v1, cxform );
	v2 = vbuf[f.z << 1]; v2 = mul4x ( v2, cxform );

	// compute bounds on y-axis	
	float p0, p1;
	fminmax3( v0.y, v1.y, v2.y, p0, p1 );
	p0 = int(p0/bdiv);	p1 = int(p1/bdiv);							// y-min and y-max bins
	if ( p0 >= bmax ) p0 = bmax-1;
	if ( p1 >= bmax ) p1 = bmax-1;
	
	// scan bins covered by triangle	
	int bndx;
	for (int y=p0; y <= p1; y++) {
		bndx = atomicAdd ( &bcnt[y], (uint) 1 );		// get bin index (and histogram bin counts)
		bndx += boff[y];								// get offset into triangle buffer (tbuf)		
		tbuf[ bndx*3   ] = v0;							// deep copy transformed vertices of face
		tbuf[ bndx*3+1 ] = v1;
		tbuf[ bndx*3+2 ] = v2;
	}	
}

extern "C" __global__ void gvdbVoxelize ( float3 vmin, float3 vmax, int3 res, uchar* obuf, uchar otype, 
										  float val_surf, float val_inside, float bdiv, int bmax, int* bcnt, int* boff, 
										  float3* tbuf )							
{
	uint3 t = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + threadIdx;
	if ( t.x >= res.x || t.y >= res.y || t.z >= res.z ) return;
	
	// solid voxelization
	float3 tdel = (vmax-vmin)/make_float3(res);						// width of voxel
	vmin += make_float3(t.x+.5f, t.y+.5f, t.z+.5f)*tdel;		// center of voxel
	float3 v0, v1, v2;
	float3 e0, e1, e2;
	float3 norm, p;		
	float rad;
	int n, cnt = 0;
	int b = vmin.y / bdiv;
	if ( b >= bmax ) b = bmax-1;
	
	for (n=boff[b]; n < boff[b]+bcnt[b]; n++ ) {
		
		v0 = tbuf[n*3];   v0 = (v0 - vmin)/tdel;
		v1 = tbuf[n*3+1]; v1 = (v1 - vmin)/tdel;
		v2 = tbuf[n*3+2]; v2 = (v2 - vmin)/tdel;
		/*f = make_int3( ebuf[n*3], ebuf[n*3+1], ebuf[n*3+2] );
		v0 = vbuf[f.x << 1];		v0 = mul4x ( v0, cxform );	v0 = (v0 - tcent)/tdel;
		v1 = vbuf[f.y << 1];		v1 = mul4x ( v1, cxform );	v1 = (v1 - tcent)/tdel;
		v2 = vbuf[f.z << 1];		v2 = mul4x ( v2, cxform );	v2 = (v2 - tcent)/tdel;*/
		e0 = v1-v0;	e1 = v2-v0;	

		//--- bounding box test
		fminmax3( v0.y, v1.y, v2.y, p.x, p.y );	
		if ( p.x > 0.5f || p.y < -0.5f ) continue; 
		fminmax3( v0.z, v1.z, v2.z, p.x, p.y );	
		if ( p.x > 0.5f || p.y < -0.5f ) continue;		
		fminmax3( v0.x, v1.x, v2.x, p.x, p.y );		
		if ( p.y < -0.5f ) continue;				// x- half space, keep x+ half space

		//--- ray-triangle intersect
		norm.x = 0;		
		e2 = make_float3(0, -e1.z, e1.y);			// P = CROSS(D, e1)		  e2 <=> P,  D={1,0,0}
		p.z = dot ( e0, e2 );						// det = DOT(e0, P)		  p.z <=> det
		if ( p.z > -0.001 && p.z < 0.001 ) norm.x=1;		
		// T=-v0;									// T = SUB(O, v0)         -v0 <=> T  O={0,0,0}
		p.y = dot ( -v0, e2 ) / p.z;				// u = DOT(T, P)*invdet   p.y <=> u
		if ( p.y < 0.f || p.y > 1.f ) norm.x=1;
		e2 = cross ( -v0, e0 );						// Q = CROSS(T, e0)		  e2 <=> Q
		rad = e2.x / p.z;							// v = DOT(D, Q)*invdet   rad <=> v
		if ( rad < 0.f || p.y+rad > 1.f ) norm.x=1;
		rad = dot ( e1, e2 ) / p.z;					// t = DOT(e1, Q)*invdet  rad <=> t
		if ( rad < 0.001f ) norm.x=1;
		if ( norm.x==0 ) cnt++;						// count crossing for inside-outside test (solid voxelize)

		if ( p.x > 0.5f ) continue;					// x+ half space

		//--- fast box-plane test
		e2 = -e1; e1 = v2-v1;					
		norm = cross ( e0, e1 );
		p.x = 0; p.y = 0;	
		if ( norm.x > 0.0f ) { p.x += norm.x*(-0.5f - v0.x); p.y += norm.x*( 0.5f - v0.x); }
		else				 { p.x += norm.x*( 0.5f - v0.x); p.y += norm.x*(-0.5f - v0.x); }
		if ( norm.y > 0.0f ) { p.x += norm.y*(-0.5f - v0.y); p.y += norm.y*( 0.5f - v0.y); }
		else				 { p.x += norm.y*( 0.5f - v0.y); p.y += norm.y*(-0.5f - v0.y); }
		if ( norm.z > 0.0f ) { p.x += norm.z*(-0.5f - v0.z); p.y += norm.z*( 0.5f - v0.z); }
		else				 { p.x += norm.z*( 0.5f - v0.z); p.y += norm.z*(-0.5f - v0.z); }
		if( p.x > 0.0f )		continue;	// do not overlap
		if( p.y < 0.0f )		continue;
			
		//--- schwarz-seidel tests
		rad = (norm.z >= 0) ? 1 : -1;		
		p = make_float3 ( -e0.y*rad, e0.x*rad, 0 );	
		if ( -(p.x+p.y)*0.5f - (p.x*v0.x + p.y*v0.y) + fmaxf(0, p.x) + fmaxf(0, p.y) < 0 ) continue; 	 // no overlap
		p = make_float3( -e1.y*rad, e1.x*rad, 0 ); 		
		if ( -(p.x+p.y)*0.5f - (p.x*v1.x + p.y*v1.y) + fmaxf(0, p.x) + fmaxf(0, p.y) < 0 ) continue; 
		p = make_float3( -e2.y*rad, e2.x*rad, 0 );
		if ( -(p.x+p.y)*0.5f - (p.x*v2.x + p.y*v2.y) + fmaxf(0, p.x) + fmaxf(0, p.y) < 0 ) continue; 
	
		rad = (norm.y >= 0) ? -1 : 1;
		p = make_float3 ( -e0.z*rad, 0, e0.x*rad );	
		if ( -(p.x+p.z)*0.5f - (p.x*v0.x + p.z*v0.z) + fmaxf(0, p.x) + fmaxf(0, p.z) < 0 ) continue; 	 // no overlap		
		p = make_float3 ( -e1.z*rad, 0, e1.x*rad ); 		
		if ( -(p.x+p.z)*0.5f - (p.x*v1.x + p.z*v1.z) + fmaxf(0, p.x) + fmaxf(0, p.z) < 0 ) continue; 
		p = make_float3 ( -e2.z*rad, 0, e2.x*rad );
		if ( -(p.x+p.z)*0.5f - (p.x*v2.x + p.z*v2.z) + fmaxf(0, p.x) + fmaxf(0, p.z) < 0 ) continue; 
	
		rad = (norm.x >= 0) ? 1 : -1;		
		p = make_float3 ( 0, -e0.z*rad, e0.y*rad );	
		if ( -(p.y+p.z)*0.5f - (p.y*v0.y + p.z*v0.z) + fmaxf(0, p.y) + fmaxf(0, p.z) < 0 ) continue; 	 // no overlap		
		p = make_float3 ( 0, -e1.z*rad, e1.y*rad ); 		
		if ( -(p.y+p.z)*0.5f - (p.y*v1.y + p.z*v1.z) + fmaxf(0, p.y) + fmaxf(0, p.z) < 0 ) continue; 
		p = make_float3 ( 0, -e2.z*rad, e2.y*rad );
		if ( -(p.y+p.z)*0.5f - (p.y*v2.y + p.z*v2.z) + fmaxf(0, p.y) + fmaxf(0, p.z) < 0 ) continue;

		//--- akenine-moller tests
		/*p.x = e0.z*v0.y - e0.y*v0.z;							// AXISTEST_X01(e0[Z], e0[Y], fez, fey);
		p.z = e0.z*v2.y - e0.y*v2.z;
		if (p.x<p.z) {min=p.x; max=p.z;} else {min=p.z; max=p.x;} 
		rad = fabsf(e0.z) * 0.5f + fabsf(e0.y) * 0.5f;  
		if (min>rad || max<-rad) continue;

		p.x = -e0.z*v0.x + e0.x*v0.z;		      				// AXISTEST_Y02(e0.z, e0.x, fez, fex);
		p.z = -e0.z*v2.x + e0.x*v2.z;
		if (p.x<p.z) {min=p.x; max=p.z;} else {min=p.z; max=p.x;}
		rad = fabsf(e0.z) * 0.5f + fabsf(e0.x) * 0.5f; 
		if (min>rad || max<-rad) continue;

		p.y = e0.y*v1.x - e0.x*v1.y;								// AXISTEST_Z12(e0.y, e0.x, fey, fex);
		p.z = e0.y*v2.x - e0.x*v2.y;
		if(p.z<p.y) {min=p.z; max=p.y;} else {min=p.y; max=p.z;}
		rad = fabsf(e0.y) * 0.5f + fabsf(e0.x) * 0.5f;  
		if(min>rad || max<-rad) continue;
 
		p.x = e1.z*v0.y - e1.y*v0.z;							// AXISTEST_X01(e1.z, e1.y, fez, fey);
		p.z = e1.z*v2.y - e1.y*v2.z;
		if(p.x<p.z) {min=p.x; max=p.z;} else {min=p.z; max=p.x;} 
		rad = fabsf(e1.z) * 0.5f + fabsf(e1.y) * 0.5f;
		if(min>rad || max<-rad) continue;

		p.x = -e1.z*v0.x + e1.x*v0.z;							// AXISTEST_Y02(e1.z, e1.x, fez, fex);
		p.z = -e1.z*v2.x + e1.x*v2.z;
		if(p.x<p.z) {min=p.x; max=p.z;} else {min=p.z; max=p.x;}
		rad = fabsf(e1.z) * 0.5f + fabsf(e1.x) * 0.5f;
		if(min>rad || max<-rad) continue;

		p.x = e1.y*v0.x - e1.x*v0.y;								// AXISTEST_Z0(e1.y, e1.x, fey, fex);
		p.y = e1.y*v1.x - e1.x*v1.y;
		if(p.x<p.y) {min=p.x; max=p.y;} else {min=p.y; max=p.x;} 
		rad = fabsf(e1.y) * 0.5f + fabsf(e1.x) * 0.5f;
		if(min>rad || max<-rad) continue;
  
		p.x = e2.z*v0.y - e2.y*v0.z;								// AXISTEST_X2(e2.z, e2.y, fez, fey);
		p.y = e2.z*v1.y - e2.y*v1.z;
		if(p.x<p.y) {min=p.x; max=p.y;} else {min=p.y; max=p.x;} 
		rad = fabsf(e2.z) * 0.5f + fabsf(e2.y) * 0.5f; 
		if(min>rad || max<-rad) continue;
	
		p.x = -e2.z*v0.x + e2.x*v0.z;		      				// AXISTEST_Y1(e2.z, e2.x, fez, fex);
		p.y = -e2.z*v1.x + e2.x*v1.z;
		if(p.x<p.y) {min=p.x; max=p.y;} else {min=p.y; max=p.x;} 
		rad = fabsf(e2.z) * 0.5f + fabsf(e2.x) * 0.5f;
		if(min>rad || max<-rad) continue;
	
		p.y = e2.y*v1.x - e2.x*v1.y;								// AXISTEST_Z12(e2.y, e2.x, fey, fex); 
		p.z = e2.y*v2.x - e2.x*v2.y;
		if(p.z<p.y) {min=p.z; max=p.y;} else {min=p.y; max=p.z;} 
		rad = fabsf(e2.y) * 0.5f + fabsf(e2.x) * 0.5f;
		if(min>rad || max<-rad) continue; */
		
		switch ( otype ) {
		case T_UCHAR:	obuf [ (t.z*res.y + t.y)*res.x + t.x ] = (uchar) val_surf;			break;
		case T_FLOAT:	((float*) obuf) [ (t.z*res.y + t.y)*res.x + t.x ] = val_surf;		break;
		case T_INT:		((int*) obuf) [ (t.z*res.y + t.y)*res.x + t.x ] = (int) val_surf;	break;
		};		
		
		break;
	}

	if ( n == boff[b]+bcnt[b] ) {
		// solid voxelization		
		if ( cnt % 2 == 1) {
			switch ( otype ) {
			case T_UCHAR:	obuf [ (t.z*res.y + t.y)*res.x + t.x ] = (uchar) val_inside;		break;
			case T_FLOAT:	((float*) obuf) [ (t.z*res.y + t.y)*res.x + t.x ] = val_inside;		break;
			case T_INT:		((int*) obuf) [ (t.z*res.y + t.y)*res.x + t.x ] = (int) val_inside;	break;
			};		
		}
	}
}

extern "C" __global__ void gvdbBitonicSort(int *dev_values, int num_pnts, int j, int k)
{
	unsigned int i, ixj; /* Sorting partners: i and ixj */
	i = threadIdx.x + blockDim.x * blockIdx.x;
	ixj = i^j;
	
	if ( i >= num_pnts ) return;
	if ( ixj >= num_pnts ) return;
	
	/* The threads with the lowest ids sort the array. */
	if ((ixj)>i) {
		if ((i&k)==0) {
			/* Sort ascending */
			if (dev_values[i]>dev_values[ixj]) {
				int temp = dev_values[i];
				dev_values[i] = dev_values[ixj];
				dev_values[ixj] = temp;
			}
		}
		if ((i&k)!=0) {
			/* Sort descending */
			if (dev_values[i]<dev_values[ixj]) {
				int temp = dev_values[i];
				dev_values[i] = dev_values[ixj];
				dev_values[ixj] = temp;
			}
		}
	}
}



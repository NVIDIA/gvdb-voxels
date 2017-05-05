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


extern "C" __global__ void gvdbInsertPoints ( int num_pnts, char* ppos, int pos_off, int pos_stride, int* pnode, int* poff, int* gcnt, float3 ptrans )
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= num_pnts ) return;

	float3 wpos = (*(float3*) (ppos + i*pos_stride + pos_off)); // NOTE: +ptrans is below. Allows check for wpos.z==NOHIT 

	if ( wpos.z == NOHIT ) { pnode[i] = ID_UNDEFL; return; }		// If position invalid, return. 
	float3 offs, vmin, vdel;										// Get GVDB node at the particle point
	uint64 nid;
	VDBNode* node = getNodeAtPoint ( wpos + ptrans, &offs, &vmin, &vdel, &nid );	
	if ( node == 0x0 ) { pnode[i] = ID_UNDEFL; return; }			// If no brick at location, return.	

	__syncthreads();

	pnode[i] = nid;													// Place point in brick
	poff[i] = atomicAdd ( &gcnt[nid], (uint) 1 );					// Increment brick pcount, and retrieve this point index at the same time
}


extern "C" __global__ void gvdbInsertSupportPoints ( int num_pnts, float offset, char* ppos, int pos_off, int pos_stride, int* pnode, int* poff, int* gcnt, char* pdir, int dir_off, int dir_stride, float3 ptrans )
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= num_pnts ) return;

	float3 wpos = (*(float3*) (ppos + i*pos_stride + pos_off)); // NOTE: +ptrans is below. Allows check for wpos.z==NOHIT 
	float3 wdir = (*(float3*) (pdir + i*dir_stride + dir_off));

	if ( wpos.z == NOHIT ) { pnode[i] = ID_UNDEFL; return; }		// If position invalid, return. 
	float3 offs, vmin, vdel;										// Get GVDB node at the particle point
	uint64 nid;
	VDBNode* node = getNodeAtPoint ( wpos + ptrans + wdir * offset, &offs, &vmin, &vdel, &nid );	
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

inline __device__ float distFunc ( float3 a, float bx, float by, float bz, float r )
{
	bx -= a.x; by -= a.y; bz -= a.z;	
	float c = (bx*bx+by*by+bz*bz) / (r*r);
	return 1.0 + c*(-3 + c*(3-c));	

	//return (r - sqrt(bx*bx+by*by+bz*bz)) / r;
}

extern "C" __global__ void gvdbScatterPointDensity ( int num_pnts, float radius, float amp, char* ppos, int pos_off, int pos_stride, char* pclr, int clr_off, int clr_stride, int* pnode, float3 ptrans, bool expand, uint* colorBuf)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= num_pnts ) return;
	if ( pnode[i] == ID_UNDEFL ) return;		// make sure point is inside a brick
	
	// Get particle position in brick	
	float3 wpos = (*(float3*) (ppos + i*pos_stride + pos_off)) + ptrans;	
	float3 vmin;
	float w;
	VDBNode* node = getNode ( 0, pnode[i], &vmin );			// Get node		
	float3 p = (wpos-vmin)/gvdb.vdel[0];
	float3 pi = make_float3(int(p.x), int(p.y), int(p.z));

	// range of pi.x,pi.y,pi.z = [0, gvdb.res0-1]
	if ( pi.x < 0 || pi.y < 0 || pi.z < 0 || pi.x >= gvdb.res[0] || pi.y >= gvdb.res[0] || pi.z >= gvdb.res[0] ) return;
	uint3 q = make_uint3(pi.x,pi.y,pi.z) + make_uint3( node->mValue );	

	w = tex3D<float>( volIn[0], q.x,q.y,q.z ) + distFunc(p, pi.x, pi.y,pi.z, radius) ;				surf3Dwrite ( w, volOut[0], q.x*sizeof(float), q.y, q.z );

	if ( expand ) {		
		w = tex3D<float> (volIn[0], q.x-1,q.y,q.z) + distFunc(p, pi.x-1, pi.y, pi.z, radius);		surf3Dwrite ( w, volOut[0], (q.x-1)*sizeof(float), q.y, q.z );
		w = tex3D<float> (volIn[0], q.x+1,q.y,q.z) + distFunc(p, pi.x+1, pi.y, pi.z, radius);		surf3Dwrite ( w, volOut[0], (q.x+1)*sizeof(float), q.y, q.z );
		w = tex3D<float> (volIn[0], q.x,q.y-1,q.z) + distFunc(p, pi.x, pi.y-1, pi.z, radius);		surf3Dwrite ( w, volOut[0], q.x*sizeof(float), (q.y-1), q.z );
		w = tex3D<float> (volIn[0], q.x,q.y+1,q.z) + distFunc(p, pi.x, pi.y+1, pi.z, radius); 		surf3Dwrite ( w, volOut[0], q.x*sizeof(float), (q.y+1), q.z );
		w = tex3D<float> (volIn[0], q.x,q.y,q.z-1) + distFunc(p, pi.x, pi.y, pi.z-1, radius);		surf3Dwrite ( w, volOut[0], q.x*sizeof(float), q.y, (q.z-1) );
		w = tex3D<float> (volIn[0], q.x,q.y,q.z+1) + distFunc(p, pi.x, pi.y, pi.z+1, radius);		surf3Dwrite ( w, volOut[0], q.x*sizeof(float), q.y, (q.z+1) );
	}

	if ( pclr != 0 ) {
		uchar4 wclr = *(uchar4*) (pclr + i*clr_stride + clr_off );

		if ( colorBuf != 0 ) {	
			// Increment index
			uint brickres = gvdb.res[0];
			uint vid = (brickres * brickres * brickres * pnode[i]) + (brickres * brickres * (uint)pi.z) + (brickres * (uint)pi.y) + (uint)pi.x;
			uint colorIdx = vid * 4;
		
			// Store in color in the colorbuf
			atomicAdd(&colorBuf[colorIdx + 0], 1);
			atomicAdd(&colorBuf[colorIdx + 1], wclr.x);
			atomicAdd(&colorBuf[colorIdx + 2], wclr.y);
			atomicAdd(&colorBuf[colorIdx + 3], wclr.z);
		}
		else {
		 	surf3Dwrite(wclr, volOut[1], q.x*sizeof(uchar4), q.y, q.z); 
		}
	}
}

extern "C" __global__ void gvdbAddSupportVoxel ( int num_pnts,  float radius, float offset, float amp, 
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
	VDBNode* node = getNode ( 0, pnode[i], &vmin );			// Get node	
	float3 p = (wpos-vmin)/gvdb.vdel[0];
	float3 pi = make_float3(int(p.x), int(p.y), int(p.z));

	// -- should be ok that pi.x,pi.y,pi.z = 0 
	if ( pi.x <= -1 || pi.y <= -1 || pi.z <= -1 || pi.x >= gvdb.res[0] || pi.y >= gvdb.res[0] || pi.z >= gvdb.res[0] ) return;
	uint3 q = make_uint3(pi.x,pi.y,pi.z) + make_uint3( node->mValue );	

	w = tex3D<float>( volIn[0], q.x, q.y, q.z ) + distFunc(p, pi.x, pi.y,pi.z, radius); 				
	surf3Dwrite ( w, volOut[0], q.x*sizeof(float), q.y, q.z );	
	surf3Dwrite ( (uchar)1, volOut[1], q.x*sizeof(uchar), q.y, q.z );
	//surf3Dwrite ( 1.0f, volOut[2], q.x*sizeof(float), q.y, q.z );	

#if 1
	// expand to 3x3 square, write to both volume and material channels
	w = tex3D<float> (volIn[0], q.x-1,q.y,q.z) + distFunc(p, pi.x-1, pi.y, pi.z, radius);		
	surf3Dwrite ( w, volOut[0], (q.x-1)*sizeof(float), q.y, q.z );
	surf3Dwrite ( (uchar)1, volOut[1], (q.x-1)*sizeof(uchar), q.y, q.z );
	//surf3Dwrite ( 1.0f, volOut[2], (q.x-1)*sizeof(float), q.y, q.z );

	w = tex3D<float> (volIn[0], q.x+1,q.y,q.z) + distFunc(p, pi.x+1, pi.y, pi.z, radius);		
	surf3Dwrite ( w, volOut[0], (q.x+1)*sizeof(float), q.y, q.z );
	surf3Dwrite ( (uchar)1, volOut[1], (q.x+1)*sizeof(uchar), q.y, q.z );
	//surf3Dwrite ( 1.0f, volOut[2], (q.x+1)*sizeof(float), q.y, q.z );

	w = tex3D<float> (volIn[0], q.x,q.y,q.z-1) + distFunc(p, pi.x, pi.y, pi.z-1, radius);		
	surf3Dwrite ( w, volOut[0], q.x*sizeof(float), q.y, (q.z-1) );
	surf3Dwrite ( (uchar)1, volOut[1], q.x*sizeof(uchar), q.y, (q.z-1) );
	//surf3Dwrite ( 1.0f, volOut[2], q.x*sizeof(float), q.y, (q.z-1) );

	w = tex3D<float> (volIn[0], q.x,q.y,q.z+1) + distFunc(p, pi.x, pi.y, pi.z+1, radius);		
	surf3Dwrite ( w, volOut[0], q.x*sizeof(float), q.y, (q.z+1) );
	surf3Dwrite ( (uchar)1, volOut[1], q.x*sizeof(uchar), q.y, (q.z+1) );
	//surf3Dwrite ( 1.0f, volOut[2], q.x*sizeof(float), q.y, (q.z+1) );

	w = tex3D<float> (volIn[0], q.x-1,q.y,q.z-1) + distFunc(p, pi.x-1, pi.y, pi.z-1, radius);		
	surf3Dwrite ( w, volOut[0], (q.x-1)*sizeof(float), q.y, (q.z-1) );
	surf3Dwrite ( (uchar)1, volOut[1], (q.x-1)*sizeof(uchar), q.y, (q.z-1) );
	//surf3Dwrite ( 1.0f, volOut[2], (q.x-1)*sizeof(float), q.y, (q.z-1) );
	w = tex3D<float> (volIn[0], q.x+1,q.y,q.z+1) + distFunc(p, pi.x+1, pi.y, pi.z+1, radius);		
	surf3Dwrite ( w, volOut[0], (q.x+1)*sizeof(float), q.y, (q.z+1) );
	surf3Dwrite ( (uchar)1, volOut[1], (q.x+1)*sizeof(uchar), q.y, (q.z+1) );
	//surf3Dwrite ( 1.0f, volOut[2], (q.x+1)*sizeof(float), q.y, (q.z+1) );
	w = tex3D<float> (volIn[0], q.x+1,q.y,q.z-1) + distFunc(p, pi.x+1, pi.y, pi.z-1, radius);		
	surf3Dwrite ( w, volOut[0], (q.x+1)*sizeof(float), q.y, (q.z-1) );
	surf3Dwrite ( (uchar)1, volOut[1], (q.x+1)*sizeof(uchar), q.y, (q.z-1) );
	//surf3Dwrite ( 1.0f, volOut[2], (q.x+1)*sizeof(float), q.y, (q.z-1) );
	w = tex3D<float> (volIn[0], q.x-1,q.y,q.z+1) + distFunc(p, pi.x-1, pi.y, pi.z+1, radius);		
	surf3Dwrite ( w, volOut[0], (q.x-1)*sizeof(float), q.y, (q.z+1) );
	surf3Dwrite ( (uchar)1, volOut[1], (q.x-1)*sizeof(uchar), q.y, (q.z+1) );
	//surf3Dwrite ( 1.0f, volOut[2], (q.x-1)*sizeof(float), q.y, (q.z+1) );
#endif
}

extern "C" __global__ void gvdbScatterPointAvgCol (int num_voxels, uint* colorBuf)
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
    uint brickres = gvdb.res[0];
    uint nid = vid / (brickres * brickres * brickres);
    float3 vmin;
    VDBNode* node = getNode(0, nid, &vmin);

    // Get local 3d indices
    uint3 pi;
    pi.x = vid % (brickres);
    pi.y = vid % (brickres * brickres) / (brickres);
    pi.z = vid % (brickres * brickres * brickres) / (brickres * brickres);
    
    // Get global atlas index
    uint3 q = make_uint3(pi.x, pi.y, pi.z) + make_uint3(node->mValue);
    
    surf3Dwrite(pclr, volOut[1], q.x*sizeof(uchar4), q.y, q.z);
  }
}

/*
extern "C" __global__ void gvdbGatherPointDensity ( int3 res, uchar chan, int num_pnts, float radius, float3* ppos,
												int num_node, int* gcnt, int* goff )
{
	uint3 bndx = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + threadIdx;
	if ( bndx.x >= gvdb.atlas_cnt.x || bndx.y >= gvdb.atlas_cnt.y || bndx.z >= gvdb.atlas_cnt.z ) return;
	int bid = (bndx.z*gvdb.atlas_cnt.y + bndx.y )*gvdb.atlas_cnt.x + bndx.x;		// brick id				
	int nid = (gvdb.atlas_map + bid)->mLeafID;
	if ( nid == ID_UNDEFL) return;
	if ( nid >= num_node ) return;
	if ( gcnt[nid] == 0 ) return;
	int3 o = (gvdb.atlas_map + bid)->mPos;		

	float3 poslist[200];			// <-- cannot used share memory here
	
	float3* pcurr = ppos + goff[nid];
	for (int j=0; j < gcnt[nid]; j++ ) {
		poslist[j] = *pcurr++;		
	}	
	__syncthreads ();

	float sum, c, R2 = radius*radius;
	float3 jpos, wpos;		
	int3 p;
	
	for ( p.z = 0; p.z < gvdb.brick_res; p.z++ ) 
	 for ( p.y = 0; p.y < gvdb.brick_res; p.y++ ) 
	  for ( p.x = 0; p.x < gvdb.brick_res; p.x++ ) {
		wpos = make_float3(p+o) * gvdb.voxelsize;
		sum = 0;
		for (int j=0; j < gcnt[nid]; j++ ) {
			jpos = poslist[j] - wpos;
			c = (jpos.x*jpos.x + jpos.y*jpos.y + jpos.z*jpos.z);		
			if ( c < R2) {		
				c = c / R2;
				sum += 1.0 + c*(-3 + c*(3 - c));		// Wyvill equation
			}
		}
		surf3Dwrite ( sum, volOut[chan], (p.x+bndx.x*gvdb.brick_res)*sizeof(float), (p.y+bndx.y*gvdb.brick_res), (p.z+bndx.z*gvdb.brick_res) );
	  }
	
}
*/

extern "C" __global__ void gvdbGatherPointDensity (  int3 res, uchar chan, int num_pnts, float radius, float3* ppos,
												int num_node, int* gcnt, int* goff, int subcell )
{
	uint3 packvox = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + threadIdx;	
	uint3 vox = packvox + (blockIdx / subcell)*gvdb.atlas_apron*2 + make_uint3(1,1,1);
	if ( vox.x >= gvdb.atlas_res.x || vox.y >= gvdb.atlas_res.y || vox.z >= gvdb.atlas_res.z ) return;

	// Get atlas node (brick)
	float3 wpos;
	int nid;
	if ( !getAtlasToWorldID ( vox, wpos, nid ) ) return;

	// Transfer brick points into shared memory	
	__shared__ float3 poslist[4000];	
	int tid = ((threadIdx.z*blockDim.y + threadIdx.y)*blockDim.x + threadIdx.x) * 4;
	for (int k=0; k < 4 && tid+k < gcnt[nid]; k++ ) {
		poslist[ tid+k ] = *(ppos + goff[nid] + tid+k);		
	}
	__syncthreads ();

	float R2 = radius*radius;
	float3 jpos;
	float c, sum = 0.0f;	
	for (int j=0; j < gcnt[nid]; j++ ) {		
		jpos = poslist[j] - wpos;		
		c = (jpos.x*jpos.x + jpos.y*jpos.y + jpos.z*jpos.z);		
		if ( c < R2) {		
			c = c / R2;
			sum += 1.0 + c*(-3 + c*(3 - c));		// Wyvill equation (Soft Objects)
		}
	}

	surf3Dwrite ( sum, volOut[chan], vox.x*sizeof(float), vox.y, vox.z );
}	


extern "C" __global__ void gvdbGatherPointVelocity (  int3 res, uchar chan, int num_pnts, float radius, float3* ppos,
												int num_node, int* gcnt, int* goff )
{
	uint3 packvox = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + threadIdx;	
	uint3 vox = packvox + (blockIdx / 2)*gvdb.atlas_apron*2 + make_uint3(1,1,1);
	if ( vox.x >= gvdb.atlas_res.x || vox.y >= gvdb.atlas_res.y || vox.z >= gvdb.atlas_res.z ) return;

	// Get atlas node (brick)
	float3 wpos;
	int nid;
	if ( !getAtlasToWorldID ( vox, wpos, nid ) ) return;

	// Transfer brick points into shared memory	
	__shared__ float3 poslist[4000];	
	int tid = ((threadIdx.z*blockDim.y + threadIdx.y)*blockDim.x + threadIdx.x) * 4;
	for (int k=0; k < 4 && tid+k < gcnt[nid]; k++ ) {
		poslist[ tid+k ] = *(ppos + goff[nid] + tid+k);		
	}
	__syncthreads ();

	float R2 = radius*radius;
	float3 jpos;
	float c, sum = 0.0f;	
	for (int j=0; j < gcnt[nid]; j++ ) {		
		jpos = poslist[j] - wpos;		
		c = (jpos.x*jpos.x + jpos.y*jpos.y + jpos.z*jpos.z);		
		if ( c < R2) {		
			c = c / R2;
			sum += 1.0 + c*(-3 + c*(3 - c));		// Wyvill equation (Soft Objects)
		}
	}

	surf3Dwrite ( sum, volOut[chan], vox.x*sizeof(float), vox.y, vox.z );
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

extern "C" __global__ void gvdbVoxelize ( float3 vmin, float3 vmax, int3 res, uchar* obuf, uchar val_surf, uchar val_inside, 
							   float bdiv, int bmax, int* bcnt, int* boff, float3* tbuf )					
							// int vcnt, int ecnt, float3* vbuf, int* ebuf )
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
		
		obuf[ (t.z*res.y + t.y)*res.x + t.x ] = val_surf;
		break;
	}

	if ( n == boff[b]+bcnt[b] ) {
		// solid voxelization		
		if ( cnt % 2 == 1)
			obuf[ (t.z*res.y + t.y)*res.x + t.x ] = val_inside;
	}
}


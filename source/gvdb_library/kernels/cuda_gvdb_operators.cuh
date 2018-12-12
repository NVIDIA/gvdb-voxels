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
// Version 1.1: Rama Hoetzlein, 3/25/2018
//----------------------------------------------------------------------------------
// File: cuda_gvdb_operators.cu
//
// GVDB Operators
// - CopyApron		- update apron voxels
// - OpSmooth		- smooth operation
// - OpNoise		- noise operation
//-----------------------------------------------

#define CMAX(x,y,z)  (imax3(x,y,z)-imin3(x,y,z))

#define RGBA2INT(r,g,b,a)	( (uint((a)*255.0f)<<24) | (uint((b)*255.0f)<<16) | (uint((g)*255.0f)<<8) | uint((r)*255.0f) )
#define CLR2INT(c)			( (uint((c.w)*255.0f)<<24) | (uint((c.z)*255.0f)<<16) | (uint((c.y)*255.0f)<<8) | uint((c.x)*255.0f) )
#define INT2CLR(c)			( make_float4( float(c & 0xFF)/255.0f, float((c>>8) & 0xFF)/255.0f, float((c>>16) & 0xFF)/255.0f, float((c>>24) & 0xFF)/255.0f ))

#define T_UCHAR			0		// channel types
#define	T_FLOAT			3
#define T_INT			6

extern "C" __global__ void gvdbUpdateApronFacesF ( VDBInfo* gvdb, uchar chan, int brickcnt, int brickres, int brickwid, int* nbrtable )
{
	// Compute brick & atlas vox	
	int side = threadIdx.z;
	uint3 suv = blockIdx * make_uint3(blockDim.x, blockDim.y, 1) + threadIdx;
	if (suv.x >= brickwid || suv.y >= brickwid || side >= 3 ) return;
	int3 vox, vinc;
	switch (side) {
	case 0:		vox = make_int3(0, suv.x, suv.y);		vinc = make_int3(1, 0, 0); break;
	case 1:		vox = make_int3(suv.x, 0, suv.y);		vinc = make_int3(0, 1, 0); break;
	case 2:		vox = make_int3(suv.x, suv.y, 0);		vinc = make_int3(0, 0, 1); break;
	}
	int3 vnbr = vox + vinc*(brickwid - 1);							// neighbor offset
	int brk = blockIdx.z;
	if (brk > brickcnt) return;

	// Get current brick
	VDBNode* node = getNode(gvdb, 0, brk);	
	if (node == 0x0) return;
	vox += make_int3(node->mValue);									// self atlas start
	
	// Get neigboring brick
	int nbr = nbrtable[brk * 6 + side];
	if (nbr == ID_UNDEFL) return;	
	node = getNode(gvdb, 0, nbr);
	vnbr += make_int3(node->mValue);								// neighbor atlas start

	// Update self and neighbor
	float v1 = tex3D<float>(gvdb->volIn[chan], vox.x + 0.5f, vox.y + 0.5f, vox.z + 0.5f);		// get self voxel
	float v2 = tex3D<float>(gvdb->volIn[chan], vnbr.x + 0.5f, vnbr.y + 0.5f, vnbr.z + 0.5f);	// get neighbor voxel
	
	surf3Dwrite(v1, gvdb->volOut[chan], (vnbr.x + vinc.x) * sizeof(float), (vnbr.y + vinc.y), (vnbr.z + vinc.z) );   // neighbor apron
	surf3Dwrite(v2, gvdb->volOut[chan], (vox.x - vinc.x) * sizeof(float), (vox.y - vinc.y), (vox.z - vinc.z));		// self apron
}


extern "C" __global__ void gvdbUpdateApronF (VDBInfo* gvdb, uchar chan, int brickcnt, int brickres, int brickwid, float boundval)
{
	// Compute brick & atlas vox	
	int side = threadIdx.z;
	uint3 suv = blockIdx * make_uint3(blockDim.x, blockDim.y, 1) + threadIdx;
	if (suv.x >= brickres || suv.y >= brickres || side >= 6) return;
	uint3 vox;
	switch (side) {
	case 0:		vox = make_uint3(0, suv.x, suv.y);			break;
	case 1:		vox = make_uint3(suv.x, 0, suv.y);			break;
	case 2:		vox = make_uint3(suv.x, suv.y, 0);			break;
	case 3:		vox = make_uint3(brickres-1, suv.x, suv.y);	break;
	case 4:		vox = make_uint3(suv.x, brickres-1, suv.y);	break;
	case 5:		vox = make_uint3(suv.x, suv.y, brickres-1);	break;	
	}	
	int brk = blockIdx.z;
	if (brk > brickcnt) return;

	// Get current brick
	VDBNode* node = getNode(gvdb, 0, brk);
	if (node == 0x0) return;
	vox += make_uint3(node->mValue) - make_uint3(1,1,1);							// self atlas start

	// Get apron value
	float3 wpos;
	if (!getAtlasToWorld(gvdb, vox, wpos)) return;
	float3 offs, vmin, vdel; uint64 nid;
	node = getNodeAtPoint(gvdb, wpos, &offs, &vmin, &vdel, &nid);		// Evaluate at world position
	offs += (wpos - vmin) / vdel;

	float v = (node == 0x0) ? boundval : tex3D<float>(gvdb->volIn[chan], offs.x, offs.y, offs.z);	// Sample at world point
	surf3Dwrite(v, gvdb->volOut[chan], vox.x * sizeof(float), vox.y, vox.z);					// Write to apron voxel
}


// Recreate the compressed axis
/*uint3 vox = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + threadIdx;
switch ( axis ) {
case 0:		vox.x = blockIdx.x * blkres + gvdb->apron_table[threadIdx.x];	break;
case 1:		vox.y = blockIdx.y * blkres + gvdb->apron_table[threadIdx.y];	break;
case 2:		vox.z = blockIdx.z * blkres + gvdb->apron_table[threadIdx.z];	break;
};
if ( vox.x >= res.x || vox.y >= res.y || vox.z >= res.z ) return;
float3 wpos;
if ( !getAtlasToWorld ( gvdb, vox, wpos )) return;

float3 offs, vmin, vdel; uint64 nid;
VDBNode* node = getNodeAtPoint ( gvdb, wpos, &offs, &vmin, &vdel, &nid );		// Evaluate at world position
offs += (wpos-vmin)/vdel;

float v = (node==0x0) ? 0.0 : tex3D<float> ( gvdb->volIn[chan], offs.x, offs.y, offs.z );		// Sample at world point

surf3Dwrite ( v, gvdb->volOut[chan], vox.x*sizeof(float), vox.y, vox.z );	// Write to apron voxel*/


extern "C" __global__ void gvdbUpdateApronF4 ( VDBInfo* gvdb, uchar chan, int brickcnt, int brickres, int brickwid, float boundval)
{
	// Compute brick & atlas vox	
	int side = threadIdx.z;
	uint3 suv = blockIdx * make_uint3(blockDim.x, blockDim.y, 1) + threadIdx;
	if (suv.x >= brickres || suv.y >= brickres || side >= 6) return;
	uint3 vox;
	switch (side) {
	case 0:		vox = make_uint3(0, suv.x, suv.y);			break;
	case 1:		vox = make_uint3(suv.x, 0, suv.y);			break;
	case 2:		vox = make_uint3(suv.x, suv.y, 0);			break;
	case 3:		vox = make_uint3(brickres-1, suv.x, suv.y);	break;
	case 4:		vox = make_uint3(suv.x, brickres-1, suv.y);	break;
	case 5:		vox = make_uint3(suv.x, suv.y, brickres-1);	break;	
	}	
	int brk = blockIdx.z;
	if (brk > brickcnt) return;

	// Get current brick
	VDBNode* node = getNode(gvdb, 0, brk);
	if (node == 0x0) return;
	vox += make_uint3(node->mValue) - make_uint3(1,1,1);							// self atlas start

	// Get apron value
	float3 wpos;
	if (!getAtlasToWorld(gvdb, vox, wpos)) return;
	float3 offs, vmin, vdel; uint64 nid;
	node = getNodeAtPoint(gvdb, wpos, &offs, &vmin, &vdel, &nid);		// Evaluate at world position
	offs += (wpos - vmin) / vdel;
	
	float4 v = (node == 0x0) ? make_float4(boundval,boundval,boundval,boundval) : tex3D<float4>(gvdb->volIn[chan], offs.x, offs.y, offs.z);	// Sample at world point
	surf3Dwrite(v, gvdb->volOut[chan], vox.x * sizeof(float4), vox.y, vox.z);					// Write to apron voxel
}

extern "C" __global__ void gvdbUpdateApronC ( VDBInfo* gvdb, uchar chan, int brickcnt, int brickres, int brickwid, float boundval)
{
	// Compute brick & atlas vox	
	int side = threadIdx.z;
	uint3 suv = blockIdx * make_uint3(blockDim.x, blockDim.y, 1) + threadIdx;
	if (suv.x >= brickres || suv.y >= brickres || side >= 6) return;
	uint3 vox;
	switch (side) {
	case 0:		vox = make_uint3(0, suv.x, suv.y);			break;
	case 1:		vox = make_uint3(suv.x, 0, suv.y);			break;
	case 2:		vox = make_uint3(suv.x, suv.y, 0);			break;
	case 3:		vox = make_uint3(brickres-1, suv.x, suv.y);	break;
	case 4:		vox = make_uint3(suv.x, brickres-1, suv.y);	break;
	case 5:		vox = make_uint3(suv.x, suv.y, brickres-1);	break;	
	}	
	int brk = blockIdx.z;
	if (brk > brickcnt) return;

	// Get current brick
	VDBNode* node = getNode(gvdb, 0, brk);
	if (node == 0x0) return;
	vox += make_uint3(node->mValue) - make_uint3(1,1,1);							// self atlas start

	// Get apron value
	float3 wpos;
	if (!getAtlasToWorld(gvdb, vox, wpos)) return;
	float3 offs, vmin, vdel; uint64 nid;
	node = getNodeAtPoint(gvdb, wpos, &offs, &vmin, &vdel, &nid);		// Evaluate at world position
	offs += (wpos - vmin) / vdel;
	
	uchar v = (node == 0x0) ? boundval : tex3D<uchar>(gvdb->volIn[chan], offs.x, offs.y, offs.z);	// Sample at world point
	surf3Dwrite(v, gvdb->volOut[chan], vox.x * sizeof(uchar), vox.y, vox.z);					// Write to apron voxel
}

extern "C" __global__ void gvdbUpdateApronC4 ( VDBInfo* gvdb, uchar chan, int brickcnt, int brickres, int brickwid, float boundval)
{
	// Compute brick & atlas vox	
	int side = threadIdx.z;
	uint3 suv = blockIdx * make_uint3(blockDim.x, blockDim.y, 1) + threadIdx;
	if (suv.x >= brickres || suv.y >= brickres || side >= 6) return;
	uint3 vox;
	switch (side) {
	case 0:		vox = make_uint3(0, suv.x, suv.y);			break;
	case 1:		vox = make_uint3(suv.x, 0, suv.y);			break;
	case 2:		vox = make_uint3(suv.x, suv.y, 0);			break;
	case 3:		vox = make_uint3(brickres-1, suv.x, suv.y);	break;
	case 4:		vox = make_uint3(suv.x, brickres-1, suv.y);	break;
	case 5:		vox = make_uint3(suv.x, suv.y, brickres-1);	break;	
	}	
	int brk = blockIdx.z;
	if (brk > brickcnt) return;

	// Get current brick
	VDBNode* node = getNode(gvdb, 0, brk);
	if (node == 0x0) return;
	vox += make_uint3(node->mValue) - make_uint3(1,1,1);							// self atlas start

	// Get apron value
	float3 wpos;
	if (!getAtlasToWorld(gvdb, vox, wpos)) return;
	float3 offs, vmin, vdel; uint64 nid;
	node = getNodeAtPoint(gvdb, wpos, &offs, &vmin, &vdel, &nid);		// Evaluate at world position
	offs += (wpos - vmin) / vdel;
	
	uchar4 v = (node == 0x0) ? make_uchar4(boundval,boundval,boundval,boundval) : tex3D<uchar4>(gvdb->volIn[chan], offs.x, offs.y, offs.z);	// Sample at world point
	surf3Dwrite(v, gvdb->volOut[chan], vox.x * sizeof(uchar4), vox.y, vox.z);					// Write to apron voxel
}

#define GVDB_COPY_SMEM_F																	\
	float3 vox;																				\
	uint3 ndx;																				\
	__shared__ float  svox[10][10][10]; 													\
	ndx = threadIdx + make_uint3(1,1,1);													\
	vox = make_float3(blockIdx) * make_float3(blockDim.x, blockDim.y, blockDim.z) + make_float3(ndx) + make_float3(0.5f,0.5f,0.5f); \
	if ( vox.x >= res.x || vox.y >= res.y || vox.z >= res.z ) return;						\
	svox[ndx.x][ndx.y][ndx.z] = tex3D<float> ( gvdb->volIn[chan], vox.x, vox.y, vox.z );	\
	if ( ndx.x==1 ) {																		\
		svox[0][ndx.y][ndx.z] = tex3D<float> ( gvdb->volIn[chan], vox.x-1, vox.y, vox.z );	\
		svox[9][ndx.y][ndx.z] = tex3D<float> ( gvdb->volIn[chan], vox.x+8, vox.y, vox.z );	\
	}																						\
	if ( ndx.y==1 ) {																		\
		svox[ndx.x][0][ndx.z] = tex3D<float> ( gvdb->volIn[chan], vox.x, vox.y-1, vox.z );	\
		svox[ndx.x][9][ndx.z] = tex3D<float> ( gvdb->volIn[chan], vox.x, vox.y+8, vox.z );	\
	}																						\
	if ( ndx.z==1 ) {																		\
		svox[ndx.x][ndx.y][0] = tex3D<float> ( gvdb->volIn[chan], vox.x, vox.y, vox.z-1 );	\
		svox[ndx.x][ndx.y][9] = tex3D<float> ( gvdb->volIn[chan], vox.x, vox.y, vox.z+8 );	\
	}																						\
	vox -= make_float3(0.5f,0.5f,0.5f);														\
	__syncthreads ();

#define GVDB_COPY_SMEM_UC4																	\
	uint3 vox, ndx;																			\
	__shared__ uchar4 svox[10][10][10]; 													\
	ndx = threadIdx + make_uint3(1,1,1);													\
	vox = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + ndx;					\
	if ( vox.x >= res.x || vox.y >= res.y || vox.z >= res.z ) return;						\
	svox[ndx.x][ndx.y][ndx.z] = tex3D<uchar4> ( gvdb->volIn[chan], vox.x, vox.y, vox.z );	\
	if ( ndx.x==1 ) {																		\
		svox[0][ndx.y][ndx.z] = tex3D<uchar4> ( gvdb->volIn[chan], vox.x-1, vox.y, vox.z );	\
		svox[9][ndx.y][ndx.z] = tex3D<uchar4> ( gvdb->volIn[chan], vox.x+8, vox.y, vox.z );	\
	}																						\
	if ( ndx.y==1 ) {																		\
		svox[ndx.x][0][ndx.z] = tex3D<uchar4> ( gvdb->volIn[chan], vox.x, vox.y-1, vox.z );	\
		svox[ndx.x][9][ndx.z] = tex3D<uchar4> ( gvdb->volIn[chan], vox.x, vox.y+8, vox.z );	\
	}																						\
	if ( ndx.z==1 ) {																		\
		svox[ndx.x][ndx.y][0] = tex3D<uchar4> ( gvdb->volIn[chan], vox.x, vox.y, vox.z-1 );	\
		svox[ndx.x][ndx.y][9]  = tex3D<uchar4> ( gvdb->volIn[chan], vox.x, vox.y, vox.z+1 );\
	}																						\
	__syncthreads ();

#define GVDB_COPY_SMEM_UC																	\
	uint3 vox, ndx;																			\
	__shared__ uchar svox[10][10][10]; 														\
	ndx = threadIdx + make_uint3(1,1,1);													\
	vox = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + ndx;					\
	if ( vox.x >= res.x || vox.y >= res.y || vox.z >= res.z ) return;						\
	svox[ndx.x][ndx.y][ndx.z] = tex3D<uchar> ( gvdb->volIn[chan], vox.x, vox.y, vox.z );	\
	if ( ndx.x==1 ) {																		\
		svox[0][ndx.y][ndx.z] = tex3D<uchar> ( gvdb->volIn[chan], vox.x-1, vox.y, vox.z );	\
		svox[9][ndx.y][ndx.z] = tex3D<uchar> ( gvdb->volIn[chan], vox.x+8, vox.y, vox.z );	\
	}																						\
	if ( ndx.y==1 ) {																		\
		svox[ndx.x][0][ndx.z] = tex3D<uchar> ( gvdb->volIn[chan], vox.x, vox.y-1, vox.z );	\
		svox[ndx.x][9][ndx.z] = tex3D<uchar> ( gvdb->volIn[chan], vox.x, vox.y+8, vox.z );	\
	}																						\
	if ( ndx.z==1 ) {																		\
		svox[ndx.x][ndx.y][0] = tex3D<uchar> ( gvdb->volIn[chan], vox.x, vox.y, vox.z-1 );	\
		svox[ndx.x][ndx.y][9]  = tex3D<uchar> ( gvdb->volIn[chan], vox.x, vox.y, vox.z+1 );	\
	}																						\
	__syncthreads ();

#define GVDB_VOX																								\
	uint3 vox = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + threadIdx + make_uint3(1,1,1);		\
	if ( vox.x >= res.x|| vox.y >= res.y || vox.z >= res.z ) return;


extern "C" __global__ void gvdbOpGrow ( VDBInfo* gvdb, int3 res, uchar chan, float p1, float p2, float p3 )
{
	GVDB_COPY_SMEM_F
	
	/*float nl;	
	float3 n;
	n.x = 0.5 * (svox[ndx.x-1][ndx.y][ndx.z] - svox[ndx.x+1][ndx.y][ndx.z]);
	n.y = 0.5 * (svox[ndx.x][ndx.y-1][ndx.z] - svox[ndx.x][ndx.y+1][ndx.z]);
	n.z = 0.5 * (svox[ndx.x][ndx.y][ndx.z-1] - svox[ndx.x][ndx.y][ndx.z+1]);
	nl = sqrt(n.x*n.x+n.y*n.y+n.z*n.z);	
	float v = svox[ndx.x][ndx.y][ndx.z];
	if ( nl > 0.2 ) v += amt;*/ 

	float v = svox[ndx.x][ndx.y][ndx.z];
	if ( v != 0.0) v += p1 * 10.0; //*0.1;
	if ( v < 0.01) v = 0.0;
	
	surf3Dwrite ( v, gvdb->volOut[chan], vox.x*sizeof(float), vox.y, vox.z );		
}

extern "C" __global__ void gvdbOpCut ( VDBInfo* gvdb, int3 res, uchar chan, float p1, float p2, float p3 )
{
	GVDB_COPY_SMEM_F			

	// Determine block and index position	
	float3 wpos;
	if ( !getAtlasToWorld ( gvdb, make_uint3(vox), wpos )) return;

	float v = svox[ndx.x][ndx.y][ndx.z];
	if ( wpos.x < 50 && v > 0 && v < 2 ) {
		v = 0.02;
		surf3Dwrite ( v, gvdb->volOut[chan], vox.x*sizeof(float), vox.y, vox.z );		
	}	
}

//-- smooth
	/*float v = 6.0*svox[ndx.x][ndx.y][ndx.z];
	v += svox[ndx.x-1][ndx.y][ndx.z];
	v += svox[ndx.x+1][ndx.y][ndx.z];
	v += svox[ndx.x][ndx.y-1][ndx.z];
	v += svox[ndx.x][ndx.y+1][ndx.z];
	v += svox[ndx.x][ndx.y][ndx.z-1];
	v += svox[ndx.x][ndx.y][ndx.z+1];
	v /= 12.0;*/

extern "C" __global__ void gvdbReduction( VDBInfo* gvdb, int3 res, uchar chan, int3 packres, float* outbuf)
{
	// voxels - not including apron
	uint3 packvox = blockIdx * make_uint3(blockDim.x, blockDim.y, 1) + threadIdx;
	int bres = (gvdb->brick_res - gvdb->atlas_apron * 2);	// brick res minus apron
	uint3 brick = packvox / bres;
	uint3 vox = (brick*gvdb->brick_res) + (packvox % bres) + make_uint3(1, 1, 1);
	if (vox.x >= res.x || vox.y >= res.y) return;

	// integrate along z-axis
	float sum = 0.0;
	float v;
	for (int z = 0; z < packres.z; z++) {
		vox.z = (int(z / bres)*gvdb->brick_res) + (z % bres) + 1;
		v = tex3D<float> ( gvdb->volIn[chan], vox.x, vox.y, vox.z);
		sum += v;   // (v > gvdb->thresh.x) ? 1.0 : 0.0;
	}

	outbuf[packvox.y * packres.x + packvox.x] = sum;
}


extern "C" __global__ void gvdbResample ( VDBInfo* gvdb, int3 res, uchar chan, int3 srcres, float* src, float* xform, float3 inr, float3 outr )
{
	GVDB_VOX

	float3 wpos;
	if ( !getAtlasToWorld ( gvdb, vox, wpos )) return;
	wpos -= make_float3(.5, .5, .5);
	
	// transform to src index
	int3 ndx;
	ndx.x = (int) (wpos.x * xform[0] + wpos.y * xform[4] + wpos.z * xform[8] + xform[12]);
	ndx.y = (int) (wpos.x * xform[1] + wpos.y * xform[5] + wpos.z * xform[9] + xform[13]);
	ndx.z = (int) (wpos.x * xform[2] + wpos.y * xform[6] + wpos.z * xform[10] + xform[14]);

	// skip if outside src
	if ( ndx.x < 0 || ndx.y < 0 || ndx.z < 0 || ndx.x >= srcres.x || ndx.y >= srcres.y || ndx.z >= srcres.z )
		return;
	
	// get value
	float v = src[ (ndx.z*srcres.y + ndx.y)*srcres.x + ndx.x ];
	v = outr.x + (v-inr.x)*(outr.y-outr.x)/(inr.y-inr.x);    // remap value

	surf3Dwrite ( v, gvdb->volOut[chan], vox.x*sizeof(float), vox.y, vox.z );
}


extern "C" __global__ void gvdbDownsample(int3 srcres, float* src, int3 destres, float3 destmax, float* dest, float* xform, float3 inr, float3 outr)
{
	uint3 vox = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + threadIdx;
	if (vox.x >= destres.x || vox.y >= destres.y || vox.z >= destres.z) return;

	float3 dmin, dmax;
	dmin = make_float3(vox.x, vox.y, vox.z) * destmax / make_float3(destres.x + 1, destres.y + 1, destres.z + 1);
	dmax = make_float3(vox.x + 1, vox.y + 1, vox.z + 1) * destmax / make_float3(destres.x + 1, destres.y + 1, destres.z + 1) - make_float3(1, 1, 1);

	// transform to src index
	int3 smin, smax;
	smin.x = (int)(dmin.x * xform[0] + dmin.y * xform[4] + dmin.z * xform[8] + xform[12]);
	smin.y = (int)(dmin.x * xform[1] + dmin.y * xform[5] + dmin.z * xform[9] + xform[13]);
	smin.z = (int)(dmin.x * xform[2] + dmin.y * xform[6] + dmin.z * xform[10] + xform[14]);
	
	smax.x = (int)(dmax.x * xform[0] + dmax.y * xform[4] + dmax.z * xform[8] + xform[12]);
	smax.y = (int)(dmax.x * xform[1] + dmax.y * xform[5] + dmax.z * xform[9] + xform[13]);
	smax.z = (int)(dmax.x * xform[2] + dmax.y * xform[6] + dmax.z * xform[10] + xform[14]);

	smin.x = (smin.x < 0) ? 0 : ((smin.x > srcres.x - 1) ? srcres.x - 1 : smin.x);
	smin.y = (smin.y < 0) ? 0 : ((smin.y > srcres.y - 1) ? srcres.y - 1 : smin.y);
	smin.z = (smin.z < 0) ? 0 : ((smin.z > srcres.z - 1) ? srcres.z - 1 : smin.z);
	
	smax.x = (smax.x < smin.x) ? smin.x : ((smax.x > srcres.x - 1) ? srcres.x - 1 : smax.x);
	smax.y = (smax.y < smin.y) ? smin.y : ((smax.y > srcres.y - 1) ? srcres.y - 1 : smax.y);
	smax.z = (smax.z < smin.z) ? smin.z : ((smax.z > srcres.z - 1) ? srcres.z - 1 : smax.z);

	// downsample
	float v = 0;
	for (int z = smin.z; z <= smax.z; z++)
		for (int y = smin.y; y <= smax.y; y++)
			for (int x = smin.x; x <= smax.x; x++) {
				v += outr.x + (src[ (z*srcres.y + y)*srcres.x + x ] - inr.x)*(outr.y - outr.x) / (inr.y - inr.x);
			}

	v /= (smax.x - smin.x + 1)*(smax.y - smin.y + 1)*(smax.z - smin.z + 1);

	// output value
	dest[(vox.z*destres.y + vox.y)*destres.x + vox.x] = v;
}

extern "C" __global__ void gvdbOpFillF  ( VDBInfo* gvdb, int3 res, uchar chan, float p1, float p2, float p3 )
{
	GVDB_VOX	

	if ( p3 < 0 ) {
		//float v = vox.y; // + (vox.z*30 + vox.x)/900.0;
		float v = sinf( vox.x*12/(3.141592*30.0) );
		v += sinf( vox.y*12/(3.141592*30.0) );
		v += sinf( vox.z*12/(3.141592*30.0) );
		surf3Dwrite ( v, gvdb->volOut[chan], vox.x*sizeof(float), vox.y, vox.z );
	} else {
		surf3Dwrite ( p1, gvdb->volOut[chan], vox.x*sizeof(float), vox.y, vox.z );
	}
}
extern "C" __global__ void gvdbOpFillC4 ( VDBInfo* gvdb, int3 res, uchar chan, float p1, float p2, float p3 )
{
	GVDB_VOX

	surf3Dwrite ( CLR2INT(make_float4(p1,p2,p3,1.0)), gvdb->volOut[chan], vox.x*sizeof(uchar4), vox.y, vox.z );
}
extern "C" __global__ void gvdbOpFillC ( VDBInfo* gvdb, int3 res, uchar chan, float p1, float p2, float p3 )
{
	GVDB_VOX	

	uchar c = p1;
	surf3Dwrite ( c, gvdb->volOut[chan], vox.x*sizeof(uchar), vox.y, vox.z );
}

extern "C" __global__ void gvdbOpSmooth ( VDBInfo* gvdb, int3 res, uchar chan, float p1, float p2, float p3 )
{
	GVDB_COPY_SMEM_F

	//-- smooth
	float v = p1 * svox[ndx.x][ndx.y][ndx.z];
	v += svox[ndx.x-1][ndx.y][ndx.z];
	v += svox[ndx.x+1][ndx.y][ndx.z];
	v += svox[ndx.x][ndx.y-1][ndx.z];
	v += svox[ndx.x][ndx.y+1][ndx.z];
	v += svox[ndx.x][ndx.y][ndx.z-1];
	v += svox[ndx.x][ndx.y][ndx.z+1];
	v = v / (p1 + 6.0) + p2;

	surf3Dwrite ( v, gvdb->volOut[chan], vox.x*sizeof(float), vox.y, vox.z );
}

extern "C" __global__ void gvdbOpClrExpand ( VDBInfo* gvdb, int3 res, uchar chan, float p1, float p2, float p3 )
{
	GVDB_COPY_SMEM_UC4

	int3 c, cs;
	int cp;
	c = make_int3( svox[ndx.x][ndx.y][ndx.z] );   cs =  c*p1;
	c = make_int3( svox[ndx.x-1][ndx.y][ndx.z] ); cs += c*p2;	
	c = make_int3( svox[ndx.x+1][ndx.y][ndx.z] ); cs += c*p2;
	c = make_int3( svox[ndx.x][ndx.y-1][ndx.z] ); cs += c*p2;
	c = make_int3( svox[ndx.x][ndx.y+1][ndx.z] ); cs += c*p2;
	c = make_int3( svox[ndx.x][ndx.y][ndx.z-1] ); cs += c*p2;
	c = make_int3( svox[ndx.x][ndx.y][ndx.z+1] ); cs += c*p2;
	cp = max(cs.x, max(cs.y,cs.z) );
	cs = (cp > 255) ? make_int3(cs.x*255/cp, cs.y*255/cp, cs.z*255/cp) : cs;

	surf3Dwrite ( make_uchar4(cs.x, cs.y, cs.z, 1), gvdb->volOut[chan], vox.x*sizeof(uchar4), vox.y, vox.z );
}

extern "C" __global__ void gvdbOpExpandC ( VDBInfo* gvdb, int3 res, uchar chan, float p1, float p2, float p3 )
{
	GVDB_COPY_SMEM_UC

	uchar c = 0;
	c = (svox[ndx.x-1][ndx.y][ndx.z] == (uchar) p1 ) ? 1 : c;
	c = (svox[ndx.x+1][ndx.y][ndx.z] == (uchar) p1 ) ? 1 : c;
	c = (svox[ndx.x][ndx.y-1][ndx.z] == (uchar) p1 ) ? 1 : c;
	c = (svox[ndx.x][ndx.y+1][ndx.z] == (uchar) p1 ) ? 1 : c;
	c = (svox[ndx.x][ndx.y][ndx.z-1] == (uchar) p1 ) ? 1 : c;
	c = (svox[ndx.x][ndx.y][ndx.z+1] == (uchar) p1 ) ? 1 : c;
	
	uchar v = svox[ndx.x][ndx.y][ndx.z];
	if ( v == 0 && c == 1 ) {
		c = p2;
		surf3Dwrite ( c, gvdb->volOut[chan], vox.x*sizeof(uchar), vox.y, vox.z );
	}
}


extern "C" __global__ void gvdbOpNoise ( VDBInfo* gvdb, int3 res, uchar chan, float p1, float p2, float p3 )
{
	GVDB_COPY_SMEM_F

	//-- noise
	float v = svox[ndx.x][ndx.y][ndx.z];
	if ( v > 0.01 ) v += random(make_float3(vox.x,vox.y,vox.z)) * p1;

	surf3Dwrite ( v, gvdb->volOut[chan], vox.x*sizeof(float), vox.y, vox.z );

}


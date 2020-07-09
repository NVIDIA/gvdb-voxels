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
	int brk = blockIdx.x;
	if (brk > brickcnt) return;

	int side = threadIdx.x;
	uint2 suv = make_uint2(blockIdx.y*blockDim.y + threadIdx.y, blockIdx.z*blockDim.z + threadIdx.z);
	if (suv.x >= brickwid || suv.y >= brickwid || side >= 3 ) return;
	int3 vox, vinc;
	switch (side) {
	case 0:		vox = make_int3(0, suv.x, suv.y);		vinc = make_int3(1, 0, 0); break;
	case 1:		vox = make_int3(suv.x, 0, suv.y);		vinc = make_int3(0, 1, 0); break;
	case 2:		vox = make_int3(suv.x, suv.y, 0);		vinc = make_int3(0, 0, 1); break;
	}
	int3 vnbr = vox + vinc*(brickwid - 1);							// neighbor offset

	// Get current brick
	VDBNode* node = getNode(gvdb, 0, brk);	
	if (node == 0x0) return;
	vox += make_int3(node->mValue);									// self atlas start
	
	// Get neigboring brick
	int nbr = nbrtable[brk * 6 + side];
	if (nbr == ID_UNDEFL) return;	
	node = getNode(gvdb, 0, nbr);
	vnbr += make_int3(node->mValue);								// neighbor atlas start

	// Update self and neighbor (gvdb->volOut[chan] is the surface object for the atlas)
	float v1 = surf3Dread<float>(gvdb->volOut[chan], vox.x * sizeof(float), vox.y, vox.z);    // get self voxel
	float v2 = surf3Dread<float>(gvdb->volOut[chan], vnbr.x * sizeof(float), vnbr.y, vnbr.z); // get neighbor voxel
	
	surf3Dwrite(v1, gvdb->volOut[chan], (vnbr.x + vinc.x) * sizeof(float), (vnbr.y + vinc.y), (vnbr.z + vinc.z) );   // neighbor apron
	surf3Dwrite(v2, gvdb->volOut[chan], (vox.x - vinc.x) * sizeof(float), (vox.y - vinc.y), (vox.z - vinc.z));		// self apron
}

// Function template for updating the apron without UpdateApronFaces' neighbor
// table. Each of the voxels of the apron computes its world-space
// (= index-space) position and looks up what its value should be, sampling
// from a neighboring brick, or using the boundary value if no brick contains
// the voxel.
// 
// This should be called using blocks with an x dimension of 6, arranged in a
// grid along the x axis. Each yz plane of the block will fill in the voxels
// for a different face.
template<class T>
__device__ void UpdateApron(VDBInfo* gvdb, const uchar channel, const int brickCount, const int paddedBrickRes, const T boundaryValue)
{
	// The brick this block processes.
	const int brick = blockIdx.x;
	if (brick > brickCount) return;

	// Determine which voxel of the apron to compute and write.
	uint3 brickVoxel; // In the local coordinate space of the brick
	{
		const int side = threadIdx.x; // Side of the brick, from 0 to 5
		uint2 suv = make_uint2(blockIdx.y*blockDim.y + threadIdx.y, blockIdx.z*blockDim.z + threadIdx.z); // Position on the side of the brick
		if (suv.x >= paddedBrickRes || suv.y >= paddedBrickRes || side >= 6) return;

		switch (side) {
		case 0:		brickVoxel = make_uint3(0, suv.x, suv.y);			break;
		case 1:		brickVoxel = make_uint3(suv.x, 0, suv.y);			break;
		case 2:		brickVoxel = make_uint3(suv.x, suv.y, 0);			break;
		case 3:		brickVoxel = make_uint3(paddedBrickRes - 1, suv.x, suv.y);	break;
		case 4:		brickVoxel = make_uint3(suv.x, paddedBrickRes - 1, suv.y);	break;
		case 5:		brickVoxel = make_uint3(suv.x, suv.y, paddedBrickRes - 1);	break;
		}
	}

	// Compute the position of the voxel in the atlas
	uint3 atlasVoxel; // In the coordinate space of the entire atlas
	{
		VDBNode* node = getNode(gvdb, 0, brick);
		if (node == 0x0) return; // This brick ID didn't correspond to a known brick, which is invalid
		// (The (1,1,1) here accounts for the 1 unit of apron padding)
		atlasVoxel = brickVoxel + make_uint3(node->mValue) - make_uint3(1, 1, 1);
	}

	// Get the value of the voxel by converting to index-space and then
	// sampling the value at that index
	T value;
	{
		float3 worldPos;
		if (!getAtlasToWorld(gvdb, atlasVoxel, worldPos)) return;
		float3 offs, vmin; uint64 nodeID;
		VDBNode* node = getNodeAtPoint(gvdb, worldPos, &offs, &vmin, &nodeID);

		if (node == 0x0) {
			// Out of range, use the boundary value
			value = boundaryValue;
		}
		else {
			offs += (worldPos - vmin); // Get the atlas position
			value = surf3Dread<T>(gvdb->volOut[channel], uint(offs.x) * sizeof(T), uint(offs.y), uint(offs.z));
		}
	}

	// Write to the apron voxel
	surf3Dwrite(value, gvdb->volOut[channel], atlasVoxel.x * sizeof(T), atlasVoxel.y, atlasVoxel.z);
}

extern "C" __global__ void gvdbUpdateApronF (VDBInfo* gvdb, uchar chan, int brickcnt, int brickres, int brickwid, float boundval)
{
	UpdateApron<float>(gvdb, chan, brickcnt, brickres, boundval);
}

extern "C" __global__ void gvdbUpdateApronF4 ( VDBInfo* gvdb, uchar chan, int brickcnt, int brickres, int brickwid, float boundval)
{
	UpdateApron<float4>(gvdb, chan, brickcnt, brickres, make_float4(boundval, boundval, boundval, boundval));
}

extern "C" __global__ void gvdbUpdateApronC ( VDBInfo* gvdb, uchar chan, int brickcnt, int brickres, int brickwid, float boundval)
{
	UpdateApron<uchar>(gvdb, chan, brickcnt, brickres, boundval);
}

extern "C" __global__ void gvdbUpdateApronC4 ( VDBInfo* gvdb, uchar chan, int brickcnt, int brickres, int brickwid, float boundval)
{
	UpdateApron<uchar4>(gvdb, chan, brickcnt, brickres, make_uchar4(boundval, boundval, boundval, boundval));
}

// Loads the shared memory for this CUDA block, given the index of this thread in the
// shared memory, and its location in the atlas. This should be called for values of
// ndx in the range [1,8]^3; this function will make additional loads as needed to
// fill in the voxels adjacent along an axis to [1,8]^3.
// This assumes that the function is being called with a block size of (8, 8, 8).
template<class T>
__device__ void LoadSharedMemory(VDBInfo* gvdb, uchar channel, T sharedVoxels[10][10][10], uint3 ndx, uint3 voxI) {
	// Copy the atlas voxel at voxI into sharedVoxels[ndx].
	sharedVoxels[ndx.x][ndx.y][ndx.z] = surf3Dread<T>(gvdb->volOut[channel], voxI.x * sizeof(T), voxI.y, voxI.z);

	// Load voxels adjacent to [1,9]^3.
	if (ndx.x == 1) {
		sharedVoxels[0][ndx.y][ndx.z] = surf3Dread<T>(gvdb->volOut[channel], (voxI.x - 1) * sizeof(T), voxI.y, voxI.z);
	}
	else if (ndx.x == 8) {
		sharedVoxels[9][ndx.y][ndx.z] = surf3Dread<T>(gvdb->volOut[channel], (voxI.x + 1) * sizeof(T), voxI.y, voxI.z);
	}

	if (ndx.y == 1) {
		sharedVoxels[ndx.x][0][ndx.z] = surf3Dread<T>(gvdb->volOut[channel], voxI.x * sizeof(T), voxI.y - 1, voxI.z);
	}
	else if (ndx.y == 8) {
		sharedVoxels[ndx.x][9][ndx.z] = surf3Dread<T>(gvdb->volOut[channel], voxI.x * sizeof(T), voxI.y + 1, voxI.z);
	}

	if (ndx.z == 1) {
		sharedVoxels[ndx.x][ndx.y][0] = surf3Dread<T>(gvdb->volOut[channel], voxI.x * sizeof(T), voxI.y, voxI.z - 1);
	}
	else if (ndx.z == 8) {
		sharedVoxels[ndx.x][ndx.y][9] = surf3Dread<T>(gvdb->volOut[channel], voxI.x * sizeof(T), voxI.y, voxI.z + 1);
	}

	// Make sure all loads from the block have completed.
	__syncthreads();
}

// Suppose you're running a kernel with a block size of (8, 8, 8). This function
// will take each thread's indices and return:
// localIdx: threadIdx + gvdb->atlas_apron
// atlasIdx: The corresponding voxel in the atlas, skipping over aprons.
// For instance, for the brick starting at (0,0,0), atlasIdx will be equal to
// localIdx. But this will not be the case for other bricks.
__device__ void GetVoxelIndicesPacked(VDBInfo* gvdb, uint3& localIdx, uint3& atlasIdx) {
	const uint3 atlasApron = make_uint3(gvdb->atlas_apron);
	localIdx = threadIdx + atlasApron;

	// What atlasIdx would be if atlas_apron were 0
	const uint3 packedVox = blockIdx * blockDim + threadIdx;

	// Find the 3D index of the brick atlasIdx corresponds to
	const int brickResNoApron = gvdb->brick_res - 2 * gvdb->atlas_apron;
	const uint3 brick = make_uint3(
		packedVox.x / brickResNoApron,
		packedVox.y / brickResNoApron,
		packedVox.z / brickResNoApron);

	// Convert to a position in the full atlas
	atlasIdx = packedVox + brick * atlasApron * 2 + atlasApron;
}

// A helper macro that sets up local and atlas coordinates, checks to see if
// they're inside the atlas bounds, and returns if not. Skips over atlas
// boundaries, which means that it can use a smaller computation grid than
// the older GVDB_VOX. Assumes a block size of (8,8,8).
#define GVDB_VOXPACKED \
	uint3 localIdx, atlasIdx; \
	GetVoxelIndicesPacked(gvdb, localIdx, atlasIdx); \
	if (atlasIdx.x >= atlasRes.x || atlasIdx.y >= atlasRes.y || atlasIdx.z >= atlasRes.z) return;

// A helper macro that sets up unpacked local and atlas coordinates, checks to
// see if they're inside the atlas bounds, and returns if not. Does not skip
// over atlas boundaries, so this covers the entire atlas. This used to be GVDB_VOX.
#define GVDB_VOXUNPACKED \
	uint3 localIdx = threadIdx + make_uint3(1, 1, 1); \
	uint3 atlasIdx = blockIdx * blockDim + localIdx; \
	if (atlasIdx.x >= atlasRes.x || atlasIdx.y >= atlasRes.y || atlasIdx.z >= atlasRes.z) return;

extern "C" __global__ void gvdbOpGrow ( VDBInfo* gvdb, int3 atlasRes, uchar channel, float p1, float p2, float p3 )
{
	__shared__ float sharedVoxels[10][10][10];
	GVDB_VOXPACKED
	LoadSharedMemory<float>(gvdb, channel, sharedVoxels, localIdx, atlasIdx);
	
	/*float nl;	
	float3 n;
	n.x = 0.5 * (svox[ndx.x-1][ndx.y][ndx.z] - svox[ndx.x+1][ndx.y][ndx.z]);
	n.y = 0.5 * (svox[ndx.x][ndx.y-1][ndx.z] - svox[ndx.x][ndx.y+1][ndx.z]);
	n.z = 0.5 * (svox[ndx.x][ndx.y][ndx.z-1] - svox[ndx.x][ndx.y][ndx.z+1]);
	nl = sqrt(n.x*n.x+n.y*n.y+n.z*n.z);	
	float v = svox[ndx.x][ndx.y][ndx.z];
	if ( nl > 0.2 ) v += amt;*/ 

	float v = sharedVoxels[localIdx.x][localIdx.y][localIdx.z];
	if ( v != 0.0) v += p1 * 10.0; //*0.1;
	if ( v < 0.01) v = 0.0;
	
	surf3Dwrite(v, gvdb->volOut[channel], atlasIdx.x * sizeof(float), atlasIdx.y, atlasIdx.z);
}

extern "C" __global__ void gvdbOpCut ( VDBInfo* gvdb, int3 atlasRes, uchar channel, float p1, float p2, float p3 )
{
	__shared__ float sharedVoxels[10][10][10];
	GVDB_VOXPACKED
	LoadSharedMemory<float>(gvdb, channel, sharedVoxels, localIdx, atlasIdx);

	// Determine block and index position	
	float3 wpos;
	if (!getAtlasToWorld(gvdb, atlasIdx, wpos)) return;

	float v = sharedVoxels[localIdx.x][localIdx.y][localIdx.z];
	if (wpos.x < 50 && v > 0 && v < 2) {
		v = 0.02;
		surf3Dwrite(v, gvdb->volOut[channel], atlasIdx.x * sizeof(float), atlasIdx.y, atlasIdx.z);
	}
}

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
		v = surf3Dread<float>(gvdb->volOut[chan], vox.x * sizeof(float), vox.y, vox.z);
		sum += v;   // (v > gvdb->thresh.x) ? 1.0 : 0.0;
	}

	outbuf[packvox.y * packres.x + packvox.x] = sum;
}


extern "C" __global__ void gvdbResample ( VDBInfo* gvdb, int3 atlasRes, uchar chan, int3 srcres, float* src, float* xform, float3 inr, float3 outr )
{
	GVDB_VOXUNPACKED

	float3 wpos;
	if (!getAtlasToWorld(gvdb, atlasIdx, wpos)) return;
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

	surf3Dwrite(v, gvdb->volOut[chan], atlasIdx.x * sizeof(float), atlasIdx.y, atlasIdx.z);
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

extern "C" __global__ void gvdbOpFillF  ( VDBInfo* gvdb, int3 atlasRes, uchar channel, float p1, float p2, float p3 )
{
	GVDB_VOXUNPACKED

	if ( p3 < 0 ) {
		//float v = vox.y; // + (vox.z*30 + vox.x)/900.0;
		float v = sinf(atlasIdx.x * 12 / (3.141592 * 30.0));
		v += sinf(atlasIdx.y * 12 / (3.141592 * 30.0));
		v += sinf(atlasIdx.z * 12 / (3.141592 * 30.0));
		surf3Dwrite(v, gvdb->volOut[channel], atlasIdx.x * sizeof(float), atlasIdx.y, atlasIdx.z);
	} else {
		surf3Dwrite(p1, gvdb->volOut[channel], atlasIdx.x * sizeof(float), atlasIdx.y, atlasIdx.z);
	}
}

extern "C" __global__ void gvdbOpFillC4 ( VDBInfo* gvdb, int3 atlasRes, uchar channel, float p1, float p2, float p3 )
{
	GVDB_VOXUNPACKED

	surf3Dwrite ( CLR2INT(make_float4(p1,p2,p3,1.0)), gvdb->volOut[channel], atlasIdx.x*sizeof(uchar4), atlasIdx.y, atlasIdx.z );
}

extern "C" __global__ void gvdbOpFillC ( VDBInfo* gvdb, int3 atlasRes, uchar channel, float p1, float p2, float p3 )
{
	GVDB_VOXUNPACKED

	const uchar c = static_cast<uchar>(p1);
	surf3Dwrite(c, gvdb->volOut[channel], atlasIdx.x * sizeof(uchar), atlasIdx.y, atlasIdx.z);
}

extern "C" __global__ void gvdbOpSmooth ( VDBInfo* gvdb, int3 atlasRes, uchar channel, float p1, float p2, float p3 )
{
	__shared__ float sharedVoxels[10][10][10];
	GVDB_VOXPACKED
	LoadSharedMemory<float>(gvdb, channel, sharedVoxels, localIdx, atlasIdx);

	//-- smooth
	float v = p1 * sharedVoxels[localIdx.x][localIdx.y][localIdx.z];
	v += sharedVoxels[localIdx.x - 1][localIdx.y][localIdx.z];
	v += sharedVoxels[localIdx.x + 1][localIdx.y][localIdx.z];
	v += sharedVoxels[localIdx.x][localIdx.y - 1][localIdx.z];
	v += sharedVoxels[localIdx.x][localIdx.y + 1][localIdx.z];
	v += sharedVoxels[localIdx.x][localIdx.y][localIdx.z - 1];
	v += sharedVoxels[localIdx.x][localIdx.y][localIdx.z + 1];
	v = v / (p1 + 6.0) + p2;

	surf3Dwrite(v, gvdb->volOut[channel], atlasIdx.x * sizeof(float), atlasIdx.y, atlasIdx.z);
}

extern "C" __global__ void gvdbOpClrExpand ( VDBInfo* gvdb, int3 atlasRes, uchar channel, float p1, float p2, float p3 )
{
	__shared__ uchar4 sharedVoxels[10][10][10];
	GVDB_VOXPACKED
	LoadSharedMemory<uchar4>(gvdb, channel, sharedVoxels, localIdx, atlasIdx);

	int3 c, cs;
	int cp;
	c = make_int3(sharedVoxels[localIdx.x][localIdx.y][localIdx.z]);   cs = c * p1;
	c = make_int3(sharedVoxels[localIdx.x - 1][localIdx.y][localIdx.z]); cs += c * p2;
	c = make_int3(sharedVoxels[localIdx.x + 1][localIdx.y][localIdx.z]); cs += c * p2;
	c = make_int3(sharedVoxels[localIdx.x][localIdx.y - 1][localIdx.z]); cs += c * p2;
	c = make_int3(sharedVoxels[localIdx.x][localIdx.y + 1][localIdx.z]); cs += c * p2;
	c = make_int3(sharedVoxels[localIdx.x][localIdx.y][localIdx.z - 1]); cs += c * p2;
	c = make_int3(sharedVoxels[localIdx.x][localIdx.y][localIdx.z + 1]); cs += c * p2;
	cp = max(cs.x, max(cs.y,cs.z) );
	cs = (cp > 255) ? make_int3(cs.x*255/cp, cs.y*255/cp, cs.z*255/cp) : cs;

	surf3Dwrite ( make_uchar4(cs.x, cs.y, cs.z, 1), gvdb->volOut[channel], atlasIdx.x*sizeof(uchar4), atlasIdx.y, atlasIdx.z );
}

extern "C" __global__ void gvdbOpExpandC ( VDBInfo* gvdb, int3 atlasRes, uchar channel, float p1, float p2, float p3 )
{
	__shared__ uchar sharedVoxels[10][10][10];
	GVDB_VOXPACKED
	LoadSharedMemory<uchar>(gvdb, channel, sharedVoxels, localIdx, atlasIdx);

	uchar c = 0;
	c = (sharedVoxels[localIdx.x - 1][localIdx.y][localIdx.z] == (uchar)p1) ? 1 : c;
	c = (sharedVoxels[localIdx.x + 1][localIdx.y][localIdx.z] == (uchar)p1) ? 1 : c;
	c = (sharedVoxels[localIdx.x][localIdx.y - 1][localIdx.z] == (uchar)p1) ? 1 : c;
	c = (sharedVoxels[localIdx.x][localIdx.y + 1][localIdx.z] == (uchar)p1) ? 1 : c;
	c = (sharedVoxels[localIdx.x][localIdx.y][localIdx.z - 1] == (uchar)p1) ? 1 : c;
	c = (sharedVoxels[localIdx.x][localIdx.y][localIdx.z + 1] == (uchar)p1) ? 1 : c;
	
	uchar v = sharedVoxels[localIdx.x][localIdx.y][localIdx.z];
	if ( v == 0 && c == 1 ) {
		c = static_cast<uchar>(p2);
		surf3Dwrite(c, gvdb->volOut[channel], atlasIdx.x * sizeof(uchar), atlasIdx.y, atlasIdx.z);
	}
}

extern "C" __global__ void gvdbOpNoise ( VDBInfo* gvdb, int3 atlasRes, uchar channel, float p1, float p2, float p3 )
{
	__shared__ float sharedVoxels[10][10][10];
	GVDB_VOXPACKED
	LoadSharedMemory<float>(gvdb, channel, sharedVoxels, localIdx, atlasIdx);

	//-- noise
	float v = sharedVoxels[localIdx.x][localIdx.y][localIdx.z];
	if (v > 0.01) v += random(make_float3(atlasIdx.x, atlasIdx.y, atlasIdx.z)) * p1;

	surf3Dwrite(v, gvdb->volOut[channel], atlasIdx.x * sizeof(float), atlasIdx.y, atlasIdx.z);
}

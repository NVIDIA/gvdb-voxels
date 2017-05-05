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
// File: cuda_gvdb_copydata.cu
//
// GVDB Data Transfers
// - CopyData		3D volume into sub-volume
// - CopyDataZYX	3D volume into sub-volume with ZYX swizzle
// - RetreiveData	3D sub-volume into cuda buffer
// - CopyTexToBuf	2D texture into cuda buffer
// - CopyBufToTex	cuda buffer into 2D texture
//-----------------------------------------------

#include "cuda_math.cuh"

texture<unsigned char, cudaTextureType3D, cudaReadModeElementType>		volTexInC;
texture<float, cudaTextureType3D, cudaReadModeElementType>		volTexInF;
surface<void, cudaTextureType3D>								volTexOut;

// Zero memory of 3D volume
extern "C" __global__ void kernelFillTex ( int3 res, int dsize )
{
	uint3 t = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + threadIdx;	
	if ( t.x >= res.x || t.y >= res.y || t.z >= res.z ) return;

	surf3Dwrite ( 0, volTexOut, t.x*dsize, t.y, t.z );
}

// Copy 3D texture into sub-volume of another 3D texture (char)
extern "C" __global__ void kernelCopyTexC ( int3 offs, int3 res )
{
	uint3 t = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + threadIdx;	
	if ( t.x >= res.x || t.y >= res.y || t.z >= res.z ) return;
	uchar val = tex3D ( volTexInC, t.x, t.y, t.z );
	surf3Dwrite ( val, volTexOut, (t.x+offs.x)*sizeof(char), (t.y+offs.y), (t.z+offs.z) );
}

// Copy 3D texture into sub-volume of another 3D texture (float)
extern "C" __global__ void kernelCopyTexF ( int3 offs, int3 res )
{
	uint3 t = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + threadIdx;	
	if ( t.x >= res.x || t.y >= res.y || t.z >= res.z ) return;	
	float val = tex3D ( volTexInF, t.x, t.y, t.z );
	surf3Dwrite ( val, volTexOut, (t.x+offs.x)*sizeof(float), (t.y+offs.y), (t.z+offs.z) );
}

// Copy linear memory as 3D volume into sub-volume of a 3D texture
extern "C" __global__ void kernelCopyBufToTexC ( int3 offs, int3 res, uchar* inbuf)
{
	uint3 t = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + threadIdx;	
	if ( t.x >= res.x || t.y >= res.y || t.z >= res.z ) return;	
	unsigned char val = inbuf[ (t.z*res.y + t.y)*res.x + t.x ];	
	surf3Dwrite ( val, volTexOut, (t.x+offs.x)*sizeof(uchar), (t.y+offs.y), (t.z+offs.z) );
}
// Copy linear memory as 3D volume into sub-volume of a 3D texture
extern "C" __global__ void kernelCopyBufToTexF ( int3 offs, int3 res, float* inbuf)
{
	uint3 t = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + threadIdx;	
	if ( t.x >= res.x || t.y >= res.y || t.z >= res.z ) return;	
	float val = inbuf[ (t.z*res.y + t.y)*res.x + t.x ];	
	surf3Dwrite ( val, volTexOut, (t.x+offs.x)*sizeof(float), (t.y+offs.y), (t.z+offs.z) );
}

// Copy 3D texture into sub-volume of another 3D texture with ZYX swizzle (float)
extern "C" __global__ void kernelCopyTexZYX (  int3 offs, int3 res )
{
	uint3 t = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + threadIdx;	
	if ( t.x >= res.x || t.y >= res.y || t.z >= res.z ) return;
	float val = tex3D ( volTexInF, t.z, t.y, t.x );
	surf3Dwrite ( val, volTexOut, (t.x+offs.x)*sizeof(float), (t.y+offs.y), (t.z+offs.z) );
}

// Retrieve 3D texture into linear memory (float)
extern "C" __global__ void kernelRetrieveTexXYZ ( int3 offs, int3 src_res, int3 res, float* buf )
{
	uint3 t = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + threadIdx;	
	if ( t.x >= src_res.x || t.y >= src_res.y || t.z >= src_res.z ) return;
	float val = tex3D ( volTexInF, t.x+offs.x, t.y+offs.y, t.z+offs.z );
	buf[ (t.x*res.y + t.y)*res.x + t.z ] = val;
}

// Copy 2D slice of 3D texture into 2D linear buffer
extern "C" __global__ void kernelSliceTexToBufF ( int slice, int3 res, float* outbuf  )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;	
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= res.x || y >= res.y ) return;
	float val = tex3D ( volTexInF, x, y, slice );
	outbuf[ y*res.x + x ] = val;
}

extern "C" __global__ void kernelSliceTexToBufC ( int slice, int3 res, uchar* outbuf  )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;	
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= res.x || y >= res.y ) return;
	uchar val = tex3D ( volTexInC, x, y, slice );
	outbuf[ y*res.x + x ] = val;
}


// Copy 2D linear buffer into the 2D slice of a 3D texture
extern "C" __global__ void kernelSliceBufToTexF ( int slice, int3 res, float* inbuf  )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;	
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= res.x || y >= res.y ) return;
	float val = inbuf[ y*res.x + x ];
	surf3Dwrite ( val, volTexOut, x*sizeof(float), y, slice );
}
extern "C" __global__ void kernelSliceBufToTexC ( int slice, int3 res, uchar* inbuf  )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;	
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= res.x || y >= res.y ) return;
	uchar val = inbuf[ y*res.x + x ];
	surf3Dwrite ( val, volTexOut, x*sizeof(uchar), y, slice );
}



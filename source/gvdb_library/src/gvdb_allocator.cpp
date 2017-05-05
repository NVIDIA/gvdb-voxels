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

#include "gvdb_allocator.h"
#include "gvdb_render.h"

#if defined(_WIN32)
#	include <windows.h>
#endif
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda.h>

using namespace nvdb;

bool		Allocator::bAllocator = false;
CUmodule	Allocator::cuAllocatorModule;
CUfunction	Allocator::cuFillTex;
CUfunction	Allocator::cuCopyTexC;
CUfunction	Allocator::cuCopyTexF;
CUfunction	Allocator::cuCopyBufToTexC;
CUfunction	Allocator::cuCopyBufToTexF;
CUfunction	Allocator::cuCopyTexZYX;
CUfunction	Allocator::cuRetrieveTexXYZ;
CUfunction	Allocator::cuSliceTexToBufF;
CUfunction	Allocator::cuSliceTexToBufC;
CUfunction	Allocator::cuSliceBufToTexF;
CUfunction	Allocator::cuSliceBufToTexC;

CUsurfref	Allocator::cuSurfWrite;
CUtexref	Allocator::cuSurfReadC;
CUtexref	Allocator::cuSurfReadF;

Allocator::Allocator ()
{
	mVFBO[0] = -1;
	
	if ( !bAllocator ) {
		bAllocator = true;
		cudaCheck ( cuModuleLoad ( &cuAllocatorModule, "cuda_gvdb_copydata.ptx" ), "cuModuleLoad", "Allocator" );
		
		cudaCheck ( cuModuleGetFunction ( &cuFillTex,		cuAllocatorModule, "kernelFillTex" ), "cuModuleGetFunc(cuFillTex)", "Allocator" );
		cudaCheck ( cuModuleGetFunction ( &cuCopyTexC,		cuAllocatorModule, "kernelCopyTexC" ), "cuModuleGetFunc(cuCopyTexC)", "Allocator" );
		cudaCheck ( cuModuleGetFunction ( &cuCopyTexF,		cuAllocatorModule, "kernelCopyTexF" ), "cuModuleGetFunct(cuCopTexF)", "Allocator" );
		cudaCheck ( cuModuleGetFunction ( &cuCopyBufToTexC,	cuAllocatorModule, "kernelCopyBufToTexC" ), "cuModuleGetFunc(cuCopyBufToTexC)", "Allocator" );
		cudaCheck ( cuModuleGetFunction ( &cuCopyBufToTexF,	cuAllocatorModule, "kernelCopyBufToTexF" ), "cuModuleGetFunc(cuCopyBufToTexF)", "Allocator" );
		cudaCheck ( cuModuleGetFunction ( &cuCopyTexZYX,	cuAllocatorModule, "kernelCopyTexZYX" ), "cuModuleGetFunc(cuCopyTexZYX)", "Allocator" );
		cudaCheck ( cuModuleGetFunction ( &cuRetrieveTexXYZ, cuAllocatorModule, "kernelRetrieveTexXYZ" ), "cuModuleGetFunc(cuRetrieveTexXYZ)", "Allocator" );
		cudaCheck ( cuModuleGetFunction ( &cuSliceTexToBufF, cuAllocatorModule, "kernelSliceTexToBufF" ), "cuModuleGetFunc(cuSliceTexToBufF)", "Allocator" );
		cudaCheck ( cuModuleGetFunction ( &cuSliceTexToBufC, cuAllocatorModule, "kernelSliceTexToBufC" ), "cuModuleGetFunc(cuSliceTexToBufC)", "Allocator" );
		cudaCheck ( cuModuleGetFunction ( &cuSliceBufToTexF, cuAllocatorModule, "kernelSliceBufToTexF" ), "cuModuleGetFunc(cuSliceBufToTexF)", "Allocator" );		
		cudaCheck ( cuModuleGetFunction ( &cuSliceBufToTexC, cuAllocatorModule, "kernelSliceBufToTexC" ), "cuModuleGetFunc(cuSliceBufToTexC)", "Allocator" );		
		
		cudaCheck ( cuModuleGetSurfRef ( &cuSurfWrite,		cuAllocatorModule, "volTexOut" ), "cuModuleGetSurfRef", "Allocator" );
		cudaCheck ( cuModuleGetTexRef  ( &cuSurfReadC,		cuAllocatorModule, "volTexInC" ), "cuModuleGetTexRef(ReadC)", "Allocator" );
		cudaCheck ( cuModuleGetTexRef  ( &cuSurfReadF,		cuAllocatorModule, "volTexInF" ), "cuModuleGetTexRef(ReadF)", "Allocator" );
	}	
}


void Allocator::PoolCreate ( uchar grp, uchar lev, uint64 width, uint64 initmax, bool bGPU )
{
	if ( grp > MAX_POOL ) {
		gprintf ( "ERROR: Exceeded maximum number of pools. %d, max %d\n", grp, MAX_POOL );
		gerror ();
	}		
	while ( mPool[grp].size() < lev ) 
		mPool[grp].push_back ( DataPtr() );

	DataPtr p;
	p.alloc = this;
	p.type = T_UCHAR;
	p.num = 0;
	p.max = initmax;
	p.size = width * initmax;	
	p.stride = width;		
	p.cpu = 0x0;
	p.gpu = 0x0;
	
	if ( p.size == 0 ) return;		// placeholder pool, do not allocate

	// cpu allocate
	p.cpu = (char*) malloc ( p.size );
	if ( p.cpu == 0x0 ) {
		gprintf ( "ERROR: Unable to malloc %lld for pool lev %d\n", p.size, lev );
		gerror ();
	}	

	// gpu allocate
	if ( bGPU ) {
		size_t sz = p.size;
		cudaCheck ( cuMemAlloc ( &p.gpu, sz ), "cuMemAlloc", "PoolCreate" );
	}
	mPool[grp].push_back ( p );
}
void Allocator::PoolCommitAll ()
{
	for (int grp=0; grp < MAX_POOL; grp++) 
		for (int lev=0; lev < mPool[grp].size(); lev++ )
			PoolCommit ( grp, lev );
}

void Allocator::PoolClearCPU()
{
	for (int grp=0; grp < MAX_POOL; grp++) 
		for (int lev=0; lev < mPool[grp].size(); lev++ )	
		{
			DataPtr* p = &mPool[grp][lev];
			memset(p->cpu, 0, p->size);
		}
}

void Allocator::PoolFetch(int grp, int lev )
{
	DataPtr* p = &mPool[grp][lev];
	cudaCheck ( cuMemcpyDtoH ( p->cpu,  p->gpu, p->num * p->stride ), "cuMemcpyDtoH", "PoolFetch" );	// Kui: fetch pool from GPU
}

void Allocator::PoolFetchAll()
{
	for (int grp=0; grp < MAX_POOL; grp++) 
		for (int lev=0; lev < mPool[grp].size() ; lev++ )	
			PoolFetch ( grp, lev );
}


void Allocator::PoolCommit ( int grp, int lev )
{
	DataPtr* p = &mPool[grp][lev];
	cudaCheck ( cuMemcpyHtoD ( p->gpu, p->cpu, p->num * p->stride ), "cuMemcpyHtoD", "PoolCommit" );	
}

void Allocator::PoolCommitAtlasMap ()
{
	DataPtr* p;
	for (int n=0; n < mAtlasMap.size(); n++ ) {
		if ( mAtlasMap[n].cpu != 0x0 ) {
			p = &mAtlasMap[n];
			cudaCheck ( cuMemcpyHtoD ( p->gpu, p->cpu, p->num * p->stride ), "cuMemcpyHtoD", "PoolCommitAtlasMap" );
		}
	}	
}

void Allocator::PoolReleaseAll ()
{
	// release all memory
	for (int grp=0; grp < MAX_POOL; grp++) 
		for (int lev=0; lev < mPool[grp].size(); lev++ )  {
			if ( mPool[grp][lev].cpu != 0x0 ) 
				free ( mPool[grp][lev].cpu );

			if ( mPool[grp][lev].gpu != 0x0 )
				cudaCheck ( cuMemFree ( mPool[grp][lev].gpu ), "cuFree", "PoolReleaseAll" );
		}


	// release pool structure	
	for (int grp=0; grp < MAX_POOL; grp++) 
		mPool[grp].clear ();
}


uint64 Allocator::PoolAlloc ( uchar grp, uchar lev, bool bGPU )
{
	if ( lev >= mPool[grp].size() ) return ID_UNDEFL;
	DataPtr* p = &mPool[grp][lev];
	
	if ( p->num >= p->max ) {
		// Expand pool
		p->max *= 2;
		p->size = p->stride * p->max;
		if ( p->cpu != 0x0 ) {
			char* new_cpu = (char*) malloc ( p->size );
			memcpy ( new_cpu, p->cpu, p->stride*p->num );
			free ( p->cpu );
			p->cpu = new_cpu;
		}
		if ( p->gpu != 0x0 ) {
			size_t sz = p->size;	
			CUdeviceptr new_gpu;
			cudaCheck ( cuMemAlloc ( &new_gpu, sz ), "cuMemAlloc", "PoolAlloc" );		
			cudaCheck ( cuMemcpy ( new_gpu, p->gpu, p->stride*p->num), "cuMemcpy", "PoolAlloc" );
			cudaCheck ( cuMemFree ( p->gpu ), "cuMemFree", "PoolAlloc" );
			p->gpu = new_gpu;
		}
	}
	// Return new element
	p->num++;	
	return Elem(grp,lev, (p->num-1) );
}

void Allocator::PoolEmptyAll ()
{
	// clear pool data (do not free)
	for (int grp=0; grp < MAX_POOL; grp++) 
		for (int lev=0; lev < mPool[grp].size(); lev++ ) 
			mPool[grp][lev].num = 0;		
}

int	Allocator::getPoolMem ()
{
	slong sz = 0;
	for (int grp=0; grp < MAX_POOL; grp++) 
		for (int lev=0; lev < mPool[grp].size(); lev++ ) 
			sz += mPool[grp][lev].size;
	return sz / slong(1024*1024);
}

char* Allocator::PoolData ( uint64 elem )
{
	register uchar g = ElemGrp(elem);
	register uchar l = ElemLev(elem);
	char* pool = mPool[g][l].cpu;
	return pool + mPool[g][l].stride * ElemNdx(elem);
}
char* Allocator::PoolData ( uchar grp, uchar lev, uint64 ndx )
{
	char* pool = mPool[grp][lev].cpu;
	return pool + mPool[grp][lev].stride * ndx;
}

uint64* Allocator::PoolData64 ( uint64 elem )
{
	return (uint64*) PoolData ( elem );
}

void PoolFree ( int elem )
{
}


uint64 Allocator::getPoolWidth ( uchar grp, uchar lev )
{
	return mPool[grp][lev].stride;
}

int	Allocator::getSize ( uchar dtype )
{	
	switch ( dtype ) {	
	case T_UCHAR:		return sizeof(uchar);	break;
	case T_UCHAR3:		return 3*sizeof(uchar);	break;
	case T_UCHAR4:		return 4*sizeof(uchar);	break;
	case T_FLOAT:		return sizeof(float);	break;
	case T_FLOAT3:		return 3*sizeof(float);	break;
	case T_FLOAT4:		return 4*sizeof(float);	break;
	case T_INT:			return sizeof(int);		break;
	case T_INT3:		return 3*sizeof(int);	break;
	case T_INT4:		return 4*sizeof(int);	break;
	}
	return 0;
}

void Allocator::CreateMemLinear ( DataPtr& p, char* dat, int sz )
{
	CreateMemLinear ( p, dat, 1, sz, false );
}

void Allocator::CreateMemLinear ( DataPtr& p, char* dat, int stride, int cnt, bool bCPU )
{
	p.alloc = this;
	p.num = cnt; p.max = cnt; 
	p.stride = stride;
	p.size = cnt * stride;
	p.subdim = Vector3DI(0,0,0);

	if ( dat==0x0 ) {
		if ( bCPU ) {
			if ( p.cpu != 0x0 ) free (p.cpu);		// release previous
			p.cpu = (char*) malloc ( p.size );		// create on cpu 
		}
	} else {
		p.cpu = dat;							// get from user
	}
	if ( p.gpu != 0x0 ) cudaCheck ( cuMemFree (p.gpu), "cuMemFree", "CreateMemLinear" );
	cudaCheck ( cuMemAlloc ( &p.gpu, p.size ), "cuMemAlloc", "CreateMemLinear" );	

	if ( dat!=0x0 ) CommitMem ( p );			// transfer from user
}

void Allocator::FreeMemLinear ( DataPtr& p )
{
	if ( p.cpu != 0x0 ) free (p.cpu);
	if ( p.gpu != 0x0 ) cudaCheck ( cuMemFree (p.gpu), "cuMemFree", "FreeMemLinear" );
	p.cpu = 0x0;
	p.gpu = 0x0;
}

void Allocator::RetrieveMem ( DataPtr& p)
{
	cudaCheck ( cuMemcpyDtoH ( p.cpu, p.gpu, p.size), "cuMemcpyDtoH", "RetrieveMem" );	
	cudaCheck ( cuCtxSynchronize (), "cuCtxSync", "RetrieveMem" );
}

void Allocator::CommitMem ( DataPtr& p)
{
	cudaCheck ( cuMemcpyHtoD ( p.gpu, p.cpu, p.size), "cuMemcpyHtoD", "CommitMem" );
}


void Allocator::AllocateTextureGPU ( DataPtr& p, uchar dtype, Vector3DI res, bool bGL, uint64 preserve )
{	
	// GPU allocate	
	if ( bGL ) {
		// OpenGL 3D texture
		#ifdef BUILD_OPENGL				

			unsigned char *pixels = 0x0; 
			int old_glid = p.glid;

			// Generate texture 
			if ( res.x==0 || res.y==0 || res.z==0 ) return;
			glGenTextures(1, (GLuint*) &p.glid );
			checkGL ( "glGenTextures (AtlasCreate)" );
			glBindTexture ( GL_TEXTURE_3D, p.glid );					
			glPixelStorei ( GL_PACK_ALIGNMENT, 4 );	
			glPixelStorei ( GL_UNPACK_ALIGNMENT, 4 );
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);		
			checkGL ( "glBindTexture (AtlasCreate)" );
	
			switch ( dtype ) {
			case T_UCHAR:	glTexImage3D ( GL_TEXTURE_3D, 0, GL_R8,		res.x, res.y, res.z, 0, GL_RED, GL_UNSIGNED_BYTE, 0);	break;
			case T_UCHAR4:	glTexImage3D ( GL_TEXTURE_3D, 0, GL_RGBA8,	res.x, res.y, res.z, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);	break;
			case T_FLOAT:	glTexImage3D ( GL_TEXTURE_3D, 0, GL_R32F,	res.x, res.y, res.z, 0, GL_RED, GL_FLOAT, 0);			break;
			};
			checkGL ( "glTexImage3D (AtlasCreate)" );

			// preserve old texture data
			if ( preserve > 0 && old_glid != -1 ) {
				Vector3DI src (res.x, res.y, preserve / (res.x*res.y*getSize(dtype)) );	 // src amount to copy 				
				glCopyImageSubData ( old_glid, GL_TEXTURE_3D, 0, 0,0,0, p.glid, GL_TEXTURE_3D, 0, 0,0,0, src.x, src.y, src.z );
			}
			if ( old_glid != -1 ) {
				GLuint id = old_glid;
				glDeleteTextures ( 1, &id );
			}

			// CUDA-GL interop for CUarray
			if ( p.grsc != 0 ) cudaCheck ( cuGraphicsUnregisterResource ( p.grsc ), "cuGraphicsUnregisterResource", "AllocateTextureGPU" );
			cudaCheck ( cuGraphicsGLRegisterImage ( &p.grsc, p.glid, GL_TEXTURE_3D, CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST ), "cuGraphicsGLRegisterImage", "AllocateTextureGPU" );
			checkGL ( "cuGraphicsGLRegisterImage" );
			cudaCheck ( cuGraphicsMapResources(1, &p.grsc, 0), "cudaGraphicsMapResources", "AllocateTextureGPU" );
			cudaCheck ( cuGraphicsSubResourceGetMappedArray ( &p.garray, p.grsc, 0, 0 ), "cuGraphicsSubResourceGetMappedArray", "AllocateTextureGPU" );			
			cudaCheck ( cuGraphicsUnmapResources(1, &p.grsc, 0), "cudaGraphicsUnmapRsrc", "AllocateTextureGPU" );
		#endif

	} else {

		// Create CUarray in CUDA
		CUDA_ARRAY3D_DESCRIPTOR desc;
		switch ( dtype ) {
		case T_FLOAT:	desc.Format = CU_AD_FORMAT_FLOAT;			desc.NumChannels = 1; break;
		case T_FLOAT3:	desc.Format = CU_AD_FORMAT_FLOAT;			desc.NumChannels = 4; break;
		case T_UCHAR:	desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;	desc.NumChannels = 1; break;	// INT8 = UCHAR
		case T_UCHAR3:	desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;	desc.NumChannels = 3; break;
		case T_UCHAR4:	desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;	desc.NumChannels = 4; break;
		};
		desc.Width = res.x;
		desc.Height = res.y;
		desc.Depth = res.z;
		desc.Flags = CUDA_ARRAY3D_SURFACE_LDST;	
		CUarray old_array = p.garray;

		if ( res.x > 0 && res.y > 0 && res.z > 0 ) {
			cudaCheck ( cuArray3DCreate( &p.garray, &desc), "cuArray3DCreate", "AllocateTextureGPU" );
			if ( preserve > 0 && old_array != 0 ) {
				CUDA_MEMCPY3D cp = {0};
				cp.dstMemoryType = CU_MEMORYTYPE_ARRAY;
				cp.dstArray = p.garray;
				cp.srcMemoryType = CU_MEMORYTYPE_ARRAY;				
				cp.srcArray = old_array;
				cp.WidthInBytes = res.x * getSize(dtype);
				cp.Height = res.y;
				cp.Depth = preserve / (res.x*res.y*getSize(dtype));	  // amount to copy (preserve)
				if ( cp.Depth < desc.Depth )
				   cudaCheck ( cuMemcpy3D ( &cp ), "cuMemcpy3D(preserve)", "AllocateTextureGPU" );
			}	
		} else {
			p.garray = 0;
		}
		if ( old_array != 0 ) cudaCheck ( cuArrayDestroy ( old_array ), "cuArrayDestroy", "AllocateTextureGPU" );		
	}	
}


void Allocator::AllocateTextureCPU ( DataPtr& p, uint64 sz, bool bCPU, uint64 preserve )
{
	if ( bCPU ) {
		char* old_cpu = p.cpu;
		p.cpu = (char*) malloc ( p.size );		
		if ( preserve > 0 && old_cpu != 0x0 ) {
			memcpy ( p.cpu, old_cpu, preserve );
		}
		if ( old_cpu != 0x0 ) free ( old_cpu );	
	}
}

void Allocator::AllocateAtlasMap ( int stride, Vector3DI axiscnt )
{
	DataPtr q; 
	if ( mAtlasMap.size()== 0 ) {
		q.cpu = 0; q.gpu = 0; q.max = 0;
		mAtlasMap.push_back( q );
	}
	q = mAtlasMap[0];
	if ( axiscnt.x*axiscnt.y*axiscnt.z == q.max ) return;	// same size, return

	// Reallocate atlas mapping 	
	q.max = axiscnt.x * axiscnt.y * axiscnt.z;	// max leaves supported
	q.subdim = axiscnt;
	q.num = q.max;
	q.stride = stride;
	q.size = stride * q.max;					// list of mapping structs			
	if ( q.cpu != 0x0 ) free ( q.cpu );
	q.cpu = (char*) malloc ( q.size );			// cpu allocate		
			
	size_t sz = q.size;							// gpu allocate
	if ( q.gpu != 0x0 ) cudaCheck ( cuMemFree ( q.gpu ), "cuMemFree", "AllocateAtlasMap" );
	cudaCheck ( cuMemAlloc ( &q.gpu, q.size ), "cuMemAlloc", "AllocateAtlasMap" );	

	mAtlasMap[0] = q;
}

bool Allocator::TextureCreate ( uchar chan, uchar dtype, Vector3DI res, bool bCPU, bool bGL )
{
	DataPtr p;
	p.alloc = this;
	p.type = dtype;	
	p.num = 0;
	p.max = res.x * res.y * res.z;			// # of voxels
	uint64 atlas_sz = uint64(getSize(dtype)) * p.max;
	p.size = atlas_sz;						// size of texture
	p.apron = 0;							// apron is 0 for texture
	p.stride = 1;							// stride is 1 for texture (see: getAtlasRes)
	p.subdim = res;							// resolution
	p.cpu = 0x0;
	p.glid = -1;
	p.grsc = 0x0;
	p.garray = 0x0;

	// Atlas
	AllocateTextureGPU ( p, dtype, res, bGL, 0 );		// GPU allocate	
	AllocateTextureCPU ( p, p.size, bCPU, 0 );			// CPU allocate	
	mAtlas.push_back ( p );

	cudaCheck ( cuCtxSynchronize(), "cuCtxSync", "TextureCreate" );

	return true;
}



bool Allocator::AtlasCreate ( uchar chan, uchar dtype, Vector3DI leafdim, Vector3DI axiscnt, char apr, uint64 map_wid, bool bCPU, bool bGL )
{
	Vector3DI axisres;
	uint64 atlas_sz; 
	
	axisres = axiscnt;
	axisres *= (leafdim + apr*2);						// number of voxels along one axis
	atlas_sz = uint64(getSize(dtype)) * axisres.x * uint64(axisres.y) * axisres.z;

	DataPtr p;
	p.alloc = this;
	p.type = dtype;
	p.apron = apr;
	p.num = 0;
	p.max = axiscnt.x * axiscnt.y * axiscnt.z;			// max leaves supported	
	p.size = atlas_sz;									// size of 3D atlas (voxel count * data type)
	p.stride = leafdim.x;								// leaf dimensions - three axes are always equal
	p.subdim = axiscnt;									// axiscnt - count on each axes may differ, defaults to same
	p.cpu = 0x0;
	p.glid = -1;
	p.grsc = 0x0;
	p.garray = 0x0;

	// Atlas
	AllocateTextureGPU ( p, dtype, axisres, bGL, 0 );		// GPU allocate	
	AllocateTextureCPU ( p, p.size, bCPU, 0 );				// CPU allocate
	mAtlas.push_back ( p );

	cudaCheck ( cuCtxSynchronize(), "cuCtxSync", "AtlasCreate" );

	return true;
}

void Allocator::AtlasSetNum ( uchar chan, int n )
{
	mAtlas[chan].num = n;
}

bool Allocator::AtlasResize ( uchar chan, int cx, int cy, int cz )
{
	DataPtr p = mAtlas[chan];
	int leafdim = p.stride;
	Vector3DI axiscnt (cx, cy, cz);
	Vector3DI axisres;
	
	// Compute axis res
	axisres = axiscnt * int(leafdim + p.apron * 2);
	uint64 atlas_sz = uint64(getSize(p.type)) * axisres.x * uint64(axisres.y) * axisres.z;
	p.max = axiscnt.x * axiscnt.y * axiscnt.z;
	p.size = atlas_sz;
	p.subdim = axiscnt;

	// Atlas		
	AllocateTextureGPU ( p, p.type, axisres, (p.glid!=-1), 0 );
	AllocateTextureCPU ( p, p.size, (p.cpu!=0x0), 0 );
	mAtlas[chan] = p;

	return true;
}


bool Allocator::AtlasResize ( uchar chan, uint64 max_leaf )
{
	DataPtr p = mAtlas[chan];
	int leafdim = p.stride;			// dimensions of brick
	Vector3DI axisres;
	Vector3DI axiscnt = p.subdim;	// number of bricks on each axis	
	uint64 preserve = p.size;		// previous size of atlas (# bytes to preserve)

	// Expand Z-axis of atlas
	axiscnt.z = ceil ( max_leaf / float(axiscnt.x*axiscnt.y) );		// expand number of bricks along Z-axis
	
	// If atlas will have the same dimensions, do not reallocate
	if ( p.subdim.x==axiscnt.x && p.subdim.y==axiscnt.y && p.subdim.z==axiscnt.z )
		return true;

	// Compute axis res
	axisres = axiscnt * int(leafdim + p.apron * 2);		// new atlas resolution
	uint64 atlas_sz = uint64(getSize(p.type)) * axisres.x * uint64(axisres.y) * axisres.z;	// new atlas size
	p.max = axiscnt.x * axiscnt.y * axiscnt.z;			// max leaves supported
	p.size = atlas_sz;				// new total # bytes
	p.subdim = axiscnt;				// new number of bricks on each axis

	// Atlas		
	AllocateTextureGPU ( p, p.type, axisres, (p.glid!=-1), preserve );
	AllocateTextureCPU ( p, p.size, (p.cpu!=0x0), preserve );
	mAtlas[chan] = p;

	return true;
}


char* Allocator::getAtlasNode ( uchar chan, Vector3DI val )
{
	int leafres = mAtlas[chan].stride + (mAtlas[chan].apron << 1);			// leaf res
	Vector3DI axiscnt = mAtlas[chan].subdim;	
	Vector3DI i = Vector3DI(val.x/leafres, val.y/leafres, val.z/leafres);	// get brick index
	int id = (i.z*axiscnt.y + i.y) * axiscnt.x + i.x;						// get brick id	
	return mAtlasMap[0].cpu + id * mAtlasMap[0].stride;						// get mapping node for brick
}

void Allocator::AtlasEmptyAll ()
{
	for (int n=0; n < mAtlas.size(); n++ )
		mAtlas[n].num = 0;
}

bool Allocator::AtlasAlloc ( uchar chan, Vector3DI& val )
{
	int id;	
	if ( mAtlas[chan].num >= mAtlas[chan].max ) {
		int layer = mAtlas[chan].subdim.x * mAtlas[chan].subdim.y;
		AtlasResize ( chan, mAtlas[chan].num + layer );
	}
	id = mAtlas[chan].num++;
	val = getAtlasPos ( chan, id );
	return true;
}

Vector3DI Allocator::getAtlasPos ( uchar chan, uint64 id )
{
	Vector3DI p;
	Vector3DI ac = mAtlas[chan].subdim;		// axis count	
	int a2 = ac.x*ac.y;						
	p.z = int( id / a2 );		id -= uint64(p.z)*a2;
	p.y = int( id / ac.x );		id -= uint64(p.y)*ac.x;
	p.x = int( id );	
	p = p * int(mAtlas[chan].stride + (mAtlas[chan].apron << 1) ) + mAtlas[chan].apron;
	return p;
}

void Allocator::AtlasAppendLinearCPU ( uchar chan, int n, float* src )
{
	// find starting position
	int br = mAtlas[chan].stride;			// brick res
	int ssize = br*br*br*sizeof(float);		// total bytes in brick
	char* start = mAtlas[chan].cpu + br*n;

	// append data
	memcpy ( start, src, ssize );
}

extern "C" CUresult cudaCopyData ( cudaArray* dest, int dx, int dy, int dz, int dest_res, cudaArray* src, int src_res );

void Allocator::AtlasCopyTex ( uchar chan, Vector3DI val, const DataPtr& src )
{
	Vector3DI atlasres = getAtlasRes(chan);
	Vector3DI brickres = src.subdim;

	Vector3DI block ( 8, 8, 8 );
	Vector3DI grid ( int(brickres.x/block.x)+1, int(brickres.y/block.y)+1, int(brickres.z/block.z)+1 );	

	cudaCheck ( cuSurfRefSetArray( cuSurfWrite, reinterpret_cast<CUarray>(mAtlas[chan].garray), 0 ), "cuSurfRefSetArray", "AtlasCopyTex" );	
	cudaCheck ( cuTexRefSetArray ( cuSurfReadC,  reinterpret_cast<CUarray>(src.garray), CU_TRSA_OVERRIDE_FORMAT ), "cuTexRefSetArray", "AtlasCopyTex" );	
	cudaCheck ( cuTexRefSetArray ( cuSurfReadF,  reinterpret_cast<CUarray>(src.garray), CU_TRSA_OVERRIDE_FORMAT ), "cuTexRefSetArray", "AtlasCopyTex" );	

	void* args[2] = { &val, &brickres };
	switch ( mAtlas[chan].type ) {
	case T_UCHAR:	cudaCheck ( cuLaunchKernel ( cuCopyTexC, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args, NULL ), "cuCopyTexC", "AtlasCopyTex" ); break;
	case T_FLOAT:	cudaCheck ( cuLaunchKernel ( cuCopyTexF, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args, NULL ), "cuCopyTexF", "AtlasCopyTex" ); break;
	};		
}
void Allocator::AtlasCopyLinear ( uchar chan, Vector3DI offset, CUdeviceptr gpu_buf )
{
	Vector3DI atlasres = getAtlasRes(chan);
	int br = mAtlas[chan].stride;		// stride = res of brick
	Vector3DI brickres = Vector3DI(br,br,br);
	Vector3DI block ( 8, 8, 8 );
	Vector3DI grid ( int(brickres.x/block.x)+1, int(brickres.y/block.y)+1, int(brickres.z/block.z)+1 );	
	
	cudaCheck ( cuSurfRefSetArray( cuSurfWrite, reinterpret_cast<CUarray>(mAtlas[chan].garray), 0 ), "cuSurfRefSetArray", "AtlasCopyLinear" );	

	void* args[3] = { &offset, &brickres, &gpu_buf };
	switch ( mAtlas[chan].type ) {
	case T_UCHAR:	cudaCheck ( cuLaunchKernel ( cuCopyBufToTexC, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args, NULL ), "cuCopyBufToTexC", "AtlasCopyLinear" ); break;
	case T_FLOAT:	cudaCheck ( cuLaunchKernel ( cuCopyBufToTexF, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args, NULL ), "cuCopyBufToTexF", "AtlasCopyLinear" ); break;
	};		

}

void Allocator::AtlasRetrieveTexXYZ ( uchar chan, Vector3DI val, DataPtr& dest )
{
	Vector3DI atlasres = getAtlasRes(chan);
	int brickres = mAtlas[chan].stride;
	
	Vector3DI block ( 8, 8, 8 );
	Vector3DI grid ( int(brickres/block.x)+1, int(brickres/block.y)+1, int(brickres/block.z)+1 );	

	cudaCheck ( cuTexRefSetArray ( cuSurfReadF,  reinterpret_cast<CUarray>(mAtlas[chan].garray), CU_TRSA_OVERRIDE_FORMAT  ), "cuTexRefSetArray", "AtlasRetrieveTexXYZ");	 

	void* args[4] = { &val, &atlasres, &brickres, &dest.gpu };
	cudaCheck ( cuLaunchKernel ( cuRetrieveTexXYZ, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args, NULL ), "cuRetrieveDataXYZ", "AtlasRetrieveTexXYZ" );

	RetrieveMem ( dest );

	cuCtxSynchronize ();
}


void Allocator::AtlasCopyTexZYX ( uchar chan, Vector3DI val, const DataPtr& src )
{
	Vector3DI atlasres = getAtlasRes(chan);
	int brickres = src.stride;

	Vector3DI block ( 8, 8, 8 );
	Vector3DI grid ( int(brickres/block.x)+1, int(brickres/block.y)+1, int(brickres/block.z)+1 );	

	cudaCheck ( cuSurfRefSetArray( cuSurfWrite, reinterpret_cast<CUarray>(mAtlas[chan].garray), 0 ), "cuSurfRefSetArray", "AtlasCopyTexZYX" );	
	cudaCheck ( cuTexRefSetArray ( cuSurfReadF,  reinterpret_cast<CUarray>(src.garray), CU_TRSA_OVERRIDE_FORMAT  ), "cuTexRefSetArray", "AtlasCopyTexZYX" );	

	void* args[2] = { &val, &brickres };
	cudaCheck ( cuLaunchKernel ( cuCopyTexZYX, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args, NULL ), "cuCopyTexZYX", "AtlasCopyTexZYX" );	

	cuCtxSynchronize ();
}

void Allocator::AtlasCommit ( uchar chan )
{
	AtlasCommitFromCPU ( chan, (uchar*) mAtlas[chan].cpu );
}
void Allocator::AtlasCommitFromCPU ( uchar chan, uchar* src )
{
	Vector3DI res = mAtlas[chan].subdim * int(mAtlas[chan].stride + (int(mAtlas[chan].apron) << 1) );		// atlas res

	CUDA_MEMCPY3D cp = {0};
	cp.dstMemoryType = CU_MEMORYTYPE_ARRAY;
	cp.dstArray = mAtlas[chan].garray;
	cp.srcMemoryType = CU_MEMORYTYPE_HOST;
	cp.srcHost = src;
	cp.WidthInBytes = res.x*sizeof(float);
	cp.Height = res.y;
	cp.Depth = res.z;
	
	cudaCheck ( cuMemcpy3D ( &cp ), "cuMemcpy3D", "AtlasCommitFromCPU" );
}

void Allocator::AtlasFill ( uchar chan )
{
	Vector3DI atlasres = getAtlasRes(chan);	
	Vector3DI block ( 8, 8, 8 );
	Vector3DI grid ( int(atlasres.x/block.x)+1, int(atlasres.y/block.y)+1, int(atlasres.z/block.z)+1 );	
	
	cudaCheck ( cuSurfRefSetArray( cuSurfWrite, reinterpret_cast<CUarray>(mAtlas[chan].garray), 0 ), "cuSurfRefSetArray", "AtlasFill" );	
	int dsize = getSize( mAtlas[chan].type );
	void* args[2] = { &atlasres, &dsize };
	cudaCheck ( cuLaunchKernel ( cuFillTex, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args, NULL ), "cuFillTex", "AtlasFill" );
}

void Allocator::AtlasRetrieveSlice ( uchar chan, int slice, int sz, CUdeviceptr gpu_buf, uchar* cpu_dest )
{
	// transfer a 3D texture slice into gpu buffer
	Vector3DI atlasres = getAtlasRes(chan);
	Vector3DI block ( 8, 8, 1 );
	Vector3DI grid ( int(atlasres.x/block.x)+1, int(atlasres.y/block.y)+1, 1 );	
	void* args[3] = { &slice, &atlasres, &gpu_buf };

	CUDA_MEMCPY3D cp = {0};	
	cp.srcMemoryType = CU_MEMORYTYPE_ARRAY;
	cp.srcArray = mAtlas[chan].garray;	
	cp.srcZ = slice;
	cp.dstMemoryType = CU_MEMORYTYPE_HOST;
	cp.dstHost = cpu_dest;
	cp.WidthInBytes = atlasres.x * getSize( mAtlas[chan].type );
	cp.Height = atlasres.y;
	cp.Depth = 1;
	cudaCheck ( cuMemcpy3D ( &cp ), "cuMemcpy3D", "AtlasRetrieveSlice" );
}

void Allocator::AtlasWriteSlice ( uchar chan, int slice, int sz, CUdeviceptr gpu_buf, uchar* cpu_src )
{
	// transfer from gpu buffer into 3D texture slice 
	Vector3DI atlasres = getAtlasRes(chan);
	Vector3DI block ( 8, 8, 1 );
	Vector3DI grid ( int(atlasres.x/block.x)+1, int(atlasres.y/block.y)+1, 1 );		
	void* args[3] = { &slice, &atlasres, &gpu_buf };


	CUDA_MEMCPY3D cp = {0};	
	cp.dstMemoryType = CU_MEMORYTYPE_ARRAY;
	cp.dstArray = mAtlas[chan].garray;	
	cp.dstZ = slice;
	cp.srcMemoryType = CU_MEMORYTYPE_HOST;
	cp.srcHost = cpu_src;
	cp.WidthInBytes = atlasres.x * getSize( mAtlas[chan].type );
	cp.Height = atlasres.y;
	cp.Depth = 1;
	cudaCheck ( cuMemcpy3D ( &cp ), "cuMemcpy3D", "AtlasWriteSlice" );

}

void Allocator::AtlasRetrieveGL ( uchar chan, char* dest )
{
	#ifdef BUILD_OPENGL
		int w, h, d;
		glFinish ();

		glBindTexture ( GL_TEXTURE_3D, mAtlas[chan].glid );

		glGetTexLevelParameteriv(GL_TEXTURE_3D, 0, GL_TEXTURE_WIDTH, &w);
		glGetTexLevelParameteriv(GL_TEXTURE_3D, 0, GL_TEXTURE_HEIGHT, &h);
		glGetTexLevelParameteriv(GL_TEXTURE_3D, 0, GL_TEXTURE_DEPTH, &d);
				
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
		checkGL ( "glBindTexture (AtlasRetrieve)" );

		switch ( mAtlas[chan].type ) {
		case T_FLOAT:	glGetTexImage ( GL_TEXTURE_3D, 0, GL_RED, GL_FLOAT, dest );	break;
		case T_UCHAR:	glGetTexImage ( GL_TEXTURE_3D, 0, GL_RED, GL_BYTE, dest );	break;
		};
		checkGL ( "glGetTexImage (AtlasRetrieve)" );
		glBindTexture ( GL_TEXTURE_3D, 0 );	

		glFinish ();
		checkGL ( "glFinish (AtlasRetrieve)" );
	#endif
}

void Allocator::AtlasReleaseAll ()
{
	for (int n=0; n < mAtlas.size(); n++ ) {

		// Free cpu memory
		if ( mAtlas[n].cpu != 0x0 ) {
			free ( mAtlas[n].cpu );
			mAtlas[n].cpu = 0x0;
		}

		// Unregister
		if ( mAtlas[n].grsc != 0x0 ) {
			cudaCheck ( cuGraphicsUnregisterResource ( mAtlas[n].grsc ), "cuGraphicsUnregisterResource", "AtlasReleaseAll" );	
			mAtlas[n].grsc = 0x0;
		}
		// Free cuda memory
		if ( mAtlas[n].garray != 0x0 && mAtlas[n].glid == -1) {
			cudaCheck ( cuArrayDestroy ( mAtlas[n].garray ), "cuArrayDestroy", "AtlasReleaseAll" );
			mAtlas[n].garray = 0x0;
		}
		// Free opengl memory	
		#ifdef BUILD_OPENGL
			if ( mAtlas[n].glid != -1 ) {
				glDeleteTextures ( 1, (GLuint*) &mAtlas[n].glid );	
				mAtlas[n].glid = -1;
			}
		#endif
	}

	mAtlas.clear ();

	for (int n=0; n < mAtlasMap.size(); n++ )  {
		// Free cpu memory
		if ( mAtlasMap[n].cpu != 0x0 ) {
				free ( mAtlasMap[n].cpu );		
				mAtlasMap[n].cpu = 0x0;
		}
		// Free cuda memory
		if ( mAtlasMap[n].gpu != 0x0 ) {
				cudaCheck ( cuMemFree ( mAtlasMap[n].gpu ), "cuMemFree(AtlasMap)", "AtlasReleaseAll" );
				mAtlasMap[n].gpu = 0x0;
		}
	}
	mAtlasMap.clear ();
}

Vector3DI Allocator::getAtlasRes ( uchar chan )
{
	return mAtlas[chan].subdim * int(mAtlas[chan].stride + (mAtlas[chan].apron<<1));	
}
int Allocator::getAtlasBrickres ( uchar chan )
{
	return (mAtlas[chan].stride + (mAtlas[chan].apron<<1));
}

int	Allocator::getAtlasMem ()
{
	Vector3DI res = getAtlasRes(0);
	uint64 mem = getSize(mAtlas[0].type)*res.x*res.y*res.z / uint64(1042*1024); 
	return mem;	
}

void Allocator::PoolWrite ( FILE* fp, uchar grp, uchar lev )
{
	fwrite ( getPoolCPU(grp, lev), getPoolWidth(grp, lev), getPoolCnt(grp, lev), fp );
}
void Allocator::PoolRead ( FILE* fp, uchar grp, uchar lev, int cnt, int wid )
{
	char* dat = getPoolCPU (grp,lev );
	fread ( dat, wid, cnt, fp );

	mPool[grp][lev].num = cnt;
	mPool[grp][lev].stride = wid;
}

void Allocator::AtlasWrite ( FILE* fp, uchar chan )
{
	fwrite ( mAtlas[chan].cpu, mAtlas[chan].size, 1, fp );
}

void Allocator::AtlasRead ( FILE* fp, uchar chan, uint64 asize )
{
	fread ( mAtlas[chan].cpu, asize, 1, fp );
}

// Global CUDA Helpers

CUcontext gcuContext;
CUdevice gcuDevice;

bool cudaCheck ( CUresult status, char* func, char* api )
{
	if ( status != CUDA_SUCCESS ) {
		const char* stat = "";
		cuGetErrorString ( status, &stat );
		gprintf ( "CUDA ERROR: %s (func: %s, caller: %s)\n", stat, func, api  );
		gerror ();
		return false;
	} 
	return true;
}


void StartCuda ( int devid, bool verbose )
{
	// NOTES:
	// CUDA and OptiX Interop: (from Programming Guide 3.8.0)
	// - CUDA must be initialized using run-time API
	// - CUDA may call driver API, but only after context created with run-time API
	// - Once app created CUDA context, OptiX will latch onto the existing CUDA context owned by run-time API
	// - Alternatively, OptiX can create CUDA context. Then set runtime API to it. (This is how Ocean SDK sample works.)

	int version = 0;
    char name[100];
    
	int cnt = 0;
	cudaDeviceProp props;
	memset ( &props, 0, sizeof(cudaDeviceProp) );
	cudaGetDeviceCount ( &cnt );
	if ( verbose ) gprintf ( "  Device List:\n" );
	for (int n=0; n < cnt; n++ ) {
		cudaGetDeviceProperties ( &props, n );
		if ( verbose ) gprintf ( "   %d. %s, Runtime Ver: %d.%d\n", n, props.name, props.major, props.minor );		
	}	

	if ( devid == -1 ) {
		// --- Create new context with Driver API 
		cudaCheck ( cuDeviceGet( &gcuDevice, 0 ), "cuDeviceGet", "StartCuda" );
		cuDeviceGetName (name, 100, gcuDevice);			
		//cudaCheck ( cuGLCtxCreate( &gcuContext, CU_CTX_SCHED_AUTO, gcuDevice ), "cuGLCtxCreate" ); 
	}

	cuCtxGetDevice ( &gcuDevice );	
	if ( verbose ) gprintf ( "   Driver  Device: %d\n", (int) gcuDevice );	
	
	cudaGetDevice ( &devid );
	if ( verbose ) gprintf ( "   Runtime Device: %d\n", devid );

	//Increase memory limits
	size_t size_heap, size_stack;
	cudaDeviceSetLimit( cudaLimitMallocHeapSize, 20000000*sizeof(double));
	cudaDeviceSetLimit( cudaLimitStackSize,12928);
	cudaDeviceGetLimit( &size_heap, cudaLimitMallocHeapSize);
	cudaDeviceGetLimit( &size_stack, cudaLimitStackSize);
	
	size_t free, total;
	float MB = 1024.0*1024.0;
	cudaMemGetInfo ( &free, &total );
	if ( verbose ) gprintf( "   CUDA Total Mem:  %6.2f MB\n", float(total) / MB );
	if ( verbose ) gprintf( "   CUDA  Free Mem:  %6.2f MB\n", float(free) / MB );
		
}

float cudaGetFreeMem ()
{
	size_t free, total;	
	cudaMemGetInfo ( &free, &total );
	return float(free)/ (1024.0f*1024.0f);
}

DataPtr* Allocator::getPool(uchar grp, uchar lev)
{
	return &mPool[grp][lev];
}
//-----------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2016 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
// 
// Version 1.0: Rama Hoetzlein, 5/1/2017
// Version 1.1: Rama Hoetzlein, 3/25/2018
//-----------------------------------------------------------------------------

// #define GVDB_DEBUG_SYNC

#include "gvdb_allocator.h"
#include "gvdb_render.h"

#if defined(_WIN32)
#	include <windows.h>
#endif

#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda.h>

using namespace nvdb;


DataPtr::DataPtr() {
	type=T_UCHAR; usedNum=0; lastEle=0; max=0; size=0; stride=0; cpu=0; glid=0; grsc=0; gpu=0; 
	filter = 0; border = 0;
}		

Allocator::Allocator ()
{
	mbDebug = false;
	mVFBO[0] = -1;

	cudaCheck ( cuModuleLoad ( &cuAllocatorModule, CUDA_GVDB_COPYDATA_PTX ), "Allocator", "Allocator", "cuModuleLoad", CUDA_GVDB_COPYDATA_PTX, mbDebug);
		
	cudaCheck ( cuModuleGetFunction ( &cuFillTex,		cuAllocatorModule, "kernelFillTex" ), "Allocator", "Allocator", "cuModuleGetFunction", "cuFillTex",  mbDebug);
	cudaCheck ( cuModuleGetFunction ( &cuCopyTexC,		cuAllocatorModule, "kernelCopyTexC" ), "Allocator", "Allocator", "cuModuleGetFunction", "cuCopyTexC", mbDebug);
	cudaCheck ( cuModuleGetFunction ( &cuCopyTexF,		cuAllocatorModule, "kernelCopyTexF" ), "Allocator", "Allocator", "cuModuleGetFunction", "cuCopyTexF", mbDebug);
	cudaCheck ( cuModuleGetFunction ( &cuCopyBufToTexC,	cuAllocatorModule, "kernelCopyBufToTexC" ), "Allocator", "Allocator", "cuModuleGetFunction", "cuCopyBufToTexC", mbDebug);
	cudaCheck ( cuModuleGetFunction ( &cuCopyBufToTexF,	cuAllocatorModule, "kernelCopyBufToTexF" ), "Allocator", "Allocator", "cuModuleGetFunction", "cuCopyBufToTexF", mbDebug);
	cudaCheck ( cuModuleGetFunction ( &cuCopyTexZYX,	cuAllocatorModule, "kernelCopyTexZYX" ), "Allocator", "Allocator", "cuModuleGetFunction", "cuCopTexZYX", mbDebug);
	cudaCheck ( cuModuleGetFunction ( &cuRetrieveTexXYZ, cuAllocatorModule, "kernelRetrieveTexXYZ" ), "Allocator", "Allocator", "cuModuleGetFunction", "cuRetrieveTexXYZ", mbDebug);
}

Allocator::~Allocator() {
	AtlasReleaseAll();
	PoolReleaseAll();

	cudaCheck(cuModuleUnload(cuAllocatorModule), "Allocator", "~Allocator", "cuModuleUnload", "cuAllocatorModule", false);
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
	p.lastEle = 0;
	p.usedNum = 0;
	p.max = initmax;
	p.size = width * initmax;	
	p.stride = width;		
	p.cpu = 0x0;
	p.gpu = 0x0;
	
	if ( p.size == 0 ) return;		// placeholder pool, do not allocate

	// cpu allocate
	p.cpu = (char*) calloc ( p.size, 1 );
	if ( p.cpu == 0x0 ) {
		gprintf ( "ERROR: Unable to malloc %lld for pool lev %d\n", p.size, lev );
		gerror ();
	}	

	// gpu allocate
	if ( bGPU ) {
		size_t sz = p.size;
		cudaCheck ( cuMemAlloc ( &p.gpu, sz ), "Allocator", "PoolCreate", "cuMemAlloc", "", mbDebug );
		cudaCheck ( cuMemsetD8 ( p.gpu, 0, sz ), "Allocator", "PoolCreate", "cuMemsetD8", "", mbDebug );
	}
	mPool[grp].push_back ( p );
}

void Allocator::PoolCommitAll()
{
	//std::cout << "Commit all\n";
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

void Allocator::PoolCommit ( int grp, int lev )
{
	DataPtr* p = &mPool[grp][lev];
	//std::cout << grp << " " << lev << " " << p->gpu << " " << p->lastEle * p->stride << std::endl;
	cudaCheck ( cuMemcpyHtoD ( p->gpu, p->cpu, p->lastEle * p->stride ), "Allocator", "PoolCommit", "cuMemcpyHtoD", "", mbDebug );	
}


void Allocator::PoolFetchAll()
{
	for (int grp=0; grp < MAX_POOL; grp++) 
		for (int lev=0; lev < mPool[grp].size() ; lev++ )	
			PoolFetch ( grp, lev );
}


void Allocator::PoolFetch(int grp, int lev )
{
	DataPtr* p = &mPool[grp][lev];
	cudaCheck ( cuMemcpyDtoH ( p->cpu,  p->gpu, p->lastEle * p->stride ), "Allocator", "PoolFetch", "cuMemcpyDtoH", "", mbDebug);	
}

void Allocator::PoolCommitAtlasMap ()
{
	DataPtr* p;
	for (int n=0; n < mAtlasMap.size(); n++ ) {
		if ( mAtlasMap[n].cpu != 0x0 ) {
			p = &mAtlasMap[n];
			cudaCheck ( cuMemcpyHtoD ( p->gpu, p->cpu, p->lastEle * p->stride ), "Allocator", "PoolCommitAtlasMap", "cuMemcpyHtoD", "", mbDebug);
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
				cudaCheck ( cuMemFree ( mPool[grp][lev].gpu ), "Allocator", "PoolReleaseAll", "cuMemFree", "", mbDebug);
		}


	// release pool structure	
	for (int grp=0; grp < MAX_POOL; grp++) 
		mPool[grp].clear ();
}


uint64 Allocator::PoolAlloc ( uchar grp, uchar lev, bool bGPU )
{
	if ( lev >= mPool[grp].size() ) return ID_UNDEFL;
	DataPtr* p = &mPool[grp][lev];
	
	if ( p->lastEle >= p->max ) {
		// Expand pool
		p->max *= 2;
		p->size = p->stride * p->max;
		if ( p->cpu != 0x0 ) {
			char* new_cpu = (char*) calloc ( p->size, 1 );
			memcpy ( new_cpu, p->cpu, p->stride*p->lastEle );
			free ( p->cpu );
			p->cpu = new_cpu;
		}
		if ( p->gpu != 0x0 ) {
			size_t sz = p->size;	
			CUdeviceptr new_gpu;
			cudaCheck ( cuMemAlloc ( &new_gpu, sz ), "Allocator", "PoolAlloc", "cuMemAlloc", "", mbDebug);
			cudaCheck ( cuMemsetD8 ( new_gpu, 0, sz ), "Allocator", "PoolAlloc", "cuMemsetD8", "", mbDebug);
			cudaCheck ( cuMemcpy ( new_gpu, p->gpu, p->stride*p->lastEle), "Allocator", "PoolAlloc", "cuMemcpy", "", mbDebug);
			cudaCheck ( cuMemFree ( p->gpu ), "Allocator", "PoolAlloc", "cuMemFree", "", mbDebug );
			p->gpu = new_gpu;
		}
	}
	// Return new element
	p->lastEle++;	
	p->usedNum++;
	return Elem(grp,lev, (p->lastEle-1) );
}

void Allocator::PoolEmptyAll ()
{
	// clear pool data (do not free)
	for (int grp=0; grp < MAX_POOL; grp++) 
		for (int lev=0; lev < mPool[grp].size(); lev++ ) 
		{
			mPool[grp][lev].usedNum = 0;	
			mPool[grp][lev].lastEle = 0;	
		}
}

uint64 Allocator::getPoolMem ()
{
	uint64 sz = 0;
	for (int grp=0; grp < MAX_POOL; grp++) 
		for (int lev=0; lev < mPool[grp].size(); lev++ ) 
			sz += mPool[grp][lev].size;
	return sz / uint64(1024*1024);
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

void Allocator::CreateMemLinear ( DataPtr& p, char* dat, int stride, uint64 cnt, bool bCPU, bool bAllocHost )
{
	//std::cout << p.lastEle << std::endl;
	p.alloc = this;
	p.lastEle = cnt; 
	p.usedNum = cnt;
	p.max = cnt; 
	p.stride = stride;
	p.size = (uint64) cnt * (uint64) stride;
	p.subdim = Vector3DI(0,0,0);

	if ( dat==0x0 ) {
		if ( bCPU ) {
			if ( p.cpu != 0x0 ) free (p.cpu);		// release previous
			
			if (bAllocHost)
				cudaCheck ( cuMemAllocHost ( (void**)&p.cpu, sizeof(float) * 3 ), "Allocator", "CreateMemLinear", "cuMemAllocHost", "", mbDebug);
			else 
				p.cpu = (char*) malloc ( p.size );		// create on cpu 
		}
	} else {
		p.cpu = dat;							// get from user
	}
	if ( p.gpu != 0x0 ) cudaCheck ( cuMemFree (p.gpu), "Allocator", "CreateMemLinear", "cuMemFree", "", mbDebug);
	cudaCheck ( cuMemAlloc ( &p.gpu, p.size ), "Allocator", "CreateMemLinear", "cuMemAlloc", "", mbDebug);

	if ( dat!=0x0 ) CommitMem ( p );			// transfer from user
}

void Allocator::FreeMemLinear ( DataPtr& p )
{
	if ( p.cpu != 0x0 ) free (p.cpu);
	if ( p.gpu != 0x0 ) cudaCheck ( cuMemFree (p.gpu), "Allocator", "FreeMemLinear", "cuMemFree", "", mbDebug);
	p.cpu = 0x0;
	p.gpu = 0x0;
}

void Allocator::RetrieveMem ( DataPtr& p)
{
	cudaCheck ( cuMemcpyDtoH ( p.cpu, p.gpu, p.size), "Allocator", "RetrieveMem", "cuMemcpyDtoH", "", mbDebug);
}

void Allocator::CommitMem ( DataPtr& p)
{
	cudaCheck ( cuMemcpyHtoD ( p.gpu, p.cpu, p.size), "Allocator", "CommitMem", "cuMemcpyHtoD", "", mbDebug);
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
			gchkGL ( "glGenTextures (AtlasCreate)" );
			glBindTexture ( GL_TEXTURE_3D, p.glid );					
			glPixelStorei ( GL_PACK_ALIGNMENT, 4 );	
			glPixelStorei ( GL_UNPACK_ALIGNMENT, 4 );
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);		
			gchkGL ( "glBindTexture (AtlasCreate)" );
	
			switch ( dtype ) {
			case T_UCHAR:	glTexImage3D ( GL_TEXTURE_3D, 0, GL_R8,		res.x, res.y, res.z, 0, GL_RED, GL_UNSIGNED_BYTE, 0);	break;
			case T_UCHAR4:	glTexImage3D ( GL_TEXTURE_3D, 0, GL_RGBA8,	res.x, res.y, res.z, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);	break;
			case T_FLOAT:	glTexImage3D ( GL_TEXTURE_3D, 0, GL_R32F,	res.x, res.y, res.z, 0, GL_RED, GL_FLOAT, 0);			break;
			};
			gchkGL ( "glTexImage3D (AtlasCreate)" );

			// preserve old texture data
			if ( preserve > 0 && old_glid != -1 ) {
				const uint64 bytesPerPlane = static_cast<uint64>(res.x)
					* static_cast<uint64>(res.y)
					* static_cast<uint64>(getSize(dtype));
				const Vector3DI src (res.x, res.y, static_cast<int>(preserve / bytesPerPlane));	 // src amount to copy 				
				glCopyImageSubData ( old_glid, GL_TEXTURE_3D, 0, 0,0,0, p.glid, GL_TEXTURE_3D, 0, 0,0,0, src.x, src.y, src.z );
			}
			if ( old_glid != -1 ) {
				GLuint id = old_glid;
				glDeleteTextures ( 1, &id );
			}

			// CUDA-GL interop for CUarray
			if ( p.grsc != 0 ) cudaCheck ( cuGraphicsUnregisterResource ( p.grsc ), "Allocator", "AllocateTextureGPU", "cuGraphicsUnregisterResource", "", mbDebug);
			cudaCheck ( cuGraphicsGLRegisterImage ( &p.grsc, p.glid, GL_TEXTURE_3D, CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST ), "Allocator", "AllocateTextureGPU", "cuGraphicsRegisterImage", "", mbDebug);
			gchkGL ( "cuGraphicsGLRegisterImage" );
			cudaCheck ( cuGraphicsMapResources(1, &p.grsc, 0), "Allocator", "AllocateTextureGPU", "cuGraphicsMapResources", "", mbDebug);
			cudaCheck ( cuGraphicsSubResourceGetMappedArray ( &p.garray, p.grsc, 0, 0 ), "Allocator", "AllocateTextureGPU", "cuGraphicsSubResourceGetMappedArray", "", mbDebug);
		
			CUDA_RESOURCE_DESC resDesc = {};
			resDesc.resType = CU_RESOURCE_TYPE_ARRAY;
			resDesc.res.array.hArray = p.garray;

		
			cudaCheck(cuSurfObjectCreate(&p.surf_obj, &resDesc), "Allocator", "AllocateTextureGPU", "cuSurfObjectCreate", "", mbDebug);


			CUDA_RESOURCE_DESC surfReadDesc = {};
			surfReadDesc.resType = CU_RESOURCE_TYPE_ARRAY;
			surfReadDesc.res.array.hArray = reinterpret_cast<CUarray>(p.garray);

			CUDA_TEXTURE_DESC texReadDesc = {};
			texReadDesc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
			texReadDesc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
			texReadDesc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;

			texReadDesc.filterMode = CU_TR_FILTER_MODE_POINT;

			cudaCheck(cuTexObjectCreate(&p.tex_obj, &surfReadDesc, &texReadDesc, nullptr), "Allocator", "AllocateTextureGPU", "cuGraphicsUnmapResources", "", mbDebug);

			
			cudaCheck ( cuGraphicsUnmapResources(1, &p.grsc, 0), "Allocator", "AllocateTextureGPU", "cuGraphicsUnmapResources", "", mbDebug);
		

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
			cudaCheck ( cuArray3DCreate( &p.garray, &desc), "Allocator", "AllocateTextureGPU", "cuArray3DCreate", "", mbDebug);
            
            CUDA_RESOURCE_DESC resDesc = {};
            resDesc.resType = CU_RESOURCE_TYPE_ARRAY;
            resDesc.res.array.hArray = p.garray;

            cudaCheck(cuSurfObjectCreate(&p.surf_obj, &resDesc), "Allocator", "AllocateTextureGPU", "cuSurfObjectCreate", "", mbDebug);

            CUDA_RESOURCE_DESC surfReadDesc = {};
            surfReadDesc.resType = CU_RESOURCE_TYPE_ARRAY;
            surfReadDesc.res.array.hArray = reinterpret_cast<CUarray>(p.garray);

            CUDA_TEXTURE_DESC texReadDesc = {};
            texReadDesc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
            texReadDesc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
            texReadDesc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;

            texReadDesc.filterMode = CU_TR_FILTER_MODE_POINT;

            cudaCheck(cuTexObjectCreate(&p.tex_obj, &surfReadDesc, &texReadDesc, nullptr), "Allocator", "AllocateTextureGPU", "cuGraphicsUnmapResources", "", mbDebug);
            
            if ( preserve > 0 && old_array != 0 ) {
				
				// Clear channel to 0
				Vector3DI block ( 8, 8, 8 );
				Vector3DI grid ( int(res.x/block.x)+1, int(res.y/block.y)+1, int(res.z/block.z)+1 );	

				int dsize = getSize( p.type );
				void* args[3] = { &res, &dsize, &p.surf_obj };
				cudaCheck ( cuLaunchKernel ( cuFillTex, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, mStream, args, NULL ), "Allocator", "AllocateTextureGPU", "cuLaunch", "cuFillTex", mbDebug);

				// Copy over preserved data
				CUDA_MEMCPY3D cp = {0};
				cp.dstMemoryType = CU_MEMORYTYPE_ARRAY;
				cp.dstArray = p.garray;
				cp.srcMemoryType = CU_MEMORYTYPE_ARRAY;
				cp.srcArray = old_array;
				cp.WidthInBytes = res.x * getSize(dtype);
				cp.Height = res.y;
				cp.Depth = preserve / (res.x*res.y*getSize(dtype));	  // amount to copy (preserve)
				if ( cp.Depth < desc.Depth )
				   cudaCheck ( cuMemcpy3D ( &cp ), "Allocator", "AllocateTextureGPU", "cuMemcpy3D", "preserve", mbDebug);
			}

		} else {
			p.garray = 0;
		}
		if ( old_array != 0 ) cudaCheck ( cuArrayDestroy ( old_array ), "Allocator", "AllocateTextureGPU", "cuArrayDestroy", "", mbDebug);
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
	q.usedNum = q.max;
	q.lastEle = q.max;
	q.stride = stride;
	q.size = stride * q.max;					// list of mapping structs			
	if ( q.cpu != 0x0 ) free ( q.cpu );
	q.cpu = (char*) malloc ( q.size );			// cpu allocate		
			
	size_t sz = q.size;							// gpu allocate
	if ( q.gpu != 0x0 ) cudaCheck ( cuMemFree ( q.gpu ), "Allocator", "AllocateAtlasMap", "cuMemFree", "", mbDebug);
	cudaCheck ( cuMemAlloc ( &q.gpu, q.size ), "Allocator", "AllocateAtlasMap", "cuMemAlloc", "", mbDebug );

	mAtlasMap[0] = q;
}

bool Allocator::TextureCreate ( uchar chan, uchar dtype, Vector3DI res, bool bCPU, bool bGL )
{
	DataPtr p;
	p.alloc = this;
	p.type = dtype;	
	p.usedNum = 0;
	p.lastEle = 0;
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

	cudaCheck ( cuCtxSynchronize(), "Allocator", "TextureCreate", "cuCtxSynchronize", "", mbDebug);

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
	p.usedNum = 0;
	p.lastEle = 0;
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

	cudaCheck ( cuCtxSynchronize(), "Allocator", "AtlasCreate", "cuCtxSynchronize", "", mbDebug);

	return true;
}

void Allocator::AtlasSetFilter ( uchar chan, int filter, int border )
{
	if ( chan < mAtlas.size() ) {
		mAtlas[chan].filter = filter;
		mAtlas[chan].border = border;
	}
}

void Allocator::AtlasSetNum ( uchar chan, int n )
{
	mAtlas[chan].usedNum = n;
	mAtlas[chan].lastEle = n;
}

bool Allocator::AtlasResize ( uchar chan, int cx, int cy, int cz )
{
	DataPtr p = mAtlas[chan];
	int leafdim = static_cast<int>(p.stride);
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

void Allocator::CopyChannel(int chanDst, int chanSrc)
{
	DataPtr pDst = mAtlas[chanDst];
	DataPtr pSrc = mAtlas[chanSrc];

	int leafdim = static_cast<int>(pSrc.stride);	
	Vector3DI axisres;
	Vector3DI axiscnt = pSrc.subdim;	// number of bricks on each axis	
	uint64 preserve = pSrc.size;		// previous size of atlas (# bytes to preserve)

	axisres = axiscnt * int(leafdim + pSrc.apron * 2);		

	CUDA_MEMCPY3D cp = {0};
	cp.dstMemoryType = CU_MEMORYTYPE_ARRAY;
	cp.dstArray = pDst.garray;
	cp.srcMemoryType = CU_MEMORYTYPE_ARRAY;
	cp.srcArray = pSrc.garray;
	cp.WidthInBytes = axisres.x * getSize(pSrc.type);
	cp.Height = axisres.y;
	cp.Depth = preserve / (axisres.x*axisres.y*getSize(pSrc.type));	  // amount to copy (preserve)
	
	cudaCheck ( cuMemcpy3D ( &cp ), "Allocator", "CopyChannel", "cuMemcpy3D", "", mbDebug);
}

bool Allocator::AtlasResize ( uchar chan, uint64 max_leaf )
{
	DataPtr p = mAtlas[chan];
	int leafdim = static_cast<int>(p.stride);			// dimensions of brick
	Vector3DI axisres;
	Vector3DI axiscnt = p.subdim;	// number of bricks on each axis	
	uint64 preserve = p.size;		// previous size of atlas (# bytes to preserve)

	// Expand Z-axis of atlas
	axiscnt.z = int(ceil ( max_leaf / float(axiscnt.x*axiscnt.y) ));		// expand number of bricks along Z-axis

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

void Allocator::AllocateNeighbors( uint64 cnt )
{
	if ( cnt <= mNeighbors.max ) return;

	uint64 brks = getPoolTotalCnt(0, 0);
	mNeighbors.usedNum = brks;
	mNeighbors.max = brks;
	mNeighbors.stride = sizeof(int) * 8;
	mNeighbors.size = mNeighbors.max * mNeighbors.stride;

	if (mNeighbors.cpu != 0x0) free(mNeighbors.cpu);
	mNeighbors.cpu = (char*) malloc(mNeighbors.size);

	if ( mNeighbors.gpu != 0x0) cudaCheck(cuMemFree(mNeighbors.gpu), "Allocator", "AllocateNeighbors", "cuMemFree", "", mbDebug);
	cudaCheck(cuMemAlloc(&mNeighbors.gpu, mNeighbors.size), "Allocator", "AllocateNeighbors", "cuMemAlloc", "", mbDebug);
}

void Allocator::CommitNeighbors()
{
	cudaCheck(cuMemcpyHtoD( mNeighbors.gpu, mNeighbors.cpu, mNeighbors.size), "Allocator", "CommitNeighbors", "cuMemcpyHtoD", "", mbDebug);
}

char* Allocator::getAtlasMapNode ( uchar chan, Vector3DI val )
{
	int leafres = static_cast<int>(mAtlas[chan].stride + (mAtlas[chan].apron << 1));			// leaf res
	Vector3DI axiscnt = mAtlas[chan].subdim;	
	Vector3DI i = Vector3DI(val.x/leafres, val.y/leafres, val.z/leafres);	// get brick index
	int id = (i.z*axiscnt.y + i.y) * axiscnt.x + i.x;						// get brick id	
	return mAtlasMap[0].cpu + id * mAtlasMap[0].stride;						// get mapping node for brick
}

void Allocator::AtlasEmptyAll ()
{
	for (int n=0; n < mAtlas.size(); n++ )
	{
		mAtlas[n].usedNum = 0;
		mAtlas[n].lastEle = 0;
	}
}

bool Allocator::AtlasAlloc ( uchar chan, Vector3DI& val )
{
	uint64 id;	
	if ( mAtlas[chan].lastEle >= mAtlas[chan].max ) {
		int layer = mAtlas[chan].subdim.x * mAtlas[chan].subdim.y;
		AtlasResize ( chan, mAtlas[chan].lastEle + layer );
	}
	id = mAtlas[chan].lastEle++;
	mAtlas[chan].usedNum++;
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
	uint64 br = mAtlas[chan].stride;			// brick res
	uint64 ssize = br*br*br*sizeof(float);		// total bytes in brick
	char* start = mAtlas[chan].cpu + br*static_cast<uint64>(n);

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

	void* args[3] = { &val, &brickres, (void*)&src.surf_obj };
	switch ( mAtlas[chan].type ) {
	case T_UCHAR:	cudaCheck ( cuLaunchKernel ( cuCopyTexC, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args, NULL ), "Allocator", "AtlasCopyTex", "cuLaunch", "cuCopyTexC", mbDebug); break;
	case T_FLOAT:	cudaCheck ( cuLaunchKernel ( cuCopyTexF, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args, NULL ), "Allocator", "AtlasCopyTex", "cuLaunch", "cuCopyTexF", mbDebug); break;
	};		
}
void Allocator::AtlasCopyLinear ( uchar chan, Vector3DI offset, CUdeviceptr gpu_buf )
{
	Vector3DI atlasres = getAtlasRes(chan);
	int br = static_cast<int>(mAtlas[chan].stride);

	Vector3DI brickres = Vector3DI(br,br,br);
	Vector3DI block ( 8, 8, 8 );
	Vector3DI grid ( int(brickres.x/block.x)+1, int(brickres.y/block.y)+1, int(brickres.z/block.z)+1 );	

	void* args[4] = { &offset, &brickres, &gpu_buf, &mAtlas[chan].surf_obj };
	switch ( mAtlas[chan].type ) {
	case T_UCHAR:	cudaCheck ( cuLaunchKernel ( cuCopyBufToTexC, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args, NULL ), "Allocator", "AtlasCopyLinear", "cuLaunch", "cuCopyBufToTexC", mbDebug); break;
	case T_FLOAT:	cudaCheck ( cuLaunchKernel ( cuCopyBufToTexF, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args, NULL ), "Allocator", "AtlasCopyLinear", "cuLaunch", "cuCopyBufToTexF", mbDebug); break;
	};		

}

void Allocator::AtlasRetrieveTexXYZ ( uchar chan, Vector3DI val, DataPtr& dest )
{
	Vector3DI atlasres = getAtlasRes(chan);
	int br = static_cast<int>(mAtlas[chan].stride);
	
	Vector3DI brickres = Vector3DI(br, br, br);
	Vector3DI block ( 8, 8, 8 );
	Vector3DI grid = (brickres + block - Vector3DI(1, 1, 1)) / block;

	void* args[4] = { &val, &brickres, &dest.gpu, &mAtlas[chan].surf_obj };
	cudaCheck ( cuLaunchKernel ( cuRetrieveTexXYZ, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args, NULL ), "Allocator", "AtlasRetrieveXYZ", "cuLaunch", "cuRetrieveTexXYZ", mbDebug);

	RetrieveMem ( dest );

	cuCtxSynchronize ();
}


void Allocator::AtlasCopyTexZYX ( uchar chan, Vector3DI val, const DataPtr& src )
{
	Vector3DI atlasres = getAtlasRes(chan);
	Vector3DI brickres = src.subdim;

	Vector3DI block ( 8, 8, 8 );
	Vector3DI grid = (brickres + block - Vector3DI(1, 1, 1)) / block;

	void* args[4] = { &val, &brickres, (void*)&src.surf_obj, &mAtlas[chan].surf_obj };
	cudaCheck(cuLaunchKernel(cuCopyTexZYX, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, NULL, args, NULL), "Allocator", "AtlasCopyTexZYX", "cuLaunch", "cuCopyTexZYX", mbDebug);

	cuCtxSynchronize();
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
	
	cudaCheck ( cuMemcpy3D ( &cp ), "Allocator", "AtlasCommitFromCPU", "cuMemcpy3D", "", mbDebug);
}

void Allocator::AtlasFill ( uchar chan )
{
	Vector3DI atlasres = getAtlasRes(chan);	
	Vector3DI block ( 8, 8, 8 );
	Vector3DI grid ( int(atlasres.x/block.x)+1, int(atlasres.y/block.y)+1, int(atlasres.z/block.z)+1 );	

	int dsize = getSize( mAtlas[chan].type );
	void* args[3] = { &atlasres, &dsize , &mAtlas[chan].surf_obj };
	cudaCheck ( cuLaunchKernel ( cuFillTex, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, mStream, args, NULL ), "Allocator", "AtlasCommitFromCPU", "cuLaunch", "cuFillTex", mbDebug);
}

void Allocator::AtlasRetrieveSlice ( uchar chan, int slice, int sz, CUdeviceptr gpu_buf, uchar* cpu_dest )
{
	// transfer a 3D texture slice into gpu buffer
	Vector3DI atlasres = getAtlasRes(chan);
	Vector3DI block ( 8, 8, 1 );
	Vector3DI grid ( int(atlasres.x/block.x)+1, int(atlasres.y/block.y)+1, 1 );	

	CUDA_MEMCPY3D cp = {0};	
	cp.srcMemoryType = CU_MEMORYTYPE_ARRAY;
	cp.srcArray = mAtlas[chan].garray;	
	cp.srcZ = slice;
	cp.dstMemoryType = CU_MEMORYTYPE_HOST;
	cp.dstHost = cpu_dest;
	cp.WidthInBytes = atlasres.x * getSize( mAtlas[chan].type );
	cp.Height = atlasres.y;
	cp.Depth = 1;
	cudaCheck ( cuMemcpy3D ( &cp ), "Allocator", "AtlasRetrieveSlice", "cuMemcpy3D", "", mbDebug);
}

void Allocator::AtlasWriteSlice ( uchar chan, int slice, int sz, CUdeviceptr gpu_buf, uchar* cpu_src )
{
	// transfer from gpu buffer into 3D texture slice 
	Vector3DI atlasres = getAtlasRes(chan);
	Vector3DI block ( 8, 8, 1 );
	Vector3DI grid ( int(atlasres.x/block.x)+1, int(atlasres.y/block.y)+1, 1 );		

	CUDA_MEMCPY3D cp = {0};	
	cp.dstMemoryType = CU_MEMORYTYPE_ARRAY;
	cp.dstArray = mAtlas[chan].garray;	
	cp.dstZ = slice;
	cp.srcMemoryType = CU_MEMORYTYPE_HOST;
	cp.srcHost = cpu_src;
	cp.WidthInBytes = atlasres.x * getSize( mAtlas[chan].type );
	cp.Height = atlasres.y;
	cp.Depth = 1;
	cudaCheck ( cuMemcpy3D ( &cp ), "Allocator", "AtlasWriteSlice", "cuMemcpy3D", "", mbDebug);

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
		gchkGL ( "glBindTexture (AtlasRetrieve)" );

		switch ( mAtlas[chan].type ) {
		case T_FLOAT:	glGetTexImage ( GL_TEXTURE_3D, 0, GL_RED, GL_FLOAT, dest );	break;
		case T_UCHAR:	glGetTexImage ( GL_TEXTURE_3D, 0, GL_RED, GL_BYTE, dest );	break;
		};
		gchkGL ( "glGetTexImage (AtlasRetrieve)" );
		glBindTexture ( GL_TEXTURE_3D, 0 );	

		glFinish ();
		gchkGL ( "glFinish (AtlasRetrieve)" );
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

		// Destroy Surf/Tex Objects
		if (mAtlas[n].surf_obj != 0x0) {
			cudaCheck(cuSurfObjectDestroy(mAtlas[n].surf_obj), "Allocator", "AtlasReleaseAll", "cuSurfObjectDestroy", "", mbDebug);
			mAtlas[n].surf_obj = 0x0;
		}

		// Destroy Surf/Tex Objects
		if (mAtlas[n].tex_obj != 0x0) {
			cudaCheck(cuTexObjectDestroy(mAtlas[n].tex_obj), "Allocator", "AtlasReleaseAll", "cuTexObjectDestroy", "", mbDebug);
			mAtlas[n].tex_obj = 0x0;
		}

		// Unregister
		if ( mAtlas[n].grsc != 0x0 ) {
			cudaCheck ( cuGraphicsUnregisterResource ( mAtlas[n].grsc ), "Allocator", "AtlasReleaseAll", "cuGraphicsUnregisterResource", "", mbDebug);
			mAtlas[n].grsc = 0x0;
		}
		// Free cuda memory
		if ( mAtlas[n].garray != 0x0 && mAtlas[n].glid == -1) {
			cudaCheck ( cuArrayDestroy ( mAtlas[n].garray ), "Allocator", "AtlasReleaseAll", "cuArrayDestroy", "", mbDebug);
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
				cudaCheck ( cuMemFree ( mAtlasMap[n].gpu ), "Allocator", "AtlasReleaseAll", "cuMemFree", "AtlasMap", mbDebug);
				mAtlasMap[n].gpu = 0x0;
		}
	}
	mAtlasMap.clear ();
}

// Dimension of entire atlas, including aprons
Vector3DI Allocator::getAtlasRes ( uchar chan )
{
	return mAtlas[chan].subdim * int(mAtlas[chan].stride + (mAtlas[chan].apron<<1));	
}
// Dimension of entire atlas, without aprons
Vector3DI Allocator::getAtlasPackres(uchar chan)
{
	return mAtlas[chan].subdim * int(mAtlas[chan].stride);
}

// Res of single brick, including apron
int Allocator::getAtlasBrickres ( uchar chan )
{
	return static_cast<int>(mAtlas[chan].stride + (mAtlas[chan].apron<<1));
}
// Res of single brick, without apron
int Allocator::getAtlasBrickwid (uchar chan)
{
	return static_cast<int>(mAtlas[chan].stride);
}

uint64 Allocator::getAtlasMem ()
{
	Vector3DI res = getAtlasRes(0);
	uint64 mem = getSize(mAtlas[0].type)*res.x*res.y*res.z / uint64(1024*1024); 
	return mem;	
}

void Allocator::PoolWrite ( FILE* fp, uchar grp, uchar lev )
{
	fwrite ( getPoolCPU(grp, lev), getPoolWidth(grp, lev), getPoolTotalCnt(grp, lev), fp );
}
void Allocator::PoolRead ( FILE* fp, uchar grp, uchar lev, int cnt, int wid )
{
	char* dat = getPoolCPU (grp,lev );
	fread ( dat, wid, cnt, fp );

	mPool[grp][lev].usedNum = cnt;
	mPool[grp][lev].lastEle = cnt;
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
#include <assert.h>

bool cudaCheck ( CUresult launch_stat, const char* obj, const char* method, const char* apicall, const char* arg, bool bDebug)
{
	CUresult kern_stat = CUDA_SUCCESS;
	
	if ( bDebug ) {
		kern_stat = cuCtxSynchronize();	
	}		
	if ( kern_stat != CUDA_SUCCESS || launch_stat != CUDA_SUCCESS ) {
		const char* launch_statmsg = "";
		const char* kern_statmsg = "";
		cuGetErrorString ( launch_stat, &launch_statmsg );
		cuGetErrorString ( kern_stat, &kern_statmsg);
		gprintf("GVDB CUDA ERROR:\n");
		gprintf("  Launch status: %s\n", launch_statmsg );
		gprintf("  Kernel status: %s\n", kern_statmsg );
		gprintf("  Caller: %s::%s\n", obj, method);
		gprintf("  Call:   %s\n", apicall);
		gprintf("  Args:   %s\n", arg);

		if (bDebug) {
			gprintf("  Generating assert so you can examine call stack.\n");
			assert(0);		// debug - trigger break (see call stack)
		} else {
			gerror();		// exit - return 0
		}		
		return false;
	} 
	return true;
}

void StartCuda ( int devsel, CUcontext ctxsel, CUdevice& dev, CUcontext& ctx, CUstream* strm, bool verbose)
{
	// NOTES:
	// CUDA and OptiX Interop: (from Programming Guide 3.8.0)
	// - CUDA must be initialized using run-time API
	// - CUDA may call driver API, but only after context created with run-time API
	// - Once app created CUDA context, OptiX will latch onto the existing CUDA context owned by run-time API
	// - Alternatively, OptiX can create CUDA context. Then set runtime API to it. (This is how Ocean SDK sample works.)

	int version = 0;
    char name[128];
    
	int cnt = 0;
	CUdevice dev_id;	
	cuInit(0);

	//--- List devices
	cuDeviceGetCount ( &cnt );
	if (cnt == 0) {
		gprintf("ERROR: No CUDA devices found.\n");
		dev = NULL; ctx = NULL;
		gerror();
		return;
	}	
	if (verbose) gprintf ( "  Device List:\n" );
	for (int n=0; n < cnt; n++ ) {
		cuDeviceGet(&dev_id, n);
		cuDeviceGetName ( name, 128, dev_id);

		int pi;
		cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH, dev_id);
		if (verbose) gprintf("Max. texture3D width: %d\n", pi);
		cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT, dev_id);
		if (verbose) gprintf("Max. texture3D height: %d\n", pi);
		cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH, dev_id);
		if (verbose) gprintf("Max. texture3D depth: %d\n", pi);

		if (verbose) gprintf ( "   %d. %s\n", n, name );
	}

	if (devsel == GVDB_DEV_CURRENT) {
		//--- Get currently active context 
		cudaCheck(cuCtxGetCurrent(&ctx), "(global)", "StartCuda", "cuCtxGetCurrent", "GVDB_DEV_CURRENT", false );
		cudaCheck(cuCtxGetDevice(&dev), "(global)", "StartCuda", "cuCtxGetDevice", "GVDB_DEV_CURRENT", false );
	}
	if (devsel == GVDB_DEV_EXISTING) {
		//--- Use existing context passed in
		ctx = ctxsel;	
		cudaCheck(cuCtxSetCurrent(ctx), "(global)", "StartCuda", "cuCtxSetCurrent", "GVDB_DEV_EXISTING", false );
		cudaCheck(cuCtxGetDevice(&dev), "(global)", "StartCuda", "cuCtxGetDevice", "GVDB_DEV_EXISTING", false );
	}
	if (devsel == GVDB_DEV_FIRST
		|| devsel >= cnt) { // Fallback to OpenGL device if additional GPUs not found
		// Get the CUDA device being used for OpenGL, so that GL examples work
		// correctly on multi-GPU systems (e.g. SLI or NVLink)
		unsigned int cudaGLDeviceCount = 0;
		int cudaGLDevices[1];
		cuGLGetDevices(&cudaGLDeviceCount, cudaGLDevices, 1, CU_GL_DEVICE_LIST_NEXT_FRAME);

		if (cudaGLDeviceCount == 0) {
			// Fall back to device 0 if no CUDA-GL devices were found
			devsel = 0;
		}
		else {
			// Use the first element in this list (which may not be device 0)
			devsel = cudaGLDevices[0];
		}
	}

	if (devsel >= 0) {		
		//--- Create new context with Driver API 
		cudaCheck(cuDeviceGet(&dev, devsel), "(global)", "StartCuda", "cuDeviceGet", "", false );
		cudaCheck(cuCtxCreate(&ctx, CU_CTX_SCHED_AUTO, dev), "(global)", "StartCuda", "cuCtxCreate", "", false );
	}
	cuDeviceGetName(name, 128, dev);
	if (verbose) gprintf("   Using Device: %d, %s, Context: %p\n", (int) dev, name, (void*) ctx);
	
	cuCtxSetCurrent( NULL );
	cuCtxSetCurrent( ctx );
		
}

Vector3DF cudaGetMemUsage ()
{
	Vector3DF mem;
	size_t free, total;	
	cuMemGetInfo ( &free, &total );
	free /= 1024ULL * 1024ULL;		// MB
	total /= 1024ULL * 1024ULL;
	mem.x = static_cast<float>(total - free);	// used
	mem.y = static_cast<float>(free);
	mem.z = static_cast<float>(total);
	return mem;
}


DataPtr* Allocator::getPool(uchar grp, uchar lev)
{
	return &mPool[grp][lev];
}
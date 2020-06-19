//----------------------------------------------------------------------------------
//
// FLUIDS v.3 - SPH Fluid Simulator for CPU and GPU
// Copyright (C) 2012-2013. Rama Hoetzlein, http://fluids3.com
//
// BSD 3-clause:
// Redistribution and use in source and binary forms, with or without modification, 
// are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this 
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this 
//    list of conditions and the following disclaimer in the documentation and/or 
//    other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors may 
//    be used to endorse or promote products derived from this software without specific 
//   prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
// SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT 
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) 
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR 
// TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//----------------------------------------------------------------------------------

#include <assert.h>
#include <stdio.h>
#include <cuda.h>	
#include "cutil_math.h"			// cutil32.lib

#include "app_perf.h"

#include "main.h"
#include "fluid_system.h"
#include "nv_gui.h"

#include <GL/glew.h>

extern bool gProfileRend;

#define EPSILON			0.00001f			// for collision detection

#define SCAN_BLOCKSIZE		512				// must match value in fluid_system_cuda.cu

// #define FLUID_INTEGRITY						// debugging, enable this to check fluid integrity 

bool cuCheck (CUresult launch_stat, char* method, char* apicall, char* arg, bool bDebug)
{
	CUresult kern_stat = CUDA_SUCCESS;

	if (bDebug) {
		kern_stat = cuCtxSynchronize();
	}
	if (kern_stat != CUDA_SUCCESS || launch_stat != CUDA_SUCCESS) {
		const char* launch_statmsg = "";
		const char* kern_statmsg = "";
		cuGetErrorString(launch_stat, &launch_statmsg);
		cuGetErrorString(kern_stat, &kern_statmsg);
		nvprintf("FLUID SYSTEM, CUDA ERROR:\n");
		nvprintf("  Launch status: %s\n", launch_statmsg);
		nvprintf("  Kernel status: %s\n", kern_statmsg);
		nvprintf("  Caller: FluidSystem::%s\n", method);
		nvprintf("  Call:   %s\n", apicall);
		nvprintf("  Args:   %s\n", arg);

		if (bDebug) {
			nvprintf("  Generating assert to examine call stack.\n");
			assert(0);		// debug - trigger break (see call stack)
		}
		else {
			nverror();		// exit - return 0
		}
		return false;
	}
	return true;

}

FluidSystem::FluidSystem ()
{
	mNumPoints = 0;
	mMaxPoints = 0;
	mbRecord = false;
	mbRecordBricks = false;
	mSelected = -1;
	m_Frame = 0;
	m_Module = 0;
	m_Thresh = 0;	
	m_NeighborTable = 0x0;
	m_NeighborDist = 0x0;		
	for (int n=0; n < FUNC_MAX; n++ ) m_Func[n] = (CUfunction) -1;
	m_Toggle [ PDEBUG ]		=	false;
	m_Toggle [ PUSE_GRID ]	=	false;
	m_Toggle [ PPROFILE ]	=	false;
	m_Toggle [ PCAPTURE ]   =	false;
}

FluidSystem::~FluidSystem() {
	ClearNeighborTable();
	if (mSaveNdx != 0x0) free(mSaveNdx);
	if (mSaveCnt != 0x0) free(mSaveCnt);
	if (mSaveNeighbors != 0x0)	free(mSaveNeighbors);

	if (m_Module != 0) {
		cuCheck(cuModuleUnload(m_Module), "~FluidSystem()", "cuModuleUnload", "m_Module", mbDebug);
	}
}

void FluidSystem::LoadKernel ( int fid, std::string func )
{
	char cfn[512];		strcpy ( cfn, func.c_str() );

	if ( m_Func[fid] == (CUfunction) -1 )
		cuCheck ( cuModuleGetFunction ( &m_Func[fid], m_Module, cfn ), "LoadKernel", "cuModuleGetFunction", cfn, mbDebug );	
}

// Must have a CUDA context to initialize
void FluidSystem::Initialize ()
{
	cuCheck ( cuModuleLoad ( &m_Module, "fluid_system_cuda.ptx" ), "LoadKernel", "cuModuleLoad", "fluid_system_cuda.ptx", mbDebug);

	LoadKernel ( FUNC_INSERT,			"insertParticles" );
	LoadKernel ( FUNC_COUNTING_SORT,	"countingSortFull" );
	LoadKernel ( FUNC_QUERY,			"computeQuery" );
	LoadKernel ( FUNC_COMPUTE_PRESS,	"computePressure" );
	LoadKernel ( FUNC_COMPUTE_FORCE,	"computeForce" );
	LoadKernel ( FUNC_ADVANCE,			"advanceParticles" );
	LoadKernel ( FUNC_EMIT,				"emitParticles" );
	LoadKernel ( FUNC_RANDOMIZE,		"randomInit" );
	LoadKernel ( FUNC_SAMPLE,			"sampleParticles" );
	LoadKernel ( FUNC_FPREFIXSUM,		"prefixSum" );
	LoadKernel ( FUNC_FPREFIXFIXUP,		"prefixFixup" );

	size_t len = 0;
	cuCheck ( cuModuleGetGlobal ( &cuFBuf, &len,		m_Module, "fbuf" ),		"LoadKernel", "cuModuleGetGlobal", "cuFBuf", mbDebug);
	cuCheck ( cuModuleGetGlobal ( &cuFTemp, &len,		m_Module, "ftemp" ),	"LoadKernel", "cuModuleGetGlobal", "cuFTemp", mbDebug);
	cuCheck ( cuModuleGetGlobal ( &cuFParams, &len,	m_Module, "fparam" ),		"LoadKernel", "cuModuleGetGlobal", "cuFParams", mbDebug);

	// Clear all buffers
	memset ( &m_Fluid, 0,		sizeof(FBufs) );
	memset ( &m_FluidTemp, 0,	sizeof(FBufs) );
	memset ( &m_FParams, 0,		sizeof(FParams) );
	//m_Param [ PMODE ]		= RUN_VALIDATE;			// debugging
	m_Param [ PMODE ]		= RUN_GPU_FULL;		
	m_Param [ PEXAMPLE ]	= 2;
	m_Param [ PGRID_DENSITY ] = 2.0;
	m_Param [ PNUM ]		= 65536 * 128;

	// Allocate the sim parameters
	AllocateBuffer ( FPARAMS,		sizeof(FParams),	0,	1,					 GPU_SINGLE, CPU_OFF );
}

void FluidSystem::Start ( int num )
{
	#ifdef TEST_PREFIXSUM
		TestPrefixSum ( 16*1024*1024 );		
		exit(-2);
	#endif

	m_Time = 0;

	ClearNeighborTable ();
	mNumPoints = 0;			// reset count
	
	SetupDefaultParams ();	
	SetupExampleParams ();	
	m_Param[PNUM] = (float) num;	// maximum number of points
	mMaxPoints = num;

	m_Param [PGRIDSIZE] = 2*m_Param[PSMOOTHRADIUS] / m_Param[PGRID_DENSITY];	

	// Setup stuff
	SetupKernels ();
	
	SetupSpacing ();

	SetupGrid ( m_Vec[PVOLMIN], m_Vec[PVOLMAX], m_Param[PSIMSCALE], m_Param[PGRIDSIZE], 1.0f );	// Setup grid
		
	FluidSetupCUDA ( mMaxPoints, m_GridSrch, *(int3*)& m_GridRes, *(float3*)& m_GridSize, *(float3*)& m_GridDelta, *(float3*)& m_GridMin, *(float3*)& m_GridMax, m_GridTotal, 0 );

	UpdateParams();

	// Allocate data
	AllocateParticles( mMaxPoints );

	AllocateGrid();

	// Create the particles (after allocate)
	SetupAddVolume(m_Vec[PINITMIN], m_Vec[PINITMAX], m_Param[PSPACING], 0.1f, (int)m_Param[PNUM]);		// increases mNumPoints

	TransferToCUDA ();		 // Initial transfer
}

void FluidSystem::UpdateParams ()
{
	// Update Params on GPU
	Vector3DF grav = m_Vec[PPLANE_GRAV_DIR] * m_Param[PGRAV];
	FluidParamCUDA (  m_Param[PSIMSCALE], m_Param[PSMOOTHRADIUS], m_Param[PRADIUS], m_Param[PMASS], m_Param[PRESTDENSITY],
					*(float3*)& m_Vec[PBOUNDMIN], *(float3*)& m_Vec[PBOUNDMAX], m_Param[PEXTSTIFF], m_Param[PINTSTIFF], 
					m_Param[PVISC], m_Param[PEXTDAMP], m_Param[PFORCE_MIN], m_Param[PFORCE_MAX], m_Param[PFORCE_FREQ], 
					m_Param[PGROUND_SLOPE], grav.x, grav.y, grav.z, m_Param[PACCEL_LIMIT], m_Param[PVEL_LIMIT], 
					(int) m_Vec[PEMIT_RATE].x );
}

void FluidSystem::SetParam (int p, float v )
{
	m_Param[p] = v;	
	UpdateParams ();
}

void FluidSystem::SetVec ( int p, Vector3DF v )	
{ 
	m_Vec[p] = v; 
	UpdateParams ();
}

void FluidSystem::Exit ()
{
	// Free fluid buffers
	for (int n=0; n < MAX_BUF; n++ ) {
		if ( m_Fluid.bufC(n) != 0x0 )
			free ( m_Fluid.bufC(n) );
	}

	//cudaExit ();
}

void FluidSystem::AllocateBuffer ( int buf_id, int stride, int cpucnt, int gpucnt, int gpumode, int cpumode )
{
	if (cpumode == CPU_YES) {
		char* src_buf = m_Fluid.bufC(buf_id);
		char* dest_buf = (char*) malloc(cpucnt*stride);
		if (src_buf != 0x0) {
			memcpy(dest_buf, src_buf, cpucnt*stride);
			free(src_buf);
		}
		m_Fluid.setBuf(buf_id, dest_buf);
	}
	if (gpumode == GPU_SINGLE || gpumode == GPU_DUAL )	{
		if (m_Fluid.gpuptr(buf_id) != 0x0) cuCheck(cuMemFree(m_Fluid.gpu(buf_id)), "AllocateBuffer", "cuMemFree", "Fluid.gpu", mbDebug);
		cuCheck( cuMemAlloc(m_Fluid.gpuptr(buf_id), stride*gpucnt), "AllocateBuffer", "cuMemAlloc", "Fluid.gpu", mbDebug);
	}
	if (gpumode == GPU_TEMP || gpumode == GPU_DUAL ) {
		if (m_FluidTemp.gpuptr(buf_id) != 0x0) cuCheck(cuMemFree(m_FluidTemp.gpu(buf_id)), "AllocateBuffer", "cuMemFree", "FluidTemp.gpu", mbDebug);
		cuCheck( cuMemAlloc(m_FluidTemp.gpuptr(buf_id), stride*gpucnt), "AllocateBuffer", "cuMemAlloc", "FluidTemp.gpu", mbDebug);
	}
}

// Allocate particle memory
void FluidSystem::AllocateParticles ( int cnt )
{
	AllocateBuffer ( FPOS,		sizeof(Vector3DF),	cnt,	m_FParams.szPnts,	GPU_DUAL, CPU_YES );
	AllocateBuffer ( FCLR,		sizeof(uint),		cnt,	m_FParams.szPnts,	GPU_DUAL, CPU_YES );
	AllocateBuffer ( FVEL,		sizeof(Vector3DF),	cnt,	m_FParams.szPnts,	GPU_DUAL, CPU_YES );
	AllocateBuffer ( FVEVAL,	sizeof(Vector3DF),	cnt,	m_FParams.szPnts,	GPU_DUAL, CPU_YES );
	AllocateBuffer ( FAGE,		sizeof(unsigned short), cnt,m_FParams.szPnts,	GPU_DUAL, CPU_YES );
	AllocateBuffer ( FPRESS,	sizeof(float),		cnt,	m_FParams.szPnts,	GPU_DUAL, CPU_YES );
	AllocateBuffer ( FDENSITY,	sizeof(float),		cnt, 	m_FParams.szPnts,	GPU_DUAL, CPU_YES );
	AllocateBuffer ( FFORCE,	sizeof(Vector3DF),	cnt,	m_FParams.szPnts,	GPU_DUAL, CPU_YES );
	AllocateBuffer ( FCLUSTER,	sizeof(uint),		cnt,	m_FParams.szPnts,	GPU_DUAL, CPU_YES );
	AllocateBuffer ( FGCELL,	sizeof(uint),		cnt,	m_FParams.szPnts,	GPU_DUAL, CPU_YES );
	AllocateBuffer ( FGNDX,		sizeof(uint),		cnt,	m_FParams.szPnts,	GPU_DUAL, CPU_YES );
	AllocateBuffer ( FGNEXT,	sizeof(uint),		cnt,	m_FParams.szPnts,	GPU_DUAL, CPU_YES );
	AllocateBuffer ( FNBRNDX,	sizeof(uint),		cnt,	m_FParams.szPnts,	GPU_DUAL, CPU_YES );
	AllocateBuffer ( FNBRCNT,	sizeof(uint),		cnt,	m_FParams.szPnts,	GPU_DUAL, CPU_YES );
	AllocateBuffer ( FSTATE,	sizeof(uint),		cnt,	m_FParams.szPnts,	GPU_DUAL, CPU_YES );
	
	// Update GPU access pointers
	cuCheck( cuMemcpyHtoD(cuFBuf, &m_Fluid, sizeof(FBufs)),			"AllocateParticles", "cuMemcpyHtoD", "cuFBuf", mbDebug);
	cuCheck( cuMemcpyHtoD(cuFTemp, &m_FluidTemp, sizeof(FBufs)),	"AllocateParticles", "cuMemcpyHtoD", "cuFTemp", mbDebug);
	cuCheck( cuMemcpyHtoD(cuFParams, &m_FParams, sizeof(FParams)),  "AllocateParticles", "cuMemcpyHtoD", "cuFParams", mbDebug);
	cuCheck(cuCtxSynchronize(), "AllocateParticles", "cuCtxSynchronize", "", mbDebug );

	m_Param[PSTAT_PMEM] = 68.0f * 2 * cnt;

	// Allocate auxiliary buffers (prefix sums)
	int blockSize = SCAN_BLOCKSIZE << 1;
	int numElem1 = m_GridTotal;
	int numElem2 = int ( numElem1 / blockSize ) + 1;
	int numElem3 = int ( numElem2 / blockSize ) + 1;

	AllocateBuffer ( FAUXARRAY1,	sizeof(uint),		0,	numElem2, GPU_SINGLE, CPU_OFF );
	AllocateBuffer ( FAUXSCAN1,	sizeof(uint),		0,	numElem2, GPU_SINGLE, CPU_OFF );
	AllocateBuffer ( FAUXARRAY2,	sizeof(uint),		0,	numElem3, GPU_SINGLE, CPU_OFF );
	AllocateBuffer ( FAUXSCAN2,	sizeof(uint),		0,	numElem3, GPU_SINGLE, CPU_OFF );
}

void FluidSystem::AllocateGrid()
{
	// Allocate grid
	int cnt = m_GridTotal;
	m_FParams.szGrid = (m_FParams.gridBlocks * m_FParams.gridThreads);
	AllocateBuffer ( FGRID,		sizeof(uint),		mMaxPoints,	m_FParams.szPnts,	GPU_SINGLE, CPU_YES );    // # grid elements = number of points
	AllocateBuffer ( FGRIDCNT,	sizeof(uint),		cnt,	m_FParams.szGrid,	GPU_SINGLE, CPU_YES );
	AllocateBuffer ( FGRIDOFF,	sizeof(uint),		cnt,	m_FParams.szGrid,	GPU_SINGLE, CPU_YES );
	AllocateBuffer ( FGRIDACT,	sizeof(uint),		cnt,	m_FParams.szGrid,	GPU_SINGLE, CPU_YES );

	// Update GPU access pointers
	cuCheck(cuMemcpyHtoD(cuFBuf, &m_Fluid, sizeof(FBufs)), "AllocateGrid", "cuMemcpyHtoD", "cuFBuf", mbDebug);
	cuCheck(cuCtxSynchronize(), "AllocateParticles", "cuCtxSynchronize", "", mbDebug);
}

int FluidSystem::AddParticle ()
{
	if ( mNumPoints >= mMaxPoints ) return -1;
	int n = mNumPoints;
	(m_Fluid.bufV3(FPOS) + n)->Set ( 0,0,0 );
	(m_Fluid.bufV3(FVEL) + n)->Set ( 0,0,0 );
	(m_Fluid.bufV3(FVEVAL) + n)->Set ( 0,0,0 );
	(m_Fluid.bufV3(FFORCE) + n)->Set ( 0,0,0 );
	*(m_Fluid.bufF(FPRESS) + n) = 0;
	*(m_Fluid.bufF(FDENSITY) + n) = 0;
	*(m_Fluid.bufI(FGNEXT) + n) = -1;
	*(m_Fluid.bufI(FCLUSTER)  + n) = -1;
	*(m_Fluid.bufF(FSTATE) + n ) = (float) rand();
	
	mNumPoints++;
	return n;
}

void FluidSystem::SetupAddVolume ( Vector3DF min, Vector3DF max, float spacing, float offs, int total )
{
	Vector3DF pos;
	int p;
	float dx, dy, dz;
	int cntx, cntz;
	cntx = (int) ceil( (max.x-min.x-offs) / spacing );
	cntz = (int) ceil( (max.z-min.z-offs) / spacing );
	int cnt = cntx * cntz;
	int c2;
	
	min += offs;
	max -= offs;

	dx = max.x-min.x;
	dy = max.y-min.y;
	dz = max.z-min.z;

	Vector3DF rnd;
		
	c2 = cnt/2;
	for (pos.y = min.y; pos.y <= max.y; pos.y += spacing ) {	
		for (int xz=0; xz < cnt; xz++ ) {
			
			pos.x = min.x + (xz % int(cntx))*spacing;
			pos.z = min.z + (xz / int(cntx))*spacing;
			p = AddParticle ();

			if ( p != -1 ) {
				rnd.Random ( 0, spacing, 0, spacing, 0, spacing );					
				*(m_Fluid.bufV3(FPOS)+p) = pos + rnd;
				
				Vector3DF clr ( (pos.x-min.x)/dx, 0, (pos.z-min.z)/dz );				
				clr *= 0.8; clr += 0.2;				
				clr.Clamp (0, 1.0);				
				m_Fluid.bufI(FCLR) [p] = COLORA( clr.x, clr.y, clr.z, 1); 				
				// = COLORA( 0.25, +0.25 + (y-min.y)*.75/dy, 0.25 + (z-min.z)*.75/dz, 1);  // (x-min.x)/dx
			}
		}
	}		
}

void FluidSystem::AddEmit ( float spacing )
{
	int p;
	Vector3DF dir;
	Vector3DF pos;
	float ang_rand, tilt_rand;
	float rnd = m_Vec[PEMIT_RATE].y * 0.15f;	
	int x = (int) sqrt(m_Vec[PEMIT_RATE].y);

	for ( int n = 0; n < m_Vec[PEMIT_RATE].y; n++ ) {
		ang_rand = (float(rand()*2.0f/RAND_MAX) - 1.0f) * m_Vec[PEMIT_SPREAD].x;
		tilt_rand = (float(rand()*2.0f/RAND_MAX) - 1.0f) * m_Vec[PEMIT_SPREAD].y;
		dir.x = cos ( ( m_Vec[PEMIT_ANG].x + ang_rand) * DEGtoRAD ) * sin( ( m_Vec[PEMIT_ANG].y + tilt_rand) * DEGtoRAD ) * m_Vec[PEMIT_ANG].z;
		dir.y = sin ( ( m_Vec[PEMIT_ANG].x + ang_rand) * DEGtoRAD ) * sin( ( m_Vec[PEMIT_ANG].y + tilt_rand) * DEGtoRAD ) * m_Vec[PEMIT_ANG].z;
		dir.z = cos ( ( m_Vec[PEMIT_ANG].y + tilt_rand) * DEGtoRAD ) * m_Vec[PEMIT_ANG].z;
		pos = m_Vec[PEMIT_POS];
		pos.x += spacing * (n/x);
		pos.y += spacing * (n%x);
		
		p = AddParticle ();
		*(m_Fluid.bufV3(FPOS)+n) = pos;
		*(m_Fluid.bufV3(FVEL)+n) = dir;
		*(m_Fluid.bufV3(FVEVAL)+n) = dir;
		*(m_Fluid.bufI(FAGE)+n) = 0;
		*(m_Fluid.bufI(FCLR)+n) = COLORA ( m_Time/10.0, m_Time/5.0, m_Time /4.0, 1 );
	}
}

void FluidSystem::ValidateCUDA ()
{
	int valid = 0, bad = 0, badl;
	// CPU results	
	uint*	cpu_gridcnt =	m_Fluid.bufI(FGRIDCNT);
	uint*	cpu_gridoff =	(uint*) malloc ( m_GridTotal * sizeof(uint) );	
	uint*	cpu_grid =		(uint*) malloc ( NumPoints() * sizeof(uint) );	
	// GPU results
	uint*	gpu_gcell =		(uint*) malloc ( NumPoints() * sizeof(uint) );
	uint*	gpu_gndx =		(uint*) malloc ( NumPoints() * sizeof(uint) );	
	uint*	gpu_gridcnt =	(uint*) malloc ( m_GridTotal * sizeof(uint) );
	uint*	gpu_gridoff =	(uint*) malloc ( m_GridTotal * sizeof(uint) );
	Vector3DF* gpu_pos =	(Vector3DF*) malloc ( NumPoints() * sizeof(Vector3DF) );	
	
	int n=0, c=0;

	// Insert Particles. Determines grid cells, and cpu grid counts (m_GridCnt)
	nvprintf ( "\nVALIDATE SIM\n" );
	nvprintf ( "Insert particles:\n" );
	InsertParticles ();					
	TransferToCUDA ();
	InsertParticlesCUDA ( gpu_gcell, gpu_gndx, gpu_gridcnt );
	for (n=0, c=0; n < NumPoints() && c < 20; n++, c++)					// show top 20
		nvprintf ( "p: %d, CPU cell: %d, GPU cell: %d\n", n, m_Fluid.bufI(FGCELL)[n], gpu_gcell[n] );	
	for (n=0, valid=0, bad=0; n < NumPoints(); n++)						// validate all
		if ( m_Fluid.bufI(FGCELL)[n]==gpu_gcell[n] ) valid++; else bad++; 
	nvprintf ( "Insert particles. VALID %d, BAD %d.  \n", valid, bad );

	// Generate test data
	// Test data for PrefixSums. Set the SCAN_BLOCKSIZE to 8
	/* srand ( 614 );
	int num_test = 512;
	for (int n=0; n < num_test; n++) {
		cpu_gridcnt[n] = rand() * 256 / RAND_MAX;
		gpu_gridcnt[n] = cpu_gridcnt[n];
	}
	cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FGRIDCNT),   gpu_gridcnt,		num_test*sizeof(uint) ), 	"Memcpy" );		
	cuCheck( cuCtxSynchronize(), "cuCtxSync" ); */

	// Compare grid counts 
	nvprintf ( "Grid Counts:\n" );	
	int cpu_cnt = 0, gpu_cnt = 0;
	for (n=0, c=0; n < m_GridTotal && c < 20; n++)						// show top 20
		if ( cpu_gridcnt[n]!=0 ) 
			{ nvprintf ( "cell: %d, CPU cnt: %d, GPU cnt: %d\n", n, (int) cpu_gridcnt[n], (int) gpu_gridcnt[n] ); c++; }
	for (n=0, valid=0, bad=0; n < m_GridTotal; n++)	{					// validate all
		cpu_cnt += cpu_gridcnt[n];
		gpu_cnt += gpu_gridcnt[n];
		if ( cpu_gridcnt[n]==gpu_gridcnt[n] ) valid++; else bad++; 
	}
	nvprintf ( "Grid Counts. VALID %d, BAD %d.  \n", valid, bad );	
	nvprintf ( "Grid Counts. Particles: %d, CPU: %d, GPU: %d\n", NumPoints(), cpu_cnt, gpu_cnt );

	// Prefix Sum. Determine grid offsets.
	PrefixSumCellsCUDA ( gpu_gridoff, 1 );		// Prefix Sum on GPU
	nvprintf ( "Prefix Sum:\n" );
	int sum = 0;
	for (n=0; n < m_GridTotal; n++) {		// Prefix Sum on CPU		
		cpu_gridoff[n] = sum;	
		sum += cpu_gridcnt[n];
	}	
	for (n=0, c=0; n < m_GridTotal && c < 512; n++)						// show top 20
		if ( cpu_gridcnt[n]!=0 ) 
			{ nvprintf ( "cell: %d, offsets CPU: %d (+%d), GPU: %d\n", n, (int) cpu_gridoff[n], (int) cpu_gridcnt[n], (int) gpu_gridoff[n] ); c++; }

	for (n=0, valid=0, bad=0; n < m_GridTotal; n++)								// validate all
		if ( cpu_gridoff[n]==gpu_gridoff[n] ) {
			valid++; 
		} else {
			bad++; 
			nvprintf ( "BAD- cell: %d, offsets CPU: %d (+%d), GPU: %d\n", n, (int) cpu_gridoff[n], (int) cpu_gridcnt[n], (int) gpu_gridoff[n] ); 
		}

	nvprintf ( "Prefix Sum. VALID %d, BAD %d.  \n", valid, bad );	

	// Counting Sort. Reinsert particles to grid.
	CountingSortFullCUDA ( gpu_pos );
	nvprintf ( "Counting Sort:\n" );	
	int gc, pi;
	for (n=0, c=0, valid=0, bad=0; n < NumPoints(); n++) {
		gc = m_Fluid.bufI(FGCELL)[n];				// grid cell of particle
		if ( gc != GRID_UNDEF ) {					
			if ( gpu_gcell[n] != gc ) nvprintf ( "ERROR: %d, cell does not match. CPU %d, GPU %d\n", gc, gpu_gcell[n] );
			pi = cpu_gridoff[gc] + gpu_gndx[n];		// sort location = cell offset + cell index		
			Vector3DF cpu_pnt = m_Fluid.bufV3(FPOS)[n];
			Vector3DF gpu_pnt = gpu_pos[ pi ];	// get _sorted_ particle at its _expected_ sort location
			badl = bad;
			if ( cpu_pnt.x==gpu_pnt.x && cpu_pnt.y==gpu_pnt.y && cpu_pnt.z==gpu_pnt.z ) valid++; else bad++;
			if ( ++c < 20 || bad != badl) nvprintf ( "ndx: %d, cell: %d, CPU pos: %2.1f,%2.1f,%2.1f, GPU pos: %2.1f,%2.1f,%2.1f\n", n, gc, cpu_pnt.x, cpu_pnt.y, cpu_pnt.z, gpu_pnt.x, gpu_pnt.y, gpu_pnt.z ); 
		} 		
	}		
	nvprintf ( "Counting Sort. VALID %d, BAD %d.\n\n", valid, bad );	
	
	free ( gpu_pos );
	free ( gpu_gridoff );
	free ( gpu_gridcnt );
	free ( gpu_gndx );
	free ( gpu_gcell );
	
	free ( cpu_grid );
	free ( cpu_gridoff );	
}
	
void FluidSystem::EmitParticles ()
{
	if ( m_Vec[PEMIT_RATE].x > 0 && (++m_Frame) % (int) m_Vec[PEMIT_RATE].x == 0 ) {
		float ss = m_Param [ PDIST ] / m_Param[ PSIMSCALE ];		// simulation scale (not Schutzstaffel)
		AddEmit ( ss ); 
	}
}


void FluidSystem::Run ()
{
	// Clear sim timers
	m_Param[ PTIME_INSERT ] = 0.0;
	m_Param[ PTIME_SORT ] = 0.0;
	m_Param[ PTIME_COUNT ] = 0.0;
	m_Param[ PTIME_PRESS ] = 0.0;
	m_Param[ PTIME_FORCE ] = 0.0;
	m_Param[ PTIME_ADVANCE ] = 0.0;

	// Run	
	#ifdef TEST_VALIDATESIM
		m_Param[PMODE] = RUN_VALIDATE;
	#endif	

	switch ( (int) m_Param[PMODE] ) {
	case RUN_SEARCH:	
		InsertParticles ();				// Insert into grid
		FindNbrsGrid ();				// Find neighbors		
		break;	
	case RUN_CPU_SLOW:					// CPU naive, no acceleration
		InsertParticles (); 
		//ComputePressureSlow ();
		//ComputeForceSlow ();
		Advance ();
		break;
	case RUN_CPU_GRID:					// CPU fast, GRID-accelerated 
		InsertParticles ();
		ComputePressureGrid ();
		ComputeForceGrid ();
		Advance ();
		break;
	case RUN_VALIDATE:					// GPU Validation
		ValidateCUDA();
		break;
	case RUN_GPU_FULL:					// Full CUDA pathway, GRID-accelerted GPU, /w deep copy sort		
		InsertParticlesCUDA ( 0x0, 0x0, 0x0 );		
		PrefixSumCellsCUDA ( 0x0, 1 );		
		CountingSortFullCUDA ( 0x0 );
		#ifdef FLUID_INTEGRITY
			IntegrityCheck();
		#endif				
		ComputePressureCUDA();		
		ComputeForceCUDA ();			
		AdvanceCUDA ( m_Time, m_DT, m_Param[PSIMSCALE] );							
		//EmitParticlesCUDA ( m_Time, (int) m_Vec[PEMIT_RATE].x );					
		TransferFromCUDA ();	// return for rendering			
		break;	
	};

	AdvanceTime ();
}

void FluidSystem::AdvanceTime ()
{
	m_Time += m_DT;
	
	m_Frame += m_FrameRange.z;

	if ( m_Frame > m_FrameRange.y && m_FrameRange.y != -1 ) {
	
		m_Frame = m_FrameRange.x;		
		mbRecord = false;
		mbRecordBricks = false;
		m_Toggle[ PCAPTURE ] = false;
		
		nvprintf ( "Exiting.\n" );
		exit ( 1 );
	}
}

void FluidSystem::DebugPrintMemory ()
{
	int psize = 4*sizeof(Vector3DF) + sizeof(uint) + sizeof(unsigned short) + 2*sizeof(float) + sizeof(int) + sizeof(int)+sizeof(int);
	int gsize = 2*sizeof(int);
	int nsize = sizeof(int) + sizeof(float);
		
	nvprintf ( "MEMORY:\n");
	nvprintf ( "  Fluid (size):			%d bytes\n", sizeof(Fluid) );
	nvprintf ( "  Particles:              %d, %f MB (%f)\n", mNumPoints, (psize*mNumPoints)/1048576.0, (psize*mMaxPoints)/1048576.0);
	nvprintf ( "  Acceleration Grid:      %d, %f MB\n",	   m_GridTotal, (gsize*m_GridTotal)/1048576.0 );
	nvprintf ( "  Acceleration Neighbors: %d, %f MB (%f)\n", m_NeighborNum, (nsize*m_NeighborNum)/1048576.0, (nsize*m_NeighborMax)/1048576.0 );
	
}

void FluidSystem::DrawDomain ()
{
	Vector3DF min, max;
	min = m_Vec[PVOLMIN];
	max = m_Vec[PVOLMAX];
	
	glColor3f ( 0.0, 0.0, 1.0 );
	glBegin ( GL_LINES );
	glVertex3f ( min.x, min.y, min.z );	glVertex3f ( max.x, min.y, min.z );
	glVertex3f ( min.x, max.y, min.z );	glVertex3f ( max.x, max.y, min.z );
	glVertex3f ( min.x, min.y, min.z );	glVertex3f ( min.x, max.y, min.z );
	glVertex3f ( max.x, min.y, min.z );	glVertex3f ( max.x, max.y, min.z );
	glEnd ();
}

void FluidSystem::Advance ()
{
	Vector3DF norm, z;
	Vector3DF dir, accel;
	Vector3DF vnext;
	Vector3DF bmin, bmax;
	Vector4DF clr;
	float adj;
	float AL, AL2, SL, SL2, ss, radius;
	float stiff, damp, speed, diff; 
	
	AL = m_Param[PACCEL_LIMIT];	AL2 = AL*AL;
	SL = m_Param[PVEL_LIMIT];	SL2 = SL*SL;
	
	stiff = m_Param[PEXTSTIFF];
	damp = m_Param[PEXTDAMP];
	radius = m_Param[PRADIUS];
	bmin = m_Vec[PBOUNDMIN];
	bmax = m_Vec[PBOUNDMAX];
	ss = m_Param[PSIMSCALE];

	// Get particle buffers
	Vector3DF*	ppos =		m_Fluid.bufV3(FPOS);
	Vector3DF*	pvel =		m_Fluid.bufV3(FVEL);
	Vector3DF*	pveleval =	m_Fluid.bufV3(FVEVAL);
	Vector3DF*	pforce =	m_Fluid.bufV3(FFORCE);
	uint*		pclr =		m_Fluid.bufI(FCLR);
	float*		ppress =	m_Fluid.bufF(FPRESS);
	float*		pdensity =	m_Fluid.bufF(FDENSITY);

	// Advance each particle
	for ( int n=0; n < NumPoints(); n++ ) {

		if ( m_Fluid.bufI(FGCELL)[n] == GRID_UNDEF) continue;

		// Compute Acceleration		
		accel = *pforce;
		accel *= m_Param[PMASS];
	
		// Boundary Conditions
		// Y-axis walls
		diff = radius - ( ppos->y - (bmin.y+ (ppos->x-bmin.x)*m_Param[PGROUND_SLOPE] ) )*ss;
		if (diff > EPSILON ) {			
			norm.Set ( -m_Param[PGROUND_SLOPE], 1.0f - m_Param[PGROUND_SLOPE], 0 );
			adj = stiff * diff - damp * (float) norm.Dot ( *pveleval );
			accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
		}		
		diff = radius - ( bmax.y - ppos->y )*ss;
		if (diff > EPSILON) {
			norm.Set ( 0, -1, 0 );
			adj = stiff * diff - damp * (float) norm.Dot ( *pveleval );
			accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
		}		
		
		// X-axis walls
		if ( !m_Toggle[PWRAP_X] ) {
			diff = radius - ( ppos->x - (bmin.x + (sin(m_Time*m_Param[PFORCE_FREQ])+1)*0.5f * m_Param[PFORCE_MIN]) )*ss;	
			//diff = 2 * radius - ( p->pos.x - min.x + (sin(m_Time*10.0)-1) * m_Param[FORCE_XMIN_SIN] )*ss;	
			if (diff > EPSILON ) {
				norm.Set ( 1.0, 0, 0 );
				adj = (m_Param[ PFORCE_MIN ]+1) * stiff * diff - damp * (float) norm.Dot ( *pveleval ) ;
				accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;					
			}

			diff = radius - ( (bmax.x - (sin(m_Time*m_Param[PFORCE_FREQ])+1)*0.5f* m_Param[PFORCE_MAX]) - ppos->x )*ss;	
			if (diff > EPSILON) {
				norm.Set ( -1, 0, 0 );
				adj = (m_Param[ PFORCE_MAX ]+1) * stiff * diff - damp * (float) norm.Dot ( *pveleval );
				accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
			}
		}

		// Z-axis walls
		diff = radius - ( ppos->z - bmin.z )*ss;			
		if (diff > EPSILON) {
			norm.Set ( 0, 0, 1 );
			adj = stiff * diff - damp * (float) norm.Dot ( *pveleval );
			accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
		}
		diff = radius - ( bmax.z - ppos->z )*ss;
		if (diff > EPSILON) {
			norm.Set ( 0, 0, -1 );
			adj = stiff * diff - damp * (float) norm.Dot ( *pveleval );
			accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
		}
		

		// Wall barrier
		if ( m_Toggle[PWALL_BARRIER] ) {
			diff = 2 * radius - ( ppos->x - 0 )*ss;					
			if (diff < 2*radius && diff > EPSILON && fabs(ppos->y) < 3 && ppos->z < 10) {
				norm.Set ( 1.0, 0, 0 );
				adj = 2*stiff * diff - damp * (float) norm.Dot ( *pveleval ) ;	
				accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;					
			}
		}
		
		// Levy barrier
		if ( m_Toggle[PLEVY_BARRIER] ) {
			diff = 2 * radius - ( ppos->x - 0 )*ss;					
			if (diff < 2*radius && diff > EPSILON && fabs(ppos->y) > 5 && ppos->z < 10) {
				norm.Set ( 1.0, 0, 0 );
				adj = 2*stiff * diff - damp * (float) norm.Dot ( *pveleval ) ;	
				accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;					
			}
		}
		// Drain barrier
		if ( m_Toggle[PDRAIN_BARRIER] ) {
			diff = 2 * radius - ( ppos->z - bmin.z-15 )*ss;
			if (diff < 2*radius && diff > EPSILON && (fabs(ppos->x)>3 || fabs(ppos->y)>3) ) {
				norm.Set ( 0, 0, 1);
				adj = stiff * diff - damp * (float) norm.Dot ( *pveleval );
				accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
			}
		}

		// Plane gravity
		accel += m_Vec[PPLANE_GRAV_DIR] * m_Param[PGRAV];

		// Point gravity
		if ( m_Vec[PPOINT_GRAV_POS].x > 0 && m_Param[PGRAV] > 0 ) {
			norm.x = ( ppos->x - m_Vec[PPOINT_GRAV_POS].x );
			norm.y = ( ppos->y - m_Vec[PPOINT_GRAV_POS].y );
			norm.z = ( ppos->z - m_Vec[PPOINT_GRAV_POS].z );
			norm.Normalize ();
			norm *= m_Param[PGRAV];
			accel -= norm;
		}

		// Acceleration limiting 
		speed = accel.x*accel.x + accel.y*accel.y + accel.z*accel.z;
		if ( speed > AL2 ) {
			accel *= AL / sqrt(speed);
		}		

		// Velocity limiting 
		speed = pvel->x*pvel->x + pvel->y*pvel->y + pvel->z*pvel->z;
		if ( speed > SL2 ) {
			speed = SL2;
			(*pvel) *= SL / sqrt(speed);
		}		

		// Leapfrog Integration ----------------------------
		vnext = accel;							
		vnext *= m_DT;
		vnext += *pvel;						// v(t+1/2) = v(t-1/2) + a(t) dt

		*pveleval = *pvel;
		*pveleval += vnext;
		*pveleval *= 0.5;					// v(t+1) = [v(t-1/2) + v(t+1/2)] * 0.5		used to compute forces later
		*pvel = vnext;
		vnext *= m_DT/ss;
		*ppos += vnext;						// p(t+1) = p(t) + v(t+1/2) dt

		/*if ( m_Param[PCLR_MODE]==1.0 ) {
			adj = fabs(vnext.x)+fabs(vnext.y)+fabs(vnext.z) / 7000.0;
			adj = (adj > 1.0) ? 1.0 : adj;
			*pclr = COLORA( 0, adj, adj, 1 );
		}
		if ( m_Param[PCLR_MODE]==2.0 ) {
			float v = 0.5 + ( *ppress / 1500.0); 
			if ( v < 0.1 ) v = 0.1;
			if ( v > 1.0 ) v = 1.0;
			*pclr = COLORA ( v, 1-v, 0, 1 );
		}*/
		if ( speed > SL2*0.1f) {
			adj = SL2*0.1f;
			clr.fromClr ( *pclr );
			clr += Vector4DF( 2/255.0f, 2/255.0f, 2/255.0f, 2/255.0f);
			clr.Clamp ( 1, 1, 1, 1);
			*pclr = clr.toClr();
		}
		if ( speed < 0.01 ) {
			clr.fromClr ( *pclr);
			clr.x -= float(1/255.0f);		if ( clr.x < 0.2f ) clr.x = 0.2f;
			clr.y -= float(1/255.0f);		if ( clr.y < 0.2f ) clr.y = 0.2f;
			*pclr = clr.toClr();
		}
		
		// Euler integration -------------------------------
		/* accel += m_Gravity;
		accel *= m_DT;
		p->vel += accel;				// v(t+1) = v(t) + a(t) dt
		p->vel_eval += accel;
		p->vel_eval *= m_DT/d;
		p->pos += p->vel_eval;
		p->vel_eval = p->vel;  */	


		if ( m_Toggle[PWRAP_X] ) {
			diff = ppos->x - (m_Vec[PBOUNDMIN].x + 2);			// -- Simulates object in center of flow
			if ( diff <= 0 ) {
				ppos->x = (m_Vec[PBOUNDMAX].x - 2) + diff*2;				
				ppos->z = 10;
			}
		}	

		ppos++;
		pvel++;
		pveleval++;
		pforce++;
		pclr++;
		ppress++;
		pdensity++;
	}

}

void FluidSystem::ClearNeighborTable ()
{
	if ( m_NeighborTable != 0x0 )	free (m_NeighborTable);
	if ( m_NeighborDist != 0x0)		free (m_NeighborDist );
	m_NeighborTable = 0x0;
	m_NeighborDist = 0x0;
	m_NeighborNum = 0;
	m_NeighborMax = 0;
}

void FluidSystem::ResetNeighbors ()
{
	m_NeighborNum = 0;
}

// Allocate new neighbor tables, saving previous data
int FluidSystem::AddNeighbor ()
{
	if ( m_NeighborNum >= m_NeighborMax ) {
		m_NeighborMax = 2*m_NeighborMax + 1;		
		int* saveTable = m_NeighborTable;
		m_NeighborTable = (int*) malloc ( m_NeighborMax * sizeof(int) );
		if ( saveTable != 0x0 ) {
			memcpy ( m_NeighborTable, saveTable, m_NeighborNum*sizeof(int) );
			free ( saveTable );
		}
		float* saveDist = m_NeighborDist;
		m_NeighborDist = (float*) malloc ( m_NeighborMax * sizeof(float) );
		if ( saveDist != 0x0 ) {
			memcpy ( m_NeighborDist, saveDist, m_NeighborNum*sizeof(int) );
			free ( saveDist );
		}
	};
	m_NeighborNum++;
	return m_NeighborNum-1;
}

void FluidSystem::ClearNeighbors ( int i )
{
	*(m_Fluid.bufI(FNBRCNT)+i) = 0;
}

int FluidSystem::AddNeighbor( int i, int j, float d )
{
	int k = AddNeighbor();
	m_NeighborTable[k] = j;
	m_NeighborDist[k] = d;
	if (*(m_Fluid.bufI(FNBRCNT)+i) == 0 ) *(m_Fluid.bufI(FNBRCNT)+i) = k;
	(*(m_Fluid.bufI(FNBRCNT)+i))++;
	return k;
}

// Ideal grid cell size (gs) = 2 * smoothing radius = 0.02*2 = 0.04
// Ideal domain size = k*gs/d = k*0.02*2/0.005 = k*8 = {8, 16, 24, 32, 40, 48, ..}
//    (k = number of cells, gs = cell size, d = simulation scale)
void FluidSystem::SetupGrid ( Vector3DF min, Vector3DF max, float sim_scale, float cell_size, float border )
{
	float world_cellsize = cell_size / sim_scale;
	
	m_GridMin = min;
	m_GridMax = max;
	m_GridSize = m_GridMax;
	m_GridSize -= m_GridMin;	
	#if 0
		m_GridRes.Set ( 6, 6, 6 );				// Fixed grid res
	#else
		m_GridRes.x = (int) ceil ( m_GridSize.x / world_cellsize );		// Determine grid resolution
		m_GridRes.y = (int) ceil ( m_GridSize.y / world_cellsize );
		m_GridRes.z = (int) ceil ( m_GridSize.z / world_cellsize );
		m_GridSize.x = m_GridRes.x * cell_size / sim_scale;				// Adjust grid size to multiple of cell size
		m_GridSize.y = m_GridRes.y * cell_size / sim_scale;
		m_GridSize.z = m_GridRes.z * cell_size / sim_scale;
	#endif
	m_GridDelta = m_GridRes;		// delta = translate from world space to cell #
	m_GridDelta /= m_GridSize;
	
	m_GridTotal = (int)(m_GridRes.x * m_GridRes.y * m_GridRes.z);

	m_Param[PSTAT_GMEM] = 12.0f * m_GridTotal;		// Grid memory used

	// Number of cells to search:
	// n = (2r / w) +1,  where n = 1D cell search count, r = search radius, w = world cell width
	//
	m_GridSrch = (int) (floor(2.0f*(m_Param[PSMOOTHRADIUS]/sim_scale) / world_cellsize) + 1.0f);
	if ( m_GridSrch < 2 ) m_GridSrch = 2;
	m_GridAdjCnt = m_GridSrch * m_GridSrch * m_GridSrch ;			// 3D search count = n^3, e.g. 2x2x2=8, 3x3x3=27, 4x4x4=64

	if ( m_GridSrch > 6 ) {
		nvprintf ( "ERROR: Neighbor search is n > 6. \n " );
		exit(-1);
	}

	int cell = 0;
	for (int y=0; y < m_GridSrch; y++ ) 
		for (int z=0; z < m_GridSrch; z++ ) 
			for (int x=0; x < m_GridSrch; x++ ) 
				m_GridAdj[cell++] = ( y*m_GridRes.z + z )*m_GridRes.x +  x ;			// -1 compensates for ndx 0=empty
				

	/*nvprintf ( "Adjacency table (CPU) \n");
	for (int n=0; n < m_GridAdjCnt; n++ ) {
		nvprintf ( "  ADJ: %d, %d\n", n, m_GridAdj[n] );
	}*/
}

int FluidSystem::getGridCell ( int p, Vector3DI& gc )
{
	return getGridCell ( m_Fluid.bufV3(FPOS)[p], gc );
}
int FluidSystem::getGridCell ( Vector3DF& pos, Vector3DI& gc )
{
	gc.x = (int)( (pos.x - m_GridMin.x) * m_GridDelta.x);			// Cell in which particle is located
	gc.y = (int)( (pos.y - m_GridMin.y) * m_GridDelta.y);
	gc.z = (int)( (pos.z - m_GridMin.z) * m_GridDelta.z);		
	return (int)( (gc.y*m_GridRes.z + gc.z)*m_GridRes.x + gc.x);		
}
Vector3DI FluidSystem::getCell ( int c )
{
	Vector3DI gc;
	int xz = m_GridRes.x*m_GridRes.z;
	gc.y = c / xz;				c -= gc.y*xz;
	gc.z = c / m_GridRes.x;		c -= gc.z*m_GridRes.x;
	gc.x = c;
	return gc;
}

void FluidSystem::InsertParticles ()
{
	int gs;	
	
	// Reset all grid pointers and neighbor tables to empty
	memset ( m_Fluid.bufC(FGNEXT),		GRID_UCHAR, NumPoints()*sizeof(uint) );
	memset ( m_Fluid.bufC(FGCELL),		GRID_UCHAR, NumPoints()*sizeof(uint) );
	memset ( m_Fluid.bufC(FCLUSTER),	GRID_UCHAR, NumPoints()*sizeof(uint) );

	// Reset all grid cells to empty	
	memset( m_Fluid.bufC(FGRID),		GRID_UCHAR, m_GridTotal*sizeof(uint));
	memset( m_Fluid.bufI(FGRIDCNT),		0, m_GridTotal*sizeof(uint));

	// Insert each particle into spatial grid
	Vector3DI gc;
	Vector3DF* ppos =	m_Fluid.bufV3(FPOS);
	uint* pgrid =		m_Fluid.bufI(FGCELL);
	uint* pnext =		m_Fluid.bufI(FGNEXT);	
	uint* pcell =		m_Fluid.bufI(FCLUSTER);

	float poff = m_Param[PSMOOTHRADIUS] / m_Param[PSIMSCALE];

	int ns = (int) pow ( (float) m_GridAdjCnt, 1.0f/3.0f );
	register int xns, yns, zns;
	xns = m_GridRes.x - m_GridSrch;
	yns = m_GridRes.y - m_GridSrch;
	zns = m_GridRes.z - m_GridSrch;

	m_Param[ PSTAT_OCCUPY ] = 0.0;
	m_Param [ PSTAT_GRIDCNT ] = 0.0;
	uint* m_Grid = m_Fluid.bufI(FGRID);
	uint* m_GridCnt = m_Fluid.bufI(FGRIDCNT);

	for ( int n=0; n < NumPoints(); n++ ) {
		gs = getGridCell ( *ppos, gc );
		if ( gc.x >= 1 && gc.x <= xns && gc.y >= 1 && gc.y <= yns && gc.z >= 1 && gc.z <= zns ) {
			// put current particle at head of grid cell, pointing to next in list (previous head of cell)
			*pgrid = gs;
			*pnext = m_Grid[gs];				
			if ( *pnext == GRID_UNDEF ) m_Param[ PSTAT_OCCUPY ] += 1.0;
			m_Grid[gs] = n;
			m_GridCnt[gs]++;
			m_Param [ PSTAT_GRIDCNT ] += 1.0;
			/* -- 1/2 cell offset search method
			gx = (int)( (-poff + ppos->x - m_GridMin.x) * m_GridDelta.x);	
			if ( gx < 0 ) gx = 0;
			if ( gx > m_GridRes.x-2 ) gx = m_GridRes.x-2;
			gy = (int)( (-poff + ppos->y - m_GridMin.y) * m_GridDelta.y);
			if ( gy < 0 ) gy = 0;
			if ( gy > m_GridRes.y-2 ) gx = m_GridRes.y-2;
			gz = (int)( (-poff + ppos->z - m_GridMin.z) * m_GridDelta.z);
			if ( gz < 0 ) gz = 0;
			if ( gz > m_GridRes.z-2 ) gz = m_GridRes.z-2;
			*pcell = (int)( (gy*m_GridRes.z + gz)*m_GridRes.x + gx) ;	// Cell in which to start 2x2x2 search*/
		} else {
			Vector3DF vel, ve;
			vel = m_Fluid.bufV3(FVEL) [n];
			ve = m_Fluid.bufV3(FVEVAL) [n];
			float pr, dn;
			pr = m_Fluid.bufF(FPRESS) [n];
			dn = m_Fluid.bufF(FDENSITY) [n];
			//printf ( "WARNING: Out of Bounds: %d, P<%f %f %f>, V<%f %f %f>, prs:%f, dns:%f\n", n, ppos->x, ppos->y, ppos->z, vel.x, vel.y, vel.z, pr, dn );
			//ppos->x = -1; ppos->y = -1; ppos->z = -1;
		}
		pgrid++;
		ppos++;
		pnext++;
		pcell++;
	}

	// STATS
	/*m_Param[ PSTAT_OCCUPY ] = 0;
	m_Param[ PSTAT_GRIDCNT ] = 0;
	for (int n=0; n < m_GridTotal; n++) {
		if ( m_GridCnt[n] > 0 )  m_Param[ PSTAT_OCCUPY ] += 1.0;
		m_Param [ PSTAT_GRIDCNT ] += m_GridCnt[n];
	}*/
}

void FluidSystem::SaveResults ()
{
	if ( mSaveNdx != 0x0 ) free ( mSaveNdx );
	if ( mSaveCnt != 0x0 ) free ( mSaveCnt );
	if ( mSaveNeighbors != 0x0 )	free ( mSaveNeighbors );

	mSaveNdx = (uint*) malloc ( sizeof(uint) * NumPoints() );
	mSaveCnt = (uint*) malloc ( sizeof(uint) * NumPoints() );
	mSaveNeighbors = (uint*) malloc ( sizeof(uint) * m_NeighborNum );
	memcpy ( mSaveNdx, m_Fluid.bufC(FNBRNDX), sizeof(uint) * NumPoints() );
	memcpy ( mSaveCnt, m_Fluid.bufC(FNBRCNT), sizeof(uint) * NumPoints() );
	memcpy ( mSaveNeighbors, m_NeighborTable, sizeof(uint) * m_NeighborNum );
}

void FluidSystem::ValidateResults ()
{
//	Setup ();
	nvprintf ( "VALIDATION:\n" );
	InsertParticles ();	nvprintf ( "  Insert. OK\n" );	
	FindNbrsSlow ();	nvprintf ( "  True Neighbors. OK\n" );	
	SaveResults ();		nvprintf ( "  Save Results. OK\n" );	
	Run ();				nvprintf ( "  New Algorithm. OK\n" );
	
	// Quick validation
	nvprintf ( "  Compare...\n" );
	int bad = 0;
	for (int n=0; n < NumPoints(); n++ ) {
		if ( *(mSaveCnt+n) != m_Fluid.bufC(FNBRCNT)[n] ) {
			m_Fluid.bufI(FCLR)[n] = COLORA(1.0,0,0,0.9);
			nvprintf ( "Error %d, correct: %d, cnt: %d\n", n, *(mSaveCnt+n), m_Fluid.bufI(FNBRCNT)[n] );
		}
	}
	if ( bad == 0 ) {
		nvprintf ( "  OK!\n" );
	}
}



void FluidSystem::FindNbrsSlow ()
{
	// O(n^2)
	// Does not require grid

	Vector3DF dst;
	float dsq;
	float d2 = m_Param[PSIMSCALE]*m_Param[PSIMSCALE];
	
	ResetNeighbors ();

	Vector3DF *ipos, *jpos;
	ipos = m_Fluid.bufV3(FPOS);
	for (int i=0; i < NumPoints(); i++ ) {
		jpos = m_Fluid.bufV3(FPOS);
		ClearNeighbors ( i );
		for (int j=0; j < NumPoints(); j++ ) {
			dst = *ipos;
			dst -= *jpos;
			dsq = d2*(dst.x*dst.x + dst.y*dst.y + dst.z*dst.z);
			if ( i != j && dsq <= m_R2 ) {
				AddNeighbor( i, j, sqrt(dsq) );
			}
			jpos++;
		}
		ipos++;
	}
}

void FluidSystem::FindNbrsGrid ()
{
	// O(n^2)
	// Does not require grid

	Vector3DF dst;
	float dsq;
	int j;
	int nadj = (m_GridRes.z + 1)*m_GridRes.x + 1;
	float d2 = m_Param[PSIMSCALE]*m_Param[PSIMSCALE];
	uint* m_Grid = m_Fluid.bufI(FGRID);
	uint* m_GridCnt = m_Fluid.bufI(FGRIDCNT);
	
	ResetNeighbors ();

	Vector3DF *ipos;
	ipos = m_Fluid.bufV3(FPOS);
	for (int i=0; i < NumPoints(); i++ ) {
		ClearNeighbors ( i );
		
		if ( m_Fluid.bufI(FGCELL)[i] != GRID_UNDEF ) {
			for (int cell=0; cell < m_GridAdjCnt; cell++) {
				j = m_Grid [ m_Fluid.bufI(FGCELL)[i] - nadj + m_GridAdj[cell] ] ;
				while ( j != GRID_UNDEF ) {
					if ( i==j ) { j = m_Fluid.bufI(FGNEXT)[j]; continue; }
					dst = *ipos;
					dst -= m_Fluid.bufV3(FPOS)[j];
					dsq = d2*(dst.x*dst.x + dst.y*dst.y + dst.z*dst.z);
					if ( dsq <= m_R2 ) {
						AddNeighbor( i, j, sqrt(dsq) );
					}
					j = m_Fluid.bufI(FGNEXT)[j];
				}
			}
		}
		ipos++;
	}
}


// Compute Pressures - Using spatial grid, and also create neighbor table
void FluidSystem::ComputePressureGrid ()
{
	int i, j, cnt = 0;	
	float sum, dsq, c;
	float d = m_Param[PSIMSCALE];
	float d2 = d*d;
	float radius = m_Param[PSMOOTHRADIUS] / m_Param[PSIMSCALE];
	
	// Get particle buffers
	Vector3DF*	ipos =		m_Fluid.bufV3(FPOS);		
	float*		ipress =	m_Fluid.bufF(FPRESS);
	float*		idensity =	m_Fluid.bufF(FDENSITY);
	uint*		inbr =		m_Fluid.bufI(FNBRNDX);
	uint*		inbrcnt =	m_Fluid.bufI(FNBRCNT);

	Vector3DF	dst;
	int			nadj = (m_GridRes.z + 1)*m_GridRes.x + 1;
	uint*		m_Grid = m_Fluid.bufI(FGRID);
	uint*		m_GridCnt = m_Fluid.bufI(FGRIDCNT);
	
	int nbrcnt = 0;
	int srch = 0;

	for ( i=0; i < NumPoints(); i++ ) {

		sum = 0.0;

		if ( m_Fluid.bufI(FGCELL)[i] != GRID_UNDEF ) {
			for (int cell=0; cell < m_GridAdjCnt; cell++) {
				j = m_Grid [   m_Fluid.bufI(FGCELL)[i] - nadj + m_GridAdj[cell] ] ;
				while ( j != GRID_UNDEF ) {
					if ( i==j ) { j = m_Fluid.bufI(FGNEXT)[j]; continue; }
					dst = m_Fluid.bufV3(FPOS)[j];
					dst -= *ipos;
					dsq = d2*(dst.x*dst.x + dst.y*dst.y + dst.z*dst.z);
					if ( dsq <= m_R2 ) {
						c =  m_R2 - dsq;
						sum += c * c * c;
						nbrcnt++;
						/*nbr = AddNeighbor();			// get memory for new neighbor						
						*(m_NeighborTable + nbr) = j;
						*(m_NeighborDist + nbr) = sqrt(dsq);
						inbr->num++;*/
					}
					srch++;
					j = m_Fluid.bufI(FGNEXT)[j];
				}
			}
		}
		*idensity = sum * m_Param[PMASS] * m_Poly6Kern ;	
		*ipress = ( *idensity - m_Param[PRESTDENSITY] ) * m_Param[PINTSTIFF];		
		*idensity = 1.0f / *idensity;

		ipos++;
		idensity++;
		ipress++;
	}
	// Stats:
	m_Param [ PSTAT_NBR ] = float(nbrcnt);
	m_Param [ PSTAT_SRCH ] = float(srch);
	if ( m_Param[PSTAT_NBR] > m_Param [ PSTAT_NBRMAX ] ) m_Param [ PSTAT_NBRMAX ] = m_Param[PSTAT_NBR];
	if ( m_Param[PSTAT_SRCH] > m_Param [ PSTAT_SRCHMAX ] ) m_Param [ PSTAT_SRCHMAX ] = m_Param[PSTAT_SRCH];
}

// Compute Forces - Using spatial grid with saved neighbor table. Fastest.
void FluidSystem::ComputeForceGrid ()
{
	Vector3DF force;
	register float pterm, vterm, dterm;
	int i, j;
	float c, d;
	float dx, dy, dz;
	float mR, visc;	

	d = m_Param[PSIMSCALE];
	mR = m_Param[PSMOOTHRADIUS];
	visc = m_Param[PVISC];
	
	// Get particle buffers
	Vector3DF*	ipos =		m_Fluid.bufV3(FPOS);		
	Vector3DF*	iveleval =	m_Fluid.bufV3(FVEVAL);		
	Vector3DF*	iforce =	m_Fluid.bufV3(FFORCE);		
	float*		ipress =	m_Fluid.bufF(FPRESS);
	float*		idensity =	m_Fluid.bufF(FDENSITY);
	
	Vector3DF	jpos;
	float		jdist;
	float		jpress;
	float		jdensity;
	Vector3DF	jveleval;
	float		dsq;
	float		d2 = d*d;
	int			nadj = (m_GridRes.z + 1)*m_GridRes.x + 1;
	uint* m_Grid = m_Fluid.bufI(FGRID);
	uint* m_GridCnt = m_Fluid.bufI(FGRIDCNT);

	for ( i=0; i < NumPoints(); i++ ) {

		iforce->Set ( 0, 0, 0 );

		if ( m_Fluid.bufI(FGCELL)[i] != GRID_UNDEF ) {
			for (int cell=0; cell < m_GridAdjCnt; cell++) {
				j = m_Grid [  m_Fluid.bufI(FGCELL)[i] - nadj + m_GridAdj[cell] ];
				while ( j != GRID_UNDEF ) {
					if ( i==j ) { j = m_Fluid.bufI(FGNEXT)[j]; continue; }
					jpos = m_Fluid.bufV3(FPOS)[j];
					dx = ( ipos->x - jpos.x);		// dist in cm
					dy = ( ipos->y - jpos.y);
					dz = ( ipos->z - jpos.z);
					dsq = d2*(dx*dx + dy*dy + dz*dz);
					if ( dsq <= m_R2 ) {

						jdist = sqrt(dsq);

						jpress = m_Fluid.bufF(FPRESS)[j];
						jdensity = m_Fluid.bufF(FDENSITY)[j];
						jveleval = m_Fluid.bufV3(FVEVAL)[j];
						dx = ( ipos->x - jpos.x);		// dist in cm
						dy = ( ipos->y - jpos.y);
						dz = ( ipos->z - jpos.z);
						c = (mR-jdist);
						pterm = d * -0.5f * c * m_SpikyKern * ( *ipress + jpress ) / jdist;
						dterm = c * (*idensity) * jdensity;
						vterm = m_LapKern * visc;
						iforce->x += ( pterm * dx + vterm * ( jveleval.x - iveleval->x) ) * dterm;
						iforce->y += ( pterm * dy + vterm * ( jveleval.y - iveleval->y) ) * dterm;
						iforce->z += ( pterm * dz + vterm * ( jveleval.z - iveleval->z) ) * dterm;
					}
					j = m_Fluid.bufI(FGNEXT)[j];
				}
			}
		}
		ipos++;
		iveleval++;
		iforce++;
		ipress++;
		idensity++;
	}
}


// Compute Forces - Using spatial grid with saved neighbor table. Fastest.
void FluidSystem::ComputeForceGridNC ()
{
	Vector3DF force;
	register float pterm, vterm, dterm;
	int i, j;
	float c, d;
	float dx, dy, dz;
	float mR, visc;	

	d = m_Param[PSIMSCALE];
	mR = m_Param[PSMOOTHRADIUS];
	visc = m_Param[PVISC];

	// Get particle buffers
	Vector3DF*	ipos =		m_Fluid.bufV3(FPOS);		
	Vector3DF*	iveleval =	m_Fluid.bufV3(FVEVAL);		
	Vector3DF*	iforce =	m_Fluid.bufV3(FFORCE);		
	float*		ipress =	m_Fluid.bufF(FPRESS);
	float*		idensity =	m_Fluid.bufF(FDENSITY);
	uint*		inbr =		m_Fluid.bufI(FNBRNDX);
	uint*		inbrcnt =	m_Fluid.bufI(FNBRCNT);

	int			jndx;
	Vector3DF	jpos;
	float		jdist;
	float		jpress;
	float		jdensity;
	Vector3DF	jveleval;

	for ( i=0; i < NumPoints(); i++ ) {

		iforce->Set ( 0, 0, 0 );
		
		jndx = *inbr;
		for (int nbr=0; nbr < (int) *inbrcnt; nbr++ ) {
			j = *(m_NeighborTable+jndx);
			jpos =		m_Fluid.bufV3(FPOS)[j];
			jpress =	m_Fluid.bufF(FPRESS)[j];
			jdensity =  m_Fluid.bufF(FDENSITY)[j];
			jveleval =  m_Fluid.bufV3(FVEVAL)[j];
			jdist = *(m_NeighborDist + jndx);			
			dx = ( ipos->x - jpos.x);		// dist in cm
			dy = ( ipos->y - jpos.y);
			dz = ( ipos->z - jpos.z);
			c = ( mR - jdist );
			pterm = d * -0.5f * c * m_SpikyKern * ( *ipress + jpress ) / jdist;
			dterm = c * (*idensity) * jdensity;
			vterm = m_LapKern * visc;
			iforce->x += ( pterm * dx + vterm * ( jveleval.x - iveleval->x) ) * dterm;
			iforce->y += ( pterm * dy + vterm * ( jveleval.y - iveleval->y) ) * dterm;
			iforce->z += ( pterm * dz + vterm * ( jveleval.z - iveleval->z) ) * dterm;
			jndx++;
		}				
		ipos++;
		iveleval++;
		iforce++;
		ipress++;
		idensity++;
		inbr++;
	}
}


void FluidSystem::SetupRender ()
{
	glEnable ( GL_TEXTURE_2D );

	glGenTextures ( 1, (GLuint*) mTex );
	glBindTexture ( GL_TEXTURE_2D, mTex[0] );
	glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);	
	glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );	
	glPixelStorei( GL_UNPACK_ALIGNMENT, 4);	
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB32F_ARB, 8, 8, 0, GL_RGB, GL_FLOAT, 0);

	glGenBuffersARB ( 3, (GLuint*) mVBO );

	// Construct a sphere in a VBO
	int udiv = 6;
	int vdiv = 6;
	float du = 180.0f / udiv;
	float dv = 360.0f / vdiv;
	float x,y,z, x1,y1,z1;

	float r = 1.0;

	Vector3DF* buf = (Vector3DF*) malloc ( sizeof(Vector3DF) * (udiv+2)*(vdiv+2)*2 );
	Vector3DF* dat = buf;
	
	mSpherePnts = 0;
	for ( float tilt=-90; tilt <= 90.0; tilt += du) {
		for ( float ang=0; ang <= 360; ang += dv) {
			x = sin ( ang*DEGtoRAD) * cos ( tilt*DEGtoRAD );
			y = cos ( ang*DEGtoRAD) * cos ( tilt*DEGtoRAD );
			z = sin ( tilt*DEGtoRAD ) ;
			x1 = sin ( ang*DEGtoRAD) * cos ( (tilt+du)*DEGtoRAD ) ;
			y1 = cos ( ang*DEGtoRAD) * cos ( (tilt+du)*DEGtoRAD ) ;
			z1 = sin ( (tilt+du)*DEGtoRAD );
		
			dat->x = x*r;
			dat->y = y*r;
			dat->z = z*r;
			dat++;
			dat->x = x1*r;
			dat->y = y1*r;
			dat->z = z1*r;
			dat++;
			mSpherePnts += 2;
		}
	}
	glBindBufferARB ( GL_ARRAY_BUFFER_ARB, mVBO[2] );
	glBufferDataARB ( GL_ARRAY_BUFFER_ARB, mSpherePnts*sizeof(Vector3DF), buf, GL_STATIC_DRAW_ARB);
	glVertexPointer ( 3, GL_FLOAT, 0, 0x0 );

	free ( buf );
}


void FluidSystem::DrawCell ( int gx, int gy, int gz )
{
	Vector3DF gd (1, 1, 1);
	Vector3DF gc;
	gd /= m_GridDelta;		
	gc.Set ( (float) gx, (float) gy, (float) gz );
	gc /= m_GridDelta;
	gc += m_GridMin;
	glBegin ( GL_LINES );
	glVertex3f ( gc.x, gc.y, gc.z ); glVertex3f ( gc.x+gd.x, gc.y, gc.z );
	glVertex3f ( gc.x, gc.y+gd.y, gc.z ); glVertex3f ( gc.x+gd.x, gc.y+gd.y, gc.z );
	glVertex3f ( gc.x, gc.y, gc.z+gd.z ); glVertex3f ( gc.x+gd.x, gc.y, gc.z+gd.z );
	glVertex3f ( gc.x, gc.y+gd.y, gc.z+gd.z ); glVertex3f ( gc.x+gd.x, gc.y+gd.y, gc.z+gd.z );

	glVertex3f ( gc.x, gc.y, gc.z ); glVertex3f ( gc.x, gc.y+gd.y, gc.z );
	glVertex3f ( gc.x+gd.x, gc.y, gc.z ); glVertex3f ( gc.x+gd.x, gc.y+gd.y, gc.z );
	glVertex3f ( gc.x, gc.y, gc.z+gd.z ); glVertex3f ( gc.x, gc.y+gd.y, gc.z+gd.z );
	glVertex3f ( gc.x+gd.x, gc.y, gc.z+gd.z ); glVertex3f ( gc.x+gd.x, gc.y+gd.y, gc.z+gd.z );

	glVertex3f ( gc.x, gc.y, gc.z ); glVertex3f ( gc.x, gc.y, gc.z+gd.x );
	glVertex3f ( gc.x, gc.y+gd.y, gc.z ); glVertex3f ( gc.x, gc.y+gd.y, gc.z+gd.z );
	glVertex3f ( gc.x+gd.x, gc.y, gc.z ); glVertex3f ( gc.x+gd.x, gc.y, gc.z+gd.z );
	glVertex3f ( gc.x+gd.x, gc.y+gd.y, gc.z); glVertex3f ( gc.x+gd.x, gc.y+gd.y, gc.z+gd.z );
	glEnd ();
}

void FluidSystem::DrawGrid ()
{
	Vector3DF gd (1, 1, 1);
	Vector3DF gc;
	gd /= m_GridDelta;		
	
	glBegin ( GL_LINES );	
	for (int z=0; z <= m_GridRes.z; z++ ) {
		for (int y=0; y <= m_GridRes.y; y++ ) {
			gc.Set ( 1.0f, float(y), float(z) );	gc /= m_GridDelta;	gc += m_GridMin;
			glVertex3f ( m_GridMin.x, gc.y, gc.z );	glVertex3f ( m_GridMax.x, gc.y, gc.z );
		}
	}
	for (int z=0; z <= m_GridRes.z; z++ ) {
		for (int x=0; x <= m_GridRes.x; x++ ) {
			gc.Set ( float(x), 1.0f, float(z) );	gc /= m_GridDelta;	gc += m_GridMin;
			glVertex3f ( gc.x, m_GridMin.y, gc.z );	glVertex3f ( gc.x, m_GridMax.y, gc.z );
		}
	}
	for (int y=0; y <= m_GridRes.y; y++ ) {
		for (int x=0; x <= m_GridRes.x; x++ ) {
			gc.Set ( float(x), float(y), 1.0f);	gc /= m_GridDelta;	gc += m_GridMin;
			glVertex3f ( gc.x, gc.y, m_GridMin.z );	glVertex3f ( gc.x, gc.y, m_GridMax.z );
		}
	}
	glEnd ();
}

void FluidSystem::DrawParticle ( int p, int r1, int r2, Vector3DF clr )
{
	Vector3DF* ppos = m_Fluid.bufV3(FPOS) + p;
	uint* pclr = m_Fluid.bufI(FCLR)  + p;
	
	glDisable ( GL_DEPTH_TEST );
	
	glPointSize ( (GLfloat) r2 );	
	glBegin ( GL_POINTS );
	glColor3f ( clr.x, clr.y, clr.z ); glVertex3f ( ppos->x, ppos->y, ppos->z );
	glEnd ();

	glEnable ( GL_DEPTH_TEST );
}

void FluidSystem::DrawNeighbors ( int p )
{
	if ( p == -1 ) return;

	Vector3DF* ppos = m_Fluid.bufV3(FPOS) + p;
	Vector3DF jpos;
	CLRVAL jclr;
	GLfloat r, g, b;
	int j;

	glBegin ( GL_LINES );
	uint cnt = m_Fluid.bufI(FNBRCNT)[p];
	uint ndx = m_Fluid.bufI(FNBRNDX)[p];
	for ( int n=0; n < (int) cnt; n++ ) {
		j = m_NeighborTable[ ndx ];
		jpos = m_Fluid.bufV3(FPOS)[j];
		jclr = m_Fluid.bufI(FCLR)[j];
		r = (GLfloat) (RED(jclr)+1.0)*0.5f;
		g = (GLfloat) (GRN(jclr)+1.0)*0.5f;
		b = (GLfloat) (BLUE(jclr)+1.0)*0.5f;
		glColor4f ( r, g, b, (GLfloat) ALPH(jclr) );
		glVertex3f ( ppos->x, ppos->y, ppos->z );
		
		jpos -= *ppos; jpos *= 0.9f;		// get direction of neighbor, 90% dist
		glVertex3f ( ppos->x + jpos.x, ppos->y + jpos.y, ppos->z + jpos.z );
		ndx++;
	}
	glEnd ();
}

void FluidSystem::DrawCircle ( Vector3DF pos, float r, Vector3DF clr, Camera3D& cam )
{
	glPushMatrix ();
	
	glTranslatef ( pos.x, pos.y, pos.z );
	glMultMatrixf ( cam.getInvView().GetDataF() );
	glColor3f ( clr.x, clr.y, clr.z );
	glBegin ( GL_LINE_LOOP );
	float x, y;
	for (float a=0; a < 360; a += 10.0f ) {
		x = cos ( a*DEGtoRAD )*r;
		y = sin ( a*DEGtoRAD )*r;
		glVertex3f ( x, y, 0 );
	}
	glEnd ();

	glPopMatrix ();
}


void FluidSystem::DrawText ()
{
	char msg[100];

	
	Vector3DF* ppos = m_Fluid.bufV3(FPOS);
	uint* pclr = m_Fluid.bufI(FCLR);
	Vector3DF clr;
	for (int n = 0; n < NumPoints(); n++) {
	
		sprintf ( msg, "%d", n );
		glColor4f ( (GLfloat) (RED(*pclr)+1.0f)*0.5f, (GLfloat) (GRN(*pclr)+1.0f)*0.5f, (GLfloat) (BLUE(*pclr)+1.0f)*0.5f, (GLfloat) ALPH(*pclr) );
		//drawText3D ( ppos->x, ppos->y, ppos->z, msg );
		ppos++;
		pclr++;
	}
}


void FluidSystem::Draw ( int frame, Camera3D& cam, float rad )
{
	Vector3DF* ppos;
	float* pdens;
	uint* pclr;

	glDisable ( GL_LIGHTING );

	switch ( (int) m_Param[PDRAWGRID] ) {	
	case 1: {
		glColor4f ( (GLfloat) 0.7, (GLfloat) 0.7, (GLfloat) 0.7, (GLfloat) 0.05 );
		DrawGrid ();
		} break;
	};
	if ( m_Param[PDRAWTEXT] == 1.0 ) {
		DrawText ();
	};

	// Draw Modes
	// DRAW_POINTS		0
	// DRAW_SPRITES		1
	// DRAW_

	// Write to File
	if ( mbRecord ) {
		SavePoints ( frame );
	}
	if ( mbRecordBricks ) {
		SaveBricks ( frame );
	}	

	

	// Render
	switch ( (int) m_Param[PDRAWMODE] ) {
	case 0: {
		glPointSize ( 2 );
		glEnable ( GL_POINT_SIZE );		
		glEnable( GL_BLEND ); 
		glBindBufferARB ( GL_ARRAY_BUFFER_ARB, mVBO[0] );
		glBufferDataARB ( GL_ARRAY_BUFFER_ARB, NumPoints()*sizeof(Vector3DF), m_Fluid.bufV3(FPOS), GL_DYNAMIC_DRAW_ARB);		
		glVertexPointer ( 3, GL_FLOAT, 0, 0x0 );				
		glBindBufferARB ( GL_ARRAY_BUFFER_ARB, mVBO[1] );
		glBufferDataARB ( GL_ARRAY_BUFFER_ARB, NumPoints()*sizeof(uint), m_Fluid.bufI(FCLR), GL_DYNAMIC_DRAW_ARB);
		glColorPointer ( 4, GL_UNSIGNED_BYTE, 0, 0x0 ); 
		glEnableClientState ( GL_VERTEX_ARRAY );
		glEnableClientState ( GL_COLOR_ARRAY );          
		glNormal3f ( 0, 0.001f, 1.0f );
		glColor3f ( 1, 1, 1 );
		//glLoadMatrixf ( view_mat );
		glDrawArrays ( GL_POINTS, 0, NumPoints() );
		glDisableClientState ( GL_VERTEX_ARRAY );
		glDisableClientState ( GL_COLOR_ARRAY );
		} break;
	
	case 1: {

		glEnable(GL_BLEND); 
	    glEnable(GL_ALPHA_TEST); 
	    glAlphaFunc( GL_GREATER, 0.5 ); 
		//glEnable ( GL_COLOR_MATERIAL );
		//glColorMaterial ( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE );
				
		// Point sprite size
		
		glEnable(GL_POINT_SPRITE_ARB); 		
		float quadratic[] =  { 1.0f, 0.01f, 0.0001f };
		glEnable (  GL_POINT_DISTANCE_ATTENUATION  );
		glPointParameterfvARB(  GL_POINT_DISTANCE_ATTENUATION, quadratic );
		//float maxSize = 10.0f;
		//glGetFloatv( GL_POINT_SIZE_MAX_ARB, &maxSize );		
		glPointSize ( 32 );		
		glPointParameterfARB( GL_POINT_SIZE_MAX_ARB, 32 );
		glPointParameterfARB( GL_POINT_SIZE_MIN_ARB, 1.0f );

		// Texture and blending mode
		/*glEnable ( GL_TEXTURE_2D );
		glBindTexture ( GL_TEXTURE_2D, mImg.getTex() );
		glTexEnvi (GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
		glTexEnvf (GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );
		glBlendFunc ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA ) ;*/

		// Point buffers
		glBindBufferARB ( GL_ARRAY_BUFFER_ARB, mVBO[0] );
		glBufferDataARB ( GL_ARRAY_BUFFER_ARB, NumPoints()*sizeof(Vector3DF), m_Fluid.bufV3(FPOS), GL_DYNAMIC_DRAW_ARB);		
		glVertexPointer ( 3, GL_FLOAT, 0, 0x0 );				
		glBindBufferARB ( GL_ARRAY_BUFFER_ARB, mVBO[1] );
		glBufferDataARB ( GL_ARRAY_BUFFER_ARB, NumPoints()*sizeof(uint), m_Fluid.bufI(FCLR), GL_DYNAMIC_DRAW_ARB);
		glColorPointer ( 4, GL_UNSIGNED_BYTE, 0, 0x0 ); 
		glEnableClientState ( GL_VERTEX_ARRAY );
		glEnableClientState ( GL_COLOR_ARRAY );
          
		// Render - Point Sprites
		glNormal3f ( 0, 1, 0.001f  );
		glColor4f ( 1, 1, 1, 1 );
		glDrawArrays ( GL_POINTS, 0, NumPoints() );

		// Restore state
		glDisableClientState ( GL_VERTEX_ARRAY );
		glDisableClientState ( GL_COLOR_ARRAY );
		glDisable (GL_POINT_SPRITE_ARB); 
		glDisable ( GL_ALPHA_TEST );
		glDisable ( GL_TEXTURE_2D );
		glDepthMask( GL_TRUE );   


		} break;
	case 2: {

		// Notes:
		// # particles, time(Render), time(Total), time(Sim), Render Overhead (%)
		//  250000,  12, 110, 98,  10%   - Point sprites
		//  250000,  36, 146, 110, 24%   - Direct rendering (drawSphere)
		//  250000, 140, 252, 110, 55%   - Batch instancing

		glEnable ( GL_LIGHTING );
		ppos = m_Fluid.bufV3(FPOS);
		pclr =  m_Fluid.bufI(FCLR);
		pdens = m_Fluid.bufF(FDENSITY);
		
		for (int n = 0; n < NumPoints(); n++) {
			glPushMatrix ();
			glTranslatef ( ppos->x, ppos->y, ppos->z );		
			glScalef ( rad, rad, rad );			
			glColor4f ( (GLfloat) RED(*pclr),(GLfloat)  GRN(*pclr), (GLfloat) BLUE(*pclr), (GLfloat) ALPH(*pclr) );
			//drawSphere ();
			glPopMatrix ();		
			ppos++;
			pclr++;
		}

		// --- HARDWARE INSTANCING
		/* cgGLEnableProfile( vert_profile );		
		// Sphere VBO
		glBindBufferARB ( GL_ARRAY_BUFFER_ARB, mVBO[2] );
		glVertexPointer ( 3, GL_FLOAT, 0, 0x0 );		
		glEnableClientState ( GL_VERTEX_ARRAY );
	
		glColor4f( 1,1,1,1 );

		CGparameter uParam = cgGetNamedParameter( cgVP, "modelViewProj" );
		glLoadMatrixf ( view_mat );
		cgGLSetStateMatrixParameter( uParam, CG_GL_MODELVIEW_PROJECTION_MATRIX, CG_GL_MATRIX_IDENTITY ); 

		uParam = cgGetNamedParameter( cgVP, "transformList" );
		int batches = NumPoints() / 768;
		int noff = 0;		
		for (int n=0; n < batches; n++ ) {
			cgGLSetParameterArray3f ( uParam, 0, 768, (float*) (m_FluidBufs.mpos + noff) ); 
			glDrawArraysInstancedARB ( GL_TRIANGLE_STRIP, 0, mSpherePnts, 768 );
			noff += 768;
		}
		cgGLDisableProfile( vert_profile );
		glDisableClientState ( GL_VERTEX_ARRAY );
		glDisableClientState ( GL_COLOR_ARRAY );  */


		//--- Texture buffer technique
		/*
		uParam = cgGetNamedParameter( cgVP, "transformList");
		cgGLSetTextureParameter ( uParam, mTex[0] );
		cgGLEnableTextureParameter ( uParam );
		uParam = cgGetNamedParameter( cgVP, "primCnt");
		cgGLSetParameter1f ( uParam, NumPoints() );		
		glBindTexture ( GL_TEXTURE_2D, mTex[0] );
		glTexImage2D ( GL_TEXTURE_2D, 0, GL_RGB32F_ARB, 2048, int(NumPoints()/2048)+1, 0, GL_RGB, GL_FLOAT, m_FluidBufs.mpos );
		glBindTexture ( GL_TEXTURE_2D, 0x0 );
		glFinish ();*/		
		} break;
	};

	//-------------------------------------- DEBUGGING
	// draw neighbors of particle i
		/*int i = 320;
		int j, jndx = (mNbrList + i )->first;
		for (int nbr=0; nbr < (mNbrList+i)->num; nbr++ ) {			
			j = *(m_NeighborTable+jndx);
			ppos = (m_FluidBufs.mpos + j );
			glPushMatrix ();
			glTranslatef ( ppos->x, ppos->y, ppos->z );		
			glScalef ( 0.25, 0.25, 0.25 );			
			glColor4f ( 0, 1, 0, 1);		// green
			drawSphere ();
			glPopMatrix ();		
			jndx++;
		}
		// draw particles in grid cells of i
		Vector3DF jpos;
		Grid_FindCells ( i );
		for (int cell=0; cell < 8; cell++) {
			j = m_Grid [ *(mClusterCell+i) + m_GridAdj[cell] ];			
			while ( j != -1 ) {
				if ( i==j ) { j = m_Fluid.bufI(FGNEXT)[j]; continue; }
				jpos = *(m_FluidBufs.mpos + j);
				glPushMatrix ();
				glTranslatef ( jpos.x, jpos.y, jpos.z );		
				glScalef ( 0.22, 0.22, 0.22 );
				glColor4f ( 1, 1, 0, 1);		// yellow
				drawSphere ();
				glPopMatrix ();
				j = m_Fluid.bufI(FGNEXT)[j];
			}
		}

		// draw grid cells of particle i		
		float poff = m_Param[PSMOOTHRADIUS] / m_Param[PSIMSCALE];
		int gx = (int)( (-poff + ppos->x - m_GridMin.x) * m_GridDelta.x);		// Determine grid cell
		int gy = (int)( (-poff + ppos->y - m_GridMin.y) * m_GridDelta.y);
		int gz = (int)( (-poff + ppos->z - m_GridMin.z) * m_GridDelta.z);
		Vector3DF gd (1, 1, 1);
		Vector3DF gc;
		gd /= m_GridDelta;

		*/

	// Error particles (debugging)
	/*for (int n=0; n < NumPoints(); n++) {
		if ( ALPH(*(mClr+n))==0.9 ) 
			DrawParticle ( n, 12, 14, Vector3DF(1,0,0) );
	}

	// Draw selected particle
	DrawNeighbors ( mSelected );
	DrawParticle ( mSelected, 8, 12, Vector3DF(1,1,1) );
	DrawCircle ( *(m_FluidBufs.mpos+mSelected), m_Param[PSMOOTHRADIUS]/m_Param[PSIMSCALE], Vector3DF(1,1,0), cam );
	Vector3DI gc;
	int gs = getGridCell ( mSelected, gc );	// Grid cell of selected
	
	glDisable ( GL_DEPTH_TEST );	
	glColor3f ( 0.8, 0.8, 0.9 );
	gs = *(mClusterCell + mSelected);		// Cluster cell
	for (int n=0; n < m_GridAdjCnt; n++ ) {		// Cluster group
		gc = getCell ( gs + m_GridAdj[n] );	DrawCell ( gc.x, gc.y, gc.z );
	}
	glColor3f ( 1.0, 1.0, 1.0 );
	DrawCell ( gc.x, gc.y, gc.z );
	glEnable ( GL_DEPTH_TEST );*/
}


void FluidSystem::StartRecord ()
{
	mbRecord = !mbRecord;	
}
void FluidSystem::StartRecordBricks ()
{
	mbRecordBricks = !mbRecordBricks;
}

void FluidSystem::SavePoints ( int frame )
{
	char buf[256];
	sprintf ( buf, "jet%04d.pts", frame );
	FILE* fp = fopen ( buf, "wb" );

	int numpnt = NumPoints();
	int numfield = 3;
	int ftype;		// 0=char, 1=int, 2=float, 3=double
	int fcnt;
	fwrite ( &numpnt, sizeof(int), 1, fp );
	fwrite ( &numfield, sizeof(int), 1, fp );
				
	// write positions
	ftype = 2; fcnt = 3;		// float, 3 channel
	fwrite ( &ftype, sizeof(int), 1, fp );		
	fwrite ( &fcnt,  sizeof(int), 1, fp );	
	fwrite ( m_Fluid.bufC(FPOS),  numpnt*sizeof(Vector3DF), 1, fp );

	// write velocities
	ftype = 2; fcnt = 3;		// float, 3 channel
	fwrite ( &ftype, sizeof(int), 1, fp );		
	fwrite ( &fcnt,  sizeof(int), 1, fp );	
	fwrite ( m_Fluid.bufC(FVEL),  numpnt*sizeof(Vector3DF), 1, fp );

	// write colors
	ftype = 0; fcnt = 4;		// char, 4 channel
	fwrite ( &ftype, sizeof(int), 1, fp );		
	fwrite ( &fcnt,  sizeof(int), 1, fp );	
	fwrite ( m_Fluid.bufC(FCLR),  numpnt*sizeof(unsigned char)*4, 1, fp );

	fclose ( fp );

	fflush ( fp );
}

float FluidSystem::Sample ( Vector3DF p )
{
	float smoothing = 16.0;

	float v;
	Vector3DF jpos, del, jvel;
	Vector3DI gc; 
	float dsq, jdist, jdensity, W;
	float d2 = m_Param[PSIMSCALE]*m_Param[PSIMSCALE];	
	int j;
	float mR = m_Param[PSMOOTHRADIUS];
	float h2 = 2.0f*mR*mR / smoothing;
	float mGaussKern = 1.0f/ pow (3.141592f * 2.0f*mR*mR, 3.0f/2.0f);		// Kelager, eq 3.18
	uint* m_Grid = m_Fluid.bufI(FGRID);
	uint* m_GridCnt = m_Fluid.bufI(FGRIDCNT);
	register int xns, yns, zns;
	xns = m_GridRes.x - m_GridSrch;
	yns = m_GridRes.y - m_GridSrch;
	zns = m_GridRes.z - m_GridSrch;

	v = 0;
	
	int gs = getGridCell ( p, gc );				// find grid cell of sample point
	if ( gc.x >= 1 && gc.x <= xns && gc.y >= 1 && gc.y <= yns && gc.z >= 1 && gc.z <= zns ) {
		if ( gs != GRID_UNDEF ) {
			
			int nadj = (m_GridRes.z + 1)*m_GridRes.x + 1;			
			for (int cell=0; cell < m_GridAdjCnt; cell++) {
				j = m_Grid [ gs - nadj + m_GridAdj[cell] ];
				while ( j != GRID_UNDEF ) {			
					jpos = m_Fluid.bufV3(FPOS)[j];
					del = p - jpos;			// dist in cm				
					dsq = d2*(del.x*del.x + del.y*del.y + del.z*del.z);
					if ( dsq <= m_R2 ) {
						jdist = sqrt(dsq);						// this is r-rj
						jdensity = m_Fluid.bufF(FDENSITY)[j];
						jvel = m_Fluid.bufV3(FVEL)[j];
						W = mGaussKern * exp ( -(jdist*jdist/ h2) );
						//v += jvel * (W / jdensity);			// evaluate variable (jveleval)											
						v += W;
					}
					j = m_Fluid.bufI(FGNEXT)[j];
				}
			}			
		}
	}	

	return v / 100.0f;
}


void FluidSystem::SaveBricks ( int frame )
{
	char buf[256];

	// Needed if we eval on CPU
	/*InsertParticles ();								// make sure paricles are binned
	ComputePressureGrid ();							// make sure we have density */

	// Needed if we eval on GPU	
	TransferToCUDA ();
	InsertParticlesCUDA ( 0x0, 0x0, 0x0 );			// make sure particles are binned
	PrefixSumCellsCUDA ( 0x0, 1 );
	CountingSortFullCUDA ( 0x0 );	
	ComputePressureCUDA();							// make sure we have density
	
	Vector3DF volmin, volmax, p;
	volmin = m_Vec[PVOLMIN];
	volmax = m_Vec[PVOLMAX];

	//Vector3DI res ( 2048, 768, 1024 );	
	Vector3DI res = m_VolRes;

	Vector3DI brk ( m_BrkRes, m_BrkRes, m_BrkRes );
	Vector3DI resdiv ( res.x/brk.x, res.y/brk.y, res.z/brk.z );
	Vector3DI b;
	Vector3DF bmin, bmax;
	float chksum;

	// Brick memory storage
	float* vdata = (float*) malloc ( brk.x*brk.y*brk.z* sizeof(float) );
	float* vchk  = (float*) malloc ( 2*2*2* sizeof(float) );	
	Vector3DI chkres ( 2, 2, 2 );

	int brkmax = resdiv.x*resdiv.y*resdiv.z;		// maximum possible bricks
	int brkcnt = 0;									// number accepted

	// Open brick file to write
	sprintf ( buf, "%s", getResolvedName ( false, m_Frame).c_str() );	// output filename	
	FILE* fp = fopen ( buf, "wb" );
	if ( fp == 0x0 ) {
		nvprintf ( "ERROR: Creating %s\n. Folder may not exist.", buf );
		exit (-1);
	}
	
	fwrite ( &brkcnt, sizeof(int), 1, fp );			// number of bricks (not known yet)

	Vector3DF bkmin, bkmax;
	bkmin.Set ( 1e20f, 1e20f, 1e20f );
	bkmax.Set (-1e20f, -1e20f,-1e20f );

	// Scan volume domain
	for (b.y=0; b.y < resdiv.y; b.y++ ) {
		for (b.z=0; b.z < resdiv.z; b.z++ ) {
			for (b.x=0; b.x < resdiv.x; b.x++ ) {				

				// Bounding box of brick
				bmin = (volmax-volmin) * Vector3DF(b*brk); bmin /= res; bmin += volmin;
				bmax = (volmax-volmin) * Vector3DF((b+Vector3DI(1,1,1))*brk); bmax /= res; bmax += volmin;
				if ( bmin.x < bkmin.x ) bkmin.x = bmin.x;
				if ( bmin.y < bkmin.y ) bkmin.y = bmin.y;
				if ( bmin.z < bkmin.z ) bkmin.z = bmin.z;
				if ( bmax.x > bkmax.x ) bkmax.x = bmax.x;
				if ( bmax.y > bkmax.y ) bkmax.y = bmax.y;
				if ( bmax.z > bkmax.z ) bkmax.z = bmax.z;
				
				// Sample corners
				SampleParticlesCUDA ( (float*) vchk, *(uint3*)& chkres, *(float3*)& bmin, *(float3*)& bmax, 1.0f/100.0f );

				/*chk[0] = Sample ( Vector3DF(bmin.x,bmin.y,bmin.z) );
				chk[1] = Sample ( Vector3DF(bmax.x,bmin.y,bmin.z) );
				chk[2] = Sample ( Vector3DF(bmax.x,bmax.y,bmin.z) );
				chk[3] = Sample ( Vector3DF(bmin.x,bmax.y,bmin.z) );
				chk[4] = Sample ( Vector3DF(bmin.x,bmin.y,bmax.z) );
				chk[5] = Sample ( Vector3DF(bmax.x,bmin.y,bmax.z) );
				chk[6] = Sample ( Vector3DF(bmax.x,bmax.y,bmax.z) );
				chk[7] = Sample ( Vector3DF(bmin.x,bmax.y,bmax.z) );*/

				chksum = 0;				
				for (int ck=0; ck<8; ck++)
					chksum += vchk[ck];
				chksum /= 8.0;

				if ( chksum > m_Thresh ) {
					// Accept this brick						
					// 1.0/1500000000.0 
					SampleParticlesCUDA ( (float*) vdata, *(uint3*)& brk, *(float3*)& bmin, *(float3*)& bmax, 1.0f/100.0f );
					brkcnt++;

					// Write brick dimensions				
					Vector3DI bndx = b * brk;
					fwrite ( &bndx, sizeof(Vector3DI), 1, fp );
					fwrite ( &bmin, sizeof(Vector3DF), 1, fp );
					fwrite ( &bmax, sizeof(Vector3DF), 1, fp );
					fwrite ( &brk,  sizeof(Vector3DI), 1, fp );
				
					// Write brick data
					fwrite ( vdata, sizeof(float), brk.x*brk.y*brk.z, fp );

					//-- debug, write to png
					/*float vmin = +1.0e20, vmax = -1.0e20;
					pix = img.getData();
					for (int py=0; py < brk.z * zd; py++ ) {
						for (int px = 0; px < brk.x * zd; px++ ) {
							vx = px % brk.x;
							vy = py % brk.y; 
							vz = (px/brk.x)*zd + int(py/brk.y);
							v = vdata[ (vz*int(brk.z)+vy)*int(brk.x) + vx ];
							*pix++ = uchar(v*255.0); *pix++ = uchar(v*255.0); *pix++ = uchar(v*255.0); *pix++ = 255;
						}
					}
					img.SavePng ( "test.png" );*/
				}
			}
		}
	}

	// write number of bricks
	fseek ( fp, 0, SEEK_SET );
	fwrite ( &brkcnt, sizeof(int), 1, fp );

	/*nvprintf ( "Bricks: %d of %d\n", brkcnt, brkmax );
	nvprintf ( "Brick min: %f,%f,%f (vmin %f,%f,%f)\n", bkmin.x, bkmin.y, bkmin.z, volmin.x, volmin.y, volmin.z);
	nvprintf ( "Brick max: %f,%f,%f (vmax %f,%f,%f)\n", bkmax.x, bkmax.y, bkmax.z, volmax.x, volmax.y, volmax.z);*/

	free ( vdata );

	fclose ( fp );

	fflush ( fp );


	//--- generate slice
	/*Vector3DI res ( 4096, 1, 2048 );		
	nvImg img;
	img.Create ( res.x, res.z, IMG_RGBA );
	uchar* pix = img.getData();
	Vector3DF* vdata = (Vector3DF*) malloc ( res.x*res.y*res.z* sizeof(float)*3 );
	
	SampleParticlesCUDA ( (float3*) vdata, *(uint3*)& res, *(float3*)& volmin, *(float3*)& volmax );

	float vmin = +1.0e20, vmax = -1.0e20;
	for (int y=0; y < res.z; y++ ) {
		for (int x = 0; x < res.x; x++ ) {
			// p = volmin + Vector3DF(x/float(xres), 0.05, y/float(yres)) * (volmax-volmin);
			// v = Sample ( p ).Length() / 1500000000.0;    // -- CPU Sampling

			v = vdata[ y*int(res.x) + x ].Length() / 1500000000.0;
			if ( v < vmin ) vmin = v;
			if ( v > vmax ) vmax = v;
			*pix++ = uchar(v*255.0); *pix++ = uchar(v*255.0); *pix++ = uchar(v*255.0); *pix++ = 255;
		}
	}
	nvprintf ( "Range: %f %f\n", vmin, vmax );
	free ( vdata );
	img.SavePng ( "test.png" );	*/
}

void FluidSystem::StartPlayback ()
{
	mbRecord = false;
	m_Param[PMODE] = RUN_PLAYBACK;
	m_Frame = 0;
}

std::string FluidSystem::getResolvedName ( bool bIn, int frame )
{
	char dataname[1024];
	std::string datastr;
	
	// Input or Output
	if ( bIn )	datastr = m_InFile;
	else		datastr = m_OutFile;

	// Frame number
	size_t lpos = datastr.find_first_of ( '#' );
	if ( lpos != std::string::npos  ) {			// data sequence 		
		size_t rpos = datastr.find_last_of ( '#' );
		if ( rpos != std::string::npos ) {			
			sprintf ( dataname, "%0*d", (int)(rpos-lpos+1), frame );
			datastr = datastr.substr ( 0, lpos ) + std::string(dataname) + datastr.substr ( rpos+1 );			
		}
	}

	if ( datastr.at(1) != ':' ) {		
		datastr = m_WorkPath + datastr;		// use relative path
	}
	
	return datastr;
}

void FluidSystem::RunPlayback ()
{	
	char buf[1024];
	sprintf ( buf, "%s", getResolvedName ( true, m_Frame).c_str() );	// input filename
	

	FILE* fp = fopen ( buf, "rb" );
	if ( fp == 0x0 ) {
		nvprintf ( "WARNING: File not found %s\n", buf );
		return;
	}

	int numpnt = 0;
	int numfield = 2;
	int ftype;		// 0=char, 1=int, 2=float, 3=double
	int fcnt;
	fread ( &numpnt, sizeof(int), 1, fp );
	fread ( &numfield, sizeof(int), 1, fp );
	mNumPoints = numpnt;
	
	
	if ( mNumPoints > mMaxPoints ) {
		// Quick setup		
		m_Param [PNUM] = (float) mNumPoints;
		AllocateParticles ( mNumPoints );
		FluidSetupCUDA ( NumPoints(), m_GridSrch, *(int3*)& m_GridRes, *(float3*)& m_GridSize, *(float3*)& m_GridDelta, *(float3*)& m_GridMin, *(float3*)& m_GridMax, m_GridTotal, (int) m_Vec[PEMIT_RATE].x );		
	}	
				
	// read positions	
	fread ( &ftype, sizeof(int), 1, fp );		
	fread ( &fcnt,  sizeof(int), 1, fp );	
	fread ( m_Fluid.bufC(FPOS),  numpnt*sizeof(Vector3DF), 1, fp );
	
	// read velocities	
	fread ( &ftype, sizeof(int), 1, fp );		
	fread ( &fcnt,  sizeof(int), 1, fp );	
	fread ( m_Fluid.bufC(FVEL),  numpnt*sizeof(Vector3DF), 1, fp );
	Vector3DF s;
	s.Set(0,0,0);
	for (int n=0; n < numpnt; n++ ) {
		s += m_Fluid.bufV3(FVEL)[n];
	}
	if ( s.Length()==0 ) {
		nvprintf ( "NO VEL DATA.\n" );

	}

	// read colors	
	fread ( &ftype, sizeof(int), 1, fp );		
	fread ( &fcnt,  sizeof(int), 1, fp );	
	fread ( m_Fluid.bufI(FCLR),  numpnt*sizeof(unsigned char)*4, 1, fp );

	fclose ( fp );

}



std::string FluidSystem::getModeStr ()
{
	char buf[100];

	switch ( (int) m_Param[PMODE] ) {
	case RUN_SEARCH:		sprintf ( buf, "SEARCH ONLY (CPU)" );		break;
	case RUN_VALIDATE:		sprintf ( buf, "VALIDATE GPU to CPU");		break;
	case RUN_CPU_SLOW:		sprintf ( buf, "SIMULATE CPU Slow");		break;
	case RUN_CPU_GRID:		sprintf ( buf, "SIMULATE CPU Grid");		break;	
	case RUN_GPU_FULL:		sprintf ( buf, "SIMULATE CUDA Full Sort" );	break;	
	case RUN_PLAYBACK:		sprintf ( buf, "PLAYBACK" ); break;
	};
	//sprintf ( buf, "RECORDING (%s, %.4f MB)", mFileName.c_str(), mFileSize ); break;
	return buf;
};


void FluidSystem::getModeClr ()
{
	glColor4f ( 1, 1, 0, 1 ); 
	/*break;
	switch ( mMode ) {
	case RUN_PLAYBACK:		glColor4f ( 0, 1, 0, 1 ); break;
	case RUN_RECORD:		glColor4f ( 1, 0, 0, 1 ); break;
	case RUN_SIM:			glColor4f ( 1, 1, 0, 1 ); break;
	}*/
}

int FluidSystem::SelectParticle ( int x, int y, int wx, int wy, Camera3D& cam )
{
	Vector4DF pnt;
	Vector3DF* ppos = m_Fluid.bufV3(FPOS);
	
	for (int n = 0; n < NumPoints(); n++ ) {
		pnt = cam.project ( *ppos );
		pnt.x = (pnt.x+1.0f)*0.5f * wx;
		pnt.y = (pnt.y+1.0f)*0.5f * wy;

		if ( x > pnt.x-8 && x < pnt.x+8 && y > pnt.y-8 && y < pnt.y+8 ) {
			mSelected = n;
			return n;
		}
		ppos++;
	}
	mSelected = -1;
	return -1;
}


void FluidSystem::DrawParticleInfo ( int p )
{
	char disp[256];

	start2D ();

	glColor4f ( 1.0, 1.0, 1.0, 1.0 );
	sprintf ( disp, "Particle: %d", p );		drawText ( 10, 20, disp, 1, 1, 1, 1 ); 

	Vector3DI gc;
	int gs = getGridCell ( p, gc );
	sprintf ( disp, "Grid Cell:    <%d, %d, %d> id: %d", gc.x, gc.y, gc.z, gs );		drawText ( 10, 40, disp, 1,1,1,1 ); 

	int cc = m_Fluid.bufI(FCLUSTER)[p];
	gc = getCell ( cc );
	sprintf ( disp, "Cluster Cell: <%d, %d, %d> id: %d", gc.x, gc.y, gc.z, cc );		drawText ( 10, 50, disp, 1,1,1,1 ); 

	sprintf ( disp, "Neighbors:    " );
	int cnt = m_Fluid.bufI(FNBRCNT)[p];
	int ndx = m_Fluid.bufI(FNBRNDX)[p];
	for ( int n=0; n < cnt; n++ ) {
		sprintf ( disp, "%s%d, ", disp, m_NeighborTable[ ndx ] );
		ndx++;
	}
	drawText ( 10, 70, disp, 1,1,1,1 );
	uint* m_Grid = m_Fluid.bufI(FGRID);
	uint* m_GridCnt = m_Fluid.bufI(FGRIDCNT);

	if ( cc != -1 ) {
		sprintf ( disp, "Cluster Group: ");		drawText ( 10, 90, disp, 1,1,1,1);
		int cadj;
		int stotal = 0;
		for (int n=0; n < m_GridAdjCnt; n++ ) {		// Cluster group
			cadj = cc+m_GridAdj[n];
			gc = getCell ( cadj );
			sprintf ( disp, "<%d, %d, %d> id: %d, cnt: %d ", gc.x, gc.y, gc.z, cc+m_GridAdj[n], m_GridCnt[ cadj ] );	
			drawText ( 20, 100+n*10.0f, disp, 1,1,1,1 );
			stotal += m_GridCnt[cadj];
		}

		sprintf ( disp, "Search Overhead: %f (%d of %d), %.2f%% occupancy", float(stotal)/ cnt, cnt, stotal, float(cnt)*100.0/stotal );
		drawText ( 10, 380, disp, 1,1,1,1 );
	}	

	end2D ();
}



void FluidSystem::SetupKernels ()
{
	m_Param [ PDIST ] = pow ( (float) m_Param[PMASS] / m_Param[PRESTDENSITY], 1.0f/3.0f );
	m_R2 = m_Param [PSMOOTHRADIUS] * m_Param[PSMOOTHRADIUS];
	m_Poly6Kern = 315.0f / (64.0f * 3.141592f * pow( m_Param[PSMOOTHRADIUS], 9.0f) );	// Wpoly6 kernel (denominator part) - 2003 Muller, p.4
	m_SpikyKern = -45.0f / (3.141592f * pow( m_Param[PSMOOTHRADIUS], 6.0f) );			// Laplacian of viscocity (denominator): PI h^6
	m_LapKern = 45.0f / (3.141592f * pow( m_Param[PSMOOTHRADIUS], 6.0f) );
}

void FluidSystem::SetupDefaultParams ()
{
	//  Range = +/- 10.0 * 0.006 (r) =	   0.12			m (= 120 mm = 4.7 inch)
	//  Container Volume (Vc) =			   0.001728		m^3
	//  Rest Density (D) =				1000.0			kg / m^3
	//  Particle Mass (Pm) =			   0.00020543	kg						(mass = vol * density)
	//  Number of Particles (N) =		4000.0
	//  Water Mass (M) =				   0.821		kg (= 821 grams)
	//  Water Volume (V) =				   0.000821     m^3 (= 3.4 cups, .21 gals)
	//  Smoothing Radius (R) =             0.02			m (= 20 mm = ~3/4 inch)
	//  Particle Radius (Pr) =			   0.00366		m (= 4 mm  = ~1/8 inch)
	//  Particle Volume (Pv) =			   2.054e-7		m^3	(= .268 milliliters)
	//  Rest Distance (Pd) =			   0.0059		m
	//
	//  Given: D, Pm, N
	//    Pv = Pm / D			0.00020543 kg / 1000 kg/m^3 = 2.054e-7 m^3	
	//    Pv = 4/3*pi*Pr^3    cuberoot( 2.054e-7 m^3 * 3/(4pi) ) = 0.00366 m
	//     M = Pm * N			0.00020543 kg * 4000.0 = 0.821 kg		
	//     V =  M / D              0.821 kg / 1000 kg/m^3 = 0.000821 m^3
	//     V = Pv * N			 2.054e-7 m^3 * 4000 = 0.000821 m^3
	//    Pd = cuberoot(Pm/D)    cuberoot(0.00020543/1000) = 0.0059 m 
	//
	// Ideal grid cell size (gs) = 2 * smoothing radius = 0.02*2 = 0.04
	// Ideal domain size = k*gs/d = k*0.02*2/0.005 = k*8 = {8, 16, 24, 32, 40, 48, ..}
	//    (k = number of cells, gs = cell size, d = simulation scale)

	// "The viscosity coefficient is the dynamic viscosity, visc > 0 (units Pa.s), 
	// and to include a reasonable damping contribution, it should be chosen 
	// to be approximately a factor larger than any physical correct viscosity 
	// coefficient that can be looked up in the literature. However, care should 
	// be taken not to exaggerate the viscosity coefficient for fluid materials.
	// If the contribution of the viscosity force density is too large, the net effect 
	// of the viscosity term will introduce energy into the system, rather than 
	// draining the system from energy as intended."
	//    Actual visocity of water = 0.001 Pa.s    // viscosity of water at 20 deg C.

	m_Time = 0.0f;							// Start at T=0
	m_DT = 0.003f;	

	m_Param [ PSIMSCALE ] =		0.005f;			// unit size
	m_Param [ PVISC ] =			0.50f;			// pascal-second (Pa.s) = 1 kg m^-1 s^-1  (see wikipedia page on viscosity)
	m_Param [ PRESTDENSITY ] =	400.0f;			// kg / m^3
	m_Param [ PSPACING ]	=	0.0f;			// spacing will be computed automatically from density in most examples (set to 0 for autocompute)
	m_Param [ PMASS ] =			0.00020543f;		// kg
	m_Param [ PRADIUS ] =		0.015f;			// m
	m_Param [ PDIST ] =			0.0059f;			// m
	m_Param [ PSMOOTHRADIUS ] =	0.015f;			// m 
	m_Param [ PINTSTIFF ] =		1.0f;
	m_Param [ PEXTSTIFF ] =		50000.0f;
	m_Param [ PEXTDAMP ] =		100.0f;
	m_Param [ PACCEL_LIMIT ] =	150.0f;			// m / s^2
	m_Param [ PVEL_LIMIT ] =	3.0f;			// m / s
	m_Param [ PMAX_FRAC ] =		1.0f;
	m_Param [ PGRAV ] =			1.0f;
	
	m_Param [ PGROUND_SLOPE ] = 0.0f;
	m_Param [ PFORCE_MIN ] =	0.0f;
	m_Param [ PFORCE_MAX ] =	0.0f;
	m_Param [ PFORCE_FREQ ] =	16.0f;
	m_Toggle [ PWRAP_X ] = false;
	m_Toggle [ PWALL_BARRIER ] = false;
	m_Toggle [ PLEVY_BARRIER ] = false;
	m_Toggle [ PDRAIN_BARRIER ] = false;

	m_Param [ PSTAT_NBRMAX ] = 0 ;
	m_Param [ PSTAT_SRCHMAX ] = 0 ;
	
	m_Vec [ PPOINT_GRAV_POS ].Set ( 0, 0, 0 );
	m_Vec [ PPLANE_GRAV_DIR ].Set ( 0, -9.8f, 0 );
	m_Vec [ PEMIT_POS ].Set ( 0, 0, 0 );
	m_Vec [ PEMIT_RATE ].Set ( 0, 0, 0 );
	m_Vec [ PEMIT_ANG ].Set ( 0, 90, 1.0f );
	m_Vec [ PEMIT_DANG ].Set ( 0, 0, 0 );

	// Default sim config
	m_Toggle [ PRUN ] = true;				// Run integrator
	m_Param [PGRIDSIZE] = m_Param[PSMOOTHRADIUS] * 2;
	m_Param [PDRAWMODE] = 1;				// Sprite drawing
	m_Param [PDRAWGRID] = 0;				// No grid 
	m_Param [PDRAWTEXT] = 0;				// No text

}

void FluidSystem::SetupExampleParams ()
{
	Vector3DF pos;
	Vector3DF min, max;
	
	switch ( (int) m_Param[PEXAMPLE] ) {

	case 0:	{	// Regression test. N x N x N static grid

		int k = (int) ceil ( pow ( (float) m_Param[PNUM], (float) 1.0f/3.0f ) );
		m_Vec [ PVOLMIN ].Set ( 0, 0, 0 );
		m_Vec [ PVOLMAX ].Set ( 2.0f+(k/2), 2.0f+(k/2), 2.0f+(k/2) );
		m_Vec [ PINITMIN ].Set ( 1.0f, 1.0f, 1.0f );
		m_Vec [ PINITMAX ].Set ( 1.0f+(k/2), 1.0f+(k/2), 1.0f+(k/2) );
		
		m_Param [ PGRAV ] = 0.0;		
		m_Vec [ PPLANE_GRAV_DIR ].Set ( 0.0, 0.0, 0.0 );			
		m_Param [ PSPACING ] = 0.5;				// Fixed spacing		Dx = x-axis density
		m_Param [ PSMOOTHRADIUS ] =	m_Param [PSPACING];		// Search radius
		m_Toggle [ PRUN ] = false;				// Do NOT run sim. Neighbors only.				
		m_Param [PDRAWMODE] = 1;				// Point drawing
		m_Param [PDRAWGRID] = 1;				// Grid drawing
		m_Param [PDRAWTEXT] = 1;				// Text drawing
		m_Param [PSIMSCALE ] = 1.0f;
	
		} break;
	case 1:		// Tower
		m_Vec [ PVOLMIN ].Set (   0,   0,   0 );
		m_Vec [ PVOLMAX ].Set (  256, 128, 256 );
		m_Vec [ PINITMIN ].Set (  5,   5,  5 );
		m_Vec [ PINITMAX ].Set ( 256*0.3f, 128*0.9f, 256*0.3f );		
		break;
	case 2:		// Wave pool
		m_Vec [ PVOLMIN ].Set (   0,   0,   0 );
		m_Vec [ PVOLMAX ].Set (  400, 200, 400 );
		m_Vec [ PINITMIN ].Set ( 100, 80,  100 );
		m_Vec [ PINITMAX ].Set ( 300, 190, 300 );
		m_Param [ PFORCE_MIN ] = 100.0f;	
		m_Param [ PFORCE_FREQ ] = 6.0f;
		m_Param [ PGROUND_SLOPE ] = 0.10f;
		break;
	case 3:		// Small dam break
		m_Vec [ PVOLMIN ].Set ( -40, 0, -40  );
		m_Vec [ PVOLMAX ].Set ( 40, 60, 40 );
		m_Vec [ PINITMIN ].Set ( 0, 8, -35 );
		m_Vec [ PINITMAX ].Set ( 35, 55, 35 );		
		m_Param [ PFORCE_MIN ] = 0.0f;
		m_Param [ PFORCE_MAX ] = 0.0f;
		m_Vec [ PPLANE_GRAV_DIR ].Set ( 0.0f, -9.8f, 0.0f );
		break;
	case 4:		// Dual-Wave pool
		m_Vec [ PVOLMIN ].Set ( -100, 0, -15 );
		m_Vec [ PVOLMAX ].Set ( 100, 100, 15 );
		m_Vec [ PINITMIN ].Set ( -80, 8, -10 );
		m_Vec [ PINITMAX ].Set ( 80, 90, 10 );
		m_Param [ PFORCE_MIN ] = 20.0;
		m_Param [ PFORCE_MAX ] = 20.0;
		m_Vec [ PPLANE_GRAV_DIR ].Set ( 0.0f, -9.8f, 0.0f );	
		break;
	case 5:		// Microgravity
		m_Vec [ PVOLMIN ].Set ( -80, 0, -80 );
		m_Vec [ PVOLMAX ].Set ( 80, 100, 80 );
		m_Vec [ PINITMIN ].Set ( -60, 40, -60 );
		m_Vec [ PINITMAX ].Set ( 60, 80, 60 );		
		m_Vec [ PPLANE_GRAV_DIR ].Set ( 0, -1, 0 );	
		m_Param [ PGROUND_SLOPE ] = 0.1f;
		break;
	}

}

void FluidSystem::SetupSpacing ()
{
	m_Param [ PSIMSIZE ] = m_Param [ PSIMSCALE ] * (m_Vec[PVOLMAX].z - m_Vec[PVOLMIN].z);	
	
	if ( m_Param[PSPACING] == 0 ) {
		// Determine spacing from density
		m_Param [PDIST] = pow ( (float) m_Param[PMASS] / m_Param[PRESTDENSITY], 1/3.0f );	
		m_Param [PSPACING] = m_Param [ PDIST ]*0.87f / m_Param[ PSIMSCALE ];			
	} else {
		// Determine density from spacing
		m_Param [PDIST] = m_Param[PSPACING] * m_Param[PSIMSCALE] / 0.87f;
		m_Param [PRESTDENSITY] = m_Param[PMASS] / pow ( (float) m_Param[PDIST], 3.0f );
	}
	nvprintf ( "Add Particles. Density: %f, Spacing: %f, PDist: %f\n", m_Param[PRESTDENSITY], m_Param [ PSPACING ], m_Param[ PDIST ] );

	// Particle Boundaries
	m_Vec[PBOUNDMIN] = m_Vec[PVOLMIN];		m_Vec[PBOUNDMIN] += 2.0*(m_Param[PGRIDSIZE] / m_Param[PSIMSCALE]);
	m_Vec[PBOUNDMAX] = m_Vec[PVOLMAX];		m_Vec[PBOUNDMAX] -= 2.0*(m_Param[PGRIDSIZE] / m_Param[PSIMSCALE]);
}


void FluidSystem::TestPrefixSum ( int num )
{
	nvprintf ( "------------------\n");
	nvprintf ( "TESTING PREFIX SUM\n");
	nvprintf ( "Num: %d\n", num );

	srand ( 2564 );		// deterministic test
	
	// Allocate input and output lists
	int* listIn = (int*) malloc( num * sizeof(int) );
	int* listOutCPU = (int*) malloc( num * sizeof(int) );
	int* listOutGPU = (int*) malloc( num * sizeof(int) );

	// Build list of pseudo-random numbers
	for (int n=0; n < num; n++) 
		listIn[n] = int ((rand()*4.0f) / RAND_MAX);
	nvprintf ( "Input: "); for (int n=num-10; n < num; n++)	printf ( "%d ", listIn[n] ); printf (" (last 10 values)\n");		// print first 10

	// Prefix Sum on CPU
	int sum = 0;	
	for (int n=0; n < num; n++) {		
		listOutCPU[n] = sum;
		sum += listIn[n];
	}
	
	// Prefix Sum on GPU	
	/*prefixSumToGPU ( (char*) listIn, num, sizeof(int) );	
	prefixSumInt ( num );	
	prefixSumFromGPU ( (char*) listOutGPU, num, sizeof(int) );	*/
	
	//nvprintf ( "GPU:   "); for (int n=num-10; n < num; n++)	printf ( "%d ", listOutGPU[n] ); printf (" (last 10 values)\n");		// print first 10

	// Validate results
	int ok = 0;
	for (int n=0; n < num; n++) {
		if ( listOutCPU[n] == listOutGPU[n] ) ok++;
	}
	nvprintf ( "Validate: %d OK. (Bad: %d)\n", ok, num-ok );
	nvprintf ( "Press any key to continue..\n");
	getchar();
}


void FluidSystem::SetupMode ( bool* cmds, Vector3DI range, std::string inf, std::string outf, std::string wpath, Vector3DI res, int br, float th )
{
	for (int n=0; n < 10; n++ ) m_Cmds[n] = cmds[n];	
	m_FrameRange = range;
	m_InFile = inf;
	m_OutFile  = outf;
	m_WorkPath = wpath;
	m_VolRes = res;
	m_BrkRes = br;
	m_Thresh = th;

	m_Frame = range.x;

	if ( m_Cmds[CMD_PLAYBACK] ) {
		m_Param[PMODE] = RUN_PLAYBACK;		
	}
	if ( m_Cmds[CMD_WRITEPTS] ) {
		mbRecord = true;
	}
	if ( m_Cmds[CMD_WRITEVOL] ) {
		mbRecordBricks = true;
	}
	if ( m_Cmds[CMD_WRITEIMG] ) {
		m_Toggle[ PCAPTURE ] = true;
	}
}

int iDivUp (int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}
void computeNumBlocks (int numPnts, int minThreads, int &numBlocks, int &numThreads)
{
    numThreads = min( minThreads, numPnts );
    numBlocks = (numThreads==0) ? 1 : iDivUp ( numPnts, numThreads );
}

/*void FluidSystem::cudaExit ()
{
	cudaDeviceReset();
}*/


void FluidSystem::TransferToTempCUDA ( int buf_id, int sz )
{
	cuCheck ( cuMemcpyDtoD ( m_FluidTemp.gpu(buf_id), m_Fluid.gpu(buf_id), sz ), "TransferToTempCUDA", "cuMemcpyDtoD", "m_FluidTemp", mbDebug);
}


// Initialize CUDA
/*void FluidSystem::cudaInit()
{   
	int count = 0;
	int i = 0;

	cudaError_t err = cudaGetDeviceCount(&count);
	if ( err==cudaErrorInsufficientDriver)	{ nvprintf ( "CUDA driver not installed.\n"); }
	if ( err==cudaErrorNoDevice)			{ nvprintf ( "No CUDA device found.\n"); }
	if ( count == 0)						{ nvprintf ( "No CUDA device found.\n"); }

	for(i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess)
			if(prop.major >= 1) break;
	}
	if(i == count) { nvprintf ( "No CUDA device found.\n");  }
	cudaSetDevice(i);

	nvprintf( "CUDA initialized.\n");
 
	cudaDeviceProp p;
	cudaGetDeviceProperties ( &p, 0);
	
	nvprintf ( "-- CUDA --\n" );
	nvprintf ( "Name:       %s\n", p.name );
	nvprintf ( "Revision:   %d.%d\n", p.major, p.minor );
	nvprintf ( "Global Mem: %d\n", p.totalGlobalMem );
	nvprintf ( "Shared/Blk: %d\n", p.sharedMemPerBlock );
	nvprintf ( "Regs/Blk:   %d\n", p.regsPerBlock );
	nvprintf ( "Warp Size:  %d\n", p.warpSize );
	nvprintf ( "Mem Pitch:  %d\n", p.memPitch );
	nvprintf ( "Thrds/Blk:  %d\n", p.maxThreadsPerBlock );
	nvprintf ( "Const Mem:  %d\n", p.totalConstMem );
	nvprintf ( "Clock Rate: %d\n", p.clockRate );	

};*/
	
void FluidSystem::FluidSetupCUDA ( int num, int gsrch, int3 res, float3 size, float3 delta, float3 gmin, float3 gmax, int total, int chk )
{	
	m_FParams.pnum = num;	
	m_FParams.gridRes = res;
	m_FParams.gridSize = size;
	m_FParams.gridDelta = delta;
	m_FParams.gridMin = gmin;
	m_FParams.gridMax = gmax;
	m_FParams.gridTotal = total;
	m_FParams.gridSrch = gsrch;
	m_FParams.gridAdjCnt = gsrch*gsrch*gsrch;
	m_FParams.gridScanMax = res;
	m_FParams.gridScanMax -= make_int3( m_FParams.gridSrch, m_FParams.gridSrch, m_FParams.gridSrch );
	m_FParams.chk = chk;

	// Build Adjacency Lookup
	int cell = 0;
	for (int y=0; y < gsrch; y++ ) 
		for (int z=0; z < gsrch; z++ ) 
			for (int x=0; x < gsrch; x++ ) 
				m_FParams.gridAdj [ cell++]  = ( y * m_FParams.gridRes.z+ z )*m_FParams.gridRes.x +  x ;			
	
	// Compute number of blocks and threads
	int threadsPerBlock = 512;

    computeNumBlocks ( m_FParams.pnum, threadsPerBlock, m_FParams.numBlocks, m_FParams.numThreads);				// particles
    computeNumBlocks ( m_FParams.gridTotal, threadsPerBlock, m_FParams.gridBlocks, m_FParams.gridThreads);		// grid cell
    
	// Compute particle buffer & grid dimensions
    m_FParams.szPnts = (m_FParams.numBlocks  * m_FParams.numThreads);     
    nvprintf ( "CUDA Config: \n" );
	nvprintf ( "  Pnts: %d, t:%dx%d=%d, Size:%d\n", m_FParams.pnum, m_FParams.numBlocks, m_FParams.numThreads, m_FParams.numBlocks*m_FParams.numThreads, m_FParams.szPnts);
    nvprintf ( "  Grid: %d, t:%dx%d=%d, bufGrid:%d, Res: %dx%dx%d\n", m_FParams.gridTotal, m_FParams.gridBlocks, m_FParams.gridThreads, m_FParams.gridBlocks*m_FParams.gridThreads, m_FParams.szGrid, (int) m_FParams.gridRes.x, (int) m_FParams.gridRes.y, (int) m_FParams.gridRes.z );		
	
	// Initialize random numbers
	int blk = int(num/16)+1;
	//randomInit<<< blk, 16 >>> ( rand(), gFluidBufs., num );

}

void FluidSystem::FluidParamCUDA ( float ss, float sr, float pr, float mass, float rest, float3 bmin, float3 bmax, float estiff, float istiff, float visc, float damp, float fmin, float fmax, float ffreq, float gslope, float gx, float gy, float gz, float al, float vl, int emit )
{
	m_FParams.psimscale = ss;
	m_FParams.psmoothradius = sr;
	m_FParams.pradius = pr;
	m_FParams.r2 = sr * sr;
	m_FParams.pmass = mass;
	m_FParams.prest_dens = rest;	
	m_FParams.pboundmin = bmin;
	m_FParams.pboundmax = bmax;
	m_FParams.pextstiff = estiff;
	m_FParams.pintstiff = istiff;
	m_FParams.pvisc = visc;
	m_FParams.pdamp = damp;
	m_FParams.pforce_min = fmin;
	m_FParams.pforce_max = fmax;
	m_FParams.pforce_freq = ffreq;
	m_FParams.pground_slope = gslope;	
	m_FParams.pgravity = make_float3( gx, gy, gz );
	m_FParams.AL = al;
	m_FParams.AL2 = al * al;
	m_FParams.VL = vl;
	m_FParams.VL2 = vl * vl;
	m_FParams.pemit = emit;

	m_FParams.pdist = pow ( m_FParams.pmass / m_FParams.prest_dens, 1/3.0f );
	m_FParams.poly6kern = 315.0f / (64.0f * 3.141592f * pow( sr, 9.0f) );
	m_FParams.spikykern = -45.0f / (3.141592f * pow( sr, 6.0f) );
	m_FParams.lapkern = 45.0f / (3.141592f * pow( sr, 6.0f) );	
	m_FParams.gausskern = 1.0f / pow(3.141592f * 2.0f*sr*sr, 3.0f/2.0f);

	m_FParams.d2 = m_FParams.psimscale * m_FParams.psimscale;
	m_FParams.rd2 = m_FParams.r2 / m_FParams.d2;
	m_FParams.vterm = m_FParams.lapkern * m_FParams.pvisc;

	// Transfer sim params to device
	cuCheck ( cuMemcpyHtoD ( cuFParams,	&m_FParams,		sizeof(FParams) ), "FluidParamCUDA", "cuMemcpyHtoD", "cuFParams", mbDebug);
}

void FluidSystem::TransferToCUDA ()
{
	// Send particle buffers	
	cuCheck(cuMemcpyHtoD(m_Fluid.gpu(FPOS), m_Fluid.bufC(FPOS),				mNumPoints *sizeof(float) * 3),			"TransferToCUDA", "cuMemcpyHtoD", "FPOS", mbDebug);
	cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FVEL),	m_Fluid.bufC(FVEL),			mNumPoints *sizeof(float)*3 ),	"TransferToCUDA", "cuMemcpyHtoD", "FVEL", mbDebug);
	cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FVEVAL),	m_Fluid.bufC(FVEVAL),	mNumPoints *sizeof(float)*3 ), "TransferToCUDA", "cuMemcpyHtoD", "FVELAL", mbDebug);
	cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FFORCE),	m_Fluid.bufC(FFORCE),	mNumPoints *sizeof(float)*3 ), "TransferToCUDA", "cuMemcpyHtoD", "FFORCE", mbDebug);
	cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FPRESS),	m_Fluid.bufC(FPRESS),	mNumPoints *sizeof(float) ),	"TransferToCUDA", "cuMemcpyHtoD", "FPRESS", mbDebug);
	cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FDENSITY), m_Fluid.bufC(FDENSITY),	mNumPoints *sizeof(float) ),	"TransferToCUDA", "cuMemcpyHtoD", "FDENSITY", mbDebug);
	cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(FCLR),	m_Fluid.bufC(FCLR),			mNumPoints *sizeof(uint) ),		"TransferToCUDA", "cuMemcpyHtoD", "FCLR", mbDebug);
}

void FluidSystem::TransferFromCUDA ()
{
	// Return particle buffers	
	cuCheck( cuMemcpyDtoH ( m_Fluid.bufC(FPOS),	m_Fluid.gpu(FPOS),	mNumPoints *sizeof(float)*3 ), "TransferFromCUDA", "cuMemcpyDtoH", "FPOS", mbDebug);
	cuCheck( cuMemcpyDtoH ( m_Fluid.bufC(FVEL),	m_Fluid.gpu(FVEL),	mNumPoints *sizeof(float)*3 ), "TransferFromCUDA", "cuMemcpyDtoH", "FVEL", mbDebug);
	cuCheck( cuMemcpyDtoH ( m_Fluid.bufC(FCLR),	m_Fluid.gpu(FCLR),	mNumPoints *sizeof(uint) ),	"TransferFromCUDA", "cuMemcpyDtoH", "FCLR", mbDebug);
}



void FluidSystem::InsertParticlesCUDA ( uint* gcell, uint* gndx, uint* gcnt )
{
	cuCheck ( cuMemsetD8 ( m_Fluid.gpu(FGRIDCNT), 0,	m_GridTotal*sizeof(int) ), "InsertParticlesCUDA", "cuMemsetD8", "FGRIDCNT", mbDebug );
	cuCheck ( cuMemsetD8 ( m_Fluid.gpu(FGRIDOFF), 0,	m_GridTotal*sizeof(int) ), "InsertParticlesCUDA", "cuMemsetD8", "FGRIDOFF", mbDebug );

	void* args[1] = { &mNumPoints };
	cuCheck(cuLaunchKernel(m_Func[FUNC_INSERT], m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL),
		"InsertParticlesCUDA", "cuLaunch", "FUNC_INSERT", mbDebug);

	// Transfer data back if requested (for validation)
	if (gcell != 0x0) {
		cuCheck( cuMemcpyDtoH ( gcell,	m_Fluid.gpu(FGCELL),		mNumPoints *sizeof(uint) ), "InsertParticlesCUDA", "cuMemcpyDtoH", "FGCELL", mbDebug );
		cuCheck( cuMemcpyDtoH ( gndx,		m_Fluid.gpu(FGNDX),		mNumPoints *sizeof(uint) ), "InsertParticlesCUDA", "cuMemcpyDtoH", "FGNDX", mbDebug);
		cuCheck( cuMemcpyDtoH ( gcnt,		m_Fluid.gpu(FGRIDCNT),	m_GridTotal*sizeof(uint) ), "InsertParticlesCUDA", "cuMemcpyDtoH", "FGRIDCNT", mbDebug);
		cuCtxSynchronize ();
	}
}
	

// #define CPU_SUMS

void FluidSystem::PrefixSumCellsCUDA ( uint* goff, int zero_offsets )
{
	#ifdef CPU_SUMS
	
		PERF_PUSH ( "PrefixSum (CPU)" );
		int numCells = m_GridTotal;
		
		cuCheck(cuMemcpyDtoH( m_Fluid.bufC(FGRIDCNT), m_Fluid.gpu(FGRIDCNT), numCells*sizeof(int)), "DtoH mgridcnt");
		cuCheck( cuCtxSynchronize(), "cuCtxSync(PrefixSum)" );

		uint* mgcnt = m_Fluid.bufI(FGRIDCNT);
		uint* mgoff = m_Fluid.bufI(FGRIDOFF);
		int sum = 0;
		for (int n=0; n < numCells; n++) {
			mgoff[n] = sum;
			sum += mgcnt[n];
		}
		cuCheck(cuMemcpyHtoD(m_Fluid.gpu(FGRIDOFF), m_Fluid.bufI(FGRIDOFF), numCells*sizeof(int)), "HtoD mgridoff");
		cuCheck( cuCtxSynchronize(), "cuCtxSync(PrefixSum)" );
		PERF_POP ();

		if ( goff != 0x0 ) {
			memcpy ( goff, mgoff, numCells*sizeof(uint) );
		}

	#else

		// Prefix Sum - determine grid offsets
		int blockSize = SCAN_BLOCKSIZE << 1;
		int numElem1 = m_GridTotal;		
		int numElem2 = int ( numElem1 / blockSize ) + 1;
		int numElem3 = int ( numElem2 / blockSize ) + 1;
		int threads = SCAN_BLOCKSIZE;
		int zon=1;

		CUdeviceptr array1  = m_Fluid.gpu(FGRIDCNT);		// input
		CUdeviceptr scan1   = m_Fluid.gpu(FGRIDOFF);		// output
		CUdeviceptr array2  = m_Fluid.gpu(FAUXARRAY1);
		CUdeviceptr scan2   = m_Fluid.gpu(FAUXSCAN1);
		CUdeviceptr array3  = m_Fluid.gpu(FAUXARRAY2);
		CUdeviceptr scan3   = m_Fluid.gpu(FAUXSCAN2);

		if ( numElem1 > SCAN_BLOCKSIZE*xlong(SCAN_BLOCKSIZE)*SCAN_BLOCKSIZE) {
			nvprintf ( "ERROR: Number of elements exceeds prefix sum max. Adjust SCAN_BLOCKSIZE.\n" );
		}

		void* argsA[5] = {&array1, &scan1, &array2, &numElem1, &zero_offsets }; // sum array1. output -> scan1, array2
		cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXSUM], numElem2, 1, 1, threads, 1, 1, 0, NULL, argsA, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXSUM", mbDebug);

		void* argsB[5] = { &array2, &scan2, &array3, &numElem2, &zon }; // sum array2. output -> scan2, array3
		cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXSUM], numElem3, 1, 1, threads, 1, 1, 0, NULL, argsB, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXSUM", mbDebug);

		if ( numElem3 > 1 ) {
			CUdeviceptr nptr = {0};
			void* argsC[5] = { &array3, &scan3, &nptr, &numElem3, &zon };	// sum array3. output -> scan3
			cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXSUM], 1, 1, 1, threads, 1, 1, 0, NULL, argsC, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);

			void* argsD[3] = { &scan2, &scan3, &numElem2 };	// merge scan3 into scan2. output -> scan2
			cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXFIXUP], numElem3, 1, 1, threads, 1, 1, 0, NULL, argsD, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);
		}

		void* argsE[3] = { &scan1, &scan2, &numElem1 };		// merge scan2 into scan1. output -> scan1
		cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXFIXUP], numElem2, 1, 1, threads, 1, 1, 0, NULL, argsE, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);

		// Transfer data back if requested
		if ( goff != 0x0 ) {
			cuCheck( cuMemcpyDtoH ( goff,		m_Fluid.gpu(FGRIDOFF),	numElem1*sizeof(int) ), "PrefixSumCellsCUDA", "cuMemcpyDtoH", "FGRIDOFF", mbDebug);
			cuCtxSynchronize ();
		}
	#endif
}

void FluidSystem::IntegrityCheck()
{
	// Retrieve data from GPU
	cuCheck(cuMemcpyDtoH( m_Fluid.bufC(FPOS),   m_Fluid.gpu(FPOS),		mNumPoints *sizeof(Vector3DF) ), "IntegrityCheck", "cuMemcpyDtoH", "FPOS", mbDebug);
	cuCheck(cuMemcpyDtoH (m_Fluid.bufC(FGCELL), m_Fluid.gpu(FGCELL),	mNumPoints *sizeof(uint) ),		"IntegrityCheck", "cuMemcpyDtoH", "FGCELL", mbDebug);
	cuCheck(cuMemcpyDtoH( m_Fluid.bufC(FGNDX),  m_Fluid.gpu(FGNDX),		mNumPoints *sizeof(uint) ),		"IntegrityCheck", "cuMemcpyDtoH", "FGNDX", mbDebug);
	
	int numElem = m_GridTotal;
	cuCheck(cuMemcpyDtoH( m_Fluid.bufC(FGRID),    m_Fluid.gpu(FGRID),	 mNumPoints *sizeof(int)),	"IntegrityCheck", "cuMemcpyDtoH", "FGRID", mbDebug);
	cuCheck(cuMemcpyDtoH( m_Fluid.bufC(FGRIDOFF), m_Fluid.gpu(FGRIDOFF), numElem*sizeof(int)),		"IntegrityCheck", "cuMemcpyDtoH", "FGRIDOFF", mbDebug);
	cuCheck(cuMemcpyDtoH( m_Fluid.bufC(FGRIDCNT), m_Fluid.gpu(FGRIDCNT), numElem*sizeof(int)),		"IntegrityCheck", "cuMemcpyDtoH", "FGRIDCNT", mbDebug);
	cuCheck(cuCtxSynchronize(), "IntegrityCheck", "cuCtxSynchronize", "", mbDebug);

	// Analysis by grid cells	
	uint p;
	nvprintf("     - Integrity Check - \n");
	for (int g = 0; g < numElem; g++) {
		int goff = m_Fluid.bufI(FGRIDOFF)[g];
		int gcnt = m_Fluid.bufI(FGRIDCNT)[g];		
		for (int gndx = 0; gndx < gcnt; gndx++) {
			p = m_Fluid.bufI(FGRID)[goff + gndx];		
			if (p != GRID_UNDEF)
				if (m_Fluid.bufI(FGCELL)[p] != g || m_Fluid.bufI(FGNDX)[p] != gndx) {
					nvprintf("ERROR-  grid %d, off %d, cnt %d, #%d, p %d, cell %d (%d), ndx %d (%d)\n", g, goff, gcnt, gndx, p, m_Fluid.bufI(FGCELL)[p], g, m_Fluid.bufI(FGNDX)[p], gndx);
				}
		}
	}
}

void FluidSystem::CountingSortFullCUDA ( Vector3DF* ppos )
{
	// Transfer particle data to temp buffers
	//  (gpu-to-gpu copy, no sync needed)	
	TransferToTempCUDA ( FPOS,		mNumPoints *sizeof(Vector3DF) );
	TransferToTempCUDA ( FVEL,		mNumPoints *sizeof(Vector3DF) );
	TransferToTempCUDA ( FVEVAL,	mNumPoints *sizeof(Vector3DF) );
	TransferToTempCUDA ( FFORCE,	mNumPoints *sizeof(Vector3DF) );
	TransferToTempCUDA ( FPRESS,	mNumPoints *sizeof(float) );
	TransferToTempCUDA ( FDENSITY,	mNumPoints *sizeof(float) );
	TransferToTempCUDA ( FCLR,		mNumPoints *sizeof(uint) );
	TransferToTempCUDA ( FGCELL,	mNumPoints *sizeof(uint) );
	TransferToTempCUDA ( FGNDX,		mNumPoints *sizeof(uint) );

	// Reset grid cell IDs
	//cuCheck(cuMemsetD32(m_Fluid.gpu(FGCELL), GRID_UNDEF, numPoints ), "cuMemsetD32(Sort)");

	void* args[1] = { &mNumPoints };
	cuCheck ( cuLaunchKernel ( m_Func[FUNC_COUNTING_SORT], m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL),
				"CountingSortFullCUDA", "cuLaunch", "FUNC_COUNTING_SORT", mbDebug );

	if ( ppos != 0x0 ) {
		cuCheck( cuMemcpyDtoH ( ppos,		m_Fluid.gpu(FPOS),	mNumPoints*sizeof(Vector3DF) ), "CountingSortFullCUDA", "cuMemcpyDtoH", "FPOS", mbDebug);
		cuCtxSynchronize ();
	}
}

int FluidSystem::ResizeBrick ( uint3 res )
{
	int sz = res.x*res.y*res.z * sizeof(float);


	if ( res.x != m_FParams.brickRes.x || res.y != m_FParams.brickRes.y || res.z != m_FParams.brickRes.z ) {
		m_FParams.brickRes = make_int3(res.x, res.y, res.z);

		if ( m_Fluid.gpu(FBRICK) != 0x0 ) {
			cuCheck ( cuMemFree ( m_Fluid.gpu(FBRICK) ), "ResizeBrick", "cuMemFree", "FBRICK", mbDebug);
		}		
		cuCheck ( cuMemAlloc ( m_Fluid.gpuptr(FBRICK), sz ), "ResizeBrick", "cuMemAlloc", "FBRICK", mbDebug);
	}
	return sz;
}

void FluidSystem::SampleParticlesCUDA ( float* outbuf, uint3 res, float3 bmin, float3 bmax, float scalar )
{
	int sz = ResizeBrick ( res );
	
	dim3 grid, blocks;
	blocks = make_uint3(8,8,8);
	grid = make_uint3( int(res.x/8)+1, int(res.y/8)+1, int(res.z/8)+1 );

	void* args[6] = { &res, &bmin, &bmax, &mNumPoints, &scalar };
	cuCheck ( cuLaunchKernel ( m_Func[FUNC_SAMPLE],  m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL), "SampleParticlesCUDA", "cuLaunch", "FUNC_SAMPLES", mbDebug);
}

void FluidSystem::ComputeQueryCUDA ()
{
	void* args[1] = { &mNumPoints };
	cuCheck ( cuLaunchKernel ( m_Func[FUNC_QUERY],  m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL), "ComputeQueryCUDA", "cuLaunch", "FUNC_QUERY", mbDebug);
}

void FluidSystem::ComputePressureCUDA ()
{
	void* args[1] = { &mNumPoints };
	cuCheck ( cuLaunchKernel ( m_Func[FUNC_COMPUTE_PRESS],  m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL), "ComputePressureCUDA", "cuLaunch", "FUNC_COMPUTE_PRESS", mbDebug);
}

void FluidSystem::ComputeForceCUDA ()
{
	void* args[1] = { &m_FParams.pnum };
	cuCheck ( cuLaunchKernel ( m_Func[FUNC_COMPUTE_FORCE],  m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL), "ComputeForceCUDA", "cuLaunch", "FUNC_COMPUTE_FORCE", mbDebug);
}

void FluidSystem::AdvanceCUDA ( float tm, float dt, float ss )
{
	void* args[4] = { &tm, &dt, &ss, &m_FParams.pnum };
	cuCheck ( cuLaunchKernel ( m_Func[FUNC_ADVANCE],  m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL), "AdvanceCUDA", "cuLaunch", "FUNC_ADVANCE", mbDebug);
}

void FluidSystem::EmitParticlesCUDA ( float tm, int cnt )
{
	void* args[3] = { &tm, &cnt, &m_FParams.pnum };
	cuCheck ( cuLaunchKernel ( m_Func[FUNC_EMIT],  m_FParams.numBlocks, 1, 1, m_FParams.numThreads, 1, 1, 0, NULL, args, NULL), "EmitParticlesCUDA", "cuLaunch", "FUNC_EMIT", mbDebug);
}

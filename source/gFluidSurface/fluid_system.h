
//-------------------------------------------------------------------
// FLUIDS v.3 - SPH Fluid Simulator for CPU and GPU
// Copyright (C) 2012-2013. Rama Hoetzlein, http://fluids3.com
//
// Attribute-ZLib license (* See additional part 4)
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.
// 4. Any published work based on this code must include public acknowledgement
//    of the origin. This includes following when applicable:
//	   - Journal/Paper publications. Credited by reference to work in text & citation.
//	   - Public presentations. Credited in at least one slide.
//	   - Distributed Games/Apps. Credited as single line in game or app credit page.	 
//	 Retaining this additional license term is required in derivative works.
//	 Acknowledgement may be provided as:
//	   Publication version:  
//	      2012-2013, Hoetzlein, Rama C. Fluids v.3 - A Large-Scale, Open Source
//	 	  Fluid Simulator. Published online at: http://fluids3.com
//	   Single line (slides or app credits):
//	      GPU Fluids: Rama C. Hoetzlein (Fluids v3 2013)
//--------------------------------------------------------------------

#ifndef DEF_FLUID_SYS
	#define DEF_FLUID_SYS

	#include <iostream>
	#include <vector>
	#include <stdio.h>
	#include <stdlib.h>
	#include <math.h>	

	#include "fluid.h"
	#include "gvdb_vec.h"
	#include "gvdb_camera.h"
	using namespace nvdb;

	#define MAX_PARAM			50
	#define GRID_UCHAR			0xFF
	#define GRID_UNDEF			4294967295	

	#define CMD_SIM				0
	#define CMD_PLAYBACK		1
	#define CMD_WRITEPTS		2
	#define CMD_WRITEVOL		3	
	#define CMD_WRITEIMG		4

	#define RUN_PAUSE			0
	#define RUN_SEARCH			1
	#define RUN_VALIDATE		2
	#define RUN_CPU_SLOW		3
	#define RUN_CPU_GRID		4	
	#define RUN_GPU_FULL		5
	#define RUN_PLAYBACK		6

	// Scalar params
	#define PMODE				0
	#define PNUM				1
	#define PEXAMPLE			2
	#define PSIMSIZE			3
	#define PSIMSCALE			4
	#define PGRID_DENSITY		5
	#define PGRIDSIZE			6
	#define PVISC				7
	#define PRESTDENSITY		8
	#define PMASS				9
	#define PRADIUS				10
	#define PDIST				11
	#define PSMOOTHRADIUS		12
	#define PINTSTIFF			13
	#define PEXTSTIFF			14
	#define PEXTDAMP			15
	#define PACCEL_LIMIT		16
	#define PVEL_LIMIT			17
	#define PSPACING			18
	#define PGROUND_SLOPE		19
	#define PFORCE_MIN			20
	#define PFORCE_MAX			21
	#define PMAX_FRAC			22
	#define PDRAWMODE			23
	#define PDRAWSIZE			24
	#define PDRAWGRID			25	
	#define PDRAWTEXT			26	
	#define PCLR_MODE			27
	#define PGRAV				28
	#define PSTAT_OCCUPY		29
	#define PSTAT_GRIDCNT		30
	#define PSTAT_NBR			31
	#define PSTAT_NBRMAX		32
	#define PSTAT_SRCH			33
	#define PSTAT_SRCHMAX		34
	#define PSTAT_PMEM			35
	#define PSTAT_GMEM			36
	#define PTIME_INSERT		37
	#define PTIME_SORT			38
	#define PTIME_COUNT			39
	#define PTIME_PRESS			40
	#define PTIME_FORCE			41
	#define PTIME_ADVANCE		42
	#define PTIME_RECORD		43
	#define PTIME_RENDER		44
	#define PTIME_TOGPU			45
	#define PTIME_FROMGPU		46
	#define PFORCE_FREQ			47	

	// Vector params
	#define PVOLMIN				0
	#define PVOLMAX				1
	#define PBOUNDMIN			2
	#define PBOUNDMAX			3
	#define PINITMIN			4
	#define PINITMAX			5
	#define PEMIT_POS			6
	#define PEMIT_ANG			7
	#define PEMIT_DANG			8
	#define PEMIT_SPREAD		9
	#define PEMIT_RATE			10
	#define PPOINT_GRAV_POS		11	
	#define PPLANE_GRAV_DIR		12	

	// Booleans
	#define PRUN				0
	#define PDEBUG				1	
	#define PUSE_CUDA			2	
	#define	PUSE_GRID			3
	#define PWRAP_X				4
	#define PWALL_BARRIER		5
	#define PLEVY_BARRIER		6
	#define PDRAIN_BARRIER		7		
	#define PPLANE_GRAV_ON		11	
	#define PPROFILE			12
	#define PCAPTURE			13

	#define BFLUID				2

	#define FUNC_INSERT			0
	#define	FUNC_COUNTING_SORT	1
	#define FUNC_QUERY			2
	#define FUNC_COMPUTE_PRESS	3
	#define FUNC_COMPUTE_FORCE	4
	#define FUNC_ADVANCE		5
	#define FUNC_EMIT			6
	#define FUNC_RANDOMIZE		7
	#define FUNC_SAMPLE			8
	#define FUNC_FPREFIXSUM		9
	#define FUNC_FPREFIXFIXUP	10
	#define FUNC_MAX			12

	#define COLORA(r,g,b,a)	( (uint((a)*255.0f)<<24) | (uint((b)*255.0f)<<16) | (uint((g)*255.0f)<<8) | uint((r)*255.0f) )
	/*#define ALPH(c)			(float(((c)>>24) & 0xFF)/255.0f)
	#define BLUE(c)			(float(((c)>>16) & 0xFF)/255.0f)
	#define GRN(c)			(float(((c)>>8)  & 0xFF)/255.0f)
	#define RED(c)			(float( (c)      & 0xFF)/255.0f)	*/

	#define GPU_OFF				0
	#define GPU_SINGLE			1
	#define GPU_TEMP			2
	#define GPU_DUAL			3
	#define CPU_OFF				4
	#define CPU_YES				5

	class FluidSystem {
	public:
		FluidSystem ();
		~FluidSystem();
		
		void LoadKernel ( int id, std::string kname );
		void Initialize ();

		// Rendering
		void Draw ( int frame, Camera3D& cam, float rad );
		void DrawDomain ();
		void DrawGrid ();
		void DrawText ();
		void DrawCell ( int gx, int gy, int gz );
		void DrawParticle ( int p, int r1, int r2, Vector3DF clr2 );
		void DrawParticleInfo ()		{ DrawParticleInfo ( mSelected ); }
		void DrawParticleInfo ( int p );
		void DrawNeighbors ( int p );
		void DrawCircle ( Vector3DF pos, float r, Vector3DF clr, Camera3D& cam );

		// Particle Utilities
		void AllocateBuffer(int buf_id, int stride, int cpucnt, int gpucnt, int gpumode, int cpumode);		
		void TransferToTempCUDA ( int buf_id, int sz );
		void AllocateParticles ( int cnt );
		int AddParticle ();
		void AddEmit ( float spacing );
		int NumPoints ()				{ return mNumPoints; }
		nvdb::Vector3DF* getPos ( int n )	{ return &m_Fluid.bufV3(FPOS)[n]; }
		nvdb::Vector3DF* getVel ( int n )	{ return &m_Fluid.bufV3(FVEL)[n]; }
		uint* getClr ( int n )			{ return &m_Fluid.bufI(FCLR)[n]; }
		
		// Setup
		void Start ( int num );
		void SetupRender ();
		void SetupKernels ();
		void SetupDefaultParams ();
		void SetupExampleParams ();
		void SetupSpacing ();
		void SetupAddVolume ( Vector3DF min, Vector3DF max, float spacing, float offs, int total );
		void SetupGrid ( Vector3DF min, Vector3DF max, float sim_scale, float cell_size, float border );		
		void AllocateGrid ();
		void IntegrityCheck();

		// Neighbor Search
		void Search ();
		void InsertParticles ();
		void BasicSortParticles ();
		void BinSortParticles ();
		void FindNbrsSlow ();
		void FindNbrsGrid ();

		// Simulation
		void Run ();		
		void ValidateCUDA ();		
		void RunPlayback ();
		void AdvanceTime ();
		
		void Advance ();
		void EmitParticles ();
		void Exit ();
		void TransferToCUDA ();
		void TransferFromCUDA ();
		void ValidateSortCUDA ();
		float Sample ( Vector3DF p );
		double GetDT()		{ return m_DT; }

		// Debugging
		void SaveResults ();
		void CaptureVideo (int width, int height);
		void ValidateResults ();
		void TestPrefixSum ( int num );
		void DebugPrintMemory ();		
		int SelectParticle ( int x, int y, int wx, int wy, Camera3D& cam );
		int GetSelected ()		{ return mSelected; }

		
		// Acceleration Grid
		int getGridCell ( int p, Vector3DI& gc );
		int getGridCell ( Vector3DF& p, Vector3DI& gc );
		int getGridTotal ()		{ return m_GridTotal; }
		int getSearchCnt ()		{ return m_GridAdjCnt; }
		Vector3DI getCell ( int gc );
		Vector3DF GetGridRes ()		{ return m_GridRes; }
		Vector3DF GetGridMin ()		{ return m_GridMin; }
		Vector3DF GetGridMax ()		{ return m_GridMax; }
		Vector3DF GetGridDelta ()	{ return m_GridDelta; }

		// Acceleration Neighbor Tables
		void AllocateNeighborTable ();
		void ClearNeighborTable ();
		void ResetNeighbors ();
		int GetNeighborTableSize ()	{ return m_NeighborNum; }
		void ClearNeighbors ( int i );
		int AddNeighbor();
		int AddNeighbor( int i, int j, float d );
		
		// Smoothed Particle Hydrodynamics		
		void ComputePressureGrid ();			// O(kn) - spatial grid
		void ComputeForceGrid ();				// O(kn) - spatial grid
		void ComputeForceGridNC ();				// O(cn) - neighbor table		

		void FluidSetupCUDA (  int num, int gsrch, int3 res, float3 size, float3 delta, float3 gmin, float3 gmax, int total, int chk );
		void FluidParamCUDA ( float ss, float sr, float pr, float mass, float rest, float3 bmin, float3 bmax, float estiff, float istiff, float visc, float damp, float fmin, float fmax, float ffreq, float gslope, float gx, float gy, float gz, float al, float vl, int emit );

		void InsertParticlesCUDA ( uint* gcell, uint* ccell, uint* gcnt );	
		void PrefixSumCellsCUDA ( uint* goff, int zero_offsets );		
		void CountingSortFullCUDA ( Vector3DF* gpos );
		void ComputePressureCUDA ();
		void ComputeQueryCUDA ();
		void ComputeForceCUDA ();	
		void SampleParticlesCUDA ( float* outbuf, uint3 res, float3 bmin, float3 bmax, float scalar );		
		void AdvanceCUDA ( float time, float dt, float ss );
		void EmitParticlesCUDA ( float time, int cnt );

		int ResizeBrick ( uint3 res );


		//void SPH_ComputePressureSlow ();			// O(n^2)	
		//void SPH_ComputeForceSlow ();				// O(n^2)
		//void SPH_ComputeForceGrid ();				// O(kn) - spatial grid

		// Recording
		void StartRecord ();
		void StartRecordBricks ();
		void StartPlayback ();
		void SavePoints ( int frame );
		void SaveBricks ( int frame );

		int getMode ()		{ return (int) m_Param[PMODE]; }
		std::string getModeStr ();
		void getModeClr ();

		// Parameters			
		void UpdateParams ();
		void SetParam (int p, float v );
		void SetParam (int p, int v )		{ m_Param[p] = (float) v; }
		float GetParam ( int p )			{ return (float) m_Param[p]; }
		float* getParamPtr ( int p )		{ return &m_Param[p]; }
		float SetParam ( int p, float v, float mn, float mx )	{ m_Param[p] = v ; if ( m_Param[p] > mx ) m_Param[p] = mn; return m_Param[p];}
		float IncParam ( int p, float v, float mn, float mx )	{ 
			m_Param[p] += v; 
			if ( m_Param[p] < mn ) m_Param[p] = mn; 
			if ( m_Param[p] > mx ) m_Param[p] = mn; 
			return m_Param[p];
		}
		void IncVec ( int p, Vector3DF v )	{ m_Vec[p] += v; }
		Vector3DF GetVec ( int p )			{ return m_Vec[p]; }
		void SetVec ( int p, Vector3DF v );
		void Toggle ( int p )				{ m_Toggle[p] = !m_Toggle[p]; }		
		bool GetToggle ( int p )			{ return m_Toggle[p]; }
		std::string		getSceneName ()		{ return mSceneName; }

		void SetupMode ( bool* cmds, Vector3DI range, std::string inf, std::string outf, std::string wpath, Vector3DI res, int brickres, float th);
		std::string getResolvedName ( bool bIn, int frame );

		CUdeviceptr getBufferGPU ( int id )	{ return m_Fluid.gpu(id); }

		void SetDebug(bool b) { mbDebug = b; }
	
	private:
		bool						m_Cmds[10];
		Vector3DI					m_FrameRange;
		Vector3DI					m_VolRes;
		int							m_BrkRes;
		std::string					m_InFile;
		std::string					m_OutFile;
		std::string					m_WorkPath;
		float						m_Thresh;
		bool						mbDebug;

		std::string					mSceneName;

		// Time
		int							m_Frame;		
		float						m_DT;
		float						m_Time;	

		// CUDA Kernels
		CUmodule					m_Module;
		CUfunction					m_Func[ FUNC_MAX ];

		// Simulation Parameters
		float						m_Param [ MAX_PARAM ];			// see defines above
		Vector3DF					m_Vec [ MAX_PARAM ];
		bool						m_Toggle [ MAX_PARAM ];		

		// SPH Kernel functions
		float						m_R2, m_Poly6Kern, m_LapKern, m_SpikyKern;		

		// Particle Buffers
		int						mNumPoints;
		int						mMaxPoints;
		int						mGoodPoints;
		FBufs					m_Fluid;				// Fluid buffers
		FBufs					m_FluidTemp;			// Fluid buffers (temporary)
		FParams					m_FParams;				// Fluid parameters

		CUdeviceptr				cuFBuf;					// GPU pointer containers
		CUdeviceptr				cuFTemp;
		CUdeviceptr				cuFParams;

		// Acceleration Grid		
		int						m_GridTotal;			// total # cells
		Vector3DI				m_GridRes;				// resolution in each axis
		Vector3DF				m_GridMin;				// volume of grid (may not match domain volume exactly)
		Vector3DF				m_GridMax;		
		Vector3DF				m_GridSize;				// physical size in each axis
		Vector3DF				m_GridDelta;
		int						m_GridSrch;
		int						m_GridAdjCnt;
		int						m_GridAdj[216];

		// Acceleration Neighbor Table
		int						m_NeighborNum;
		int						m_NeighborMax;
		int*					m_NeighborTable;
		float*					m_NeighborDist;

		int						mVBO[3];

		// Record/Playback
		bool					mbRecord;		
		bool					mbRecordBricks;
		int						mSpherePnts;
		int						mTex[1];		

		// Selected particle
		int						mSelected;


		// Saved results (for algorithm validation)
		uint*					mSaveNdx;
		uint*					mSaveCnt;
		uint*					mSaveNeighbors;		
	};	

	

#endif

//--------------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2017, NVIDIA Corporation
//
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
// 
// Version 1.0: Rama Hoetzlein, 5/1/2017
//----------------------------------------------------------------------------------

#ifndef DEF_OPTIX_SCENE
	#define DEF_OPTIX_SCENE

	#include <optix.h>
	#include <optixu/optixu.h>
	#include <optixu/optixpp_namespace.h>
	using namespace optix;

	#include "gvdb.h"

	struct MaterialParams {
		char		name[64];
		float		light_width;		// light scatter
		float		shadow_width;		// shadow scatter
		Vector3DF	env_color;			// 0.5,0.5,0.5
		Vector3DF	diff_color;			// .6,.7,.7
		Vector3DF	spec_color;			// 3,3,3
		float		spec_power;			// 400		
		float		refl_width;			// reflect scatter
		Vector3DF	refl_color;			// 1,1,1		
		float		refr_width;			// refract scatter
		float		refr_ior;			// 1.2
		Vector3DF	refr_color;			// .35, .4, .4
		float		refr_amount;		// 10
		float		refr_offset;		// 15
	};

	class OptixScene {
	public:
		OptixScene ();

		enum OptixParam {
			uView = 0,
			uProj = 1,
			uModel = 2,
			uLightPos = 3,
			uLightTarget = 4,
			uClrAmb = 5,
			uClrDiff = 6
		};

		void	InitializeOptix ( int w, int h );
		Buffer	CreateOutputOptix ( RTformat format, unsigned int width, unsigned int height );
		Program CreateProgramOptix ( std::string name, std::string prog_func );
		void	ClearGraph ();
		int		AddMaterial ( std::string fname, std::string cast_prog, std::string shadow_prog );
		void	AddPolygons ( Model* model, int mat_id, Matrix4F& xform );
		void	AddVolume ( int atlas_glid, Vector3DF vmin, Vector3DF vmax, int mat_id, Matrix4F& xform, bool deep, bool lset );
		void	ValidateGraph ();

		void	SetCamera ( Vector3DF pos, Vector3DF U, Vector3DF V, Vector3DF W, float asp );
		void	SetSample ( int frame, int sample );
		void	SetLight ( Vector3DF pos );		
		void    SetShading ( int stype );
		void	SetVolumeParams ( Vector3DF steps, Vector3DF extinct, Vector3DF cutoff );
		void	SetMaterialParams ( int n );
		void	SetTransferFunc ( Vector4DF* src );
		void	AssignGVDB ( int sz, char* dat );
		void	Launch ();

		void	ReadOutputTex ( int out_tex );
		
		void*	getContext()	{ return &m_OptixContext; }
		MaterialParams* getMaterialParams( int n )	{ return &m_OptixMatParams[n]; }
	
	/*void ClearOptix();
		void AddShaderOptix ( Scene* scene, char* vertname, char* fragname );
		int	 AddMaterialOptix ( Scene* scene, std::string cast_prog, std::string shadow_prog, std::string name );	
		void AddModelOptix ( Model* model, int oid, Matrix4F& xform );
		void AddVolumeOptix ( Vector3DF vmin, Vector3DF vmax, int mat_id, Matrix4F& xform, bool deep, bool lset );
		void AssignSamplerOptix ( int glid );
		void AssignGVDBOptix ( int vdb_sz, void* vdb_dat, int shade_type );
		void ValidateOptix ();
		float RenderOptix ( Scene* scene, int rend, int frame, int sample, int spp, int msz, void* mdat );
		void SetupOptixGL ( Scene* scene, int prog );

		int renderGetOptixGLID ();
		int renderGetOptixCUDADevice ();*/

	private:

		optix::Context	m_OptixContext;
		Group			m_OptixMainGroup;
		TextureSampler	m_OptixVolSampler;
		Program			m_OptixVolIntersectSurfProg;
		Program			m_OptixVolIntersectLevelSetProg;
		Program			m_OptixVolIntersectDeepProg;
		Program			m_OptixVolBBoxProg;
		Program			m_OptixMeshIntersectProg;
		Program			m_OptixMeshBBoxProg;
		Buffer			m_OptixBuffer;
		Buffer			m_OptixTransferFunc;
		std::vector< Transform >		m_OptixVolumes;
		std::vector< Transform >		m_OptixModels;
		std::vector< optix::Material >	m_OptixMats;
		std::vector< MaterialParams >	m_OptixMatParams;
		int				m_OptixTex;
		int				m_OptixProgram;

		int				m_OptixParam[10];
	};

#endif
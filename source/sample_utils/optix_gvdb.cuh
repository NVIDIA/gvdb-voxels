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

#ifndef OPTIX_GVDB_H
#define OPTIX_GVDB_H

	#include "optix_extra_math.cuh"			
	#include "texture_fetch_functions.h"

	#define ANY_RAY			0
	#define	SHADOW_RAY		1
	#define VOLUME_RAY		2
	#define MESH_RAY		3
	#define REFRACT_RAY		4

	//------ GVDB Headers
	#define OPTIX_PATHWAY
	#include "cuda_gvdb_scene.cuh"				// GVDB Scene
	#include "cuda_gvdb_nodes.cuh"				// GVDB Node structure
	#include "cuda_gvdb_geom.cuh"				// GVDB Geom helpers
	#include "cuda_gvdb_dda.cuh"				// GVDB DDA 
	#include "cuda_gvdb_raycast.cuh"			// GVDB Raycasting
	//------

	//----- GVDB Objects for OptiX
	rtDeclareVariable(VDBInfo, gvdbObj, , );			// GVDB object
	rtDeclareVariable(int, gvdbChan, , );				// GVDB render channel	

	// ScnInfo is defined in cuda_gvdb_raycast.cuh	   	

#endif
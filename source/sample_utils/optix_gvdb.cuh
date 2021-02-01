//-----------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2017 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
// 
// Version 1.0: Rama Hoetzlein, 5/1/2017
//-----------------------------------------------------------------------------

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
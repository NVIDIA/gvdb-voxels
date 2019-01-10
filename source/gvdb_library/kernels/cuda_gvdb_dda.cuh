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
// File: cuda_gvdb_dda.cuh
//
// DDA header
// - Transfer function 
// - DDA stepping macros
//-----------------------------------------------

// Tranfser function
inline __device__ float4 transfer ( VDBInfo* gvdb, float v )
{
	return TRANSFER_FUNC [ int(min( 1.0, max( 0.0,  (v - SCN_THRESH) / (SCN_VMAX - SCN_VMIN) ) ) * 16300.0f) ];
}

// Prepare DDA - This macro sets up DDA variables to start stepping sequence
#define PREPARE_DDA {											\
	p = ( pos + t.x*dir - vmin) / gvdb->vdel[lev];				\
	tDel = fabs3 ( gvdb->vdel[lev] / dir );						\
	tSide	= (( floor3(p) - p + 0.5)*pStep+0.5) * tDel + t.x;	\
	p = floor3(p);												\
}
#define PREPARE_DDA_LEAF {										\
	p = ( pos + t.x*dir - vmin) / gvdb->vdel[0];				\
	tDel = fabs3 ( gvdb->vdel[0] / dir );						\
	tSide	= (( floor3(p) - p + 0.5)*pStep+0.5) * tDel;		\
	p = floor3(p);												\
}

// Next DDA - This macro computes the next time step from DDA
#define NEXT_DDA {														\
	mask.x = float ( (tSide.x < tSide.y) & (tSide.x <= tSide.z) );		\
	mask.y = float ( (tSide.y < tSide.z) & (tSide.y <= tSide.x) );		\
	mask.z = float ( (tSide.z < tSide.x) & (tSide.z <= tSide.y) );		\
	t.y = mask.x ? tSide.x : (mask.y ? tSide.y : tSide.z);				\
}					
// Step DDA - This macro steps the DDA to next point, updating t
#define STEP_DDA {						\
	t.x = t.y;							\
	tSide	+= mask * tDel;				\
	p		+= mask * pStep;			\
}

// Advance DDA - This macro advances the DDA, *without* updating t
#define ADVANCE_DDA {					\
	tSide	+= mask * tDel;				\
	p		+= mask * pStep;			\
}

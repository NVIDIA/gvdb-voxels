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
// Version 1.1.1: Neil Bickford, 7/7/2020
//----------------------------------------------------------------------------------
// File: cuda_gvdb_dda.cuh
//
// DDA header
// - Transfer function 
// - DDA stepping struct
//-----------------------------------------------

// Transfer function
// Evaluates the transfer function.
// Values of `v` from SCN_THRESH to SCN_THRESH + (SCN_VMAX - SCN_VMIN) will be mapped to entries from 0 to 16,300
// of the transfer function.
inline __device__ float4 transfer(VDBInfo* gvdb, float v)
{
	return TRANSFER_FUNC[int(min(1.0, max(0.0, (v - SCN_THRESH) / (SCN_VMAX - SCN_VMIN))) * 16300.0f)];
}

// Represents the state of a point on a ray marching through a hierarchical DDA.
// In index-space, the ray is pos + t.x*dir.
// Essentially, we're always marching within a parent node, over the children of that node.
// We can always prepare this state given a point and a node.
// At each step, we advance to the next child node in the X, Y, or Z directions.
//
// This mainly gets used in cuda_gvdb_raycast.cuh.
// Typically, a function will call SetFromRay once per ray, then call Prepare initially
// and whenever the traversal level changes (this essentially performs some precomputation).
// It'll then usually alternate between calling Next (determines next cell intersection)
// and Step (moves to that next intersection).
//
// Reference: http://ramakarl.com/pdfs/2016_Hoetzlein_GVDB.pdf
struct HDDAState {
	// Constant per ray:
	float3 pos; // The origin of the ray in index-space.
	float3 dir; // The direction of the ray in index-space.
	int3 pStep; // Signs of dir
	// Constant per node:
	float3 tDel; // (voxels per child node) / fabs3(dir)
	// Varying:
	float3 t; // (current position, next time step, hit status). The point position in index-space is pos + t.x * dir.
	int3 p; // The index of the child node relative to the parent node.
	float3 tSide; // Value of t.x that intersects the next plane in the x, y, and z direction.
	int3 mask; // Which coordinates to move along next, each 1 or 0.

	// Initializes the HDDA given an index-space point (startPos), direction (startDir), and
	// a starting value of t (startT).
	__device__ void SetFromRay(float3 startPos, float3 startDir, float3 startT) {
		pos = startPos;
		dir = startDir;
		pStep = isign3(dir);
		t = startT;
	}

	// Set up variables for a node to prepare for stepping.
	__device__ void Prepare(float3 vmin, float3 vdel) {
		tDel = fabs3(vdel / dir);
		float3 pFlt = (pos + t.x * dir - vmin) / vdel;
		tSide = ((floor3(pFlt) - pFlt + 0.5f) * make_float3(pStep) + 0.5) * tDel + t.x;
		p = make_int3(floor3(pFlt));
	}

	// Prepare a DDA for stepping through a brick, which omits the addition by t.x from Prepare.
	// For bricks, the size of a child node in voxels (vdel) is always 1.
	__device__ void PrepareLeaf(float3 vmin){
		tDel = fabs3(1.0f / dir);
		float3 pFlt = pos + t.x * dir - vmin;
		tSide = ((floor3(pFlt) - pFlt + 0.5f) * make_float3(pStep) + 0.5) * tDel;
		p = make_int3(floor3(pFlt));
	}

	// Compute the next time step and direction from DDA, but don't step yet.
	__device__ void Next() {
		mask.x = int((tSide.x < tSide.y) & (tSide.x <= tSide.z));
		mask.y = int((tSide.y < tSide.z) & (tSide.y <= tSide.x));
		mask.z = int((tSide.z < tSide.x) & (tSide.z <= tSide.y));
		t.y = mask.x ? tSide.x : (mask.y ? tSide.y : tSide.z);
	}

	// Step the DDA to the next point.
	__device__ void Step() {
		t.x = t.y;
		tSide += make_float3(mask) * tDel;
		p += mask * pStep;
	}
};
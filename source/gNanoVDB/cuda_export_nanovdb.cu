//--------------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2020, NVIDIA Corporation. 
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
// Version 1.1.1: Neil Bickford, 8/12/2020
//----------------------------------------------------------------------------------

#include <cuda.h>
#include <float.h>
#include <stdint.h>

// NanoVDB
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/Ray.h>
#include <nanovdb/util/SampleFromVoxels.h>

#define CUDA_PATHWAY
#include <cuda_math.cuh>
#include <cuda_gvdb_scene.cuh>
#include <cuda_gvdb_nodes.cuh>

// Shorter type name for a NanoVDB grid with a given value type, log base 2 (branching) dimension
// for level-2 InternalNodes, log base 2 dimension for level-1 InternalNodes, and log base 2
// dimension for leaves. (These are the same values as returned by gvdb.getLD(level).)
template<class ValueT, int Node2LogDim, int Node1LogDim, int LeafLogDim>
using NanoGridCustom = nanovdb::Grid<nanovdb::Tree<nanovdb::RootNode<
	nanovdb::InternalNode<nanovdb::InternalNode<
	nanovdb::LeafNode<ValueT, nanovdb::Coord, nanovdb::Mask, LeafLogDim>,
	Node1LogDim>, Node2LogDim>>>>;

// Overloaded function to get the maximum value of a ValueT, since nanovdb::Maximum
// doesn't handle Vec3f at the moment
template<class T> __device__ T ExportToNanoVDB_MaximumValue() {
	return FLT_MAX;
}
template<> __device__ int ExportToNanoVDB_MaximumValue<int>() {
	return INT_MAX;
}
template<> __device__ nanovdb::Vec3f ExportToNanoVDB_MaximumValue<nanovdb::Vec3f>() {
	return { FLT_MAX, FLT_MAX, FLT_MAX };
}

// Implementation of min and max for nanovdb::Vec3f types.
__device__ nanovdb::Vec3f min(const nanovdb::Vec3f& a, const nanovdb::Vec3f& b) {
	return {
		min(a[0], b[0]),
		min(a[1], b[1]),
		min(a[2], b[2])
	};
}

__device__ nanovdb::Vec3f max(const nanovdb::Vec3f& a, const nanovdb::Vec3f& b) {
	return {
		max(a[0], b[0]),
		max(a[1], b[1]),
		max(a[2], b[2])
	};
}

// Reads a value from the atlas.
template<class ValueT>
__device__ ValueT ReadValue(const cudaSurfaceObject_t& atlas, const int x, const int y, const int z) {
	return surf3Dread<ValueT>(atlas, x * sizeof(ValueT), y, z);
}
// Partial specialization for Vec3f, where we have to switch to using float4s:
template<>
__device__ nanovdb::Vec3f ReadValue<nanovdb::Vec3f>(const cudaSurfaceObject_t& atlas, const int x, const int y, const int z) {
	float4 value = surf3Dread<float4>(atlas, x * sizeof(float3), y, z); // Note float4/float3 change!
	return { value.x, value.y, value.z };
}

// A ProcessLeafFunc is a function that takes a VDBInfo*, void*, cudaSurfaceObject_t, and int, and returns nothing.
using ProcessLeafFunc = void(*)(VDBInfo*, void*, cudaSurfaceObject_t, int);

template<class ValueT, int LOG2DIM>
__device__ void ProcessLeaf(VDBInfo* gvdb, void* nanoVDBLeafNodes, cudaSurfaceObject_t atlas, int numLeaves)
{
	const int leafIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (leafIdx >= numLeaves) return;

	VDBNode* gvdbNode = getNode(gvdb, 0, leafIdx);

	// Sometimes, the GVDB node list can contain nodes that were previously part of trees.
	// These will be ignored, although there will still be space for them in the NanoVDB volume.
	// (This isn't a requirement of exporting - we could prune these nodes as well.)
	if (gvdbNode->mChildList != ID_UNDEFL) return;

	// The NanoVDB node and data
	using LeafT = nanovdb::LeafNode<ValueT, nanovdb::Coord, nanovdb::Mask, LOG2DIM>;
	const int brickres = 1 << LOG2DIM;

	LeafT* leafNodes = reinterpret_cast<LeafT*>(nanoVDBLeafNodes);
	LeafT& node = leafNodes[leafIdx];
	LeafT::DataType& nodeData = *reinterpret_cast<LeafT::DataType*>(&node);

	// All values in a brick are active in GVDB
	nodeData.mValueMask.set(true);

	// Minimum and maximum value of all elements
	ValueT minValue = ExportToNanoVDB_MaximumValue<ValueT>();
	ValueT maxValue = -minValue;

	const int3 brickValue = gvdbNode->mValue;

	// Copy the brick in the order ((x * T) + y) * T + z.
	for (int x = 0; x < brickres; x++) {
		for (int y = 0; y < brickres; y++) {
			for (int z = 0; z < brickres; z++) {
				const ValueT value = ReadValue<ValueT>(atlas, x + brickValue.x, y + brickValue.y, z + brickValue.z);
				nodeData.mValues[(x * brickres + y) * brickres + z] = value;
				minValue = min(minValue, value);
				maxValue = max(maxValue, value);
			}
		}
	}

	nodeData.mMinimum = minValue;
	nodeData.mMaximum = maxValue;
	// Note that mAverage and mStdDevi currently aren't set.

	// Since all voxels are active, the bounding box of active values of this node is the
	// bounding box of the node itself:
	nodeData.mBBoxMin = { gvdbNode->mPos.x, gvdbNode->mPos.y, gvdbNode->mPos.z };
	nodeData.mBBoxDif[0] = brickres;
	nodeData.mBBoxDif[1] = brickres;
	nodeData.mBBoxDif[2] = brickres;
}

// Autogenerated list of ProcessLeaf instantiation function pointers. You can generate this list
// using the following Python code:
/*
print('static const __device__ ProcessLeafFunc processLeafFuncs[3][6] = {')
for type in ['float', 'nanovdb::Vec3f', 'int']:
	print('{', end='')
	for ld in range(2, 8):
		print(f'ProcessLeaf<{type}, {ld}>', end='')
		if ld != 7:
			print(', ', end='')
	if(type == 'int'):
		print('}')
	else:
		print('},')
print('};')
*/
static const __device__ ProcessLeafFunc processLeafFuncs[3][6] = {
{ProcessLeaf<float, 2>, ProcessLeaf<float, 3>, ProcessLeaf<float, 4>, ProcessLeaf<float, 5>, ProcessLeaf<float, 6>, ProcessLeaf<float, 7>},
{ProcessLeaf<nanovdb::Vec3f, 2>, ProcessLeaf<nanovdb::Vec3f, 3>, ProcessLeaf<nanovdb::Vec3f, 4>, ProcessLeaf<nanovdb::Vec3f, 5>, ProcessLeaf<nanovdb::Vec3f, 6>, ProcessLeaf<nanovdb::Vec3f, 7>},
{ProcessLeaf<int, 2>, ProcessLeaf<int, 3>, ProcessLeaf<int, 4>, ProcessLeaf<int, 5>, ProcessLeaf<int, 6>, ProcessLeaf<int, 7>}
};

extern "C" __global__ void gvdbToNanoVDBProcessLeaves(
	VDBInfo * gvdb, void* nanoVDBLeafNodes, int typeTableIndex, cudaSurfaceObject_t atlas, int numLeaves)
{
	// Redirect to the appropriate function instantiation
	processLeafFuncs[typeTableIndex][gvdb->dim[0] - 2](gvdb, nanoVDBLeafNodes, atlas, numLeaves);
}

//-------------------------------------------------------------------------------------------------
// The code to process internal nodes from PTX involves probably the most complex function
// indirection in this file. If we wanted to template this over a single function, we would need to
// know four things:
// - the level of the node (if level 1, then its child is a LeafNode, while if level 2, then its
// child is another InternalNode)
// - the value type
// - the log2dim of this node
// - the log2dim of the child node
//
// If we expressed this all in a 4D array of templates, we would have 2 * 3 * 6 * 6 = 216 cases.
// Which could be reasonable! Instead, we reduce this to two smaller tables of cases:
// gvdbToNanoVDBProcessInternalNodes starts out by switching over a 2D table of 3 * 6 = 18 cases
//   (value type and log2dim of this node).
// When getting the extents of the child node, we switch over a 3D table of 2 * 3 * 6 = 36 cases.
// These functions return their extent data in a common format.

// A union type for the possible values of a ValueT.
union ValueUnion {
	float f;
	nanovdb::Vec3f f3;
	int i;
};

template<class T>
__device__ T* getValueUnion(ValueUnion& value) { return reinterpret_cast<T*>(&value.f3); }

// Common structure for the extents of a node.
struct NodeRangeData {
	ValueUnion valueMin;
	ValueUnion valueMax;
	nanovdb::CoordBBox aabb;
};

// Gets information about the range of the given node in a C-like format.
template<class NodeT>
__device__ NodeRangeData GetNodeRange(uint8_t* nodeStart, int nodeIdx) {
	using ValueT = NodeT::ValueType;
	NodeT* node = reinterpret_cast<NodeT*>(nodeStart) + nodeIdx;

	NodeRangeData result;

	*getValueUnion<ValueT>(result.valueMin) = node->valueMin();
	*getValueUnion<ValueT>(result.valueMax) = node->valueMax();
	result.aabb = node->bbox();

	return result;
}

// A NodeRangeFunc is a function that takes a uint8_t* and an int and returns a NodeRangeData.
using NodeRangeFunc = NodeRangeData(*)(uint8_t*, int);

// Short versions of leaf node and internal node
template<class ValueT, int LOG2DIM>
using LeafNodeSmpl = nanovdb::LeafNode<ValueT, nanovdb::Coord, nanovdb::Mask, LOG2DIM>;
// Using 3 for the leaf node's LOG2DIM suffices here
template<class ValueT, int LOG2DIM>
using INodeSmpl = nanovdb::InternalNode<nanovdb::LeafNode<ValueT>, LOG2DIM>;

// 3D table of [child node is leaf or level-1][value type][child node log2dim - 2].
// This was autogenerated using the following Python code:
/*
print('static const __device__ NodeRangeFunc rangeFunctions[2][3][6] = {{')
for type in ['float', 'nanovdb::Vec3f', 'int']:
	print('{', end='')
	for ld in range(2, 8):
		print(f'GetNodeRange<LeafNodeSmpl<{type}, {ld}>>', end='')
		if ld != 7:
			print(', ', end='')
	if(type == 'int'):
		print('}')
	else:
		print('},')
print('},{',)
for type in ['float', 'nanovdb::Vec3f', 'int']:
	print('{', end='')
	for ld in range(2, 8):
	# Using 3 for the leaf node's LOG2DIM suffices here
		print(f'GetNodeRange<INodeSmpl<{type}, {ld}>>', end='')
		if ld != 7:
			print(', ', end='')
	if(type == 'int'):
		print('}')
	else:
		print('},')
print('}};')
*/
static const __device__ NodeRangeFunc rangeFunctions[2][3][6] = { {
{GetNodeRange<LeafNodeSmpl<float, 2>>, GetNodeRange<LeafNodeSmpl<float, 3>>, GetNodeRange<LeafNodeSmpl<float, 4>>, GetNodeRange<LeafNodeSmpl<float, 5>>, GetNodeRange<LeafNodeSmpl<float, 6>>, GetNodeRange<LeafNodeSmpl<float, 7>>},
{GetNodeRange<LeafNodeSmpl<nanovdb::Vec3f, 2>>, GetNodeRange<LeafNodeSmpl<nanovdb::Vec3f, 3>>, GetNodeRange<LeafNodeSmpl<nanovdb::Vec3f, 4>>, GetNodeRange<LeafNodeSmpl<nanovdb::Vec3f, 5>>, GetNodeRange<LeafNodeSmpl<nanovdb::Vec3f, 6>>, GetNodeRange<LeafNodeSmpl<nanovdb::Vec3f, 7>>},
{GetNodeRange<LeafNodeSmpl<int, 2>>, GetNodeRange<LeafNodeSmpl<int, 3>>, GetNodeRange<LeafNodeSmpl<int, 4>>, GetNodeRange<LeafNodeSmpl<int, 5>>, GetNodeRange<LeafNodeSmpl<int, 6>>, GetNodeRange<LeafNodeSmpl<int, 7>>}
},{
{GetNodeRange<INodeSmpl<float, 2>>, GetNodeRange<INodeSmpl<float, 3>>, GetNodeRange<INodeSmpl<float, 4>>, GetNodeRange<INodeSmpl<float, 5>>, GetNodeRange<INodeSmpl<float, 6>>, GetNodeRange<INodeSmpl<float, 7>>},
{GetNodeRange<INodeSmpl<nanovdb::Vec3f, 2>>, GetNodeRange<INodeSmpl<nanovdb::Vec3f, 3>>, GetNodeRange<INodeSmpl<nanovdb::Vec3f, 4>>, GetNodeRange<INodeSmpl<nanovdb::Vec3f, 5>>, GetNodeRange<INodeSmpl<nanovdb::Vec3f, 6>>, GetNodeRange<INodeSmpl<nanovdb::Vec3f, 7>>},
{GetNodeRange<INodeSmpl<int, 2>>, GetNodeRange<INodeSmpl<int, 3>>, GetNodeRange<INodeSmpl<int, 4>>, GetNodeRange<INodeSmpl<int, 5>>, GetNodeRange<INodeSmpl<int, 6>>, GetNodeRange<INodeSmpl<int, 7>>}
} };

using ProcessInternalNodeFunc = void(*)(VDBInfo*, VDBNode*, uint8_t*, uint8_t*, int, int, int, int, ValueUnion);

// Here, LOG2DIM is the log2dim of the internal node (not the child node).
// childLevel is this node's level minus 1.
template<class ValueT, int LOG2DIM> __device__ void ProcessInternalNode(
	VDBInfo* gvdb, VDBNode* gvdbNode,
	uint8_t* nanoVDBNodes, uint8_t* nanoVDBChildNodes,
	int numNodes, int nodeIdx, int level, int childLog2Dim,
	ValueUnion backgroundUnion)
{
	using NodeT = INodeSmpl<ValueT, LOG2DIM>;
	using DataT = NodeT::DataType;
	const uint32_t res = 1 << LOG2DIM;
	const uint32_t numChildren = res * res * res;
	DataT* nodeData = reinterpret_cast<DataT*>(nanoVDBNodes) + nodeIdx;

	// We never have any values to fill in:
	nodeData->mValueMask.setOff();
	// We'll turn on the child mask as needed:
	nodeData->mChildMask.setOff();

	// There are different ways that we could combine mOffset and childID, but for now we
	// simply make mOffset give the number of InternalNodes until the start of the next node
	// section:
	nodeData->mOffset = numNodes - nodeIdx;

	// Initialize counters for the min and max value and bounding box
	ValueT valueMin = ExportToNanoVDB_MaximumValue<ValueT>();
	ValueT valueMax = -valueMin;
	nanovdb::Coord aabbMin = { INT_MAX, INT_MAX, INT_MAX };
	nanovdb::Coord aabbMax = { -INT_MAX, -INT_MAX, -INT_MAX };

	// Get a pointer to the function instantiation to use for getting ndoe extents.
	NodeRangeFunc rangeFunc;
	const int childLevel = level - 1;
	if (std::is_same<ValueT, float>::value) {
		rangeFunc = rangeFunctions[childLevel][0][childLog2Dim - 2];
	}
	else if (std::is_same<ValueT, nanovdb::Vec3f>::value) {
		rangeFunc = rangeFunctions[childLevel][1][childLog2Dim - 2];
	}
	else {
		rangeFunc = rangeFunctions[childLevel][2][childLog2Dim - 2];
	}

	// The child list contains either ID_UNDEF64 or a 64-bit pool reference (grp, lev, index)
	// for each child. This translates well into NanoVDB's mask + indices!
	// (and also means that yes, you could implement a DAG with GVDB or NanoVDB)

	// At the moment, NanoVDB's internal nodes store their children in (x*T+y)*T+z order, while
	// GVDB stores its children in (z*T+y)*T+x order. (See the implementations of CoordToOffset
	// inside NanoVDB.)
	const uint32_t resMask = res - 1;
	const uint32_t resSquared = res * res;
	const uint32_t middleMask = res * resMask;
	for (uint32_t gvdbChildIdx = 0; gvdbChildIdx < numChildren; gvdbChildIdx++) {
		// Swap the order of the x, y, and z components, by interpreting gvdbChildIdx
		// as a base-res number.
		const uint32_t nanoVDBChildIdx =
			(gvdbChildIdx / resSquared)
			+ (gvdbChildIdx & middleMask)
			+ (gvdbChildIdx & resMask) * resSquared;

		const int childID = getChild(gvdb, gvdbNode, gvdbChildIdx);

		// Skip children that don't exist, filling them with the background value.
		// These aren't active, but still have values.
		if (childID == (int)ID_UNDEF64) {
			nodeData->mTable[nanoVDBChildIdx].value = *getValueUnion<ValueT>(backgroundUnion);
			continue;
		}

		nodeData->mChildMask.setOn(nanoVDBChildIdx);
		nodeData->mTable[nanoVDBChildIdx].childID = static_cast<uint32_t>(childID);

		// Now, since we've already filled in the lower levels of the tree, update this node's
		// min and max values and bounding box:
		NodeRangeData rangeData = rangeFunc(nanoVDBChildNodes, childID);
		valueMin = min(valueMin, *getValueUnion<ValueT>(rangeData.valueMin));
		valueMax = max(valueMax, *getValueUnion<ValueT>(rangeData.valueMax));

		const nanovdb::BBox<nanovdb::Coord> childAABB = rangeData.aabb;
		for (int c = 0; c < 3; c++) {
			aabbMin[c] = min(aabbMin[c], childAABB.min()[c]);
			aabbMax[c] = max(aabbMax[c], childAABB.max()[c]);
		}
	}

	nodeData->mMinimum = valueMin;
	nodeData->mMaximum = valueMax;
	nodeData->mBBox.min() = aabbMin;
	nodeData->mBBox.max() = aabbMax;
}

// Table of instantiations of ProcessInternalNode, [value type][node log2dim - 2]. This was
// autogenerated using the following Python script:
/*
print('static const __device__ ProcessInternalNodeFunc processInternalNodeFuncs[3][6] = {')
for type in ['float', 'nanovdb::Vec3f', 'int']:
	print('{', end='')
	for ld in range(2, 8):
		print(f'ProcessInternalNode<{type}, {ld}>', end='')
		if ld != 7:
			print(', ', end='')
	if(type == 'int'):
		print('}')
	else:
		print('},')
print('};')
*/
static const __device__ ProcessInternalNodeFunc processInternalNodeFuncs[3][6] = {
{ProcessInternalNode<float, 2>, ProcessInternalNode<float, 3>, ProcessInternalNode<float, 4>, ProcessInternalNode<float, 5>, ProcessInternalNode<float, 6>, ProcessInternalNode<float, 7>},
{ProcessInternalNode<nanovdb::Vec3f, 2>, ProcessInternalNode<nanovdb::Vec3f, 3>, ProcessInternalNode<nanovdb::Vec3f, 4>, ProcessInternalNode<nanovdb::Vec3f, 5>, ProcessInternalNode<nanovdb::Vec3f, 6>, ProcessInternalNode<nanovdb::Vec3f, 7>},
{ProcessInternalNode<int, 2>, ProcessInternalNode<int, 3>, ProcessInternalNode<int, 4>, ProcessInternalNode<int, 5>, ProcessInternalNode<int, 6>, ProcessInternalNode<int, 7>}
};

extern "C" __global__ void gvdbToNanoVDBProcessInternalNodes(
	VDBInfo * gvdb,
	uint8_t* nanoVDBNodes, uint8_t* nanoVDBChildNodes,
	int numNodes, int level, int nodeLog2Dim, int childLog2Dim,
	ValueUnion backgroundUnion, int typeTableIndex)
{
	const int nodeIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (nodeIdx >= numNodes) return;

	VDBNode* gvdbNode = getNode(gvdb, level, nodeIdx);

	// Get the child list of the node, skipping nodes with no children. (Note: These will still
	// take up space, though! We could prune these if we wanted to.)
	if (gvdbNode->mChildList == ID_UNDEFL) {
		// All fields will be 0
		return;
	}

	// Redirect to the correct instantiation of ProcessInternalNode.
	processInternalNodeFuncs[typeTableIndex][nodeLog2Dim - 2](
		gvdb, gvdbNode, // Volume and GVDB node
		nanoVDBNodes, nanoVDBChildNodes, // Pointers to NanoVDB data
		numNodes, // Number of level-`level` nodes
		nodeIdx, // Index of the current node
		level, // The node level
		childLog2Dim, // The child log2dim
		backgroundUnion // The background
		);
}

//-------------------------------------------------------------------------------------------------

// Here's an implementation of a simple transfer function accessible from NanoVDB. You could also
// look at GVDB's memory used for its transfer function directly.
__device__ float4 transferFunction(float value) {
	// Emulating the transfer function in cuda_gvdb_dda.cuh:
	const float scnThresh = .1f;
	const float scnVmax = .1f;
	const float scnVmin = 0.0f;
	const float t = max(0.0f, min(1.0f, (value - scnThresh) / (scnVmax - scnVmin)));

	// Applying a linear transfer function:
	const float4 keys[5] = {
		make_float4(0.0f, 0.0f, 0.0f, 0.0f),
		make_float4(1.5f, 1.5f, 0.0f, 0.1f),
		make_float4(1.5f, 0.0f, 0.0f, 0.3f),
		make_float4(0.3f, 0.3f, 0.3f, 0.1f),
		make_float4(0.0f, 0.0f, 0.0f, 0.0f)
	};
	const float t4 = t * 4;
	const int baseKey = int(min(3.0f, floorf(t4)));
	const float ft4 = t4 - float(baseKey);
	return keys[baseKey] * (1.0f - ft4) + keys[baseKey] * ft4;
}

__device__ unsigned char floatToUchar(float v) {
	return static_cast<unsigned char>(255.0f * fminf(1.0f, v));
}

extern "C" __global__ void gvdbExportNanoVDBRender(void* vPtrGrid,
	nanovdb::Vec3f eye, float4 camTopLeftWS, float4 camRightWS, float4 camDownWS,
	unsigned char* image, uint32_t width, uint32_t height)
{
	using GridT = NanoGridCustom<float, 5, 4, 3>;
	using AccT = GridT::AccessorType;
	GridT* grid = reinterpret_cast<GridT*>(vPtrGrid);

	const uint32_t xi = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t yi = blockIdx.y * blockDim.y + threadIdx.y;
	if (xi >= width || yi >= height) return;

	AccT acc = grid->getAccessor();

	// 0----> u
	// |    |
	// V ---*
	// v
	float u = static_cast<float>(xi) / static_cast<float>(width);
	float v = static_cast<float>(yi) / static_cast<float>(height);
	const nanovdb::Vec3f direction(
		camTopLeftWS.x + u * camRightWS.x + v * camDownWS.x,
		camTopLeftWS.y + u * camRightWS.y + v * camDownWS.y,
		camTopLeftWS.z + u * camRightWS.z + v * camDownWS.z
	);
	nanovdb::Ray<float> rayWS(eye, direction);
	nanovdb::SampleFromVoxels<AccT, 1, true> sampler(acc);
	// Convert to index-space
	nanovdb::Ray<float> iRay = rayWS.worldToIndexF(*grid);

	float3 color = make_float3(0.0f, 0.0f, 0.0f);
	float transmittance = 1.0f;
	// Compute transmittance if the ray intersects the bounding box
	if (iRay.clip(grid->tree().bbox())) {
		// Integrate color and transmittance.
		// This is done in a brute-force manner for this sample, but it's possible to do better
		// (e.g. using delta tracking)
		const float dt = 0.25f; // Roughly, 1/(number of samples per voxel).
		for (float t = iRay.t0(); t < iRay.t1(); t += dt) {
			float sigma = sampler(iRay(t));
			float4 value = transferFunction(sigma); // Color and opacity
			value.w = exp(-dt * value.w);
			color.x += value.x * transmittance * (1 - value.w);
			color.y += value.y * transmittance * (1 - value.w);
			color.z += value.z * transmittance * (1 - value.w);
			transmittance *= value.w;
		}
	}

	// Composite against the background
	const int checkerboard = 1 << 7;
	const float checkerboardValue = (((xi & checkerboard) == (yi & checkerboard)) ? 1.0f : 0.0f);

	float3 finalColor = color * (1.0f - transmittance) + checkerboardValue * transmittance;

	// Note that this function doesn't do any linear->sRGB conversion! In a production setting,
	// it's more physically accurate to do rendering in linear-space, then convert to sRGB.
	image[4 * (yi * width + xi) + 0] = floatToUchar(finalColor.x);
	image[4 * (yi * width + xi) + 1] = floatToUchar(finalColor.y);
	image[4 * (yi * width + xi) + 2] = floatToUchar(finalColor.z);
	image[4 * (yi * width + xi) + 3] = 255;
}
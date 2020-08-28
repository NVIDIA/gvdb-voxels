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

#ifndef DEF_GVDB_EXPORT_NANOVDB
#define DEF_GVDB_EXPORT_NANOVDB

// NanoVDB
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/Ray.h>

// GVDB
#define NOMINMAX
#include "gvdb.h"

#ifdef USE_BITMASKS
#error As of this writing, GVDB-to-NanoVDB conversion requires not using bitmasks.
#endif // #ifdef USE_BITMASKS

#ifdef NDEBUG
static const bool DEBUG_EXPORT_NANOVDB = false;
#else
static const bool DEBUG_EXPORT_NANOVDB = true;
#endif

#define GVDB_EXPORT_NANOVDB_NULL 0

namespace nvdb {
	// Shorter type name for a NanoVDB grid with a given value type, log base 2 (branching) dimension
	// for level-2 InternalNodes, log base 2 dimension for level-1 InternalNodes, and log base 2
	// dimension for leaves. (These are the same values as returned by gvdb.getLD(level).)
	template<class ValueT, int Node2LogDim, int Node1LogDim, int LeafLogDim>
	using NanoGridCustom = nanovdb::Grid<nanovdb::Tree<nanovdb::RootNode<
		nanovdb::InternalNode<nanovdb::InternalNode<
		nanovdb::LeafNode<ValueT, nanovdb::Coord, nanovdb::Mask, LeafLogDim>,
		Node1LogDim>, Node2LogDim>>>>;

	// Creates a copy of the GVDB volume as a NanoVDB volume on the GPU, and returns an owning
	// pointer to it. Returns GVDB_EXPORT_NANOVDB_NULL if exporting failed.
	//
	// The CUDA device pointer returned will point to an object of the NanoVDB type
	//
	// NanoGridCustom<ValueT, Node2LogDim, Node1LogDim, LeafLogDim>
	//
	// where ValueT is a NanoVDB type (described below), LeafLogDim is gvdb.getLD(0), Node1Log2Dim
	// is gvdb.getLD(1), and Node2Log2Dim is gvdb.getLD(2). When LeafLogDim == 3, Node1Log2Dim == 4,
	// and Node2Log2Dim == 5, this is equivalent to a NanoGrid.
	//
	// This function uses the following mapping between GVDB types and NanoVDB GridTypes (which may
	// change to add additional types in the future):
	//   GVDB type    NanoVDB type
	//   T_UCHAR      None (returns nullptr)
	//   T_UCHAR3     None (returns nullptr)
	//   T_UCHAR4     None (returns nullptr)
	//   T_FLOAT      GridType::Float
	//   T_FLOAT3     GridType::Vec3f
	//   T_FLOAT4     None (returns nullptr)
	//   T_INT        GridType::Int32
	//   T_INT3       None (returns nullptr)
	//   T_INT4       None (returns nullptr)
	//
	// Additionally, this function requires that the log-2 dim of each of the three lowest levels
	// is between 2 and 7 inclusive. (This makes it so that we can copy data into a NanoVDB
	// structure without rebuilding topology.)
	//
	// Note that the memory returned by this function will initially only be accessible to GVDB's
	// CUDA context (see gvdb.getContext()). To use this memory, you can switch to this context
	// before accessing it, or call cuCtxEnablePeerAccess(gvdb.getContext(), 0) to give your
	// context access to all of GVDB's memory allocations.
	//
	// We plan to move this function into GVDB's core API in the future.
	CUdeviceptr ExportToNanoVDB(VolumeGVDB& gvdb, uchar channel, void* backgroundPtr,
		const char gridName[nanovdb::GridData::MaxNameSize], nanovdb::GridClass gridClass, size_t* outTotalSize);

	// Renders a level set of a NanoGridCustom<float, 5, 4, 3> on the GPU. Switches to the given
	// CUDA context before rendering.
	void RenderNanoVDB(CUcontext context, CUdeviceptr nanoVDB, Camera3D* camera,
		uint width, uint height, uchar* outImage);

	//---------------------------------------------------------------------------------------------
	// Utility functions. These may be moved out of the header in the future.

	// Gets the number of nodes at a given level. If the number of nodes is greater than INT_MAX,
	// prints an error and returns false; otherwise, returns true and sets numNodes.
	inline bool ExportToNanoVDB_GetNumNodes(VolumeGVDB& gvdb, int level, int& numNodes) {
		uint64 numNodes64 = gvdb.getNumUsedNodes(level);
		if (numNodes64 > INT_MAX) {
			gprintf("Error in ExportToNanoVDB: Number of nodes at level %d (%llu) was greater than "
				"INT_MAX.", level, numNodes64);
			return false;
		}

		numNodes = static_cast<int>(numNodes64);
		return true;
	}

	template<class T>
	inline T ExportToNanoVDB_Min(T a, T b) {
		return (a < b ? a : b);
	}

	// Overload for nanovdb::Vec3f
	inline nanovdb::Vec3f ExportToNanoVDB_Min(nanovdb::Vec3f a, nanovdb::Vec3f b) {
		return {
			ExportToNanoVDB_Min(a[0], b[0]),
			ExportToNanoVDB_Min(a[1], b[1]),
			ExportToNanoVDB_Min(a[2], b[2])
		};
	}

	template<class T>
	inline T ExportToNanoVDB_Max(T a, T b) {
		return (a > b ? a : b);
	}

	// Overload for nanovdb::Vec3f
	inline nanovdb::Vec3f ExportToNanoVDB_Max(nanovdb::Vec3f a, nanovdb::Vec3f b) {
		return {
			ExportToNanoVDB_Max(a[0], b[0]),
			ExportToNanoVDB_Max(a[1], b[1]),
			ExportToNanoVDB_Max(a[2], b[2])
		};
	}

	//---------------------------------------------------------------------------------------------
	// For development purposes, we also include a reference CPU implementation of GVDB-to-NanoVDB
	// conversion and rendering. This can be useful when testing modifications to the GPU
	// implementation, but is generally significantly slower.

	/*
	template<class NodeT>
	inline bool ExportToNanoVDB_ProcessInternalNodes(VolumeGVDB& gvdb, NodeT* levelNodes, int numNodes, typename NodeT::ValueType background) {

		// Get the level we're working at from the NodeT type:
		const int level = NodeT::LEVEL;
		assert(level == 1 || level == 2);

		using ValueT = NodeT::ValueType;

		// Get number of children and range. Perform bounds checking.
		const uint64 numChildren64 = gvdb.getVoxCnt(level);
		if (numChildren64 > INT_MAX) {
			gprintf("Error in ExportToNanoVDB: Number of children for a level-%d internal node (%ull) "
				"was more than INT_MAX.", level, numChildren64);
			return false;
		}
		const uint32 numChildren = static_cast<uint32>(numChildren64);

		const uint32 res = gvdb.getRes(level); // Dimensions of this node (not in voxels)

		for (int nodeIdx = 0; nodeIdx < numNodes; nodeIdx++) {
			nvdb::Node* gvdbNode = gvdb.getNodeAtLevel(nodeIdx, level);

			// Get the child list of the node, skipping nodes with no children. (Note: These will still
			// take up space, though! We could prune these if we wanted to.
			if (gvdbNode->mChildList == ID_UNDEFL) {
				// All fields will be 0
				continue;
			}

			NodeT& node = levelNodes[nodeIdx];
			NodeT::DataType& nodeData = *reinterpret_cast<NodeT::DataType*>(&node);

			// We never have any values to fill in
			nodeData.mValueMask.setOff();
			// We'll turn on the child mask as needed:
			nodeData.mChildMask.setOff();

			// There are different ways that we could combine mOffset and childID, but for now we
			// simply make mOffset give the number of InternalNodes until the start of the next node
			// section:
			nodeData.mOffset = numNodes - nodeIdx;

			// Initialize counters for the min and max value and bounding box
			ValueT minValue = nanovdb::Maximum<ValueT>::value();
			ValueT maxValue = -minValue;
			nanovdb::Coord aabbMin = { INT_MAX, INT_MAX, INT_MAX };
			nanovdb::Coord aabbMax = { -INT_MAX, -INT_MAX, -INT_MAX };

			// The child list contains either ID_UNDEF64 or a 64-bit pool reference (grp, lev, index)
			// for each child. This translates well into NanoVDB's mask + indices!
			// (and also means that yes, you could implement a DAG with GVDB or NanoVDB)

			// At the moment, NanoVDB's internal nodes store their children in (x*T+y)*T+z order, while
			// GVDB stores its children in (z*T+y)*T+x order. (See the implementations of CoordToOffset
			// inside NanoVDB.)
			for (uint32 gvdbChildIdx = 0; gvdbChildIdx < numChildren; gvdbChildIdx++) {
				// Separate into x, y, and z components to get the NanoVDB child index
				const Vector3DI localChildPos = gvdb.getPosFromBit(level, gvdbChildIdx);
				const uint32 nanoVDBChildIdx = ((localChildPos.x * res) + localChildPos.y) * res
					+ localChildPos.z;

				const uint64 childReference = gvdb.getChildRefAtBit(gvdbNode, gvdbChildIdx);

				// Skip children that don't exist, filling them with the background value.
				// These aren't active, but still have values.
				if (childReference == ID_UNDEF64) {
					nodeData.mTable[nanoVDBChildIdx].value = background;
					continue;
				}

				// Consistency checks
				assert(ElemLev(childReference) == level - 1);
				assert(ElemGrp(childReference) == 0);

				// Get the child node from the child reference
				nodeData.mChildMask.setOn(nanoVDBChildIdx);
				nodeData.mTable[nanoVDBChildIdx].childID = static_cast<uint32_t>(ElemNdx(childReference));

				// Now, since we've already filled in the lower levels of the tree, update this node's
				// min and max values and bounding box:
				const NodeT::ChildNodeType* child = nodeData.child(nanoVDBChildIdx);
				minValue = ExportToNanoVDB_Min(minValue, child->valueMin());
				maxValue = ExportToNanoVDB_Max(maxValue, child->valueMax());

				const nanovdb::BBox<nanovdb::Coord> childAABB = child->bbox();
				for (int c = 0; c < 3; c++) {
					aabbMin[c] = std::min(aabbMin[c], childAABB.min()[c]);
					aabbMax[c] = std::max(aabbMax[c], childAABB.max()[c]);
				}
			}

			nodeData.mValueMin = minValue;
			nodeData.mValueMax = maxValue;
			nodeData.mBBox.min() = aabbMin;
			nodeData.mBBox.max() = aabbMax;
		}

		return true;
	}

	// Creates a copy of the GVDB volume as a NanoVDB volume, and returns an owning pointer to it.
	// Returns nullptr if exporting failed.
	// The main requirement for this function is that the template parameters must match the
	// lowest-level dimensions of the GVDB volume (which can be obtained using gvdb.getLD(level)), and
	// ValueT must match the type of the corresponding channel of the GVDB volume, and must be a
	// NanoVDB type (a struct, not the GridType enum).
	// This function uses the following mapping between GVDB types and NanoVDB GridTypes (which may
	// change to add additional types in the future):
	//   GVDB type    NanoVDB type
	//   T_UCHAR      None (returns nullptr)
	//   T_UCHAR3     None (returns nullptr)
	//   T_UCHAR4     None (returns nullptr)
	//   T_FLOAT      GridType::Float
	//   T_FLOAT3     GridType::Vec3f
	//   T_FLOAT4     None (returns nullptr)
	//   T_INT        GridType::Int32
	//   T_INT3       None (returns nullptr)
	//   T_INT4       None (returns nullptr)
	template<class ValueT, int Node2LogDim, int Node1LogDim, int LeafLogDim>
	inline NanoGridCustom<ValueT, Node2LogDim, Node1LogDim, LeafLogDim>*
		ExportToNanoVDB_CPU(VolumeGVDB& gvdb, uchar channel, const ValueT background,
			const char gridName[nanovdb::GridData::MaxNameSize], nanovdb::GridClass gridClass, size_t& outTotalSize)
	{
		// First, make sure that the type of the VolumeGVDB object matches the type of the NanoVDB tree
		// we will return.
		// TODO: Check assert(gvdbPool->getAtlas(channel).type == T_FLOAT);

		// Check that the dimensions of the NanoVDB tree type match the dimensions of the GVDB tree.
		// If these match, then this allows us to go much faster - the lowest-level topologies of the
		// trees will match, so we don't have to rebuild topology, and the main difficulty is copying
		// data into the right places!
		if (gvdb.getLD(0) != LeafLogDim) {
			gprintf("Error in ExportToNanoVDB: The GVDB object's level-0 log dimension (%d) did not "
				"match the template's level-0 log dimension (%d)! Please consider exporting to a NanoVDB "
				"object of the type that matches the GVDB object.", gvdb.getLD(0), LeafLogDim);
			return nullptr;
		}
		if (gvdb.getLD(1) != Node1LogDim) {
			gprintf("Error in ExportToNanoVDB: The GVDB object's level-0 log dimension (%d) did not "
				"match the template's level-0 log dimension (%d)! Please consider exporting to a NanoVDB "
				"object of the type that matches the GVDB object.", gvdb.getLD(0), LeafLogDim);
			return nullptr;
		}
		if (gvdb.getLD(2) != Node2LogDim) {
			gprintf("Error in ExportToNanoVDB: The GVDB object's level-0 log dimension (%d) did not "
				"match the template's level-0 log dimension (%d)! Please consider exporting to a NanoVDB "
				"object of the type that matches the GVDB object.", gvdb.getLD(0), LeafLogDim);
			return nullptr;
		}

		// A NanoVDB volume is essentially a single block of memory, storing data and offsets into this
		// memory.
		// So, the first thing we need to do is compute how much memory we'll need for the entire volume.
		// A NanoGrid is a Grid<NanoTree<RootNode<InternalNode<InternalNode<LeafNode<value type, Coord, Log2Dim=3>, Log2Dim1>, Log2Dim2>>
		// At this moment, NanoVDB assumes that the tree only has depth 3. However, I think the
		// LOG2DIM=3 restriction is not too bad, and can be worked around (even in a library!) without
		// having to do more complicated things.
		// In memory, it can be viewed like this:
		//   GridData
		//   number of GridBlindMetaData objects (contains offsets to contents)
		//   TreeData (contains pointers to root, internal nodes, and leaves)
		//   root
		//     (number of level-2 nodes) Tiles
		//   level-2 nodes
		//   level-1 nodes
		//   leaves
		//   contents of GridBlindMetaData

		// Copy all atlas data to the CPU
		gvdb.FetchPoolCPU();

		// Create some aliases for types to make the code below shorter
		using GridT = NanoGridCustom<ValueT, Node2LogDim, Node1LogDim, LeafLogDim>;
		using TreeT = GridT::TreeType;
		using RootT = TreeT::RootType;
		using Node2T = RootT::ChildNodeType;
		using Node1T = Node2T::ChildNodeType;
		using LeafT = Node1T::ChildNodeType;

		// Minimum and maximum values for all types
		const ValueT valueTMaxBound = nanovdb::Maximum<ValueT>::value();
		const ValueT valueTMinBound = -valueTMaxBound;

		// Denotes the different regions of a NanoVDB file/memory representation.
		enum Region {
			R_GRID,
			R_TREE,
			R_ROOT,
			R_NODE2,
			R_NODE1,
			R_LEAF,
			R_COUNT
		};

		// Count the number of nodes at each level. At the moment, limit the number of nodes of each
		// type to INT_MAX (2^31-1).
		int numNode2s, numNode1s, numLeaves;
		if (!ExportToNanoVDB_GetNumNodes(gvdb, 2, numNode2s)) {
			return nullptr;
		}
		if (!ExportToNanoVDB_GetNumNodes(gvdb, 1, numNode1s)) {
			return nullptr;
		}
		if (!ExportToNanoVDB_GetNumNodes(gvdb, 0, numLeaves)) {
			return nullptr;
		}

		// Compute the size of each region.
		size_t dataSizes[R_COUNT];
		static_assert(sizeof(GridT) == sizeof(nanovdb::GridData), "GridT must inherit from GridData.");
		dataSizes[R_GRID] = sizeof(nanovdb::GridData);
		dataSizes[R_TREE] = sizeof(TreeT);
		dataSizes[R_ROOT] = sizeof(RootT) + sizeof(RootT::Tile) * numNode2s;
		dataSizes[R_NODE2] = numNode2s * sizeof(Node2T);
		dataSizes[R_NODE1] = numNode1s * sizeof(Node1T);
		dataSizes[R_LEAF] = numLeaves * sizeof(LeafT);

		// Compute offsets into memory using an exclusive prefix sum; the last element in this array
		// will hold the size of all of the memory we need to allocate.
		// (e.g. this turns {3, 5, 2, 5} into {0, 3, 8, 10, 15}.)
		size_t dataOffsetsBytes[R_COUNT + 1];
		dataOffsetsBytes[0] = 0;
		for (int i = 1; i <= R_COUNT; i++) {
			dataOffsetsBytes[i] = dataOffsetsBytes[i - 1] + dataSizes[i - 1];
		}

		// Allocate the memory!
		uint8_t* bufferData = new uint8_t[dataOffsetsBytes[R_COUNT]];
		// Set the data to 0 initially for reproducibility.
		memset(bufferData, 0, dataOffsetsBytes[R_COUNT]);

		// We now fill in the data in this block of memory manually.
		// In order to compute bounding boxes and min/max values (which exist in NanoVDB, but aren't
		// stored in GVDB), we work from the leaf nodes up to the level-2 nodes.
		// For higher-level nodes, it turns out it works to operate in this order:
		// - Leaf nodes
		// - Level-1 nodes
		// - Level-2 nodes
		// - Basic grid setup
		// - Tree
		// - Root
		// - Grid

		//---------------------------------------------------------------------------------------------
		// Leaves (GVDB bricks)

		// Start by skipping directly to the location of the leaves in memory.
		LeafT* leafNodes = reinterpret_cast<LeafT*>(bufferData + dataOffsetsBytes[R_LEAF]);
		// Get number of children and range. Perform bounds checking.
		const uint64 leafVoxels64 = gvdb.getVoxCnt(0);
		const uint64 bytesPerLeaf64 = leafVoxels64 * sizeof(float);
		if (bytesPerLeaf64 > INT_MAX) {
			gprintf("Error in ExportToNanoVDB: Bytes per brick (%ull) was more than INT_MAX.",
				bytesPerLeaf64);
			return nullptr;
		}
		const uint32 leafVoxels = static_cast<uint32>(leafVoxels64);
		const Vector3DI leafRange = gvdb.getRange(0);
		assert(leafVoxels == leafRange.x * leafRange.y * leafRange.z); // Consistency check
		if (leafRange.x >= 256) { // All leafRange components should be the same
			gprintf("Error in ExportToNanoVDB: Brick dimension (%d) was larger than supported by "
				"NanoVDB as of this writing.", leafRange.x);
			return nullptr;
		}

		// Used to store the result of copying the brick from the atlas (without apron) back to the CPU
		const int bytesPerLeaf = static_cast<int>(bytesPerLeaf64);
		DataPtr brickCopy{};
		gvdb.AllocData(brickCopy, bytesPerLeaf, 1, true);

		for (int leafIdx = 0; leafIdx < numLeaves; leafIdx++) {
			nvdb::Node* gvdbNode = gvdb.getNodeAtLevel(leafIdx, 0);

			// Sometimes, the GVDB node list can contain nodes that were previously part of trees.
			// These will be ignored, although there will still be space for them in the NanoVDB volume.
			// (This isn't a requirement of exporting - we could prune these nodes as well.)
			if (gvdbNode->mChildList != ID_UNDEFL) {
				continue;
			}

			// The NanoVDB node and data
			LeafT& node = leafNodes[leafIdx];
			LeafT::DataType& nodeData = *reinterpret_cast<LeafT::DataType*>(&node);

			// All values in a brick are active in GVDB
			nodeData.mValueMask.set(true);

			// Minimum and maximum value of all elements
			ValueT minValue = valueTMaxBound;
			ValueT maxValue = valueTMinBound;

			// Copy the brick from the GPU to the CPU in linear order, the same as NanoVDB - i.e. in
			// the order ((x * T) + y) * T + z. Here, there's no need to swizzle the indexes.
			gvdb.AtlasRetrieveBrickXYZ(channel, gvdbNode->mValue, brickCopy);
			assert(brickCopy.cpu != nullptr);
			ValueT* brickData = reinterpret_cast<ValueT*>(brickCopy.cpu);

			for (uint32 voxel = 0; voxel < leafVoxels; voxel++) {
				const float value = brickData[voxel];
				nodeData.mValues[voxel] = value;


				minValue = ExportToNanoVDB_Min(minValue, value);
				maxValue = ExportToNanoVDB_Max(maxValue, value);
			}

			nodeData.mValueMin = minValue;
			nodeData.mValueMax = maxValue;
			// Since all voxels are active, the bounding box of active values of this node is the
			// bounding box of the node itself:
			nodeData.mBBoxMin = { gvdbNode->mPos.x, gvdbNode->mPos.y, gvdbNode->mPos.z };
			nodeData.mBBoxDif[0] = leafRange.x;
			nodeData.mBBoxDif[1] = leafRange.y;
			nodeData.mBBoxDif[2] = leafRange.z;
		}

		gvdb.FreeData(brickCopy);

		//---------------------------------------------------------------------------------------------
		// Level-1 nodes
		// As of this writing, Root doesn't have an easy way to go directly to the nth node, so we just
		// skip to the region directly:
		Node1T* level1Nodes = reinterpret_cast<Node1T*>(bufferData + dataOffsetsBytes[R_NODE1]);
		if (!ExportToNanoVDB_ProcessInternalNodes<Node1T>(gvdb, level1Nodes, numNode1s, background)) {
			return nullptr;
		}

		//---------------------------------------------------------------------------------------------
		// Level-2 nodes
		Node2T* level2Nodes = reinterpret_cast<Node2T*>(bufferData + dataOffsetsBytes[R_NODE2]);
		if (!ExportToNanoVDB_ProcessInternalNodes<Node2T>(gvdb, level2Nodes, numNode2s, background)) {
			return nullptr;
		}

		//---------------------------------------------------------------------------------------------
		// Grid setup
		// Fill in just the minimum to get going from the grid to the tree to work:
		nanovdb::GridData* gridData = reinterpret_cast<nanovdb::GridData*>(bufferData);
		if (std::is_same<ValueT, float>::value) {
			gridData->mGridType = nanovdb::GridType::Float;
		}
		else if (std::is_same<ValueT, nanovdb::Vec3f>::value) {
			gridData->mGridType = nanovdb::GridType::Vec3f;
		}
		else if (std::is_same<ValueT, int32_t>::value) {
			gridData->mGridType = nanovdb::GridType::Int32;
		}
		gridData->mBlindDataCount = 0;

		// We can't call handle.grid<ValueT>, because our grid dimensions might not match NanoGrid,
		// which handle.grid returns. So we reproduce the implementation of handle.grid() here:
		GridT* gridPtr = reinterpret_cast<GridT*>(bufferData);
		assert(gridPtr != nullptr);
		assert(gridPtr->memUsage() == dataSizes[R_GRID]);

		//---------------------------------------------------------------------------------------------
		// Tree
		const TreeT& treePtr = gridPtr->tree();
		// Consistency check
		assert((uint8_t*)&treePtr - (uint8_t*)gridPtr == dataOffsetsBytes[R_TREE]);
		{
			// Get the tree data, and *remove const-ness* so that we can modify the memory:
			TreeT::DataType* treeData = const_cast<TreeT::DataType*>(treePtr.data());
			if (treeData == nullptr) {
				gprintf("Internal error in ExportToNanoVDB: treeData was nullptr!");
				return nullptr;
			}
			static_assert(sizeof(TreeT::DataType::mBytes) == sizeof(uint64_t) * 4, "Tree must have a depth of 3.");
			// Filling in the tree is much simpler; we simply give the offsets from treePtr to each of
			// the regions, and the number of nodes in each region. Note that the indices of mBytes
			// and mCount refer to the level of the nodes.
			treeData->mBytes[0] = dataOffsetsBytes[R_LEAF] - dataOffsetsBytes[R_TREE];
			treeData->mBytes[1] = dataOffsetsBytes[R_NODE1] - dataOffsetsBytes[R_TREE];
			treeData->mBytes[2] = dataOffsetsBytes[R_NODE2] - dataOffsetsBytes[R_TREE];
			treeData->mBytes[3] = dataOffsetsBytes[R_ROOT] - dataOffsetsBytes[R_TREE];

			treeData->mCount[0] = numLeaves;
			treeData->mCount[1] = numNode1s;
			treeData->mCount[2] = numNode2s;
			treeData->mCount[3] = 1; // There's only one root!
		}
		assert(treePtr.memUsage() == dataSizes[R_TREE]);

		//---------------------------------------------------------------------------------------------
		// Root
		// This also computes the bounding box and min and max values of the grid from the level-2 nodes.
		const RootT& rootPtr = treePtr.root();

		nanovdb::CoordBBox indexAABB; // Index-space bounding box
		ValueT rootMaxValue = valueTMinBound;
		ValueT rootMinValue = valueTMaxBound;

		// Consistency check
		assert((uint8_t*)&rootPtr - (uint8_t*)gridPtr == dataOffsetsBytes[R_ROOT]);

		{
			// Get the root data, and remove const-ness so that we can modify the memory:
			RootT::DataType* rootData = const_cast<RootT::DataType*>(rootPtr.data());
			if (rootData == nullptr) {
				gprintf("Internal error in ExportToNanoVDB: rootData was nullptr!");
				return nullptr;
			}

			// All voxels in the leaves of the GVDB volume are active, so this is the total volume of
			// of the leaves:
			rootData->mActiveVoxelCount = numLeaves * gvdb.getVoxCnt(0);

			rootData->mBackground = background;

			// One tile per level-2 node:
			rootData->mTileCount = numNode2s;

			// Iterate over tiles. We'll initially write them in linear order, then sort them by key
			// afterwards.
			for (int tileIdx = 0; tileIdx < numNode2s; tileIdx++) {
				// Copy data from the GVDB level-2 node to the NanoVDB tile:
				RootT::Tile& tile = rootData->tile(tileIdx);
				nvdb::Node* gvdbNode = gvdb.getNodeAtLevel(tileIdx, 2);

				// Consistency checks
				assert(gvdbNode != nullptr);
				assert(gvdbNode->mLev == 2);

				// Get the index of the minimum corner (may not be within the bounding box!)
				nanovdb::Coord minIndex(gvdbNode->mPos.x, gvdbNode->mPos.y, gvdbNode->mPos.y);

				if (gvdbNode->mFlags != 0) {
					// Node is active; set its child index.
					tile.setChild(minIndex, tileIdx);
				}
				else {
					// For inactive internal GVDB nodes, we simply set their value to the background.
					// This could be changed in the future if needed.
					uint8_t isActiveState = gvdbNode->mFlags;
					tile.setValue(minIndex, isActiveState, background);
				}

				// Then get the actual child, and process its bounding box and min/max values.
				const Node2T& child = rootData->child(tile);
				const nanovdb::CoordBBox childAABB = child.bbox();
				const ValueT childMinValue = child.valueMin();
				const ValueT childMaxValue = child.valueMax();

				for (int c = 0; c < 3; c++) {
					indexAABB.min()[c] = std::min(indexAABB.min()[c], childAABB.min()[c]);
					indexAABB.max()[c] = std::max(indexAABB.max()[c], childAABB.max()[c]);
				}
				rootMinValue = ExportToNanoVDB_Min(rootMinValue, childMinValue);
				rootMaxValue = ExportToNanoVDB_Max(rootMaxValue, childMaxValue);
			}

			// Sort the tiles so that their keys are in ascending order. This makes it so that
			// RootNode::findTile can efficiently find them.
			{
				RootT::Tile* startTile = &rootData->tile(0);
				RootT::Tile* endTile = &rootData->tile(numNode2s - 1) + 1; // Note that this is one after the last tile

				// This says "Treat startTile and endTile as random access iterators. Sort them using
				// this comparison operator: two tiles b...a in order are sorted if a.key < b.key."
				std::sort(startTile, endTile, [](const RootT::Tile& a, const RootT::Tile& b) {
					return a.key < b.key;
				});
			}

			// Set the bounding box and min/max values for the whole volume:
			rootData->mBBox = indexAABB;
			rootData->mValueMin = rootMinValue;
			rootData->mValueMax = rootMaxValue;
		}
		assert(rootPtr.memUsage() == dataSizes[R_ROOT]);

		//---------------------------------------------------------------------------------------------
		// Grid
		{
			nanovdb::GridData* gridData = reinterpret_cast<nanovdb::GridData*>(bufferData);
			gridData->mMagic = NANOVDB_MAGIC_NUMBER;
			memcpy(gridData->mGridName, gridName, nanovdb::GridData::MaxNameSize);

			// Get the GVDB index-to-world transform and copy it to a format Map can read
			Matrix4F xform = gvdb.getTransform(); // Make a copy (this is inefficient!)
			{
				float indexToWorld[4][4];
				for (int row = 0; row < 4; row++) {
					for (int col = 0; col < 4; col++) {
						indexToWorld[row][col] = xform(row, col);
					}
				}
				float worldToIndex[4][4];
				xform.InvertTRS();
				for (int row = 0; row < 4; row++) {
					for (int col = 0; col < 4; col++) {
						worldToIndex[row][col] = xform(row, col);
					}
				}
				gridData->mMap.set(indexToWorld, worldToIndex, 1.0); // mTaper seems to be unused
			}

			// For the world bounding box, we compute a bounding box containing GVDB's bounding box
			// after transformation by GVDB's matrix:
			{
				nanovdb::BBox<nanovdb::Vec3R> worldAABB(
					{ FLT_MAX, FLT_MAX, FLT_MAX }, // Initial AABB min
					{ -FLT_MAX, -FLT_MAX, -FLT_MAX } // Initial AABB max
				);

				nanovdb::BBox<nanovdb::Vec3R> indexAABB;
				for (int minMax = 0; minMax < 2; minMax++) {
					for (int c = 0; c < 3; c++) {
						indexAABB[minMax][c] = static_cast<double>(rootPtr.bbox()[minMax][c]);
					}
				}

				if (indexAABB.min() == indexAABB.max()) {
					gprintf("Warning from ExportToNanoVDB: Bounding box had zero volume!");
				}

				for (int choiceFlags = 0; choiceFlags < 8; choiceFlags++) {
					nanovdb::Vec3R vertex;
					vertex[0] = indexAABB[(choiceFlags & 1)][0];
					vertex[1] = indexAABB[(choiceFlags & 2) >> 1][1];
					vertex[2] = indexAABB[(choiceFlags & 4) >> 2][2];
					vertex = gridData->mMap.applyMap(vertex);
					worldAABB.expand(vertex);
				}

				gridData->mWorldBBox = worldAABB;
			}

			// GridData would like a uniform scale, but that's not really possible to provide, since
			// GVDB supports arbitrary voxel transforms (e.g. think of squished voxels).
			// For now, we use the approach GridBuilder uses, which is ||map((1, 0, 0)) - map((0,0,0))||.
			// However, for a different approximation, we could use something like sqrt(tr(A*A)/3),
			// where A is the upper-left 3x3 block of xform; if A is normal, this gives the root mean
			// square of the singular values of A.
			gridData->mUniformScale = (gridData->applyMap(nanovdb::Vec3d(1, 0, 0))
				- gridData->applyMap(nanovdb::Vec3d(0))).length();

			gridData->mGridClass = gridClass;
			// We've already filled in mBlindDataCount and mGridType
		}

		outTotalSize = dataOffsetsBytes[R_COUNT];
		return gridPtr;
	}

	template<class TreeType>
	inline void FreeExportedNanoVDB(TreeType* treePtr) {
		uint8_t* originalBuffer = reinterpret_cast<uint8_t*>(treePtr);
		delete[] originalBuffer;
	}

	// image must have length width * height * 4
	template<class ValueT, int Node2LogDim, int Node1LogDim, int LeafLogDim>
	inline void RenderNanoVDB_CPU(NanoGridCustom<ValueT, Node2LogDim, Node1LogDim, LeafLogDim>* grid,
		Camera3D* camera, uchar* image, uint width, uint height)
	{
		if (grid == nullptr) return;
		if (image == nullptr) return;
		if (camera == nullptr) return;

		// Render the image
		auto acc = grid->getAccessor();
		// Camera origin in world-space
		const nanovdb::Vec3f eye(camera->from_pos.x, camera->from_pos.y, camera->from_pos.z);
		// Camera directions in world-space
		const Vector4DF camTopLeftWS = camera->tlRayWorld;
		const Vector4DF camRightWS = camera->trRayWorld - camera->tlRayWorld;
		const Vector4DF camDownWS = camera->blRayWorld - camera->tlRayWorld;

		for (uint yi = 0; yi < height; yi++) {
			for (uint xi = 0; xi < width; xi++) {
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
				// Convert to index-space
				nanovdb::Ray<float> iRay = rayWS.worldToIndexF(*grid);
				// Find the closest zero-crossing
				nanovdb::Coord intersectionPoint;
				float value = 0.0f;
				float time = 0.0f;
				if (nanovdb::ZeroCrossing(iRay, acc, intersectionPoint, value, time)) {
					// Compute shading using the gradient
					nanovdb::Vec3f gradient(-acc.getValue(intersectionPoint)); // DEBUG, can maybe use -value?
					intersectionPoint[0] += 1;
					gradient[0] += acc.getValue(intersectionPoint);
					intersectionPoint[0] -= 1;
					intersectionPoint[1] += 1;
					gradient[1] += acc.getValue(intersectionPoint);
					intersectionPoint[1] -= 1;
					intersectionPoint[2] += 1;
					gradient[2] += acc.getValue(intersectionPoint);
					gradient.normalize();

					float lighting = std::abs(gradient.dot(iRay.dir()));
					unsigned char exportVal = static_cast<unsigned char>(255.0f * std::min(1.0f, lighting));
					// Debug
					image[4 * (yi * width + xi) + 0] = exportVal;
					image[4 * (yi * width + xi) + 1] = exportVal;
					image[4 * (yi * width + xi) + 2] = exportVal;
				}
				else {
					// Draw a checkerboard that alternates every 128 pixels
					const int checkerboard = 1 << 7;
					if ((xi & checkerboard) != (yi & checkerboard)) {
						image[4 * (yi * width + xi) + 0] = 255;
						image[4 * (yi * width + xi) + 1] = 255;
						image[4 * (yi * width + xi) + 2] = 255;
					}
					else {
						image[4 * (yi * width + xi) + 0] = 0;
						image[4 * (yi * width + xi) + 1] = 0;
						image[4 * (yi * width + xi) + 2] = 0;
					}
				}
				image[4 * (yi * width + xi) + 3] = 255;
			}
		}
	}
	*/
}

#endif // #ifndef DEF_GVDB_EXPORT_NANOVDB
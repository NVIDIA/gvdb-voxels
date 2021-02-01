//-----------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2020 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Version 1.1.1: Neil Bickford, 8/12/2020
//----------------------------------------------------------------------------------

#include <algorithm> // for min/max

// GVDB export to NanoVDB
#include "gvdb_export_nanovdb.h"

// Sample utilities
#include "file_png.h"

VolumeGVDB gvdb;

int main(int argc, char* argv) {
	// Width and height for rendering
	const int width = 1024, height = 768;

	// Initialize GVDB
	gprintf("Starting GVDB.\n");
	gvdb.SetVerbose(true); // Enable/disable console output from gvdb
#ifndef NDEBUG
	gvdb.SetDebug(true);
#endif // #ifdef _DEBUG
	gvdb.SetCudaDevice(GVDB_DEV_FIRST);
	gvdb.Initialize();

	// Add search paths for gvdb.FindFile()
	gvdb.AddPath("../source/shared_assets/");
	gvdb.AddPath("../shared_assets/");
	gvdb.AddPath(ASSET_PATH);

	// Load VBX
	char scnpath[1024];
	if (!gvdb.FindFile("explosion.vbx", scnpath)) {
		gprintf("Cannot find vbx file.\n");
		exit(-1);
	}
	printf("Loading VBX. %s\n", scnpath);
	if (!gvdb.LoadVBX(scnpath)) {
		gerror();
	}

	// Export this to a new NanoVDB volume.
	// This allocation belongs to GVDB's CUDA context, so we make sure to switch to that context
	// when we render it later on. However, we could also avoid having to do this context switch
	// by making GVDB's memory allocations accessible to the current context, by using CUDA's
	// Peer Context Memory Access functions such as cuCtxEnablePeerAccess.
	gprintf("Export to a NanoVDB volume.\n");

	gvdb.TimerStart();
	float background = 0.0f; // The background value of the volume - the value all unspecified voxels have.
	const char gridName[256] = "explosion"; // The name of the grid - GVDB doesn't have this, but NanoVDB does.
	size_t gridSize = 0; // Will store the size of the resulting memory buffer.
	CUdeviceptr deviceGrid = ExportToNanoVDB(gvdb, 0, &background, gridName, nanovdb::GridClass::LevelSet, &gridSize);
	gprintf("Finished converting to a NanoVDB volume in %f ms.\n", gvdb.TimerStop());

	// Render the volume using the render kernel in cuda_export_nanovdb.cu.
	gprintf("Rendering...\n");
	gvdb.TimerStart();

	// Set up the camera
	Camera3D camera;
	camera.setFov(38.0f);
	float maxAABB = std::max(std::max(gvdb.getVolMax().x, gvdb.getVolMax().y), gvdb.getVolMax().z);
	const float voxelsizeApprox
		= static_cast<float>((gvdb.getTransform() * Vector4DF(1.0f, 0.0f, 0.0f, 0.0f)).Length());
	camera.setOrbit(Vector3DF(20, 30, 0), gvdb.getVolMax() * 0.5f * voxelsizeApprox,
		3.0f * maxAABB * voxelsizeApprox, 1.0f);
	camera.setAspect(static_cast<float>(width) / static_cast<float>(height));

	// Allocate space for the image
	uchar* image = new uchar[width * height * 4];

	// Get the GVDB context
	CUcontext gvdbContext = gvdb.getContext();
	RenderNanoVDB(gvdbContext, deviceGrid, &camera, width, height, image);
	gprintf("Rendered in %f ms. Saving to out_nanovdb.png.\n", gvdb.TimerStop());

	save_png("out_nanovdb.png", image, width, height, 4);

	// Clean up
	delete[] image;

	cudaCheck(cuCtxPushCurrent(gvdbContext), "", "main", "cuCtxPushCurrent", "gvdbContext", DEBUG_EXPORT_NANOVDB);
	cudaCheck(cuMemFree(deviceGrid), "", "main", "cuMemFree", "deviceGrid", DEBUG_EXPORT_NANOVDB);
	CUcontext tempContext;
	cudaCheck(cuCtxPopCurrent(&tempContext), "", "main", "cuCtxPopCurrent", "tempContext", DEBUG_EXPORT_NANOVDB);

	printf("Done!\n");
}
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
// This sample shows how to export a GVDB volume to a NanoVDB volume.
// It loads a GVDB volume from a VBX volume, then calls ExportToNanoVDB to allocate and create a
// NanoVDB grid on the GPU. Finally, it renders this volume using a small NanoVDB kernel in
// cuda_export_nanovdb.cu, mimicking how gRenderToFile renders a volume. It also shows how to
// use a GVDB Camera class with NanoVDB rendering.
//
// Since NanoVDB is a header-only library, you don't need to compile any dependencies other than
// GVDB in order to compile this sample - only add the correct NanoVDB folder to the list of
// include directories.
//
// At the moment, GVDB-to-NanoVDB export is implemented entirely in this sample in
// gvdb_export_nanovdb.h, gvdb_export_nanovdb.cpp, and cuda_export_nanovdb.cu, but the plan is to
// move this to the core GVDB library in the future.
//
// In addition to this file, make sure to check out the rendering code in cuda_export_nanovdb.cu!
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
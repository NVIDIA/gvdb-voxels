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

#include <algorithm>
#include <stdio.h>

// GVDB export to NanoVDB
#include "gvdb_export_nanovdb.h"

// Sample utilities
#include "file_png.h"

int main(int argc, char* argv) {
	const int width = 1024, height = 768;

	VolumeGVDB gvdb;
	printf("Starting GVDB.\n");
	gvdb.SetVerbose(true); // Enable/disable console output from gvdb
#ifndef NDEBUG
	gvdb.SetDebug(true);
#endif // #ifdef _DEBUG
	gvdb.SetCudaDevice(GVDB_DEV_FIRST);
	gvdb.Initialize();
	gvdb.AddPath("../source/shared_assets/");
	gvdb.AddPath("../shared_assets/");
	gvdb.AddPath(ASSET_PATH);

	// Load VBX
	char scnpath[1024];
	if (!gvdb.FindFile("explosion.vbx", scnpath)) {
		printf("Cannot find vbx file.\n");
		exit(-1);
	}
	printf("Loading VBX. %s\n", scnpath);
	if (!gvdb.LoadVBX(scnpath)) {
		gerror();
	}

	const char gridName[256] = "dragon_ls";

	printf("Export to a NanoVDB volume.\n");

	gvdb.TimerStart();

	float background = 0.0f;
	size_t gridSize = 0;
	CUdeviceptr deviceGrid = ExportToNanoVDB(gvdb, 0, &background, gridName, nanovdb::GridClass::LevelSet, &gridSize);

	gprintf("Finished converting to a NanoVDB volume in %f ms.\n", gvdb.TimerStop());

	gprintf("Rendering...\n");
	gvdb.TimerStart();

	// Set up the camera and space for the image
	Camera3D camera;
	camera.setFov(38.0f);
	float maxAABB = std::max(std::max(gvdb.getVolMax().x, gvdb.getVolMax().y), gvdb.getVolMax().z);
	const float voxelsizeApprox
		= static_cast<float>((gvdb.getTransform() * Vector4DF(1.0f, 0.0f, 0.0f, 0.0f)).Length());
	camera.setOrbit(Vector3DF(20, 30, 0), gvdb.getVolMax() * 0.5f * voxelsizeApprox,
		3.0f * maxAABB * voxelsizeApprox, 1.0f);
	camera.setAspect(static_cast<float>(width) / static_cast<float>(height));
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
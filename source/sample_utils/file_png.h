//-----------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2022 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#ifndef DEF_GVDB_FILE_PNG_H
#define DEF_GVDB_FILE_PNG_H

#include "lodepng.h"

// Helper function for saving a PNG file.
// Automatically fills out lodepng arguments. Prints a message on error.
// This function is defined in file_png.h to preserve backwards compatibility.
inline void save_png(char* filename, unsigned char* image_data, int width, int height, int num_channels)
{
	unsigned error = lodepng::encode(filename, image_data, width, height,
		(num_channels == 3) ? LodePNGColorType::LCT_RGB : LodePNGColorType::LCT_RGBA,
		8);

	if(error) printf("png encoder error: %s\n", lodepng_error_text(error));
}

#endif
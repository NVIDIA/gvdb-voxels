#ifndef __CUH_GVDB_CUTILS__
#define __CUH_GVDB_CUTILS__

#pragma once

#include<cuda.h>
#include<stdint.h>

#include<thrust/device_ptr.h>
#include<thrust/device_vector.h>
#include<thrust/extrema.h>
#include<thrust/sort.h>


void gvdbDeviceRadixSort(CUdeviceptr inSource, const uint32_t inCount);

void gvdbDeviceMaxElementF(CUdeviceptr inSource, CUdeviceptr inDest, const uint32_t inCount);
void gvdbDeviceMinElementF(CUdeviceptr inSource, CUdeviceptr inDest, const uint32_t inCount);


#endif
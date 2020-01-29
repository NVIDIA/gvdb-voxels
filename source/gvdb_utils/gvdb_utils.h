#ifndef __H_GVDB_UTILS__
#define __H_GVDB_UTILS__

#pragma once

#include<cuda.h>
#include<stdint.h>

void gvdbRadixSort(CUdeviceptr inSource, const uint32_t inCount);

void gvdbMaxElementF(CUdeviceptr inSource, CUdeviceptr inDest, const uint32_t inCount);
void gvdbMinElementF(CUdeviceptr inSource, CUdeviceptr inDest, const uint32_t inCount);




#endif

// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt 
// in the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
* @file
* cudpp_compress.h
*
* @brief Compress functionality header file - contains CUDPP interface (not public)
*/

#ifndef _CUDPP_COMPRESS_H_
#define _CUDPP_COMPRESS_H_

class CUDPPCompressPlan;
class CUDPPBwtPlan;
class CUDPPMtfPlan;

// Compress
extern "C"
void allocCompressStorage(CUDPPCompressPlan* plan);

extern "C"
void freeCompressStorage(CUDPPCompressPlan* plan);

extern "C"
void cudppCompressDispatch(unsigned char *d_uncompressed,
                           int *d_bwtIndex,
                           unsigned int *d_histSize,
                           unsigned int *d_hist,
                           unsigned int *d_encodeOffset,
                           unsigned int *d_compressedSize,
                           unsigned int *d_compressed,
                           size_t numElements,
                           const CUDPPCompressPlan *plan);

// BWT
extern "C"
void allocBwtStorage(CUDPPBwtPlan* plan);

extern "C"
void freeBwtStorage(CUDPPBwtPlan* plan);

extern "C"
void cudppBwtDispatch(unsigned char *d_in,
                      unsigned char *d_out,
                      int *d_index,
                      size_t numElements,
                      const CUDPPBwtPlan *plan);

// MTF
extern "C"
void allocMtfStorage(CUDPPMtfPlan* plan);

extern "C"
void freeMtfStorage(CUDPPMtfPlan* plan);

extern "C"
void cudppMtfDispatch(unsigned char *d_in,
                      unsigned char *d_out,
                      size_t numElements,
                      const CUDPPMtfPlan *plan);

#endif // _CUDPP_COMPRESS_H_

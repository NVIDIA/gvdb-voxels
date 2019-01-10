// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
* @file
* cudpp_segscan.h
*
* @brief Scan functionality header file - contains CUDPP interface (not public)
*/

#ifndef _CUDPP_SEGMENTEDSCAN_H_
#define _CUDPP_SEGMENTEDSCAN_H_

class CUDPPSegmentedScanPlan;

extern "C"
void allocSegmentedScanStorage(CUDPPSegmentedScanPlan *plan);

extern "C"
void freeSegmentedScanStorage(CUDPPSegmentedScanPlan *plan);

extern "C"
void cudppSegmentedScanDispatch(void                   *d_out, 
                                const void             *d_idata,
                                const unsigned int     *d_iflags,
                                size_t                 numElements,
                                const CUDPPSegmentedScanPlan *plan);

#endif // _CUDPP_SEGMENTEDSCAN_H_

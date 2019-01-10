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
* cudpp_compact.h
*
* @brief Compact functionality header file - contains CUDPP interface (not public)
*/

#ifndef _CUDPP_COMPACT_H_
#define _CUDPP_COMPACT_H_

class CUDPPCompactPlan;

extern "C"
void allocCompactStorage(CUDPPCompactPlan* plan);

extern "C"
void freeCompactStorage(CUDPPCompactPlan* plan);

extern "C"
void cudppCompactDispatch(void                   *d_out, 
                          size_t                 *d_numValidElements,
                          const void             *d_in, 
                          const unsigned int     *d_isValid,
                          size_t                 numElements,
                          const CUDPPCompactPlan *plan);

#endif // _CUDPP_COMPACT_H_

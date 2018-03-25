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
* cudpp_rand.h
*
* @brief rand functionality header file - contains CUDPP interface (not public)
*/

#ifndef __CUDPP_RAND_H__
#define __CUDPP_RAND_H__

#include "cudpp_globals.h"
#include "cudpp.h"
#include "cudpp_plan.h"

extern "C"
void cudppRandDispatch(void * d_out, size_t num_elements, const CUDPPRandPlan * plan);

#endif //__CUDPP_RAND_H__



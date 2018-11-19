// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt 
// in the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
* @file
* cudpp_reduce.h
*
* @brief Reduce functionality header file - contains CUDPP interface (not public)
*/

#ifndef _CUDPP_REDUCE_H_
#define _CUDPP_REDUCE_H_

class CUDPPReducePlan;


void allocReduceStorage(CUDPPReducePlan *plan);

void freeReduceStorage(CUDPPReducePlan *plan);

void cudppReduceDispatch(void                *d_out, 
                         const void          *d_in, 
                         size_t              numElements,
                         const CUDPPReducePlan *plan);

#endif // _CUDPP_REDUCE_H_

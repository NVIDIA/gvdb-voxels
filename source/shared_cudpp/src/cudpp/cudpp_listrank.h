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
* cudpp_listrank.h
*
* @brief ListRank functionality header file - contains CUDPP interface (not public)
*/

#ifndef _CUDPP_LISTRANK_H_
#define _CUDPP_LISTRANK_H_

class CUDPPListRankPlan;

// ListRank
extern "C"
void allocListRankStorage(CUDPPListRankPlan* plan);

extern "C"
void freeListRankStorage(CUDPPListRankPlan* plan);

extern "C"
CUDPPResult cudppListRankDispatch(void *d_ranked_values,
                           void *d_unranked_values,
                           void *d_next_indices,
                           size_t head,
                           size_t numElements,
                           const CUDPPListRankPlan *plan);

#endif // _CUDPP_LISTRANK_H_
// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// ------------------------------------------------------------- 
#ifndef   __MERGESORT_H__
#define   __MERGESORT_H__

#include "cudpp_globals.h"
#include "cudpp.h"
#include "cudpp_plan.h"

extern "C"
void allocMergeSortStorage(CUDPPMergeSortPlan* plan);

extern "C"
void freeMergeSortStorage(CUDPPMergeSortPlan* plan);

extern "C"
void cudppMergeSortDispatch(void       *keys,
                            void       *values,
                            size_t      numElements,
                            const       CUDPPMergeSortPlan *plan);


#endif // __MERGESORT_H__

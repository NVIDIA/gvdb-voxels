// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// ------------------------------------------------------------- 
#ifndef   __RADIXSORT_H__
#define   __RADIXSORT_H__

#include "cudpp_globals.h"
#include "cudpp.h"
#include "cudpp_plan.h"


void allocRadixSortStorage(CUDPPRadixSortPlan* plan);

void freeRadixSortStorage(CUDPPRadixSortPlan* plan);

void cudppRadixSortDispatch(void       *keys,
                            void       *values,
                            size_t      numElements,
                            const       CUDPPRadixSortPlan *plan);


#endif // __RADIXSORT_H__

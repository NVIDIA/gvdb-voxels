// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// ------------------------------------------------------------- 
#include <cudpp.h>
#include <stdio.h>
#include <limits.h>

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set for list ranking
//! Each element is a node in the linked-list.
//! @param reference        reference data, computed but preallocated
//! @param ivalues          const input values as provided to device
//! @param inextindices     const input next indices as provided to device
//! @param head             input head node index
//! @param count            number of elements in reference / linked-list
////////////////////////////////////////////////////////////////////////////////
template <typename T>
void listRankGold( T* reference, const T* ivalues, 
                  const int* inextindices, const unsigned int head,
                  const unsigned int count) 
{
    int cur_id = head;
    for(unsigned int i=0; i<count; i++){
        reference[i] = ivalues[cur_id];
        cur_id = inextindices[cur_id];
    }
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
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
//! Compute reference data set for exclusive sum-scan
//! Each element is the sum of the elements before it in the array.
//! @param reference  reference data, computed but preallocated
//! @param idata      const input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
template <typename T>
unsigned int
compactGold( T* reference, const T* idata, 
             const unsigned int *isValid, const unsigned int len,
             const CUDPPConfiguration & config) 
{

    unsigned int count = 0;
    unsigned int numValidElements = 0;

    if (config.options & CUDPP_OPTION_BACKWARD)
    {
        // first have to count total valid elements
        for( unsigned int i = 0; i < len; ++i) 
        {
            if (isValid[i] != 0.0f) 
                numValidElements++;
        }
    }        

    for( unsigned int i = 0; i < len; ++i) 
    {
        if (isValid[i] != 0.0f) {
            if (config.options & CUDPP_OPTION_BACKWARD)
                reference[numValidElements-count-1] = idata[i];
            else
                reference[count] = idata[i];
            ++count;
        }
    }

    return count;
}


// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

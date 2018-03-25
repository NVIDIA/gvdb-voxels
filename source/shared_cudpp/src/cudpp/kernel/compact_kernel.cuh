// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
//  $Revision: 3049 $
//  $Date: 2007-02-26 10:42:36 -0800 (Mon, 26 Feb 2007) $
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
 * @file
 * compact_kernel.cu
 * 
 * @brief CUDPP kernel-level compact routines
 */

#include <cudpp_globals.h>
#include "sharedmem.h"
// #include <stdio.h>

/** \addtogroup cudpp_kernel
  * @{
  */

/** @name Compact Functions
 * @{
 */

/**
 * @brief Consolidate non-null elements - for each non-null element
 * in \a d_in write it to \a d_out, in the position specified by 
 * \a d_isValid. Called by compactArray().
 *
 * @param[out] d_out    Output array of compacted values.
 * @param[out] d_numValidElements The number of elements in d_in with valid flags set to 1.
 * @param[in]  d_indices Positions where non-null elements will go in d_out.
 * @param[in]  d_isValid Flags indicating valid (1) and invalid (0) elements.  
 *             Only valid elements will be copied to \a d_out.
 * @param[in]  d_in     The input array
 * @param[in]  numElements The length of the \a d_in in elements.
 *
 */
template <class T, bool isBackward>
__global__ void compactData(T                  *d_out, 
                            size_t             *d_numValidElements,
                            const unsigned int *d_indices, // Exclusive Sum-Scan Result
                            const unsigned int *d_isValid,
                            const T            *d_in,
                            unsigned int       numElements)
{
    if (threadIdx.x == 0)
    {
        if (isBackward)
            d_numValidElements[0] = d_isValid[0] + d_indices[0];
        else
            d_numValidElements[0] = d_isValid[numElements-1] + d_indices[numElements-1];
    }

    // The index of the first element (in a set of eight) that this
    // thread is going to set the flag for. We left shift
    // blockDim.x by 3 since (multiply by 8) since each block of 
    // threads processes eight times the number of threads in that
    // block
    unsigned int iGlobal = blockIdx.x * (blockDim.x << 3) + threadIdx.x;

    // Repeat the following 8 (SCAN_ELTS_PER_THREAD) times
    // 1. Check if data in input array d_in is null
    // 2. If yes do nothing
    // 3. If not write data to output data array d_out in
    //    the position specified by d_isValid
    if (iGlobal < numElements && d_isValid[iGlobal] > 0) {
        d_out[d_indices[iGlobal]] = d_in[iGlobal];
    }
    iGlobal += blockDim.x;  
    if (iGlobal < numElements && d_isValid[iGlobal] > 0) {
        d_out[d_indices[iGlobal]] = d_in[iGlobal];       
    }
    iGlobal += blockDim.x;
    if (iGlobal < numElements && d_isValid[iGlobal] > 0) {
        d_out[d_indices[iGlobal]] = d_in[iGlobal];
    }
    iGlobal += blockDim.x;
    if (iGlobal < numElements && d_isValid[iGlobal] > 0) {
        d_out[d_indices[iGlobal]] = d_in[iGlobal];
    }
    iGlobal += blockDim.x;
    if (iGlobal < numElements && d_isValid[iGlobal] > 0) {
        d_out[d_indices[iGlobal]] = d_in[iGlobal];
    }
    iGlobal += blockDim.x;
    if (iGlobal < numElements && d_isValid[iGlobal] > 0) {
        d_out[d_indices[iGlobal]] = d_in[iGlobal];
    }
    iGlobal += blockDim.x;
    if (iGlobal < numElements && d_isValid[iGlobal] > 0) {
        d_out[d_indices[iGlobal]] = d_in[iGlobal];
    }
    iGlobal += blockDim.x;
    if (iGlobal < numElements && d_isValid[iGlobal] > 0) {
        d_out[d_indices[iGlobal]] = d_in[iGlobal];
    }
}

/** @} */ // end compact functions
/** @} */ // end cudpp_kernel

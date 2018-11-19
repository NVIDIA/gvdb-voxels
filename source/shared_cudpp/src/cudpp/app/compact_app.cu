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
  * compact_app.cu
  * 
  * @brief CUDPP application-level compact routines
  */

#include "cuda_util.h"
#include "cudpp_globals.h"
#include "cudpp_util.h"
#include "cudpp.h"
#include "cudpp_plan.h"
#include "cudpp_scan.h"
#include "kernel/compact_kernel.cuh"
#include <cstdlib>
#include <cstdio>
#include <assert.h>

/** \addtogroup cudpp_app 
  * @{
  */

/** @name Compact Functions
 * @{
 */

/** @brief Calculate launch parameters for compactArray().
  *
  * Calculates the block size and number of blocks from the total
  * number of elements and the maximum threads per block. Called by
  * compactArray().
  *
  * The calculation is pretty straightforward - the number of blocks
  * is calculated by dividing the number of input elements by the product
  * of the number of threads in each CTA and the number of elements each thread
  * will process. numThreads and numEltsPerBlock are also simple to 
  * calculate. Please note that in cases where numElements is not an exact
  * multiple of SCAN_ELTS_PER_THREAD * CTA_SIZE we would have threads
  * which do nothing or have a thread which will process less than
  * SCAN_ELTS_PER_THREAD elements.
  *
  *
  * @param[in]  numElements Number of elements to sort
  * @param[out] numThreads Number of threads in each block
  * @param[out] numBlocks  Number of blocks
  * @param[out] numEltsPerBlock Number of elements processed per block
  *
  */
void calculateCompactLaunchParams(const unsigned int numElements,
                 unsigned int       &numThreads, 
                 unsigned int       &numBlocks,
                 unsigned int       &numEltsPerBlock)
{
    numBlocks = 
        max(1, (int)ceil((float)numElements / 
                         ((float)SCAN_ELTS_PER_THREAD * SCAN_CTA_SIZE)));

    if (numBlocks > 1)
    {  
        numThreads = SCAN_CTA_SIZE;
    }    
    else
    {
        numThreads = (unsigned int)ceil((float)numElements / (float)SCAN_ELTS_PER_THREAD);
    }

    numEltsPerBlock = numThreads * SCAN_ELTS_PER_THREAD;
}

/** @brief Compact the non-zero elements of an array.
  * 
  * Given an input array \a d_in, compactArray() outputs a compacted version 
  * which does not have null (zero) elements. Also ouputs the number of non-zero 
  * elements in the compacted array. Called by ::cudppCompactDispatch().
  *
  * The algorithm is straightforward, involving two steps (most of the 
  * complexity is hidden in scan, invoked with cudppScanDispatch() ).
  *
  * -# scanArray() performs a prefix sum on \a d_isValid to compute output 
  *                indices.
  * -# compactData() takes \a d_in and an intermediate array of output indices
  *    as input and writes the values with valid flags in \a d_isValid into 
  *    \a d_out using the output indices.
  *
  * @param[out] d_out         Array of compacted non-null elements
  * @param[out] d_numValidElements Pointer to unsigned int to store number of 
  *                                non-null elements
  * @param[in]  d_in          Input array
  * @param[out] d_isValid     Array of flags, 1 for each non-null element, 0 
  *                           for each null element. Same length as \a d_in
  * @param[in]  numElements   Number of elements in input array
  * @param[in]  plan          Pointer to the plan object used for this compact
  *
  */
template<class T>
void compactArray(T                      *d_out, 
                  size_t                 *d_numValidElements,
                  const T                *d_in, 
                  const unsigned int     *d_isValid,
                  size_t                 numElements,
                  const CUDPPCompactPlan *plan)
{
    unsigned int numThreads = 0;
    unsigned int numBlocks = 0;
    unsigned int numEltsPerBlock = 0;

    // Calculate CUDA launch parameters - number of blocks, number of threads
    // @todo What is numEltsPerBlock doing here?
    calculateCompactLaunchParams((unsigned)numElements, numThreads, numBlocks, numEltsPerBlock);

    // Run prefix sum on isValid array to find the addresses in the compacted
    // output array where each non-null element of d_in will go to
    cudppScanDispatch((void*)plan->m_d_outputIndices, (void*)d_isValid, 
                      numElements, 1, plan->m_scanPlan);

    // For every non-null element in d_in write it to its proper place in the
    // d_out. This is indicated by the corresponding element in isValid array
    if (plan->m_config.options & CUDPP_OPTION_BACKWARD)
        compactData<T, true><<<numBlocks, numThreads>>>(d_out,
                                                        d_numValidElements,
                                                        plan->m_d_outputIndices, 
                                                        d_isValid, d_in, (unsigned)numElements);
    else
        compactData<T, false><<<numBlocks, numThreads>>>(d_out, 
                                                         d_numValidElements,
                                                         plan->m_d_outputIndices, 
                                                         d_isValid, d_in, (unsigned)numElements);
                                                         
    CUDA_CHECK_ERROR("compactArray -- compactData");
}

#ifdef __cplusplus
extern "C" 
{
#endif

/** @brief Allocate intermediate arrays used by cudppCompact().
  *
  * In addition to the internal CUDPPScanPlan contained in CUDPPCompactPlan,
  * CUDPPCompact also needs a temporary device array of output indices, which
  * is allocated by this function.
  *
  * @param plan Pointer to CUDPPCompactPlan object within which intermediate 
  *             storage is allocated.
  */
void allocCompactStorage(CUDPPCompactPlan *plan)
{
    CUDA_SAFE_CALL( cudaMalloc((void**)&plan->m_d_outputIndices, sizeof(unsigned int) * plan->m_numElements) );
}

/** @brief Deallocate intermediate storage used by cudppCompact().
  *
  * Deallocates the output indices array allocated by allocCompactStorage().
  *
  * @param plan Pointer to CUDPPCompactPlan object initialized by allocCompactStorage().
  */
void freeCompactStorage(CUDPPCompactPlan *plan)
{
    CUDA_SAFE_CALL( cudaFree(plan->m_d_outputIndices));
}

/** @brief Dispatch compactArray for the specified datatype.
 *
 * A thin wrapper on top of compactArray which calls compactArray() for the data type
 * specified in \a config. This is the app-level interface to compact used by 
 * cudppCompact().
 *
 * @param[out] d_out         Compacted array of non-zero elements
 * @param[out] d_numValidElements Pointer to an unsigned int to store the 
 *                                 number of non-zero elements
 * @param[in]  d_in          Input array 
 * @param[in]  d_isValid     Array of boolean valid flags with same length as 
 *                           \a d_in
 * @param[in]  numElements   Number of elements to compact
 * @param[in]  plan          Pointer to plan object for this compact
 
 */
void cudppCompactDispatch(void                   *d_out, 
                          size_t                 *d_numValidElements,
                          const void             *d_in, 
                          const unsigned int     *d_isValid,
                          size_t                 numElements,
                          const CUDPPCompactPlan *plan)
{
    switch (plan->m_config.datatype)
    {
    case CUDPP_CHAR:
        compactArray<char>((char*)d_out, d_numValidElements, 
                           (const char*)d_in, d_isValid, numElements, plan);
        break;
    case CUDPP_UCHAR:
        compactArray<unsigned char>((unsigned char*)d_out, d_numValidElements, 
                                    (const unsigned char*)d_in, d_isValid, 
                                    numElements, plan);
        break;
    case CUDPP_INT:
        compactArray<int>((int*)d_out, d_numValidElements, 
                          (const int*)d_in, d_isValid, numElements, plan);
        break;
    case CUDPP_UINT:
        compactArray<unsigned int>((unsigned int*)d_out, d_numValidElements, 
                                   (const unsigned int*)d_in, d_isValid, 
                                   numElements, plan);
        break;
    case CUDPP_FLOAT:
        compactArray<float>((float*)d_out, d_numValidElements, 
                            (const float*)d_in, d_isValid, numElements, plan);
        break;
    case CUDPP_DOUBLE:
        compactArray<double>((double*)d_out, d_numValidElements, 
                            (const double*)d_in, d_isValid, numElements, plan);
        break;
    case CUDPP_LONGLONG:
        compactArray<long long>((long long*)d_out, d_numValidElements, 
                            (const long long*)d_in, d_isValid, numElements, plan);
        break;
    case CUDPP_ULONGLONG:
        compactArray<unsigned long long>((unsigned long long*)d_out, d_numValidElements, 
                                         (const unsigned long long*)d_in, d_isValid, numElements, plan);
        break;
    default:
        break;
    }
}

#ifdef __cplusplus
}
#endif

/** @} */ // end compact functions
/** @} */ // end cudpp_app

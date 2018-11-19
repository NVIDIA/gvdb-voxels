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
 * radixsort_app.cu
 *   
 * @brief CUDPP application-level radix sorting routines
 */

/** @addtogroup cudpp_app 
 * @{
 */

/** @name RadixSort Functions
 * @{
 */
 
#include "cuda_util.h"
#include "cudpp.h"
#include "cudpp_util.h"
#include "cudpp_radixsort.h"
#include "cudpp_scan.h"
#if 0
#include "kernel/radixsort_kernel.cuh"
#include "cudpp_maximal_launch.h"
#include <cstdlib>
#include <cstdio>
#include <assert.h>
#endif

#if 0
typedef unsigned int uint;

/** @brief Perform one step of the radix sort.  Sorts by nbits key bits per step, 
* starting at startbit.
* 
* Uses cudppScanDispatch() for the prefix sum of radix counters.
* 
* @param[in,out] keys Keys to be sorted.
* @param[in,out] values Associated values to be sorted (through keys).
* @param[in] plan Configuration information for RadixSort.
* @param[in] numElements Number of elements in the sort.
**/
template<uint nbits, uint startbit, bool flip, bool unflip>
void radixSortStep(uint *keys, 
                   uint *values, 
                   const CUDPPRadixSortPlan *plan,
                   uint numElements)
{
    const uint eltsPerBlock = SORT_CTA_SIZE * 4;
    const uint eltsPerBlock2 = SORT_CTA_SIZE * 2;

    bool fullBlocks = ((numElements % eltsPerBlock) == 0);
    uint numBlocks = (fullBlocks) ? 
        (numElements / eltsPerBlock) : 
    (numElements / eltsPerBlock + 1);
    uint numBlocks2 = ((numElements % eltsPerBlock2) == 0) ?
        (numElements / eltsPerBlock2) : 
    (numElements / eltsPerBlock2 + 1);

    bool loop = numBlocks > 65535;

    uint blocks = loop ? 65535 : numBlocks;
    uint blocksFind = loop ? 65535 : numBlocks2;
    uint blocksReorder = loop ? 65535 : numBlocks2;

    uint threshold = fullBlocks ? plan->m_persistentCTAThresholdFullBlocks[0] : plan->m_persistentCTAThreshold[0];

    bool persist = plan->m_bUsePersistentCTAs && (numElements >= threshold);

    if (persist)
    {
        loop = (numElements > 262144) || (numElements >= 32768 && numElements < 65536);
        
        blocks = numBlocks;
        blocksFind = numBlocks2;
        blocksReorder = numBlocks2;

        // Run an empty kernel -- this seems to reset some of the CTA scheduling hardware
        // on GT200, resulting in better scheduling and lower run times
        if (startbit > 0)
        {
            emptyKernel<<<plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_EK], SORT_CTA_SIZE>>>();
        }
    }

    if (fullBlocks)
    {
        if (loop)
        {
            if (persist) 
            {
                blocks = flip? plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RSB_4_0_T_T_T] : 
                               plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RSB_4_0_T_F_T];
            }

            radixSortBlocks<nbits, startbit, true, flip, true>
                <<<blocks, SORT_CTA_SIZE, 4 * SORT_CTA_SIZE * sizeof(uint)>>>
                ((uint4*)plan->m_tempKeys, (uint4*)plan->m_tempValues, (uint4*)keys, (uint4*)values, numElements, numBlocks);
        }
        else
        {
            radixSortBlocks<nbits, startbit, true, flip, false>
                <<<blocks, SORT_CTA_SIZE, 4 * SORT_CTA_SIZE * sizeof(uint)>>>
                ((uint4*)plan->m_tempKeys, (uint4*)plan->m_tempValues, (uint4*)keys, (uint4*)values, numElements, numBlocks);
        }
    }
    else
    {
        if (loop)
        {
            if (persist) 
            {
                blocks = flip ? plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RSB_4_0_F_T_T] : 
                                plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RSB_4_0_F_F_T];
            }

            radixSortBlocks<nbits, startbit, false, flip, true>
                <<<blocks, SORT_CTA_SIZE, 4 * SORT_CTA_SIZE * sizeof(uint)>>>
                ((uint4*)plan->m_tempKeys, (uint4*)plan->m_tempValues, (uint4*)keys, (uint4*)values, numElements, numBlocks);
        }
        else
        {
            radixSortBlocks<nbits, startbit, false, flip, false>
                <<<blocks, SORT_CTA_SIZE, 4 * SORT_CTA_SIZE * sizeof(uint)>>>
                ((uint4*)plan->m_tempKeys, (uint4*)plan->m_tempValues, (uint4*)keys, (uint4*)values, numElements, numBlocks);
        }
    }

    CUDA_CHECK_ERROR("radixSortBlocks");

    if (fullBlocks)
    {
        if (loop)
        {
            if (persist) 
            {
                blocksFind = plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_FRO_0_T_T];
            }
            findRadixOffsets<startbit, true, true>
                <<<blocksFind, SORT_CTA_SIZE, 3 * SORT_CTA_SIZE * sizeof(uint)>>>
                ((uint2*)plan->m_tempKeys, plan->m_counters, plan->m_blockOffsets, numElements, numBlocks2);
        }
        else
        {
            findRadixOffsets<startbit, true, false>
                <<<blocksFind, SORT_CTA_SIZE, 3 * SORT_CTA_SIZE * sizeof(uint)>>>
                ((uint2*)plan->m_tempKeys, plan->m_counters, plan->m_blockOffsets, numElements, numBlocks2);
        }
    }
    else
    {
        if (loop)
        {
            if (persist) 
            {
                blocksFind = plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_FRO_0_F_T];
            }
            findRadixOffsets<startbit, false, true>
                <<<blocksFind, SORT_CTA_SIZE, 3 * SORT_CTA_SIZE * sizeof(uint)>>>
                ((uint2*)plan->m_tempKeys, plan->m_counters, plan->m_blockOffsets, numElements, numBlocks2);
        }
        else
        {
            findRadixOffsets<startbit, false, false>
                <<<blocksFind, SORT_CTA_SIZE, 3 * SORT_CTA_SIZE * sizeof(uint)>>>
                ((uint2*)plan->m_tempKeys, plan->m_counters, plan->m_blockOffsets, numElements, numBlocks2);
        }
    }

    CUDA_CHECK_ERROR("findRadixOffsets");

    cudppScanDispatch(plan->m_countersSum, plan->m_counters, 16*numBlocks2, 1, plan->m_scanPlan);

    if (fullBlocks)
    {
        if (plan->m_bManualCoalesce)
        {
            if (loop)
            {
                if (persist) 
                {
                    blocksReorder = unflip ? plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RD_0_T_T_T_T] :
                                             plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RD_0_T_T_F_T];
                }
                reorderData<startbit, true, true, unflip, true>
                    <<<blocksReorder, SORT_CTA_SIZE>>>
                    (keys, values, (uint2*)plan->m_tempKeys, (uint2*)plan->m_tempValues, 
                    plan->m_blockOffsets, plan->m_countersSum, plan->m_counters, numElements, numBlocks2);
            }
            else
            {
                reorderData<startbit, true, true, unflip, false>
                    <<<blocksReorder, SORT_CTA_SIZE>>>
                    (keys, values, (uint2*)plan->m_tempKeys, (uint2*)plan->m_tempValues, 
                    plan->m_blockOffsets, plan->m_countersSum, plan->m_counters, numElements, numBlocks2);
            }
        }
        else
        {
            if (loop)
            {
                if (persist) 
                {
                    blocksReorder = unflip ? plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RD_0_T_F_T_T] :
                                             plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RD_0_T_F_F_T];
                }
                reorderData<startbit, true, false, unflip, true>
                    <<<blocksReorder, SORT_CTA_SIZE>>>
                    (keys, values, (uint2*)plan->m_tempKeys, (uint2*)plan->m_tempValues, 
                    plan->m_blockOffsets, plan->m_countersSum, plan->m_counters, numElements, numBlocks2);
            }
            else
            {
                reorderData<startbit, true, false, unflip, false>
                    <<<blocksReorder, SORT_CTA_SIZE>>>
                    (keys, values, (uint2*)plan->m_tempKeys, (uint2*)plan->m_tempValues, 
                    plan->m_blockOffsets, plan->m_countersSum, plan->m_counters, numElements, numBlocks2);
            }
        }
    }
    else
    {
        if (plan->m_bManualCoalesce)
        {
            if (loop)
            {
                if (persist) 
                {
                    blocksReorder = unflip ? 
                        plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RD_0_F_T_T_T] :
                        plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RD_0_F_T_F_T];
                }
                reorderData<startbit, false, true, unflip, true>
                    <<<blocksReorder, SORT_CTA_SIZE>>>
                    (keys, values, (uint2*)plan->m_tempKeys, (uint2*)plan->m_tempValues, 
                    plan->m_blockOffsets, plan->m_countersSum, plan->m_counters, numElements, numBlocks2);
            }
            else
            {
                reorderData<startbit, false, true, unflip, false>
                    <<<blocksReorder, SORT_CTA_SIZE>>>
                    (keys, values, (uint2*)plan->m_tempKeys, (uint2*)plan->m_tempValues, 
                    plan->m_blockOffsets, plan->m_countersSum, plan->m_counters, numElements, numBlocks2);
            }
        }
        else
        {
            if (loop)
            {
                if (persist) 
                {
                    blocksReorder = unflip ?
                        plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RD_0_F_F_T_T] :
                        plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RD_0_F_F_F_T];
                }
                reorderData<startbit, false, false, unflip, true>
                    <<<blocksReorder, SORT_CTA_SIZE>>>
                    (keys, values, (uint2*)plan->m_tempKeys, (uint2*)plan->m_tempValues, 
                    plan->m_blockOffsets, plan->m_countersSum, plan->m_counters, numElements, numBlocks2);
            }
            else
            {
                reorderData<startbit, false, false, unflip, false>
                    <<<blocksReorder, SORT_CTA_SIZE>>>
                    (keys, values, (uint2*)plan->m_tempKeys, (uint2*)plan->m_tempValues, 
                    plan->m_blockOffsets, plan->m_countersSum, plan->m_counters, numElements, numBlocks2);
            }
        }
    }

    CUDA_CHECK_ERROR("radixSortStep");
}

/**
 * @brief Single-block optimization for sorts of fewer than 4 * CTA_SIZE elements
 * 
 * @param[in,out] keys  Keys to be sorted.
 * @param[in,out] values Associated values to be sorted (through keys).
 * @param numElements Number of elements in the sort.
**/
template <bool flip>
void radixSortSingleBlock(uint *keys, 
                          uint *values, 
                          uint numElements)
{
    bool fullBlocks = (numElements % (SORT_CTA_SIZE * 4) == 0);
    if (fullBlocks)
    {
        radixSortBlocks<32, 0, true, flip, false>
            <<<1, SORT_CTA_SIZE, 4 * SORT_CTA_SIZE * sizeof(uint)>>>
                ((uint4*)keys, (uint4*)values, 
                 (uint4*)keys, (uint4*)values, 
                 numElements, 0);
    }
    else
    {
        radixSortBlocks<32, 0, false, flip, false>
            <<<1, SORT_CTA_SIZE, 4 * SORT_CTA_SIZE * sizeof(uint)>>>
                ((uint4*)keys, (uint4*)values, 
                 (uint4*)keys, (uint4*)values, 
                 numElements, 0);
    }

    if (flip) unflipFloats<<<1, SORT_CTA_SIZE>>>(keys, numElements);

    CUDA_CHECK_ERROR("radixSortSingleBlock");
}

/**
 * @brief Main radix sort function
 * 
 * Main radix sort function.  Sorts in place in the keys and values arrays,
 * but uses the other device arrays as temporary storage.  All pointer 
 * parameters are device pointers.  Uses cudppScan() for the prefix sum of 
 * radix counters.
 * 
 * While the interface supports forward and backward sorts (via \a plan),
 * only forward is currently implemented.
 * 
 * @param[in,out] keys Keys to be sorted.
 * @param[in,out] values Associated values to be sorted (through keys).
 * @param[in] plan Configuration information for RadixSort.
 * @param[in] numElements Number of elements in the sort.
 * @param[in] flipBits Is set true if key datatype is a float 
 *                 (neg. numbers) for special float sorting operations.
 * @param[in] keyBits Number of interesting bits in the key
 **/
void radixSort(uint *keys,                         
               uint* values,               
               const CUDPPRadixSortPlan *plan,               
               size_t numElements,
               bool flipBits,
               int keyBits)
{
    if(numElements <= WARP_SIZE)
    {
        if (flipBits)
            radixSortSingleWarp<true><<<1, numElements>>>
                (keys, values, numElements);
        else
            radixSortSingleWarp<false><<<1, numElements>>>
                (keys, values, numElements);

        CUDA_CHECK_ERROR("radixSortSingleWarp");        
        return;
    }
    
    if(numElements <= SORT_CTA_SIZE * 4)
    {
        if (flipBits)
            radixSortSingleBlock<true>(keys, values, numElements);
        else
            radixSortSingleBlock<false>(keys, values, numElements);
        return;
    }
        
    // flip float bits on the first pass, unflip on the last pass    
    if (flipBits) 
    {               
        radixSortStep<4,  0, true, false>
            (keys, values, plan, numElements);            
    }
    else
    {     
        radixSortStep<4,  0, false, false>
            (keys, values, plan, numElements);           
    }

    if (keyBits > 4)
    {                   
        radixSortStep<4,  4, false, false>
            (keys, values, plan, numElements);            
    }
    if (keyBits > 8)
    {                                   
        radixSortStep<4,  8, false, false>
            (keys, values, plan, numElements);            
    }
    if (keyBits > 12)
    {                   
        radixSortStep<4, 12, false, false>
            (keys, values, plan, numElements);            
    }
    if (keyBits > 16)
    {                   
        radixSortStep<4, 16, false, false>
            (keys, values, plan, numElements);            
    }
    if (keyBits > 20)
    {                   
        radixSortStep<4, 20, false, false>
            (keys, values, plan, numElements);            
    }
    if (keyBits > 24)
    {                   
        radixSortStep<4, 24, false, false>
            (keys, values, plan, numElements);         
    }
    if (keyBits > 28)
    {
        if (flipBits) // last pass
        {                       
            radixSortStep<4, 28, false, true>
                (keys, values, plan, numElements);
        }
        else
        {                       
            radixSortStep<4, 28, false, false>
                (keys, values, plan, numElements);            
        }
    }
}

/**
 * @brief Wrapper to call main radix sort function. For float configuration.
 * 
 * Calls the main radix sort function. For float configuration.
 * 
 * @param[in,out] keys Keys to be sorted.
 * @param[in,out] values Associated values to be sorted (through keys).
 * @param[in] plan Configuration information for RadixSort.
 * @param[in] numElements Number of elements in the sort.
 * @param[in] negativeKeys Is set true if key datatype has neg. numbers.
 * @param[in] keyBits Number of interesting bits in the key
 **/
void radixSortFloatKeys(float* keys, 
                        uint* values, 
                        const CUDPPRadixSortPlan *plan,
                        size_t numElements,            
                        bool  negativeKeys,
                        int keyBits)
{
   
    radixSort((uint*)keys, (uint*)values, plan, 
              numElements, negativeKeys, keyBits);
}

/** @brief Perform one step of the radix sort.  Sorts by nbits key bits per step, 
 * starting at startbit.
 * 
 * @param[in,out] keys  Keys to be sorted.
 * @param[in] plan Configuration information for RadixSort.
 * @param[in] numElements Number of elements in the sort. 
**/
template<uint nbits, uint startbit, bool flip, bool unflip>
void radixSortStepKeysOnly(uint *keys, 
                           const CUDPPRadixSortPlan *plan,
                           uint numElements)
{
    const uint eltsPerBlock = SORT_CTA_SIZE * 4;
    const uint eltsPerBlock2 = SORT_CTA_SIZE * 2;

    bool fullBlocks = ((numElements % eltsPerBlock) == 0);
    uint numBlocks = (fullBlocks) ? 
        (numElements / eltsPerBlock) : 
    (numElements / eltsPerBlock + 1);
    uint numBlocks2 = ((numElements % eltsPerBlock2) == 0) ?
        (numElements / eltsPerBlock2) : 
    (numElements / eltsPerBlock2 + 1);

    bool loop = numBlocks > 65535;
    
    uint blocks = loop ? 65535 : numBlocks;
    uint blocksFind = loop ? 65535 : numBlocks2;
    uint blocksReorder = loop ? 65535 : numBlocks2;

    uint threshold = fullBlocks ? plan->m_persistentCTAThresholdFullBlocks[1] : plan->m_persistentCTAThreshold[1];

    bool persist = plan->m_bUsePersistentCTAs && (numElements >= threshold);

    if (persist)
    {
        loop = (numElements > 262144) || (numElements >= 32768 && numElements < 65536);
        
        blocks = numBlocks;
        blocksFind = numBlocks2;
        blocksReorder = numBlocks2;
    }

    if (fullBlocks)
    {
        if (loop)
        {
            if (persist) 
            {
                blocks = flip ? plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RSBKO_4_0_T_T_T] : 
                                plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RSBKO_4_0_T_F_T];
            }

            radixSortBlocksKeysOnly<nbits, startbit, true, flip, true>
                <<<blocks, SORT_CTA_SIZE, 4 * SORT_CTA_SIZE * sizeof(uint)>>>
                ((uint4*)plan->m_tempKeys, (uint4*)keys, numElements, numBlocks);
        }
        else
            radixSortBlocksKeysOnly<nbits, startbit, true, flip, false>
                <<<blocks, SORT_CTA_SIZE, 4 * SORT_CTA_SIZE * sizeof(uint)>>>
                ((uint4*)plan->m_tempKeys, (uint4*)keys, numElements, numBlocks);
    }
    else
    {
        if (loop)
        {
            if (persist) 
            {
                blocks = flip ? plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RSBKO_4_0_F_T_T] : 
                                plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RSBKO_4_0_F_F_T];
            }

            radixSortBlocksKeysOnly<nbits, startbit, false, flip, true>
                <<<blocks, SORT_CTA_SIZE, 4 * SORT_CTA_SIZE * sizeof(uint)>>>
                ((uint4*)plan->m_tempKeys, (uint4*)keys, numElements, numBlocks);
        }
        else
            radixSortBlocksKeysOnly<nbits, startbit, false, flip, false>
                <<<blocks, SORT_CTA_SIZE, 4 * SORT_CTA_SIZE * sizeof(uint)>>>
                ((uint4*)plan->m_tempKeys, (uint4*)keys, numElements, numBlocks);

    }

    if (fullBlocks)
    {
        if (loop)
        {
            if (persist) 
            {
                blocksFind = plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_FRO_0_T_T];
            }
            findRadixOffsets<startbit, true, true>
                <<<blocksFind, SORT_CTA_SIZE, 3 * SORT_CTA_SIZE * sizeof(uint)>>>
                ((uint2*)plan->m_tempKeys, plan->m_counters, plan->m_blockOffsets, numElements, numBlocks2);
        }
        else
            findRadixOffsets<startbit, true, false>
                <<<blocksFind, SORT_CTA_SIZE, 3 * SORT_CTA_SIZE * sizeof(uint)>>>
                ((uint2*)plan->m_tempKeys, plan->m_counters, plan->m_blockOffsets, numElements, numBlocks2);
    }
    else
    {
        if (loop)
        {
            if (persist) 
            {
                blocksFind = plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_FRO_0_F_T];
            }
            findRadixOffsets<startbit, false, true>
                <<<blocksFind, SORT_CTA_SIZE, 3 * SORT_CTA_SIZE * sizeof(uint)>>>
                ((uint2*)plan->m_tempKeys, plan->m_counters, plan->m_blockOffsets, numElements, numBlocks2);
        }
        else
            findRadixOffsets<startbit, false, false>
                <<<blocksFind, SORT_CTA_SIZE, 3 * SORT_CTA_SIZE * sizeof(uint)>>>
                ((uint2*)plan->m_tempKeys, plan->m_counters, plan->m_blockOffsets, numElements, numBlocks2);

    }

    cudppScanDispatch(plan->m_countersSum, plan->m_counters, 16*numBlocks2, 1, plan->m_scanPlan);

    if (fullBlocks)
    {
        if (plan->m_bManualCoalesce)
        {
            if (loop)
            {
                if (persist) 
                {
                    blocksReorder = unflip ? 
                        plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RDKO_0_T_T_T_T] : 
                        plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RDKO_0_T_T_F_T];
                }
                reorderDataKeysOnly<startbit, true, true, unflip, true>
                    <<<blocksReorder, SORT_CTA_SIZE>>>
                    (keys, (uint2*)plan->m_tempKeys, plan->m_blockOffsets, plan->m_countersSum, plan->m_counters, 
                    numElements, numBlocks2);
            }
            else
                reorderDataKeysOnly<startbit, true, true, unflip, false>
                    <<<blocksReorder, SORT_CTA_SIZE>>>
                    (keys, (uint2*)plan->m_tempKeys, plan->m_blockOffsets, plan->m_countersSum, plan->m_counters, 
                     numElements, numBlocks2);
        }
        else
        {
            if (loop)
            {
                if (persist) 
                {
                    blocksReorder = unflip ?
                        plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RDKO_0_T_F_T_T] :
                        plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RDKO_0_T_F_F_T];
                }
                reorderDataKeysOnly<startbit, true, false, unflip, true>
                    <<<blocksReorder, SORT_CTA_SIZE>>>
                    (keys, (uint2*)plan->m_tempKeys, plan->m_blockOffsets, plan->m_countersSum, plan->m_counters, 
                    numElements, numBlocks2);
            }
            else
                reorderDataKeysOnly<startbit, true, false, unflip, false>
                    <<<blocksReorder, SORT_CTA_SIZE>>>
                    (keys, (uint2*)plan->m_tempKeys, plan->m_blockOffsets, plan->m_countersSum, plan->m_counters, 
                     numElements, numBlocks2);
        }
    }
    else
    {
        if (plan->m_bManualCoalesce)
        {
            if (loop)
            {
                if (persist) 
                {
                    blocksReorder = unflip ? 
                        plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RDKO_0_F_T_T_T] :
                        plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RDKO_0_F_T_F_T];
                }
                reorderDataKeysOnly<startbit, false, true, unflip, true>
                    <<<blocksReorder, SORT_CTA_SIZE>>>
                    (keys, (uint2*)plan->m_tempKeys, plan->m_blockOffsets, plan->m_countersSum, plan->m_counters, 
                    numElements, numBlocks2);
            }
            else
                reorderDataKeysOnly<startbit, false, true, unflip, false>
                <<<blocksReorder, SORT_CTA_SIZE>>>
                (keys, (uint2*)plan->m_tempKeys, plan->m_blockOffsets, plan->m_countersSum, plan->m_counters, 
                numElements, numBlocks2);
        }
        else
        {
            if (loop)
            {
                if (persist) 
                {
                    blocksReorder = unflip ?
                        plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RDKO_0_F_F_T_T] :
                        plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RDKO_0_F_F_F_T];
                }
                reorderDataKeysOnly<startbit, false, false, unflip, true>
                    <<<blocksReorder, SORT_CTA_SIZE>>>
                    (keys, (uint2*)plan->m_tempKeys, plan->m_blockOffsets, plan->m_countersSum, plan->m_counters, 
                    numElements, numBlocks2);
            }
            else
                reorderDataKeysOnly<startbit, false, false, unflip, false>
                <<<blocksReorder, SORT_CTA_SIZE>>>
                (keys, (uint2*)plan->m_tempKeys, plan->m_blockOffsets, plan->m_countersSum, plan->m_counters, 
                numElements, numBlocks2);
        }
    }

    CUDA_CHECK_ERROR("radixSortStepKeysOnly");
}

/**
 * @brief Optimization for sorts of fewer than 4 * CTA_SIZE elements (keys only).
 * 
 * @param[in,out] keys Keys to be sorted.
 * @param numElements Number of elements in the sort.
**/
template <bool flip>
void radixSortSingleBlockKeysOnly(uint *keys, 
                                  uint numElements)
{
    bool fullBlocks = (numElements % (SORT_CTA_SIZE * 4) == 0);
    if (fullBlocks)
    {
        radixSortBlocksKeysOnly<32, 0, true, flip, false>
            <<<1, SORT_CTA_SIZE, 4 * SORT_CTA_SIZE * sizeof(uint)>>>
            ((uint4*)keys, (uint4*)keys, numElements, 1 );
    }
    else
    {
        radixSortBlocksKeysOnly<32, 0, false, flip, false>
            <<<1, SORT_CTA_SIZE, 4 * SORT_CTA_SIZE * sizeof(uint)>>>
            ((uint4*)keys, (uint4*)keys, numElements, 1 );
    }

    if (flip)
        unflipFloats<<<1, SORT_CTA_SIZE>>>(keys, numElements);


    CUDA_CHECK_ERROR("radixSortSingleBlock");
}

/** 
 * @brief Main radix sort function. For keys only configuration.
 *
 * Main radix sort function.  Sorts in place in the keys array,
 * but uses the other device arrays as temporary storage.  All pointer 
 * parameters are device pointers.  Uses scan for the prefix sum of
 * radix counters.
 * 
 * @param[in,out] keys Keys to be sorted.
 * @param[in] plan Configuration information for RadixSort.
 * @param[in] flipBits Is set true if key datatype is a float (neg. numbers) 
 *        for special float sorting operations.
 * @param[in] numElements Number of elements in the sort.
 * @param[in] keyBits Number of interesting bits in the key
**/
void radixSortKeysOnly(uint *keys,
                       const CUDPPRadixSortPlan *plan, 
                       size_t numElements,
                       bool flipBits, 
                       int keyBits)
{

    if(numElements <= WARP_SIZE)
    {
        if (flipBits)
            radixSortSingleWarpKeysOnly<true><<<1, numElements>>>(keys, numElements);
        else
            radixSortSingleWarpKeysOnly<false><<<1, numElements>>>(keys, numElements);
        return;
    }
    if(numElements <= SORT_CTA_SIZE * 4)
    {
        if (flipBits)
            radixSortSingleBlockKeysOnly<true>(keys, numElements);
        else
            radixSortSingleBlockKeysOnly<false>(keys, numElements);
        return;
    }

    // flip float bits on the first pass, unflip on the last pass
    if (flipBits) 
    {
        radixSortStepKeysOnly<4,  0, true, false>(keys, plan, numElements);
    }
    else
    {
        radixSortStepKeysOnly<4,  0, false, false>(keys, plan, numElements);
    }

    if (keyBits > 4)
    {
        radixSortStepKeysOnly<4,  4, false, false>(keys, plan, numElements);
    }
    if (keyBits > 8)
    {
        radixSortStepKeysOnly<4,  8, false, false>(keys, plan, numElements);
    }
    if (keyBits > 12)
    {
        radixSortStepKeysOnly<4, 12, false, false>(keys, plan, numElements);
    }
    if (keyBits > 16)
    {
        radixSortStepKeysOnly<4, 16, false, false>(keys, plan, numElements);
    }
    if (keyBits > 20)
    {
        radixSortStepKeysOnly<4, 20, false, false>(keys, plan, numElements);
    }
    if (keyBits > 24)
    {
       radixSortStepKeysOnly<4, 24, false, false>(keys, plan, numElements);
    }
    if (keyBits > 28)
    {
        if (flipBits) // last pass
        {
            radixSortStepKeysOnly<4, 28, false, true>(keys, plan, numElements);
        }
        else
        {
            radixSortStepKeysOnly<4, 28, false, false>(keys, plan, numElements);
        }
    }
}

/**
 * @brief Wrapper to call main radix sort function. For floats and keys only.
 *
 * Calls the radixSortKeysOnly function setting parameters for floats.
 * 
 * @param[in,out] keys Keys to be sorted.
 * @param[in] plan Configuration information for RadixSort.
 * @param[in] negativeKeys Is set true if key flipBits is to be true in 
 *                     radixSortKeysOnly().
 * @param[in] numElements Number of elements in the sort.
 * @param[in] keyBits Number of interesting bits in the key
**/
void radixSortFloatKeysOnly(float *keys, 
                            const CUDPPRadixSortPlan *plan,                        
                            size_t numElements,
                            bool  negativeKeys,
                            int keyBits)
{
    radixSortKeysOnly((uint*)keys, plan, numElements, negativeKeys, keyBits);
}

void initDeviceParameters(CUDPPRadixSortPlan *plan)
{
    int deviceID = -1;
    if (cudaSuccess == cudaGetDevice(&deviceID))
    {
        cudaDeviceProp devprop;
        plan->m_planManager->getDeviceProps(devprop);

        int smVersion = devprop.major * 10 + devprop.minor;

        // sm_12 and later devices don't need help with coalesce in reorderData kernel
        plan->m_bManualCoalesce = (smVersion < 12);

        // sm_20 and later devices are better off not using persistent CTAs
        plan->m_bUsePersistentCTAs = (smVersion < 20);

        if (plan->m_bUsePersistentCTAs)
        {
            // The following is only true on pre-sm_20 devices (pre-Fermi):
            // Empirically we have found that for some (usually larger) sort
            // sizes it is better to use exactly as many "persistent" CTAs 
            // as can fill the GPU, which loop over the "blocks" of work. For smaller 
            // arrays it is better to use the typical CUDA approach of launching one CTA
            // per block of work.
            // 0-element of these two-element arrays is for key-value sorts
            // 1-element is for key-only sorts
            plan->m_persistentCTAThreshold[0] = plan->m_bManualCoalesce ? 16777216 : 524288;
            plan->m_persistentCTAThresholdFullBlocks[0] = plan->m_bManualCoalesce ? 2097152: 524288;
            plan->m_persistentCTAThreshold[1] = plan->m_bManualCoalesce ? 16777216 : 8388608;
            plan->m_persistentCTAThresholdFullBlocks[1] = plan->m_bManualCoalesce ? 2097152: 0;

            // create a map of function pointers to register counts for more accurate occupancy calculation
            // Must pass in the dynamic shared memory used by each kernel, since the runtime doesn't know it
            // Note we only insert the "loop" version of the kernels (the one with the last template param = true)
            // Because those are the only ones that require persistent CTAs that maximally fill the device.
            
            plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RSB_4_0_F_F_T] = maxBlocks(radixSortBlocks<4, 0, false, false, true>, 4 * SORT_CTA_SIZE * sizeof(uint), SORT_CTA_SIZE);
            plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RSB_4_0_F_T_T] = maxBlocks(radixSortBlocks<4, 0, false, true,  true>, 4 * SORT_CTA_SIZE * sizeof(uint), SORT_CTA_SIZE);
            plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RSB_4_0_T_F_T] = maxBlocks(radixSortBlocks<4, 0, true,  false, true>, 4 * SORT_CTA_SIZE * sizeof(uint), SORT_CTA_SIZE);
            plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RSB_4_0_T_T_T] = maxBlocks(radixSortBlocks<4, 0, true,  true,  true>, 4 * SORT_CTA_SIZE * sizeof(uint), SORT_CTA_SIZE);
            
            plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RSBKO_4_0_F_F_T] = maxBlocks(radixSortBlocksKeysOnly<4, 0, false, false, true>, 4 * SORT_CTA_SIZE * sizeof(uint), SORT_CTA_SIZE);
            plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RSBKO_4_0_F_T_T] = maxBlocks(radixSortBlocksKeysOnly<4, 0, false, true, true>,  4 * SORT_CTA_SIZE * sizeof(uint), SORT_CTA_SIZE);
            plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RSBKO_4_0_T_F_T] = maxBlocks(radixSortBlocksKeysOnly<4, 0, true, false, true>,  4 * SORT_CTA_SIZE * sizeof(uint), SORT_CTA_SIZE);
            plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RSBKO_4_0_T_T_T] = maxBlocks(radixSortBlocksKeysOnly<4, 0, true, true, true>,   4 * SORT_CTA_SIZE * sizeof(uint), SORT_CTA_SIZE);

            plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_FRO_0_F_T] = maxBlocks(findRadixOffsets<0, false, true>, 3 * SORT_CTA_SIZE * sizeof(uint), SORT_CTA_SIZE);
            plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_FRO_0_T_T] = maxBlocks(findRadixOffsets<0, true, true>,  3 * SORT_CTA_SIZE * sizeof(uint), SORT_CTA_SIZE);

            plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RD_0_F_F_F_T] = maxBlocks(reorderData<0, false, false, false, true>, 0, SORT_CTA_SIZE);
            plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RD_0_F_F_T_T] = maxBlocks(reorderData<0, false, false, true, true>,  0, SORT_CTA_SIZE);
            plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RD_0_F_T_F_T] = maxBlocks(reorderData<0, false, true, false, true>,  0, SORT_CTA_SIZE);
            plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RD_0_F_T_T_T] = maxBlocks(reorderData<0, false, true, true, true>,   0, SORT_CTA_SIZE);
            plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RD_0_T_F_F_T] = maxBlocks(reorderData<0, true, false, false, true>,  0, SORT_CTA_SIZE);
            plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RD_0_T_F_T_T] = maxBlocks(reorderData<0, true, false, true, true>,   0, SORT_CTA_SIZE);
            plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RD_0_T_T_F_T] = maxBlocks(reorderData<0, true, true, false, true>,   0, SORT_CTA_SIZE);
            plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RD_0_T_T_T_T] = maxBlocks(reorderData<0, true, true, true, true>,    0, SORT_CTA_SIZE);

            plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RDKO_0_F_F_F_T] = maxBlocks(reorderDataKeysOnly<0, false, false, false, true>, 0, SORT_CTA_SIZE);
            plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RDKO_0_F_F_T_T] = maxBlocks(reorderDataKeysOnly<0, false, false, true, true>,  0, SORT_CTA_SIZE);
            plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RDKO_0_F_T_F_T] = maxBlocks(reorderDataKeysOnly<0, false, true, false, true>,  0, SORT_CTA_SIZE);
            plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RDKO_0_F_T_T_T] = maxBlocks(reorderDataKeysOnly<0, false, true, true, true>,   0, SORT_CTA_SIZE);
            plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RDKO_0_T_F_F_T] = maxBlocks(reorderDataKeysOnly<0, true, false, false, true>,  0, SORT_CTA_SIZE);
            plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RDKO_0_T_F_T_T] = maxBlocks(reorderDataKeysOnly<0, true, false, true, true>,   0, SORT_CTA_SIZE);
            plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RDKO_0_T_T_F_T] = maxBlocks(reorderDataKeysOnly<0, true, true, false, true>,   0, SORT_CTA_SIZE);
            plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_RDKO_0_T_T_T_T] = maxBlocks(reorderDataKeysOnly<0, true, true, true, true>,    0, SORT_CTA_SIZE);
                   
            plan->m_numCTAs[CUDPPRadixSortPlan::KERNEL_EK] = maxBlocks(emptyKernel, 0, SORT_CTA_SIZE);
        }
    }
}
#endif
/**
 * @brief From the programmer-specified sort configuration, 
 *        creates internal memory for performing the sort.
 * 
 * @param[in] plan Pointer to CUDPPRadixSortPlan object
**/
void allocRadixSortStorage(CUDPPRadixSortPlan *plan)
{               
#if 0
    unsigned int numElements = plan->m_numElements;

    unsigned int numBlocks = 
        ((numElements % (SORT_CTA_SIZE * 4)) == 0) ? 
            (numElements / (SORT_CTA_SIZE * 4)) : 
            (numElements / (SORT_CTA_SIZE * 4) + 1);
                        
    switch(plan->m_config.datatype)
    {
    case CUDPP_UINT:
        CUDA_SAFE_CALL(cudaMalloc((void **)&plan->m_tempKeys, 
                                  numElements * sizeof(unsigned int)));

        if (!plan->m_bKeysOnly)
            CUDA_SAFE_CALL(cudaMalloc((void **)&plan->m_tempValues, 
                           numElements * sizeof(unsigned int)));

        CUDA_SAFE_CALL(cudaMalloc((void **)&plan->m_counters, 
                       WARP_SIZE * numBlocks * sizeof(unsigned int)));

        CUDA_SAFE_CALL(cudaMalloc((void **)&plan->m_countersSum,
                       WARP_SIZE * numBlocks * sizeof(unsigned int)));

        CUDA_SAFE_CALL(cudaMalloc((void **)&plan->m_blockOffsets, 
                       WARP_SIZE * numBlocks * sizeof(unsigned int)));
    break;

    case CUDPP_FLOAT:
        CUDA_SAFE_CALL(cudaMalloc((void **)&plan->m_tempKeys,
                                   numElements * sizeof(float)));

        if (!plan->m_bKeysOnly)
            CUDA_SAFE_CALL(cudaMalloc((void **)&plan->m_tempValues,
                           numElements * sizeof(float)));

        CUDA_SAFE_CALL(cudaMalloc((void **)&plan->m_counters,
                       WARP_SIZE * numBlocks * sizeof(float)));

        CUDA_SAFE_CALL(cudaMalloc((void **)&plan->m_countersSum,
                       WARP_SIZE * numBlocks * sizeof(float)));

        CUDA_SAFE_CALL(cudaMalloc((void **)&plan->m_blockOffsets,
                       WARP_SIZE * numBlocks * sizeof(float)));     
    break;
    }
        
    initDeviceParameters(plan);
#endif
}

/** @brief Deallocates intermediate memory from allocRadixSortStorage.
 *
 *
 * @param[in] plan Pointer to CUDPPRadixSortPlan object
**/
void freeRadixSortStorage(CUDPPRadixSortPlan* plan)
{
#if 0
    CUDA_SAFE_CALL( cudaFree(plan->m_tempKeys));
    CUDA_SAFE_CALL( cudaFree(plan->m_tempValues));
    CUDA_SAFE_CALL( cudaFree(plan->m_counters));
    CUDA_SAFE_CALL( cudaFree(plan->m_countersSum));
    CUDA_SAFE_CALL( cudaFree(plan->m_blockOffsets));
#endif
}

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/reverse.h>
template<typename T>
void runSort(T *pkeys, 
             unsigned int *pvals,
             size_t numElements, 
             const CUDPPRadixSortPlan *plan)
{
    thrust::device_ptr<T> keys((T*)pkeys);
    thrust::device_ptr<unsigned int> vals((unsigned int*)pvals);

    if (plan->m_bKeysOnly)
        thrust::sort(keys, keys + numElements);
    else
        thrust::sort_by_key(keys, keys + numElements, vals);
            
    if (plan->m_bBackward)
    {
        thrust::reverse(keys, keys + numElements);
        if (!plan->m_bKeysOnly)
            thrust::reverse(vals, vals + numElements);
    }
    
    CUDA_CHECK_ERROR("cudppRadixSortDispatch");
}

/** @brief Dispatch function to perform a sort on an array with 
 * a specified configuration.
 *
 * This is the dispatch routine which calls radixSort...() with 
 * appropriate template parameters and arguments as specified by 
 * the plan.
 * @param[in,out] keys Keys to be sorted.
 * @param[in,out] values Associated values to be sorted (through keys).
 * @param[in] numElements Number of elements in the sort.
 * @param[in] plan Configuration information for RadixSort.
**/

void cudppRadixSortDispatch(void  *keys,
                            void  *values,
                            size_t numElements,
                            const CUDPPRadixSortPlan *plan)
{
    switch(plan->m_config.datatype)
    {
    case CUDPP_CHAR:
        runSort<char>((char*)keys, (unsigned int*)values, numElements, plan);
        break;
    case CUDPP_UCHAR:
        runSort<unsigned char>((unsigned char*)keys, (unsigned int*)values, numElements, plan);
        break;
    case CUDPP_INT:
        runSort<int>((int*)keys, (unsigned int*)values, numElements, plan);
        break;
    case CUDPP_UINT:
        runSort<unsigned int>((unsigned int*)keys, (unsigned int*)values, numElements, plan);
        break;
    case CUDPP_FLOAT:
        runSort<float>((float*)keys, (unsigned int*)values, numElements, plan);
        break;
    case CUDPP_DOUBLE:
        runSort<double>((double*)keys, (unsigned int*)values, numElements, plan);
        break;
    case CUDPP_LONGLONG:
        runSort<long long>((long long*)keys, (unsigned int*)values, numElements, plan);        
        break;
    case CUDPP_ULONGLONG:
        runSort<unsigned long long>((unsigned long long*)keys, (unsigned int*)values, numElements, plan);
        break;
    }

    /*if (plan->m_bKeysOnly)
    {
        switch(plan->m_config.datatype)
        {
        case CUDPP_UINT:
            radixSortKeysOnly((uint*)keys, plan, 
                              numElements, false, 32);
            break;
        case CUDPP_FLOAT:
            radixSortFloatKeysOnly((float*)keys, plan, 
                                   numElements, true, 32);
        }
    }
    else
    {
        switch(plan->m_config.datatype)
        {
        case CUDPP_UINT:      
            radixSort((uint*)keys, (uint*) values, plan, 
                      numElements, false, 32);
            break;
        case CUDPP_FLOAT: 
            radixSortFloatKeys((float*)keys, (uint*) values, plan, 
                               numElements, true, 32);
        }
    }*/
    
}                            

/** @} */ // end radixsort functions
/** @} */ // end cudpp_app

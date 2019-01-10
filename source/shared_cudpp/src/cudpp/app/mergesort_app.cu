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
 * mergesort_app.cu
 *
 * @brief CUDPP application-level merge sorting routines
 */

/** @addtogroup cudpp_app
 * @{
 */

/** @name MergeSort Functions
 * @{
 */

#include "cuda_util.h"
#include "cudpp.h"
#include "cudpp_util.h"
#include "cudpp_mergesort.h"
#include "kernel/mergesort_kernel.cuh"
#include "limits.h"


#define BLOCKSORT_SIZE 1024
#define DEPTH 8

/** @brief Performs merge sort utilizing 3 stages:
 * (1) Blocksort, (2) simple merge and (3) multi merge
 *
 *
 * @param[in,out] pkeys Keys to be sorted.
 * @param[in,out] pvals Associated values to be sorted
 * @param[in] numElements Number of elements in the sort.
 * @param[in] plan Configuration information for mergesort.
 **/
template<typename T>
void runMergeSort(T *pkeys,
                  unsigned int *pvals,
                  size_t numElements,
                  const CUDPPMergeSortPlan *plan)
{
    int numPartitions = (numElements+BLOCKSORT_SIZE-1)/BLOCKSORT_SIZE;
    int numBlocks = numPartitions/2;
    int partitionSize = BLOCKSORT_SIZE;
    int subPartitions = plan->m_subPartitions;



    unsigned int swapPoint = plan->m_swapPoint;



    int numThreads = 128;
    blockWiseSort<T, DEPTH>
        <<<numPartitions, BLOCKSORT_SIZE/DEPTH, (BLOCKSORT_SIZE)*sizeof(T) + (BLOCKSORT_SIZE)*sizeof(unsigned int)>>>(pkeys, pvals, BLOCKSORT_SIZE, numElements);

    int mult = 1; int count = 0;

    //we run p stages of simpleMerge until numBlocks <= some Critical level
    while(numPartitions > swapPoint )
    {
        if(count%2 == 0)
        {
            simpleMerge_lower<T, 2>
                <<<numBlocks, CTASIZE_simple, sizeof(T)*(INTERSECT_B_BLOCK_SIZE_simple+4)>>>
                (pkeys, pvals, (T*)plan->m_tempKeys, plan->m_tempValues, partitionSize*mult, (int)numElements);
            simpleMerge_higher<T, 2>
                <<<numBlocks, CTASIZE_simple, sizeof(T)*(INTERSECT_B_BLOCK_SIZE_simple+4)>>>
                (pkeys, pvals, (T*)plan->m_tempKeys, plan->m_tempValues, partitionSize*mult, (int)numElements);
            if(numPartitions%2 == 1)
            {

                int offset = (partitionSize*mult*(numPartitions-1));
                int numElementsToCopy = numElements-offset;
                simpleCopy<T>
                    <<<(numElementsToCopy+numThreads-1)/numThreads, numThreads>>>(pkeys, pvals, (T*)plan->m_tempKeys, plan->m_tempValues, offset, numElementsToCopy);
            }
        }
        else
        {
            simpleMerge_lower<T, 2>
                <<<numBlocks, CTASIZE_simple, sizeof(T)*(INTERSECT_B_BLOCK_SIZE_simple+4)>>>
                ((T*)plan->m_tempKeys, plan->m_tempValues, pkeys, pvals, partitionSize*mult, numElements);
            simpleMerge_higher<T, 2>
                <<<numBlocks, CTASIZE_simple, sizeof(T)*(INTERSECT_B_BLOCK_SIZE_simple+4)>>>
                ((T*)plan->m_tempKeys, plan->m_tempValues, pkeys, pvals, partitionSize*mult, numElements);
            if(numPartitions%2 == 1)
            {
                int offset = (partitionSize*mult*(numPartitions-1));
                int numElementsToCopy = numElements-offset;
                simpleCopy<T>
                    <<<(numElementsToCopy+numThreads-1)/numThreads, numThreads>>>((T*)plan->m_tempKeys, plan->m_tempValues, pkeys, pvals, offset, numElementsToCopy);
            }
        }

        mult*=2;
        count++;
        numPartitions = (numPartitions+1)/2;
        numBlocks=numPartitions/2;
    }



    //End of simpleMerge, now blocks cooperate to merge partitions
    while (numPartitions > 1)
    {
        int secondBlocks = (numBlocks*subPartitions+numThreads-1)/numThreads;
        if(count%2 == 1)
        {
            findMultiPartitions<T><<<secondBlocks, numThreads>>>( (T*)plan->m_tempKeys, subPartitions, numBlocks*2,
                                                                  partitionSize*mult, plan->m_partitionBeginA, plan->m_partitionSizeA, numElements);
            mergeMulti_lower<T, 4>
                <<<numBlocks*subPartitions, CTASIZE_multi, (INTERSECT_B_BLOCK_SIZE_multi+3)*sizeof(T)>>>
                (pkeys, pvals, (T*)plan->m_tempKeys, plan->m_tempValues, subPartitions, numBlocks, plan->m_partitionBeginA,
                 plan->m_partitionSizeA, mult*partitionSize, numElements);


            mergeMulti_higher<T, 4>
                <<<numBlocks*subPartitions, CTASIZE_multi, (INTERSECT_B_BLOCK_SIZE_multi+3)*sizeof(T)>>>
                (pkeys, pvals, (T*)plan->m_tempKeys, plan->m_tempValues, subPartitions, numBlocks, plan->m_partitionBeginA,
                 plan->m_partitionSizeA, mult*partitionSize, numElements);

            if(numPartitions%2 == 1)
            {
                int offset = (partitionSize*mult*(numPartitions-1));
                int numElementsToCopy = numElements-offset;
                simpleCopy<T>
                    <<<(numElementsToCopy+numThreads-1)/numThreads, numThreads>>>((T*)plan->m_tempKeys, plan->m_tempValues, pkeys, pvals, offset, numElementsToCopy);
            }

        }
        else
        {

            findMultiPartitions <T> <<<secondBlocks, numThreads>>>(pkeys, subPartitions, numBlocks*2, partitionSize*mult,
                                                                   plan->m_partitionBeginA, plan->m_partitionSizeA, numElements);


            mergeMulti_lower<T, 4>
                <<<numBlocks*subPartitions, CTASIZE_multi, (INTERSECT_B_BLOCK_SIZE_multi+3)*sizeof(T)>>>
                ((T*)plan->m_tempKeys, plan->m_tempValues, pkeys, pvals, subPartitions, numBlocks, plan->m_partitionBeginA,
                 plan->m_partitionSizeA, mult*partitionSize, numElements);

            mergeMulti_higher<T, 4>
                <<<numBlocks*subPartitions, CTASIZE_multi, (INTERSECT_B_BLOCK_SIZE_multi+3)*sizeof(T)>>>
                ((T*)plan->m_tempKeys, plan->m_tempValues, pkeys, pvals, subPartitions, numBlocks, plan->m_partitionBeginA,
                 plan->m_partitionSizeA, mult*partitionSize, numElements);

            if(numPartitions%2 == 1)
            {
                int offset = (partitionSize*mult*(numPartitions-1));
                int numElementsToCopy = numElements-offset;
                simpleCopy<T>
                    <<<(numElementsToCopy+numThreads-1)/numThreads, numThreads>>>(pkeys, pvals, (T*)plan->m_tempKeys, plan->m_tempValues, offset, numElementsToCopy);
            }

        }


        count++;
        mult*=2;
        numBlocks/=2;
        subPartitions*=2;
        numPartitions = (numPartitions+1)/2;
        numBlocks = numPartitions/2;
    }


    if(count%2==1)
    {
        CUDA_SAFE_CALL( cudaMemcpy(pkeys, plan->m_tempKeys, numElements*sizeof(T), cudaMemcpyDeviceToDevice));
        CUDA_SAFE_CALL( cudaMemcpy(pvals, plan->m_tempValues, numElements*sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    }
}

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief From the programmer-specified sort configuration,
 *        creates internal memory for performing the sort.
 *
 * @param[in] plan Pointer to CUDPPMergeSortPlan object
 **/
void allocMergeSortStorage(CUDPPMergeSortPlan *plan)
{
    CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_tempValues,    sizeof(unsigned int)*plan->m_numElements));
    CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_partitionBeginA, plan->m_swapPoint*plan->m_subPartitions*sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_partitionSizeA, plan->m_swapPoint*plan->m_subPartitions*sizeof(unsigned int)));
    switch(plan->m_config.datatype)
    {
    case CUDPP_CHAR:
        CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_tempKeys,    sizeof(char)*plan->m_numElements));
        break;
    case CUDPP_UCHAR:
        CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_tempKeys,    sizeof(unsigned char)*plan->m_numElements));
        break;
    case CUDPP_SHORT:
        CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_tempKeys,    sizeof(short)*plan->m_numElements));
        break;
    case CUDPP_USHORT:
        CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_tempKeys,    sizeof(unsigned short)*plan->m_numElements));
        break;
    case CUDPP_INT:
        CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_tempKeys,    sizeof(int)*plan->m_numElements));
        break;
    case CUDPP_UINT:
        CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_tempKeys,    sizeof(unsigned int)*plan->m_numElements));

        break;
    case CUDPP_FLOAT:
        CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_tempKeys,    sizeof(float)*plan->m_numElements));
        break;
    case CUDPP_DOUBLE:
        CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_tempKeys,    sizeof(double)*plan->m_numElements));
        break;
    case CUDPP_LONGLONG:
        CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_tempKeys,    sizeof(long long)*plan->m_numElements));

        break;
    case CUDPP_ULONGLONG:
        CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_tempKeys,    sizeof(unsigned long long)*plan->m_numElements));

        break;
    default:
        break;
    }

}

/** @brief Deallocates intermediate memory from allocRadixSortStorage.
 *
 *
 * @param[in] plan Pointer to CUDPPMergeSortPlan object
 **/

void freeMergeSortStorage(CUDPPMergeSortPlan* plan)
{
    cudaFree(plan->m_tempKeys);
    cudaFree(plan->m_tempValues);
    cudaFree(plan->m_partitionSizeA);
    cudaFree(plan->m_partitionBeginA);
}

/** @brief Dispatch function to perform a sort on an array with
 * a specified configuration.
 *
 * This is the dispatch routine which calls mergeSort...() with
 * appropriate template parameters and arguments as specified by
 * the plan.
 * Currently only sorts keys of type int, unsigned int, and float.
 * @param[in,out] keys Keys to be sorted.
 * @param[in,out] values Associated values to be sorted (through keys).
 * @param[in] numElements Number of elements in the sort.
 * @param[in] plan Configuration information for mergeSort.
 **/

void cudppMergeSortDispatch(void  *keys,
                            void  *values,
                            size_t numElements,
                            const CUDPPMergeSortPlan *plan)
{
    switch(plan->m_config.datatype)
    {
    case CUDPP_INT:
        runMergeSort<int>((int*)keys, (unsigned int*)values, numElements, plan);
        break;
    case CUDPP_UINT:
        runMergeSort<unsigned int>((unsigned int*)keys, (unsigned int*)values, numElements, plan);
        break;
    case CUDPP_FLOAT:
        runMergeSort<float>((float*)keys, (unsigned int*)values, numElements, plan);
        break;
    default:
        /* do nothing, not handled */
        break;
    }
}

#ifdef __cplusplus
}
#endif

/** @} */ // end mergesort functions
/** @} */ // end cudpp_app

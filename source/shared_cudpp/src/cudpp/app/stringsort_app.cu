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
* stringsort_app.cu
*
* @brief CUDPP application-level merge sorting routines
*/

/** @addtogroup cudpp_app
* @{
*/

/** @name StringSort Functions
* @{
*/

#include "cuda_util.h"
#include "cudpp.h"
#include "cudpp_util.h"
#include "cudpp_stringsort.h"
#include "kernel/stringsort_kernel.cuh"
#include "limits.h"


#define BLOCKSORT_SIZE 1024
#define DEPTH 8

void dotAdd(unsigned int* d_address,
           unsigned int* numSpaces,
           unsigned int* packedAddress,
           size_t numElements,
           size_t stringArrayLength)
{
        int numThreads = 128;
        int numBlocks = (numElements+numThreads-1)/numThreads;
        dotAddInclusive<<<numBlocks, numThreads>>>(numSpaces, d_address, packedAddress, numElements, stringArrayLength);
}
void calculateAlignedOffsets(unsigned int* d_address,
                                                         unsigned int* numSpaces,
                                                         unsigned char* d_stringVals,
                                                         unsigned char termC,
                                                         size_t numElements,
                                                         size_t stringArrayLength)
{
        int numThreads = 128;
        int numBlocks = (numElements+numThreads-1)/numThreads;

        alignedOffsets<<<numBlocks, numThreads>>>(numSpaces, d_address, d_stringVals, termC, numElements, stringArrayLength);

}
void packStrings(unsigned int* packedStrings,
                                                 unsigned char* d_stringVals,
                                                 unsigned int* d_keys,
                                                 unsigned int* packedAddress,
                                                 unsigned int* address,
                                                 size_t numElements,
                                                 size_t stringArrayLength,
                                                 unsigned char termC)
{
        unsigned int numThreads = 128;
        unsigned int numBlocks = (numElements + numThreads - 1)/numThreads;

        //Each thread handles one string (irregular parrallelism) other option is to do per character (set of chars)
        //but that requires a binary search per character. Efficiency depends on the dataset
        alignString<<<numBlocks, numThreads>>>(packedStrings, d_stringVals, packedAddress, address, numElements, stringArrayLength, termC);
        createKeys<<<numBlocks, numThreads>>>(d_keys, packedStrings, packedAddress, numElements);

}


void unpackStrings(unsigned int* packedAddress,
                                   unsigned int* packedAddressRef,
                                   unsigned int* address,
                                   unsigned int* addressRef,
                                   size_t numElements)
{
        unsigned int numThreads = 128;
        unsigned int numBlocks = (numElements + numThreads - 1)/numThreads;

        unpackAddresses<<<numBlocks, numThreads>>>(packedAddress, packedAddressRef, address, addressRef, numElements);
}

/** @brief Performs merge sor utilzing three stages.
* (1) Blocksort, (2) simple merge and (3) multi merge on a
* set of strings
*
* @param[in,out] pkeys Keys (first four characters of string) to be sorted.
* @param[in,out] pvals Addresses of string locations for tie-breaks
* @param[out] stringVals global string value array (four characters stuffed into a uint)
* @param[in] numElements Number of elements in the sort.
* @param[in] stringArrayLength The size of our string array in uints (4 chars per uint)
* @param[in] plan Configuration information for mergesort.
* @param[in] termC Termination character for our strings
**/
void runStringSort(unsigned int *pkeys,
                                   unsigned int *pvals,
                                   unsigned int *stringVals,
                                   size_t numElements,
                                   size_t stringArrayLength,
                                   unsigned char termC,
                                   const CUDPPStringSortPlan *plan)
{
        int numPartitions = (numElements+BLOCKSORT_SIZE-1)/BLOCKSORT_SIZE;
        int numBlocks = numPartitions/2;
        int partitionSize = BLOCKSORT_SIZE;




        unsigned int swapPoint = plan->m_swapPoint;
        unsigned int subPartitions = plan->m_subPartitions;



        int numThreads = 128;

        blockWiseStringSort<unsigned int, DEPTH> <<<numPartitions, BLOCKSORT_SIZE/DEPTH, 2*(BLOCKSORT_SIZE)*sizeof(unsigned int)>>>
                                     (pkeys, pvals, stringVals, BLOCKSORT_SIZE, numElements, stringArrayLength, termC);


        int mult = 1; int count = 0;

        //we run p stages of simpleMerge until numBlocks <= some Critical level
        while(numPartitions > swapPoint || (partitionSize*mult < 16384 && numPartitions > 1)/* && numPartitions > 1*/)
        {
                //printf("Running simple merge for %d partitions of size %d\n", numPartitions, partitionSize*mult);
                numBlocks = (numPartitions&0xFFFE);
                if(count%2 == 0)
                {
                        simpleStringMerge<unsigned int, 2>
                                <<<numBlocks, CTASIZE_simple, sizeof(unsigned int)*(2*INTERSECT_B_BLOCK_SIZE_simple+4)>>>(pkeys, plan->m_tempKeys,
                                pvals, plan->m_tempAddress, stringVals, partitionSize*mult, numElements, count, stringArrayLength, termC);

                        if(numPartitions%2 == 1)
                        {

                                int offset = (partitionSize*mult*(numPartitions-1));
                                int numElementsToCopy = numElements-offset;
                                simpleCopy<unsigned int>
                                        <<<(numElementsToCopy+numThreads-1)/numThreads, numThreads>>>(pkeys, pvals, plan->m_tempKeys, plan->m_tempAddress, offset, numElementsToCopy);
                        }
                }
                else
                {
                        simpleStringMerge<unsigned int, 2>
                                <<<numBlocks, CTASIZE_simple, sizeof(unsigned int)*(2*INTERSECT_B_BLOCK_SIZE_simple+4)>>>(plan->m_tempKeys, pkeys,
                                plan->m_tempAddress, pvals, stringVals, partitionSize*mult, numElements, count, stringArrayLength, termC);

                        if(numPartitions%2 == 1)
                        {
                                int offset = (partitionSize*mult*(numPartitions-1));
                                int numElementsToCopy = numElements-offset;
                                simpleCopy<unsigned int>
                                        <<<(numElementsToCopy+numThreads-1)/numThreads, numThreads>>>(plan->m_tempKeys, plan->m_tempAddress, pkeys, pvals, offset, numElementsToCopy);
                        }
                }

                mult*=2;
                count++;
                numPartitions = (numPartitions+1)/2;
        }




        //End of simpleMerge, now blocks cooperate to merge partitions
        while (numPartitions > 1)
        {
                numBlocks = (numPartitions&0xFFFE);
                int secondBlocks = ((numBlocks)*subPartitions+numThreads-1)/numThreads;
                if(count%2 == 1)
                {
                        findMultiPartitions<unsigned int>
                                <<<secondBlocks, numThreads>>>(plan->m_tempKeys, plan->m_tempAddress, stringVals, subPartitions, numBlocks, partitionSize*mult, plan->m_partitionStartA, plan->m_partitionSizeA,
                                plan->m_partitionStartB, plan->m_partitionSizeB, numElements, stringArrayLength, termC);


                        //int lastSubPart = getLastSubPart(numBlocks, subPartitions, partitionSize, mult, numElements);
                        stringMergeMulti<unsigned int, DEPTH_multi>
                                <<<numBlocks*subPartitions, CTASIZE_multi, (2*INTERSECT_B_BLOCK_SIZE_multi+4)*sizeof(unsigned int)>>>(plan->m_tempKeys, pkeys, plan->m_tempAddress,
                                pvals, stringVals, subPartitions, numBlocks, plan->m_partitionStartA, plan->m_partitionSizeA, plan->m_partitionStartB, plan->m_partitionSizeB, mult*partitionSize,
                                count, numElements, stringArrayLength, termC);
                        if(numPartitions%2 == 1)
                        {
                                int offset = (partitionSize*mult*(numPartitions-1));
                                int numElementsToCopy = numElements-offset;
                                simpleCopy<unsigned int>
                                        <<<(numElementsToCopy+numThreads-1)/numThreads, numThreads>>>(plan->m_tempKeys, plan->m_tempAddress, pkeys, pvals, offset, numElementsToCopy);
                        }

                }
                else
                {

                        findMultiPartitions<unsigned int>
                                <<<secondBlocks, numThreads>>>(pkeys, pvals, stringVals, subPartitions, numBlocks, partitionSize*mult, plan->m_partitionStartA, plan->m_partitionSizeA,
                                plan->m_partitionStartB, plan->m_partitionSizeB, numElements, stringArrayLength, termC);
                        //int lastSubPart = getLastSubPart(numBlocks, subPartitions, partitionSize, mult, numElements);
                        stringMergeMulti<unsigned int, DEPTH_multi>
                                <<<numBlocks*subPartitions, CTASIZE_multi, (2*INTERSECT_B_BLOCK_SIZE_multi+4)*sizeof(unsigned int)>>>(pkeys, plan->m_tempKeys, pvals,
                                plan->m_tempAddress, stringVals, subPartitions, numBlocks, plan->m_partitionStartA, plan->m_partitionSizeA, plan->m_partitionStartB, plan->m_partitionSizeB, mult*partitionSize,
                                count, numElements, stringArrayLength, termC);

                        if(numPartitions%2 == 1)
                        {
                                int offset = (partitionSize*mult*(numPartitions-1));
                                int numElementsToCopy = numElements-offset;
                                simpleCopy<unsigned int>
                                        <<<(numElementsToCopy+numThreads-1)/numThreads, numThreads>>>(pkeys, pvals, plan->m_tempKeys, plan->m_tempAddress, offset, numElementsToCopy);
                        }

                }


                count++;
                mult*=2;
                subPartitions*=2;
                numPartitions = (numPartitions+1)/2;
        }

        if(count%2==1)
        {
                CUDA_SAFE_CALL(cudaMemcpy(pkeys, plan->m_tempKeys, numElements*sizeof(unsigned int), cudaMemcpyDeviceToDevice));
                CUDA_SAFE_CALL(cudaMemcpy(pvals, plan->m_tempAddress, numElements*sizeof(unsigned int), cudaMemcpyDeviceToDevice));
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
        * @param[in] plan Pointer to CUDPPStringSortPlan object
        **/
        void allocStringSortStorage(CUDPPStringSortPlan *plan)
        {


                CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_keys, sizeof(unsigned int)*plan->m_numElements));
                CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_tempKeys,    sizeof(unsigned int)*plan->m_numElements));
                CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_tempAddress,    sizeof(unsigned int)*plan->m_numElements));
                CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_packedAddress, sizeof(unsigned int)*(plan->m_numElements+1)));
                CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_packedAddressRef, sizeof(unsigned int)*(plan->m_numElements)));
                CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_addressRef, sizeof(unsigned int)*(plan->m_numElements)));

                CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_spaceScan, sizeof(unsigned int)*(plan->m_numElements+1)));
                CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_numSpaces, sizeof(unsigned int)*(plan->m_numElements+1)));

                CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_partitionSizeA, sizeof(unsigned int)*(plan->m_swapPoint*plan->m_subPartitions*4)));
                CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_partitionSizeB, sizeof(unsigned int)*(plan->m_swapPoint*plan->m_subPartitions*4)));
                CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_partitionStartA, sizeof(unsigned int)*(plan->m_swapPoint*plan->m_subPartitions*4)));
                CUDA_SAFE_CALL(cudaMalloc((void**)&plan->m_partitionStartB, sizeof(unsigned int)*(plan->m_swapPoint*plan->m_subPartitions*4)));
        }

        /** @brief Deallocates intermediate memory from allocStringSortStorage.
        *
        *
        * @param[in] plan Pointer to CUDPStringSortPlan object
        **/

        void freeStringSortStorage(CUDPPStringSortPlan* plan)
        {
                cudaFree(plan->m_keys);
                cudaFree(plan->m_packedAddress);
                cudaFree(plan->m_packedAddressRef);
                cudaFree(plan->m_tempKeys);
                cudaFree(plan->m_tempAddress);
                cudaFree(plan->m_addressRef);

                cudaFree(plan->m_numSpaces);
                cudaFree(plan->m_spaceScan);

                cudaFree(plan->m_partitionSizeA);
                cudaFree(plan->m_partitionSizeB);
                cudaFree(plan->m_partitionStartA);
                cudaFree(plan->m_partitionStartB);
        }

        /** @brief Dispatch function to perform a sort on an array with
        * a specified configuration.
        *
        * This is the dispatch routine which calls stringSort...() with
        * appropriate template parameters and arguments as specified by
        * the plan.
        * @param[in,out] keys Keys (first four chars of string) to be sorted.
        * @param[in,out] values Address of string values in array of null terminated strings
        * @param[in] stringVals Global string array
        * @param[in] numElements Number of elements in the sort.
        * @param[in] stringArrayLength The size of our string array in uints (4 chars per uint)
        * @param[in] termC Termination character for our strings
        * @param[in] plan Configuration information for mergeSort.
        **/

        void cudppStringSortDispatch(unsigned int  *keys,
                                         unsigned int  *values,
                                         unsigned int  *stringVals,
                                         size_t numElements,
                                                                 size_t stringArrayLength,
                                                                 unsigned char termC,
                                         const CUDPPStringSortPlan *plan)
        {
                runStringSort(keys, values, stringVals, numElements, stringArrayLength, termC, plan);
        }

#ifdef __cplusplus
}
#endif






/** @} */ // end stringsort functions
/** @} */ // end cudpp_app

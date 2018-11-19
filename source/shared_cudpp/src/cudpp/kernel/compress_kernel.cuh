// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt 
// in the root directory of this source distribution.
// ------------------------------------------------------------- 

#include <cudpp_globals.h>
#include "sharedmem.h"
#include <stdio.h>
#include "cta/compress_cta.cuh"

/**
 * @file
 * compress_kernel.cu
 * 
 * @brief CUDPP kernel-level compress routines
 */

/** \addtogroup cudpp_kernel
 * @{
 */

/** @name Compress Functions
 * @{
 */

typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;

/** @brief Compute final BWT
 *
 * This is the final stage in the BWT. This stage computes the final
 * values of the BWT output. It is given the indices of where each of
 * the cyclical rotations of the initial input were sorted to. It uses
 * these indices to figure out the last "column" of the sorted
 * cyclical rotations which is the final BWT output.
 *
 *
 * @param[in]  d_bwtIn      Input char array to perform the BWT on.
 * @param[in]  d_values     Input array that gives the indices of where each
                            of the cyclical rotations of the intial input
                            were sorted to.
 * @param[out] d_bwtIndex   Output pointer to store the BWT index. The index
                            tells us where the original string sorted to.
 * @param[out] d_bwtOut     Output char array of the BWT.
 * @param[in]  numElements  The number of elements we are performing a BWT on.
 * @param[in]  tThreads     The total threads we have dispatched on the device.
 *
 **/
__global__ void
bwt_compute_final_kernel(const uchar *d_bwtIn,
                         const uint *d_values,
                         int *d_bwtIndex,
                         uchar *d_bwtOut,
                         uint numElements,
                         uint tThreads)
{
    // Global, local IDs
    uint idx = threadIdx.x + (blockIdx.x * blockDim.x);

    for(int i = idx; i < numElements; i += tThreads)
    {
        uint val = d_values[i];

        if(val == 0) *d_bwtIndex = i;
        d_bwtOut[i] = (val == 0) ? d_bwtIn[numElements-1] : d_bwtIn[val-1];
    }

}

/** @brief Multi merge
 * @param[in]  A_keys        keys to be sorted
 * @param[out] A_keys_out    keys after being sorted
 * @param[in]  A_values      associated values to keys
 * @param[out]  A_values_out  associated values after sort
 * @param[in]  stringValues  keys of each of the cyclical rotations
 * @param[in]  subPartitions  Number of blocks working on a partition (number of sub-partitions)
 * @param[in]  numBlocks 
 * @param[out] partitionBeginA  Where each partition/subpartition will begin in A
 * @param[in]  partitionSizeA Partition sizes decided by function findMultiPartitions
 * @param[out] partitionBeginB  Where each partition/subpartition will begin in B
 * @param[in]  partitionSizeB Partition sizes decided by function findMultiPartitions
 * @param[in] entirePartitionSize The size of an entire partition (before it is split up)
 * @param[in]     numElements   Size of the enitre array
 *
 **/
template<class T, int depth>
__global__ void
stringMergeMulti(T      *A_keys,
                 T      *A_keys_out,
                 T      *A_values,
                 T      *A_values_out,
                 T      *stringValues,
                 int    subPartitions,
                 int    numBlocks, 
                 int    *partitionBeginA,
                 int    *partitionSizeA,
                 int    *partitionBeginB,
                 int    *partitionSizeB,
                 int    entirePartitionSize,
                 size_t numElements)
{
    int myId = blockIdx.x;
    int myStartId = (myId%subPartitions) + (myId/(2*subPartitions))*2*subPartitions;
    int myStartIdxA, myStartIdxB, localAPartSize, localBPartSize, localCPartSize;

    T finalMaxB;
    myStartIdxA = partitionBeginA[myId];
    myStartIdxB = partitionBeginB[myId];
    localAPartSize = partitionSizeA[myId];
    localBPartSize = partitionSizeB[myId];

    int myStartIdxC;                    
    myStartIdxC = myStartIdxA + myStartIdxB - ((myStartId+subPartitions)/(subPartitions))*entirePartitionSize;  
    localCPartSize = localAPartSize + localBPartSize;   

    if(myId%subPartitions != subPartitions-1 && myStartIdxB + localBPartSize < (myId/subPartitions)*entirePartitionSize+2*entirePartitionSize)
        finalMaxB = A_keys[myStartIdxB+localBPartSize+1];
    else
        finalMaxB = UINT_MAX-1;

    //Now we have the beginning and end points of our subpartitions, merge the two together
    T cmpValue;
    int mid, index;
    int bIndex = 0; int aIndex = 0;     

    __shared__ T      BValues[2*BWT_INTERSECT_B_BLOCK_SIZE_multi+3];
    T * BKeys =      &BValues[BWT_INTERSECT_B_BLOCK_SIZE_multi];
    T * BMax =       &BValues[2*BWT_INTERSECT_B_BLOCK_SIZE_multi];
    T * lastAIndex = &BValues[2*BWT_INTERSECT_B_BLOCK_SIZE_multi+3];

    bool breakout = false;
    int tid = threadIdx.x;

    T localMaxB, localMaxA;                     
    T localMinB = 0;    

    T myKey[depth];
    T myValue[depth];

#pragma unroll
    for(int i =0; i <depth; i++)
    {
        myKey[i] =   (depth*tid + i < localAPartSize ? A_keys  [myStartIdxA + depth*tid + i]   : UINT_MAX-3);           
        myValue[i] = (depth*tid + i < localAPartSize ? A_values[myStartIdxA + depth*tid + i]   : UINT_MAX-3);           
    }

    if(bIndex + BWT_INTERSECT_B_BLOCK_SIZE_multi < localBPartSize) 
    {
        int bi = tid;                                   
#pragma unroll
        for(int i = 0;i < BWT_INTERSECT_B_BLOCK_SIZE_multi/BWT_CTASIZE_multi; i++, bi+=BWT_CTASIZE_multi) 
        {
            BKeys[bi] =   A_keys  [myStartIdxB + bi];
            BValues[bi] = A_values[myStartIdxB + bi];
        }
    }
    else {
        int bi = tid;
#pragma unroll
        for(int i = 0;i < BWT_INTERSECT_B_BLOCK_SIZE_multi/BWT_CTASIZE_multi; i++, bi+=BWT_CTASIZE_multi)
        {
            BKeys[bi] =   ((bIndex + bi < localBPartSize) ? A_keys  [myStartIdxB + bi]   : UINT_MAX-1);
            BValues[bi] = ((bIndex + bi < localBPartSize) ? A_values[myStartIdxB + bi]   : UINT_MAX-1);
        }
    }

    if(tid == BWT_CTASIZE_multi-1)
    {
        BMax[1] =  myKey[depth-1];
        BMax[0] =  (BWT_INTERSECT_B_BLOCK_SIZE_multi < localBPartSize ? A_keys  [myStartIdxB + BWT_INTERSECT_B_BLOCK_SIZE_multi-1] : UINT_MAX-1);
    }   

    __syncthreads();

    localMaxB = BMax[0];
    localMaxA = BMax[1];

    do 
    {
        index = 0;

        if(1)
        {
            index = -1;
            int cumulativeAddress = myStartIdxA+aIndex+threadIdx.x*depth;
            mid = (BWT_INTERSECT_B_BLOCK_SIZE_multi/2)-1;

            if(BWT_INTERSECT_B_BLOCK_SIZE_multi >= 1024)
                binSearch_frag_mult<T, depth> (BKeys, BValues, 256, mid, cmpValue, myKey[0], myValue[0], cumulativeAddress, A_values, stringValues, myStartIdxB+bIndex+index, numElements);

            if(BWT_INTERSECT_B_BLOCK_SIZE_multi>= 512)
                binSearch_frag_mult<T, depth> (BKeys, BValues, 128, mid, cmpValue, myKey[0], myValue[0], cumulativeAddress, A_values, stringValues, myStartIdxB+bIndex+index, numElements);

            if(BWT_INTERSECT_B_BLOCK_SIZE_multi >= 256)
                binSearch_frag_mult<T, depth> (BKeys, BValues, 64, mid, cmpValue, myKey[0], myValue[0], cumulativeAddress, A_values, stringValues, myStartIdxB+bIndex+index, numElements);

            binSearch_frag_mult<T, depth> (BKeys, BValues, 32, mid, cmpValue, myKey[0], myValue[0], cumulativeAddress, A_values, stringValues, myStartIdxB+bIndex+index, numElements);                                          
            binSearch_frag_mult<T, depth> (BKeys, BValues, 16, mid, cmpValue, myKey[0], myValue[0], cumulativeAddress, A_values, stringValues, myStartIdxB+bIndex+index, numElements);                  
            binSearch_frag_mult<T, depth> (BKeys, BValues,  8, mid, cmpValue, myKey[0], myValue[0], cumulativeAddress, A_values, stringValues, myStartIdxB+bIndex+index, numElements);                                          
            binSearch_frag_mult<T, depth> (BKeys, BValues,  4, mid, cmpValue, myKey[0], myValue[0], cumulativeAddress, A_values, stringValues, myStartIdxB+bIndex+index, numElements);                  
            binSearch_frag_mult<T, depth> (BKeys, BValues,  2, mid, cmpValue, myKey[0], myValue[0], cumulativeAddress, A_values, stringValues, myStartIdxB+bIndex+index, numElements);                  
            binSearch_frag_mult<T, depth> (BKeys, BValues,  1, mid, cmpValue, myKey[0], myValue[0], cumulativeAddress, A_values, stringValues, myStartIdxB+bIndex+index, numElements);

            index = mid;                        
            cmpValue = BKeys[index];
            if(cmpValue < myKey[0] && index < (localBPartSize-bIndex) && index < BWT_INTERSECT_B_BLOCK_SIZE_multi)                              
                cmpValue = BKeys[++index];

            if(cmpValue == myKey[0] && index < BWT_INTERSECT_B_BLOCK_SIZE_multi && index < (localBPartSize-bIndex))
            {
                int count = 1;
                T tmpKey, cmpKey;
                tmpKey = myKey[0];
                cmpKey = cmpValue;

                while(tmpKey == cmpKey )
                {
                    tmpKey = (myValue[0]+4*count > numElements-1) ? stringValues[myValue[0] + 4*count - numElements] : stringValues[myValue[0] + 4*count];
                    cmpKey = (BValues[index]+4*count > numElements-1) ? stringValues[BValues[index] + 4*count - numElements] : stringValues[BValues[index] + 4*count];

                    if(cmpKey < tmpKey)
                    {   cmpValue = BKeys[++index];      break; }        

                    count++;                            
                }                               
            }


            if(cmpValue < myKey[0])
                index++;

            if(cmpValue == myKey[0]) 
            {
                int count = 1;
                T tmpKey, cmpKey;
                tmpKey = myKey[0];
                cmpKey = cmpValue;

                while(tmpKey == cmpKey)
                {
                    tmpKey = (myValue[0]+4*count > numElements-1) ? stringValues[myValue[0] + 4*count - numElements] : stringValues[myValue[0] + 4*count];
                    cmpKey = (A_values[myStartIdxB+bIndex+index]+4*count > numElements-1) ?
                        stringValues[A_values[myStartIdxB+bIndex+index] + 4*count - numElements] : stringValues[A_values[myStartIdxB+bIndex+index] + 4*count];

                    if(cmpKey < tmpKey)
                    {index++;   break; }                
                    count++;            
                }                                                       
            }

            int globalCAddress = (myStartIdxC + index + bIndex + aIndex + tid*depth);

            if(((myKey[0] < localMaxB && myKey[0] > localMinB) || (bIndex+index) >= (localBPartSize) || 
                (index > 0 && index <BWT_INTERSECT_B_BLOCK_SIZE_multi)) && globalCAddress < (myStartIdxC+localCPartSize) && myKey[0] < finalMaxB)
            {
                A_keys_out  [globalCAddress] = myKey[0];                                                                                        
                A_values_out[globalCAddress] = myValue[0];
            }
            else if((myKey[0] == localMaxB && myKey[0] <= finalMaxB && index > 0 && index <=1024) && globalCAddress < (myStartIdxC+localCPartSize))
            {
                //tie break
                unsigned int tmpAdd = A_values[myStartIdxA+aIndex+depth*tid];
                unsigned int cmpAdd = A_values[myStartIdxB+bIndex+index];
                int count = 1;
                unsigned int tmpKey = (tmpAdd+count > numElements-1) ? stringValues[tmpAdd + count - numElements] : stringValues[tmpAdd + count];
                unsigned int cmpKey = (cmpAdd+count > numElements-1) ? stringValues[cmpAdd + count - numElements] : stringValues[cmpAdd + count];

                while(tmpKey == cmpKey )
                {
                    count++;

                    tmpKey = (tmpAdd+count > numElements-1) ? stringValues[tmpAdd + count - numElements] : stringValues[tmpAdd + count];
                    cmpKey = (cmpAdd+count > numElements-1) ? stringValues[cmpAdd + count - numElements] : stringValues[cmpAdd + count];
                }
                if(tmpKey < cmpKey)
                {
                    A_keys_out  [myStartIdxC + bIndex + aIndex+depth*tid+index] = myKey[0];     
                    A_values_out[myStartIdxC + bIndex + aIndex+depth*tid+index] = myValue[0];   
                }
            }
            else if(myKey[0] == localMinB && globalCAddress < (myStartIdxC+localCPartSize))
            {
                unsigned int tmpAdd = A_values[myStartIdxA+aIndex+depth*tid];
                unsigned int cmpAdd = A_values[myStartIdxB+bIndex+index];

                int count = 1;

                unsigned int tmpKey = (tmpAdd+4*count > numElements-1) ? stringValues[tmpAdd + 4*count - numElements] : stringValues[tmpAdd + 4*count];
                unsigned int cmpKey = (cmpAdd+4*count > numElements-1) ? stringValues[cmpAdd + 4*count - numElements] : stringValues[cmpAdd + 4*count];

                while(tmpKey == cmpKey)
                {
                    count++;

                    tmpKey = (tmpAdd+4*count > numElements-1) ? stringValues[tmpAdd + 4*count - numElements] : stringValues[tmpAdd + 4*count];
                    cmpKey = (cmpAdd+4*count > numElements-1) ? stringValues[cmpAdd + 4*count - numElements] : stringValues[cmpAdd + 4*count];
                }       
                if(tmpKey > cmpKey)
                {
                    A_keys_out  [myStartIdxC + bIndex + aIndex+depth*tid+index] = myKey[0];     
                    A_values_out[myStartIdxC + bIndex + aIndex+depth*tid+index] = myValue[0];   
                }
            }

            if(myKey[1] <= localMaxB)
                linearStringMerge<T, depth>(BKeys, BValues, A_values, myKey[1], myValue[1], index, cmpValue, A_keys_out, A_values_out, stringValues, 
                                            myStartIdxC, myStartIdxA, myStartIdxB, localAPartSize, localBPartSize, localCPartSize, localMaxB, finalMaxB, localMinB, tid, aIndex, bIndex, 
                                            1, subPartitions, numElements);
        }

        if(threadIdx.x == blockDim.x - 1) { *lastAIndex = index; }

        bool reset = false;
        __syncthreads();
        if(localMaxA == localMaxB)
        {       
            //Break the tie
            if(tid == (blockDim.x-1))
            {
                unsigned int tmpAdd = myValue[depth-1]; 
                unsigned int cmpAdd = BValues[BWT_INTERSECT_B_BLOCK_SIZE_multi-1];

                int count = 1;

                unsigned int tmpKey = (tmpAdd+4*count > numElements-1) ? stringValues[tmpAdd + 4*count - numElements] : stringValues[tmpAdd + 4*count];
                unsigned int cmpKey = (cmpAdd+4*count > numElements-1) ? stringValues[cmpAdd + 4*count - numElements] : stringValues[cmpAdd + 4*count];

                while(tmpKey == cmpKey)
                {
                    count++;
                    tmpKey = (tmpAdd+4*count > numElements-1) ? stringValues[tmpAdd + 4*count - numElements] : stringValues[tmpAdd + 4*count];
                    cmpKey = (cmpAdd+4*count > numElements-1) ? stringValues[cmpAdd + 4*count - numElements] : stringValues[cmpAdd + 4*count];
                }

                if(tmpKey > cmpKey)
                    BMax[1]++;
                else
                    BMax[0]++;                          

            }
            __syncthreads();
            localMaxB = BMax[0];
            localMaxA = BMax[1];
            reset = true;               
        }

        __syncthreads();                
        __threadfence();
        if((localMaxA < localMaxB || (bIndex+BWT_INTERSECT_B_BLOCK_SIZE_multi-1) >= localBPartSize) && (aIndex+BWT_INTERSECT_A_BLOCK_SIZE_multi)< localAPartSize)
        {

            aIndex += BWT_INTERSECT_A_BLOCK_SIZE_multi;

            if(aIndex + BWT_INTERSECT_A_BLOCK_SIZE_multi < localAPartSize) 
            {           
#pragma unroll
                for(int i = 0;i < depth; i++) 
                { myKey[i] = A_keys[myStartIdxA + aIndex + depth*tid + i]; myValue[i] = A_values[myStartIdxA + aIndex + depth*tid + i]; }
            }
            else 
            {

#pragma unroll
                for(int i = 0;i < depth; i++) 
                { myKey[i] =   (aIndex+depth*tid + i < localAPartSize ? A_keys[myStartIdxA + aIndex+ depth*tid + i]   : UINT_MAX-3); 
                    myValue[i] = (aIndex+depth*tid + i < localAPartSize ? A_values[myStartIdxA + aIndex+ depth*tid + i]   : UINT_MAX-3);}
            }

            if(tid == BWT_CTASIZE_multi-1)              
            {
                BMax[1] = myKey[depth-1];               
                if(reset)
                    BMax[0]--;                  
            }
            reset = false;
        }                       
        else if(localMaxB < localMaxA && (bIndex+BWT_INTERSECT_B_BLOCK_SIZE_multi-1) < localBPartSize)
        {                               
            bIndex += BWT_INTERSECT_B_BLOCK_SIZE_multi-1;       
            if(bIndex + BWT_INTERSECT_B_BLOCK_SIZE_multi < localBPartSize) 
            {
                int bi = tid;                                   
#pragma unroll
                for(int i = 0;i < BWT_INTERSECT_B_BLOCK_SIZE_multi/BWT_CTASIZE_multi; i++, bi+=BWT_CTASIZE_multi) 
                {
                    BKeys[bi] =   A_keys[myStartIdxB + bIndex + bi];
                    BValues[bi] = A_values[myStartIdxB + bIndex + bi];
                }
            }
            else {
                int bi = tid;
#pragma unroll
                for(int i = 0;i < BWT_INTERSECT_B_BLOCK_SIZE_multi/BWT_CTASIZE_multi; i++, bi+=BWT_CTASIZE_multi) 
                {
                    BKeys[bi] =   ((bIndex + bi < localBPartSize) ? A_keys  [myStartIdxB + bIndex + bi]   : UINT_MAX-1);
                    BValues[bi] = ((bIndex + bi < localBPartSize)? A_values[myStartIdxB + bIndex + bi]   : UINT_MAX-1);
                }
            }

            if(tid ==BWT_CTASIZE_multi-1)
            {
                BMax[0] =  (bIndex + BWT_INTERSECT_B_BLOCK_SIZE_multi < localBPartSize ? A_keys[myStartIdxB + bIndex + BWT_INTERSECT_B_BLOCK_SIZE_multi-1] : UINT_MAX-1);
                if(reset)
                    BMax[1]--;
            }
            reset = false;
            __syncthreads();
            localMinB = BKeys[0];
        }
        else
            breakout = true;    
        __syncthreads();
        __threadfence();

        localMaxB = BMax[0];
        localMaxA = BMax[1];
    }
    while(!breakout);
}

/** @brief Merges the indices for the "upper" block (right block)
 *  
 * Utilizes a "ping-pong" strategy
 * @param[in]  A                Global array of keys
 * @param[in]  splitsPP         Global array of values to be merged
 * @param[in]  numPartitions    number of partitions being considered
 * @param[in]  partitionSize    Size of each partition being considered
 * @param[out] partitionBeginA  Where each partition/subpartition will begin in A
 * @param[out] partitionSizesA  Size of each partition/subpartition in A
 * @param[out] partitionBeginB  Where each partition/subpartition will begin in B
 * @param[out] partitionSizesB  Size of each partition/subpartition in B
 * @param[in]  sizeA            Size of the entire array
 *
 **/
template<class T>
__global__ void
findMultiPartitions(T       *A,
                    int     splitsPP,
                    int     numPartitions,
                    int     partitionSize,
                    int     *partitionBeginA,
                    int     *partitionSizesA,
                    int     *partitionBeginB,
                    int     *partitionSizesB,
                    int     sizeA)
{
    int myId = threadIdx.x + blockIdx.x*blockDim.x;
    if (myId >= (numPartitions*splitsPP))
        return;

    int myStartA, myEndA;
    T testSample, myStartSample, myEndSample;
    int testIdx;
    int subPartitionSize = partitionSize/splitsPP;
    int myPartitionId = myId/splitsPP;
    int mySubPartitionId = myId%splitsPP;

    myStartA = (myPartitionId)*partitionSize + (mySubPartitionId)*subPartitionSize; // we are at the beginning of a partition
    T mySample = A[myStartA];

    if(mySubPartitionId != 0)
    {
        //need to ensure that we don't start inbetween duplicates
        // we have sampled in the middle of a repeated sequence search until we are at a new sequence
        if(threadIdx.x%2 == 1)
        {
            testSample = (myId == 0 ? 0 : A[myStartA-1]);
            int count = 1; testIdx = myStartA;
            if(testSample == mySample)
            {
                while(testSample == mySample && (testIdx+count) < (myPartitionId)*partitionSize+partitionSize)  
                    testSample = A[testIdx + (count++)];
                myStartA = (testIdx + count-1);
            }
        }
        else
        {
            testSample = (myId == 0 ? 0 : A[myStartA-1]);
            int count = 1; testIdx = myStartA;

            if(testSample == mySample)
            {
                while(testSample == mySample && (testIdx+count) < (myPartitionId)*partitionSize+partitionSize)  
                    testSample = A[testIdx + (count++)];
                myStartA = (testIdx + count-1);
            }                   
        }                       
    }


    partitionBeginA[myId] = myStartA; //partitionBegin found for first set
    myStartSample = mySample;
    myEndA = ((myId+1)/splitsPP)*partitionSize+((myId+1)%splitsPP)*subPartitionSize;

    if(mySubPartitionId!= splitsPP-1 )
    {
        mySample = A[myEndA];   
        //need to ensure that we don't start inbetween duplicates

        if(threadIdx.x%2 == 0)
        {
            testSample = A[myEndA-1];                   
            int count = 1; testIdx = myEndA;

            if(testSample == mySample)
            {
                while(testSample == mySample && (testIdx+count) < (myPartitionId)*partitionSize+partitionSize)  
                    testSample = A[testIdx + (count++)];
                myEndA = (testIdx + count-1);
            }
        }
        else
        {
            testSample = A[myEndA-1];                   
            int count = 1; testIdx = myEndA;

            if(testSample == mySample)
            {
                while(testSample == mySample && (testIdx+count) < (myPartitionId)*partitionSize+partitionSize)  
                    testSample = A[testIdx + (count++)];
                myEndA = (testIdx + count-1);
            }
        }
        myEndSample = A[(myEndA < (myPartitionId+1)*partitionSize && myEndA < sizeA) ? myEndA : myEndA];


    }
    else
    {
        myEndA = (myPartitionId)*partitionSize + partitionSize;                 
        myEndSample = A[myEndA-1];

    }

    partitionSizesA[myId] = myEndA-myStartA ;

    int myStartRange = (myPartitionId)*partitionSize + partitionSize - 2*(myPartitionId%2)*partitionSize;
    int myEndRange = myStartRange + partitionSize;
    int first = myStartRange;
    int last = myEndRange;
    int mid = (first + last)/2;
    testSample = A[mid];

    while(testSample != myStartSample)
    {   
        if(testSample < myStartSample)          
            first = mid;                                        
        else            
            last = mid;

        mid = (first+last)/2;           
        testSample = A[mid];
        if(mid == last || mid == first )
            break;      
    }

    while (testSample >= myStartSample && mid > myStartRange)   
        testSample = A[--mid];

    myStartA = mid;     
    first = myStartA;
    last = myEndRange;
    mid = (first + last)/2;     
    testSample = A[mid];

    while(testSample != myEndSample)
    {
        if(testSample <= myEndSample)           
            first = mid;                                        
        else            
            last = mid;

        mid = (first+last)/2;                                   
        testSample = A[mid];
        if(mid == last || mid == first )
            break;
    }

    while (myEndSample >= testSample && mid < myEndRange)
        testSample = A[++mid];

    myEndA = mid;

    if(mySubPartitionId  == splitsPP-1)
        myEndA = myStartRange + partitionSize;

    partitionBeginB[myId] = myStartA;
    partitionSizesB[myId] = myEndA-myStartA;
}

/** @brief Simple merge
*
 * @param[in]  A_keys           keys to be sorted
 * @param[out] A_keys_out       keys after being sorted
 * @param[in]  A_values         associated values to keys
 * @param[out] A_values_out     associated values after sort
 * @param[in]  stringValues     BWT string manipulated to words
 * @param[in]  sizePerPartition Size of each partition being merged
 * @param[in]  size             Size of total Array being sorted
 * @param[in]  stringValues2    keys of each of the cyclical rotations
 * @param[in]  numElements      Number of elements being sorted
 *
 **/
template<class T, int depth>
__global__ void
simpleStringMerge(T         *A_keys,
                  T         *A_keys_out,
                  T         *A_values,
                  T         *A_values_out,
                  T         *stringValues,
                  int       sizePerPartition,
                  int       size,
                  T         *stringValues2,
                  size_t    numElements)
{
    //each block will be responsible for a submerge
    int myStartIdxA, myStartIdxB, myStartIdxC;
    int myId = blockIdx.x;

    int totalSize;

    //Slight difference in loading if we are an odd or even block
    if(myId%2 == 0)
    {
        myStartIdxA = (myId/2)*2*sizePerPartition; myStartIdxB = myStartIdxA+sizePerPartition; myStartIdxC = myStartIdxA;
        totalSize = myStartIdxB + sizePerPartition;
    }
    else
    {
        myStartIdxB = (myId/2)*2*sizePerPartition; myStartIdxA = myStartIdxB + sizePerPartition; myStartIdxC = myStartIdxB;
        totalSize = myStartIdxA + sizePerPartition;
    }

    T cmpValue;
    int mid, index;     int bIndex = 0; int aIndex = 0;

    //Shared Memory pool
    __shared__ T BValues[BWT_INTERSECT_B_BLOCK_SIZE_simple*2+2];
    T* BKeys = (T*) &BValues[BWT_INTERSECT_B_BLOCK_SIZE_simple];
    T* BMax = (T*) &BValues[2*BWT_INTERSECT_B_BLOCK_SIZE_simple];


    bool breakout = false;
    int tid = threadIdx.x;

    T localMaxB, localMaxA;
    T myKey[depth];
    T myValue[depth];

    //Load Registers
    if(aIndex + BWT_INTERSECT_A_BLOCK_SIZE_simple < sizePerPartition)
    {
#pragma unroll
        for(int i = 0;i < depth; i++)
        {
            myKey[i]   = A_keys  [myStartIdxA + aIndex+ depth*tid + i];
            myValue[i] = A_values[myStartIdxA + aIndex+ depth*tid + i];
        }

    }

    else
    {
#pragma unroll
        for(int i = 0;i < depth; i++)
        {
            myKey[i] =   (aIndex+depth*tid + i < sizePerPartition ? A_keys  [myStartIdxA + aIndex+ depth*tid + i]   : UINT_MAX-1); // ==ADDED==
            myValue[i] = (aIndex+depth*tid + i < sizePerPartition ? A_values[myStartIdxA + aIndex+ depth*tid + i]   : UINT_MAX-1); // ==ADDED==
        }
    }

    //Load Shared-Memory
    if(bIndex + BWT_INTERSECT_B_BLOCK_SIZE_simple < sizePerPartition)
    {
        int bi = tid;
#pragma unroll
        for(int i = 0;i < BWT_INTERSECT_B_BLOCK_SIZE_simple/BWT_CTASIZE_simple; i++, bi+=BWT_CTASIZE_simple)
        {
            BKeys[bi] =   A_keys[myStartIdxB + bIndex + bi];
            BValues[bi] = A_values[myStartIdxB + bIndex + bi];
        }

    }
    else
    {
        int bi = tid;
#pragma unroll
        for(int i = 0;i < BWT_INTERSECT_B_BLOCK_SIZE_simple/BWT_CTASIZE_simple; i++, bi+=BWT_CTASIZE_simple)
        {
            BKeys[bi] =   (bIndex + bi < sizePerPartition ? A_keys  [myStartIdxB + bIndex + bi] : UINT_MAX);
            BValues[bi] = (bIndex + bi < sizePerPartition ? A_values[myStartIdxB + bIndex + bi] : UINT_MAX);
        }
    }

    //Save localMaxA and localMaxB
    if(tid == BWT_CTASIZE_simple-1)
        BMax[1] = myKey[depth-1];
    if(tid == 0)
        BMax[0] =  (bIndex + BWT_INTERSECT_B_BLOCK_SIZE_simple - 1 < sizePerPartition ?
                    A_keys[myStartIdxB + bIndex + BWT_INTERSECT_B_BLOCK_SIZE_simple - 1] : UINT_MAX);

    __syncthreads();

    //Maximum values for B and A in this stream
    localMaxB = BMax[0];
    localMaxA = BMax[1];

    T localMinB = 0;

    __syncthreads();
    __threadfence(); //Extra Added

    do
    {
        __syncthreads();

        index = 0;

        int cumulativeAddress = myStartIdxA+aIndex+threadIdx.x*depth;                   
        if((myKey[0] <= localMaxB && myKey[depth-1] >= localMinB) ||  (bIndex+BWT_INTERSECT_B_BLOCK_SIZE_simple-1) >= (sizePerPartition) && cumulativeAddress < sizePerPartition) // ==ADDED==
        {
            index = -1;

            mid = (BWT_INTERSECT_B_BLOCK_SIZE_simple/2)-1;
            if(BWT_INTERSECT_B_BLOCK_SIZE_simple >= 1024)
                binSearch_fragment<T,depth> (BKeys, BValues, 256, mid, cmpValue, myKey[0], myValue[0], stringValues, stringValues2, numElements);
            if(BWT_INTERSECT_B_BLOCK_SIZE_simple >= 512)
                binSearch_fragment<T,depth> (BKeys, BValues, 128, mid, cmpValue, myKey[0], myValue[0], stringValues, stringValues2, numElements);
            if(BWT_INTERSECT_B_BLOCK_SIZE_simple >= 256)
                binSearch_fragment<T,depth> (BKeys, BValues, 64, mid, cmpValue, myKey[0], myValue[0], stringValues, stringValues2, numElements);

            binSearch_fragment<T,depth> (BKeys, BValues, 32, mid, cmpValue, myKey[0], myValue[0], stringValues, stringValues2, numElements);
            binSearch_fragment<T,depth> (BKeys, BValues, 16, mid, cmpValue, myKey[0], myValue[0], stringValues, stringValues2, numElements);
            binSearch_fragment<T,depth> (BKeys, BValues,  8, mid, cmpValue, myKey[0], myValue[0], stringValues, stringValues2, numElements);
            binSearch_fragment<T,depth> (BKeys, BValues,  4, mid, cmpValue, myKey[0], myValue[0], stringValues, stringValues2, numElements);
            binSearch_fragment<T,depth> (BKeys, BValues,  2, mid, cmpValue, myKey[0], myValue[0], stringValues, stringValues2, numElements);
            binSearch_fragment<T,depth> (BKeys, BValues,  1, mid, cmpValue, myKey[0], myValue[0], stringValues, stringValues2, numElements);

            index = mid;
            cmpValue = BKeys[index];

            //correct search if needed
            if(cmpValue < myKey[0] && index < BWT_INTERSECT_B_BLOCK_SIZE_simple)
                cmpValue = BKeys[++index];

            //Tied version of previous if statement
            if(cmpValue == myKey[0] && index < BWT_INTERSECT_B_BLOCK_SIZE_simple)
            {
                int count = 1;
                T tmpKey, cmpKey;
                tmpKey = myKey[0];
                cmpKey = cmpValue;

                while(tmpKey == cmpKey)
                {
                    tmpKey = (myValue[0]+4*count > numElements-1) ? stringValues2[myValue[0] + 4*count - numElements] : stringValues2[myValue[0] + 4*count];
                    cmpKey = (BValues[index]+4*count > numElements-1) ? stringValues2[BValues[index] + 4*count - numElements] : stringValues2[BValues[index] + 4*count];

                    if(cmpKey < tmpKey)
                    {
                        cmpValue = BKeys[++index];
                        break;
                    }
                    count++;
                }
            }


            if(cmpValue < myKey[0] && (bIndex+index) < sizePerPartition)
            {
                index++;
                cmpValue =  A_keys[myStartIdxB+bIndex + (index)];
            }

            //Tied version of previous if statement
            if(cmpValue == myKey[0])
            {
                int count = 1;
                T tmpKey, cmpKey;
                tmpKey = myKey[0];
                cmpKey = cmpValue;

                while(tmpKey == cmpKey && (bIndex+index) < sizePerPartition)
                {
                    tmpKey = (myValue[0]+4*count > numElements-1) ? stringValues2[myValue[0] + 4*count - numElements] : stringValues2[myValue[0] + 4*count];
                    cmpKey = (A_values[myStartIdxB+bIndex+index]+4*count > numElements-1) ?
                        stringValues2[A_values[myStartIdxB+bIndex+index] + 4*count - numElements] : stringValues2[A_values[myStartIdxB+bIndex+index] + 4*count];

                    if(cmpKey < tmpKey)
                    {
                        index++;
                        break;
                    }
                    count++;
                }
            }

            //End Binary Search
            //binary search done for first element in our set (A_0)

            //Save Value if it is valid (correct window)
            //If we are on the edge of a window, and we are tied with the localMax or localMin value
            //we must go to global memory to find out if we are valid
            if((myKey[0] < localMaxB && myKey[0] > localMinB) || (index==BWT_INTERSECT_B_BLOCK_SIZE_simple && (bIndex+index)>=sizePerPartition) || (index > 0 && index <BWT_INTERSECT_B_BLOCK_SIZE_simple))
            {
                A_keys_out[myStartIdxC + bIndex + aIndex + depth*tid + index] = myKey[0];
                A_values_out[myStartIdxC + bIndex + aIndex + depth*tid + index] = myValue[0];

            }
            else if(myKey[0] == localMaxB && index == BWT_INTERSECT_B_BLOCK_SIZE_simple)
            {
                //tie break
                unsigned int tmpAdd = myValue[0];
                unsigned int cmpAdd = A_values[myStartIdxB+bIndex+index];

                int count = 1;

                T tmpKey = (tmpAdd+4*count > numElements-1) ? stringValues2[tmpAdd + 4*count - numElements] : stringValues2[tmpAdd + 4*count];
                T cmpKey = (cmpAdd+4*count > numElements-1) ? stringValues2[cmpAdd + 4*count - numElements] : stringValues2[cmpAdd + 4*count];

                while(tmpKey == cmpKey)
                {
                    count++;
                    tmpKey = (tmpAdd+4*count > numElements-1) ? stringValues2[tmpAdd + 4*count - numElements] : stringValues2[tmpAdd + 4*count];
                    cmpKey = (cmpAdd+4*count > numElements-1) ? stringValues2[cmpAdd + 4*count - numElements] : stringValues2[cmpAdd + 4*count];
                }

                if(tmpKey < cmpKey)
                {
                    A_keys_out[myStartIdxC + bIndex + aIndex + depth*tid + index] = myKey[0];
                    A_values_out[myStartIdxC + bIndex + aIndex + depth*tid + index] = myValue[0];
                }

            }
            else if(myKey[0] == localMinB && index == 0)
            {
                unsigned int tmpAdd = myValue[0];
                unsigned int cmpAdd = BValues[0];

                int count = 1;

                unsigned int tmpKey = (tmpAdd+4*count > numElements-1) ? stringValues2[tmpAdd + 4*count - numElements] : stringValues2[tmpAdd + 4*count];
                unsigned int cmpKey = (cmpAdd+4*count > numElements-1) ? stringValues2[cmpAdd + 4*count - numElements] : stringValues2[cmpAdd + 4*count];

                while(tmpKey == cmpKey)
                {
                    count++;
                    tmpKey = (tmpAdd+4*count > numElements-1) ? stringValues2[tmpAdd + 4*count - numElements] : stringValues2[tmpAdd + 4*count];
                    cmpKey = (cmpAdd+4*count > numElements-1) ? stringValues2[cmpAdd + 4*count - numElements] : stringValues2[cmpAdd + 4*count];
                }

                if(tmpKey > cmpKey)
                {
                    A_keys_out[myStartIdxC + bIndex + aIndex+depth*tid+index] = myKey[0];
                    A_values_out[myStartIdxC + bIndex + aIndex+depth*tid+index] = myValue[0];
                }
            }

            //After binary search, linear merge
            lin_merge_simple<T, depth>(cmpValue, myKey[1], myValue[1], index, BKeys, BValues, stringValues, A_values, A_keys_out, A_values_out,
                                       myStartIdxA, myStartIdxB, myStartIdxC, localMinB, localMaxB, aIndex+tid*depth, bIndex, totalSize, sizePerPartition, 1, stringValues2, numElements);

        }

        bool reset = false;
        __syncthreads();
        if(localMaxA == localMaxB)
        {

            //Break the tie
            if(tid == (blockDim.x-1))
            {
                unsigned int tmpAdd = myValue[1];
                unsigned int cmpAdd = BValues[BWT_INTERSECT_B_BLOCK_SIZE_simple-1];

                int count = 1;

                unsigned int tmpKey = (tmpAdd+4*count > numElements-1) ? stringValues2[tmpAdd + 4*count - numElements] : stringValues2[tmpAdd + 4*count];
                unsigned int cmpKey = (cmpAdd+4*count > numElements-1) ? stringValues2[cmpAdd + 4*count - numElements] : stringValues2[cmpAdd + 4*count];

                while(tmpKey == cmpKey)
                {
                    count++;
                    tmpKey = (tmpAdd+4*count > numElements-1) ? stringValues2[tmpAdd + 4*count - numElements] : stringValues2[tmpAdd + 4*count];
                    cmpKey = (cmpAdd+4*count > numElements-1) ? stringValues2[cmpAdd + 4*count - numElements] : stringValues2[cmpAdd + 4*count];
                }

                if(tmpKey > cmpKey)
                    BMax[1]++;
                else
                    BMax[0]++;
            }
            __syncthreads();
            localMaxB = BMax[0];
            localMaxA = BMax[1];
            reset = true;

        }

        __syncthreads();
        if((localMaxA < localMaxB || (bIndex+BWT_INTERSECT_B_BLOCK_SIZE_simple-1) >= sizePerPartition) && (aIndex+BWT_INTERSECT_A_BLOCK_SIZE_simple)< sizePerPartition)
        {
            __syncthreads();
            __threadfence();

            aIndex += BWT_INTERSECT_A_BLOCK_SIZE_simple;

            if(aIndex + BWT_INTERSECT_A_BLOCK_SIZE_simple < sizePerPartition)
            {
#pragma unroll
                for(int i = 0;i < depth; i++)
                {
                    myKey[i] = A_keys[myStartIdxA + aIndex + depth*tid + i];
                    myValue[i] = A_values[myStartIdxA + aIndex + depth*tid + i];

                }
            }
            else
            {

#pragma unroll
                for(int i = 0;i < depth; i++)
                {
                    myKey[i] = (aIndex+depth*tid + i < sizePerPartition ? A_keys[myStartIdxA + aIndex + depth*tid + i]   : UINT_MAX-1);
                    myValue[i] = (aIndex+depth*tid + i < sizePerPartition ? A_values[myStartIdxA + aIndex + depth*tid + i]   : UINT_MAX-1);

                }
            }
            if(tid == BWT_CTASIZE_simple-1)
            {
                BMax[1] = myKey[depth-1];
                if(reset)
                    BMax[0]--;
            }
            reset = false;

        }
        else if(localMaxB < localMaxA && (bIndex+BWT_INTERSECT_B_BLOCK_SIZE_simple-1) < sizePerPartition)
        {

            //Use UINT_MAX as an "invalid/no-value" type in case the streaming window cannot be filled
            bIndex += BWT_INTERSECT_B_BLOCK_SIZE_simple-1;

            if(bIndex + BWT_INTERSECT_B_BLOCK_SIZE_simple < sizePerPartition)
            {
                int bi = tid;
#pragma unroll
                for(int i = 0;i < BWT_INTERSECT_B_BLOCK_SIZE_simple/BWT_CTASIZE_simple; i++, bi+=BWT_CTASIZE_simple)
                {
                    BKeys[bi] =   A_keys[myStartIdxB + bIndex + bi];
                    BValues[bi] = A_values[myStartIdxB + bIndex + bi];

                }
            }
            else
            {
                int bi = tid;
#pragma unroll
                for(int i = 0;i < BWT_INTERSECT_B_BLOCK_SIZE_simple/BWT_CTASIZE_simple; i++, bi+=BWT_CTASIZE_simple)
                {
                    BKeys[bi] =   (bIndex + bi < sizePerPartition ? A_keys[myStartIdxB + bIndex + bi]   : UINT_MAX);
                    BValues[bi] = (bIndex + bi < sizePerPartition ? A_values[myStartIdxB + bIndex + bi] : UINT_MAX);

                }
            }

            if(tid == 0)
            {
                BMax[0] =  (bIndex + BWT_INTERSECT_B_BLOCK_SIZE_simple < sizePerPartition ? A_keys[myStartIdxB + bIndex + BWT_INTERSECT_B_BLOCK_SIZE_simple - 1] : UINT_MAX);
                if(reset)
                    BMax[1]--;
            }
            __syncthreads();
            localMinB = BKeys[0];

            __syncthreads();
            __threadfence(); //Extra Added
            reset = false;
        }
        else
            breakout = true;
        __syncthreads();



        localMaxB = BMax[0];
        localMaxA = BMax[1];

        __syncthreads();
        __threadfence(); //Extra Added

    }
    while(!breakout);

    __syncthreads();

}

/** @brief Sorts blocks of data of size blockSize
 * @param[in,out] A_keys        keys to be sorted
 * @param[in,out] A_address     associated values to keys
 * @param[in]     stringVals    BWT string manipulated to words
 * @param[in]     stringVals2   keys of each of the cyclical rotations
 * @param[in]     blockSize     Size of the chunks being sorted
 * @param[in]     numElements   Size of the enitre array
 **/
template<class T, int depth>
__global__ void blockWiseStringSort(T*      A_keys,
                                    T*      A_address,
                                    const   T* stringVals,
                                    T*      stringVals2,
                                    int     blockSize,
                                    size_t  numElements)
{
    //load into registers
    T Aval[depth]; // keys
    T saveValue[depth];

    __shared__ T scratchPad[2*BWT_BLOCKSORT_SIZE];

    // half of scratch pad is taken up by addresses
    // there are BWT_BLOCKSORT_SIZE addresses
    T* addressPad = (T*) &scratchPad[BWT_BLOCKSORT_SIZE];


    int bid = blockIdx.x;
    int tid = threadIdx.x;


    for(int i = 0; i < depth; i++)
    {
        // Brining keys into registers
        Aval[i] = A_keys[bid*blockSize+tid*depth+i];
    }

    // bringing adressess into shared mem (coalesced reads)
    for(int i = tid; i < blockSize; i+=BWT_CTA_BLOCK)
        addressPad[i] = A_address[blockSize*bid+i];


    __syncthreads();

    //Sort first 8 values
    // Bitonic sort -- each thread sorts 8 string using bitonc
    int offset = tid*depth;

    compareSwapVal<T>(Aval[0], Aval[1], offset, offset+1, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[2], Aval[3], offset+2, offset+3, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[0], Aval[2], offset+0, offset+2, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[1], Aval[3], offset+1, offset+3, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[1], Aval[2], offset+1, offset+2, addressPad, stringVals, stringVals2, numElements);
    //4-way sort on second set of values
    compareSwapVal<T>(Aval[4], Aval[5], offset+4, offset+5, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[6], Aval[7], offset+6, offset+7, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[4], Aval[6], offset+4, offset+6, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[5], Aval[7], offset+5, offset+7, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[5], Aval[6], offset+5, offset+6, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[0], Aval[4], offset+0, offset+4, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[1], Aval[5], offset+1, offset+5, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[2], Aval[6], offset+2, offset+6, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[3], Aval[7], offset+3, offset+7, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[2], Aval[4], offset+2, offset+4, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[3], Aval[5], offset+3, offset+5, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[1], Aval[2], offset+1, offset+2, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[3], Aval[4], offset+3, offset+4, addressPad, stringVals, stringVals2, numElements);
    compareSwapVal<T>(Aval[5], Aval[6], offset+5, offset+6, addressPad, stringVals, stringVals2, numElements);

    __syncthreads();

    int j;
#pragma unroll
    // loading all keys into shared mem., used to find where to merge into
    for(int i=0;i<depth;i++)
        scratchPad[tid*depth+i] = Aval[i];

    __syncthreads();

    // 1st half of scratch pad has keys (first 4 chars of each 1024 strings)
    // 2nd half of scratch pad has values

    T * in = scratchPad;

    int mult = 1;
    int count = 0;
    int steps = 128;

    while (mult < steps)
    {
        // What is first, last, midpoint?
        int first, last;
        first = (tid>>(count+1))*depth*2*mult;
        int midPoint = first+mult*depth;

        T cmpValue;
        T tmpVal;

        //first half or second half
        int addPart = threadIdx.x%(mult<<1) >= mult ? 1 : 0;

        if(addPart == 0)
            first += depth*mult;
        last = first+depth*mult-1;

        j = (first+last)/2;

        int startAddress = threadIdx.x*depth-midPoint;

        int range = last-first;

        __syncthreads();
        tmpVal = Aval[0];

        //Begin binary search
        switch(range)
        {
        case 1023: bin_search_block<T, depth>(cmpValue, tmpVal, in, addressPad, stringVals, j, 256, stringVals2, numElements);
        case 511: bin_search_block<T, depth>(cmpValue, tmpVal, in, addressPad, stringVals, j, 128, stringVals2, numElements);
        case 255: bin_search_block<T, depth>(cmpValue, tmpVal, in, addressPad, stringVals, j, 64, stringVals2, numElements);
        case 127: bin_search_block<T, depth>(cmpValue, tmpVal, in, addressPad, stringVals, j, 32, stringVals2, numElements);
        case 63: bin_search_block<T, depth>(cmpValue, tmpVal, in, addressPad, stringVals, j, 16, stringVals2, numElements);
        case 31: bin_search_block<T, depth>(cmpValue, tmpVal, in, addressPad, stringVals, j, 8, stringVals2, numElements);
        case 15: bin_search_block<T, depth>(cmpValue, tmpVal, in, addressPad, stringVals, j, 4, stringVals2, numElements);
        case 7: bin_search_block<T, depth>(cmpValue, tmpVal, in,  addressPad,stringVals, j, 2, stringVals2, numElements);
        case 3: bin_search_block<T, depth>(cmpValue, tmpVal, in, addressPad, stringVals, j, 1, stringVals2, numElements);
        }

        cmpValue = in[j];
        if(cmpValue == tmpVal)
        {
            T tmp = (addressPad[depth*tid]+4*1 > numElements-1) ?
                stringVals2[addressPad[depth*tid]+4*1-numElements] : stringVals2[addressPad[depth*tid]+4*1];
            T tmp2 = (addressPad[j]+4*1 > numElements-1) ? stringVals2[addressPad[j]+4*1-numElements] : stringVals2[addressPad[j]+4*1];

            int i = 2;
            while(tmp == tmp2)
            {
                tmp = (addressPad[depth*tid]+4*i > numElements-1) ?
                    stringVals2[addressPad[depth*tid]+4*i-numElements] : stringVals2[addressPad[depth*tid]+4*i];
                tmp2 = (addressPad[j]+4*i > numElements-1) ? stringVals2[addressPad[j]+4*i-numElements] : stringVals2[addressPad[j]+4*i];

                i++;
            }
            j = (tmp2 < tmp ? j +1 : j);
            cmpValue = in[j];
        }
        else if(cmpValue < tmpVal)
            cmpValue = in[++j];

        if(cmpValue == tmpVal && j == last)
        {
            T tmp = (addressPad[depth*tid]+4*1 > numElements-1) ?
                stringVals2[addressPad[depth*tid]+4*1-numElements] : stringVals2[addressPad[depth*tid]+4*1];
            T tmp2 = (addressPad[j]+4*1 > numElements-1) ? stringVals2[addressPad[j]+4*1-numElements] : stringVals2[addressPad[j]+4*1];

            int i = 2;
            while(tmp == tmp2)
            {
                tmp = (addressPad[depth*tid]+4*i > numElements-1) ?
                    stringVals2[addressPad[depth*tid]+4*i-numElements] : stringVals2[addressPad[depth*tid]+4*i];
                tmp2 = (addressPad[j]+4*i > numElements-1) ? stringVals2[addressPad[j]+4*i-numElements] : stringVals2[addressPad[j]+4*i];

                i++;
            }
            j = (tmp2 < tmp ? j +1 : j);
        }
        else if(cmpValue < tmpVal && j == last)
            j++;

        Aval[0] = j+startAddress;
        lin_search_block<T, depth>(cmpValue,  Aval[1], in, addressPad, stringVals, j, 1, last, startAddress, 0, stringVals2, numElements);
        lin_search_block<T, depth>(cmpValue,  Aval[2], in, addressPad, stringVals, j, 2, last, startAddress, 0, stringVals2, numElements);
        lin_search_block<T, depth>(cmpValue,  Aval[3], in, addressPad, stringVals, j, 3, last, startAddress, 0, stringVals2, numElements);
        lin_search_block<T, depth>(cmpValue,  Aval[4], in, addressPad, stringVals, j, 4, last, startAddress, 0, stringVals2, numElements);
        lin_search_block<T, depth>(cmpValue,  Aval[5], in, addressPad, stringVals, j, 5, last, startAddress, 0, stringVals2, numElements);
        lin_search_block<T, depth>(cmpValue,  Aval[6], in, addressPad, stringVals, j, 6, last, startAddress, 0, stringVals2, numElements);
        lin_search_block<T, depth>(cmpValue,  Aval[7], in, addressPad, stringVals, j, 7, last, startAddress, 0, stringVals2, numElements);
        __syncthreads();
        saveValue[0] = in[tid*depth];
        saveValue[1] = in[tid*depth+1];
        saveValue[2] = in[tid*depth+2];
        saveValue[3] = in[tid*depth+3];
        saveValue[4] = in[tid*depth+4];
        saveValue[5] = in[tid*depth+5];
        saveValue[6] = in[tid*depth+6];
        saveValue[7] = in[tid*depth+7];
        __syncthreads();
        in[Aval[0]] = saveValue[0];
        in[Aval[1]] = saveValue[1];
        in[Aval[2]] = saveValue[2];
        in[Aval[3]] = saveValue[3];
        in[Aval[4]] = saveValue[4];
        in[Aval[5]] = saveValue[5];
        in[Aval[6]] = saveValue[6];
        in[Aval[7]] = saveValue[7];
        __syncthreads();
        saveValue[0] = addressPad[tid*depth];
        saveValue[1] = addressPad[tid*depth+1];
        saveValue[2] = addressPad[tid*depth+2];
        saveValue[3] = addressPad[tid*depth+3];
        saveValue[4] = addressPad[tid*depth+4];
        saveValue[5] = addressPad[tid*depth+5];
        saveValue[6] = addressPad[tid*depth+6];
        saveValue[7] = addressPad[tid*depth+7];
        __syncthreads();
        addressPad[Aval[0]] = saveValue[0];
        addressPad[Aval[1]] = saveValue[1];
        addressPad[Aval[2]] = saveValue[2];
        addressPad[Aval[3]] = saveValue[3];
        addressPad[Aval[4]] = saveValue[4];
        addressPad[Aval[5]] = saveValue[5];
        addressPad[Aval[6]] = saveValue[6];
        addressPad[Aval[7]] = saveValue[7];
        __syncthreads();



        mult*=2;
        count++;

        if(mult < steps)
        {
            __syncthreads();

#pragma unroll
            for(int i=0;i<depth;i++)
                Aval[i] = in[tid*depth+i];
        }
        __syncthreads();
    }

    __syncthreads();
#pragma unroll
    for(int i=tid;i<blockSize;i+= BWT_CTA_BLOCK)
    {
        A_keys[bid*blockSize+i] = in[i];
        A_address[bid*blockSize+i] = addressPad[i];
        __syncthreads();
    }
}

/** @brief Massage input to set up for merge sort
 * @param[in]  d_bwtIn      A char array of the input data stream to perform the BWT on.
 * @param[out] d_bwtInRef   BWT string manipulated to words.
 * @param[out] d_keys       An array of associated keys to sort by the first four chars
                            of the cyclical rotations.
 * @param[out] d_values     Array of values associates with the keys to sort.
 * @param[out] d_bwtInRef2  keys of each of the cyclical rotations.
 * @param[in]  tThreads     Pointer to the plan object used for this BWT.
 **/
__global__ void
bwt_keys_construct_kernel(uchar4    *d_bwtIn,
                          uint      *d_bwtInRef,
                          uint      *d_keys,
                          uint      *d_values,
                          uint      *d_bwtInRef2,
                          uint      tThreads)
{
    // Global, local IDs
    uint idx = threadIdx.x + (blockIdx.x * blockDim.x);

    if(idx < tThreads)
    {
        uint start = idx*4;
        uchar4 prefix = d_bwtIn[idx];
        uchar4 prefix_end;

        uint keys[4];

        if(idx == (tThreads-1))
            prefix_end = d_bwtIn[0];
        else
            prefix_end = d_bwtIn[idx+1];

        // Manipulate ordering of d_bwtIn for when we typecast
        // if 1st four chars of d_bwtIn[0,1,2,3] = [a][b][c][d]
        // then 1st int of d_bwtInRef[0] = [abcd] instead of [dcba]
        uint word = 0;
        word |= (uint)prefix.x<<24;
        word |= (uint)prefix.y<<16;
        word |= (uint)prefix.z<<8;
        word |= (uint)prefix.w;
        d_bwtInRef[idx] = word;

        // key0
        keys[0] = (uint)prefix.x << 24;
        keys[0] |= (uint)prefix.y << 16;
        keys[0] |= (uint)prefix.z << 8;
        keys[0] |= (uint)prefix.w;
        d_keys[start] = keys[0];
        d_values[start] = start;

        // key1
        keys[1] = (uint)prefix.y << 24;
        keys[1] |= (uint)prefix.z << 16;
        keys[1] |= (uint)prefix.w << 8;
        keys[1] |= (uint)prefix_end.x;;
        d_keys[start+1] = keys[1];
        d_values[start+1] = start+1;

        // key2
        keys[2] = (uint)prefix.z << 24;
        keys[2] |= (uint)prefix.w << 16;
        keys[2] |= (uint)prefix_end.x << 8;
        keys[2] |= (uint)prefix_end.y;
        d_keys[start+2] = keys[2];
        d_values[start+2] = start+2;

        // key3
        keys[3] = (uint)prefix.w << 24;
        keys[3] |= (uint)prefix_end.x << 16;
        keys[3] |= (uint)prefix_end.y << 8;
        keys[3] |= (uint)prefix_end.z;
        d_keys[start+3] = keys[3];
        d_values[start+3] = start+3;

        d_bwtInRef2[start] = keys[0];
        d_bwtInRef2[start+1] = keys[1];
        d_bwtInRef2[start+2] = keys[2];
        d_bwtInRef2[start+3] = keys[3];
    }
}



/** @brief First stage in MTF (Reduction)
 * @param[in]  d_mtfIn      A char array of the input data stream to perform the MTF on.
 * @param[out] d_lists      A pointer to the start of MTF lists.
 * @param[out] d_list_sizes An array storing the size of each MTF list.
 * @param[in]  nLists       Total number of MTF lists.
 * @param[in]  offset       The offset during the reduction stage. Initialized to two.
 * @param[in]  numElements  Total number of input elements MTF transform.
 **/
__global__ void
mtf_reduction_kernel(const uchar * d_mtfIn,
                     uchar       * d_lists,
                     ushort      * d_list_sizes,
                     uint          nLists,
                     uint          offset,
                     uint          numElements)
{
#if (__CUDA_ARCH__ >= 200)
    __shared__ uchar shared[(MTF_PER_THREAD+256)*(MTF_THREADS_BLOCK/2)*sizeof(uchar) + MTF_THREADS_BLOCK*sizeof(ushort)];
    __shared__ int FLAG;

    // Global, local IDs
    uint idx = threadIdx.x + (blockIdx.x * blockDim.x);
    uint lid = threadIdx.x;

    // Shared mem setup
    uchar* sdata = (uchar*)&shared[0];
    ushort* s_sizes = (ushort*)&sdata[(MTF_PER_THREAD+256)*(blockDim.x/2)*sizeof(uchar)];

    if(lid == 0) {
        FLAG = 0;
    }
    __syncthreads();

    int C[8];
#pragma unroll
    for(int i=0; i<8; i++)
        C[i] = 0;

    ushort list_size = 0;
    uint sdata_index = (lid%2 == 0) ? (lid/2*(256+MTF_PER_THREAD)) : ((lid+1)/2*(256+MTF_PER_THREAD) - 256);

    uchar mtfVal;
    uchar C_ID;
    uchar C_bit;

    if(idx < nLists)
    {
        // Take initial input and generate initial 
        // MTF lists every PER_THREAD elements

        uint start = (idx+1)*MTF_PER_THREAD-1;
        uint finish = idx*MTF_PER_THREAD;
        uint list_start = idx*256;
        int j = 0;

        for(int i=(int)start; i>=(int)finish; i--)
        {
            // Each working thread traverses through PER_THREAD elements,
            // and generate nLists (numElements/PER_THREAD) lists
            mtfVal = (i<numElements) ? d_mtfIn[i] : 0;
            C_ID = mtfVal/32;
            C_bit = mtfVal%32;

            int bit_to_set = 1 << (31-C_bit);
            if( !((C[C_ID]<<C_bit) & 0x80000000) )
            {
                d_lists[list_start] = mtfVal;   // Add to device list
                sdata[sdata_index+j] = mtfVal;  // Also store copy of list in shared memory, later used for reduction
                list_start++;
                list_size++;
                j++;
                C[C_ID] |= bit_to_set;          // Set bit so this value is not added to list again
            }

        }

        d_list_sizes[idx] = list_size;      // Store each initial list's size
        s_sizes[lid] = list_size;           // Store list size in smem for reduction phase
    }

    __syncthreads();

    uint    l_offset = offset;              // Initial offset for reduction phase
    uchar   done = 0;
    uchar   working = 0;
    ushort  initListSize = list_size;       // Init size of list being added to
    uint    liveThreads = blockDim.x/2;     // Keep track of max number of working threads

    ushort  listSize1 = 0;                  // Size of list being added from
    ushort  listSize2 = list_size;          // Size of list being added to
    uint    tid1 = 0;                       // Used for location in smem of list being added from

    while(FLAG < blockDim.x)
    {
        if( ((lid+1)%l_offset == 0) && (idx < nLists) && (liveThreads > 0) )
        {
            tid1 = lid-l_offset/2;
            listSize1 = s_sizes[tid1];

            for(uint i=0; i<listSize1; i++)
            {
                if(tid1%2 == 0) {
                    mtfVal = sdata[(256+MTF_PER_THREAD)*(tid1/2)+i];
                } else {
                    mtfVal = sdata[(tid1+1)/2*(256+MTF_PER_THREAD) - 256 + i];
                }

                C_ID = mtfVal/32;
                C_bit = mtfVal%32;

                int bit_to_set = 1 << (31-C_bit);
                if( !((C[C_ID]<<C_bit) & 0x80000000) )
                {
                    C[C_ID] |= bit_to_set;
                    sdata[sdata_index+listSize2] = mtfVal;  // Add to list if value is not present in list
                    listSize2++;                            // Increment size of list being added to
                }
            }

            liveThreads = liveThreads/2;
            working = 1;

            // Update list size for next list
            s_sizes[lid] = listSize2;

        } else if(!done) {
            // Idle thread
            done = 1;
            atomicAdd(&FLAG, 1);
        }
        l_offset *= 2;
        __syncthreads();
    }

    if(working && ((lid+1)%(l_offset/4))==0)
    {
        // This thread was the last one to do any work
        // Update d_lists with it's reduced value
        for(ushort i=initListSize; i<listSize2; i++)
        {
            d_lists[256*idx+i] = sdata[sdata_index+i];
        }
        d_list_sizes[idx] = listSize2;
    }
#endif
}


/** @brief Second stage in MTF (Global reduction)
 * @param[in,out] d_lists      A pointer to the start of MTF lists.
 * @param[in,out] d_list_sizes An array storing the size of each MTF list.
 * @param[in]     offset       The offset during the reduction stage. Initialized to two.
 * @param[in]     tThreads     Total number of threads dispatched.
 * @param[in]     nLists       Total number of MTF lists.
 **/
__global__ void
mtf_GLreduction_kernel(uchar  * d_lists,
                       ushort * d_list_sizes,
                       uint     offset,
                       uint     tThreads,
                       uint     nLists)
{
#if (__CUDA_ARCH__ >= 200)
    __shared__ uchar sdata[256*MTF_THREADS_BLOCK];
    __shared__ int FLAG;

    uint idx = threadIdx.x + (blockIdx.x * blockDim.x);
    uint lid = threadIdx.x;

    if(lid == 0) {
        FLAG = 0;
    }
    __syncthreads();

    uint liveThreads = blockDim.x;
    uint l_offset = offset;
    uint listID2 = ((idx+1)*(l_offset)-1);
    uint listID1 = listID2 - l_offset/2;

    int C[8];
#pragma unroll
    for(int i=0; i<8; i++)
        C[i] = 0;

    uchar mtfVal = 0;
    uchar C_ID = 0;
    uchar C_bit = 0;

    ushort listSize1 = 0;
    ushort listSize2 = 0;
    ushort initListSize = 0;

    uchar done = 0;
    uchar working = 0;
    ushort loopCnt = 1;

    if( ((listID2+1)%l_offset == 0) && (listID2 < nLists) && (idx < tThreads) )
    {
        listSize1 = d_list_sizes[listID1];
        listSize2 = d_list_sizes[listID2];
        initListSize = listSize2;

        for(ushort i=0; i<listSize2; i++)
        {
            mtfVal = d_lists[256*listID2+i];
            sdata[256*lid+i] = mtfVal;

            C_ID = mtfVal/32;
            C_bit = mtfVal%32;

            int bit_to_set = 1 << (31-C_bit);
            C[C_ID] |= ( !((C[C_ID]<<C_bit) & 0x80000000) ) ? bit_to_set : 0;
        }

    }
    __syncthreads();

    while(FLAG < blockDim.x)
    {
        if( ((listID2+1)%l_offset == 0) && (liveThreads > 0) && (listID2 < nLists) && (idx < tThreads) )
        {
            listSize1 = d_list_sizes[listID2 - l_offset/2];

            for(uint i=0; i<listSize1; i++)
            {
                if(loopCnt>1)
                    mtfVal = sdata[256*(lid-loopCnt/2)+i];
                else
                    mtfVal = d_lists[256*listID1+i];

                C_ID = mtfVal/32;
                C_bit = mtfVal%32;

                int bit_to_set = 1 << (31-C_bit);
                if( !((C[C_ID]<<C_bit) & 0x80000000) )
                {
                    C[C_ID] |= bit_to_set;
                    sdata[256*(lid)+listSize2] = mtfVal;
                    listSize2++;
                }

            }

            l_offset *= 2;
            liveThreads = liveThreads/2;
            working = 1;
            loopCnt *= 2;

            // Update list size for next thread
            d_list_sizes[listID2] = listSize2;

        } else if(working && !done) {

            // This thread is done working
            // Update d_lists at end
            for(uint i=initListSize; i<listSize2; i++)
            {
                d_lists[256*listID2+i] = sdata[256*(lid)+i];
            }
            atomicAdd(&FLAG, 1);
            done = 1;
        } else if(!done) {
            // Idle thread
            atomicAdd(&FLAG, 1);
            done = 1;
        }
        __syncthreads();
    }
#endif
}

/** @brief Third stage in MTF (Global downsweep)
 * @param[in,out] d_lists      A pointer to the start of MTF lists.
 * @param[in,out] d_list_sizes An array storing the size of each MTF list.
 * @param[in]     offset       The offset during the reduction stage.
 * @param[in]     lastLevel    The limit to which offset can be set to.
 * @param[in]     nLists       Total number of MTF lists.
 * @param[in]     tThreads     Total number of threads dispatched.
 **/
__global__ void
mtf_GLdownsweep_kernel(uchar    *d_lists,
                       ushort   *d_list_sizes,
                       uint     offset,
                       uint     lastLevel,
                       uint     nLists,
                       uint     tThreads)
{
#if (__CUDA_ARCH__ >= 200)
    __shared__ uchar sdata[256*MTF_THREADS_BLOCK];

    uint idx = threadIdx.x + (blockIdx.x * blockDim.x);
    uint lid = threadIdx.x;

    uint l_offset = offset;
    uint listID = ((idx+1)*lastLevel*2-1-lastLevel);

    if( (listID >= lastLevel) && (listID < (nLists-1)) && ((listID+1)%lastLevel == 0) && (idx<tThreads) )
    {
        ushort listSize1 = d_list_sizes[listID-lastLevel];

        for(ushort i=0; i<listSize1; i++)
        {
            sdata[256*lid+i] = d_lists[256*(listID-lastLevel)+i];
        }
    }
    __syncthreads();

    while(l_offset>=lastLevel)
    {
        if( (((listID-lastLevel+1)%l_offset == 0) && ((listID-lastLevel+1)%(l_offset*2) != 0) &&
             (listID >= lastLevel+l_offset) && (listID < (nLists-1)) && (idx<tThreads)) ||
            ( (l_offset==lastLevel) &&  (listID >= lastLevel+l_offset) && (listID < (nLists-1)) &&
              (idx<tThreads) )
            )
        {
            int C[8];
#pragma unroll
            for(int i=0; i<8; i++)
                C[i] = 0;

            ushort listSize1 = 0;
            ushort listSize2 = 0;
            uchar mtfVal = 0;

            if(l_offset == lastLevel) 
            {
                listSize2 = d_list_sizes[listID-lastLevel];
                listSize1 = d_list_sizes[listID];
            }
            else 
            {
                listSize2 = d_list_sizes[listID-lastLevel-l_offset];
                listSize1 = d_list_sizes[listID-lastLevel];
            }

            for(uint i=0; i<listSize1; i++)
            {
                if(l_offset == lastLevel)
                    mtfVal = d_lists[256*listID+i];
                else
                    mtfVal = sdata[256*lid+i];

                uchar C_ID = mtfVal/32;
                uchar C_bit = mtfVal%32;

                C[C_ID] |= 1<<(31-C_bit);

            }

            for(uint i=0; i<listSize2; i++)
            {
                if(l_offset == lastLevel)
                    mtfVal = sdata[256*lid+i];
                else
                    mtfVal = sdata[256*(lid-l_offset/(2*lastLevel))+i];

                uchar C_ID = mtfVal/32;
                uchar C_bit = mtfVal%32;

                if( !((C[C_ID]<<C_bit) & 0x80000000) )
                {
                    C[C_ID] |= 1<<(31-C_bit);
                    if(l_offset == lastLevel) {
                        d_lists[256*listID+listSize1] = mtfVal;
                    } else {
                        sdata[256*(lid)+listSize1] = mtfVal;
                        d_lists[256*(listID-lastLevel)+listSize1] = mtfVal;
                    }
                    listSize1++;
                }
            }

            // Update list size for next thread
            if(l_offset == lastLevel) {
                d_list_sizes[listID] = listSize1;
            } else {
                d_list_sizes[listID-lastLevel] = listSize1;
            }

        }

        l_offset = l_offset/2;
        __syncthreads();

    }
#endif
}

/** @brief Compute final MTF lists and final MTF output
 * @param[in]     d_mtfIn      A char array of the input data stream to perform the MTF on.
 * @param[out]     d_mtfOut     A char array of the output with the transformed MTF string.
 * @param[in,out] d_lists      A pointer to the start of MTF lists.
 * @param[in]     d_list_sizes An array storing the size of each MTF list.
 * @param[in]     nLists       Total number of MTF lists.
 * @param[in]     offset       The offset during the reduction stage.
 * @param[in]     numElements  Total number of elements to perform the MTF on.
 **/
__global__ void
mtf_localscan_lists_kernel(const uchar * d_mtfIn,
                           uchar       * d_mtfOut,
                           uchar       * d_lists,
                           ushort      * d_list_sizes,
                           uint          nLists,
                           uint          offset,
                           uint          numElements)
{
#if (__CUDA_ARCH__ >= 200)
    uint idx = threadIdx.x + (blockIdx.x * blockDim.x);
    uint lid = threadIdx.x;

    __shared__ uchar shared[MTF_LIST_SIZE*MTF_THREADS_BLOCK*sizeof(uchar) + MTF_THREADS_BLOCK*sizeof(ushort)];
    __shared__ uchar s_mtfIn[MTF_THREADS_BLOCK*MTF_PER_THREAD];

    // Each thread has it's own MTF list, stored in sLists (each list is 256-bytes)
    uchar* sLists = (uchar*)shared;
    uchar* sMyList = (uchar*)&shared[MTF_LIST_SIZE*lid];

    // The number of elements present in each MTF list (max = 256)
    ushort* s_sizes = (ushort*)&sLists[MTF_LIST_SIZE*blockDim.x*sizeof(uchar)];

    __shared__ int FLAG;

    if(lid == 0) {
        FLAG = 0;
    }
    __syncthreads();

    int C[8];
#pragma unroll
    for(int i=0; i<8; i++)
        C[i] = 0;

    ushort list_size = 0;
    
    uchar mtfVal;
    uchar C_ID;
    uchar C_bit;

    // There is no unique MTF list at the start of the input sequence
    // The MTF list at the beg is simply all characters placed in order, e.g. [0, 1, 2, .. 255]
    if(idx==0)
        s_sizes[0] = 0;

    // Use this are a temporary storage for C0, C1, etc.
    int* tmpPreviouslySeen = (int*)&s_mtfIn[0];

    // Clear tmpPreviouslySeen[] + s_sdata_index[]
    for(int i=lid; i<(MTF_THREADS_BLOCK*MTF_PER_THREAD/4); i += blockDim.x)
        tmpPreviouslySeen[i] = 0;
    __syncthreads();

    for(int tid = 0; tid < blockDim.x; tid++)
    {
        if(tid == 0 && blockIdx.x == 0) continue;

        uint sd_tid = 0;
        int* C = &tmpPreviouslySeen[tid*8];
        ushort tid_list_size = d_list_sizes[tid+blockIdx.x*blockDim.x-1];
        if(lid == tid)
        {
            list_size = tid_list_size;
            s_sizes[lid] = tid_list_size;
        }
        
        for(int i = (int)lid; i < tid_list_size; i += blockDim.x)
        {
            mtfVal = d_lists[(tid+blockIdx.x*blockDim.x-1)*256+i];
            if(i < MTF_LIST_SIZE) {
                sd_tid++;
                sLists[tid*MTF_LIST_SIZE+i] = mtfVal;
            }

            // Finds which bit needs to be set to '1'
            C_ID = mtfVal/32;
            C_bit = mtfVal%32;

            int bit_to_set = 1 << (31-C_bit);
            atomicOr(&C[C_ID], bit_to_set);
        }
    }

    __syncthreads();

    // Copy tmpPreviouslySeen into respective registers
    if(idx>0)
    {
#pragma unroll
        for(int i=0; i<8; i++)
            C[i] = tmpPreviouslySeen[lid*8+i];
    }

    //===================================================================================================
    //                                      MTF Local Scan
    // Each thread now has a copy of its own list. From prior steps, we have also computed the MTF list
    // at block intervals. Using these lists, we are able to construct each thread's MTF list using 
    // a local parallel scan (Reduction + "down-sweep" phases).
    //===================================================================================================

    //==========================
    //  Local Reduction Phase
    //==========================

    uint l_offset = offset;
    uchar done = 0;

    while(FLAG < blockDim.x)
    {
        if(idx<nLists && (lid+1)%l_offset==0) {

            int add_threadId = lid-l_offset/2;                  // The ID of the list we are appending to current list
            ushort add_size = s_sizes[add_threadId];            // The size of the list we are appending to current list
            int add_data_index = add_threadId*MTF_LIST_SIZE;    // The location of the list we are appending to current list

            // Start appending previous MTF list with current MTF list
            // We only append the characters that are not
            // present in our the list we are appending to
            for(int i=0; i<(int)add_size; i++) {
                mtfVal = (i<MTF_LIST_SIZE) ?
                    sLists[add_data_index+i] : d_lists[256*(add_threadId+blockIdx.x*blockDim.x-1)+i];

                C_ID = mtfVal/32;
                C_bit = mtfVal%32;

                int bit_to_set = 1 << (31-C_bit);
                if(!((C[C_ID]<<C_bit) & 0x80000000))
                { 
                    C[C_ID] |= bit_to_set;
                    if(list_size < MTF_LIST_SIZE) {
                        sMyList[list_size] = mtfVal;      // Append to current MTF list if value is not present in the list
                     } else if(idx>0)
                        d_lists[256*(idx-1)+list_size] = mtfVal;
                    list_size++;
                }
            }

            // Update the size of current MTF list
            // current size += sizeof(list that was appended)
            s_sizes[lid] = list_size;

        } else if(!done) {
            atomicAdd(&FLAG, 1);
            done = 1;
        }

        l_offset *= 2;
        __syncthreads();
    }

    //==========================================================================
    //                  Local "Down-Sweep" Phase
    // Now we are done with the reduction phase. We now continue to build our
    // final MTF lists by performing the down-sweep phase.
    //==========================================================================

    l_offset = l_offset/16;
    while(l_offset>0)
    {
        if(lid>=l_offset && (lid+1)%l_offset==0 && (lid+1)%(l_offset*2)!=0) {

            int add_threadId = lid-l_offset;                // The ID of the list we are appending to current list
            ushort add_size = s_sizes[add_threadId];        // The size of the list we are appending to current list
            int add_data_index = add_threadId*MTF_LIST_SIZE;    // The location of the list we are appending to current list
            for(int i=0; i<add_size; i++) {

                mtfVal = (i<MTF_LIST_SIZE) ?
                    sLists[add_data_index+i] : d_lists[256*(add_threadId+blockIdx.x*blockDim.x-1)+i];
                
                C_ID = mtfVal/32;
                C_bit = mtfVal%32;

                int bit_to_set = 1 << (31-C_bit);

                if(!((C[C_ID]<<C_bit) & 0x80000000))
                {
                    if(list_size < MTF_LIST_SIZE) {
                        sMyList[list_size] = mtfVal;      // Append to current MTF list if value is not present in the list
                    } else if (idx>0)
                        d_lists[256*(idx-1)+list_size] = mtfVal;
                    list_size++;
                    C[C_ID] |= bit_to_set;
                }
            }

            // Update the size of current MTF list
            // current size += sizeof(list that was appended)
            s_sizes[lid] = list_size;

        }

        l_offset = l_offset/2;
        __syncthreads();
    }


    // Read in d_mtfIn to s_mtfIn
    for(int i = 0; i < MTF_PER_THREAD; i++)
    {
        // Coalesced reads
        int index = blockIdx.x*MTF_PER_THREAD*MTF_THREADS_BLOCK + lid+MTF_THREADS_BLOCK*i;

	if(lid+MTF_THREADS_BLOCK*i < MTF_THREADS_BLOCK*MTF_PER_THREAD)
        s_mtfIn[lid+MTF_THREADS_BLOCK*i] = (index<numElements) ? d_mtfIn[index] : 0;
    }
    __syncthreads();
    //========================================================================
    //                      Final MTF
    // Done computing each MTF list. Now, compute final MTF values
    // using these MTF lists. Each MTF list can be a maximum of 256 chars
    //========================================================================

    if(idx < nLists)
    {
        for(int i=0; i<MTF_PER_THREAD; i++)
        {
            uchar mtfOut = 0;
            bool found = false;

            // Read next MTF input
            mtfVal = s_mtfIn[lid*MTF_PER_THREAD + i];
            C_ID = mtfVal/32;
            C_bit = mtfVal%32;

            int bit_to_set = 1 << (31-C_bit);
            if( !((C[C_ID]<<C_bit) & 0x80000000) ) C[C_ID] |= bit_to_set;
            else found = true;
	    if(mtfVal==37){
}
            if(found)
            { 
                // Element already exists in list, Moving to front
                uint tmp1 = mtfVal;
                uint tmp2;
                for(int j=0; j<(int)list_size; j++)
                {
                    if(j < MTF_LIST_SIZE) {
                        tmp2 = sMyList[j];
                        sMyList[j] = tmp1; 
		    }else if(idx>0){
                           tmp2 = d_lists[256*(idx-1)+j];
                           d_lists[256*(idx-1)+j] = tmp1; 
		    }

                    tmp1 = tmp2;
                    if(tmp1 == mtfVal) {
                        mtfOut = j;
                        break;
                    }
                }

            }
            else
	    {
                // Adding new element to front of the list, shift all other elements
                uchar greater_cnt = 0;
                for(int j=(int)list_size; j>0; j--)
                {
                    if(j > MTF_LIST_SIZE) {
		        if(idx>0){   
                        d_lists[256*(idx-1)+j] = d_lists[256*(idx-1)+j-1];
                        if(d_lists[256*(idx-1)+j] > mtfVal) greater_cnt++;
			}
                    } else if(j == MTF_LIST_SIZE) {
		        if(idx>0) 
                        d_lists[256*(idx-1)+j] = sMyList[j-1]; // Shifting elements

                        if(sMyList[j-1] > mtfVal) greater_cnt++;
                    } else if(j < MTF_LIST_SIZE) {
                        sMyList[j] = sMyList[j-1]; // shifting elements
                        if(sMyList[j] > mtfVal) greater_cnt++;
                    }
                }
                sMyList[0] = mtfVal;
                list_size++;
                mtfOut = mtfVal+greater_cnt;
            }

            // Write MTF output
            s_mtfIn[lid*MTF_PER_THREAD + i] = mtfOut;
        }
        
    }

    __syncthreads();

    for(int i = 0; i < MTF_PER_THREAD; i++)
    {
        // Coalesced writes
        int index = blockIdx.x*MTF_PER_THREAD*MTF_THREADS_BLOCK + lid+MTF_THREADS_BLOCK*i;
        if(index<numElements)
            d_mtfOut[index] = s_mtfIn[lid+MTF_THREADS_BLOCK*i];
    }
#endif
}








/** @brief Compute 256-entry histogram
 * @param[in]  d_input      An array of words we will use to build our histogram.
 * @param[out] d_histograms A pointer where we store our global histograms.
 * @param[in]  numElements  The total number of elements to build our histogram from.
 **/
__global__ void
huffman_build_histogram_kernel(uint     *d_input, // Read in as words, instead of bytes
                               uint     *d_histograms,
                               uint     numElements)
{
#if (__CUDA_ARCH__ >= 200)
    // Per-thread Histogram - Each "bin" will be 1 byte
    __shared__ uchar threadHist[HUFF_THREADS_PER_BLOCK_HIST*256]; // Each thread has 256 1-byte bins

    uint* blockHist = &d_histograms[blockIdx.x*256];

    // Global, local IDs
    uint lid = threadIdx.x;

    // Clear thread histograms
    for(uint i=lid; i<HUFF_THREADS_PER_BLOCK_HIST*256; i += blockDim.x)
    {
        threadHist[i] = 0;
    }

    // Clear block histogram
    for(uint i=lid; i<256; i += blockDim.x)
    {
        blockHist[i] = 0;
    }
    __syncthreads();

    uint wordsPerThread = HUFF_WORK_PER_THREAD_HIST/4;   // We can change this later, possibly increase?

    for(uint i=0; i<wordsPerThread; i++)
    {
        // Coalesced reads
        uint word = d_input[blockIdx.x*wordsPerThread*HUFF_THREADS_PER_BLOCK_HIST + lid+HUFF_THREADS_PER_BLOCK_HIST*i];

        // Extract each byte from word
        uchar bin1 = (uchar)(word & 0xff);
        uchar bin2 = (uchar)((word >> 8) & 0xff);
        uchar bin3 = (uchar)((word >> 16) & 0xff);
        uchar bin4 = (uchar)((word >> 24) & 0xff);

        // Update thread histograms + spill to shared if overflowing
        if(threadHist[lid*256+bin1]==255)
        {
            atomicAdd(&blockHist[bin1], 255);
            threadHist[lid*256+bin1] = 0;
        }
        threadHist[lid*256+bin1]++;

        if(threadHist[lid*256+bin2]==255)
        {
            atomicAdd(&blockHist[bin2], 255);
            threadHist[lid*256+bin2] = 0;
        }
        threadHist[lid*256+bin2]++;

        if(threadHist[lid*256+bin3]==255)
        {
            atomicAdd(&blockHist[bin3], 255);
            threadHist[lid*256+bin3] = 0;
        }
        threadHist[lid*256+bin3]++;

        if(threadHist[lid*256+bin4]==255)
        {
            atomicAdd(&blockHist[bin4], 255);
            threadHist[lid*256+bin4] = 0;
        }
        threadHist[lid*256+bin4]++;
    }

    __syncthreads();

    // Merge thread histograms into a block histogram
    for(uint i=0; i<(256/HUFF_THREADS_PER_BLOCK_HIST); i++)
    {
        uint count = 0;
        for(uint j=0; j<HUFF_THREADS_PER_BLOCK_HIST; j++)
        {
            count += threadHist[lid*(256/HUFF_THREADS_PER_BLOCK_HIST)+i + j*256];
        }

        blockHist[lid*(256/HUFF_THREADS_PER_BLOCK_HIST)+i] += count;
    }
#endif
}

/* Compute 256-entry histogram of an array of char
   d_input An array of chars
   d_histograms A pointer where we store our global histograms
   numElements The total number of elements to build the histogram
*/

__global__ void
histo_kernel(uchar *d_input,
             uint  *d_histograms,
             uint  numElements)
{
#if (__CUDA_ARCH__ >= 200)
    // Per-thread Histogram
    __shared__ uchar threadHist[HUFF_THREADS_PER_BLOCK_HIST*256]; // Eash thread has 256 1-byte bins

    uint* blockHist = &d_histograms[blockIdx.x*256];

    // Global, local IDs
    uint lid = threadIdx.x;

    //Clear thread histograms
    for(uint i=lid; i<HUFF_THREADS_PER_BLOCK_HIST*256; i+=blockDim.x) //64 threads per block
    {
	threadHist[i]=0;
    }

    //Clear block histogram
    for(uint i=lid; i<256; i+= blockDim.x)
    {
	blockHist[i]=0;
    }
    __syncthreads();

    // Update thread histograms + spill to shared if overflowing
    for(uint i=0; i<HUFF_WORK_PER_THREAD_HIST;++i)
    {
	uchar word=d_input[lid+blockIdx.x*HUFF_THREADS_PER_BLOCK_HIST*256*HUFF_WORK_PER_THREAD_HIST+i*HUFF_THREADS_PER_BLOCK_HIST];
	if(threadHist[lid*256+word]==255)
        {
	   atomicAdd(&blockHist[word], 255);
           threadHist[lid*256+word]=0;
	}
	threadHist[lid*256+word]++;

    }

    __syncthreads();

    // Merge thread histograms into a block histogram
    for(uint i=0; i<(256/HUFF_THREADS_PER_BLOCK_HIST); ++i)
    {
	uint count = 0;
	for(uint j=0; j<HUFF_THREADS_PER_BLOCK_HIST; ++j)
	{
	    count += threadHist[lid*(256/HUFF_THREADS_PER_BLOCK_HIST) + i + j*256];
	}	
	
	blockHist[lid*(256/HUFF_THREADS_PER_BLOCK_HIST)+i] += count;
    }
#endif
}


/** @brief Build Huffman tree/codes
 * @param[in] d_input               An array of input elements to encode
 * @param[out] d_huffCodesPacked    An array of huffman bit codes packed together
 * @param[out] d_huffCodeLocations  An array which stores the starting bit locations of each
                                    Huffman bit code
 * @param[out] d_huffCodeLengths    An array which stores the lengths of each Huffman bit code
 * @param[in] d_histograms          An input array of histograms to combine
 * @param[out] d_histogram          Final histogram combined
 * @param[out] d_nCodesPacked       Number of chars it took to store all Huffman bit codes
 * @param[out] d_totalEncodedSize   Total number of words it takes to hold the compressed data
 * @param[in] histBlocks            Total number of histograms we will combine into one
 * @param[in] numElements           Number of elements to compress
 **/
__global__ void
huffman_build_tree_kernel(const uchar *d_input,
                          uchar     *d_huffCodesPacked,
                          uint      *d_huffCodeLocations,
                          uchar     *d_huffCodeLengths,
                          uint      *d_histograms,
                          uint      *d_histogram,
                          uint      *d_nCodesPacked,
                          uint      *d_totalEncodedSize,
                          uint      histBlocks,
                          uint      numElements)
{
#if (__CUDA_ARCH__ >= 200)
    // Global, local IDs
    uint idx = threadIdx.x + (blockIdx.x * blockDim.x);
    uint lid = threadIdx.x;

    __shared__ uint histogram[HUFF_NUM_CHARS];
    __shared__ my_huffman_node_t h_huffmanArray[HUFF_NUM_CHARS*2-1];

    // Used during building of Huffman codes
    __shared__ huffman_code tempCode;
    __shared__ huffman_code tempCode2;

    // Huffman codes packed together + lengths
    __shared__ uchar s_codesPacked[(HUFF_NUM_CHARS*(HUFF_NUM_CHARS+1)/2)/8+1]; // Array size [4145] -- estimating the average bit code is no more than 8 bits
    __shared__ uchar s_codeLengths[HUFF_NUM_CHARS];

    // Set codesPacked and codeLocations to 0
    uint workPerThread = (idx == (blockDim.x-1) ) ? 81 : 32; // Only works for 128 threads

    if(idx==0)
    {
        *d_nCodesPacked = 0; // Only works when 1 thread block is present
        *d_totalEncodedSize = 0;
    }

    for(uint i = lid; i < (HUFF_NUM_CHARS*(HUFF_NUM_CHARS+1)/2)/8+1; i += blockDim.x)
    {
        s_codesPacked[i] = 0;
    }

    __syncthreads();

    // Set codes to 0
    if(idx==0)
    {
        for(int j=0; j<HUFF_NUM_CHARS; j++)
        {
            histogram[j] = 0;
            s_codeLengths[j] = 0;
        }
        histogram[HUFF_EOF_CHAR] = 1;

        for(int j=0; j<HUFF_NUM_CHARS; j++)
        {
            h_huffmanArray[j].iter = (uint)j;
            h_huffmanArray[j].value = j;
            h_huffmanArray[j].ignore = HUFF_TRUE;
            h_huffmanArray[j].count = 0;
            h_huffmanArray[j].level = 0;
            h_huffmanArray[j].left = -1;
            h_huffmanArray[j].right = -1;
            h_huffmanArray[j].parent = -1;
        }
        for(int j=HUFF_NUM_CHARS; j<HUFF_NUM_CHARS*2-1; j++)
        {
            h_huffmanArray[j].iter = (uint)j;
            h_huffmanArray[j].value = 0;
            h_huffmanArray[j].ignore = HUFF_TRUE;
            h_huffmanArray[j].count = 0;
            h_huffmanArray[j].level = 0;
            h_huffmanArray[j].left = -1;
            h_huffmanArray[j].right = -1;
            h_huffmanArray[j].parent = -1;
        }

    }
    __syncthreads();

    // Merge block histograms into final histogram
    workPerThread = 256/blockDim.x;
    for(uint i=0; i<workPerThread; i++)
    {
        uint count = 0;
        for(uint j=0; j<histBlocks; j++)
        {
            count += d_histograms[lid+(blockDim.x)*i + j*256]; // Coalesced Reads
        }
        histogram[lid+(blockDim.x)*i] += count;
    }

    __syncthreads();

    // Update global histogram (used during decode)
    for(int i=idx; i<256; i+=blockDim.x)
        d_histogram[i] = histogram[i];

    //--------------------------------------------------------------------------
    //          Huffman tree: Build the Huffman tree
    //--------------------------------------------------------------------------

    __shared__ uint nNodes;
    __shared__ int min1, min2; // two nodes with the lowest count
    __shared__ int head_node; // root of tree
    __shared__ int current_node; // location on the tree

    if(idx == 0)
    {
        nNodes = 0;
        min1 = HUFF_NONE;
        min2 = HUFF_NONE;

        for(int j=0; j<(HUFF_NUM_CHARS); j++)
        {
            if(histogram[j] > 0)
            {
                h_huffmanArray[nNodes].count = histogram[j];
                h_huffmanArray[nNodes].ignore = 0;
                h_huffmanArray[nNodes].value = j;
                nNodes++;
            }
        }
    }

    __threadfence();
    __syncthreads();

    if(idx == 0)
    {
        // keep looking until no more nodes can be found
        for (;;)
        {
            // find node with lowest count
            min1 = FindMinimumCount(&h_huffmanArray[0], nNodes);
            if (min1 == HUFF_NONE) break; // No more nodes to combine

            h_huffmanArray[min1].ignore = 1; // remove from consideration

            // find node with second lowest count
            min2 = FindMinimumCount(&h_huffmanArray[0], nNodes);
            if (min2 == HUFF_NONE) break; // No more nodes to combine

            // Move min1 to the next available slot

            h_huffmanArray[min1].ignore = 0;
            uchar min1_replacement = 0;

            for(int i = (int)nNodes; i<(HUFF_NUM_CHARS*2-1); i++)
            {
                if(h_huffmanArray[i].count==0)
                {
                    // Found next available slot
                    h_huffmanArray[i] = h_huffmanArray[min1];
                    h_huffmanArray[i].iter = (uint)i;
                    h_huffmanArray[i].ignore = 1;
                    h_huffmanArray[i].parent = h_huffmanArray[min1].iter;

                    if(h_huffmanArray[i].left >= 0) {
                        h_huffmanArray[h_huffmanArray[i].left].parent = i;
                    }
                    if(h_huffmanArray[i].right >= 0) {
                        h_huffmanArray[h_huffmanArray[i].right].parent = i;
                    }
                    h_huffmanArray[min1].left = i;
                    min1_replacement = 1;
                    break;
                }
            }

            if(min1_replacement == 0)
            {
                printf("ERROR: Tree size too small\n");
                break;
            }

            h_huffmanArray[min2].ignore = 1;

            // Combines both nodes into composite node
            h_huffmanArray[min1].value = HUFF_COMPOSITE_NODE;
            h_huffmanArray[min1].ignore = 0;
            h_huffmanArray[min1].count = h_huffmanArray[min1].count + h_huffmanArray[min2].count;
            h_huffmanArray[min1].level = max(h_huffmanArray[min1].level, h_huffmanArray[min2].level) + 1;

            h_huffmanArray[min1].right = h_huffmanArray[min2].iter;
            h_huffmanArray[min2].parent =  h_huffmanArray[min1].iter;
            h_huffmanArray[min1].parent = -1;

        }

        // 'head_node' tells us which node in the h_huffmanArray
        // represents the head node of the Huffman tree (Starting point)
        head_node = min1;
    }
    __threadfence();
    __syncthreads();


    //--------------------------------------------------------------------------
    //      Huffman Codes: Use the tree that is build to create
    //                     the Huffman codes -- MakeCodeList() on the CPU
    //      Store huff. codes in codes[] and length in code_lengths[]
    //--------------------------------------------------------------------------

    if(idx == 0)
    {
        uint writeBit = 0; // Use for d_huffCodesPacked
        uchar buffer = 8;
        uint section = 0;
        uint offset = 0;
        uint numBits = 0;

        uchar depth = 0;
        current_node = head_node;
        tempCode.numBits = 256;
        memset(&tempCode.code, 0, 32*sizeof(uchar)); // clear code

        for(;;)
        {
            // Follow tree branch all of the way to the left
            // Going left = 0, Going right = 1

            while (h_huffmanArray[current_node].left != -1)
            { // Moving Left
                BitArrayShiftLeft(&tempCode, 1);
                current_node = h_huffmanArray[current_node].left;
                depth++;
            }

            if (h_huffmanArray[current_node].value != HUFF_COMPOSITE_NODE)
            { // Enter results in list
                memcpy((void *)(&tempCode2.code[0]), (void *)(&tempCode.code[0]), BITS_TO_CHARS(tempCode.numBits));
                tempCode2.numBits = tempCode.numBits;

                // Left justify code
                BitArrayShiftLeft(&tempCode2, 256 - depth);

                // Pack Huffman codes into char array
                s_codeLengths[h_huffmanArray[current_node].value] = depth;
                d_huffCodeLocations[h_huffmanArray[current_node].value] = writeBit;
                writeBit += depth;
                offset = 0;
                numBits = depth;


                while(numBits >= 8)
                {
                    if(buffer < numBits) {
                        s_codesPacked[section] |= tempCode2.code[offset] >> (8-buffer);
                        section++;
                        s_codesPacked[section] |= tempCode2.code[offset] << (buffer);
                    } else {
                        s_codesPacked[section] |= tempCode2.code[offset];
                        section++;
                    }
                    numBits -= 8;
                    offset++;
                }

                if(numBits > 0)
                {
                    uchar tmp = tempCode2.code[offset];
                    while(numBits > 0)
                    {
                        s_codesPacked[section] |= (tmp&0x80) >> (8-buffer);
                        tmp <<= 1;
                        buffer--;
                        numBits--;
                        if(buffer == 0) {
                            buffer = 8;
                            section++;
                        }
                    }
                }

            }

            while (h_huffmanArray[current_node].parent != -1)
            {
                if (current_node != h_huffmanArray[h_huffmanArray[current_node].parent].right)
                { // try the parent's right
                    BitArraySetBit(&tempCode, 255);
                    current_node = h_huffmanArray[h_huffmanArray[current_node].parent].right;
                    break;
                }
                else
                { // parent's right tried, go up one level yet
                    depth--;
                    BitArrayShiftRight(&tempCode, 1);
                    current_node = h_huffmanArray[current_node].parent;
                }
            }

            if (h_huffmanArray[current_node].parent == -1)
            { // we're at the top with nowhere to go
                break;
            }
        }

        *d_nCodesPacked = (writeBit%8 == 0) ? (writeBit/8) : (writeBit/8+1);

        for(int i=0; i <= section; i++)
        {
            d_huffCodesPacked[i] = s_codesPacked[i];
        }

        for(uint i=0; i<HUFF_NUM_CHARS; i++)
        {
            if(histogram[i] > 0) d_huffCodeLengths[i] = s_codeLengths[i];
        }

    }    
#endif
}

/** @brief Perform parallel Huffman encoding
 * @param[in] d_input           Input array to encode
 * @param[in] d_codes           Array of packed Huffman bit codes
 * @param[in] d_code_locations  Array of starting Huffman bit locations
 * @param[in] d_huffCodeLengths An array storing the bit lengths of the Huffman codes
 * @param[out] d_encoded        An array of encoded classes which stores the size and data of
                                encoded data
 * @param[in] nCodesPacked      Number of chars it took to store all Huffman bit codes
 * @param[in] nThreads          Total number of dispatched threads
 **/
__global__ void
huffman_kernel_en(uchar4    *d_input,              // Input to encode
                  uchar     *d_codes,               // Packed Huffman Codes
                  uint      *d_code_locations,       // Location of each huffman code
                  uchar     *d_huffCodeLengths,
                  encoded   *d_encoded,
                  uint      nCodesPacked,
                  uint      nThreads)
{
#if (__CUDA_ARCH__ >= 200)
    // Global, local IDs
    uint idx = threadIdx.x + (blockIdx.x * blockDim.x);
    uint lid = threadIdx.x;

    // Shared mem setup
    extern __shared__ uchar s_codes[];              // Huffman codes
    __shared__ uint s_locations[HUFF_NUM_CHARS];         // Which bit to find the huffman codes
    __shared__ uint s_codeLengths[HUFF_NUM_CHARS];

    __shared__ uchar4 s_input[HUFF_THREADS_PER_BLOCK*HUFF_WORK_PER_THREAD/sizeof(uchar4)];    // Input data to encode
    __shared__ uint s_write_locations[HUFF_THREADS_PER_BLOCK];                           // Starting write location for each thread
    __shared__ uint s_encoded[HUFF_THREADS_PER_BLOCK*HUFF_WORK_PER_THREAD/4];                   // Encoded data
    __shared__ uint total_bits_block;

    encoded* my_encoded = (encoded*)&d_encoded[blockIdx.x];

    //-----------------------------------------------
    //    Copy data from global to shared memory
    //-----------------------------------------------

    s_write_locations[lid] = 0;
    uint total_bits = 0; // Keep track of the total number of bits each thread encodes

    // Store the packed codes into shared memory
    if(lid == 0) {
        total_bits_block = 0;
        my_encoded->block_size = 0;
    }

#pragma unroll
    for(uint i = lid; i < HUFF_CODE_BYTES; i += blockDim.x)
        my_encoded->code[i] = 0;

#pragma unroll
    for(uint i = lid; i < nCodesPacked; i += blockDim.x)
        s_codes[i] = d_codes[i];

    // Store code locations into shared memory
#pragma unroll
    for(uint i = lid; i < HUFF_NUM_CHARS; i += blockDim.x) {
        s_locations[i] = d_code_locations[i];
        s_codeLengths[i] = d_huffCodeLengths[i];
    }

    __syncthreads();

    // Clear encoded-data array in shared memory
    for(uint i=lid; i<HUFF_THREADS_PER_BLOCK*HUFF_WORK_PER_THREAD/4; i += blockDim.x)
        s_encoded[i] = 0;

    // Store un-encoded input data and calculate starting write-locations for each thread into shared memory
    if(idx < nThreads)
    {
        for(uint i=0; i<(HUFF_WORK_PER_THREAD/sizeof(uchar4)); i++)
        {
            uchar4 val = d_input[idx*(HUFF_WORK_PER_THREAD/sizeof(uchar4))+i];

            s_input[lid*(HUFF_WORK_PER_THREAD/sizeof(uchar4))+i].x = val.x;
            s_input[lid*(HUFF_WORK_PER_THREAD/sizeof(uchar4))+i].y = val.y;
            s_input[lid*(HUFF_WORK_PER_THREAD/sizeof(uchar4))+i].z = val.z;
            s_input[lid*(HUFF_WORK_PER_THREAD/sizeof(uchar4))+i].w = val.w;

            s_write_locations[lid] += s_codeLengths[val.x];
            total_bits += s_codeLengths[val.x];

            s_write_locations[lid] += s_codeLengths[val.y];
            total_bits += s_codeLengths[val.y];

            s_write_locations[lid] += s_codeLengths[val.z];
            total_bits += s_codeLengths[val.z];

            s_write_locations[lid] += s_codeLengths[val.w];
            total_bits += s_codeLengths[val.w];

        }

        // Add up total bits of this block in shared memory
        atomicAdd(&total_bits_block, total_bits);
    }

    __syncthreads();

    // Write the total amount of 4-bytes for this block
    if(lid==0)
        my_encoded->block_size = (total_bits_block%32==0) ? total_bits_block/32 : total_bits_block/32+1;

    //-------------------------------------------------------------------
    //  SCAN - Need to perform a scan operation on 's_write_locations' 
    //         to determine where to start writing the encoded data
    //-------------------------------------------------------------------

    if(lid==0)
    {
        uint sum = s_write_locations[0];
        for(uint i=1; i<HUFF_THREADS_PER_BLOCK; i++)
        {
            sum += s_write_locations[i];
            s_write_locations[i] = sum;
        }
    }

    __syncthreads();


    //--------------------------
    //     Huffman Encoding
    //--------------------------

    if(idx < nThreads)
    {
        uint WR = 0;
        uint my_write_loc = (lid==0) ? 0 : s_write_locations[lid-1];
        uint write_block = my_write_loc/32;
        uint write_bit = my_write_loc%32;

        for(uint i=0; i<(HUFF_WORK_PER_THREAD/sizeof(uchar4)); i++)
        {
            uchar val[4];
            val[0] = s_input[lid*(HUFF_WORK_PER_THREAD/sizeof(uchar4))+i].x;
            val[1] = s_input[lid*(HUFF_WORK_PER_THREAD/sizeof(uchar4))+i].y;
            val[2] = s_input[lid*(HUFF_WORK_PER_THREAD/sizeof(uchar4))+i].z;
            val[3] = s_input[lid*(HUFF_WORK_PER_THREAD/sizeof(uchar4))+i].w;

            for(uchar j=0; j<4; j++)
            {
                uint CodeLen = s_codeLengths[val[j]];

                uint CodeBlock = s_locations[val[j]]/8;
                uint CodeOffset = s_locations[val[j]]%8+24;
                uint Code = (uint)s_codes[CodeBlock];

                // Encoding
                for(uint k=0; k<CodeLen; k++)
                {
                    if(write_bit == 32) 
                    {
                        write_bit = 0;
                        if(write_block < (HUFF_THREADS_PER_BLOCK*HUFF_WORK_PER_THREAD/4)) {
                            atomicOr(&s_encoded[write_block], WR);
                        } else {
                            atomicOr(&my_encoded->code[write_block], WR);
                        }
                        WR = 0;
                        write_block++;
                    }

                    if(CodeOffset == 32)
                    {
                        CodeOffset = 24;
                        CodeBlock++;
                        Code = (uint)s_codes[CodeBlock];
                    }

                    WR |= ((Code<<CodeOffset)&0x80000000) >> write_bit;
                    write_bit++;
                    CodeOffset++;

                }
            }
        }
        if(write_bit > 0) 
        {
            if(write_block < (HUFF_THREADS_PER_BLOCK*HUFF_WORK_PER_THREAD/4)) {
                atomicOr(&s_encoded[write_block], WR);
            } else {
                atomicOr(&my_encoded->code[write_block], WR);
            }
        }
    }
    __syncthreads();

    for(int i=lid; i<(HUFF_THREADS_PER_BLOCK*HUFF_WORK_PER_THREAD/4); i += blockDim.x)
        my_encoded->code[i] = s_encoded[i];
#endif
}

/** @brief Pack together encoded blocks.
 * @param[in] d_encoded An array of encoded objects with stored size and data of the encoded data.
 * @param[out] d_encodedData An in array to store all encoded data.
 * @param[out] d_totalEncodedSize Total number words of the encoded data.
 * @param[out] d_eOffsets Array holding the word offsets of each encoded data block.
 **/
__global__ void
huffman_datapack_kernel(encoded     *d_encoded,
                        uint        *d_encodedData,
                        uint        *d_totalEncodedSize,
                        uint        *d_eOffsets)
{
#if (__CUDA_ARCH__ >= 200)
    // Global, local IDs
    uint lid = threadIdx.x;

    __shared__ uint prevWords;
    encoded* my_encodedData = (encoded*)&d_encoded[blockIdx.x];
    uint nWords = my_encodedData[0].block_size;

    if(lid==0)
    {
        prevWords = 0;
        for(uint i=0; i<blockIdx.x; i++) {
            prevWords += 1+d_encoded[i].block_size;
        }
        d_eOffsets[blockIdx.x] = prevWords;
        d_encodedData[prevWords] = my_encodedData[0].block_size;
        atomicAdd(&d_totalEncodedSize[0], 1+nWords);
    }

    __syncthreads();

    uint j = lid;
    while(j<nWords)
    {
        d_encodedData[prevWords+1+j] = my_encodedData->code[j];
        j += blockDim.x;
    }
#endif
}

/** @} */ // end compress functions
/** @} */ // end cudpp_kernel

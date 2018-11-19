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
#include "cudpp_mergesort.h"
#include <cudpp.h>
#include <stdio.h>

#include <cudpp_util.h>
#include <math.h>
#include "sharedmem.h"

/**
 * @file
 * sort_cta.cu
 * 
 * @brief CUDPP CTA-level sort routines
 */

/** \addtogroup cudpp_cta 
* @{
*/

/** @name Merge Sort Functions
* @{
*/

#define BLOCKSORT_SIZE 1024
#define CTA_BLOCK 128
#define DEPTH_simple 2
#define DEPTH_multi 4
#define CTASIZE_simple 256
#define CTASIZE_multi 128

#define INTERSECT_A_BLOCK_SIZE_simple DEPTH_simple*CTASIZE_simple
#define INTERSECT_B_BLOCK_SIZE_simple 2*DEPTH_simple*CTASIZE_simple

#define INTERSECT_A_BLOCK_SIZE_multi DEPTH_multi*CTASIZE_multi
#define INTERSECT_B_BLOCK_SIZE_multi 2*DEPTH_multi*CTASIZE_multi
typedef unsigned int uint;

/** @brief Binary search within a single block (blockSort)
 * @param[in,out] cmpValue Value being considered from other partition
 * @param[in] tmpVal My Value
 * @param[in] in input keys
 * @param[in,out] j The index we are considering
 * @param[in] bump The offset we update by
 * @param[in] addPart Tie break (left partition vs right partition) 
 **/
template<class T, int depth>
__device__ void bin_search_block(T &cmpValue, T tmpVal, T* in, unsigned int & j, unsigned int bump, unsigned int addPart)
{

	cmpValue = in[j]; 

    j = ((cmpValue < tmpVal || cmpValue == tmpVal && addPart == 1) ? j + bump : j - bump);  
    __syncthreads();

}
/** @brief Linear search within a single block (blockSort)
 * @param[in,out] cmpValue Value being considered from other partition
 * @param[in] mVal Value in our partition
 * @param[in,out] tmpVal Temporary register which is used to store the final address after our search
 * @param[in] in, addressPad, in = keys and addressPad = values
 * @param[in] j index in B partition we are considering
 * @param[in] offset Since this is register packed, offset is the ith iteration of linear search
 * @param[in] last The end of partition B we are allowed to look upto
 * @param[in] startAddress The beginning of our partition 
 * @param[in] addPart Tie break (left partition vs right partition) 
 **/
			
template<class T, int depth>
__device__ void lin_search_block(T &cmpValue, T mVal, unsigned int &tmpVal, T* in, unsigned int* addressPad, unsigned int &j, 
								 unsigned int offset, unsigned int last, unsigned int startAddress, unsigned int addPart)
{			
	
	while (cmpValue < mVal && j < last)		
		cmpValue = in[++j];			
	while (cmpValue == mVal && j < last && addPart == 1)
		cmpValue = in[++j];	
	
	//Corner case to handle being at the edge of our shared memory search
    j = (j==last && (cmpValue < mVal || (cmpValue == mVal && addPart == 1)) ? j+1 : j);	
	
    tmpVal = j+startAddress+offset;
}

/** @brief For blockSort. Compares two values and decides to swap if A1 > A2
 * @param[in,out] A1 First value being compared
 * @param[in,out] A2 Second value being compared
 * @param[in,out] ref1 Local address of A1
 * @param[in,out] ref2 Local address of A2                        
 **/
template<class T>
__device__ void compareSwapVal(T &A1, T &A2, unsigned int& ref1, unsigned int& ref2)
{
    if(A1 > A2)
    {
        T tmp = A1;
        A1 = A2;
        A2 = tmp;

        unsigned int tmp2 = ref1;
        ref1 = ref2;
        ref2 = tmp2;
    }   
}

template<class T>
__device__ 
inline void  binSearch_fragment_lower(T* binArray, int offset, int &mid, T testValue)
{	 mid = (binArray[mid] >= testValue ? mid-offset : mid+offset);  }
//Binary Search fragment for later block
template<class T>
__device__ 
inline void  binSearch_fragment_higher(T* binArray, int offset, int &mid, T testValue)
{	 mid = (binArray[mid] > testValue ? mid-offset : mid+offset); }


template<class T>
__device__
inline void binSearch_whole_lower(T* BKeys, int &index, T myKey)
{	
	index = (INTERSECT_B_BLOCK_SIZE_simple/2)-1;
	binSearch_fragment_lower<T> (BKeys, 256, index, myKey);
	binSearch_fragment_lower<T> (BKeys, 128, index, myKey);
	binSearch_fragment_lower<T> (BKeys, 64,  index, myKey);
	binSearch_fragment_lower<T> (BKeys, 32,  index, myKey);
	binSearch_fragment_lower<T> (BKeys, 16,  index, myKey);
	binSearch_fragment_lower<T> (BKeys, 8,   index, myKey);
	binSearch_fragment_lower<T> (BKeys, 4,   index, myKey);
	binSearch_fragment_lower<T> (BKeys, 2,   index, myKey);
	binSearch_fragment_lower<T> (BKeys, 1,   index, myKey);								
}

template<class T>
__device__
inline void binSearch_whole_higher(T* BKeys, int &index, T myKey)
{
	index = (INTERSECT_B_BLOCK_SIZE_simple/2)-1;		
	binSearch_fragment_higher<T> (BKeys, 256, index, myKey);		
	binSearch_fragment_higher<T> (BKeys, 128, index, myKey);		
	binSearch_fragment_higher<T> (BKeys, 64,  index, myKey);
	binSearch_fragment_higher<T> (BKeys, 32,  index, myKey);
	binSearch_fragment_higher<T> (BKeys, 16,  index, myKey);
	binSearch_fragment_higher<T> (BKeys, 8,   index, myKey);
	binSearch_fragment_higher<T> (BKeys, 4,   index, myKey);
	binSearch_fragment_higher<T> (BKeys, 2,   index, myKey);
	binSearch_fragment_higher<T> (BKeys, 1,   index, myKey);							
}


/** @brief Performs a linear search in our shared memory (done after binary search).
* It merges the partition on the left side with the associated partition on the right side
* @param[in] searchArray Array of keys
* @param[in] myKey Current key being considered
* @param[in] myVal Associated value of key
* @param[in,out] index Index in local B partition we are comparing with
* @param[out] saveGlobalArray Array of Keys after merge is complete
* @param[out] saveValueArray Array of values after merge is complete
* @param[in] myStartIdxC Global starting index of both partitions being considered
* @param[in] nextMaxB Minimum value in the partition NEXT to the one we are comparing against
* @param[in] localAPartSize Size of the partition we are considering
* @param[in] localBPartSize Size of the partition we are comparing against
* @param[in] localMaxB Largest element in THIS partition we are comparing against
* @param[in] localMinB Smallest element in THIS partition we are comparing against
* @param[in] aIndex The first global index our block is considering (thread 0 key 0)
* @param[in] bIndex The first global index our block is comparing against (value 0 in shared memory)
* @param[in] offset Count of key this thread is considering (between 1 and depth)
 
**/

template<class T, int depth>
__device__
inline void linearMerge_lower(T* searchArray, T myKey, unsigned int myVal, int &index, T* saveGlobalArray, unsigned int* saveValueArray, int myStartIdxC, 
					    T nextMaxB, int localAPartSize, int localBPartSize, T localMaxB, T localMinB, int aIndex, int bIndex, int offset)
{		
	
	
	while(searchArray[index] < myKey && index < INTERSECT_B_BLOCK_SIZE_multi )
		index++;
	
	int globalCAddress = myStartIdxC + index + bIndex + aIndex + offset + threadIdx.x*depth; 
	


	//Save Key-Val Pair
	if(((myKey <=  nextMaxB || myKey <= localMaxB) && myKey > localMinB)  && offset+threadIdx.x*depth+aIndex < localAPartSize)			
	{ 
		saveGlobalArray[globalCAddress] =  myKey;   saveValueArray[globalCAddress] = myVal;			
	}
		
				
}

/** @brief Performs a linear search in our shared memory (done after binary search).
* It merges the partition on the right side with the associated partition on the left side
* @param[in] searchArray Array of keys
* @param[in] myKey Current key being considered
* @param[in] myVal Associated value of key
* @param[in,out] index Index in local B partition we are comparing with
* @param[out] saveGlobalArray Array of Keys after merge is complete
* @param[out] saveValueArray Array of values after merge is complete
* @param[in] myStartIdxC Global starting index of both partitions being considered
* @param[in] localMinB Smallest element in THIS partition we are comparing against
* @param[in] nextMaxB Minimum value in the partition NEXT to the one we are comparing against
* @param[in] aIndex The first global index our block is considering (thread 0 key 0)
* @param[in] bIndex The first global index our block is comparing against (value 0 in shared memory)
* @param[in] offset Count of key this thread is considering (between 1 and depth)
* @param[in] localAPartSize Size of the partition we are considering
* @param[in] localBPartSize Size of the partition we are comparing against
**/
template<class T, int depth>
__device__
inline void linearMerge_higher(T* searchArray, T myKey, unsigned int myVal, int &index, T* saveGlobalArray, unsigned int* saveValueArray, int myStartIdxC, 
															T localMinB, T nextMaxB, int aIndex, int bIndex, int offset, int localAPartSize, int localBPartSize)
{		
	
	while(searchArray[index] <= myKey && index < INTERSECT_B_BLOCK_SIZE_multi && index < (localBPartSize-bIndex) )
		index++;

	int globalCAddress = myStartIdxC + index + bIndex + aIndex + offset + threadIdx.x*depth;

	//Save Key-Val Pair
	if((myKey <= nextMaxB && myKey >= localMinB /*|| bIndex + index >= localBPartSize*/)&& offset+threadIdx.x*depth+aIndex < localAPartSize)	
	{ saveGlobalArray[globalCAddress] =  myKey;	saveValueArray [globalCAddress] =  myVal;	}
			
}



/** @} */ // end merte  sort functions
/** @} */ // end cudpp_cta

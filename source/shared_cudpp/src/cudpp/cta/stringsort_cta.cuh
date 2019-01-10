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
#include "cudpp_stringsort.h"
#include <cudpp.h>
#include <stdio.h>

#include <cudpp_util.h>
#include <math.h>
#include "sharedmem.h"

/**
 * @file
 * stringsort_cta.cu
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
#define DEPTH_multi 2
#define CTASIZE_simple 256
#define CTASIZE_multi 256

#define INTERSECT_A_BLOCK_SIZE_simple DEPTH_simple*CTASIZE_simple
#define INTERSECT_B_BLOCK_SIZE_simple 2*DEPTH_simple*CTASIZE_simple

#define INTERSECT_A_BLOCK_SIZE_multi DEPTH_multi*CTASIZE_multi
#define INTERSECT_B_BLOCK_SIZE_multi 2*DEPTH_multi*CTASIZE_multi
typedef unsigned int uint;

/** @brief Breaks ties in keys (first four characters) returns true if cmpVal > myVal false otherwise
 * @param[in] myLoc, cmpLoc Location of the two inputs
 * @param[in] myBound, cmpBound Local memory bounds for the two addresses
 * @param[in] myAdd Address into global memory of our current value
 * @param[in] cmpAdd Address into global memory of the value we are comparing against
 * @param[in] stringLoc Global memory array (input string)
 * @param[in] stringSize Size of our input string
 * @param[in] termC Termination character for our strings
 * @return Returns 1 if cmpVal > myVal 0 otherwise 
 **/
__device__ int tie_break_simp(unsigned int myLoc, unsigned int cmpLoc, unsigned int myBound, unsigned int cmpBound, unsigned int myAdd, unsigned int cmpAdd, unsigned int* stringLoc, unsigned int stringSize, unsigned char termC)
{
	
	if(myLoc >= myBound && cmpLoc >= cmpBound)
		return cmpLoc > myLoc;
	else if(myLoc >= myBound)
		return 0;
	else if(cmpLoc >= cmpBound)
		return 1;
		
	//Our tie is in bounds therefore we can break the tie using traditional means
	if(myAdd >= stringSize)
		return 0;
	else if(cmpAdd >= stringSize)
		return 1;
	
	unsigned int a = stringLoc[myAdd];
	unsigned int b = stringLoc[cmpAdd];
	
	
	while(a == b && ((a&255) != termC) && ((b&255) != termC) && myAdd < (stringSize-1) && cmpAdd < (stringSize-1) )
	{
		

	    a = stringLoc[++myAdd];
	    b = stringLoc[++cmpAdd];
		
	}
	if(a==b)
		return (myAdd > cmpAdd ? 0 : 1);

	return a > b ? 0 : 1;
	
}
/** @brief Binary search within a single block (blockSort)
 * @param[in,out] cmpValue Value being considered from other partition
 * @param[in] tmpVal My Value
 * @param[in] in input keys
 * @param[in] addressPad addresses of string locations in case of tie breaks
 * @param[in] stringVals global string array used to break ties
 * @param[in,out] j The index we are considering
 * @param[in] bump The offset we update by
 * @param[in] sizeRemain Size of our block (if it's smaller than blockSize)
 * @param[in] stringSize Size of our global string array (for tie breaks)
 * @param[in] termC Termination character for our strings
 **/
template<class T, int depth>
__device__ void bin_search_block_string(T &cmpValue, T tmpVal, T* in, T* addressPad, T* stringVals, int & j, int bump, int sizeRemain, unsigned int stringSize, unsigned char termC)
{


    cmpValue = in[j]; 

	__syncthreads();
	__threadfence();
    if(cmpValue == tmpVal)
    {

		unsigned int myAdd = addressPad[depth*threadIdx.x];
		unsigned int cmpAdd = addressPad[j];

		
		j = (tie_break_simp(depth*threadIdx.x, j, sizeRemain, sizeRemain, myAdd, cmpAdd, stringVals, stringSize, termC) == 0 ? j + bump : j - bump);                
		
    }
    else
        j = (cmpValue < tmpVal ? j + bump : j - bump);  
    __syncthreads();

}

/** @brief Linear search within a single block (blockSort)
 * @param[in,out] cmpValue Value being considered from other partition
 * @param[in,out] tmpVal Temporary register which is used initially to compare our value, and then to store the final address
 *                        after our search
 * @param[in] in, addressPad, stringVals in = keys, addressPad = values, stringVals = global string array for tie breaks
 * @param[in] j index in B partition we are considering
 * @param[in] offset Since this is register packed, offset is the ith iteration of linear search
 * @param[in] last The end of partition B we are allowed to look upto
 * @param[in] startAddress The beginning of our partition 
 * @param[in] stringSize Size of our global string array
 * @param[in] termC Termination character for our strings
 **/
template<class T, int depth>
__device__ void lin_search_block_string(T &cmpValue, T &tmpVal, T* in, T* addressPad, T* stringVals, int &j, int offset, int last, int startAddress, int stringSize, unsigned char termC){			
	
	
	while (cmpValue < tmpVal && j < last)		
		cmpValue = in[++j];			
	
	__threadfence();
	__syncthreads();


	//If we need to tie break while linearly searching	
    while(cmpValue == tmpVal && j < last)              
    {	
		
		
		unsigned int myAdd = addressPad[depth*threadIdx.x+offset];
		unsigned int cmpAdd = addressPad[j];
        T myTmp = 0, cmpTmp = 0;
        
		//printf("tie break occured in linear at index %d for value %u compared to index %d (%d) unless corner case comparing addresses %d %d\n",
		//	depth*threadIdx.x+offset, cmpValue, j, j+startAddress+offset, myAdd, cmpAdd);
        while( (myAdd != cmpAdd) &&(++myAdd) < stringSize && (++cmpAdd) < stringSize && myTmp == cmpTmp)
        {
			myTmp = stringVals[myAdd];
			cmpTmp = stringVals[cmpAdd];
		}      

		if(myAdd > stringSize || (cmpAdd > stringSize))
			break;

		
        if((cmpTmp < myTmp)  && j < last)
           cmpValue = in[++j];
        else  if (cmpTmp > myTmp || j == last)
           break;		
		
		
    }

	

	__syncthreads();
	//Corner case to handle being at the edge of our shared memory search
    j = ((j==last && cmpValue < tmpVal) ? j+1 : j);	
	if (j == last && cmpValue == tmpVal)
	{
		int myLoc = depth*threadIdx.x + offset;
		int cmpLoc = j;
		int myAdd = addressPad[depth*threadIdx.x+offset];
		int cmpAdd = addressPad[j];

		j = tie_break_simp(myLoc, cmpLoc, BLOCKSORT_SIZE, BLOCKSORT_SIZE, myAdd, cmpAdd, stringVals, stringSize, termC) == 0 ? j+1 : j;	 
        
	}
    tmpVal = j+startAddress+offset;
	
}

/** @brief For blockSort. Compares two values and decides to swap if A1 > A2
 * @param[in,out] A1 First value being compared
 * @param[in,out] A2 Second value being compared
 * @param[in] index1 Local address of A1
 * @param[in] index2 Local address of A2 
 * @param[in,out] scratch Scratch memory storing the addresses
 * @param[in] stringVals String Values for tie breaks
 * @param[in] size size of our array
 * @param[in] termC Termination character for our strings
 *                        
 **/
template<class T>
__device__ void compareSwapVal(T &A1, T &A2, const int index1, const int index2, T* scratch, T* stringVals, unsigned int size, unsigned char termC)
{


    if(A1 > A2)
    {
        T tmp = A1;
        A1 = A2;
        A2 = tmp;
        tmp = scratch[index1];
        scratch[index1] = scratch[index2];
        scratch[index2] = tmp;
    }
    else if(A1 == A2 && index1 < size && index2 < size)
    {
        //bad case (hopefully infrequent) we have to gather from global memory to find out if we wanna swap
        T tmp = stringVals[scratch[index1]+1];
        T tmp2 = stringVals[scratch[index2]+1];
        int i = 2;
        while(tmp == tmp2)
        {
            tmp = stringVals[scratch[index1]+i];
            tmp2 = stringVals[scratch[index2]+i];
			i++;
        }

        if(tmp > tmp2)
        {
			tmp = A1;
            A1 = A2;
            A2 = tmp;
            tmp = scratch[index1];
            scratch[index1] = scratch[index2];
            scratch[index2] = tmp;
        } 
    }
}
/** @brief Performs a binary search in our shared memory, with tie breaks for strings
 * @param[in] keys, address Keys and address from our array
 * @param[in] offset, mid The current "middle" we are searching and the offset we will move to next
 * @param[in] cmpValue, testValue testValue is the value we are searching for from array A, cmpValue the value we have currently in B
 * @param[in] myAddress, myLoc, cmpLoc, myBound, cmpBound Same values from tie_break_simp which will be passed along
 * @param[in] globalStringArray, stringSize Our string array for breaking ties, and stringSize so we don't go out of bounds
 * @param[in] termC Termination character for our strings
**/
template<class T, int depth>
__device__ 
void  binSearch_fragment(T* keys, T* address, int offset, int &mid, T cmpValue, T testValue, T myAddress,  
						 int myLoc, int cmpLoc, int myBound, int cmpBound, T* globalStringArray, int stringSize, unsigned char termC)
{						

	cmpValue = keys[mid];
	
	if(cmpValue != testValue)
		mid = (cmpValue > testValue ? mid-offset : mid+offset); 
	
	T cmpKey = cmpValue; 

	if(cmpKey == testValue)
	{
		unsigned int cmpAdd = address[mid];				
		mid = tie_break_simp(myLoc, cmpLoc, myBound, cmpBound, myAddress, cmpAdd, globalStringArray, stringSize, termC) == 0 ? mid + offset : mid - offset;					
	}	

	
	
}

//TODO: merge binsearch_mult w/ regular

template<class T, int depth>
__device__ 
void  binSearch_frag_mult(T* keyArraySmem, T* valueArraySmem, int offset, int &mid, T cmpValue, T testValue, int myAddress, 
						 T* globalStringArray, int myStartIdxA, int myStartIdxB, int aIndex, int bIndex, int size, int stringSize, unsigned char termC)
{			
	cmpValue = keyArraySmem[mid];
	if(cmpValue != testValue)
		mid = (cmpValue > testValue ? mid-offset : mid+offset); 
 
	if(cmpValue == testValue)
	{
		int myLoc = myStartIdxA + aIndex + depth*threadIdx.x;
		int cmpLoc = myStartIdxB + bIndex + mid;
		mid = (tie_break_simp(myLoc, cmpLoc, size, size, myAddress, valueArraySmem[mid], globalStringArray, stringSize, termC) == 0 ? mid + offset : mid - offset);
	}
}


/** @brief Performs a linear search in our shared memory (done after binary search), with tie breaks for strings
 * @param[in, out] cmpValue The current value we are looking at in our B array
 * @param[in] myKey, myAddress Keys and address from our array
 * @param[in] index Current index we are considering in our B array
 * @param[in] BKeys, BValues Keys and Addresses for array B
 * @param[in, out] stringValues, A_keys, A_values, A_keys_out, A_values_out Global arrays for our strings, keys, values
 * @param[in] myStartIdxA, myStartIdxB, myStartIdxC Beginning indices for our partitions
 * @param[in] localMinB, localMaxB The minimum and maximum values in our B partition
 * @param[in] aCont, bCont, totalSize, mySizeA, mySizeB, stringSize Address bounds and calculation helpers
 * @param[in] i The index of the local element we are merging
 * @param[in] stepNum Debug helper
 * @param[in] placed Whether value has been placed yet or not
 * @param[in] termC Termination character for our strings
**/
template<class T, int depth>
__device__
void lin_merge_simple(T& cmpValue, T myKey, T myAddress, int& index, T* BKeys, T* BValues, T* stringValues, T* A_keys, T* A_values, T*A_keys_out, T*A_values_out,
					  int myStartIdxA, int myStartIdxB, int myStartIdxC, T localMinB, T localMaxB, int aCont, int bCont, int totalSize, int mySizeA,
					  int mySizeB, unsigned int stringSize, int i, int stepNum, bool &placed, unsigned char termC)
{

	int tid = threadIdx.x;	
	 
	//Here we keep climbing until we either reach the end of our partitions
	//Or we pass a value greater than ours
	while(cmpValue < myKey && index < INTERSECT_B_BLOCK_SIZE_simple)
	{

		index++;

		if(index < INTERSECT_B_BLOCK_SIZE_simple)
		    cmpValue = BKeys[index];
		else if(index == INTERSECT_B_BLOCK_SIZE_simple && bCont+index < mySizeB)
		{			
			cmpValue = A_keys[myStartIdxB+bCont+index];
		}
		else
			cmpValue = UINT_MAX;
	}

	//If we have a tie, brea it
	while(cmpValue == myKey && index < INTERSECT_B_BLOCK_SIZE_simple)
	{		
		int myLoc = myStartIdxA + depth*threadIdx.x + i;
		int cmpLoc = myStartIdxB + bCont + index;
		int cmpAdd = BValues[index];	
	
		//if(myKey == 1820065792)
		//	printf("B: (%d %d %d) (%d %d %d) %u %u\n", myLoc, cmpLoc, totalSize, cmpAdd, myAddress, stringSize, stringValues[cmpAdd], stringValues[myAddress]);
		if(tie_break_simp(myLoc, cmpLoc, totalSize, totalSize, myAddress, cmpAdd, stringValues, stringSize, termC) == 0)
		{
			index = index + 1;	
			if(index < INTERSECT_B_BLOCK_SIZE_simple)
			    cmpValue = BKeys[index];
			else if(index == INTERSECT_B_BLOCK_SIZE_simple-1 && bCont+index < mySizeB)			
			    cmpValue = A_keys[myStartIdxB+bCont+index];			
			else
				cmpValue = UINT_MAX;

		}
		else
			break;
	}
	

	int globalCAddress = myStartIdxC + bCont + index + aCont + i;


	

	bool isInWindow = (index > 0 && index < (INTERSECT_B_BLOCK_SIZE_simple));

	isInWindow = (isInWindow || (index == 0 && myKey > localMinB));
	isInWindow = (isInWindow || (index >= (INTERSECT_B_BLOCK_SIZE_simple-1) && myKey < localMaxB));
	isInWindow = (isInWindow || (bCont+index >= mySizeB));

	

	//if(myKey >= 1820065792 && myKey <= 1820916440)
	//			printf("key %u %u index %u placed %u %d (min %u max %u %d)\n", myKey, cmpValue, globalCAddress, placed, myStartIdxA+aCont+i, localMinB, localMaxB, index);

	if(!isInWindow && index == 0 && myKey <= localMinB)
	{
		
		if(!placed)
			isInWindow = true;
	}
	else if(!isInWindow && index >= (INTERSECT_B_BLOCK_SIZE_simple-1) && myKey == localMaxB && cmpValue <= myKey)
	{
		
		//Here we must check if our string is greater than our tie @ index INTERSECT_B_BLOCK_SIZE_simple (or our shared memory partition)
		unsigned int myLoc = myStartIdxA + depth*tid + i;
		unsigned int cmpLoc = myStartIdxB + bCont + index;
		unsigned int cmpAdd = (bCont+index < mySizeB ? A_values[cmpLoc] : UINT_MAX);
		
		
		if(cmpAdd > totalSize || tie_break_simp(myLoc, cmpLoc, totalSize, totalSize, myAddress, cmpAdd, stringValues, stringSize, termC) == 1)
			isInWindow = true;
	}	
	 
	if(globalCAddress >= totalSize)
		return;
	//Save Value if it is valid (correct window)
	//If we are on the edge of a window, and we are tied with the localMax or localMin value
	//we must go to global memory to find out if we are valid
	if(!placed && isInWindow)
    {		
		A_keys_out[globalCAddress] = myKey;	
        A_values_out[globalCAddress] = myAddress;			
		placed = true;
	}				
		
}
/** @brief Performs a linear search in our shared memory, used by multiMerge kernel 
 * @param[in] BKeys, BValues Keys and Addresses for array B
 * @param[in] myKey, myAddress Keys and address from our array
 * @param[in] placed Whether value has been placed yet or not
 * @param[in] index Current index we are considering in our B array
 * @param[in, out] cmpValue The current value we are looking at in our B array
 * @param[in, out] stringValues, A_keys, A_values, A_keys_out, A_values_out Global arrays for our strings, keys, values
 * @param[in] myStartIdxA, myStartIdxB, myStartIdxC Beginning indices for our partitions
 * @param[in] localAPartSize, localBPartSize, localCPartSize Array of partition sizes for our inputs and outputs
 * @param[in] localMinB, localMaxB The minimum and maximum values in our B partition
 * @param[in] tid thread ID
 * @param[in] aIndex, bIndex, totalSize, stringSize Address bounds and calculation helpers
 * @param[in] i The index of the local element we are merging
 * @param[in] termC Termination character for our strings 
**/
template<class T, int depth>
__device__
void linearStringMerge(T* BKeys, T* BValues, T myKey, T myAddress, bool &placed, int &index,  T &cmpValue, T* A_keys, T* A_values, T* A_keys_out, 
					   T* A_values_out, T* stringValues, int myStartIdxC, int myStartIdxA, int myStartIdxB, int localAPartSize, int localBPartSize, int localCPartSize,
                       T localMaxB, T localMinB, int tid, int aIndex, int bIndex, int i, int stringSize, int totalSize, unsigned char termC)
{		
	
	while(cmpValue < myKey && index < INTERSECT_B_BLOCK_SIZE_multi )
	{
		index++;

		if(index < INTERSECT_B_BLOCK_SIZE_multi)
		    cmpValue = BKeys[index];
		else if(index == INTERSECT_B_BLOCK_SIZE_multi && bIndex+index < localBPartSize)
		{			
			cmpValue = A_keys[myStartIdxB+bIndex+index];
		}
		else
			cmpValue = UINT_MAX;
	}
		
	
	while(cmpValue == myKey && index < INTERSECT_B_BLOCK_SIZE_multi)
	{
		int myLoc = myStartIdxA + depth*threadIdx.x + i;
		int cmpLoc = myStartIdxB + bIndex + index;
		int cmpAdd = BValues[index];	
	
		if(tie_break_simp(myLoc, cmpLoc, totalSize, totalSize, myAddress, cmpAdd, stringValues, stringSize, termC) == 0)
		{
			index = index + 1;	
			if(index < INTERSECT_B_BLOCK_SIZE_multi)
			    cmpValue = BKeys[index];
			else if(index == INTERSECT_B_BLOCK_SIZE_multi-1 && bIndex+index < localBPartSize)			
			    cmpValue = A_keys[myStartIdxB+bIndex+index];			
			else
				cmpValue = UINT_MAX;

		}
		else
			break;

	}
	
	int globalCAddress = myStartIdxC + index + bIndex + aIndex + i + tid*depth; 



	
	
	bool isInWindow = (index > 0 && index < (INTERSECT_B_BLOCK_SIZE_multi));

	isInWindow = (isInWindow || (index == 0 && myKey > localMinB));
	isInWindow = (isInWindow || (index >= (INTERSECT_B_BLOCK_SIZE_multi-1) && myKey < localMaxB));
	isInWindow = (isInWindow || (bIndex+index >= localBPartSize));
	

		
	if((myKey == localMaxB) && index >= (INTERSECT_B_BLOCK_SIZE_multi-1) && globalCAddress <= (myStartIdxC+localCPartSize))
	{
		//Here we must check if our string is greater than our tie @ index INTERSECT_B_BLOCK_SIZE_simple (or our shared memory partition)
		unsigned int myLoc = myStartIdxA + depth*tid + i;
		unsigned int cmpLoc = myStartIdxB + bIndex + index;
		unsigned int cmpAdd = (bIndex+index < localBPartSize ? A_values[cmpLoc] : UINT_MAX);
		
		
		if(cmpAdd > totalSize || tie_break_simp(myLoc, cmpLoc, totalSize, totalSize, myAddress, cmpAdd, stringValues, stringSize, termC) == 1)
			isInWindow = true;
	
	}
	else if(!isInWindow && index == 0 && myKey <= localMinB)
	{
		//Here we must check if our string is greater than our tie @ index 0 (or our shared memory partition)
		int myLoc = myStartIdxA + depth*tid + i;
		int cmpLoc = myStartIdxB + bIndex;

		int cmpAdd = BValues[0];
		
		

		if(!placed || tie_break_simp(myLoc, cmpLoc, totalSize, totalSize, myAddress, cmpAdd, stringValues, stringSize, termC) == 0)
			isInWindow = true;
		
	}

	if(!placed && isInWindow)
    {
		A_keys_out [globalCAddress] = myKey;	
        A_values_out[globalCAddress] = myAddress;	
		placed = true;
	}		 
}


/** @} */ // end stringsort functions
/** @} */ // end cudpp_cta

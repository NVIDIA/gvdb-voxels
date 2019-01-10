// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt 
// in the root directory of this source distribution.
// ------------------------------------------------------------- 

#include "cudpp_stringsort.h"
#include <cudpp_globals.h>
#include "sharedmem.h"
#include "cta/stringsort_cta.cuh"


/**
 * @file
 * stringsort_kernel.cu
 *   
 * @brief CUDPP kernel-level radix sorting routines
 */

/* doxygen: this include is already wrapped in "StringSort Functions" */

/** \addtogroup cudpp_kernel
 * @{
 */
/*
*/
__global__
void dotAddInclusive(unsigned int* numSpaces, unsigned int* d_address, unsigned int* packedAddress, unsigned int numElements, unsigned int stringSize)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int myId = bid*blockDim.x + tid;

	if(myId >= numElements)
		return;

	if(myId > 0 && myId < (numElements-1))
	{
		packedAddress[myId] = (d_address[myId] + numSpaces[myId])  >> 2;		
	}
	else if (myId == (numElements-1))
	{
		packedAddress[myId] = (stringSize + numSpaces[myId]) >> 2;		
	}
	else
		packedAddress[myId] = 0;

	
	
}

/** @brief Calculate the number of spaces required for each string to align the string array.
 * @param[out] numSpaces Number of spaces required for each string
 * @param[in] d_address Input addresses of each string
 * @param[in] d_stringVals String array
 * @param[in] termC Termination character for the strings
 * @param[in] numElements Number of strings
 * @param[in] stringSize Number of characters in the string array
 **/

__global__
void alignedOffsets(unsigned int* numSpaces, unsigned int* d_address, 
					unsigned char* d_stringVals, unsigned char termC, unsigned int numElements, unsigned int stringSize)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int myId = bid*blockDim.x + tid;

	if(myId >= numElements)
		return;

	unsigned int startAddress = d_address[myId];
	unsigned int endAddress = startAddress;

	unsigned char c;
	do
	{
		c = d_stringVals[endAddress++];			
	}
	while (c != termC && endAddress < stringSize);

	

	int length = endAddress - startAddress;
	numSpaces[myId] = ((4-(length%4))%4);

	//printf("Id %d starts %d ends %d numChars %d packedSpace %d\n", myId, startAddress, endAddress, length, (length + (4-(length%4))%4) >> 2);

}

/** @brief Packs strings into unsigned ints to be sorted later. These packed strings will also
 * be aligned
 * @param[out] packedStrings Resulting packed strings. 
 * @param[in] d_stringVals Unpacked string array which we will pack
 * @param[out] packedAddress Resulting addresses for each string to the packedStrings array
 * @param[in] address Input addresses of unpacked strings 
 * @param[in] numElements Number of strings
 * @param[in] stringArrayLength Number of characters in the string array
 * @param[in] termC Termination character for the strings
 **/

__global__
void alignString(unsigned int* packedStrings, unsigned char* d_stringVals, unsigned int* packedAddress, unsigned int* address, 
				 unsigned int numElements, unsigned int stringArrayLength, unsigned char termC)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int myId = bid*blockDim.x + tid;

	if(myId >= numElements)
		return;
	int oldAddress = address[myId];
	int newAddress = packedAddress[myId];
	


	unsigned char c1, c2, c3, c4;

	int i = 0;
	do
	{
		c1 = d_stringVals[oldAddress + i];
		c2 = ((oldAddress+i+1) < stringArrayLength && c1 != termC) ? d_stringVals[oldAddress+i+1] : termC;
		c3 = ((oldAddress+i+2) < stringArrayLength && c2 != termC) ? d_stringVals[oldAddress+i+2] : termC;
		c4 = ((oldAddress+i+3) < stringArrayLength && c3 != termC) ? d_stringVals[oldAddress+i+3] : termC;

		
		unsigned int val = (c1 << 24) + (c2 << 16) + (c3 << 8) + c4;
		packedStrings[newAddress + (i>>2)] = val;
		//printf("Id %d is saving %d in %d (%c%c%c%c)\n", myId, val, newAddress+i/4, c1, c2, c3, c4);
		i+=4;
	}
	while(c1 != termC && c2 != termC && c3 != termC && c4 != termC && i < stringArrayLength);

}

/** @brief Create keys (first four characters stuffed in an uint) from the addresses to the strings, and the string array.
 * @param[out] d_keys Resulting keys
 * @param[in] packedStrings Packed string array.
 * @param[in] packedAddress Addresses which point to the string array. 
 * @param[in] numElements Number of strings
 **/
__global__
void createKeys(unsigned int* d_keys, unsigned int* packedStrings, unsigned int* packedAddress, unsigned int numElements)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int myId = bid*blockDim.x + tid;

	if(myId >= numElements)
		return;

	int newAddress = packedAddress[myId];
	d_keys[myId] = packedStrings[newAddress];
}


/** @brief Converts addresses from packed (unaligned) form to unpacked and unaligned form
 * Resulting aligned strings begin in our string array packed in an unsigned int
 * and aligned such that each string begins at the start of a uint (divisible by 4)
 * @param[in] packedAddress Resulting packed addresses that have been sorted. All strings are aligned.
 * @param[in] packedAddressRef Original array after packing (before sort). Used as a reference.
 * @param[out] address Final output of sorted addresses in unpacked form.
 * @param[in] addressRef Reference array of original unpacked addresses. 
 * @param[in] numElements Number of strings
 **/
__global__
void unpackAddresses(unsigned int* packedAddress,
				     unsigned int* packedAddressRef,
				     unsigned int* address,
				     unsigned int* addressRef,				   
				     size_t numElements)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int myId = bid*blockDim.x + tid;

	if(myId >= numElements)
		return;

	int myAddress = packedAddress[myId];
	int begin = 0;
	int last = numElements-1;
	int mid = (begin+last)/2;
	
	unsigned int val = packedAddressRef[mid];
	while(val != myAddress && (mid != begin) && (mid != last))
	{
		if(val > myAddress)
			last = mid;
		else
			begin = mid;

		mid = (begin + last) / 2;		
		val = packedAddressRef[mid];
	}

	while(val < myAddress && mid < (numElements-1))
		val = packedAddressRef[++mid];

	while(val > myAddress && mid > 0)
		val = packedAddressRef[--mid];

	address[myId] = addressRef[mid];


}


/** @brief Copies unused portions of arrays in our ping-pong strategy
 * @param[in] A_keys_dev The keys we will be copying
 * @param[in] A_vals_dev The values we will be copying
 * @param[out] A_keys_out_dev The destination keys array
 * @param[out] A_vals_out_dev The destination values array
 * @param[in] offset The offset we are starting to copy from
 * @param[in] numElementsToCopy The number of elements we copy starting from the offset
 **/
template <class T>
__global__
void simpleCopy(T* A_keys_dev, unsigned int* A_vals_dev, T* A_keys_out_dev, unsigned int* A_vals_out_dev, int offset, int numElementsToCopy)
{
    int myId = blockIdx.x*blockDim.x + threadIdx.x;
    if(myId >= numElementsToCopy)
        return;
    A_keys_out_dev[offset+myId] = A_keys_dev[offset+myId];
    A_vals_out_dev[offset+myId] = A_vals_dev[offset+myId];

}

/** @brief Does an initial blockSort based on the size of our partition (limited by shared memory size)
 * @param[in,out] A_keys, A_address This sort is in-place. A_keys and A_address store the key (first four characters) and addresses of our strings
 * @param[in] stringVals Global array of strings for tie breaks
 * @param[in] blockSize size of each block
 * @param[in] totalSize The total size of the array we are sorting
 * @param[in] stringSize The size of our string array (stringVals)
 * @param[in] termC Termination character for the strings
 **/
template<class T, int depth>
__global__
void blockWiseStringSort(T *A_keys, T* A_address, T* stringVals, int blockSize, int totalSize, unsigned int stringSize, unsigned char termC)
{
        
    //load into registers
    T Aval[depth]; T saveValue[depth];
        
    //__shared__ T scratchPad[2*BLOCKSORT_SIZE];
	extern __shared__ T BWSshared[];   
	T* scratchPad = (T*) BWSshared;
    T* addressPad = (T*) &scratchPad[BLOCKSORT_SIZE];
                

    int bid = blockIdx.x;
    int tid = threadIdx.x;

    //Grab keys and addresses
    for(int i = 0; i < depth; i++)    
        Aval[i] = bid*blockSize+tid*depth + i < totalSize ? A_keys[bid*blockSize+tid*depth+i] : UINT_MAX;        
    
    for(int i = tid; i < blockSize; i+=CTA_BLOCK)    
        addressPad[i] = bid*blockSize + i < totalSize? A_address[blockSize*bid+i] : UINT_MAX;

    __syncthreads();
    __threadfence();
    //manually unrolled for performance
        
        
    //Sort first 8 values

    
    int offset = tid*depth;
    int sizeRemaining = totalSize - bid*blockSize;
    compareSwapVal<T>(Aval[0], Aval[1], offset,     offset+1, addressPad, stringVals, sizeRemaining, termC);       
    compareSwapVal<T>(Aval[1], Aval[2], offset+1,   offset+2, addressPad, stringVals, sizeRemaining, termC);       
    compareSwapVal<T>(Aval[2], Aval[3], offset+2,   offset+3, addressPad, stringVals, sizeRemaining, termC);       
    compareSwapVal<T>(Aval[3], Aval[4], offset+3,   offset+4, addressPad, stringVals, sizeRemaining, termC);       
    compareSwapVal<T>(Aval[4], Aval[5], offset+4,   offset+5, addressPad, stringVals, sizeRemaining, termC);       
    compareSwapVal<T>(Aval[5], Aval[6], offset+5,   offset+6, addressPad, stringVals, sizeRemaining, termC);       
    compareSwapVal<T>(Aval[6], Aval[7], offset+6,   offset+7, addressPad, stringVals, sizeRemaining, termC);       
        
    compareSwapVal<T>(Aval[0], Aval[1], offset,     offset+1, addressPad, stringVals, sizeRemaining, termC);       
    compareSwapVal<T>(Aval[1], Aval[2], offset+1,   offset+2, addressPad, stringVals, sizeRemaining, termC);       
    compareSwapVal<T>(Aval[2], Aval[3], offset+2,   offset+3, addressPad, stringVals, sizeRemaining, termC);       
    compareSwapVal<T>(Aval[3], Aval[4], offset+3,   offset+4, addressPad, stringVals, sizeRemaining, termC);       
    compareSwapVal<T>(Aval[4], Aval[5], offset+4,   offset+5, addressPad, stringVals, sizeRemaining, termC);       
    compareSwapVal<T>(Aval[5], Aval[6], offset+5,   offset+6, addressPad, stringVals, sizeRemaining, termC);       

    compareSwapVal<T>(Aval[0], Aval[1], offset,     offset+1, addressPad, stringVals, sizeRemaining, termC);       
    compareSwapVal<T>(Aval[1], Aval[2], offset+1,   offset+2, addressPad, stringVals, sizeRemaining, termC);       
    compareSwapVal<T>(Aval[2], Aval[3], offset+2,   offset+3, addressPad, stringVals, sizeRemaining, termC);       
    compareSwapVal<T>(Aval[3], Aval[4], offset+3,   offset+4, addressPad, stringVals, sizeRemaining, termC);       
    compareSwapVal<T>(Aval[4], Aval[5], offset+4,   offset+5, addressPad, stringVals, sizeRemaining, termC);       

    compareSwapVal<T>(Aval[0], Aval[1], offset,     offset+1, addressPad, stringVals, sizeRemaining, termC);       
    compareSwapVal<T>(Aval[1], Aval[2], offset+1,   offset+2, addressPad, stringVals, sizeRemaining, termC);       
    compareSwapVal<T>(Aval[2], Aval[3], offset+2,   offset+3, addressPad, stringVals, sizeRemaining, termC);       
    compareSwapVal<T>(Aval[3], Aval[4], offset+3,   offset+4, addressPad, stringVals, sizeRemaining, termC);       

    compareSwapVal<T>(Aval[0], Aval[1], offset,     offset+1, addressPad, stringVals, sizeRemaining, termC);       
    compareSwapVal<T>(Aval[1], Aval[2], offset+1,   offset+2, addressPad, stringVals, sizeRemaining, termC);       
    compareSwapVal<T>(Aval[2], Aval[3], offset+2,   offset+3, addressPad, stringVals, sizeRemaining, termC);       

    compareSwapVal<T>(Aval[0], Aval[1], offset,     offset+1, addressPad, stringVals, sizeRemaining, termC);       
    compareSwapVal<T>(Aval[1], Aval[2], offset+1,   offset+2, addressPad, stringVals, sizeRemaining, termC);       

    compareSwapVal<T>(Aval[0], Aval[1], offset,     offset+1, addressPad, stringVals, sizeRemaining, termC);       
   
        
    __syncthreads();

    int j;

        
#pragma unroll
    for(int i=0;i<depth;i++)                        
        scratchPad[tid*depth+i] = Aval[i];      
        
    __syncthreads();

    T * in = scratchPad;       

    int mult = 1;
    int count = 0;
    int steps = 128;    

        
    while (mult < steps)
    {
        T cmpValue;
        T tmpVal;

        int first, last;
        first = (tid>>(count+1))*depth*2*mult;
        int midPoint = first+mult*depth;

        //first half or second half
        int addPart = threadIdx.x%(mult<<1) >= mult ? 1 : 0;

        //calculate range of values
        if(addPart == 0)                
            first += depth*mult;                            
        last = first+depth*mult-1;


        

        j = (first+last)/2;             

        int startAddress = offset-midPoint;             
        int range = last-first;         
                                                
        __syncthreads();


        tmpVal = Aval[0];                                                           
                
        //Begin binary search
        switch(range)
        {
        case 1023: bin_search_block_string<T, depth>(cmpValue, tmpVal, in, addressPad, stringVals, j, 256, sizeRemaining, stringSize, termC);                                   
        case 511: bin_search_block_string<T, depth>(cmpValue, tmpVal, in, addressPad, stringVals, j, 128, sizeRemaining, stringSize, termC);            
        case 255: bin_search_block_string<T, depth>(cmpValue, tmpVal, in, addressPad, stringVals, j, 64, sizeRemaining, stringSize, termC);            
        case 127: bin_search_block_string<T, depth>(cmpValue, tmpVal, in, addressPad, stringVals, j, 32, sizeRemaining, stringSize, termC);                    
        case 63: bin_search_block_string<T, depth>(cmpValue, tmpVal, in, addressPad, stringVals, j, 16, sizeRemaining, stringSize, termC);     
        case 31: bin_search_block_string<T, depth>(cmpValue, tmpVal, in, addressPad, stringVals, j, 8, sizeRemaining, stringSize, termC);                       
        case 15: bin_search_block_string<T, depth>(cmpValue, tmpVal, in, addressPad, stringVals, j, 4, sizeRemaining, stringSize, termC);            
        case 7: bin_search_block_string<T, depth>(cmpValue, tmpVal, in,  addressPad,stringVals, j, 2, sizeRemaining, stringSize, termC);            
        case 3: bin_search_block_string<T, depth>(cmpValue, tmpVal, in, addressPad, stringVals, j, 1, sizeRemaining, stringSize, termC);                        
        }

                
                

        //possible need for slight correction   
		//TODO: blockSort doesn't handle duplicate strings yet
		//Need to fix
        cmpValue = in[j];
        if(cmpValue == tmpVal && offset < (sizeRemaining) && j < sizeRemaining)              
        {
            T tmp = stringVals[addressPad[offset]+1];
            T tmp2 = stringVals[addressPad[j]+1];
            int i = 2;
            while(tmp == tmp2)
            {
                tmp = stringVals[addressPad[offset]+i];
                tmp2 = stringVals[addressPad[j]+i];
                i++;
            }        

            j = (tmp2 < tmp ? j + 1 : j);                
            cmpValue = in[j];
        }
        else if(cmpValue < tmpVal && j < sizeRemaining)         
            cmpValue = in[++j];                     

        __syncthreads();
        __threadfence();
                
        if(cmpValue == tmpVal && j == last && j <= sizeRemaining && depth*tid < sizeRemaining)              
        {
            T tmp = stringVals[addressPad[depth*tid]+1];
            T tmp2 = stringVals[addressPad[j]+1];
            int i = 2;
            while(tmp == tmp2)
            {
                tmp = stringVals[addressPad[depth*tid]+i];
                tmp2 = stringVals[addressPad[j]+i];
                i++;
            }                      
            j = (tmp2 < tmp ? j + 1 : j);                
        }
        else if(cmpValue < tmpVal && j == last && j < sizeRemaining && depth*tid < sizeRemaining)                       
            j++;          
                
                
        //TODO: unrolled this loop because the loop overhead/compiler was slowing things down
        //template unroll this!
                

        __syncthreads();        
        __threadfence();
        Aval[0] = j+startAddress;                       
        lin_search_block_string<T, depth>(cmpValue,  Aval[1], in, addressPad, stringVals, j, 1, last, startAddress, stringSize, termC);                                
        lin_search_block_string<T, depth>(cmpValue,  Aval[2], in, addressPad, stringVals, j, 2, last, startAddress, stringSize, termC);                
        lin_search_block_string<T, depth>(cmpValue,  Aval[3], in, addressPad, stringVals, j, 3, last, startAddress, stringSize, termC);                
        lin_search_block_string<T, depth>(cmpValue,  Aval[4], in, addressPad, stringVals, j, 4, last, startAddress, stringSize, termC);                
        lin_search_block_string<T, depth>(cmpValue,  Aval[5], in, addressPad, stringVals, j, 5, last, startAddress, stringSize, termC);                
        lin_search_block_string<T, depth>(cmpValue,  Aval[6], in, addressPad, stringVals, j, 6, last, startAddress, stringSize, termC);                
        lin_search_block_string<T, depth>(cmpValue,  Aval[7], in, addressPad, stringVals, j, 7, last, startAddress, stringSize, termC);
                
        __threadfence();
        __syncthreads();
        saveValue[0] = in[tid*depth];
        saveValue[1] = in[tid*depth+1];
        saveValue[2] = in[tid*depth+2];
        saveValue[3] = in[tid*depth+3];
        saveValue[4] = in[tid*depth+4];
        saveValue[5] = in[tid*depth+5];
        saveValue[6] = in[tid*depth+6];
        saveValue[7] = in[tid*depth+7];
        __threadfence();
        __syncthreads();
        in[Aval[0]] = saveValue[0];
        in[Aval[1]] = saveValue[1];
        in[Aval[2]] = saveValue[2];
        in[Aval[3]] = saveValue[3];
        in[Aval[4]] = saveValue[4];
        in[Aval[5]] = saveValue[5];
        in[Aval[6]] = saveValue[6];
        in[Aval[7]] = saveValue[7];
        __threadfence();
        __syncthreads();   
        saveValue[0] = addressPad[tid*depth];
        saveValue[1] = addressPad[tid*depth+1];
        saveValue[2] = addressPad[tid*depth+2];
        saveValue[3] = addressPad[tid*depth+3];
        saveValue[4] = addressPad[tid*depth+4];
        saveValue[5] = addressPad[tid*depth+5];
        saveValue[6] = addressPad[tid*depth+6];
        saveValue[7] = addressPad[tid*depth+7];
        __threadfence();
        __syncthreads();
        addressPad[Aval[0]] = saveValue[0];
        addressPad[Aval[1]] = saveValue[1];
        addressPad[Aval[2]] = saveValue[2];
        addressPad[Aval[3]] = saveValue[3];
        addressPad[Aval[4]] = saveValue[4];
        addressPad[Aval[5]] = saveValue[5];
        addressPad[Aval[6]] = saveValue[6];
        addressPad[Aval[7]] = saveValue[7];
        __threadfence();
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
    for(int i=tid;i<blockSize;i+= CTA_BLOCK)
    {           
        if(bid*blockSize+i < totalSize)
        {
                
            A_keys[bid*blockSize+i] = in[i];
            A_address[bid*blockSize+i] = addressPad[i];
        }


        __syncthreads();

                
    }    

}
/** @brief Merges two independent sets. Each CUDA block works on two partitions of data without cooperating
 * @param[in] A_keys First four characters (input) of our sets to merge
 * @param[in] A_values Addresses of the strings (for tie breaks)
 * @param[in] stringValues Global string array for tie breaks
 * @param[out] A_keys_out, A_values_out Keys and values array after merge step
 * @param[in] sizePerPartition The size of each partition for this merge step
 * @param[in] size Global size of our array
 * @param[in] step Number of merges done so far
 * @param[in] stringSize global string length
 * @param[in] termC Termination character for the strings
 **/
template<class T, int depth>
__global__
void simpleStringMerge(T *A_keys, T *A_keys_out, T *A_values, T* A_values_out, T* stringValues, int sizePerPartition, int size, int step, int stringSize, unsigned char termC)
{
    //each block will be responsible for a submerge
    int myStartIdxA, myStartIdxB, myStartIdxC;
    int myId = blockIdx.x;


    int totalSize;
    int mySizeA, mySizeB;

    //Slight difference in loading if we are an odd or even block
    if(myId%2 == 0)
    {
        myStartIdxA = (myId/2)*2*sizePerPartition; 
        myStartIdxB = myStartIdxA + sizePerPartition; 
        myStartIdxC = myStartIdxA; 
        totalSize = (myStartIdxB + sizePerPartition > size ? size : myStartIdxB + sizePerPartition);            
        mySizeA = sizePerPartition;
        mySizeB = totalSize-myStartIdxB;
    }
    else
    {
        myStartIdxB = (myId/2)*2*sizePerPartition; 
        myStartIdxA = myStartIdxB + sizePerPartition; 
        myStartIdxC = myStartIdxB; 
        totalSize = (myStartIdxA + sizePerPartition > size ? size : myStartIdxA + sizePerPartition);
        mySizeB = sizePerPartition;
        mySizeA = totalSize-myStartIdxA;
    }
        

    T cmpValue; 
    int mid, index;  int bIndex = 0; int aIndex = 0;        
        

        
    //Shared Memory pool
    //__shared__ T BValues[INTERSECT_B_BLOCK_SIZE_simple*2+4]; //(T*) shared; 
	extern __shared__ T BValues[];
    T* BKeys = (T*) &BValues[INTERSECT_B_BLOCK_SIZE_simple];        
    T* BMax = (T*) &BValues[2*INTERSECT_B_BLOCK_SIZE_simple];               
    T* lastIndex = (T*) &BMax[3];
        

        
    bool breakout = false;
    int tid = threadIdx.x;  

    T localMaxB, localMinB;                                 
        

    T myKey[depth]; T myValue[depth]; bool placed[depth];   

    //Load Registers
    if(aIndex + INTERSECT_A_BLOCK_SIZE_simple < mySizeA) 
    {
#pragma unroll
        for(int i = 0;i < depth; i++) 
        { 
            myKey[i]   = A_keys  [myStartIdxA + aIndex+ depth*tid + i]; 
            myValue[i] = A_values[myStartIdxA + aIndex+ depth*tid + i]; 
            placed[i] = false;
        }       
    }
    else
    {
#pragma unroll
        for(int i = 0;i < depth; i++) 
        { 
            myKey[i] =   (aIndex+depth*tid + i < mySizeA ? A_keys  [myStartIdxA + aIndex+ depth*tid + i]   : UINT_MAX); 
            myValue[i] = (aIndex+depth*tid + i < mySizeA ? A_values[myStartIdxA + aIndex+ depth*tid + i]   : UINT_MAX);
            placed[i] =  (aIndex+depth*tid + i < mySizeA ?  false : true);
        }
    }       
                
        
    //Load Shared-Memory
    if(bIndex + INTERSECT_B_BLOCK_SIZE_simple < mySizeB) 
    {
        int bi = tid;                                   
#pragma unroll
        for(int i = 0;i < INTERSECT_B_BLOCK_SIZE_simple/CTASIZE_simple; i++, bi+=CTASIZE_simple) 
        {BKeys[bi] =   A_keys[myStartIdxB + bIndex + bi];  BValues[bi] = A_values[myStartIdxB + bIndex + bi]; }
                
    }       
    else 
    {
        int bi = tid;
#pragma unroll
        for(int i = 0;i < INTERSECT_B_BLOCK_SIZE_simple/CTASIZE_simple; i++, bi+=CTASIZE_simple) 
        { BKeys[bi] =   (bIndex + bi < mySizeB ? A_keys  [myStartIdxB + bIndex + bi] : UINT_MAX); 
            BValues[bi] = (bIndex + bi < mySizeB ? A_values[myStartIdxB + bIndex + bi] : UINT_MAX);}              
    }

        
    //Save localMaxA and localMaxB
    if(tid == CTASIZE_simple-1)             
        BMax[1] = myKey[depth-1];                       
    if(tid == 0)
        BMax[0] =  (bIndex + INTERSECT_B_BLOCK_SIZE_simple - 1 < mySizeB ?
                    A_keys[myStartIdxB + bIndex + INTERSECT_B_BLOCK_SIZE_simple - 1] : UINT_MAX);

    __syncthreads();                                        

    unsigned int myLast;

    do
    {                       
        __syncthreads();                                

        localMinB = BKeys[0];
        localMaxB = BKeys[1023];    

        __syncthreads();                                

        index = 0;
        cmpValue = localMinB;
        int cumulativeAddressA = myStartIdxA+aIndex+threadIdx.x*depth;                                                                                                                  

        if(( !(myKey[0] > localMaxB) || bIndex+INTERSECT_B_BLOCK_SIZE_multi >= mySizeB) || (myKey[0] < localMinB && !placed[0]) || (myKey[1] < localMinB && !placed[1]))
        {                               
                        
            int cumulateAddressB = myStartIdxB+bIndex+index;
            index = -1;                                             
            mid = (INTERSECT_B_BLOCK_SIZE_simple/2)-1;
            if(INTERSECT_B_BLOCK_SIZE_simple >= 1024)
                binSearch_fragment<T,depth>(BKeys, BValues, 256, mid, cmpValue, myKey[0], myValue[0], cumulativeAddressA, cumulateAddressB, size, size, stringValues, stringSize, termC);              
            if(INTERSECT_B_BLOCK_SIZE_simple >= 512)
                binSearch_fragment<T,depth>(BKeys, BValues, 128, mid, cmpValue, myKey[0], myValue[0], cumulativeAddressA, cumulateAddressB, size, size, stringValues, stringSize, termC);              
            if(INTERSECT_B_BLOCK_SIZE_simple >= 256)
                binSearch_fragment<T,depth>(BKeys, BValues, 64, mid, cmpValue, myKey[0], myValue[0], cumulativeAddressA, cumulateAddressB, size, size, stringValues, stringSize, termC);       

            binSearch_fragment<T,depth>(BKeys, BValues, 32, mid, cmpValue, myKey[0], myValue[0], cumulativeAddressA, cumulateAddressB, size, size, stringValues, stringSize, termC);                       
            binSearch_fragment<T,depth>(BKeys, BValues, 16, mid, cmpValue, myKey[0], myValue[0], cumulativeAddressA, cumulateAddressB, size, size, stringValues, stringSize, termC);                       
            binSearch_fragment<T,depth>(BKeys, BValues,  8, mid, cmpValue, myKey[0], myValue[0], cumulativeAddressA, cumulateAddressB, size, size, stringValues, stringSize, termC);                       
            binSearch_fragment<T,depth>(BKeys, BValues,  4, mid, cmpValue, myKey[0], myValue[0], cumulativeAddressA, cumulateAddressB, size, size, stringValues, stringSize, termC);                       
            binSearch_fragment<T,depth>(BKeys, BValues,  2, mid, cmpValue, myKey[0], myValue[0], cumulativeAddressA, cumulateAddressB, size, size, stringValues, stringSize, termC);                       
            binSearch_fragment<T,depth>(BKeys, BValues,  1, mid, cmpValue, myKey[0], myValue[0], cumulativeAddressA, cumulateAddressB, size, size, stringValues, stringSize, termC);                       
                
            index = mid;                    
                        
            cmpValue = BKeys[index];
        
            //correct search if needed                      

                        
            if(cmpValue < myKey[0] && index < INTERSECT_B_BLOCK_SIZE_simple)                        
                cmpValue = BKeys[++index];                                                                              
            else if(cmpValue == myKey[0] && index < INTERSECT_B_BLOCK_SIZE_simple)
            {
                //Tied version of previous if statement
                int myLoc = myStartIdxA + depth*threadIdx.x;
                int cmpLoc = myStartIdxB + bIndex + index;
                int cmpAdd = BValues[index];    

                //if(myKey[0] == 1820065792)
			     //   printf("%d %d %d)\n", myLoc, cmpLoc, size);              
                if(tie_break_simp(myLoc, cmpLoc, size, size, myValue[0], cmpAdd, stringValues, stringSize, termC) == 0)
                    cmpValue = BKeys[++index];                                      
                                
            }       
                        
                        
            if(cmpValue < myKey[0] && (bIndex+index)<mySizeB)
            {       
                index++;                
                cmpValue =  A_keys[myStartIdxB+bIndex + (index)];
            }
            else if(cmpValue == myKey[0] && (bIndex+index)<mySizeB)
            {
                //Tied version of previous if statement
                int myLoc = myStartIdxA + depth*threadIdx.x;
                int cmpLoc = myStartIdxB + bIndex + index;
                int cmpAdd = BValues[index];    

				//if(myKey[0] == 1820065792)
			     //   printf("(%d %d %d) (%d %d %d) %u %u\n", myLoc, cmpLoc, size, cmpAdd, myValue[0], stringSize, stringValues[cmpAdd], stringValues[myValue[0]]);              
                                
                if(tie_break_simp(myLoc, cmpLoc, size, size, myValue[0], cmpAdd, stringValues, stringSize, termC) == 0)
                    index++;
                cmpValue = A_keys[myStartIdxB+bIndex+index];
                                                                                        
            }
                
                

            //End Binary Search 
            //binary search done for first element in our set (A_0)         
            //Save Value if it is valid (correct window)
            //If we are on the edge of a window, and we are tied with the localMax or localMin value
            //we must go to global memory to find out if we are valid
            //unsigned int tmpKey, cmpKey;
            //int count = 0 ;

			//if(myKey[0] >= 1820065792 && myKey[0] <= 1820916440)
			//	printf("key %u cmpValue %u index %u placed %u %d (min %u max %u index %d)\n",
			//	myKey[0], cmpValue, myStartIdxC + bIndex + aIndex + depth*tid + index, placed[0], myStartIdxA+aIndex, localMinB, localMaxB, index);
                        
            //Base case
            if(!placed[0] && ((index > 0 && index < INTERSECT_B_BLOCK_SIZE_simple) || (myKey[0] < localMaxB) || (index+bIndex) >= mySizeB))
            {                               
                                
                A_keys_out  [myStartIdxC + bIndex + aIndex + depth*tid + index] = myKey[0];     
                A_values_out[myStartIdxC + bIndex + aIndex + depth*tid + index] = myValue[0];
                placed[0] = true;
            }                       
            else if(!placed[0] && myKey[0] == localMaxB && index >= (INTERSECT_B_BLOCK_SIZE_simple-1))
            {                               
                //If we are on the edge and have a tie break
                int myLoc = myStartIdxA + depth*threadIdx.x;
                int cmpLoc = myStartIdxB + bIndex + index;
                int cmpAdd = A_values[cmpLoc];     

                        
                if(tie_break_simp(myLoc, cmpLoc, size, size, myValue[0], cmpAdd, stringValues, stringSize, termC) == 1)
                {
                    A_keys_out  [myStartIdxC + bIndex + aIndex + depth*tid + index] = myKey[0];     
                    A_values_out[myStartIdxC + bIndex + aIndex + depth*tid + index] = myValue[0];           
                    placed[0] = true;
                }

            }
            else if(!placed[0] && myKey[0] == localMinB && index == 0)
            {
                //If we are on the edge and have a tie break
                        
                A_keys_out[myStartIdxC + bIndex + aIndex+depth*tid+index] = myKey[0];   
                A_values_out[myStartIdxC + bIndex + aIndex+depth*tid+index] = myValue[0];
                placed[0] = true;

            }
        }
                
                
        //After binary search, linear merge
        if(aIndex+depth*tid+1 < mySizeA)
            lin_merge_simple<T, depth>(cmpValue, myKey[1], myValue[1], index, BKeys, BValues, stringValues, A_keys, A_values, A_keys_out, A_values_out,
                                       myStartIdxA, myStartIdxB, myStartIdxC, localMinB, localMaxB, aIndex+tid*depth, bIndex, totalSize, mySizeA, mySizeB, stringSize, 1, step, placed[1], termC);


        
        __syncthreads();        

        if(tid == CTASIZE_simple-1)
            lastIndex[0] = index;   
                
        __syncthreads();

        myLast = lastIndex[0];
                
        __syncthreads();

        if( (myLast < INTERSECT_B_BLOCK_SIZE_simple || bIndex+INTERSECT_B_BLOCK_SIZE_simple >= mySizeB) 
            && aIndex+INTERSECT_A_BLOCK_SIZE_simple < mySizeA)
        {       
            __syncthreads();
                        
            aIndex += INTERSECT_A_BLOCK_SIZE_simple;        
            //Use UINT_MAX-1 as an "invalid/no-value" type in case we are out of values to check
            if(aIndex + INTERSECT_A_BLOCK_SIZE_simple < mySizeA) 
            {               
                for(int i = 0;i < depth; i++) 
                { myKey[i] = A_keys[myStartIdxA + aIndex + depth*tid + i]; myValue[i] = A_values[myStartIdxA + aIndex + depth*tid + i]; placed[i] = false; }
            }
            else 
            {


                for(int i = 0;i < depth; i++) 
                { 
                    myKey[i] =   (aIndex+depth*tid + i < mySizeA ? A_keys[myStartIdxA + aIndex+ depth*tid + i]   : UINT_MAX); 
                    myValue[i] = (aIndex+depth*tid + i < mySizeA ? A_values[myStartIdxA + aIndex+ depth*tid + i]   : UINT_MAX);
                    placed[i] =  (aIndex+depth*tid + i < mySizeA ? false : true);
                }
            }                               
            if(tid == CTASIZE_simple-1)             
            {
                BMax[1] = myKey[depth-1]; //localMaxA for all threads                                                   
            }       
            __syncthreads();
                        
        }                       
        else if( (myLast == INTERSECT_B_BLOCK_SIZE_simple || aIndex+INTERSECT_A_BLOCK_SIZE_simple >= mySizeA) && 
                 (bIndex+INTERSECT_B_BLOCK_SIZE_simple) < mySizeB)
        {                                       
            //Use UINT_MAX as an "invalid/no-value" type in case the streaming window cannot be filled
            bIndex += INTERSECT_B_BLOCK_SIZE_simple;                        

            __syncthreads();
                        
            if(bIndex + INTERSECT_B_BLOCK_SIZE_simple < mySizeB) 
            {
                int bi = tid;                                   
                for(int i = 0;i < INTERSECT_B_BLOCK_SIZE_simple/CTASIZE_simple; i++, bi+=CTASIZE_simple) 
                { BKeys[bi] =   A_keys[myStartIdxB + bIndex + bi]; BValues[bi] = A_values[myStartIdxB + bIndex + bi]; }
            }
            else 
            {
                int bi = tid;                   
                for(int i = 0;i < INTERSECT_B_BLOCK_SIZE_simple/CTASIZE_simple; i++, bi+=CTASIZE_simple) 
                { 
                    BKeys[bi] =   (bIndex + bi < mySizeB ? A_keys[myStartIdxB + bIndex + bi]   : UINT_MAX); 
                    BValues[bi] = (bIndex + bi < mySizeB ? A_values[myStartIdxB + bIndex + bi] : UINT_MAX);
                }
            }

                        
            if(tid == 0)
            {
                BMax[0] =  (bIndex + INTERSECT_B_BLOCK_SIZE_simple  < mySizeB ? A_keys[myStartIdxB + bIndex + INTERSECT_B_BLOCK_SIZE_simple ] : UINT_MAX);                                              
            }
            __syncthreads();
                        
                                                
        }               
        else
            breakout = true;

                
        __syncthreads();                        
                
    }
    while(!breakout);

}


/** @brief For our multiMerge kernels we need to divide our partitions into smaller partitions. This kernel breaks up a set of partitions into splitsPP*numPartitions subpartitions.
 * @param[in] A_keys, A_address First four characters (input), and addresses of our inputs
 * @param[in] stringValues Global string array for tie breaks
 * @param[in] splitsPP, numPartitions, partitionSize Partition information for this routine (splitsPP=splits Per Partition) 
 * @param[in] partitionBeginA, partitionSizesA Partition starting points and sizes for each new subpartition in our original set in A
 * @param[in] partitionBeginB, partitionSizesB Partition starting points and sizes for each new subpartition in our original set in B
 * @param[in] size, stringSize Number of elements in our set, and size of our global string array
 * @param[in] termC Termination character for the strings
 **/
template<class T>
__global__
void findMultiPartitions(T *A_keys, T* A_address, T* stringValues, int splitsPP, int numPartitions, int partitionSize,  unsigned int* partitionBeginA, unsigned int* partitionSizesA, 
                         unsigned int* partitionBeginB, unsigned int* partitionSizesB, size_t size, size_t stringSize, unsigned char termC)
{
    int myId = threadIdx.x + blockIdx.x*blockDim.x;
    int myIdLoc = myId%splitsPP + (myId/splitsPP)*splitsPP*2;
    if (myId >= (numPartitions*splitsPP)/2)
        return;

    int myStartA, myEndA;   
    T testSample, myStartSample, myEndSample;
    int subPartitionSize;

        
    int myPartitionId = myId/splitsPP;
    int mySubPartitionId = myId%splitsPP;

        
    subPartitionSize = partitionSize/splitsPP;
 
    //printf("tid %d\n", threadIdx.x);

        
    myStartA = (myPartitionId)*2*partitionSize + (mySubPartitionId)*subPartitionSize; // we are at the beginning of a partition                             
        
    myEndA = myStartA + subPartitionSize;

    myEndA = myEndA < size ? myEndA : size; 
    myEndSample = myEndA == size ? A_keys[myEndA-1] : A_keys[myEndA];



    int myStartRange = (myPartitionId)*2*partitionSize + partitionSize;
    int myEndRange = myStartRange + partitionSize;

                
    
    myEndRange = myEndRange < size ? myEndRange : size;

        
    if(myStartA > size)
        return;

        
    int first = myStartRange;
    int last = myEndRange;
    uint BMin = A_keys[myStartRange];
    int mid = (first + last)/2;

    T myPrevSample;
    myStartSample = A_keys[myStartA];
    if (mySubPartitionId == 0)
    {               
        mid = myStartRange;                             
    }
    else
    {               
        myPrevSample = A_keys[myStartA-1];

        while(mid != first && mid != last)
        {
                        
            testSample = A_keys[mid];
            if(testSample < myPrevSample)
                first = mid;
            else if(testSample > myPrevSample)
                last = mid;
            else
            {
                if(tie_break_simp(myStartA-1, mid, size, size, A_address[myStartA-1], A_address[mid], stringValues, stringSize, termC) == 1)
                    last = mid;
                else
                    first = mid;
            }
            mid =(first+last)/2;

        }       
                
        T prevSample = mid > myStartRange ? A_keys[--mid] : 0;

        //testSample needs to be greater than myStartSample: A[myStartA] < A[mid] 
        //and myNextSample must be greater than prevSample: A[myStartA+1] > A[mid-1]
        //printf("thread %d mid %d myStartSample %u prevSample %u\n", threadIdx.x, mid, myStartSample, prevSample);
        while(prevSample > myStartSample && mid > myStartRange)
            prevSample = A_keys[--mid];
        while(prevSample == myStartSample && mid > myStartRange)
        {
            if(tie_break_simp(myStartA, mid, size, size, A_address[myStartA], A_address[mid], stringValues, stringSize, termC) == 1)
                prevSample = A_keys[--mid];
            else
                break;
        }
        testSample = prevSample < BMin ? BMin : prevSample;

        //printf("thread %d mid %d myStartSample %u prevSample %u %u < %u ?\n", threadIdx.x, mid, myStartSample, prevSample, testSample, myPrevSample);
        while(testSample < myPrevSample && mid < myEndRange-1)
            testSample = A_keys[++mid];
        while(testSample == myPrevSample && mid < myEndRange-1)
        {
            if(tie_break_simp(myStartA-1, mid, size, size, A_address[myStartA-1], A_address[mid], stringValues, stringSize, termC) == 0)
                testSample = A_keys[++mid];
            else
                break;
        }

        if(testSample < myPrevSample && mid < myEndRange)
            mid++;
        if(testSample == myPrevSample && mid < myEndRange)
        {
            if(tie_break_simp(myStartA-1, mid, size, size, A_address[myStartA-1], A_address[mid], stringValues, stringSize, termC) == 0)
                mid++;
        }


        //printf("final start %d mid %d myStartSample %u prevSample %u\n", threadIdx.x, mid, myStartSample, prevSample);
    }


    //printf("idx %d myStartA %d myEndA %d BMin %u BMax %u AMin %u AMax %u %d %d\n", 
    //      threadIdx.x+blockIdx.x*blockDim.x, myStartA, myEndA, BMin, BMax, AMin, AMax, mid, myEndRange);

    //Check here
        
        
    unsigned int myStartB = mid;
    unsigned int myEndB;
        

    first = myStartRange; 
    last = myEndRange;
    mid = (first + last)/2; 

    if (mySubPartitionId == splitsPP-1)
    {
        mid = myEndRange;
        myEndSample = UINT_MAX;
        //myEndA = myEndRange;
    }
    else
    {
        myEndSample = A_keys[myEndA];
        myPrevSample = A_keys[myEndA-1];

        while(mid != first && mid != last)
        {
                        
            testSample = A_keys[mid];
            if(testSample < myPrevSample)
                first = mid;
            else if(testSample > myPrevSample)
                last = mid;
            else 
            {
                if(tie_break_simp(myEndA-1, mid, size, size, A_address[myEndA-1], A_address[mid], stringValues, stringSize, termC) == 1)
                    last = mid;
                else
                    first = mid;
            }
            mid =(first+last)/2;                    
        }

                
        T prevSample = mid > myStartRange ? A_keys[--mid] : 0;

        //testSample needs to be greater than myStartSample: A[myStartA] < A[mid] 
        //and myNextSample must be greater than prevSample: A[myStartA+1] > A[mid-1]
        while(prevSample > myEndSample && mid > myStartRange)
            prevSample = A_keys[--mid];
        while(prevSample == myEndSample && mid > myStartRange)
        {
            if(tie_break_simp(myEndA, mid, size, size, A_address[myEndA], A_address[mid], stringValues, stringSize, termC) == 1)
                prevSample = A_keys[--mid];
            else
                break;
        }
        testSample = prevSample < BMin ? BMin : prevSample;

        while(testSample < myPrevSample && mid < myEndRange-1)
            testSample = A_keys[++mid];
        while(testSample == myPrevSample && mid < myEndRange-1)
        {
            if(tie_break_simp(myEndA-1, mid, size, size, A_address[myEndA-1], A_address[mid], stringValues, stringSize, termC) == 0)
                testSample = A_keys[++mid];
            else
                break;
        }

        if(testSample < myPrevSample && mid < myEndRange)
            mid++;
        if(testSample == myPrevSample && mid < myEndRange)
        {
            if(tie_break_simp(myEndA-1, mid, size, size, A_address[myEndA-1], A_address[mid], stringValues, stringSize, termC) == 0)
                mid++;
                        
        }               
    }       
    myEndB = mid;

    partitionBeginA[myIdLoc] = myStartA; //partitionBegin found for first set
    partitionBeginB[myIdLoc+splitsPP] =  myStartA;
        
    partitionBeginB[myIdLoc] = myStartB; 
    partitionBeginA[myIdLoc+splitsPP] = myStartB;

    partitionSizesA[myIdLoc] = myEndA-myStartA;
    partitionSizesB[myIdLoc+splitsPP] = myEndA-myStartA;

    partitionSizesB[myIdLoc] = myEndB-myStartB;
    partitionSizesA[myIdLoc+splitsPP] = myEndB-myStartB;

        
}

/** @brief Main merge kernel where multiple CUDA blocks cooperate to merge a partition(s)
 * @param[in] A_keys, A_values First four characters (input), and addresses of our inputs
 * @param[out] A_keys_out, A_values_out First four characters, and addresses for our outputs(ping-pong)
 * @param[in] stringValues string array for tie breaks
 * @param[out] subPartitions, numBlocks Number of splits per partitions and number of partitions respectively
 * @param[in] partitionBeginA, partitionSizeA Where partitions begin and how large they are for Segment A
 * @param[in] partitionBeginB, partitionSizeB Where partitions begin and how large they are for Segment B
 * @param[in] entirePartitionSize The maximum length of a partition
 * @param[in] step Number of merge cycles done
 * @param[in] size Number of total strings being sorted
 * @param[in] stringSize Length of string array
 * @param[in] termC Termination character for the strings
 **/


template<class T, int depth>
__global__
void stringMergeMulti(T *A_keys, T*A_keys_out, T* A_values, T *A_values_out, T* stringValues, int subPartitions, int numBlocks, 
                      unsigned int *partitionBeginA, unsigned int *partitionSizeA, unsigned int *partitionBeginB, unsigned int* partitionSizeB, 
                      int entirePartitionSize, int step, size_t size, size_t stringSize, unsigned char termC)
{
    int myId = blockIdx.x;
    int myMergeId = (myId/(2*subPartitions))*2;
    int partStart = myMergeId*entirePartitionSize;

    int myStartIdxA, myStartIdxB, localAPartSize, localBPartSize, localCPartSize;

    //T finalMaxB;
        
        
    myStartIdxA = partitionBeginA[myId];
    myStartIdxB = partitionBeginB[myId];

    localAPartSize = partitionSizeA[myId];
    localBPartSize = partitionSizeB[myId];

        

    int myStartIdxC;                        
        
    myStartIdxC = myStartIdxA + (myStartIdxB - entirePartitionSize) - partStart;    
    localCPartSize = localAPartSize + localBPartSize;       

        
    //If at the end of a partition, resize
    if (localCPartSize + myStartIdxC > size)
        localCPartSize = size - myStartIdxC;

        
    //Now we have the beginning and end points of our subpartitions, merge the two together                 
                
    T cmpValue; 

        
    int mid, index; 
    int bIndex = 0; int aIndex = 0; 

    //__shared__ T      BValues[2*INTERSECT_B_BLOCK_SIZE_multi+3];    
	extern __shared__ T BValues[];
    T * BKeys =      &BValues[INTERSECT_B_BLOCK_SIZE_multi];
    T * BMax =       &BValues[2*INTERSECT_B_BLOCK_SIZE_multi];
    T * lastIndex = &BValues[2*INTERSECT_B_BLOCK_SIZE_multi+3];
        
    unsigned int myLast;
    bool breakout = false;
    int tid = threadIdx.x;

        
    T localMaxB;                    
    T localMinB = 0;                
        
    T myKey[depth];
    T myValue[depth];
    bool placed[depth];

        

#pragma unroll
    for(int i =0; i <depth; i++)
    {
        myKey[i] =   (depth*tid + i < localAPartSize ? A_keys  [myStartIdxA + depth*tid + i]   : UINT_MAX);             
        myValue[i] = (depth*tid + i < localAPartSize ? A_values[myStartIdxA + depth*tid + i]   : UINT_MAX);             
        placed[i] = false;
    }


        

    if(bIndex + INTERSECT_B_BLOCK_SIZE_multi < localBPartSize) 
    {
        int bi = tid;                                   
#pragma unroll
        for(int i = 0;i < INTERSECT_B_BLOCK_SIZE_multi/CTASIZE_multi; i++, bi+=CTASIZE_multi) 
        {
            BKeys[bi] =   A_keys  [myStartIdxB + bi];
            BValues[bi] = A_values[myStartIdxB + bi];
        }
    }
    else {
        int bi = tid;
#pragma unroll
        for(int i = 0;i < INTERSECT_B_BLOCK_SIZE_multi/CTASIZE_multi; i++, bi+=CTASIZE_multi)
        {
            BKeys[bi] =   ((bIndex + bi < localBPartSize) ? A_keys  [myStartIdxB + bi]   : UINT_MAX);
            BValues[bi] = ((bIndex + bi < localBPartSize) ? A_values[myStartIdxB + bi]   : UINT_MAX);
        }
    }

    __syncthreads();        


    if(tid == CTASIZE_multi-1)
    {
        BMax[1] =  myKey[depth-1];              
    }               
        
        
    do 
    {
        __syncthreads();
        localMaxB = BKeys[INTERSECT_B_BLOCK_SIZE_multi-1];
        localMinB = BKeys[0];

                
        __syncthreads();        
        index = 0;                                              
        cmpValue = localMinB;
                        
        __syncthreads();
        __threadfence(); //Extra Added                                  
        if(!(myKey[0] > localMaxB) || bIndex+INTERSECT_B_BLOCK_SIZE_multi >= localBPartSize || (myKey[0] < localMinB && !placed[0]) || (myKey[1] < localMinB && !placed[1]))
        {       

            mid = (INTERSECT_B_BLOCK_SIZE_multi/2)-1;
            if(INTERSECT_B_BLOCK_SIZE_multi >= 1024)
                binSearch_fragment<T, depth> (BKeys, BValues, 256, mid, cmpValue, myKey[0], myValue[0], myStartIdxA+aIndex+tid*depth, myStartIdxB+bIndex+index, myStartIdxA+localAPartSize, myStartIdxB+localBPartSize, stringValues, stringSize, termC);

                        
            if(INTERSECT_B_BLOCK_SIZE_multi>= 512)
                binSearch_fragment<T, depth> (BKeys, BValues, 128, mid, cmpValue, myKey[0], myValue[0], myStartIdxA+aIndex+tid*depth, myStartIdxB+bIndex+index, myStartIdxA+localAPartSize, myStartIdxB+localBPartSize, stringValues, stringSize, termC);

            if(INTERSECT_B_BLOCK_SIZE_multi >= 256)
                binSearch_fragment<T, depth> (BKeys, BValues, 64, mid, cmpValue, myKey[0], myValue[0], myStartIdxA+aIndex+tid*depth, myStartIdxB+bIndex+index, myStartIdxA+localAPartSize, myStartIdxB+localBPartSize, stringValues, stringSize, termC);

            binSearch_fragment<T, depth> (BKeys, BValues, 32, mid, cmpValue, myKey[0], myValue[0], myStartIdxA+aIndex+tid*depth, myStartIdxB+bIndex+index, myStartIdxA+localAPartSize, myStartIdxB+localBPartSize, stringValues, stringSize, termC);
            binSearch_fragment<T, depth> (BKeys, BValues, 16, mid, cmpValue, myKey[0], myValue[0], myStartIdxA+aIndex+tid*depth, myStartIdxB+bIndex+index, myStartIdxA+localAPartSize, myStartIdxB+localBPartSize, stringValues, stringSize, termC);
            binSearch_fragment<T, depth> (BKeys, BValues, 8, mid, cmpValue, myKey[0], myValue[0], myStartIdxA+aIndex+tid*depth, myStartIdxB+bIndex+index, myStartIdxA+localAPartSize, myStartIdxB+localBPartSize, stringValues, stringSize, termC);
            binSearch_fragment<T, depth> (BKeys, BValues, 4, mid, cmpValue, myKey[0], myValue[0], myStartIdxA+aIndex+tid*depth, myStartIdxB+bIndex+index, myStartIdxA+localAPartSize, myStartIdxB+localBPartSize, stringValues, stringSize, termC);
            binSearch_fragment<T, depth> (BKeys, BValues, 2, mid, cmpValue, myKey[0], myValue[0], myStartIdxA+aIndex+tid*depth, myStartIdxB+bIndex+index, myStartIdxA+localAPartSize, myStartIdxB+localBPartSize, stringValues, stringSize, termC);
            binSearch_fragment<T, depth> (BKeys, BValues, 1, mid, cmpValue, myKey[0], myValue[0], myStartIdxA+aIndex+tid*depth, myStartIdxB+bIndex+index, myStartIdxA+localAPartSize, myStartIdxB+localBPartSize, stringValues, stringSize, termC);
        
            index = mid;                    
            cmpValue = BKeys[index];
            if(cmpValue < myKey[0] && index < INTERSECT_B_BLOCK_SIZE_multi)                         
                cmpValue = BKeys[++index];                                              
            else if(cmpValue == myKey[0] && index < INTERSECT_B_BLOCK_SIZE_multi)
            {
                int myLoc, cmpLoc;
                myLoc = myStartIdxA + aIndex + tid*depth;
                cmpLoc = myStartIdxB + bIndex + index;
                int cmpAdd = BValues[index];
                        
                if(cmpLoc != myLoc && cmpAdd > 0 && tie_break_simp(myLoc, cmpLoc, size, size, myValue[0], cmpAdd, stringValues, stringSize, termC) == 0)
                {
                    cmpValue = BKeys[++index];              
                }

            }
                

            if(cmpValue < myKey[0] && bIndex+index < localBPartSize)
            {       
                index++;                
                cmpValue = (bIndex + index < localBPartSize ? A_keys[myStartIdxB+bIndex + (index)] : UINT_MAX);
            }
                        
            if(cmpValue == myKey[0] && bIndex+index < localBPartSize)
            {
                int myLoc, cmpLoc;
                myLoc = myStartIdxA + aIndex + tid*depth;
                cmpLoc = myStartIdxB + bIndex + index;
                                
                int cmpAdd = BValues[index];    
                        
                if(cmpAdd > 0 && tie_break_simp(myLoc, cmpLoc, size, size, myValue[0], cmpAdd, stringValues, stringSize, termC) == 0)
                {
                    index++;                                                
                    cmpValue = bIndex+index < localBPartSize ? A_keys[myStartIdxB+bIndex+index] : UINT_MAX;
                }
                else if(cmpAdd < 0)
                    cmpValue = UINT_MAX;
            }

                
            int globalCAddress = (myStartIdxC + index + bIndex + aIndex + tid*depth);                               
                                                
            if(!placed[0] && ((index > 0 && index < INTERSECT_B_BLOCK_SIZE_multi) || (myKey[0] < localMaxB && myKey[0] > localMinB) || (bIndex+index) >= (localBPartSize)) && globalCAddress < (myStartIdxC+localCPartSize))
            {       

                                
                A_keys_out  [globalCAddress] = myKey[0];                                                                                        
                A_values_out[globalCAddress] = myValue[0];
                placed[0] = true;
            }
            else if(!placed[0] && (myKey[0] == localMaxB &&  index >= INTERSECT_B_BLOCK_SIZE_multi-1) && globalCAddress < (myStartIdxC+localCPartSize))
            {
                //tie break
                int myLoc = myStartIdxA+aIndex+depth*tid;
                int cmpLoc = myStartIdxB+bIndex+index;
                                
                unsigned int cmpAdd = A_values[cmpLoc];                                                         
                                
                if(tie_break_simp(myLoc, cmpLoc, size, size, myValue[0], cmpAdd, stringValues, stringSize, termC) == 1)
                {                                       
                    A_keys_out  [myStartIdxC + bIndex + aIndex+depth*tid+index] = myKey[0]; 
                    A_values_out[myStartIdxC + bIndex + aIndex+depth*tid+index] = myValue[0];       
                    placed[0] = true;
                }
            }
            else if(!placed[0] && myKey[0] <= localMinB && index == 0 && globalCAddress < (myStartIdxC+localCPartSize))
            {
                int myLoc = myStartIdxA+aIndex+depth*tid;
                int cmpLoc = myStartIdxB+bIndex+index;
                                
                unsigned int cmpAdd = A_values[cmpLoc];


                if(!placed[0] && myKey[0] < localMinB)
                {
                    //place
                    A_keys_out  [myStartIdxC + bIndex + aIndex+depth*tid+index] = myKey[0]; 
                    A_values_out[myStartIdxC + bIndex + aIndex+depth*tid+index] = myValue[0];       
                    placed[0] = true;
                }
                if(!placed[0] && myKey[0] == localMinB && cmpAdd > 0 || tie_break_simp(myLoc, cmpLoc, size, size, myValue[0], cmpAdd, stringValues, stringSize, termC) == 0)
                {                                       
                    A_keys_out  [myStartIdxC + bIndex + aIndex+depth*tid+index] = myKey[0]; 
                    A_values_out[myStartIdxC + bIndex + aIndex+depth*tid+index] = myValue[0];       
                    placed[0] = true;
                }                               
            }
                
            if(aIndex+depth*tid+1 < localAPartSize)
                linearStringMerge<T, depth>(BKeys, BValues, myKey[1], myValue[1], placed[1], index, cmpValue, A_keys, A_values, A_keys_out, A_values_out, stringValues, 
                                            myStartIdxC, myStartIdxA, myStartIdxB, localAPartSize, localBPartSize, localCPartSize, localMaxB, localMinB, tid, aIndex, bIndex, 
                                            1, stringSize, size, termC);                       
                        
        }       
 
                                                

        if(myKey[0] > localMaxB)
            index = INTERSECT_B_BLOCK_SIZE_multi;


                
        
        __syncthreads();
        if(tid == CTASIZE_multi-1)
            lastIndex[0] = index;
        __syncthreads();

        myLast = lastIndex[0];
        __syncthreads();
                
        

                
        if((myLast < INTERSECT_B_BLOCK_SIZE_multi || bIndex+INTERSECT_B_BLOCK_SIZE_multi >= localBPartSize) 
           && aIndex+INTERSECT_A_BLOCK_SIZE_multi < localAPartSize)
        {
        
            aIndex += INTERSECT_A_BLOCK_SIZE_multi; 
            //Use UINT_MAX-1 as an "invalid/no-value" type in case we are out of values to check

            if(aIndex + INTERSECT_A_BLOCK_SIZE_multi < localAPartSize) 
            {               
#pragma unroll
                for(int i = 0;i < depth; i++) 
                { myKey[i] = A_keys[myStartIdxA + aIndex + depth*tid + i]; myValue[i] = A_values[myStartIdxA + aIndex + depth*tid + i]; placed[i] = false;}
            }
            else 
            {

#pragma unroll
                for(int i = 0;i < depth; i++) 
                { myKey[i] =   (aIndex+depth*tid + i < localAPartSize ? A_keys[myStartIdxA + aIndex+ depth*tid + i]   : UINT_MAX); 
                    myValue[i] = (aIndex+depth*tid + i < localAPartSize ? A_values[myStartIdxA + aIndex+ depth*tid + i]   : UINT_MAX);
                    placed[i] =  false;}
            }
        
            if(tid == CTASIZE_multi-1)              
            {
                BMax[1] = myKey[depth-1]; //localMaxA for all threads                                                           
            }
        }                       
        else if((myLast == INTERSECT_B_BLOCK_SIZE_multi || aIndex+INTERSECT_A_BLOCK_SIZE_multi >= localAPartSize) && 
                (bIndex+INTERSECT_B_BLOCK_SIZE_multi) < localBPartSize)
        {                       
                        
            //Use UINT_MAX as an "invalid/no-value" type in case the streaming window cannot be filled                                      
            bIndex += INTERSECT_B_BLOCK_SIZE_multi; 
            if(bIndex + INTERSECT_B_BLOCK_SIZE_multi < localBPartSize) 
            {
                int bi = tid;                                   
#pragma unroll
                for(int i = 0;i < INTERSECT_B_BLOCK_SIZE_multi/CTASIZE_multi; i++, bi+=CTASIZE_multi) 
                {
                    BKeys[bi] =   A_keys[myStartIdxB + bIndex + bi];
                    BValues[bi] = A_values[myStartIdxB + bIndex + bi];
                }
            }
            else {
                int bi = tid;
#pragma unroll
                for(int i = 0;i < INTERSECT_B_BLOCK_SIZE_multi/CTASIZE_multi; i++, bi+=CTASIZE_multi) 
                {
                    BKeys[bi] =   ((bIndex + bi < localBPartSize) ? A_keys  [myStartIdxB + bIndex + bi]   : UINT_MAX);
                    BValues[bi] = ((bIndex + bi < localBPartSize)? A_values[myStartIdxB + bIndex + bi]   : UINT_MAX);
                }
            }
                        
            if(tid ==CTASIZE_multi-1)
            {
                BMax[0] =  (bIndex + INTERSECT_B_BLOCK_SIZE_multi < localBPartSize ? A_keys[myStartIdxB + bIndex + INTERSECT_B_BLOCK_SIZE_multi-1] : UINT_MAX);                         
            }
                        
            __syncthreads();                        
        }
        else
            breakout = true;        
                
        __syncthreads();
                
                
    }
    while(!breakout);
}

                
/** @} */ // end cudpp_kernel

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

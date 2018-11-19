// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt 
// in the root directory of this source distribution.
// ------------------------------------------------------------- 

#include "cudpp_mergesort.h"
#include <cudpp_globals.h>
#include <cudpp_util.h>
#include "sharedmem.h"
#include "cta/mergesort_cta.cuh"

/**
 * @file
 * mergesort_kernel.cu
 *   
 * @brief CUDPP kernel-level radix sorting routines
 */

/** \addtogroup cudpp_kernel
  * @{
 */

/** @name MergeSort Functions
 * @{
 */



typedef unsigned int uint;

/** @brief Copies unused portions of arrays in our ping-pong strategy
 * @param[in] A_keys_dev, A_vals_dev The keys and values we will be copying
 * @param[out] A_keys_out_dev, A_vals_out_dev The keys and values array we will copy to
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
/** @brief Sorts blocks of data of size blockSize
 * @param[in,out] A_keys keys to be sorted
 * @param[in,out] A_values associated values to keys
 * @param[in] blockSize Size of the chunks being sorted
 * @param[in] totalSize Size of the enitre array
 **/

template<class T, int depth>
__global__
void blockWiseSort(T *A_keys, unsigned int* A_values, int blockSize, size_t totalSize)
{
    //load into registers
    T myKey[depth];
    unsigned int myValue[depth];
    unsigned int myAddress[depth];

#if (__CUDA_ARCH__ >= 200)
    extern __shared__ char shared[];
#else
	extern __shared__ unsigned int shared[];
#endif
    //scratchPad is for stuffing keys
    T* scratchPad =  (T*) shared;
    unsigned int* addressPad = (unsigned int*) &scratchPad[BLOCKSORT_SIZE];
        

    int bid = blockIdx.x;
    int tid = threadIdx.x;

	T MAX_VAL = getMax<T>();
    
    //Grab values in coalesced fashion 
    //out of order, but since no sorting has been done, doesn't matter
    for(int i = 0; i < depth; i++)
    {
        myKey[i] =    ((bid*blockSize+i*blockDim.x + tid) < totalSize ? A_keys  [bid*blockSize+i*blockDim.x + tid] : MAX_VAL);
        myValue[i]  = ((bid*blockSize+i*blockDim.x + tid) < totalSize ? A_values[bid*blockSize+i*blockDim.x + tid] : 0);
    }	

    //Register Sort - Begin    
    compareSwapVal<T>(myKey[0], myKey[1], myValue[0], myValue[1]);	
    compareSwapVal<T>(myKey[1], myKey[2], myValue[1], myValue[2]);    
    compareSwapVal<T>(myKey[2], myKey[3], myValue[2], myValue[3]);
    compareSwapVal<T>(myKey[3], myKey[4], myValue[3], myValue[4]);
    compareSwapVal<T>(myKey[4], myKey[5], myValue[4], myValue[5]);    
    compareSwapVal<T>(myKey[5], myKey[6], myValue[5], myValue[6]);
    compareSwapVal<T>(myKey[6], myKey[7], myValue[6], myValue[7]);
        
    compareSwapVal<T>(myKey[0], myKey[1], myValue[0], myValue[1]); 	
    compareSwapVal<T>(myKey[1], myKey[2], myValue[1], myValue[2]);    
    compareSwapVal<T>(myKey[2], myKey[3], myValue[2], myValue[3]);
    compareSwapVal<T>(myKey[3], myKey[4], myValue[3], myValue[4]);
    compareSwapVal<T>(myKey[4], myKey[5], myValue[4], myValue[5]);    
    compareSwapVal<T>(myKey[5], myKey[6], myValue[5], myValue[6]);
    
	compareSwapVal<T>(myKey[0], myKey[1], myValue[0], myValue[1]);
    compareSwapVal<T>(myKey[1], myKey[2], myValue[1], myValue[2]);		    
    compareSwapVal<T>(myKey[2], myKey[3], myValue[2], myValue[3]);
    compareSwapVal<T>(myKey[3], myKey[4], myValue[3], myValue[4]);
    compareSwapVal<T>(myKey[4], myKey[5], myValue[4], myValue[5]);

    compareSwapVal<T>(myKey[0], myKey[1], myValue[0], myValue[1]);
    compareSwapVal<T>(myKey[1], myKey[2], myValue[1], myValue[2]);
    compareSwapVal<T>(myKey[2], myKey[3], myValue[2], myValue[3]);
    compareSwapVal<T>(myKey[3], myKey[4], myValue[3], myValue[4]);

    compareSwapVal<T>(myKey[0], myKey[1], myValue[0], myValue[1]);
    compareSwapVal<T>(myKey[1], myKey[2], myValue[1], myValue[2]);
    compareSwapVal<T>(myKey[2], myKey[3], myValue[2], myValue[3]);

	compareSwapVal<T>(myKey[0], myKey[1], myValue[0], myValue[1]);
    compareSwapVal<T>(myKey[1], myKey[2], myValue[1], myValue[2]);

	compareSwapVal<T>(myKey[0], myKey[1], myValue[0], myValue[1]);

    //Register Sort - End	
    

    

    //Manually unroll save for performance
    //TODO: Use template unrolling?
    scratchPad[tid*depth  ] = myKey[0]; scratchPad[tid*depth+1] = myKey[1]; scratchPad[tid*depth+2] = myKey[2]; scratchPad[tid*depth+3] = myKey[3];
    scratchPad[tid*depth+4] = myKey[4]; scratchPad[tid*depth+5] = myKey[5]; scratchPad[tid*depth+6] = myKey[6]; scratchPad[tid*depth+7] = myKey[7];
    
    __syncthreads();	
    //now we merge
   


    unsigned int j;
    unsigned int mult = 1;	
    unsigned int steps = 128;

    //Seven Merge steps (2^7)
    while (mult < steps)
    {				
        unsigned int first, last;
        //Determine the search space for each thread
        first = (tid/(mult*2))*depth*2*mult;
        unsigned int midPoint = first+mult*depth;
        
        //If you are the "right" block or "left" block
        unsigned int addPart = threadIdx.x%(mult<<1) >= mult ? 1 : 0;
        //if "right" block search in "left", otherwise search in "right"
        if(addPart == 0)		
            first += depth*mult;				

        last = first+depth*mult-1;
        j = (first+last)/2;

        unsigned int startAddress = threadIdx.x*depth-midPoint;		
        unsigned int range = last-first;
        
        
        T cmpValue;										
        __syncthreads();		
    
        //Binary Search
        switch(range)
        {
        case 1023: bin_search_block<T, depth>(cmpValue, myKey[0], scratchPad, j, 256, addPart);          			 
        case 511: bin_search_block<T, depth>(cmpValue, myKey[0],  scratchPad, j, 128, addPart);            
        case 255: bin_search_block<T, depth>(cmpValue, myKey[0],  scratchPad, j, 64, addPart);            
        case 127: bin_search_block<T, depth>(cmpValue, myKey[0],  scratchPad, j, 32, addPart);			
        case 63: bin_search_block<T, depth>(cmpValue, myKey[0],   scratchPad, j, 16, addPart);	
        case 31: bin_search_block<T, depth>(cmpValue, myKey[0],   scratchPad, j, 8, addPart);			 
        case 15: bin_search_block<T, depth>(cmpValue, myKey[0],   scratchPad, j, 4, addPart);            
        case 7: bin_search_block<T, depth>(cmpValue, myKey[0],    scratchPad, j, 2, addPart);            
        case 3: bin_search_block<T, depth>(cmpValue, myKey[0],    scratchPad, j, 1, addPart);                        
        }		
        cmpValue = scratchPad[j];

        //Binary search done, some post search correction
        if(cmpValue < myKey[0] || (cmpValue == myKey[0] && addPart == 1))		
            cmpValue = scratchPad[++j];				
        if((cmpValue < myKey[0] || (cmpValue == myKey[0] && addPart == 1)) && j == last)			
            j++;	

        //Save first address, then perform linear searches
        __syncthreads();
        myAddress[0] = j + startAddress;		
        addressPad[myAddress[0]] = myValue[0]; //Save address in new slot, unless we want to ping-pong in shared memory need extra registers		
        lin_search_block<T, depth>(cmpValue, myKey[1], myAddress[1], scratchPad, addressPad, j, 1, last, startAddress, addPart);				
        addressPad[myAddress[1]] = myValue[1];
        lin_search_block<T, depth>(cmpValue, myKey[2], myAddress[2], scratchPad, addressPad, j, 2, last, startAddress, addPart);	
        addressPad[myAddress[2]] = myValue[2];
        lin_search_block<T, depth>(cmpValue, myKey[3], myAddress[3], scratchPad, addressPad, j, 3, last, startAddress, addPart);		
        addressPad[myAddress[3]] = myValue[3];
        lin_search_block<T, depth>(cmpValue, myKey[4], myAddress[4], scratchPad, addressPad, j, 4, last, startAddress, addPart);		
        addressPad[myAddress[4]] = myValue[4];
        lin_search_block<T, depth>(cmpValue, myKey[5], myAddress[5], scratchPad, addressPad, j, 5, last, startAddress, addPart);		
        addressPad[myAddress[5]] = myValue[5];
        lin_search_block<T, depth>(cmpValue, myKey[6], myAddress[6], scratchPad, addressPad, j, 6, last, startAddress, addPart);		
        addressPad[myAddress[6]] = myValue[6];
        lin_search_block<T, depth>(cmpValue, myKey[7], myAddress[7], scratchPad, addressPad, j, 7, last, startAddress, addPart);		
        addressPad[myAddress[7]] = myValue[7];       

        //Save Key values in correct addresses -- Unrolled for performance
        __syncthreads();
        scratchPad[myAddress[0]] = myKey[0]; scratchPad[myAddress[1]] = myKey[1]; scratchPad[myAddress[2]] = myKey[2]; scratchPad[myAddress[3]] = myKey[3];
        scratchPad[myAddress[4]] = myKey[4]; scratchPad[myAddress[5]] = myKey[5]; scratchPad[myAddress[6]] = myKey[6]; scratchPad[myAddress[7]] = myKey[7];		 
        __syncthreads();	
        
        
        if(mult < steps)
        {
            __syncthreads();
            //Grab new key values -- Unrolled for performance
            myKey[0] = scratchPad[tid*depth];     myKey[1] = scratchPad[tid*depth+1];   myKey[2] = scratchPad[tid*depth+2];   myKey[3] = scratchPad[tid*depth+3];
            myKey[4] = scratchPad[tid*depth+4];   myKey[5] = scratchPad[tid*depth+5];   myKey[6] = scratchPad[tid*depth+6];   myKey[7] = scratchPad[tid*depth+7];
            myValue[0] = addressPad[tid*depth];   myValue[1] = addressPad[tid*depth+1]; myValue[2] = addressPad[tid*depth+2]; myValue[3] = addressPad[tid*depth+3];
            myValue[4] = addressPad[tid*depth+4]; myValue[5] = addressPad[tid*depth+5]; myValue[6] = addressPad[tid*depth+6]; myValue[7] = addressPad[tid*depth+7];
        }
        __syncthreads();			
        mult*=2;
    }   
    __syncthreads();

    //Coalesced Write back to Memory
#pragma unroll
    for(int i=tid;i<blockSize && bid*blockSize+i < totalSize ;i+= CTA_BLOCK)
    {	
        
        A_keys[bid*blockSize+i] = scratchPad[i];
        A_values[bid*blockSize+i] = addressPad[i];			
    }   
        
}


/** @brief Merges the indices for the "lower" block (left block)
 *  
 * Utilizes a "ping-pong" strategy
 * @param[in] A_keys  Global array of keys to be merged
 * @param[in] A_values  Global array of values to be merged
 * @param[out] A_keys_out  Resulting array of keys merged
 * @param[out] A_values_out  Resulting array of values merged
 * @param[in] sizePerPartition Size of each partition being merged
 * @param[in] size Size of total Array being sorted
 **/
template<class T, int depth>
__global__
void simpleMerge_lower(T *A_keys, unsigned int* A_values, T *A_keys_out, unsigned int *A_values_out, int sizePerPartition, int size)
{
    //each block will be responsible for a submerge
    int myId = blockIdx.x; int tid = threadIdx.x;	
    int myStartIdxA = 2*myId*sizePerPartition;   int myStartIdxB = (2*myId+1)*sizePerPartition;  int myStartIdxC = myStartIdxA;	
    int partitionSizeB = sizePerPartition < (size - myStartIdxB) ? sizePerPartition : size - myStartIdxB;	

	T MAX_VAL = getMax<T>();
	T MIN_VAL = getMin<T>();
	unsigned int UMAX_VAL = getMax<unsigned int>();
    //__shared__ T BKeys[INTERSECT_B_BLOCK_SIZE_simple+2];	
#if (__CUDA_ARCH__ >= 200)
    extern __shared__ char shared[];
#else
	extern __shared__ unsigned int shared[];
#endif
    T* BKeys =  (T*) shared;
    T* BMax = (T*) &BKeys[INTERSECT_B_BLOCK_SIZE_simple];			
    T localMaxB, localMaxA, localMinB;					
    
    int globalCAddress;			
    int index, bIndex = 0, aIndex = 0;			
    bool breakout = false;

    T myKey[depth]; 
    unsigned int myValue[depth];
    
    
    //Load Registers
    if(aIndex + INTERSECT_A_BLOCK_SIZE_simple < sizePerPartition) 
    {
    #pragma unroll
        for(int i = 0;i < depth; i++) 
        { 
            myKey[i]   = A_keys  [myStartIdxA + aIndex + depth*tid + i]; 
            myValue[i] = A_values[myStartIdxA + aIndex + depth*tid + i]; 
        }	
    }
    else
    {
    #pragma unroll
        for(int i = 0;i < depth; i++) 
        { 
            myKey[i] =   (aIndex+depth*tid + i < sizePerPartition ? A_keys  [myStartIdxA + aIndex+ depth*tid + i]   : MAX_VAL); 
            myValue[i] = (aIndex+depth*tid + i < sizePerPartition ? A_values[myStartIdxA + aIndex+ depth*tid + i]   : UMAX_VAL);	
        }
    }	

    //load smem values
    if(bIndex + INTERSECT_B_BLOCK_SIZE_simple < partitionSizeB) 
    {
        int bi = tid;					
    #pragma unroll
        for(int i = 0;i < INTERSECT_B_BLOCK_SIZE_simple/CTASIZE_simple; i++, bi+=CTASIZE_simple) 
        {
            BKeys[bi] = A_keys[myStartIdxB + bIndex + bi]; 
        }
    }
    else 
    {
        int bi = tid;
    #pragma unroll
        for(int i = 0;i < INTERSECT_B_BLOCK_SIZE_simple/CTASIZE_simple; i++, bi+=CTASIZE_simple) 
        { BKeys[bi] = (bIndex + bi < partitionSizeB ? A_keys[myStartIdxB + bIndex + bi] : MAX_VAL);}
    }


    //Save localMaxA and localMaxB
    if(tid == CTASIZE_simple-1)		
        BMax[1] = myKey[depth-1];			
    if(tid == 0)
        BMax[0] =  (bIndex + INTERSECT_B_BLOCK_SIZE_simple - 1 < partitionSizeB ?
            A_keys[myStartIdxB + bIndex + INTERSECT_B_BLOCK_SIZE_simple - 1] : MAX_VAL);

    __syncthreads();	

    //Maximum values for B and A in this stream	
    localMinB = MIN_VAL;
    localMaxB = BMax[0]; 
    localMaxA = BMax[1];
    
    do
    {		
        __syncthreads();	
        globalCAddress = myStartIdxC + bIndex + aIndex + tid*depth;
        index = 0;
        
        
        if((!(myKey[depth-1] < localMinB || myKey[0] > localMaxB) ||  
            (bIndex+INTERSECT_B_BLOCK_SIZE_simple) >= (partitionSizeB-1)) && (aIndex + tid*depth) < sizePerPartition)
        {				
            binSearch_whole_lower<T>(BKeys, index, myKey[0]);

            T cmpValue = BKeys[index];
            if(cmpValue < myKey[0] && index < INTERSECT_B_BLOCK_SIZE_simple)			
                cmpValue = BKeys[++index];
    
            
            index = (cmpValue < myKey[0] ? index+1 : index);						
            
            
            //Save Key-Value Pair
            if((myKey[0] < localMaxB && myKey[0] > localMinB) || (bIndex+index) >= (partitionSizeB)  || (index > 0 && index <INTERSECT_B_BLOCK_SIZE_simple))
            {
        
                A_keys_out  [globalCAddress + index] = myKey[0]; A_values_out[globalCAddress + index] = myValue[0]; }
                
            while(BKeys[index] < myKey[1] && index < INTERSECT_B_BLOCK_SIZE_simple)				
                index++;				
            //save Key-Value Pair
            if(((myKey[1] <= localMaxB && myKey[1] > localMinB) || bIndex+index >= (partitionSizeB)) && (aIndex+tid*depth+1< sizePerPartition))										 
            { A_keys_out[globalCAddress+index+1] =  myKey[1];	A_values_out[globalCAddress+index+1] = myValue[1]; }			
        }		
        
        __syncthreads();
        if((localMaxA <= localMaxB || (bIndex+INTERSECT_B_BLOCK_SIZE_simple) >= partitionSizeB) && (aIndex+INTERSECT_A_BLOCK_SIZE_simple) < sizePerPartition)
        {	
            
        
    
            aIndex += INTERSECT_A_BLOCK_SIZE_simple;	
            
            if(aIndex + INTERSECT_A_BLOCK_SIZE_simple < sizePerPartition) {
                for(int i = 0;i < depth; i++) {		
                    myKey[i] =   A_keys  [myStartIdxA + aIndex+ depth*tid + i];	
                    myValue[i] = A_values[myStartIdxA + aIndex + depth*tid + i];
                }
            }
            else {
                for(int i = 0;i < depth; i++) {		
                    myKey[i] = (aIndex+depth*tid + i < sizePerPartition ? A_keys[myStartIdxA + aIndex+ depth*tid + i]   : MAX_VAL);					
                    myValue[i] = (aIndex+depth*tid + i < sizePerPartition ? A_values[myStartIdxA + aIndex+ depth*tid + i]   : UMAX_VAL);
                }
            }			
            if(tid == CTASIZE_simple-1)		
                BMax[1] = myKey[depth-1]; //localMaxA for all threads	
        }			
        else if(localMaxB < localMaxA && (bIndex+INTERSECT_B_BLOCK_SIZE_simple) < partitionSizeB)
        {				
            localMinB = localMaxB;
            //Use INT_MAX as an "invalid/no-value" type in case the streaming window cannot be filled
            bIndex += INTERSECT_B_BLOCK_SIZE_simple;			

            if(bIndex + INTERSECT_B_BLOCK_SIZE_simple < partitionSizeB) {
                    int bi = tid;					
            #pragma unroll
                    for(int i = 0;i < INTERSECT_B_BLOCK_SIZE_simple/CTASIZE_simple; i++, bi+=CTASIZE_simple) {
                        BKeys[bi] =   A_keys[myStartIdxB + bIndex + bi];
                    }
                }
                else {
                    int bi = tid;
            #pragma unroll
                    for(int i = 0;i < INTERSECT_B_BLOCK_SIZE_simple/CTASIZE_simple; i++, bi+=CTASIZE_simple) {
                        BKeys[bi] =   (bIndex + bi < partitionSizeB ? A_keys[myStartIdxB + bIndex + bi]   : MAX_VAL);
                    }
                }

            if(tid == 0)
                BMax[0] =  (bIndex + INTERSECT_B_BLOCK_SIZE_simple < partitionSizeB ? A_keys[myStartIdxB + bIndex + INTERSECT_B_BLOCK_SIZE_simple] : MAX_VAL);
        }
        else
            breakout = true;

    
        __syncthreads();

        localMaxB = BMax[0];
        localMaxA = BMax[1];					
    }
    while(!breakout);
}

/** @brief Merges the indices for the "upper" block (right block)
 *  
 * Utilizes a "ping-pong" strategy
 * @param[in] A_keys  Global array of keys to be merged
 * @param[in] A_values  Global array of values to be merged
 * @param[out] A_keys_out  Resulting array of keys merged
 * @param[out] A_values_out  Resulting array of values merged
 * @param[in] sizePerPartition Size of each partition being merged
 * @param[in] size Size of total Array being sorted
 **/
template<class T, int depth>
__global__
void simpleMerge_higher(T *A_keys, unsigned int* A_values, T* A_keys_out, unsigned int *A_values_out, int sizePerPartition, int size)
{

	T MAX_VAL = getMax<T>();
	T MIN_VAL = getMin<T>();
	unsigned int UMAX_VAL = getMax<unsigned int>();

    int myId = blockIdx.x;
    int myStartIdxB = 2*myId*sizePerPartition; 
    int myStartIdxA = (2*myId+1)*sizePerPartition; 
    int myStartIdxC = myStartIdxB;
    T nextMaxB, nextMaxA, localMaxB, localMinB;

    int partitionSizeA = (sizePerPartition < (size - myStartIdxA) ? sizePerPartition : size - myStartIdxA);

    
    int index, bIndex = 0, aIndex = 0;	
#if (__CUDA_ARCH__ >= 200)
    extern __shared__ char shared[];
#else
	extern __shared__ unsigned int shared[];
#endif
    T* BKeys =  (T*) shared;
    //__shared__ T BKeys[INTERSECT_B_BLOCK_SIZE_simple+3];			
    T* BMax = (T*) &BKeys[INTERSECT_B_BLOCK_SIZE_simple];	

    bool breakout = false;
    int tid = threadIdx.x;	
    
    T myKey[depth]; 
    unsigned int myValue[depth];
            
#pragma unroll
    for(int i =0; i <depth; i++)
    {
        myKey[i] =   (aIndex+depth*tid + i < partitionSizeA ? A_keys  [myStartIdxA + aIndex+depth*tid+i] : MAX_VAL);
        myValue[i] = (aIndex+depth*tid + i < partitionSizeA ? A_values[myStartIdxA + aIndex+depth*tid+i] : UMAX_VAL);
    }

    if(bIndex + INTERSECT_B_BLOCK_SIZE_simple < sizePerPartition) {
        int bi = tid;					
#pragma unroll
        for(int i = 0;i < INTERSECT_B_BLOCK_SIZE_simple/CTASIZE_simple; i++, bi+=CTASIZE_simple) {
            BKeys[bi] =   A_keys[myStartIdxB + bIndex + bi];
        }
    }
    else {
        int bi = tid;
#pragma unroll
        for(int i = 0;i < INTERSECT_B_BLOCK_SIZE_simple/CTASIZE_simple; i++, bi+=CTASIZE_simple) {
            BKeys[bi] =   (bIndex + bi < sizePerPartition ? A_keys[myStartIdxB + bIndex + bi]   : MAX_VAL);
        }
    }
    if(tid == CTASIZE_simple-1)
    {
        BMax[0] =  (bIndex + INTERSECT_B_BLOCK_SIZE_simple < sizePerPartition ? A_keys[myStartIdxB + bIndex + INTERSECT_B_BLOCK_SIZE_simple] : MAX_VAL);
        BMax[1] = (aIndex + INTERSECT_A_BLOCK_SIZE_simple < partitionSizeA ? A_keys[myStartIdxA + aIndex + INTERSECT_A_BLOCK_SIZE_simple] :  A_keys[myStartIdxA + partitionSizeA-1]+1);				
    }	
    __syncthreads();

    localMinB = MIN_VAL;
    localMaxB = BKeys[INTERSECT_B_BLOCK_SIZE_simple-1];	

    nextMaxB = BMax[0];		
    nextMaxA = BMax[1];						
        
    int globalCAddress;
    do
    {	
        __syncthreads();
        index = 0;
        globalCAddress = myStartIdxC + bIndex + aIndex + depth*tid;
        //if(myKey[0] >= DVAL1 && myKey[0] <= DVAL2 && sizePerPartition == 2048)
        //	printf("higher myKey0 %u %d %d\n", myKey[0], globalCAddress + index, aIndex+tid*depth);
        
        if((myKey[0] < nextMaxB && myKey[depth-1] >= localMinB ||  (bIndex+INTERSECT_B_BLOCK_SIZE_simple) >= sizePerPartition) && (aIndex+depth*tid < partitionSizeA))
        {	
            binSearch_whole_higher(BKeys, index, myKey[0]);
            T cmpValue = BKeys[index];
            if(cmpValue <= myKey[0] && index < INTERSECT_B_BLOCK_SIZE_simple)			
                cmpValue = BKeys[++index];

            index = (cmpValue <= myKey[0] ? index + 1 : index);


            //End Binary Search
            //binary search done for first element in our set (A_0)			
            if(myKey[0] >= localMinB)
            { A_keys_out[globalCAddress+index] = myKey[0]; A_values_out[globalCAddress+index] =  myValue[0];	}		

            while(BKeys[index] <= myKey[1] && index < INTERSECT_B_BLOCK_SIZE_simple )
                index++;

            //Save Key-Value Pair
            if(((myKey[1] < nextMaxB && myKey[1] >= localMinB) || bIndex+index >=sizePerPartition) && (aIndex+depth*tid+1 < partitionSizeA))	
            {
                A_keys_out[globalCAddress + index + 1] =  myKey[1]; A_values_out[globalCAddress + index + 1] =  myValue[1];	}			
                        
            }	

        // if(threadIdx.x == blockDim.x - 1) { *lastAIndex = index; }				
        __syncthreads();		

        if((nextMaxA <= nextMaxB || (bIndex+INTERSECT_B_BLOCK_SIZE_simple) >= sizePerPartition ) && (aIndex+INTERSECT_A_BLOCK_SIZE_simple)< partitionSizeA)
        {	
            aIndex += INTERSECT_A_BLOCK_SIZE_simple;
    
            //Use INT_MAX-1 as an "invalid/no-value" type in case we are out of values to check
#pragma unroll
            for(int i=0;i <depth;i++)
            {
                myKey[i]   = (aIndex+depth*tid+i   < partitionSizeA ? A_keys[myStartIdxA + aIndex + depth * tid + i]   : MAX_VAL);
                myValue[i] = (aIndex+depth*tid+i   < partitionSizeA ? A_values[myStartIdxA + aIndex + depth * tid + i]   : UMAX_VAL);
            }
    
            if(tid == CTASIZE_simple-1)		
            {
                BMax[1] = (aIndex + INTERSECT_A_BLOCK_SIZE_simple < partitionSizeA ? 
                    A_keys[myStartIdxA + aIndex + INTERSECT_A_BLOCK_SIZE_simple] : A_keys[myStartIdxA + partitionSizeA - 1] + 1);		
                BMax[2] = myKey[depth-1];
            }
        }			
        else if(nextMaxB <= nextMaxA && (bIndex+INTERSECT_B_BLOCK_SIZE_simple) < sizePerPartition)
        {				
            localMinB = localMaxB;
            //Use INT_MAX as an "invalid/no-value" type in case the streaming window cannot be filled
            bIndex += INTERSECT_B_BLOCK_SIZE_simple;	
            if(bIndex + INTERSECT_B_BLOCK_SIZE_simple < sizePerPartition) {
                    int bi = tid;					
            #pragma unroll
                    for(int i = 0;i < INTERSECT_B_BLOCK_SIZE_simple/CTASIZE_simple; i++, bi+=CTASIZE_simple) {
                        BKeys[bi] =   A_keys[myStartIdxB + bIndex + bi];
                    }
                }
                else {
                    int bi = tid;
            #pragma unroll
                    for(int i = 0;i < INTERSECT_B_BLOCK_SIZE_simple/CTASIZE_simple; i++, bi+=CTASIZE_simple) {
                        BKeys[bi] =   (bIndex + bi < sizePerPartition ? A_keys[myStartIdxB + bIndex + bi]   : MAX_VAL);
                    }
                }

            if(tid == 0)
                BMax[0] =  (bIndex + INTERSECT_B_BLOCK_SIZE_simple < sizePerPartition ? A_keys[myStartIdxB + bIndex + INTERSECT_B_BLOCK_SIZE_simple] : MAX_VAL);							
        }
        else
            breakout = true;	
        __syncthreads();

        //For each thread grab your value ranges for B and A	
        //These will look at the end of our window, and the beginning of the next window for A and B
        //We make decisions on whether to advance a window, or save our merged value based on these
        nextMaxB = BMax[0];		
        nextMaxA = BMax[1];		
        localMaxB = BKeys[INTERSECT_B_BLOCK_SIZE_simple-1];				
        
    }
    while(!breakout);

}



/** @brief Merges the indices for the "upper" block (right block)
 *  
 * Utilizes a "ping-pong" strategy
 * @param[in] A Global array of keys
 * @param[in] splitsPP  Global array of values to be merged
 * @param[in] numPartitions  number of partitions being considered
 * @param[in] partitionSize Size of each partition being considered
 * @param[out] partitionBeginA Where each partition/subpartition will begin in A
 * @param[out] partitionSizesA Size of each partition/subpartition in A
 * @param[in] sizeA Size of the entire array
 **/
template<class T>
__global__
void findMultiPartitions(T *A, int splitsPP, int numPartitions, int partitionSize,  int* partitionBeginA, int* partitionSizesA, int sizeA)
{
	T MY_INVALID = getMax<T>();
    int myId = threadIdx.x + blockIdx.x*blockDim.x;
    if (myId >= (numPartitions*splitsPP)/2)
        return;

    int myStartA, myEndA;		
    int testIdx;
    int subPartitionSize = partitionSize/splitsPP;
    int myPartitionId = myId/splitsPP;
    int mySubPartitionId = myId%splitsPP;
    int saveId = 2*(myPartitionId*splitsPP) + mySubPartitionId;

    // we are at the beginning of a partition
    myStartA = 2*(myPartitionId)*partitionSize + (mySubPartitionId)*subPartitionSize; 
    myEndA = myStartA + (subPartitionSize)-1;		

    
    T mySample = A[myStartA];		
    T nextSample, testSample, myStartSample, myEndSample;

    if(mySubPartitionId != 0);
    {
        //need to ensure that we don't start inbetween duplicates			
        testSample = (myId == 0 ? MY_INVALID : A[myStartA-1]);			
        int count = 1; testIdx = myStartA;
        // we have sampled in the middle of a repeated sequence search until we are at a new sequence
        while(testSample == mySample && (testIdx-count) >= 2*(myPartitionId)*partitionSize) 	
           testSample = A[testIdx - (count++)];
        myStartA = (testIdx - count+1); 		
    }
    partitionBeginA[saveId] = myStartA; //partitionBegin found for first set
    myStartSample = mySample;
    
    mySample = A[myEndA];	

    int count;
    if(mySubPartitionId!= splitsPP-1 )
    {
        //need to ensure that we don't start inbetween duplicates			
        testSample = (A[myEndA+1]);		
        count = 1; testIdx = myEndA;
        while(testSample == mySample && (testIdx+count) < (2*myPartitionId+1)*partitionSize ) 	
           testSample = A[testIdx + (count++)];	
        myEndA = myEndA+count-1;
        myEndSample = A[(myEndA < (2*myPartitionId+1)*partitionSize && myEndA < sizeA) ? myEndA+1 : sizeA-1];
    }
    else
    {
        myEndA = (2*myPartitionId)*partitionSize + partitionSize-1;			
        myEndSample = A[myEndA]; //<---Start Sample			
    }
    partitionSizesA[saveId] = myEndA-myStartA + 1;

    //Now that we have found our range for "A" we search our neighboring partition "B" for its corresponding range
    //Therefore partitions [0] will match with partitions [splitsPP] [1] with [splitsPP + 1] .... [splitsPP-1] [2*splitsPP-1]

    int myStartRange = (2*myPartitionId)*partitionSize + partitionSize;
    int myEndRange = min(myStartRange + partitionSize, sizeA);

    //search for myStartSample in between range
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
        if(testSample > myEndSample)
            myEndRange = mid;
                
        mid = (first+last)/2;		
        testSample = A[mid];
        if(mid == last || mid == first )
            break;
    }		
    while (testSample > myStartSample && mid > myStartRange)	
        testSample = A[--mid];
        
        
    nextSample = (mid > myStartRange ? A[mid] : MY_INVALID);
    if(testSample == nextSample)
    {
        while(testSample == nextSample && mid > myStartRange)
            testSample = A[--mid];
        
        
    }

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

    if(testSample <= myEndSample)
        mid++;

    nextSample = (mid < myEndRange ? A[mid] : MY_INVALID);

    while (myEndSample >= nextSample && mid < myEndRange)
        nextSample = A[++mid];



    myEndA = mid;
    if(mySubPartitionId % splitsPP== 0)
        myStartA = (2*myPartitionId)*partitionSize + partitionSize;
    if(mySubPartitionId % splitsPP == splitsPP-1)
        myEndA = (2*myPartitionId)*partitionSize + 2*partitionSize;
    
    if(myEndA > sizeA)
        myEndA = sizeA;

    
    partitionBeginA[saveId + splitsPP] = myStartA;
    partitionSizesA[saveId + splitsPP] = myEndA-myStartA;

    
}


/** @brief Blocks cooperatively Merge two partitions for the indices in the "lower" block (left block)
 *  
 * Utilizes a "ping-pong" strategy
 * @param[out] A_keys_out  Resulting array of keys merged
 * @param[out] A_vals_out  Resulting array of values merged
 * @param[in] A_keys  Global array of keys to be merged
 * @param[in] A_vals  Global array of values to be merged
 * @param[in] subPartitions Number of blocks working on a partition (number of sub-partitions)
 * @param[in] numBlocks 
 * @param[in] partitionBeginA Partition starting points decided by function findMultiPartitions
 * @param[in] partitionSizeA Partition sizes decided by function findMultiPartitions
 * @param[in] entirePartitionSize The size of an entire partition (before it is split up)
 * @param[in] sizeA The total size of our array
 **/
template<class T, int depth>
__global__
void mergeMulti_lower(T *A_keys_out, unsigned int* A_vals_out, T *A_keys, unsigned int *A_vals, int subPartitions, int numBlocks, int *partitionBeginA, int *partitionSizeA, 
                      int entirePartitionSize, int sizeA)
{

	T MAX_VAL = getMax<T>();
	T MIN_VAL = getMin<T>();

	unsigned int UMAX_VAL = getMax<unsigned int>();

    int myId = blockIdx.x; int tid = threadIdx.x;
    int myStartId = (myId%subPartitions) + 2*(myId/subPartitions)*subPartitions; 
    int myStartIdxA = partitionBeginA[myStartId];
    int myStartIdxB = partitionBeginA[myStartId+subPartitions];
    int localAPartSize = partitionSizeA[myStartId];
    int localBPartSize = partitionSizeA[myStartId+subPartitions];		
    int myStartIdxC;		
    myStartIdxC = myStartIdxA + myStartIdxB - ((myStartId+subPartitions)/subPartitions)*entirePartitionSize;	

    int partitionEndA = 2*(myId/subPartitions)*entirePartitionSize + entirePartitionSize < sizeA ? (myId/subPartitions)*entirePartitionSize*2 + entirePartitionSize : sizeA;
    int partitionEndB = partitionEndA + entirePartitionSize < sizeA ? partitionEndA + entirePartitionSize : sizeA;

    if(localAPartSize == 0)
        return;
    
    
    //Now we have the beginning and end points of our subpartitions, merge the two together			
    T nextMaxB, nextMaxA, localMinB, localMaxB, cmpValue;			
    int index, bIndex = 0; int aIndex = 0;	
    int localAIndex = aIndex+depth*tid;
#if (__CUDA_ARCH__ >= 200)
    extern __shared__ char shared[];
#else
	extern __shared__ unsigned int shared[];
#endif
    T* BKeys =  (T*) shared;
    //__shared__ T BKeys[INTERSECT_B_BLOCK_SIZE_multi+3];		
    T* BMax = (T*) &BKeys[INTERSECT_B_BLOCK_SIZE_multi];	
    
    bool breakout = false;	
    //bool endPartition = myStartIdxA+localAPartSize >= partitionEndA;
    
        

    if(myStartIdxB == 0 || myId%subPartitions == 0 || myStartIdxB == partitionEndA )
        localMinB = MIN_VAL;
    else
        localMinB = A_keys[myStartIdxB-1];

    T myKey[depth];
    unsigned int myVal[depth];

#pragma unroll
    for(int i =0; i <depth; i++)
    {
        myKey[i] = (localAIndex + i   < localAPartSize ? A_keys[myStartIdxA + aIndex+depth*tid+i]   : MAX_VAL);		
        myVal[i] = (localAIndex + i   < localAPartSize ? A_vals[myStartIdxA + aIndex+depth*tid+i]   : UMAX_VAL);
    }

    if(bIndex + INTERSECT_B_BLOCK_SIZE_multi < localBPartSize) {
        int bi = tid;					
#pragma unroll
        for(int i = 0;i < INTERSECT_B_BLOCK_SIZE_multi/CTASIZE_multi; i++, bi+=CTASIZE_multi) {
            BKeys[bi] =   A_keys[myStartIdxB + bIndex + bi];		
        }
    }
    else {
        int bi = tid;
#pragma unroll
        for(int i = 0;i < INTERSECT_B_BLOCK_SIZE_multi/CTASIZE_multi; i++, bi+=CTASIZE_multi) {			
            BKeys[bi] =   (bIndex + bi < localBPartSize ? A_keys[myStartIdxB + bIndex + bi]   : MAX_VAL);
            
        }
    }
    __syncthreads();	

    if(tid == CTASIZE_multi-1)
    {
        if(bIndex + INTERSECT_B_BLOCK_SIZE_multi < localBPartSize)
            BMax[0] = A_keys[myStartIdxB + bIndex + INTERSECT_B_BLOCK_SIZE_multi];
        else
            BMax[0] = (myStartIdxB + bIndex + localBPartSize < partitionEndB ? A_keys[myStartIdxB + bIndex + localBPartSize] : MAX_VAL);
        BMax[1] = (myStartIdxA + aIndex + INTERSECT_A_BLOCK_SIZE_multi < partitionEndA ? A_keys[myStartIdxA + aIndex + INTERSECT_A_BLOCK_SIZE_multi] :  MAX_VAL);		

    }	
    __syncthreads();	

    
    if(localBPartSize == 0)
    {
        localMaxB = MAX_VAL;
    }
    else
    {
        localMaxB = (localBPartSize < INTERSECT_B_BLOCK_SIZE_multi ? BKeys[localBPartSize-1] :  BKeys[INTERSECT_B_BLOCK_SIZE_multi-1]);
    }
    nextMaxB = BMax[0];	
    nextMaxA = BMax[1];

    do{				
        __syncthreads();		
        index = 0;		
        if(myKey[0] <= nextMaxB && myKey[depth-1] >= localMinB && localAIndex < localAPartSize)
        {		
            index = (INTERSECT_B_BLOCK_SIZE_multi/2)-1;			
            binSearch_fragment_lower<T> (BKeys, 256, index, myKey[0]);			
            binSearch_fragment_lower<T> (BKeys, 128, index, myKey[0]);			
            binSearch_fragment_lower<T> (BKeys, 64, index, myKey[0]);			
            binSearch_fragment_lower<T> (BKeys, 32, index, myKey[0]);			
            binSearch_fragment_lower<T> (BKeys, 16, index, myKey[0]);			
            binSearch_fragment_lower<T> (BKeys, 8, index, myKey[0]);			
            binSearch_fragment_lower<T> (BKeys, 4, index, myKey[0]);			
            binSearch_fragment_lower<T> (BKeys, 2, index, myKey[0]);			
            binSearch_fragment_lower<T> (BKeys, 1, index, myKey[0]);			

            cmpValue = BKeys[index];			
            if(cmpValue < myKey[0] && index < (localBPartSize-bIndex) && index < INTERSECT_B_BLOCK_SIZE_multi)				
                cmpValue = BKeys[++index];
            
            if(cmpValue < myKey[0])
                index++;

                
            int globalCAddress = (myStartIdxC + bIndex + aIndex + tid*depth);	

            //Save Key-Value Pair (after bin search)
            if(((myKey[0] <=  nextMaxB || myKey[0] <= localMaxB) && myKey[0] >= localMinB) && localAIndex < localAPartSize)
            { 
                A_keys_out[globalCAddress+index] = myKey[0]; A_vals_out[globalCAddress+index] = myVal[0]; 

            }

                        
            if(localAIndex + 1 < localAPartSize)
                linearMerge_lower<T, depth>(BKeys, myKey[1], myVal[1], index, A_keys_out, A_vals_out, myStartIdxC, nextMaxB,
                    localAPartSize, localBPartSize, localMaxB, localMinB, aIndex, bIndex, 1);

            if(localAIndex + 2 < localAPartSize)
                linearMerge_lower<T, depth>(BKeys, myKey[2], myVal[2], index, A_keys_out, A_vals_out, myStartIdxC, nextMaxB,
                    localAPartSize, localBPartSize, localMaxB, localMinB, aIndex, bIndex, 2);
            if(localAIndex + 3 < localAPartSize)
            {
                linearMerge_lower<T, depth>(BKeys, myKey[3], myVal[3], index, A_keys_out, A_vals_out, myStartIdxC, nextMaxB,
                    localAPartSize, localBPartSize, localMaxB, localMinB, aIndex, bIndex, 3);			
            }

        }		

        //We try to cleverly move the memory window ahead to get more overlap between our register window and smem window		 
        __syncthreads();		
        if((nextMaxA <= nextMaxB || (bIndex+INTERSECT_B_BLOCK_SIZE_multi) >= localBPartSize) && (aIndex+INTERSECT_A_BLOCK_SIZE_multi)< localAPartSize)
        {			
 
            aIndex += INTERSECT_A_BLOCK_SIZE_multi;	
            //Use INT_MAX-1 as an "invalid/no-value" type in case we are out of values to check
#pragma unroll
            for(int i=0;i <depth;i++)
            {
                myKey[i] = (aIndex+depth*tid + i   < localAPartSize  ? A_keys[myStartIdxA + aIndex+depth*tid+i]   : MAX_VAL);
                myVal[i] = (aIndex+depth*tid + i   < localAPartSize  ? A_vals[myStartIdxA + aIndex+depth*tid+i]   : UMAX_VAL);
            }
    
            if(tid == CTASIZE_multi-1)		
            {
                BKeys[INTERSECT_B_BLOCK_SIZE_multi+1] = (myStartIdxA + aIndex + INTERSECT_A_BLOCK_SIZE_multi < partitionEndA ? A_keys[myStartIdxA + aIndex + INTERSECT_A_BLOCK_SIZE_multi] :  MAX_VAL);		
                BKeys[INTERSECT_B_BLOCK_SIZE_multi+2] = myKey[depth-1];
            }
        }			
        else if(nextMaxB <= nextMaxA && (bIndex+INTERSECT_B_BLOCK_SIZE_multi) < localBPartSize)
        {				
            //Use INT_MAX as an "invalid/no-value" type in case the streaming window cannot be filled
            localMinB = nextMaxB;				
            bIndex += INTERSECT_B_BLOCK_SIZE_multi;	
            if(bIndex + INTERSECT_B_BLOCK_SIZE_multi < localBPartSize) {
                int bi = tid;					
        #pragma unroll
                for(int i = 0;i < INTERSECT_B_BLOCK_SIZE_multi/CTASIZE_multi; i++, bi+=CTASIZE_multi) {
                    BKeys[bi] =   A_keys[myStartIdxB + bIndex + bi];
                }
            }
            else {
                int bi = tid;
        #pragma unroll
                for(int i = 0;i < INTERSECT_B_BLOCK_SIZE_multi/CTASIZE_multi; i++, bi+=CTASIZE_multi) {
                    BKeys[bi] =   (bIndex + bi < localBPartSize ? A_keys[myStartIdxB + bIndex + bi]   : MAX_VAL);
                }
            }
            
            if(tid == CTASIZE_multi-1)
            {
                if(bIndex + INTERSECT_B_BLOCK_SIZE_multi < localBPartSize)
                    BMax[0] = A_keys[myStartIdxB + bIndex + INTERSECT_B_BLOCK_SIZE_multi];
                else
                    BMax[0] = (myStartIdxB + bIndex + localBPartSize < partitionEndB ? A_keys[myStartIdxB + bIndex + localBPartSize] : MAX_VAL);
            }
                            
        }
        else
            breakout = true;	
        __syncthreads();
        
            
        localMaxB = ( (localBPartSize-bIndex) < INTERSECT_B_BLOCK_SIZE_multi && (localBPartSize - bIndex) > 0 ? 
            BKeys[localBPartSize-bIndex-1] : BKeys[INTERSECT_B_BLOCK_SIZE_multi-1]);
        nextMaxB = BMax[0];		
        nextMaxA = BMax[1];						
        __syncthreads();
        
        
    }
    while(!breakout);

}

/** @brief Blocks cooperatively Merge two partitions for the indices in the "upper" block (right block)
 *  
 * Utilizes a "ping-pong" strategy
 * @param[out] A_keys_out  Resulting array of keys merged
 * @param[out] A_vals_out  Resulting array of values merged
 * @param[in] A_keys  Global array of keys to be merged
 * @param[in] A_vals  Global array of values to be merged
 * @param[in] subPartitions Number of blocks working on a partition (number of sub-partitions)
 * @param[in] numBlocks 
 * @param[in] partitionBeginA Partition starting points decided by function findMultiPartitions
 * @param[in] partitionSizeA Partition sizes decided by function findMultiPartitions
 * @param[in] entirePartitionSize The size of an entire partition (before it is split up)
 * @param[in] sizeA The total size of our array
 **/

template<class T, int depth>
__global__
void mergeMulti_higher(T *A_keys_out, unsigned int* A_vals_out, T *A_keys, unsigned int*A_vals, int subPartitions, int numBlocks, int *partitionBeginA, int *partitionSizeA, 
                       int entirePartitionSize, int sizeA)
{

	T MAX_VAL = getMax<T>();
	T MIN_VAL = getMin<T>();

	unsigned int UMAX_VAL = getMax<unsigned int>();

    int myId = blockIdx.x;
    int myStartId = (myId%subPartitions) + 2*(myId/subPartitions)*subPartitions;
    int myStartIdxB = partitionBeginA[myStartId];
    int myStartIdxA = partitionBeginA[myStartId+subPartitions];
    int myStartIdxC;
        
    int localBPartSize = partitionSizeA[myStartId];
    int localAPartSize = partitionSizeA[myStartId+subPartitions];


    if(localAPartSize == 0)
        return;
    
    myStartIdxC = myStartIdxA + myStartIdxB - ((myStartId+subPartitions)/subPartitions)*entirePartitionSize;		
    //Now we have the beginning and end points of our subpartitions, merge the two together
    T nextMaxB, nextMaxA, cmpValue, localMaxB, localMinB;

    int index, bIndex = 0, aIndex = 0;	

    
    T myKey[depth];
    unsigned int myVal[depth];
#if (__CUDA_ARCH__ >= 200)
    extern __shared__ char shared[];
#else
	extern __shared__ unsigned int shared[];
#endif
    T* BKeys =  (T*) shared;
    //__shared__ T BKeys[INTERSECT_B_BLOCK_SIZE_multi+3];
    T* BMax = (T*) &BKeys[INTERSECT_B_BLOCK_SIZE_multi];	
        
    bool breakout = false;
    int tid = threadIdx.x;
    int localAIndex = aIndex+depth*tid;

    //bool endPartition = (myId%subPartitions == subPartitions-1);

    int partitionEndB = (myId/subPartitions)*(entirePartitionSize*2) + entirePartitionSize < sizeA ? (myId/subPartitions)*entirePartitionSize*2 + entirePartitionSize : sizeA;
    int partitionEndA = partitionEndB + entirePartitionSize < sizeA ? partitionEndB + entirePartitionSize : sizeA;
    
    
#pragma unroll
    for(int i =0; i <depth; i++)
    {
        myKey[i] = (localAIndex+ i   < localAPartSize ? A_keys[myStartIdxA + localAIndex+i]   : MAX_VAL);
        myVal[i] = (localAIndex + i   < localAPartSize ? A_vals[myStartIdxA + localAIndex+i]   : UMAX_VAL);
    }
    
        
    if(bIndex + INTERSECT_B_BLOCK_SIZE_multi < localBPartSize) {
        int bi = tid;					
#pragma unroll
        for(int i = 0;i < INTERSECT_B_BLOCK_SIZE_multi/CTASIZE_multi; i++, bi+=CTASIZE_multi) {
            BKeys[bi] =   A_keys[myStartIdxB + bIndex + bi];
        }
    }
    else {
        int bi = tid;
#pragma unroll
        for(int i = 0;i < INTERSECT_B_BLOCK_SIZE_multi/CTASIZE_multi; i++, bi+=CTASIZE_multi) {
            BKeys[bi] =   (myStartIdxB + bIndex + bi < partitionEndB ? A_keys[myStartIdxB + bIndex + bi]   : MAX_VAL);
        }
    }



    if(tid == 0)
    {
        if(bIndex + INTERSECT_B_BLOCK_SIZE_multi < localBPartSize)
            BMax[0] = A_keys[myStartIdxB + bIndex + INTERSECT_B_BLOCK_SIZE_multi];
        else
            BMax[0] = (myStartIdxB + bIndex + localBPartSize < partitionEndB ? A_keys[myStartIdxB + bIndex + localBPartSize] : MAX_VAL);

        BMax[1] = (myStartIdxA + aIndex + INTERSECT_A_BLOCK_SIZE_multi < partitionEndA ? A_keys[myStartIdxA + aIndex + INTERSECT_A_BLOCK_SIZE_multi] : MAX_VAL);				
    }	
    __syncthreads();

    if(myStartIdxB == 0 || myId%subPartitions == 0 )
        localMinB = MIN_VAL;
    else
        localMinB = A_keys[myStartIdxB-1];
    
    
    if(localBPartSize == 0)
        localMaxB = MAX_VAL;
    else
        localMaxB = ( localBPartSize < INTERSECT_B_BLOCK_SIZE_multi ? BKeys[localBPartSize-1] : BKeys[INTERSECT_B_BLOCK_SIZE_multi-1]);
        
    nextMaxB = BMax[0];
    nextMaxA = BMax[1];
    
    do
    {
        
        
        __syncthreads();		
        index = 0;
        
        if((myKey[0] <= nextMaxB) && myKey[depth-1] >= localMinB && localAIndex < localAPartSize)		
        {	
    
            
            index = (INTERSECT_B_BLOCK_SIZE_multi/2)-1;
            
            binSearch_fragment_higher<T> (BKeys, 256, index, myKey[0]);			
            binSearch_fragment_higher<T> (BKeys, 128, index, myKey[0]);						
            binSearch_fragment_higher<T> (BKeys, 64, index, myKey[0]);			
            binSearch_fragment_higher<T> (BKeys, 32, index, myKey[0]);
            binSearch_fragment_higher<T> (BKeys, 16, index, myKey[0]);
            binSearch_fragment_higher<T> (BKeys, 8, index, myKey[0]);
            binSearch_fragment_higher<T> (BKeys, 4, index, myKey[0]);
            binSearch_fragment_higher<T> (BKeys, 2, index, myKey[0]);
            binSearch_fragment_higher<T> (BKeys, 1, index, myKey[0]);

            cmpValue = BKeys[index];
            
                                    
            if(cmpValue <= myKey[0] && index < INTERSECT_B_BLOCK_SIZE_multi)							
                cmpValue = BKeys[++index];

            if(cmpValue <= myKey[0])
                index++;
            index = index >= (localBPartSize-bIndex) ? localBPartSize-bIndex : index;
                    
            if((myKey[0] >= localMinB && myKey[0] < nextMaxB /*|| bIndex+index >= localBPartSize*/) && aIndex+depth*tid < localAPartSize)
            {
                A_keys_out[myStartIdxC + bIndex + aIndex+depth*tid+index] = myKey[0];											
                A_vals_out[myStartIdxC + bIndex + aIndex+depth*tid+index] = myVal[0];											
        
            }

     
            if(localAIndex + 1 < localAPartSize)
                linearMerge_higher<T, depth>(BKeys, myKey[1], myVal[1], index, A_keys_out, A_vals_out, myStartIdxC, localMinB, nextMaxB, aIndex, bIndex, 1, localAPartSize, localBPartSize);
            if(localAIndex+2 < localAPartSize)
                linearMerge_higher<T, depth>(BKeys, myKey[2], myVal[2], index, A_keys_out, A_vals_out, myStartIdxC, localMinB, nextMaxB, aIndex, bIndex, 2, localAPartSize, localBPartSize);
            
     
            if(localAIndex+3 < localAPartSize)
                linearMerge_higher<T, depth>(BKeys, myKey[3], myVal[3], index, A_keys_out, A_vals_out, myStartIdxC, localMinB, nextMaxB, aIndex, bIndex, 3, localAPartSize, localBPartSize);
            
        }
        
        
        __syncthreads();
        __threadfence();
        
        if((nextMaxA <= nextMaxB /*&& localMaxA != nextMaxB*/ || (bIndex+INTERSECT_B_BLOCK_SIZE_multi) >= localBPartSize) 
            && (aIndex+INTERSECT_A_BLOCK_SIZE_multi)< localAPartSize)
        {			

            aIndex += INTERSECT_A_BLOCK_SIZE_multi;
    
            //Use INT_MAX-1 as an "invalid/no-value" type in case we are out of values to check
#pragma unroll
            for(int i=0;i <depth;i++)
            {
                myKey[i] = (aIndex+depth*tid+i   < localAPartSize ? A_keys[myStartIdxA + aIndex+depth*tid+i]   : MAX_VAL);
                myVal[i] = (aIndex+depth*tid+i   < localAPartSize ? A_vals[myStartIdxA + aIndex+depth*tid+i]   : UMAX_VAL);
            }
    
            if(tid == CTASIZE_multi-1)		
            {
                BMax[1] = (myStartIdxA+aIndex + INTERSECT_A_BLOCK_SIZE_multi < partitionEndA ? A_keys[myStartIdxA + aIndex + INTERSECT_A_BLOCK_SIZE_multi] :  MAX_VAL);						
            }
        }			
        else if(nextMaxB <= nextMaxA && (bIndex+INTERSECT_B_BLOCK_SIZE_multi) < localBPartSize)
        {				
            localMinB = localMaxB;
            //Use INT_MAX as an "invalid/no-value" type in case the streaming window cannot be filled

            bIndex += INTERSECT_B_BLOCK_SIZE_multi;	
            if(bIndex + INTERSECT_B_BLOCK_SIZE_multi < localBPartSize) {
                int bi = tid;					
        #pragma unroll
                for(int i = 0;i < INTERSECT_B_BLOCK_SIZE_multi/CTASIZE_multi; i++, bi+=CTASIZE_multi) {
                    BKeys[bi] =   A_keys[myStartIdxB + bIndex + bi];
                }
            }
            else {
                int bi = tid;
        #pragma unroll
                for(int i = 0;i < INTERSECT_B_BLOCK_SIZE_multi/CTASIZE_multi; i++, bi+=CTASIZE_multi) {
                    BKeys[bi] =   (myStartIdxB+bIndex + bi < partitionEndB ? A_keys[myStartIdxB + bIndex + bi]   : MAX_VAL);
                }
            }

    
            if(tid == 0)
            {
                if(bIndex + INTERSECT_B_BLOCK_SIZE_multi < localBPartSize)
                    BMax[0] = A_keys[myStartIdxB + bIndex + INTERSECT_B_BLOCK_SIZE_multi];
                else
                    BMax[0] = (myStartIdxB + bIndex + localBPartSize < partitionEndB ? A_keys[myStartIdxB + bIndex + localBPartSize] : MAX_VAL);
            }
        
        }
        else
            breakout = true;	
        __syncthreads();
        

        //For each thread grab your value ranges for B and A	
        //These will look at the end of our window, and the beginning of the next window for A and B
        //We make decisions on whether to advance a window, or save our merged value based on these
        nextMaxB = BMax[0]; 
        nextMaxA = BMax[1]; 		
       
        localMaxB = ( (localBPartSize-bIndex) < INTERSECT_B_BLOCK_SIZE_multi && (localBPartSize - bIndex) > 0 ? 
            BKeys[localBPartSize-bIndex-1] : BKeys[INTERSECT_B_BLOCK_SIZE_multi-1]);
        __syncthreads();
        
        
    }
    while(!breakout);

}

/** @} */ // end MergeSort functions
/** @} */ // end cudpp_kernel


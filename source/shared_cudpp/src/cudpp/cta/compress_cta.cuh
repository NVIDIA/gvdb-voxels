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
 * compress_cta.cu
 *
 * @brief CUDPP CTA-level compress routines
 */

/** \addtogroup cudpp_cta 
 * @{
 */

/** @name Compress Functions
 * @{
 */

#include <stdio.h>
#include <cudpp_globals.h>

template<class T, int depth>
__device__  void 
binSearch_frag_mult(T* keyArraySmem, T* valueArraySmem, int offset, int &mid, T cmpValue, T testValue, int myAddress, int testGlobalIndex,
                    T* globalPointerArray, T* globalStringArray, int bIndex, size_t numElements)
{
    cmpValue = keyArraySmem[mid];
    if(cmpValue != testValue)
        mid = (cmpValue > testValue ? mid-offset : mid+offset);

    int count = 1;
    T cmpKey = cmpValue; 
    T tstKey = testValue;

    while(cmpKey == tstKey)
	{
	    tstKey = (myAddress+4*count > numElements-1) ? globalStringArray[myAddress + 4*count - numElements] : globalStringArray[myAddress + 4*count];
	    cmpKey = (valueArraySmem[mid] + 4*count > numElements-1) ?
		globalStringArray[valueArraySmem[mid] + 4*count - numElements] : globalStringArray[valueArraySmem[mid] + 4*count];

	    if(cmpKey > tstKey)
		mid -= offset;
	    else if(cmpKey < tstKey)
		mid += offset;

	    count++;
	}
}

template<class T, int depth>
__device__ void
linearStringMerge(T        *searchArray,       T       *pointerArray,
                  T        *A_values,          T       myKey,
                  T        myAddress,          int     &index,
                  T        &cmpValue,          T       *saveGlobalArray,
                  T        *savePointerArray,  T       *stringValues, 
                  int      myStartIdxC,        int     myStartIdxA,
                  int      myStartIdxB,        int     localAPartSize,
                  int      localBPartSize,     int     localCPartSize,
                  T        localMaxB,          T       finalMaxB,
                  T        localMinB,          int     tid,
                  int      aIndex,             int     bIndex,
                  int      offset,             int     subPartitions,
                  size_t   numElements)
{

    while(cmpValue < myKey && index < BWT_INTERSECT_B_BLOCK_SIZE_multi )
        cmpValue = searchArray[++index];

    bool breakNext = false;

    while(cmpValue == myKey && index < BWT_INTERSECT_B_BLOCK_SIZE_multi && !breakNext /*&& cmpValue != UINT_MAX*/)
	{
	    int count = 1;
	    T tmpKey = myKey;
	    T cmpKey = cmpValue;
	    while(tmpKey == cmpKey)
		{
		    tmpKey = (myAddress+4*count > numElements-1) ? stringValues[myAddress + 4*count - numElements] : stringValues[myAddress + 4*count];
		    cmpKey = (pointerArray[index] + 4*count > numElements-1) ?
			stringValues[pointerArray[index] + 4*count - numElements] : stringValues[pointerArray[index] + 4*count];

		    if(cmpKey < tmpKey)                 
			cmpValue = searchArray[++index];
		    else if(cmpKey > tmpKey || myAddress == pointerArray[index])
			breakNext = true;
		    count++;
		}                                                               
	}

    int globalCAddress = myStartIdxC + index + bIndex + aIndex + offset + tid*depth; 

    if(((myKey < localMaxB && myKey > localMinB) || bIndex+index >= (localBPartSize) || 
        (index > 0 && index <BWT_INTERSECT_B_BLOCK_SIZE_multi)) && globalCAddress < (myStartIdxC+localCPartSize) && myKey < finalMaxB)
	{
	    saveGlobalArray [globalCAddress] = myKey;       
	    savePointerArray[globalCAddress] = myAddress;   
	}                   
    else if((myKey == localMaxB  && myKey <= finalMaxB) && index == BWT_INTERSECT_B_BLOCK_SIZE_multi && globalCAddress <= (myStartIdxC+localCPartSize))
	{
	    unsigned int tmpAdd = myAddress;
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
	    if(tmpKey < cmpKey)
		{
		    saveGlobalArray [globalCAddress] = myKey;   
		    savePointerArray[globalCAddress] = myAddress;       
		}
	}
    else if(myKey == localMinB && globalCAddress < (myStartIdxC+localCPartSize))
	{
	    unsigned int tmpAdd = myAddress;
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
		    saveGlobalArray [globalCAddress] = myKey;   
		    savePointerArray[globalCAddress] = myAddress;       
		}
	}
}

template<class T, int depth>
__device__ void
binSearch_fragment(T*       binArray,
                   T*       pointerBinArray,
                   int      offset,
                   int      &mid,
                   T        cmpValue,
                   T        testValue,
                   T        myAddress,
                   T*       globalStringArray,
                   T*       globalStringArray2,
                   size_t   numElements)
{

    cmpValue = binArray[mid];
    if(cmpValue != testValue)
        mid = (cmpValue > testValue ? mid-offset : mid+offset);

    int count = 1;
    T cmpKey = cmpValue;

    while(cmpKey == testValue)
	{
	    testValue = (myAddress+4*count > numElements-1) ? globalStringArray2[myAddress + 4*count - numElements] : globalStringArray2[myAddress + 4*count];
	    cmpKey = (pointerBinArray[mid]+4*count > numElements-1) ?
		globalStringArray2[pointerBinArray[mid] + 4*count - numElements] : globalStringArray2[pointerBinArray[mid] + 4*count];

	    if(cmpKey > testValue)
		mid -= offset;
	    else if(cmpKey < testValue)
		mid += offset;

	    count++;
	}
}

template<class T, int depth>
__device__ void
lin_merge_simple(T      &cmpValue,
                 T      myKey,
                 T      myAddress,
                 int    &index,
                 T*     BKeys,
                 T*     BValues,
                 T*     stringValues,
                 T*     A_values,
                 T*     A_keys_out,
                 T*     A_values_out,
                 int    myStartIdxA,
                 int    myStartIdxB,
                 int    myStartIdxC,
                 T      localMinB,
                 T      localMaxB,
                 int    aCont,
                 int    bCont,
                 int    totalSize,
                 int    sizePerPartition,
                 int    i,
                 T*     stringValues2,
                 size_t numElements)
{
    while(cmpValue < myKey && index < BWT_INTERSECT_B_BLOCK_SIZE_simple)
        cmpValue = BKeys[++index];

    bool breakNext = false;

    while(cmpValue == myKey && index < BWT_INTERSECT_B_BLOCK_SIZE_simple && !breakNext)
	{
	    int count = 1;
	    T tmpKey = myKey;
	    T cmpKey = cmpValue;

	    while(tmpKey == cmpKey)
		{
		    tmpKey = (myAddress+4*count > numElements-1) ? stringValues2[myAddress + 4*count - numElements] : stringValues2[myAddress + 4*count];
		    cmpKey = (BValues[index]+4*count > numElements-1) ?
			stringValues2[BValues[index] + 4*count - numElements] : stringValues2[BValues[index] + 4*count];

		    if(cmpKey < tmpKey)
			cmpValue = BKeys[++index];
		    else if(cmpKey > tmpKey)
			breakNext = true;

		    count++;
		}
	}

    int globalCAddress = myStartIdxC + bCont + index + aCont + i;

    //Save Value if it is valid (correct window)
    //If we are on the edge of a window, and we are tied with the localMax or localMin value
    //we must go to global memory to find out if we are valid
    if((myKey < localMaxB && myKey > localMinB) || (index==BWT_INTERSECT_B_BLOCK_SIZE_simple && (bCont+index)>=sizePerPartition) ||
       (index > 0 && index <BWT_INTERSECT_B_BLOCK_SIZE_simple))
	{
	    A_keys_out[globalCAddress] = myKey;
	    A_values_out[globalCAddress] = myAddress;

	}
    else if(myKey == localMaxB && index == BWT_INTERSECT_B_BLOCK_SIZE_simple)
	{
	    unsigned int tmpAdd = myAddress;
	    unsigned int cmpAdd = A_values[myStartIdxB+bCont+index];
	    int          count = 1;
	    unsigned int tmpKey = (tmpAdd+4*count > numElements-1) ? stringValues2[tmpAdd + 4*count - numElements] : stringValues2[tmpAdd + 4*count];
	    unsigned int cmpKey = (cmpAdd+4*count > numElements-1) ? stringValues2[cmpAdd + 4*count - numElements] : stringValues2[cmpAdd + 4*count];

	    while(tmpKey == cmpKey)
		{
		    count++;
		    tmpKey = (tmpAdd+4*count > numElements-1) ? stringValues2[tmpAdd + 4*count - numElements] : stringValues2[tmpAdd + 4*count];
		    cmpKey = (cmpAdd+4*count > numElements-1) ? stringValues2[cmpAdd + 4*count - numElements] : stringValues2[cmpAdd + 4*count];
		}

	    if(tmpKey < cmpKey)
		{
		    A_keys_out[globalCAddress] = myKey;
		    A_values_out[globalCAddress] = myAddress;
		}

	}
    else if(myKey == localMinB)
	{
	    unsigned int    tmpAdd = myAddress;
	    unsigned int    cmpAdd = A_values[myStartIdxB+bCont+index];
	    int             count = 1;
	    unsigned int    tmpKey = (tmpAdd+4*count > numElements-1) ? stringValues2[tmpAdd + 4*count - numElements] : stringValues2[tmpAdd + 4*count];
	    unsigned int    cmpKey = (cmpAdd+4*count > numElements-1) ? stringValues2[cmpAdd + 4*count - numElements] : stringValues2[cmpAdd + 4*count];

	    while(tmpKey == cmpKey)
		{
		    count++;
		    tmpKey = (tmpAdd+4*count > numElements-1) ? stringValues2[tmpAdd + 4*count - numElements] : stringValues2[tmpAdd + 4*count];
		    cmpKey = (cmpAdd+4*count > numElements-1) ? stringValues2[cmpAdd + 4*count - numElements] : stringValues2[cmpAdd + 4*count];
		}
	    if(tmpKey > cmpKey)
		{
		    A_keys_out[globalCAddress] = myKey;
		    A_values_out[globalCAddress] = myAddress;
		}
	}
}

template<class T, int depth>
__device__ void bin_search_block(T &cmpValue,
                                 T tmpVal,
                                 T* in,
                                 T* addressPad,
                                 const T* stringVals,
                                 int & j,
                                 int bump,
                                 T* stringVals2,
                                 size_t numElements)
{
    cmpValue = in[j];

    if(cmpValue == tmpVal)
	{
	    T tmp = (addressPad[depth*threadIdx.x]+4*1 > numElements-1) ?
		stringVals2[addressPad[depth*threadIdx.x]+4*1-numElements] : stringVals2[addressPad[depth*threadIdx.x]+4*1];
	    T tmp2 = (addressPad[j]+4*1 > numElements-1) ? stringVals2[addressPad[j]+4*1-numElements] : stringVals2[addressPad[j]+4*1];

	    int i = 2;
	    while(tmp == tmp2)
		{
		    tmp = (addressPad[depth*threadIdx.x]+4*i > numElements-1) ?
			stringVals2[addressPad[depth*threadIdx.x]+4*i-numElements] : stringVals2[addressPad[depth*threadIdx.x]+4*i];
		    tmp2 = (addressPad[j]+4*i > numElements-1) ? stringVals2[addressPad[j]+4*i-numElements] : stringVals2[addressPad[j]+4*i];
		    i++;
		}

	    j = (tmp2 < tmp ? j + bump : j - bump);
	}
    else
        j = (cmpValue < tmpVal ? j + bump : j - bump);
    __syncthreads();

}

template<class T, int depth>
__device__ void lin_search_block(T &cmpValue,
                                 T &tmpVal,
                                 T* in,
                                 T* addressPad,
                                 const T* stringVals,
                                 int &j,
                                 int offset,
                                 int last,
                                 int startAddress,
                                 int addPart,
                                 T* stringVals2,
                                 size_t numElements)
{

    while (cmpValue < tmpVal && j < last)
        cmpValue = in[++j];

    //If we need to tie break while linearly searching
    while(cmpValue == tmpVal && j < last)
	{
	    T tmp = (addressPad[depth*threadIdx.x+offset]+4*1 > numElements-1) ?
		stringVals2[addressPad[depth*threadIdx.x+offset]+4*1-numElements] : stringVals2[addressPad[depth*threadIdx.x+offset]+4*1];
	    T tmp2 = (addressPad[j]+4*1 > numElements-1) ? stringVals2[addressPad[j]+4*1-numElements] : stringVals2[addressPad[j]+4*1];

	    int i = 2;
	    while(tmp == tmp2)
		{
		    tmp = (addressPad[depth*threadIdx.x+offset]+4*i > numElements-1) ?
		      stringVals2[addressPad[depth*threadIdx.x+offset]+4*i-numElements] : stringVals2[addressPad[depth*threadIdx.x+offset]+4*i];
		    tmp2 = (addressPad[j]+4*i > numElements-1) ? stringVals2[addressPad[j]+4*i-numElements] : stringVals2[addressPad[j]+4*i];
		    i++;
		}

	    if(tmp2 < tmp)
		cmpValue = in[++j];
	    else if(tmp2 > tmp)
		break;
	}

    //Corner case to handle being at the edge of our shared memory search
    j = ((j==last && cmpValue < tmpVal) ? j+1 : j);
    if (j == last && cmpValue == tmpVal)
	{
	    T tmp = (addressPad[depth*threadIdx.x+offset]+4*1 > numElements-1) ?
		stringVals2[addressPad[depth*threadIdx.x+offset]+4*1-numElements] : stringVals2[addressPad[depth*threadIdx.x+offset]+4*1];
	    T tmp2 = (addressPad[j]+4*1 > numElements-1) ? stringVals2[addressPad[j]+4*1-numElements] : stringVals2[addressPad[j]+4*1];

	    int i = 2;
	    while(tmp == tmp2)
		{
		    tmp = (addressPad[depth*threadIdx.x+offset]+4*i > numElements-1) ?
			stringVals2[addressPad[depth*threadIdx.x+offset]+4*i-numElements] : stringVals2[addressPad[depth*threadIdx.x+offset]+4*i];
		    tmp2 = (addressPad[j]+4*i > numElements-1) ? stringVals2[addressPad[j]+4*i-numElements] : stringVals2[addressPad[j]+4*i];
		    i++;
		}

	    if(tmp2 < tmp)
		j++;
	}
    tmpVal = j+startAddress+offset + addPart;
}

template<class T>
__device__ void
compareSwapVal(T            &A1,
               T            &A2,
               const int    index1,
               const int    index2,
               T*           scratch,
               const T*     stringVals,
               T*           stringVals2,
               size_t       numElements)
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
    else if(A1 == A2)
	{
	    T tmp = (scratch[index1]+4*1 > numElements-1) ? stringVals2[scratch[index1]+4*1-numElements] : stringVals2[scratch[index1]+1*4];
	    T tmp2 = (scratch[index2]+4*1 > numElements-1) ? stringVals2[scratch[index2]+4*1-numElements] : stringVals2[scratch[index2]+1*4];

	    int i = 2;
	    while(tmp == tmp2)
		{
		    tmp = (scratch[index1]+4*i > numElements-1) ? stringVals2[scratch[index1]+4*i-numElements] : stringVals2[scratch[index1]+4*i];
		    tmp2 = (scratch[index2]+4*i > numElements-1) ? stringVals2[scratch[index2]+4*i-numElements] : stringVals2[scratch[index2]+4*i];
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

__device__ void BitArraySetBit(huffman_code *ba, unsigned int bit)
{
    if (ba->numBits <= bit)
	{
	    return; // bit out of range
	}

    ba->code[bit/8] |= BIT_IN_CHAR(bit);
}

__device__ void BitArrayShiftLeft(huffman_code *ba, unsigned int shifts)
{
    int i, j;
    int chars = shifts / 8;  // number of whole byte shifts
    shifts = shifts % 8;     // number of bit shifts remaining

    if (shifts >= ba->numBits)
	{
	    // all bits have been shifted off
	    // set bits in all bytes to 0
	    memset((void *)(ba->code), 0, BITS_TO_CHARS(ba->numBits));
	    return;
	}

    // first handle big jumps of bytes
    if (chars > 0)
	{
	    for (i = 0; (i + chars) <  (int)BITS_TO_CHARS(ba->numBits); i++)
		{
		    ba->code[i] = ba->code[i + chars];
		}

	    // now zero out new bytes on the right
	    for (i = BITS_TO_CHARS(ba->numBits); chars > 0; chars--)
		{
		    ba->code[i - chars] = 0;
		}
	}

    // now we have at most CUDPP_CHAR_BIT - 1 bit shifts across the whole array
    for (i = 0; i < (int)shifts; i++)
	{
	    for (j = 0; j < (int)BIT_CHAR(ba->numBits - 1); j++)
		{
		    ba->code[j] <<= 1;

		    // handle shifts across byte bounds
		    if (ba->code[j + 1] & MS_BIT)
			{
			    ba->code[j] |= 0x01;
			}
		}

	    ba->code[BIT_CHAR(ba->numBits - 1)] <<= 1;
	}
}

__device__ void BitArrayShiftRight(huffman_code *ba, unsigned int shifts)
{
    int i, j;
    unsigned char mask;
    int chars = shifts / CUDPP_CHAR_BIT;  // number of whole byte shifts
    shifts = shifts % CUDPP_CHAR_BIT;     // number of bit shifts remaining

    if (shifts >= ba->numBits)
	{
	    // all bits have been shifted off
	    memset((void *)(ba->code), 0, BITS_TO_CHARS(ba->numBits));
	    return;
	}

    // first handle big jumps of bytes
    if (chars > 0)
	{
	    for (i = BIT_CHAR(ba->numBits - 1); (i - chars) >= 0; i--)
		{
		    ba->code[i] = ba->code[i - chars];
		}

	    // now zero out new bytes on the right
	    for (; chars > 0; chars--)
		{
		    ba->code[chars - 1] = 0;
		}
	}

    // now we have at most CUDPP_CHAR_BIT - 1 bit shifts across the whole array
    for (i = 0; i < (int)shifts; i++)
	{
	    for (j = BIT_CHAR(ba->numBits - 1); j > 0; j--)
		{
		    ba->code[j] >>= 1;

		    // handle shifts across byte bounds
		    if (ba->code[j - 1] & 0x01)
			{
			    ba->code[j] |= MS_BIT;
			}
		}

	    ba->code[0] >>= 1;
	}

    /***********************************************************************
     * zero any spare bits that are beyond the end of the bit array so
     * increment and decrement are consistent.
     ***********************************************************************/
    i = ba->numBits % CUDPP_CHAR_BIT;
    if (i != 0)
	{
	    mask = UCHAR_MAX << (CUDPP_CHAR_BIT - i);
	    ba->code[BIT_CHAR(ba->numBits - 1)] &= mask;
	}
}

__device__ int FindMinimumCount(my_huffman_node_t* ht, int elements)
{
    int currentIndex = HUFF_NONE;   // index with lowest count seen so far
    int currentCount = INT_MAX;     // lowest count seen so far
    int currentLevel = INT_MAX;     // level of lowest count seen so far

    // sequentially search array
    for (int i = 0; i < elements; i++)
    {
        // check for lowest count (or equally as low, but not as deep)
        if( (!ht[i].ignore) &&
            (ht[i].count < currentCount ||
            (ht[i].count == currentCount && ht[i].level < currentLevel)) )
        {
            currentIndex = i;
            currentCount = ht[i].count;
            currentLevel = ht[i].level;
        }
    }

    return currentIndex;
}

/** @} */ // end compress functions
/** @} */ // end cudpp_cta

// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// ------------------------------------------------------------- 
#ifndef   __STRINGSORT_H__
#define   __STRINGSORT_H__

#include "cudpp_globals.h"
#include "cudpp.h"
#include "cudpp_plan.h"

extern "C"
void allocStringSortStorage(CUDPPStringSortPlan* plan);

extern "C"
void freeStringSortStorage(CUDPPStringSortPlan* plan);



extern "C"
void cudppStringSortDispatch(unsigned int       *keys,
                            unsigned int        *values,
                            unsigned int        *stringVals,
                            size_t      numElements,
							size_t      stringArrayLength,
							unsigned char termC,
                            const       CUDPPStringSortPlan *plan);

//Some helper functions needed to transform data
extern "C"
void dotAdd(unsigned int* d_address,	
	   unsigned int* numSpaces,
	   unsigned int* packedAddress,
	   size_t numElements,
	   size_t stringArrayLength);

extern "C"
void calculateAlignedOffsets(unsigned int* d_address,
							 unsigned int* numSpaces,
							 unsigned char* d_stringVals, 
							 unsigned char termC,
							 size_t numElements,
							 size_t stringArrayLength);
extern "C"
void packStrings(unsigned int* packedStrings, 
						 unsigned char* d_stringVals, 
						 unsigned int* d_keys, 						 
						 unsigned int* packedAddress, 
						 unsigned int* address, 
						 size_t numElements, 	
						 size_t stringArrayLength,
						 unsigned char termC);

extern "C"
void unpackStrings(unsigned int* packedAddress,
				   unsigned int* packedAddressRef,
				   unsigned int* address,
				   unsigned int* addressRef,				   
				   size_t numElements);

#endif // __STRINGSORT_H__

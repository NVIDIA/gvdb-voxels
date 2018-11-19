// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
* @file
* cudpp_sa.h
*
* @brief Suffix Array functionality header file - contains CUDPP interface (not public)
*/

#ifndef _CUDPP_SA_H_
#define _CUDPP_SA_H_

class CUDPPSaPlan;

extern "C" 
void allocSaStorage(CUDPPSaPlan* plan);

extern "C" 
void freeSaStorage(CUDPPSaPlan* plan);

extern "C"
void cudppSuffixArrayDispatch(unsigned char* d_str, 
                              unsigned int* d_keys_sa, 
                              size_t d_str_length,
                              const CUDPPSaPlan *plan);

#endif // _CUDPP_SA_H_

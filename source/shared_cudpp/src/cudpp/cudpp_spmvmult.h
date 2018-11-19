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
* cudpp_spmvmult.h
*
* @brief Scan functionality header file - contains CUDPP interface (not public)
*/

#ifndef _CUDPP_SPMVMULT_H_
#define _CUDPP_SPMVMULT_H_

class CUDPPSparseMatrixVectorMultiplyPlan;

extern "C"
void allocSparseMatrixVectorMultiplyStorage(CUDPPSparseMatrixVectorMultiplyPlan *plan,
                                            const void                          *A,
                                            const unsigned int                  *rowindx,
                                            const unsigned int                  *indx);

extern "C"
void freeSparseMatrixVectorMultiplyStorage(CUDPPSparseMatrixVectorMultiplyPlan *plan);

extern "C"
void cudppSparseMatrixVectorMultiplyDispatch(void                                      *d_y,
                                             const void                                *d_x,
                                             const CUDPPSparseMatrixVectorMultiplyPlan *plan);

#endif // _CUDPP_SPMVMULT_H_

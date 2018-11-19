// ***************************************************************
//  cuDPP -- CUDA Data Parallel Primitives library
//  -------------------------------------------------------------
//  $Revision: 3505 $
//  $Date: 2007-07-06 09:26:06 -0700 (Fri, 06 Jul 2007) $
//  -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
 * @file
 * spmvmult_app.cu
 *
 * @brief CUDPP application-level scan routines
 */

/** \addtogroup cudpp_app
  *
  */

#include <cstdlib>
#include <cstdio>
#include <assert.h>

#include "cuda_util.h"
#include "cudpp.h"
#include "cudpp_util.h"
#include "cudpp_plan.h"
#include "cudpp_globals.h"
#include "kernel/spmvmult_kernel.cuh"

extern "C"
void cudppSegmentedScanDispatch (void                   *d_out, 
                                 const void             *d_idata,
                                 const unsigned int     *d_iflags,
                                 int                    numElements,
                                 const CUDPPSegmentedScanPlan *plan
                                );

/** @name Sparse Matrix-Vector Multiply Functions
 * @{
 */

/** @brief Perform matrix-vector multiply for sparse matrices and vectors of arbitrary size.
  *
  * This function performs the sparse matrix-vector multiply by executing four steps. 
  *
  * 1. The sparseMatrixVectorFetchAndMultiply() kernel does an element-wise multiplication of a
  *    each element e in CUDPPSparseMatrixVectorMultiplyPlan::m_d_A with the corresponding 
  *    (i.e. in the same row as the column index of e in CUDPPSparseMatrixVectorMultiplyPlan::m_d_A) 
  *    element in d_x and stores the product in CUDPPSparseMatrixVectorMultiplyPlan::m_d_prod. It 
  *    also sets all elements of CUDPPSparseMatrixVectorMultiplyPlan::m_d_flags to 0.
  *
  * 2. The sparseMatrixVectorSetFlags() kernel iterates over each element in 
  *    CUDPPSparseMatrixVectorMultiplyPlan::m_d_rowIndex and sets 
  *    the corresponding position (indicated by CUDPPSparseMatrixVectorMultiplyPlan::m_d_rowIndex) in 
  *    CUDPPSparseMatrixVectorMultiplyPlan::m_d_flags to 1.
  *
  * 3. Perform a segmented scan on CUDPPSparseMatrixVectorMultiplyPlan::m_d_prod with 
  *    CUDPPSparseMatrixVectorMultiplyPlan::m_d_flags as the flag vector. The output is 
  *    stored in CUDPPSparseMatrixVectorMultiplyPlan::m_d_prod.
  *
  * 4. The yGather() kernel goes over each element in CUDPPSparseMatrixVectorMultiplyPlan::m_d_rowFinalIndex 
  *    and picks the corresponding element (indicated by CUDPPSparseMatrixVectorMultiplyPlan::m_d_rowFinalIndex) 
  *    element from CUDPPSparseMatrixVectorMultiplyPlan::m_d_prod and stores it in d_y.
  *
  * @param[out] d_y The output array for the sparse matrix-vector multiply (y vector)
  * @param[in] d_x The input x vector
  * @param[in] plan Pointer to the CUDPPSparseMatrixVectorMultiplyPlan object which stores the 
  *                 configuration and pointers to temporary buffers needed by this routine
  */
template <class T>
void sparseMatrixVectorMultiply(
                                 T                       *d_y, 
                                 const T                 *d_x,  
                                 const CUDPPSparseMatrixVectorMultiplyPlan *plan
                                )
{
    unsigned int numEltsBlocks = 
        max(1, (int)ceil((double)plan->m_numNonZeroElements / 
                         ((double)SEGSCAN_ELTS_PER_THREAD * SCAN_CTA_SIZE)));
    
    bool fullBlock = 
        (plan->m_numNonZeroElements == (numEltsBlocks * SEGSCAN_ELTS_PER_THREAD * SCAN_CTA_SIZE));  

    dim3  gridElts(max(1, numEltsBlocks), 1, 1);
    dim3  threads(SCAN_CTA_SIZE, 1, 1);

    if (fullBlock)
        sparseMatrixVectorFetchAndMultiply<T, true><<<gridElts, threads>>>
            (plan->m_d_flags, (T*)plan->m_d_prod, (T*)plan->m_d_A, d_x, plan->m_d_index, (unsigned)plan->m_numNonZeroElements);
    else
        sparseMatrixVectorFetchAndMultiply<T, false><<<gridElts, threads>>>
            (plan->m_d_flags, (T*)plan->m_d_prod, (T*)plan->m_d_A, d_x, plan->m_d_index, (unsigned)plan->m_numNonZeroElements);

    unsigned int numRowBlocks = 
        max(1, (int)ceil((double)plan->m_numRows / 
                         ((double)SEGSCAN_ELTS_PER_THREAD * SCAN_CTA_SIZE)));

    dim3  gridRows(max(1, numRowBlocks), 1, 1);

    sparseMatrixVectorSetFlags<<<gridRows, threads>>>
        (plan->m_d_flags, plan->m_d_rowIndex, (unsigned)plan->m_numRows);

    cudppSegmentedScanDispatch ((T*)plan->m_d_prod, 
                                (const T*)plan->m_d_prod,
                                plan->m_d_flags,
                                (unsigned)plan->m_numNonZeroElements, plan->m_segmentedScanPlan);

    yGather<<<gridRows, threads>>>
        (d_y, (T*)plan->m_d_prod, plan->m_d_rowFinalIndex, (unsigned)plan->m_numRows); 
}

#ifdef __cplusplus
extern "C" 
{
#endif

// file scope
/** @brief Allocate intermediate product, flags and rowFindx (index of the last
  *        element of each row) array .
  *  
  * @param[in] plan Pointer to CUDPPSparseMatrixVectorMultiplyPlan class containing sparse 
  *             matrix-vector multiply options, number of non-zero elements and number 
  *             of rows which is used to compute storage requirements
  * @param[in]  A The matrix A
  * @param[in]  rowindx The indices of elements in A which are the first element of their row
  * @param[in]  indx The column number for each element in A
  */
void allocSparseMatrixVectorMultiplyStorage(CUDPPSparseMatrixVectorMultiplyPlan *plan,
                                            const void         *A,
                                            const unsigned int *rowindx, 
                                            const unsigned int *indx)
{
    
    switch(plan->m_config.datatype)
    {
    case CUDPP_INT:
        CUDA_SAFE_CALL(cudaMalloc(&(plan->m_d_prod),  
                                  plan->m_numNonZeroElements * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc(&(plan->m_d_A),  
                                  plan->m_numNonZeroElements * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemcpy(plan->m_d_A, (int *)A, 
                                  plan->m_numNonZeroElements * sizeof(int),
                                  cudaMemcpyHostToDevice) );
        break;
    case CUDPP_UINT:
        CUDA_SAFE_CALL(cudaMalloc(&(plan->m_d_prod),  
                                  plan->m_numNonZeroElements * sizeof(unsigned int)));
        CUDA_SAFE_CALL(cudaMalloc(&(plan->m_d_A),  
                                  plan->m_numNonZeroElements * sizeof(unsigned int)));
        CUDA_SAFE_CALL(cudaMemcpy(plan->m_d_A, (unsigned int *)A, 
                                  plan->m_numNonZeroElements * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice) );
        break;
    case CUDPP_FLOAT:
        CUDA_SAFE_CALL(cudaMalloc(&(plan->m_d_prod),  
                                  plan->m_numNonZeroElements * sizeof(float)));
        CUDA_SAFE_CALL(cudaMalloc(&(plan->m_d_A),  
                                  plan->m_numNonZeroElements * sizeof(float)));
        CUDA_SAFE_CALL(cudaMemcpy(plan->m_d_A, (float *)A, 
                                  plan->m_numNonZeroElements * sizeof(float),
                                  cudaMemcpyHostToDevice) );
        break;
    default:
        break;
    }

    CUDA_SAFE_CALL(cudaMalloc((void **)&(plan->m_d_flags),  
                              plan->m_numNonZeroElements * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&(plan->m_d_index),  
                              plan->m_numNonZeroElements * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&(plan->m_d_rowFinalIndex),  
                              plan->m_numRows * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&(plan->m_d_rowIndex),  
                              plan->m_numRows * sizeof(unsigned int)));

    CUDA_SAFE_CALL(cudaMemcpy(plan->m_d_rowFinalIndex, plan->m_rowFinalIndex, 
                              plan->m_numRows * sizeof(unsigned int),
                              cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(plan->m_d_rowIndex, rowindx, 
                               plan->m_numRows * sizeof(unsigned int),
                               cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(plan->m_d_index, indx, 
                               plan->m_numNonZeroElements * sizeof(unsigned int),
                               cudaMemcpyHostToDevice) );


    CUDA_CHECK_ERROR("allocSparseMatrixVectorMultiplyStorage");
}

/** @brief Deallocate intermediate product, flags and rowFindx (index of the last
  *        element of each row) array .
  *
  * These arrays must have been allocated by allocSparseMatrixVectorMultiplyStorage(), which is called
  * by the constructor of CUDPPSparseMatrixVectorMultiplyPlan.  
  *
  * @param[in] plan Pointer to CUDPPSparseMatrixVectorMultiplyPlan plan initialized by its constructor.
  */
void freeSparseMatrixVectorMultiplyStorage(CUDPPSparseMatrixVectorMultiplyPlan *plan)
{
    CUDA_CHECK_ERROR("freeSparseMatrixVectorMultiply");

    cudaFree(plan->m_d_prod);
    cudaFree(plan->m_d_A);
    cudaFree((void*)plan->m_d_flags);
    cudaFree((void*)plan->m_d_index);
    cudaFree((void*)plan->m_d_rowFinalIndex);
    cudaFree((void*)plan->m_d_rowIndex);

    plan->m_d_prod = 0;
    plan->m_d_A = 0;
    plan->m_d_flags = 0;
    plan->m_d_index = 0;
    plan->m_d_rowFinalIndex = 0;
    plan->m_d_rowIndex = 0;
    plan->m_numNonZeroElements = 0;
    plan->m_numRows = 0;
}

/** @brief Dispatch function to perform a sparse matrix-vector multiply
  * with the specified configuration.
  *
  * This is the dispatch routine which calls sparseMatrixVectorMultiply() with 
  * appropriate template parameters and arguments
  * 
  * @param[out] d_y The output vector for y = A*x
  * @param[in]  d_x The x vector for y = A*x
  * @param[in]  plan The sparse matrix plan and data
  */
void cudppSparseMatrixVectorMultiplyDispatch (
                                              void                                      *d_y,
                                              const void                                *d_x,
                                              const CUDPPSparseMatrixVectorMultiplyPlan *plan
                                             )                            
{    
    switch(plan->m_config.datatype)
    {
        case CUDPP_INT:
            sparseMatrixVectorMultiply<int>((int *)d_y, (int *)d_x, plan);
            break;
        case CUDPP_UINT:
            sparseMatrixVectorMultiply<unsigned int>((unsigned int *)d_y, (unsigned int *)d_x, 
                                                      plan);
                break;
        case CUDPP_FLOAT:
            sparseMatrixVectorMultiply<float>((float *)d_y, (float *)d_x, plan);
                break;
        default:
            break;
    }
}

#ifdef __cplusplus
}
#endif

/** @} */ // end sparse matrix-vector multiply functions
/** @} */ // end cudpp_app

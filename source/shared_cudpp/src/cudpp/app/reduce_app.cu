// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: 5636 $
// $Date: 2009-07-02 13:39:38 +1000 (Thu, 02 Jul 2009) $
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt 
// in the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
 * @file
 * reduce_app.cu
 *
 * @brief CUDPP application-level reduction routines
 */
 
#include <stdio.h>

#include "cuda_util.h"
#include "cudpp_plan.h"
#include "cudpp_util.h"
#include "kernel/reduce_kernel.cuh"

/** \addtogroup cudpp_app
  *
  */

/** @name Reduce Functions
 * @{
 */

/**
  * @brief Per-block reduction function
  *
  * This function dispatches the appropriate reduction kernel given the size of the blocks.
  *
  * @param[out] d_odata The output data pointer.  Each block writes a single output element.
  * @param[in]  d_idata The input data pointer.  
  * @param[in]  numElements The number of elements to be reduced.
  * @param[in]  plan A pointer to the plan structure for the reduction.
*/
template <class T, class Oper>
void reduceBlocks(T *d_odata, const T *d_idata, size_t numElements, const CUDPPReducePlan *plan)
{
    unsigned int numThreads = (unsigned int)(((unsigned int)numElements > 2 * plan->m_threadsPerBlock) ?
        plan->m_threadsPerBlock : max(1, ceilPow2((unsigned int)numElements) / 2));
    dim3 dimBlock(numThreads, 1, 1);
    unsigned int numBlocks =
        min(plan->m_maxBlocks,
        ((unsigned int)(numElements) +
         (2*plan->m_threadsPerBlock - 1)) / (2*plan->m_threadsPerBlock));

    dim3 dimGrid(numBlocks, 1, 1);
    int smemSize = plan->m_threadsPerBlock * sizeof(T);

    // choose which of the optimized versions of reduction to launch
    
    if (isPowerOfTwo((unsigned int)numElements))
    {
        switch (dimBlock.x)
        {
        case 512:
            reduce<T, Oper, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, (unsigned)numElements); break;
        case 256:
            reduce<T, Oper, 256, true><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, (unsigned)numElements); break;
        case 128:
            reduce<T, Oper, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, (unsigned)numElements); break;
        case 64:
            reduce<T, Oper, 64, true><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, (unsigned)numElements); break;
        case 32:
            reduce<T, Oper, 32, true><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, (unsigned)numElements); break;
        case 16:
            reduce<T, Oper, 16, true><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, (unsigned)numElements); break;
        case  8:
            reduce<T, Oper,  8, true><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, (unsigned)numElements); break;
        case  4:
            reduce<T, Oper,  4, true><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, (unsigned)numElements); break;
        case  2:
            reduce<T, Oper,  2, true><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, (unsigned)numElements); break;
        case  1:
            reduce<T, Oper,  1, true><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, (unsigned)numElements); break;
        }
    }
    else
    {
        switch (dimBlock.x)
        {
        case 512:
            reduce<T, Oper, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, (unsigned)numElements); break;
        case 256:
            reduce<T, Oper, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, (unsigned)numElements); break;
        case 128:
            reduce<T, Oper, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, (unsigned)numElements); break;
        case 64:
            reduce<T, Oper,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, (unsigned)numElements); break;
        case 32:
            reduce<T, Oper,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, (unsigned)numElements); break;
        case 16:
            reduce<T, Oper,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, (unsigned)numElements); break;
        case  8:
            reduce<T, Oper,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, (unsigned)numElements); break;
        case  4:
            reduce<T, Oper,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, (unsigned)numElements); break;
        case  2:
            reduce<T, Oper,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, (unsigned)numElements); break;
        case  1:
            reduce<T, Oper,   1, false><<< dimGrid, dimBlock, smemSize >>>(d_odata, d_idata, (unsigned)numElements); break;
        }
    }

     CUDA_CHECK_ERROR("Reduce");
}
/**
  * @brief Array reduction function.
  *
  * Performs multi-level reduction on large arrays using reduceBlocks().  
  *
  * @param [out] d_odata The output data pointer.  This is a pointer to a single element.
  * @param [in]  d_idata The input data pointer.  
  * @param [in]  numElements The number of elements to be reduced.
  * @param [in]  plan A pointer to the plan structure for the reduction.
*/
template <class Oper, class T>
void reduceArray(T *d_odata, const T *d_idata, size_t numElements, const CUDPPReducePlan *plan)
{
    unsigned int numBlocks =
        min(plan->m_maxBlocks,
        ((unsigned int)(numElements) +
         (2*plan->m_threadsPerBlock - 1)) / (2*plan->m_threadsPerBlock));

    if (numBlocks > 1)
    {
        reduceBlocks<T, Oper>((T*)plan->m_blockSums, d_idata, numElements, plan);
        reduceBlocks<T, Oper>(d_odata, (const T*)plan->m_blockSums, numBlocks, plan);
    }
    else
    {
        reduceBlocks<T, Oper>(d_odata, d_idata, numElements, plan);
    }
}

/** @brief Allocate intermediate arrays used by reductions.
  *
  * Reductions of large arrays must be split into multiple blocks, 
  * where each block is reduced by a single CUDA thread block.  
  * Each block writes its partial sum to global memory where it is reduced
  * to a single element in a second pass.
  *
  * @param [in,out] plan Pointer to CUDPPReducePlan object containing options and number 
  *                      of elements, which is used to compute storage requirements, and
  *                      within which intermediate storage is allocated.
  */
void allocReduceStorage(CUDPPReducePlan *plan)
{
    unsigned int blocks =
        min(plan->m_maxBlocks,
        ((unsigned int)(plan->m_numElements) +
         plan->m_threadsPerBlock - 1) / plan->m_threadsPerBlock);
  
    switch (plan->m_config.datatype)
    {
    case CUDPP_INT:
        cudaMalloc(&plan->m_blockSums, blocks * sizeof(int));
        break;
    case CUDPP_UINT:
        cudaMalloc(&plan->m_blockSums, blocks * sizeof(unsigned int));
        break;
    case CUDPP_SHORT:
        cudaMalloc(&plan->m_blockSums, blocks * sizeof(short));
        break;
    case CUDPP_USHORT:
        cudaMalloc(&plan->m_blockSums, blocks * sizeof(unsigned short));
        break;    
    case CUDPP_FLOAT:
        cudaMalloc(&plan->m_blockSums, blocks * sizeof(float));
        break;
    case CUDPP_DOUBLE:
        cudaMalloc(&plan->m_blockSums, blocks * sizeof(double));
        break;
    case CUDPP_LONGLONG:
        cudaMalloc(&plan->m_blockSums, blocks * sizeof(long long));
        break;
    case CUDPP_ULONGLONG:
        cudaMalloc(&plan->m_blockSums, blocks * sizeof(unsigned long long));
        break;
    default:
        //! @todo should this flag an error? 
        break;
    }
   
    CUDA_CHECK_ERROR("allocReduceStorage");
}

/** @brief Deallocate intermediate block sums arrays in a CUDPPReducePlan object.
  *
  * These arrays must have been allocated by allocScanStorage(), which is called
  * by the constructor of cudppReducePlan().  
  *
  * @param[in,out] plan Pointer to CUDPPReducePlan object initialized by allocScanStorage().
  */
void freeReduceStorage(CUDPPReducePlan *plan)
{
    cudaFree(plan->m_blockSums);

    CUDA_CHECK_ERROR("freeReduceStorage");

    plan->m_blockSums = 0;
}

/** @brief Dispatch function to perform a parallel reduction on an
  * array with the specified configuration.
  *
  * This is the dispatch routine which calls reduceArray() with 
  * appropriate template parameters and arguments to achieve the scan as 
  * specified in \a plan. 
  * 
  * @param[out] d_odata     The output array of scan results
  * @param[in]  d_idata     The input array
  * @param[in]  numElements The number of elements to scan
  * @param[in]  plan     Pointer to CUDPPReducePlan object containing reduce options
  *                      and intermediate storage
  */
void cudppReduceDispatch(void *d_odata, const void *d_idata, size_t numElements, const CUDPPReducePlan *plan)
{
    switch (plan->m_config.datatype)
    {
    case CUDPP_SHORT:
        switch (plan->m_config.op)
        {
        case CUDPP_ADD:
        default:
            reduceArray< OperatorAdd<short> >((short*)d_odata, (short*)d_idata, numElements, plan);
            break;
        case CUDPP_MULTIPLY:
            reduceArray< OperatorMultiply<short> >((short*)d_odata, (short*)d_idata, numElements, plan);
            break;
        case CUDPP_MAX:
            reduceArray< OperatorMax<short> >((short*)d_odata, (short*)d_idata, numElements, plan);
            break;
        case CUDPP_MIN:
            reduceArray< OperatorMin<short> >((short*)d_odata, (short*)d_idata, numElements, plan);
            break;
        }
        break;
    case CUDPP_USHORT:
        switch (plan->m_config.op)
        {
        case CUDPP_ADD:
        default:
            reduceArray< OperatorAdd<unsigned short> >((unsigned short*)d_odata, (unsigned short*)d_idata, numElements, plan);
            break;
        case CUDPP_MULTIPLY:
            reduceArray< OperatorMultiply<unsigned short> >((unsigned short*)d_odata, (unsigned short*)d_idata, numElements, plan);
            break;
        case CUDPP_MAX:
            reduceArray< OperatorMax<unsigned short> >((unsigned short*)d_odata, (unsigned short*)d_idata, numElements, plan);
            break;
        case CUDPP_MIN:
            reduceArray< OperatorMin<unsigned short> >((unsigned short*)d_odata, (unsigned short*)d_idata, numElements, plan);
            break;
        }
        break;
    case CUDPP_CHAR:
        switch (plan->m_config.op)
        {
        case CUDPP_ADD:
        default:
            reduceArray< OperatorAdd<char> >((char*)d_odata, (char*)d_idata, numElements, plan);
            break;
        case CUDPP_MULTIPLY:
            reduceArray< OperatorMultiply<char> >((char*)d_odata, (char*)d_idata, numElements, plan);
            break;
        case CUDPP_MAX:
            reduceArray< OperatorMax<char> >((char*)d_odata, (char*)d_idata, numElements, plan);
            break;
        case CUDPP_MIN:
            reduceArray< OperatorMin<char> >((char*)d_odata, (char*)d_idata, numElements, plan);
            break;
        }
        break;
    case CUDPP_UCHAR:
        switch (plan->m_config.op)
        {
        case CUDPP_ADD:
        default:
            reduceArray< OperatorAdd<unsigned char> >((unsigned char*)d_odata, (unsigned char*)d_idata, numElements, plan);
            break;
        case CUDPP_MULTIPLY:
            reduceArray< OperatorMultiply<unsigned char> >((unsigned char*)d_odata, (unsigned char*)d_idata, numElements, plan);
            break;
        case CUDPP_MAX:
            reduceArray< OperatorMax<unsigned char> >((unsigned char*)d_odata, (unsigned char*)d_idata, numElements, plan);
            break;
        case CUDPP_MIN:
            reduceArray< OperatorMin<unsigned char> >((unsigned char*)d_odata, (unsigned char*)d_idata, numElements, plan);
            break;
        }
        break;
    case CUDPP_INT:
        switch (plan->m_config.op)
        {
        case CUDPP_ADD:
        default:
            reduceArray< OperatorAdd<int> >((int*)d_odata, (int*)d_idata, numElements, plan);
            break;
        case CUDPP_MULTIPLY:
            reduceArray< OperatorMultiply<int> >((int*)d_odata, (int*)d_idata, numElements, plan);
            break;
        case CUDPP_MAX:
            reduceArray< OperatorMax<int> >((int*)d_odata, (int*)d_idata, numElements, plan);
            break;
        case CUDPP_MIN:
            reduceArray< OperatorMin<int> >((int*)d_odata, (int*)d_idata, numElements, plan);
            break;
        }
        break;
    case CUDPP_UINT:
        switch (plan->m_config.op)
        {
        case CUDPP_ADD:
        default:
            reduceArray< OperatorAdd<unsigned int> >((unsigned int*)d_odata, (unsigned int*)d_idata, numElements, plan);
            break;
        case CUDPP_MULTIPLY:
            reduceArray< OperatorMultiply<unsigned int> >((unsigned int*)d_odata, (unsigned int*)d_idata, numElements, plan);
            break;
        case CUDPP_MAX:
            reduceArray< OperatorMax<unsigned int> >((unsigned int*)d_odata, (unsigned int*)d_idata, numElements, plan);
            break;
        case CUDPP_MIN:
            reduceArray< OperatorMin<unsigned int> >((unsigned int*)d_odata, (unsigned int*)d_idata, numElements, plan);
            break;
        }
        break;
    case CUDPP_FLOAT:
        switch (plan->m_config.op)
        {
        case CUDPP_ADD:
        default:
            reduceArray< OperatorAdd<float> >((float*)d_odata, (float*)d_idata, numElements, plan);
            break;
        case CUDPP_MULTIPLY:
            reduceArray< OperatorMultiply<float> >((float*)d_odata, (float*)d_idata, numElements, plan);
            break;
        case CUDPP_MAX:
            reduceArray< OperatorMax<float> >((float*)d_odata, (float*)d_idata, numElements, plan);
            break;
        case CUDPP_MIN:
            reduceArray< OperatorMin<float> >((float*)d_odata, (float*)d_idata, numElements, plan);
            break;
        }
        break;
    case CUDPP_DOUBLE:
        switch (plan->m_config.op)
        {
        case CUDPP_ADD:
        default:
            reduceArray< OperatorAdd<double> >((double*)d_odata, (double*)d_idata, numElements, plan);
            break;
        case CUDPP_MULTIPLY:
            reduceArray< OperatorMultiply<double> >((double*)d_odata, (double*)d_idata, numElements, plan);
            break;
        case CUDPP_MAX:
            reduceArray< OperatorMax<double> >((double*)d_odata, (double*)d_idata, numElements, plan);
            break;
        case CUDPP_MIN:
            reduceArray< OperatorMin<double> >((double*)d_odata, (double*)d_idata, numElements, plan);
            break;
        }
        break;
    case CUDPP_LONGLONG:
        switch (plan->m_config.op)
        {
        case CUDPP_ADD:
        default:
            reduceArray< OperatorAdd<long long> >((long long*)d_odata, (long long*)d_idata, numElements, plan);
            break;
        case CUDPP_MULTIPLY:
            reduceArray< OperatorMultiply<long long> >((long long*)d_odata, (long long*)d_idata, numElements, plan);
            break;
        case CUDPP_MAX:
            reduceArray< OperatorMax<long long> >((long long*)d_odata, (long long*)d_idata, numElements, plan);
            break;
        case CUDPP_MIN:
            reduceArray< OperatorMin<long long> >((long long*)d_odata, (long long*)d_idata, numElements, plan);
            break;
        }
        break;
    case CUDPP_ULONGLONG:
        switch (plan->m_config.op)
        {
        case CUDPP_ADD:
        default:
            reduceArray< OperatorAdd<unsigned long long> >((unsigned long long*)d_odata, (unsigned long long*)d_idata, numElements, plan);
            break;
        case CUDPP_MULTIPLY:
            reduceArray< OperatorMultiply<unsigned long long> >((unsigned long long*)d_odata, (unsigned long long*)d_idata, numElements, plan);
            break;
        case CUDPP_MAX:
            reduceArray< OperatorMax<unsigned long long> >((unsigned long long*)d_odata, (unsigned long long*)d_idata, numElements, plan);
            break;
        case CUDPP_MIN:
            reduceArray< OperatorMin<unsigned long long> >((unsigned long long*)d_odata, (unsigned long long*)d_idata, numElements, plan);
            break;
        }
        break;
    default:
        break;
    }
}

/** @} */ // end reduce functions
/** @} */ // end cudpp_app

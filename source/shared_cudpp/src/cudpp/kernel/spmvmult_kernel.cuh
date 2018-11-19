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
 * spmvmult_kernel.cu
 *
 * @brief CUDPP kernel-level scan routines
 */

/** \defgroup cudpp_kernel CUDPP Kernel-Level API
  * The CUDPP Kernel-Level API contains functions that run on the GPU 
  * device across a grid of Cooperative Thread Array (CTA, aka Thread
  * Block).  These kernels are declared \c __global__ so that they 
  * must be invoked from host (CPU) code.  They generally invoke GPU 
  * \c __device__ routines in the CUDPP \link cudpp_cta CTA-Level API\endlink. 
  * Kernel-Level API functions are used by CUDPP 
  * \link cudpp_app Application-Level\endlink functions to implement their 
  * functionality.
  * @{
  */

/** @name Sparse Matrix-Vector multiply Functions
* @{
*/

#include <cudpp_globals.h>
#include <cudpp_util.h>

/** 
  * @brief Fetch and multiply kernel
  *
  * This __global__ device function takes an element from the vector d_A, finds its 
  * column in d_indx and multiplies the element from d_A with its 
  * corresponding (that is having the same row) element in d_x and stores the resulting
  * product in d_prod. It also sets all the elements of d_flags to 0.
  * 
  * Template parameter \a T is the datatype of the matrix A and x. 
  *
  * @param[out] d_flags The output flags array
  * @param[out] d_prod The output products array
  * @param[in] d_A The input matrix A
  * @param[in] d_x The input array x
  * @param[in] d_indx The input array of column indices for each element in A
  * @param[in] numNZElts The number of non-zero elements in matrix A
  */
template <class T, bool isFullBlock>
__global__ 
void sparseMatrixVectorFetchAndMultiply(
                                        unsigned int       *d_flags, 
                                        T                  *d_prod, 
                                        const T            *d_A,
                                        const T            *d_x,
                                        const unsigned int *d_indx,
                                        unsigned int       numNZElts
                                        )
{
// This method is currently slow
#if 0 
    typeToVector<T,4>::Result  tempData, outData;
    typeToVector<unsigned int,4>::Result tempIndx, tempFlags;

    typeToVector<T,4>::Result* inData = (typeToVector<T,4>::Result*)d_A;
    typeToVector<T,4>::Result* outProd = (typeToVector<T,4>::Result*)d_prod;

    typeToVector<unsigned int,4>::Result* inIndx = 
        (typeToVector<unsigned int,4>::Result*)d_indx;
    typeToVector<unsigned int,4>::Result* outFlags = 
        (typeToVector<unsigned int,4>::Result*)d_flags;

    unsigned int aiDev = blockIdx.x * (blockDim.x << 1) + threadIdx.x;
    unsigned int i = aiDev * 4;
    if (isFullBlock || ((i+3) < numNZElts) )
    {
        tempData = inData[aiDev];
        tempIndx = inIndx[aiDev];

        outData.x = tempData.x * d_x[tempIndx.x];
        outData.y = tempData.y * d_x[tempIndx.y];
        outData.z = tempData.z * d_x[tempIndx.z];
        outData.w = tempData.w * d_x[tempIndx.w];

        outProd[aiDev] = outData;

        tempFlags.x = tempFlags.y = tempFlags.z = tempFlags.w = 0;
        outFlags[aiDev] = tempFlags;
    }
    else
    {
        if (i < numNZElts)
        {
            d_prod[i] = d_A[i] * d_x[d_indx[i]];
            d_flags[i] = 0;
        }
        if ((i+1) < numNZElts)
        {
            d_prod[i+1] = d_A[i+1] * d_x[d_indx[i+1]];
            d_flags[i+1] = 0;
        }
        if ((i+2) < numNZElts)
        {
            d_prod[i+2] = d_A[i+2] * d_x[d_indx[i+2]];
            d_flags[i+2] = 0;
        }
        if ((i+3) < numNZElts)
        {
            d_prod[i+3] = d_A[i+3] * d_x[d_indx[i+3]];
            d_flags[i+3] = 0;
        }
    }

    unsigned int biDev = aiDev + blockDim.x;
    i = biDev * 4;
    if (isFullBlock || ((i+3) < numNZElts) )
    {
        tempData = inData[biDev];
        tempIndx = inIndx[biDev];

        outData.x = tempData.x * d_x[tempIndx.x];
        outData.y = tempData.y * d_x[tempIndx.y];
        outData.z = tempData.z * d_x[tempIndx.z];
        outData.w = tempData.w * d_x[tempIndx.w];

        outProd[biDev] = outData;

        tempFlags.x = tempFlags.y = tempFlags.z = tempFlags.w = 0;
        outFlags[biDev] = tempFlags;
    }
    else
    {
        if (i < numNZElts)
        {
            d_prod[i] = d_A[i] * d_x[d_indx[i]];
            d_flags[i] = 0;
        }
        if ((i+1) < numNZElts)
        {
            d_prod[i+1] = d_A[i+1] * d_x[d_indx[i+1]];
            d_flags[i+1] = 0;
        }
        if ((i+2) < numNZElts)
        {
            d_prod[i+2] = d_A[i+2] * d_x[d_indx[i+2]];
            d_flags[i+2] = 0;
        }
        if ((i+3) < numNZElts)
        {
            d_prod[i+3] = d_A[i+3] * d_x[d_indx[i+3]];
            d_flags[i+3] = 0;
        }
    }
#endif

#if 1
    bool isLastBlock = (blockIdx.x == (gridDim.x-1));
    unsigned int iGlobal = (blockIdx.x * (blockDim.x << 3)) + threadIdx.x;
    
    for (unsigned int i = 0; i < 8; ++i)
    {
        if (isFullBlock)
        {
            d_prod[iGlobal] = d_A[iGlobal] * d_x[d_indx[iGlobal]];
            d_flags[iGlobal] = 0;
        }
        else
        {
            if (isLastBlock)
            {
                if (iGlobal < numNZElts)
                {
                    d_prod[iGlobal] = d_A[iGlobal] * d_x[d_indx[iGlobal]];
                    d_flags[iGlobal] = 0;
                }
            }
            else
            {
                d_prod[iGlobal] = d_A[iGlobal] * d_x[d_indx[iGlobal]];
                d_flags[iGlobal] = 0;
            }
        }

        iGlobal += blockDim.x;
    }
#endif
    __syncthreads();
}

/** 
  * @brief Set Flags kernel
  *
  * This __global__ device function takes an element from the vector d_rowindx, 
  * and sets the corresponding position in d_flags to 1 
  *
  * @param[out] d_flags The output flags array
  * @param[in] d_rowindx The starting index of each row in the "flattened" version
  *                      of matrix A
  * @param[in] numRows The number of rows in matrix A
  */
__global__ 
void sparseMatrixVectorSetFlags(
                                unsigned int             *d_flags, 
                                const unsigned int       *d_rowindx, 
                                unsigned int             numRows
                                )
{
    unsigned int iGlobal = (blockIdx.x * (blockDim.x << 3)) + threadIdx.x;

    bool isLastBlock = (blockIdx.x == (gridDim.x-1));

    for (unsigned int i = 0; i < 8; ++i)
    {
        if (isLastBlock)
        {
            if (iGlobal < numRows)
            {
                d_flags[d_rowindx[iGlobal]] = 1;
            }
        }
        else
        {
            d_flags[d_rowindx[iGlobal]] = 1;
        }

        iGlobal += blockDim.x;
    }
    
    __syncthreads();
}

/** 
  * @brief Gather final y values kernel
  *
  * This __global__ device function takes an element from the vector d_rowFindx,
  * which for each row gives the index of the last element of that row, reads the
  * corresponding position in d_prod and write it in d_y
  *
  * Template parameter \a T is the datatype of the matrix A and x.
  *
  * @param[out] d_y The output result array
  * @param[in] d_prod The input products array (which now contains sums for each row)
  * @param[in] d_rowFindx The starting index of each row in the "flattened" version
  *                       of matrix A
  * @param[in] numRows The number of rows in matrix A
  */
template <class T>
__global__ 
void yGather(
             T                  *d_y, 
             const T            *d_prod,
             const unsigned int *d_rowFindx,
             unsigned int       numRows
             )
{
    
    unsigned int iGlobal = (blockIdx.x * (blockDim.x << 3)) + threadIdx.x;
    
    bool isLastBlock = (blockIdx.x == (gridDim.x-1));
    
    for (unsigned int i=0; i < 8; ++i)
    {
        if (isLastBlock)
        {
            if (iGlobal < numRows)
            {
                d_y[iGlobal] += d_prod[d_rowFindx[iGlobal]-1];
            }
        }
        else
        {
            d_y[iGlobal] += d_prod[d_rowFindx[iGlobal]-1];
        }

        iGlobal += blockDim.x;
    }

    __syncthreads();

}

/** @} */ // end sparse matrix vector multiply functions
/** @} */ // end cudpp_kernel

// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// -------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#include "cuda_util.h"
#include "cudpp_globals.h"
#include "cudpp.h"
#include "cudpp_util.h"
#include "cudpp_plan.h"
#include "cudpp_sa.h"

#include "kernel/compress_kernel.cuh"

using namespace std;


/**
 * @file
 * compress_app.cu
 *
 * @brief CUDPP application-level compress routines
 */

/** \addtogroup cudpp_app
 * @{
 */

/** @name Compress Functions
 * @{
 */

/** @brief Perform Huffman encoding
 *
 *
 * Performs Huffman encoding on the input data stream. The input data
 * stream is the output data stream from the previous stage (MTF) in our
 * compress stream.
 *
 * The input is given by the output of the Move-to-Front transform (MTF).
 * There are a few things that need to be store along with the compressed
 * data. We also store the word offset of the compressed data stream because
 * our data is compressed into indepedent blocks (word granularity) so that
 * they can be encoded and decoded in parallel. The number of independent blocks
 * is HUFF_THREADS_PER_BLOCK*HUFF_WORK_PER_THREAD.
 *
 *
 * @param[out] d_hist           Histogram array of the input data stream used for decoding.
 * @param[out] d_encodeOffset   An array of the word offsets of the independent compressed data blocks.
 * @param[out] d_compressedSize Pointer to the total size in words of all compressed data blocks combined.
 * @param[out] d_compressed     A pointer to the compressed data blocks.
 * @param[in]  numElements      Total number of input elements to compress.
 * @param[in]  plan             Pointer to the plan object used for this compress.
 *
 */
void huffmanEncoding(unsigned int               *d_hist,
                     unsigned int               *d_encodeOffset,
                     unsigned int               *d_compressedSize,
                     unsigned int               *d_compressed,
                     size_t                     numElements,
                     const CUDPPCompressPlan    *plan)
{
    unsigned char* d_input  = plan->m_d_mtfOut;

    // Set work dimensions
    size_t nCodesPacked = 0;
    size_t histBlocks = (numElements%(HUFF_WORK_PER_THREAD_HIST*HUFF_THREADS_PER_BLOCK_HIST)==0) ?
        numElements/(HUFF_WORK_PER_THREAD_HIST*HUFF_THREADS_PER_BLOCK_HIST) : numElements%(HUFF_WORK_PER_THREAD_HIST*HUFF_THREADS_PER_BLOCK_HIST)+1;
    size_t tThreads = ((numElements%HUFF_WORK_PER_THREAD) == 0) ? numElements/HUFF_WORK_PER_THREAD : numElements/HUFF_WORK_PER_THREAD+1;
    size_t nBlocks = ( (tThreads%HUFF_THREADS_PER_BLOCK) == 0) ? tThreads/HUFF_THREADS_PER_BLOCK : tThreads/HUFF_THREADS_PER_BLOCK+1;

    dim3 grid_hist(histBlocks, 1, 1);
    dim3 threads_hist(HUFF_THREADS_PER_BLOCK_HIST, 1, 1);

    dim3 grid_tree(1, 1, 1);
    dim3 threads_tree(128, 1, 1);

    dim3 grid_huff(nBlocks, 1, 1);
    dim3 threads_huff(HUFF_THREADS_PER_BLOCK, 1, 1);

    //---------------------------------------
    //  1) Build histogram from MTF output
    //---------------------------------------
    huffman_build_histogram_kernel<<< grid_hist, threads_hist>>>
        ((unsigned int*)d_input, plan->m_d_histograms, numElements);

    //----------------------------------------------------
    //  2) Compute final Histogram + Build Huffman codes
    //----------------------------------------------------
    huffman_build_tree_kernel<<< grid_tree, threads_tree>>>
        (d_input, plan->m_d_huffCodesPacked, plan->m_d_huffCodeLocations, plan->m_d_huffCodeLengths, plan->m_d_histograms,
         d_hist, plan->m_d_nCodesPacked, d_compressedSize, histBlocks, numElements);

    //----------------------------------------------
    //  3) Main Huffman encoding step (encode data)
    //----------------------------------------------
    CUDA_SAFE_CALL(cudaMemcpy((void*)&nCodesPacked,  plan->m_d_nCodesPacked, sizeof(size_t), cudaMemcpyDeviceToHost));
    huffman_kernel_en<<< grid_huff, threads_huff, nCodesPacked*sizeof(unsigned char)>>>
        ((uchar4*)d_input, plan->m_d_huffCodesPacked, plan->m_d_huffCodeLocations, plan->m_d_huffCodeLengths,
         plan->m_d_encoded, nCodesPacked, tThreads);

    //--------------------------------------------------
    //  4) Pack together encoded data to determine how
    //     much encoded data needs to be transferred
    //--------------------------------------------------
    huffman_datapack_kernel<<<grid_huff, threads_huff>>>
        (plan->m_d_encoded, d_compressed, d_compressedSize, d_encodeOffset);
}


/** @brief Perform the Move-to-Front Transform (MTF)
 *
 * Performs a Move-to-Front (MTF) transform on the input data stream.
 * The MTF transform is the second stage in our compress pipeline. The
 * MTF manipulates the input data stream to improve the performance of
 * entropy encoding.
 *
 * @param[in]  d_mtfIn      An array of the input data stream to perform the MTF transform on.
 * @param[out] d_mtfOut     An array to store the output of the MTF transform.
 * @param[in]  numElements  Total number of input elements of the MTF transform.
 * @param[in]  plan         Pointer to the plan object used for this MTF transform.
 *
 */
template <class T>
void moveToFrontTransform(unsigned char             *d_mtfIn,
                          unsigned char             *d_mtfOut,
                          size_t                    numElements,
                          const T                   *plan)
{
    unsigned int npad = numElements-1;
    npad |= npad >> 1;
    npad |= npad >> 2;
    npad |= npad >> 4;
    npad |= npad >> 8;
    npad |= npad >> 16;
    npad++;

    unsigned int nThreads = MTF_THREADS_BLOCK;
    unsigned int nLists = npad/MTF_PER_THREAD;
    unsigned int tThreads = npad/MTF_PER_THREAD;
    unsigned int offset = 2;

    bool fullBlocks = (tThreads%nThreads == 0);
    unsigned int nBlocks = (fullBlocks) ? (tThreads/nThreads) : (tThreads/nThreads + 1);

    //-------------------------------------------
    //  Initial MTF lists + Initial Reduction
    //-------------------------------------------

    // Set work-item dimensions
    dim3 grid(nBlocks, 1, 1);
    dim3 threads(nThreads, 1, 1);

    // Kernel call
    mtf_reduction_kernel<<< grid, threads>>>
        (d_mtfIn, plan->m_d_lists, plan->m_d_list_sizes, nLists, offset, numElements);
    if(nBlocks > 1)
    {
        //----------------------
        //  MTF Global Reduce
        //----------------------

        unsigned int init_offset = offset * nThreads;
        offset = init_offset;
        tThreads = nBlocks/2;
        fullBlocks = (tThreads%nThreads == 0);
        nBlocks = (fullBlocks) ? (tThreads/nThreads) : (tThreads/nThreads + 1);

        // Set work dimensions
        dim3 grid_GLred(nBlocks, 1, 1);
        dim3 threads_GLred(nThreads, 1, 1);

        while(offset <= nLists)
        {
            mtf_GLreduction_kernel<<< grid_GLred, threads_GLred>>>
                (plan->m_d_lists, plan->m_d_list_sizes, offset, tThreads, nLists);
            offset *= 2*nThreads;
        }

        //-----------------------------
        //  MTF Global Down-sweep
        //-----------------------------
        offset = nLists/4;
        unsigned int lastLevel = 0;

        // Work-dimensions
        dim3 grid_GLsweep(nBlocks, 1, 1);
        dim3 threads_GLsweep(nThreads, 1, 1);

        while(offset >= init_offset/2)
        {
            lastLevel = offset/nThreads;
            lastLevel = (lastLevel>=(init_offset/2)) ? lastLevel : init_offset/2;

            mtf_GLdownsweep_kernel<<< grid_GLsweep, threads_GLsweep>>>
                (plan->m_d_lists, plan->m_d_list_sizes, offset, lastLevel, nLists, tThreads);
            offset = lastLevel/2;
        }
    }

    //------------------------
    //      Local Scan
    //------------------------
    tThreads = npad/MTF_PER_THREAD;
    offset = 2;
    fullBlocks = (tThreads%nThreads == 0);
    nBlocks = (fullBlocks) ? (tThreads/nThreads) : (tThreads/nThreads + 1);

    dim3 grid_loc(nBlocks, 1, 1);
    dim3 threads_loc(nThreads, 1, 1);

    mtf_localscan_lists_kernel<<< grid_loc, threads_loc>>>
        (d_mtfIn, d_mtfOut, plan->m_d_lists, plan->m_d_list_sizes, nLists, offset, numElements);
}

/** @brief Perform the Burrows-Wheeler Transform (BWT)
 *
 * Performs the Burrows-Wheeler Transform (BWT) on a given
 * character string. The BWT is an algorithm which is commonly used
 * in compression applications, mainly bzip2. The BWT orders the
 * characters in such a way that the output tends to have many long
 * runs of repeated characters. This bodes well for later stages in
 * compression pipelines which perform better with repeated characters.
 *
 *
 * @param[in]  d_uncompressed       A char array of the input data stream to perform the BWT on.
 * @param[out] d_bwtIndex           The index at which the original string in the BWT sorts to.
 * @param[out] d_bwtOut             An array to store the output of the BWT.
 * @param[in]  numElements          Total number of input elements of the BWT.
 * @param[in]  plan                 Pointer to the plan object used for this BWT.
 *
 *
 */
template <class T>
void burrowsWheelerTransform(unsigned char              *d_uncompressed,
                             int                        *d_bwtIndex,
                             unsigned char              *d_bwtOut,
                             size_t                     numElements,
                             const T    *plan)
{
    size_t tThreads = (numElements%4 == 0) ? numElements/4 : numElements/4 + 1;
    size_t nThreads = BWT_CTA_BLOCK;
    bool fullBlocks = (tThreads%nThreads == 0);
    uint nBlocks = (fullBlocks) ? (tThreads/nThreads) : (tThreads/nThreads+1);
    dim3 grid_construct(nBlocks, 1, 1);
    dim3 threads_construct(nThreads, 1, 1);
    uint* d_result;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_result, sizeof(unsigned int)*(numElements+1)));

    cudppSuffixArrayDispatch((unsigned char*)d_uncompressed, (unsigned int*)d_result, numElements, plan->m_saPlan);
    d_result += 1;
    CUDA_SAFE_CALL(cudaMemcpy(plan->m_d_values, d_result, numElements*sizeof(uint), cudaMemcpyDeviceToDevice));
    d_result -= 1;
    CUDA_SAFE_CALL(cudaFree(d_result));

    bwt_compute_final_kernel<<< grid_construct, threads_construct >>>
            (d_uncompressed, plan->m_d_values, d_bwtIndex, d_bwtOut, numElements, tThreads);
}

/** @brief Wrapper for calling the Burrows-Wheeler Transform (BWT).
 *
 * This is a wrapper function for calling the BWT. This wrapper is used
 * internally via the compress application to call burrowsWheelerTransform().
 *
 *
 * @param[in]  d_in         A char array of the input data stream to perform the BWT on.
 * @param[out] d_bwtIndex   The index at which the original string in the BWT sorts to.
 * @param[in]  numElements  Total number of input elements to the compress stream.
 * @param[in]  plan         Pointer to the plan object used for this compress.
 *
 *
 */
void burrowsWheelerTransformWrapper(unsigned char *d_in,
                                    int *d_bwtIndex,
                                    size_t numElements,
                                    const CUDPPCompressPlan *plan)
{
    burrowsWheelerTransform<CUDPPCompressPlan>(d_in, d_bwtIndex, plan->m_d_bwtOut, numElements, plan);
}

/** @brief Wrapper for calling the Burrows-Wheeler Transform (BWT).
 *
 * This is a wrapper function for calling the BWT. This wrapper is used
 * internally via the BWT primitive to call burrowsWheelerTransform().
 *
 *
 * @param[in]  d_in         A char array of the input data stream to perform the BWT on.
 * @param[out] d_bwtIndex   The index at which the original string in the BWT sorts to.
 * @param[out] d_bwtOut     An array to store the output of the BWT.
 * @param[in]  numElements  Total number of input elements to the BWT.
 * @param[in]  plan         Pointer to the plan object used for this BWT.
 *
 *
 */
void burrowsWheelerTransformWrapper(unsigned char *d_in,
                                    int *d_bwtIndex,
                                    unsigned char *d_bwtOut,
                                    size_t numElements,
                                    const CUDPPBwtPlan *plan)
{
    burrowsWheelerTransform<CUDPPBwtPlan>(d_in, d_bwtIndex, d_bwtOut, numElements, plan);
}

/** @brief Wrapper for calling the Move-to-Front (MTF) transform.
 *
 * This is a wrapper function for calling the MTF. This wrapper is used
 * internally via the compress application to call moveToFrontTransform().
 *
 *
 * @param[in]  numElements  Total number of input elements to the MTF transform.
 * @param[in]  plan         Pointer to the plan object used for this compress.
 *
 *
 */
void moveToFrontTransformWrapper(size_t numElements,
                                 const CUDPPCompressPlan *plan)
{
    moveToFrontTransform<CUDPPCompressPlan>(plan->m_d_bwtOut, plan->m_d_mtfOut, numElements, plan);
}

/** @brief Wrapper for calling the Move-to-Front (MTF) transform.
 *
 * This is a wrapper function for calling the MTF. This wrapper is used
 * internally via the MTF primitive to call moveToFrontTransform().
 *
 *
 * @param[in]  d_in         An input char array to perform the MTF on.
 * @param[in]  d_mtfOut     An output char array to store the MTF transformed
                            stream.
 * @param[in]  numElements  Total number of input elements to the MTF transform.
 * @param[in]  plan         Pointer to the plan object used for this MTF.
 *
 *
 */
void moveToFrontTransformWrapper(unsigned char *d_in,
                                 unsigned char *d_mtfOut,
                                 size_t numElements,
                                 const CUDPPMtfPlan *plan)
{
    moveToFrontTransform<CUDPPMtfPlan>(d_in, d_mtfOut, numElements, plan);
}

#ifdef __cplusplus
extern "C"
{
#endif

/** @brief Allocate intermediate arrays used by BWT.
 *
 *
 * @param [in,out] plan Pointer to CUDPPBwtPlan object containing options and number
 *                      of elements, which is used to compute storage requirements, and
 *                      within which intermediate storage is allocated.
 */
void allocBwtStorage(CUDPPBwtPlan *plan)
{
    size_t numElts = plan->m_numElements;

    // BWT
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_values), numElts*sizeof(unsigned int) ));

}

/** @brief Allocate intermediate arrays used by MTF.
 *
 *
 * @param [in,out] plan Pointer to CUDPPMtfPlan object containing
 *                      options and number of elements, which is used
 *                      to compute storage requirements, and within
 *                      which intermediate storage is allocated.
 */
void allocMtfStorage(CUDPPMtfPlan *plan)
{
    // Number of padding
    size_t tmp = plan->m_numElements-1;
    tmp |= tmp >> 1;
    tmp |= tmp >> 2;
    tmp |= tmp >> 4;
    tmp |= tmp >> 8;
    tmp |= tmp >> 16;
    tmp++;
    plan->npad = tmp;

    // MTF
    CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_lists), (tmp/MTF_PER_THREAD)*256*sizeof(unsigned char)));
    CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_list_sizes), (tmp/MTF_PER_THREAD)*sizeof(unsigned short)));
    CUDA_SAFE_CALL(cudaMemset(plan->m_d_lists, 0, (tmp/MTF_PER_THREAD)*256*sizeof(unsigned char)));
    CUDA_SAFE_CALL(cudaMemset(plan->m_d_list_sizes, 0, (tmp/MTF_PER_THREAD)*sizeof(unsigned short)));
}

/** @brief Allocate intermediate arrays used by compression.
 *
 *
 * @param [in,out] plan Pointer to CUDPPCompressPlan object
 *                      containing options and number of elements,
 *                      which is used to compute storage
 *                      requirements, and within which intermediate
 *                      storage is allocated.
 */
void allocCompressStorage(CUDPPCompressPlan *plan)
{
    size_t numElts = plan->m_numElements;
    plan->npad = numElts;

    // BWT
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_values), numElts*sizeof(unsigned int) ));
    CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_bwtOut), numElts*sizeof(unsigned char) ));

    // MTF
    CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_lists), (numElts/MTF_PER_THREAD)*256*sizeof(unsigned char)));
    CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_list_sizes), (numElts/MTF_PER_THREAD)*sizeof(unsigned short)));
    CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_mtfOut), numElts*sizeof(unsigned char) ));

    // Huffman
    size_t numBitsAlloc = HUFF_NUM_CHARS*(HUFF_NUM_CHARS+1)/2;
    size_t numCharsAlloc = (numBitsAlloc%8 == 0) ? numBitsAlloc/8 : numBitsAlloc/8 + 1;
    size_t histBlocks = (numElts%(HUFF_WORK_PER_THREAD_HIST*HUFF_THREADS_PER_BLOCK_HIST)==0) ?
        numElts/(HUFF_WORK_PER_THREAD_HIST*HUFF_THREADS_PER_BLOCK_HIST) : numElts%(HUFF_WORK_PER_THREAD_HIST*HUFF_THREADS_PER_BLOCK_HIST)+1;
    size_t tThreads = ((numElts%HUFF_WORK_PER_THREAD) == 0) ? numElts/HUFF_WORK_PER_THREAD : numElts/HUFF_WORK_PER_THREAD+1;
    size_t nBlocks = ( (tThreads%HUFF_THREADS_PER_BLOCK) == 0) ? tThreads/HUFF_THREADS_PER_BLOCK : tThreads/HUFF_THREADS_PER_BLOCK+1;

    CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_huffCodesPacked), numCharsAlloc*sizeof(unsigned char) ));
    CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_huffCodeLocations), HUFF_NUM_CHARS*sizeof(size_t) ));
    CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_huffCodeLengths), HUFF_NUM_CHARS*sizeof(unsigned char) ));
    CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_histograms), histBlocks*256*sizeof(size_t) ));
    CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_nCodesPacked), sizeof(size_t)));
    CUDA_SAFE_CALL(cudaMalloc( (void**) &(plan->m_d_encoded), sizeof(encoded)*nBlocks));

    CUDA_CHECK_ERROR("allocCompressStorage");
}

/** @brief Deallocate intermediate block arrays in a CUDPPCompressPlan object.
 *
 *
 * @param[in,out] plan Pointer to CUDPPCompressPlan object initialized by allocCompressStorage().
 */
void freeCompressStorage(CUDPPCompressPlan *plan)
{
    // BWT
    CUDA_SAFE_CALL( cudaFree(plan->m_d_values));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_bwtOut));

    // MTF
    CUDA_SAFE_CALL( cudaFree(plan->m_d_lists));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_list_sizes));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_mtfOut));

    // Huffman
    CUDA_SAFE_CALL(cudaFree(plan->m_d_histograms));
    CUDA_SAFE_CALL(cudaFree(plan->m_d_huffCodeLengths));
    CUDA_SAFE_CALL(cudaFree(plan->m_d_huffCodesPacked));
    CUDA_SAFE_CALL(cudaFree(plan->m_d_huffCodeLocations));
    CUDA_SAFE_CALL(cudaFree(plan->m_d_nCodesPacked));
    CUDA_SAFE_CALL(cudaFree(plan->m_d_encoded));

    CUDA_CHECK_ERROR("freeCompressStorage");
}

/** @brief Deallocate intermediate block arrays in a CUDPPBwtPlan object.
 *
 *
 * @param[in,out] plan Pointer to CUDPPBwtPlan object initialized by allocBwtStorage().
 */
void freeBwtStorage(CUDPPBwtPlan *plan)
{
    // BWT
    CUDA_SAFE_CALL( cudaFree(plan->m_d_values));

}

/** @brief Deallocate intermediate block arrays in a CUDPPMtfPlan object.
 *
 *
 * @param[in,out] plan Pointer to CUDPPMtfPlan object initialized by allocMtfStorage().
 */
void freeMtfStorage(CUDPPMtfPlan *plan)
{
    // MTF
    CUDA_SAFE_CALL( cudaFree(plan->m_d_lists));
    CUDA_SAFE_CALL( cudaFree(plan->m_d_list_sizes));
}

/** @brief Dispatch function to perform parallel compression on an
 *         array with the specified configuration.
 *
 *
 * @param[in]  d_uncompressed Uncompressed data
 * @param[out] d_bwtIndex BWT Index
 * @param[out] d_histSize Histogram size
 * @param[out] d_hist Histogram
 * @param[out] d_encodeOffset Encoded offset table
 * @param[out] d_compressedSize Size of compressed data
 * @param[out] d_compressed Compressed data
 * @param[in]  numElements Number of elements to compress
 * @param[in]  plan     Pointer to CUDPPCompressPlan object containing
 *                      compress options and intermediate storage
 */
void cudppCompressDispatch(void *d_uncompressed,
                           void *d_bwtIndex,
                           void *d_histSize, // ignore
                           void *d_hist,
                           void *d_encodeOffset,
                           void *d_compressedSize,
                           void *d_compressed,
                           size_t numElements,
                           const CUDPPCompressPlan *plan)
{
    // Call to perform the Burrows-Wheeler transform
    burrowsWheelerTransformWrapper((unsigned char*)d_uncompressed, (int*)d_bwtIndex,
                                   numElements, plan);

    // Call to perform the move-to-front transform
    moveToFrontTransformWrapper(numElements, plan);
    // Call to perform the Huffman encoding
    huffmanEncoding((unsigned int*)d_hist, (unsigned int*)d_encodeOffset,
                    (unsigned int*)d_compressedSize, (unsigned int*)d_compressed, numElements, plan);
}


/** @brief Dispatch function to perform the Burrows-Wheeler transform
 *
 *
 * @param[in]  d_in        Input data
 * @param[out] d_out       Transformed data
 * @param[out] d_index     BWT Index
 * @param[in]  numElements Number of elements to compress
 * @param[in]  plan        Pointer to CUDPPBwtPlan object containing
 *                         compress options and intermediate storage
 */
void cudppBwtDispatch(void *d_in,
                      void *d_out,
                      void *d_index,
                      size_t numElements,
                      const CUDPPBwtPlan *plan)
{
    // Call to perform the Burrows-Wheeler transform
    burrowsWheelerTransformWrapper((unsigned char*)d_in, (int*)d_index,
                                   (unsigned char*) d_out, numElements,
                                   plan);
}


/** @brief Dispatch function to perform the Move-to-Front transform
 *
 *
 * @param[in]  d_in        Input data
 * @param[out] d_out       Transformed data
 * @param[in]  numElements Number of elements to compress
 * @param[in]  plan        Pointer to CUDPPMtfPlan object containing
 *                         compress options and intermediate storage
 */
void cudppMtfDispatch(void *d_in,
                      void *d_out,
                      size_t numElements,
                      const CUDPPMtfPlan *plan)
{
    // Call to perform the Burrows-Wheeler transform
    moveToFrontTransformWrapper((unsigned char*) d_in,
                                (unsigned char*) d_out, numElements, plan);
}

#ifdef __cplusplus
}
#endif

/** @} */ // end compress functions
/** @} */ // end cudpp_app

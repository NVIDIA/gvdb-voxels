// --------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// --------------------------------------------------------------
// $Revision$
// $Date$
//---------------------------------------------------------------
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// --------------------------------------------------------------


#include <stdio.h>
#include <stdlib.h>

#include "cuda_util.h"
#include "cudpp_globals.h"
#include "cudpp.h"
#include "cudpp_util.h"
#include "cudpp_plan.h"

#include "moderngpu.cuh"
#include "cub/cub.cuh"
#include "kernel/sa_kernel.cuh"

#define SA_BLOCK 128

template <typename T>
struct my_less {
  __device__ bool operator()(T x, T y)
  {
      if(y.d == 1) return ((y.a == x.a) ? (x.c < y.b) : (x.a < y.a));
      else return ((y.a == x.a) ? ((y.b == x.b) ? (x.d<y.c) : (x.b<y.b)):(x.a<y.a));    
  }

};

typedef my_less<typename std::iterator_traits<Vector*>::value_type> Comp;
typedef typename std::iterator_traits<unsigned int*>::value_type T;

/**
  * @file
  * sa_app.cu
  *
  * @brief CUDPP application-level suffix array routines
  */

/** \addtogroup cudpp_app
 * @{
 */

/** @name Suffix Array Functions
 * @{
 */

/** @brief Radix Sort kernel from NVlab cub library.
 *
 * @param[in] num_elements      Number of elements to sort.
 * @param[in] d_keys            Key values of the elements in the array to be sorted.
 * @param[in, out] d_values     Positions of the elements in the array.
 */
void KeyValueSort(unsigned int num_elements,
                  unsigned int* d_keys,
                  unsigned int* d_values)
{
    using namespace cub;
    size_t temp_storage_bytes = 0;
    void *d_temp_storage = NULL;
    cub::DoubleBuffer<unsigned int> d_cub_keys;
    cub::DoubleBuffer<unsigned int> d_cub_values;
    d_cub_keys.d_buffers[d_cub_keys.selector] = d_keys;
    d_cub_values.d_buffers[d_cub_values.selector] = d_values;

    CUDA_SAFE_CALL(
        cudaMalloc((void**) &d_cub_keys.d_buffers[d_cub_keys.selector ^ 1],
                   sizeof(uint) * num_elements));
    CUDA_SAFE_CALL(
        cudaMalloc((void**) &d_cub_values.d_buffers[d_cub_values.selector ^ 1],
                   sizeof(uint) * num_elements));

    // Initialize the d_temp_storage
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    d_cub_keys, d_cub_values, num_elements);
    CUDA_SAFE_CALL(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Run
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    d_cub_keys, d_cub_values, num_elements);

    CUDA_SAFE_CALL(
        cudaMemcpy(d_keys, d_cub_keys.d_buffers[d_cub_keys.selector],
                   sizeof(uint) * num_elements, cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(
        cudaMemcpy(d_values, d_cub_values.d_buffers[d_cub_values.selector],
                   sizeof(uint) * num_elements, cudaMemcpyDeviceToDevice));

    cudaFree(d_temp_storage);

    // Cleanup "ping-pong" storage
    if (d_cub_values.d_buffers[1]) cudaFree(d_cub_values.d_buffers[1]);
    if (d_cub_keys.d_buffers[1]) cudaFree(d_cub_keys.d_buffers[1]);
}

/** @brief Perform Suffix Array (SA) using skew algorithm
 *
 *
 * Performs recursive skew kernel on a given character string.
 * A suffix array is a sorted array of all suffixes of a string.
 * Skew algorithm is a linear-time algorithm based on divde and conquer.
 * The SA of a string can be used as an index to quickly locate every
 * occurrence of a substring pattern within the string. Suffix sorting
 * algorithms can be used to compute the Burrows-Wheeler Transform(BWT).
 * The BWT requires sorting of all cyclic permutations of a string, thus
 * can be computed in linear time by using a suffix array of the string.
 *
 *
 * @param[in]      d_str           An unsigned int array of the input data stream to perform the SA on.
 * @param[out]     d_keys_sa:      An array to store the output of the SA.
 * @param[in]      str_length      Total number of input elements.
 * @param[in]      context         Context format required by mgpu functions.
 * @param[in,out]  plan            Pointer to the plan object used for this suffix array.
 * @param[in,out]  offset          Offset to move head pointer to find the memory for each iteration.
 * @param[in,out]  stage           Stage for each iteration.
 */

void ComputeSA(unsigned int* d_str,
               unsigned int* d_keys_sa,
               size_t str_length,
               mgpu::CudaContext& context,
               CUDPPSaPlan *plan,
               unsigned int offset,
               unsigned int stage)
{
    size_t mod_1 = (str_length+1)/3 + ((str_length+1)%3 > 0 ? 1:0);
    size_t mod_2 = (str_length+1)/3 + ((str_length+1)%3 > 1 ? 1:0);
    size_t mod_3 = (str_length+1)/3;
    size_t tThreads1 = mod_1+mod_2;
    size_t tThreads2 = mod_3;
    size_t bound = mod_1+mod_2+mod_3;

    bool *unique = new bool[1];
    unique[0] = 1;
    size_t nThreads = SA_BLOCK;
    bool fullBlocks1 = (tThreads1%nThreads==0);
    bool fullBlocks2 = (tThreads2%nThreads==0);
    size_t nBlocks1 = (fullBlocks1) ? (tThreads1/nThreads) :
                                      (tThreads1/nThreads+1);
    size_t nBlocks2 = (fullBlocks2) ? (tThreads2/nThreads) :
                                      (tThreads2/nThreads+1);
    dim3 grid_construct1(nBlocks1,1,1);
    dim3 grid_construct2(nBlocks2,1,1);
    dim3 threads_construct(nThreads,1,1);

    plan->m_d_keys_srt_12 = plan->m_d_keys_srt_12+offset;
    CUDA_SAFE_CALL(cudaMemcpy(plan->m_d_unique, unique, sizeof(bool),
                              cudaMemcpyHostToDevice));


    
   // extract the positions of i%3 != 0 to construct SA12
   // d_str: input,the original string
   // d_keys_sa: output, extracted string value with SA1 before SA2
   // m_d_keys_srt_12: output, store the positions of SA12 in original str
   ////////////////////////////////////////////////////////////////////
   sa12_keys_construct<<< grid_construct1, threads_construct >>>
         (d_str, d_keys_sa, plan->m_d_keys_srt_12, mod_1, tThreads1);
   
   // LSB radix sort the triplets character by character
   // d_keys_sa store the value of the character from the triplets r->l
   // m_d_keys_srt_12 store the sorted position of each char from each round
   // 3 round to sort the SA12 triplets
   /////////////////////////////////////////////////////////////////////
   KeyValueSort((mod_1+mod_2), d_keys_sa, plan->m_d_keys_srt_12);

   sa12_keys_construct_0<<< grid_construct1, threads_construct >>>
       (d_str, d_keys_sa, plan->m_d_keys_srt_12, tThreads1);

   KeyValueSort((mod_1+mod_2), d_keys_sa, plan->m_d_keys_srt_12);

   sa12_keys_construct_1<<< grid_construct1, threads_construct >>>
       (d_str, d_keys_sa, plan->m_d_keys_srt_12, tThreads1);

   KeyValueSort(tThreads1, d_keys_sa, plan->m_d_keys_srt_12);

    // Compare each SA12 position's rank to its previous position
    // and  mark 1 if different and 0 for same
    // Input: m_d_keys_srt_12, first round SA12 (may not fully sorted)
    // Output: d_keys_sa,1 if two position's ranks are the same
    //         m_d_unique,1 if SA12 are fully sorted
    ////////////////////////////////////////////////////////////////
    compute_rank<<< grid_construct1, threads_construct >>>
          (d_str, plan->m_d_keys_srt_12, d_keys_sa, plan->m_d_unique, tThreads1,
          str_length);

    CUDA_SAFE_CALL(cudaMemcpy(unique, plan->m_d_unique,  sizeof(bool),
                              cudaMemcpyDeviceToHost));

    // If not fully sorted
    if(!unique[0])
    {
        // Inclusive scan to compute the ranks of SA12
        plan->m_d_new_str = ((stage==0) ? plan->m_d_new_str :
                                          plan->m_d_new_str+offset+3);

        mgpu::Scan<mgpu::MgpuScanTypeInc>(
                d_keys_sa, (mod_1+mod_2), (T)0, mgpu::plus<T>(), (T*)0,
                (T*)0, d_keys_sa, context);

        // Construct new string with 2/3 str_length of original string
        // Place the ranks of SA1 before SA2 to construct the new str
        ///////////////////////////////////////////////////////////////
        new_str_construct<<< grid_construct1, threads_construct >>>
               (plan->m_d_new_str, plan->m_d_keys_srt_12, d_keys_sa, mod_1,
                tThreads1);

        // recurse
        ComputeSA(plan->m_d_new_str, plan->m_d_keys_srt_12, tThreads1-1,
                  context, plan, tThreads1, stage+1);
        plan->m_d_keys_srt_12 = plan->m_d_keys_srt_12 - tThreads1 ;

        // translate the sorted SA12 to original position and compute ISA12
        // Input: m_d_keys_srt_12, fully sorted SA12 named by local position
        // Output: m_d_isa_12, ISA12 to store the rank regard to local position
        // m_d_keys_srt_12, SA12 with regard to global position
        // d_keys_sa, flag to mark those with i mod 3 = 1 and i > 1
        ////////////////////////////////////////////////////////////////////
        reconstruct<<< grid_construct1, threads_construct >>>
               (plan->m_d_keys_srt_12, plan->m_d_isa_12, d_keys_sa, mod_1,
                tThreads1);

     }

// SA12 already fully sorted with results stored in d_keys_srt_12
// in their original position, no need to reconstruct, construct ISA12
// Input: m_d_keys_srt_12, fully sorted SA12 named by global position
// Output: m_d_isa_12, ISA12 to store the rank regard to local position
//         d_keys_sa, flag to mark those with i mod 3 = 1
//////////////////////////////////////////////////////////////////////
else
{
   isa12_construct<<< grid_construct1, threads_construct >>>
             (plan->m_d_keys_srt_12, plan->m_d_isa_12, d_keys_sa, mod_1,
              tThreads1);

}

// Exclusive scan to compute the position of SA1
  mgpu::ScanExc(d_keys_sa, (mod_1+mod_2), context);
  // Construct SA3 keys and positions based on SA1's ranks
  // Input: m_d_keys_srt_12, sorted SA12
  //        d_keys_sa, positions of sorted SA1
  //        tThreads1, mod_1+mod_2 tThreads2, mod_3
  // Output:m_d_keys_srt_3, positions of i mod 3 = 3 in the same order of SA1
  //        d_keys_sa, ith character value according to d_keys_srt_3
  ////////////////////////////////////////////////////////////////////////
  sa3_srt_construct<<< grid_construct1, threads_construct >>>
      (plan->m_d_keys_srt_3, d_str, plan->m_d_keys_srt_12, d_keys_sa, tThreads1,
       tThreads2, str_length);
  sa3_keys_construct<<<grid_construct2, threads_construct>>>
          (plan->m_d_keys_srt_3, d_keys_sa, d_str, tThreads2, str_length);
  // Only one radix sort based on the result of SA1 (induced sorting)
  KeyValueSort(mod_3, d_keys_sa, plan->m_d_keys_srt_3);

  // Construct SA12 keys in terms of Vector
  // With SA1 composed of 1st char's value, 2nd char's rank, 0 and 1
  // With SA2 composed of 1st char's value, 2nd char's value,
  //          3rd char's rank, 0
  // Input: m_d_keys_srt_12 the order of aKeys
  //        m_d_isa_12 storing the ranks in sorted SA12 order
  // Output: m_d_aKeys, storing SA12 keys in Vectors
  //////////////////////////////////////////////////////////////////
  merge_akeys_construct<<< grid_construct1, threads_construct >>>
    (d_str, plan->m_d_keys_srt_12, plan->m_d_isa_12, plan->m_d_aKeys, tThreads1,
     mod_1, bound, str_length);

  // Construct SA3 keys in terms of Vector
  // Composed of 1st char's value, 2nd char's value, 2nd char's rank
  // and 3rd char's rank
  // Input: m_d_keys_srt_3 the order of bKeys
  //        m_d_isa_12 storing the ranks of chars behind the first char
  // Output:m_d_bKeys, storing SA3 keys in Vectors
  ////////////////////////////////////////////////////////////////////
  merge_bkeys_construct<<< grid_construct2, threads_construct >>>
    (d_str, plan->m_d_keys_srt_3, plan->m_d_isa_12, plan->m_d_bKeys, tThreads2,
     mod_1, bound, str_length);
#if (__CUDA_ARCH__ >= 200) || (CUB_PTX_VERSION == 0)

  // Merge SA12 and SA3 based on aKeys and bKeys
  // Output: m_d_cKeys storing the merged aKeys and bKeys
  //         d_keys_sa storing the merged SA12 and SA3 (positions)
  /////////////////////////////////////////////////////////////////

  mgpu::MergePairs(plan->m_d_aKeys, plan->m_d_keys_srt_12, tThreads1,
                   plan->m_d_bKeys, plan->m_d_keys_srt_3, tThreads2,
                   plan->m_d_cKeys, d_keys_sa, Comp(), context);

  _SafeDeleteArray(unique);
#endif
}


#ifdef __cplusplus
extern "C"
{
#endif

/** @brief Allocate intermediate arrays used by suffix array.
 *
 *
 * @param [in,out] plan Pointer to CUDPPSaPlan object
 *                      containing options and number of elements,
 *                      which is used to compute storage
 *                      requirements, and within which intermediate
 *                      storage is allocated.
 */

void allocSaStorage(CUDPPSaPlan *plan)
{
    size_t str_length = plan->m_numElements;
    size_t mod_1 = (str_length+1)/3 + ((str_length+1)%3 > 0 ? 1:0);
    size_t mod_2 = (str_length+1)/3 + ((str_length+1)%3 > 1 ? 1:0);
    size_t mod_3 = (str_length+1)/3;
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->d_str_value), (str_length+3)*sizeof(uint)));
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_isa_12), (mod_1+mod_2) * sizeof(uint)));
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_keys_srt_12), 2*str_length * sizeof(uint)));
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_unique), sizeof(bool)));
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_keys_srt_3), mod_3 * sizeof(uint)));
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_new_str), 2*str_length * sizeof(uint)));
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_aKeys), (mod_1+mod_2) * sizeof(Vector)));
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_bKeys), mod_3 * sizeof(Vector)));
    CUDA_SAFE_CALL(cudaMalloc((void**) &(plan->m_d_cKeys), (mod_1+mod_2+mod_3) * sizeof(Vector)));
    CUDA_CHECK_ERROR("allocSaStorage");
}


/** @brief Deallocate intermediate block arrays in a CUDPPSaPlan object.
 *
 *
 * @param[in,out] plan Pointer to CUDPPSaPlan object initialized by allocSaStorage().
 */
void freeSaStorage(CUDPPSaPlan *plan)
{
    CUDA_SAFE_CALL(cudaFree(plan->d_str_value));
    CUDA_SAFE_CALL(cudaFree(plan->m_d_isa_12));
    CUDA_SAFE_CALL(cudaFree(plan->m_d_keys_srt_12));
    CUDA_SAFE_CALL(cudaFree(plan->m_d_unique));
    CUDA_SAFE_CALL(cudaFree(plan->m_d_keys_srt_3));
    CUDA_SAFE_CALL(cudaFree(plan->m_d_aKeys));
    CUDA_SAFE_CALL(cudaFree(plan->m_d_bKeys));
    CUDA_SAFE_CALL(cudaFree(plan->m_d_cKeys));
    CUDA_CHECK_ERROR("freeSaStorage");
}

/** @brief Dispatch function to perform parallel suffix array on a
 *         string with the specified configuration.
 *
 *
 * @param[in]  d_str input string with three $
 * @param[out] d_keys_sa lexicographically sorted suffix position array
 * @param[in]  d_str_length Number of elements in the string including $
 * @param[in]  plan     Pointer to CUDPPSaPlan object containing
 *                      suffix_array options and intermediate storage
 */


void cudppSuffixArrayDispatch(void* d_str,
                              unsigned int* d_keys_sa,
                              size_t d_str_length,
                              CUDPPSaPlan *plan)
{
    mgpu::ContextPtr context = mgpu::CreateCudaDevice(0);
    size_t nThreads = SA_BLOCK;
    size_t tThreads = d_str_length+3;
    bool fullBlocks = (tThreads%nThreads==0);
    size_t nBlocks = (fullBlocks) ? (tThreads/nThreads) : (tThreads/nThreads+1);
    dim3 grid_construct(nBlocks,1,1);
    dim3 threads_construct(nThreads,1,1);
//    size_t freeMem, totalMem;
//    CUDA_SAFE_CALL(cudaMemGetInfo(&freeMem, &totalMem));
//    printf("freeMem=%u, totalMem=%u\n", freeMem, totalMem);
    strConstruct<<< grid_construct, threads_construct >>>
             ((unsigned char*)d_str, plan->d_str_value,  d_str_length);
//    CUDA_SAFE_CALL(cudaMemGetInfo(&freeMem, &totalMem));
//    printf("freeMem=%u, totalMem=%u\n", freeMem, totalMem);

    ComputeSA((unsigned int*)plan->d_str_value, (unsigned int*)d_keys_sa,
              d_str_length, *context, plan, 0, 0);

    d_keys_sa = d_keys_sa + 1;
    resultConstruct<<< grid_construct, threads_construct >>>
              (d_keys_sa, d_str_length);
}

#ifdef __cplusplus
}
#endif

/** @} */ // end suffix array functions
/** @} */ // end cudpp_app


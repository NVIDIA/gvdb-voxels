// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision:$
// $Date:$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
 * @file hash_multivalue.cu
 *
 * @brief Implements hash tables that store multiple values per key.
 */

#include "hash_multivalue.h"
#include "hash_table.cuh"

#include "cuda_util.h"
#include <cudpp.h>

namespace CudaHT {
namespace CuckooHashing {

//! @name Internal
/// @{

//! Compacts the unique keys down and stores the location of its values as the value.
__global__ void compact_keys(const unsigned keys[],
                             const unsigned is_unique[],
                             const unsigned locations[],
                             uint2          index_counts[],
                             unsigned       compacted[],
                             size_t         kSize) {
    unsigned index = threadIdx.x +
        blockIdx.x * blockDim.x +
        blockIdx.y * blockDim.x * gridDim.x;
    if (index < kSize && is_unique[index]) {
        unsigned array_index = locations[index] - 1;
        compacted[array_index] = keys[index];
        index_counts[array_index].x = index;
    }
}


//! Finds unique keys by checking neighboring items in a sorted list.
__global__ void check_if_unique(const unsigned *keys,
                                unsigned       *is_unique,
                                size_t          kSize) {
    unsigned id = threadIdx.x +
        blockIdx.x * blockDim.x +
        blockIdx.y * blockDim.x * gridDim.x;
    if (id == 0) {
        is_unique[0] = 1;
    } else if (id < kSize) {
        is_unique[id] = (keys[id] != keys[id - 1] ? 1 : 0);
    }
}


//! Counts how many values each key has.
__global__ void count_values(uint2     index_counts[],
                             unsigned  kSize,
                             unsigned  num_unique) {
    unsigned index = threadIdx.x +
        blockIdx.x * blockDim.x +
        blockIdx.y * blockDim.x * gridDim.x;
    if (index < num_unique - 1) {
        index_counts[index].y = index_counts[index+1].x - index_counts[index].x;
    } else if (index == num_unique-1) {
        index_counts[index].y = kSize - index_counts[index].x;
    }
}

//! Creates an array of values equal to the array index.
__global__ void prepare_indices(const unsigned num_keys,
                                unsigned *data) {
    unsigned index = threadIdx.x +
        blockIdx.x * blockDim.x +
        blockIdx.y * blockDim.x * gridDim.x;
    if (index < num_keys) {
        data[index] = index;
    }
}                            
/// @}


template <unsigned kNumHashFunctions> __global__
void hash_retrieve_multi_sorted(const unsigned   n_queries,
                                const unsigned  *keys_in, 
                                const unsigned   table_size, 
                                const Entry     *table, 
                                const uint2     *index_counts, 
                                const Functions<kNumHashFunctions>  constants,
                                const uint2      stash_constants,
                                const unsigned   stash_count,
                                uint2     *location_count)
{
    // Get the key & perform the query.
    unsigned thread_index = threadIdx.x + blockIdx.x*blockDim.x + 
        blockIdx.y*blockDim.x*gridDim.x;
    if (thread_index >= n_queries)
        return;
    unsigned key = keys_in[thread_index];
    unsigned result = retrieve(key,
                               table_size,
                               table,
                               constants,
                               stash_constants,
                               stash_count,
                               NULL);

    // Return the location of the key's values and the count.
    uint2 index_count;
    if (result == kNotFound) {
        index_count = make_uint2(0, 0);
    } else {
        index_count = index_counts[result];
    }
    location_count[thread_index] = index_count;
}       


//! @name Internal
/// @{
namespace CUDAWrapper {

void CallCheckIfUnique(const unsigned *d_sorted_keys,
                       const size_t    n,
                             unsigned *d_scratch_is_unique) {
    dim3 gridDim = ComputeGridDim((unsigned int)n);
    check_if_unique <<<gridDim, kBlockSize>>> (d_sorted_keys,
                                               d_scratch_is_unique,
                                               n);
    CUDA_CHECK_ERROR("Failed to check uniqueness");
}

void CallCompactKeys(const unsigned *d_sorted_keys,
                     const unsigned *d_is_unique,
                     const unsigned *d_offsets,
                     const size_t    kSize,
                           uint2    *d_index_counts,
                           unsigned *d_compacted_keys) {
    dim3 gridDim = ComputeGridDim((unsigned int)kSize);
    compact_keys<<<gridDim, 512>>> (d_sorted_keys,
                                    d_is_unique,
                                    d_offsets,
                                    d_index_counts,
                                    d_compacted_keys,
                                    kSize);
    CUDA_CHECK_ERROR("Failed to compact the arrays.");
}                           

void CallCountValues(uint2    *d_index_counts,
                     unsigned  kSize,
                     unsigned  num_unique) {
    count_values<<<ComputeGridDim(num_unique), 512>>>
        (d_index_counts, kSize, num_unique);
    CUDA_CHECK_ERROR("Failed to count number of values for each key.");
}    

void CallPrepareIndices(const unsigned  num_unique_keys,
                              unsigned *d_indices) {
    prepare_indices<<<ComputeGridDim(num_unique_keys), kBlockSize>>>
        (num_unique_keys, d_indices);
    CUDA_CHECK_ERROR("Failed to create index array.");        
}                              

void CallHashRetrieveMultiSorted(const unsigned      n_queries,
                                 const unsigned      num_hash_functions,
                                 const unsigned     *d_query_keys, 
                                 const unsigned      table_size, 
                                 const Entry        *d_contents, 
                                 const uint2        *d_index_counts, 
                                 const Functions<2>  constants_2,
                                 const Functions<3>  constants_3,
                                 const Functions<4>  constants_4,
                                 const Functions<5>  constants_5,
                                 const uint2         stash_constants,
                                 const unsigned      stash_count,
                                       uint2        *d_location_counts) {
    if (num_hash_functions == 2) {
        hash_retrieve_multi_sorted<2> <<<ComputeGridDim(n_queries),kBlockSize>>>
            (n_queries,
             d_query_keys,
             table_size,
             d_contents,
             d_index_counts,
             constants_2,
             stash_constants,
             stash_count,
             d_location_counts);
    } else if (num_hash_functions == 3) {
        hash_retrieve_multi_sorted<3> <<<ComputeGridDim(n_queries),kBlockSize>>>
            (n_queries,
             d_query_keys,
             table_size,
             d_contents,
             d_index_counts,
             constants_3,
             stash_constants,
             stash_count,
             d_location_counts);
    } else if (num_hash_functions == 4) {
        hash_retrieve_multi_sorted<4> <<<ComputeGridDim(n_queries),kBlockSize>>>
            (n_queries,
             d_query_keys,
             table_size,
             d_contents,
             d_index_counts,
             constants_4,
             stash_constants,
             stash_count,
             d_location_counts);
    } else {
        hash_retrieve_multi_sorted<5> <<<ComputeGridDim(n_queries),kBlockSize>>>
            (n_queries,
             d_query_keys,
             table_size,
             d_contents,
             d_index_counts,
             constants_5,
             stash_constants,
             stash_count,
             d_location_counts);
    }

    CUDA_CHECK_ERROR("Retrieval failed.\n");
}

};  // namespace CUDAWrapper
/// @}

};  // namespace CuckooHashing
};  // namespace CudaHT

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

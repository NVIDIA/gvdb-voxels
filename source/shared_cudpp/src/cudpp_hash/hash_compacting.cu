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
 * @file hash_compacting.cu
 *
 * @brief Implements hash tables that assign each unique key an ID.
 */

#include "debugging.h"
#include "hash_compacting.h"
#include "hash_functions.h"
#include "hash_table.cuh"

#include <cudpp.h>
#include "cuda_util.h"

#include <set>

namespace CudaHT {
namespace CuckooHashing {

/* --------------------------------------------------------------------------
   Retrieval functions.
   -------------------------------------------------------------------------- */
//! Answers a single query from a compacting hash table.
/*! @ingroup PublicInterface
 *  @param[in]  key                   Query key
 *  @param[in]  table_size            Size of the hash table
 *  @param[in]  table                 The contents of the hash table
 *  @param[in]  constants             The hash functions used to build the table
 *  @param[in]  stash_constants       Constants used by the stash hash function
 *  @param[in]  stash_count           Number of items contained in the stash
 *  @param[out] num_probes_required   Debug only: The number of probes required to resolve the query.
 *
 *  @returns The ID of the query key is returned if the key exists in the table.  Otherwise, \ref kNotFound will be returned.
 */
template <unsigned kNumHashFunctions> __device__
unsigned retrieve_compacting(const unsigned                      query_key,
                             const unsigned                      table_size,
                             const Entry                        *table,
                             const Functions<kNumHashFunctions>  constants,
                             const uint2                         stash_constants,
                             const unsigned                      stash_count,
                             unsigned                     *num_probes_required = NULL)
{
    // Identify all of the locations that the key can be located in.
    unsigned locations[kNumHashFunctions];
    KeyLocations(constants, table_size, query_key, locations);

    // Check each location until the key is found.
    // Short-circuiting is disabled because the duplicate removal step breaks
    // it.
    unsigned num_probes = 1;
    Entry    entry      = table[locations[0]];

#pragma unroll
    for (unsigned i = 1; i < kNumHashFunctions; ++i) {
        if (get_key(entry) != query_key) {
            num_probes++;
            entry = table[locations[i]];
        }
    }

    // Check the stash.
    if (stash_count && get_key(entry) != query_key) {
        num_probes++;
        const Entry *stash = table + table_size;
        unsigned slot = stash_hash_function(stash_constants, query_key);
        entry = stash[slot];
    }

#ifdef TRACK_ITERATIONS
    if (num_probes_required) {
        *num_probes_required = num_probes;
    }
#endif

    if (get_key(entry) == query_key) {
        return get_value(entry);
    } else {
        return kNotFound;
    }
}


//! Returns the unique identifier for every query key.  Each thread manages a single query.
/*! @param[in]  n_queries             Number of query keys
 *  @param[in]  keys_in               Query keys
 *  @param[in]  table_size            Size of the hash table
 *  @param[in]  table                 The contents of the hash table
 *  @param[in]  constants             The hash functions used to build the table
 *  @param[in]  stash_constants       Constants used by the stash hash function
 *  @param[in]  stash_count           Number of items contained in the stash
 *  @param[out] values_out            The unique identifiers for each query key
 *  @param[out] num_probes_required   Debug only: The number of probes required to resolve the query.
 *
 *  The ID of the query key is written out if the key exists in the table.
 *  Otherwise, \ref kNotFound will be.
 */
template <unsigned kNumHashFunctions> __global__
void hash_retrieve_compacting(const unsigned                      n_queries,
                              const unsigned                     *keys_in,
                              const unsigned                      table_size,
                              const Entry                        *table,
                              const Functions<kNumHashFunctions>  constants,
                              const uint2                         stash_constants,
                              const unsigned                      stash_count,
                              unsigned                     *values_out,
                              unsigned                     *num_probes_required = NULL)
{
    // Get the key.
    unsigned thread_index = threadIdx.x +
        blockIdx.x * blockDim.x +
        blockIdx.y * blockDim.x * gridDim.x;
    if (thread_index >= n_queries)
        return;
    unsigned key = keys_in[thread_index];

    values_out[thread_index] = retrieve_compacting<kNumHashFunctions>
        (key,
         table_size,
         table,
         constants,
         stash_constants,
         stash_count,
         (num_probes_required ? num_probes_required + thread_index : NULL));
}       


/*! @name Internal
 *  @{
 */
//! Builds a compacting hash table.
template <unsigned kNumHashFunctions>
__global__
void hash_build_compacting(const int                           n,
                           const unsigned                     *keys,
                           const unsigned                      table_size,
                           const Functions<kNumHashFunctions>  constants,
                           const uint2                         stash_constants,
                           const unsigned                      max_iteration_attempts,
                           unsigned                     *table,
                           unsigned                     *stash_count,
                           unsigned                     *failures)
{       
    // Check if this thread has an item and if any previous threads failed.
    unsigned int thread_index = threadIdx.x +
        blockIdx.x * blockDim.x +
        blockIdx.y * blockDim.x * gridDim.x;
    if (thread_index >= n || *failures)
        return;

    // Read the key that this thread should insert.  It always uses its first
    // slot.
    unsigned key      = keys[thread_index];
    unsigned location = hash_function(constants, 0, key) % table_size;

    // Keep inserting until an empty slot is found, a copy was found,
    // or the eviction chain grows too large.
    unsigned old_key = kKeyEmpty;
    for (int its = 1; its < max_iteration_attempts; its++) {
        old_key = key;

        // Insert the new entry.
        key = atomicExch(&table[location], key);

        // If no unique key was evicted, we're done.
        if (key == kKeyEmpty || key == old_key)
            return;

        location = determine_next_location(constants, table_size, key,
                                           location);
    };

    // Shove it into the stash.
    if (key != kKeyEmpty) {
        unsigned slot = stash_hash_function(stash_constants, key);
        unsigned *stash = table + table_size;
        unsigned  replaced_key = atomicExch(stash + slot, key);
        if (replaced_key == kKeyEmpty || replaced_key == key) {
            atomicAdd(stash_count, 1);
            return;
        }
    }

    // The eviction chain grew too large.  Report failure.
#ifdef COUNT_UNINSERTED
    atomicAdd(failures, 1);
#else
    *failures = 1;
#endif
}       


//! Removes all key duplicates from a compacting hash table.
/*! The unspecialized version is significantly slower than the explicitly
 * specialized ones.
 */
template <unsigned kNumHashFunctions> __global__
void hash_remove_duplicates(const unsigned                      table_size,
                            const unsigned                      total_table_size,
                            const Functions<kNumHashFunctions>  constants,
                            const uint2                         stash_constants,
                            unsigned                     *keys,
                            unsigned                     *is_unique) {
    // Read out the key that may be duplicated.
    unsigned int thread_index = threadIdx.x +
        blockIdx.x * blockDim.x +
        blockIdx.y * blockDim.x * gridDim.x;
    if (thread_index >= total_table_size)
        return;
    unsigned key = keys[thread_index];

    // Determine all the locations that the key could be in.
    unsigned first_location = table_size + stash_hash_function(stash_constants,
                                                               key);
#pragma unroll
    for (int i = kNumHashFunctions-1; i >= 0; --i) {
        unsigned location = hash_function(constants, i, key) % table_size;
        first_location = (keys[location] == key ? location : first_location);
    }

    // If this thread got a later copy of the key, remove this thread's copy
    // from the table.
    if (first_location != thread_index || key == kKeyEmpty) {
        keys[thread_index] = kKeyEmpty;
        is_unique[thread_index] = 0;
    } else {
        is_unique[thread_index] = 1;
    }
}                                  
/// @}

//! @name Explicit template specializations
/// @{
#if 1
template <> __global__
void hash_remove_duplicates<2>(const unsigned      table_size,
                               const unsigned      total_table_size,
                               const Functions<2>  constants,
                               const uint2         stash_constants,
                               unsigned     *keys,
                               unsigned     *is_unique) {       
    // Read out the key that may be duplicated.
    unsigned int thread_index = threadIdx.x +
        blockIdx.x * blockDim.x +
        blockIdx.y * blockDim.x * gridDim.x;
    if (thread_index >= total_table_size)
        return;
    unsigned key = keys[thread_index];

    // Determine all the locations that the key could be in.
    unsigned location_0 = hash_function(constants, 0, key) % table_size;
    unsigned location_1 = hash_function(constants, 1, key) % table_size;
    unsigned stash_loc  = table_size + stash_hash_function(stash_constants,
                                                           key);

    // Figure out where the key is first located.
    unsigned first_index;
    if (keys[location_0] == key) first_index = location_0;
    else if (keys[location_1] == key) first_index = location_1;
    else                              first_index = stash_loc;

    // If this thread got a later copy of the key, remove this thread's copy
    // from the table.
    if (first_index != thread_index || key == kKeyEmpty) {
        keys[thread_index] = kKeyEmpty;
        is_unique[thread_index] = 0;
    } else {
        is_unique[thread_index] = 1;
    }
}       

template <> __global__
void hash_remove_duplicates<3>(const unsigned      table_size,
                               const unsigned      total_table_size,
                               const Functions<3>  constants,
                               const uint2         stash_constants,
                               unsigned     *keys,
                               unsigned     *is_unique) {       
    // Read out the key that may be duplicated.
    unsigned int thread_index = threadIdx.x +
        blockIdx.x * blockDim.x +
        blockIdx.y * blockDim.x * gridDim.x;
    if (thread_index >= total_table_size)
        return;
    unsigned key = keys[thread_index];

    // Determine all the locations that the key could be in.
    unsigned location_0 = hash_function(constants, 0, key) % table_size;
    unsigned location_1 = hash_function(constants, 1, key) % table_size;
    unsigned location_2 = hash_function(constants, 2, key) % table_size;
    unsigned stash_loc  = table_size + stash_hash_function(stash_constants,
                                                           key);

    // Figure out where the key is first located.
    unsigned first_index;
    if (keys[location_0] == key) first_index = location_0;
    else if (keys[location_1] == key) first_index = location_1;
    else if (keys[location_2] == key) first_index = location_2;
    else                              first_index = stash_loc;

    // If this thread got a later copy of the key, remove this thread's copy
    // from the table.
    if (first_index != thread_index || key == kKeyEmpty) {
        keys[thread_index] = kKeyEmpty;
        is_unique[thread_index] = 0;
    } else {
        is_unique[thread_index] = 1;
    }
}       

template <> __global__
void hash_remove_duplicates<4>(const unsigned      table_size,
                               const unsigned      total_table_size,
                               const Functions<4>  constants,
                               const uint2         stash_constants,
                               unsigned     *keys,
                               unsigned     *is_unique) {       
    // Read out the key that may be duplicated.
    unsigned int thread_index = threadIdx.x +
        blockIdx.x * blockDim.x +
        blockIdx.y * blockDim.x * gridDim.x;
    if (thread_index >= total_table_size)
        return;
    unsigned key = keys[thread_index];

    // Determine all the locations that the key could be in.
    unsigned location_0 = hash_function(constants, 0, key) % table_size;
    unsigned location_1 = hash_function(constants, 1, key) % table_size;
    unsigned location_2 = hash_function(constants, 2, key) % table_size;
    unsigned location_3 = hash_function(constants, 3, key) % table_size;
    unsigned stash_loc  = table_size + stash_hash_function(stash_constants,
                                                           key);

    // Figure out where the key is first located.
    unsigned first_index;
    if (keys[location_0] == key) first_index = location_0;
    else if (keys[location_1] == key) first_index = location_1;
    else if (keys[location_2] == key) first_index = location_2;
    else if (keys[location_3] == key) first_index = location_3;
    else                              first_index = stash_loc;

    // If this thread got a later copy of the key, remove this thread's copy
    // from the table.
    if (first_index != thread_index || key == kKeyEmpty) {
        keys[thread_index] = kKeyEmpty;
        is_unique[thread_index] = 0;
    } else {
        is_unique[thread_index] = 1;
    }
}       


template <> __global__
void hash_remove_duplicates<5>(const unsigned      table_size,
                               const unsigned      total_table_size,
                               const Functions<5>  constants,
                               const uint2         stash_constants,
                               unsigned     *keys,
                               unsigned     *is_unique) {       
    // Read out the key that may be duplicated.
    unsigned int thread_index = threadIdx.x +
        blockIdx.x * blockDim.x +
        blockIdx.y * blockDim.x * gridDim.x;
    if (thread_index >= total_table_size)
        return;
    unsigned key = keys[thread_index];

    // Determine all the locations that the key could be in.
    unsigned location_0 = hash_function(constants, 0, key) % table_size;
    unsigned location_1 = hash_function(constants, 1, key) % table_size;
    unsigned location_2 = hash_function(constants, 2, key) % table_size;
    unsigned location_3 = hash_function(constants, 3, key) % table_size;
    unsigned location_4 = hash_function(constants, 4, key) % table_size;
    unsigned stash_loc  = table_size + stash_hash_function(stash_constants,
                                                           key);

    // Figure out where the key is first located.
    unsigned first_index;
    if (keys[location_0] == key) first_index = location_0;
    else if (keys[location_1] == key) first_index = location_1;
    else if (keys[location_2] == key) first_index = location_2;
    else if (keys[location_3] == key) first_index = location_3;
    else if (keys[location_4] == key) first_index = location_4;
    else                              first_index = stash_loc;

    // If this thread got a later copy of the key, remove this thread's copy
    // from the table.
    if (first_index != thread_index || key == kKeyEmpty) {
        keys[thread_index] = kKeyEmpty;
        is_unique[thread_index] = 0;
    } else {
        is_unique[thread_index] = 1;
    }
}       
/// @}
#endif


//! @name Internal
//! @{

//! Interleave the keys and their unique IDs in the cuckoo hash table, then compact down the keys.
__global__ void hash_compact_down(const unsigned  table_size,
                                  Entry    *table_entry,
                                  unsigned *unique_keys,
                                  const unsigned *table,
                                  const unsigned *indices) {
    // Read out the table entry.
    unsigned int thread_index = threadIdx.x + blockIdx.x*blockDim.x +
        blockIdx.y*blockDim.x*gridDim.x;
    if (thread_index >= table_size)
        return;
    unsigned key = table[thread_index];
    unsigned index = indices[thread_index] - 1;
    Entry entry = make_entry(key, index);

    // Write the key and value interleaved.  The value for an invalid key
    // doesn't matter.
    table_entry[thread_index] = entry;

    // Compact down the keys.
    if (key != kKeyEmpty) {
        unique_keys[index] = key;
    }
}
//! @}


namespace CUDAWrapper {
void CallHashBuildCompacting(const int           n,
                             const unsigned      num_hash_functions,
                             const unsigned     *d_keys,
                             const unsigned      table_size,
                             const Functions<2>  constants_2,
                             const Functions<3>  constants_3,
                             const Functions<4>  constants_4,
                             const Functions<5>  constants_5,
                             const uint2         stash_constants,
                             const unsigned      max_iterations,
                             unsigned           *d_scratch_cuckoo_keys,
                             unsigned           *d_stash_count,
                             unsigned           *d_failures) {
    if (num_hash_functions == 2) {
        hash_build_compacting <<<ComputeGridDim(n), kBlockSize>>>
            (n, 
             d_keys,
             table_size,
             constants_2,
             stash_constants,
             max_iterations,
             d_scratch_cuckoo_keys,
             d_stash_count,
             d_failures);
    } else if (num_hash_functions == 3) {
        hash_build_compacting <<<ComputeGridDim(n), kBlockSize>>>
            (n, 
             d_keys,
             table_size,
             constants_3,
             stash_constants,
             max_iterations,
             d_scratch_cuckoo_keys,
             d_stash_count,
             d_failures);
    } else if (num_hash_functions == 4) {
        hash_build_compacting <<<ComputeGridDim(n), kBlockSize>>>
            (n, 
             d_keys,
             table_size,
             constants_4,
             stash_constants,
             max_iterations,
             d_scratch_cuckoo_keys,
             d_stash_count,
             d_failures);
    } else {                             
        hash_build_compacting <<<ComputeGridDim(n), kBlockSize>>>
            (n, 
             d_keys,
             table_size,
             constants_5,
             stash_constants,
             max_iterations,
             d_scratch_cuckoo_keys,
             d_stash_count,
             d_failures);
    }

    CUDA_CHECK_ERROR("Failed to build.\n");
}    

void CallHashRemoveDuplicates(const unsigned      num_hash_functions,
                              const unsigned      table_size,
                              const unsigned      total_table_size,
                              const Functions<2>  constants_2,
                              const Functions<3>  constants_3,
                              const Functions<4>  constants_4,
                              const Functions<5>  constants_5,
                              const uint2         stash_constants,
                              unsigned           *d_scratch_cuckoo_keys,
                              unsigned           *d_scratch_counts) {
    // Remove any duplicated keys from the hash table and set values to one.
    if (num_hash_functions == 2) {
        hash_remove_duplicates <<<ComputeGridDim(total_table_size),
            kBlockSize>>>
            (table_size,
             total_table_size,
             constants_2,
             stash_constants,
             d_scratch_cuckoo_keys,
             d_scratch_counts);
    } else if (num_hash_functions == 3) {
        hash_remove_duplicates <<<ComputeGridDim(total_table_size),
            kBlockSize>>>
            (table_size,
             total_table_size,
             constants_3,
             stash_constants,
             d_scratch_cuckoo_keys,
             d_scratch_counts);
    } else if (num_hash_functions == 4) {
        hash_remove_duplicates <<<ComputeGridDim(total_table_size),
            kBlockSize>>>
            (table_size,
             total_table_size,
             constants_4,
             stash_constants,
             d_scratch_cuckoo_keys,
             d_scratch_counts);
    } else {                              
        hash_remove_duplicates <<<ComputeGridDim(total_table_size),
            kBlockSize>>>
            (table_size,
             total_table_size,
             constants_5,
             stash_constants,
             d_scratch_cuckoo_keys,
             d_scratch_counts);
    }
    CUDA_CHECK_ERROR("!!! Failed to remove duplicates. \n");
}                                  

void CallHashCompactDown(const unsigned  table_size,
                         Entry          *d_contents,
                         unsigned       *d_unique_keys,
                         const unsigned *d_scratch_cuckoo_keys,
                         const unsigned *d_scratch_unique_ids) {
    hash_compact_down <<<ComputeGridDim(table_size), kBlockSize>>>
        (table_size, 
         d_contents, 
         d_unique_keys, 
         d_scratch_cuckoo_keys, 
         d_scratch_unique_ids);

    CUDA_CHECK_ERROR("Compact down failed.\n");
}

void CallHashRetrieveCompacting(const unsigned      n_queries,
                                const unsigned      num_hash_functions,
                                const unsigned     *d_keys,
                                const unsigned      table_size,
                                const Entry        *d_contents,
                                const Functions<2>  constants_2,
                                const Functions<3>  constants_3,
                                const Functions<4>  constants_4,
                                const Functions<5>  constants_5,
                                const uint2         stash_constants,
                                const unsigned      stash_count,
                                unsigned           *d_values) {
    unsigned *d_retrieval_probes = NULL;
#ifdef TRACK_ITERATIONS
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_retrieval_probes, 
                              sizeof(unsigned) * n_queries));
#endif

    if (num_hash_functions == 2) {
        hash_retrieve_compacting<<<ComputeGridDim(n_queries), kBlockSize>>>
            (n_queries,
             d_keys,
             table_size,
             d_contents,
             constants_2,
             stash_constants,
             stash_count,
             d_values,
             d_retrieval_probes);
    } else if (num_hash_functions == 3) {
        hash_retrieve_compacting<<<ComputeGridDim(n_queries), kBlockSize>>>
            (n_queries,
             d_keys,
             table_size,
             d_contents,
             constants_3,
             stash_constants,
             stash_count,
             d_values,
             d_retrieval_probes);
    } else if (num_hash_functions == 4) {
        hash_retrieve_compacting<<<ComputeGridDim(n_queries), kBlockSize>>>
            (n_queries,
             d_keys,
             table_size,
             d_contents,
             constants_4,
             stash_constants,
             stash_count,
             d_values,
             d_retrieval_probes);
    } else {
        hash_retrieve_compacting<<<ComputeGridDim(n_queries), kBlockSize>>>
            (n_queries,
             d_keys,
             table_size,
             d_contents,
             constants_5,
             stash_constants,
             stash_count,
             d_values,
             d_retrieval_probes);
    }
  
    CUDA_CHECK_ERROR("Retrieval failed.\n");

#ifdef TRACK_ITERATIONS
    OutputRetrievalStatistics(n_queries,
                              d_retrieval_probes,
                              num_hash_functions);
    CUDA_SAFE_CALL(cudaFree(d_retrieval_probes));
#endif
}

void ClearTable(const unsigned  slots_in_table,
                const unsigned  fill_value,
                      unsigned *d_contents) {
    clear_table<<<ComputeGridDim(slots_in_table), kBlockSize>>>
        (slots_in_table, fill_value, d_contents);
    CUDA_CHECK_ERROR("Error occurred during hash table clear.\n");
}

};  // namespace CUDAWrapper

};  // namespace CuckooHashing
};  // namespace CudaHT

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

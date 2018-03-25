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
 * @file hash_multivalue.cpp
 *
 * @brief Implements hash tables that store multiple values per key.
 */

#include "hash_multivalue.h"

#include "cuda_util.h"
#include <cuda_runtime_api.h>
#include <cudpp.h>


namespace CudaHT {
namespace CuckooHashing {


bool MultivalueHashTable::Build(const unsigned  n,
                                const unsigned *d_keys,
                                const unsigned *d_vals)
{
    CUDA_CHECK_ERROR("Failed before build.");

    unsigned *d_sorted_keys = NULL;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_sorted_keys, sizeof(unsigned) * n));
    CUDA_SAFE_CALL(cudaMemcpy(d_sorted_keys, d_keys, sizeof(unsigned) * n, 
                              cudaMemcpyDeviceToDevice));

    unsigned *d_sorted_vals = NULL;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_sorted_vals, sizeof(unsigned) * n));
    CUDA_SAFE_CALL(cudaMemcpy(d_sorted_vals, d_vals, sizeof(unsigned) * n, 
                              cudaMemcpyDeviceToDevice));
    CUDA_CHECK_ERROR("Failed to allocate.");

    CUDPPConfiguration sort_config;
    sort_config.algorithm = CUDPP_SORT_RADIX;                  
    sort_config.datatype = CUDPP_UINT;
    sort_config.options = CUDPP_OPTION_KEY_VALUE_PAIRS;

    CUDPPHandle sort_plan;
    CUDPPResult sort_result = cudppPlan(theCudpp, &sort_plan, sort_config, n,
                                        1, 0);
    cudppRadixSort(sort_plan, d_sorted_keys, (void*)d_sorted_vals, n);

    if (sort_result != CUDPP_SUCCESS)
    {
        printf("Error in plan creation in MultivalueHashTable::build\n");
        bool retval = false;
        cudppDestroyPlan(sort_plan);
        return retval;
    }
    CUDA_CHECK_ERROR("Failed to sort");

    // Find the first key-value pair for each key.
    CUDAWrapper::CallCheckIfUnique(d_sorted_keys, n, d_scratch_is_unique_);

    // Assign a unique index from 0 to k-1 for each of the keys.
    cudppScan(scanplan_, d_scratch_offsets_, d_scratch_is_unique_, n);
    CUDA_CHECK_ERROR("Failed to scan");

    // Check how many unique keys were found.
    unsigned num_unique_keys;
    CUDA_SAFE_CALL(cudaMemcpy(&num_unique_keys, d_scratch_offsets_ + n - 1,
                              sizeof(unsigned), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR("Failed to get # unique keys");

    // Keep a list of the unique keys, and store info on each key's data
    // (location in the values array, how many there are).
    unsigned *d_compacted_keys = NULL;
    uint2 *d_index_counts_tmp = NULL;
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_compacted_keys,
                              sizeof(unsigned) * num_unique_keys));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_index_counts_tmp,
                              sizeof(uint2) * num_unique_keys));
    CUDAWrapper::CallCompactKeys(d_sorted_keys,
                                 d_scratch_is_unique_,
                                 d_scratch_offsets_,
                                 n,
                                 d_index_counts_tmp,
                                 d_compacted_keys);

    // Determine the counts.
    CUDAWrapper::CallCountValues(d_index_counts_tmp, n, num_unique_keys);

    // Reinitialize the cuckoo hash table using the information we discovered.
    HashTable::Initialize(num_unique_keys,
                          target_space_usage_,
                          num_hash_functions_);

    d_index_counts_  = d_index_counts_tmp;
    d_unique_keys_   = d_compacted_keys;
    d_sorted_values_ = d_sorted_vals;
    sorted_values_size_ = n;

    // Build the cuckoo hash table with each key assigned a unique index.
    // Re-uses the sorted key memory as an array of values from 0 to k-1.
    CUDAWrapper::CallPrepareIndices(num_unique_keys, d_sorted_keys);
    bool success = HashTable::Build(num_unique_keys, d_unique_keys_,
                                    d_sorted_keys);
    CUDA_SAFE_CALL(cudaFree(d_sorted_keys));
    return success;
}

void MultivalueHashTable::Retrieve(const unsigned  n_queries,
                                   const unsigned *d_query_keys,
                                         uint2    *d_location_counts)
{
    CUDAWrapper::CallHashRetrieveMultiSorted(n_queries,
                                             num_hash_functions_,
                                             d_query_keys,
                                             table_size_,
                                             d_contents_,
                                             d_index_counts_,
                                             constants_2_,
                                             constants_3_,
                                             constants_4_,
                                             constants_5_,
                                             stash_constants_,
                                             stash_count_,
                                             d_location_counts);
}

bool MultivalueHashTable::Initialize(const unsigned   max_table_entries,
                                     const float      space_usage,
                                     const unsigned   num_hash_functions)
{                                    
    bool success = HashTable::Initialize(max_table_entries, space_usage,
                                             num_hash_functions);
    target_space_usage_ = space_usage;

    // + 2N 32-bit entries
    CUDA_SAFE_CALL(cudaMalloc( (void**)&d_scratch_offsets_, 
                               sizeof(unsigned) * max_table_entries ));
    CUDA_SAFE_CALL(cudaMalloc( (void**)&d_scratch_is_unique_,
                               sizeof(unsigned) * max_table_entries ));

    success &= (d_scratch_offsets_ != NULL);
    success &= (d_scratch_is_unique_ != NULL);

    // Allocate memory for the scan.
    // + Unknown memory usage
    CUDPPConfiguration config;
    config.op            = CUDPP_ADD;
    config.datatype      = CUDPP_UINT;
    config.algorithm     = CUDPP_SCAN;
    config.options       = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
    CUDPPResult result   = cudppPlan(theCudpp, &scanplan_, config, 
                                     max_table_entries, 1, 0);
    if (CUDPP_SUCCESS != result) {
        fprintf(stderr, "Failed to create plan.");
        return false;
    }
    return success;
}

MultivalueHashTable::MultivalueHashTable() :
    d_index_counts_(NULL),
    d_sorted_values_(NULL),
    d_scratch_offsets_(NULL),
    d_scratch_is_unique_(NULL),
    d_unique_keys_(NULL),
    scanplan_(0)
{
}

void MultivalueHashTable::Release() {
    HashTable::Release();

    if (scanplan_) {
      cudppDestroyPlan(scanplan_);
      scanplan_ = 0;
    }

    cudaFree(d_index_counts_);
    cudaFree(d_sorted_values_);
    cudaFree(d_scratch_offsets_);
    cudaFree(d_scratch_is_unique_);
    cudaFree(d_unique_keys_);

    d_index_counts_      = NULL;
    d_sorted_values_     = NULL;
    d_scratch_offsets_   = NULL;
    d_scratch_is_unique_ = NULL;
    d_unique_keys_       = NULL;
}

};  // namespace CuckooHashing
};  // namespace CudaHT

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

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
 * @file hash_compacting.cpp
 *
 * @brief Implements hash tables that assign each unique key an ID.
 */

#include "hash_compacting.h"
#include "hash_functions.h"

#include <cuda_runtime_api.h>
#include <cudpp.h>
#include "cuda_util.h"

#include <set>

namespace CudaHT {
namespace CuckooHashing {


bool CompactingHashTable::Build(const unsigned  n,
                                const unsigned *d_keys,
                                const unsigned *d_values)
{
    CUDA_CHECK_ERROR("Failed before attempting to build.");

    unsigned num_failures = 1;
    unsigned num_attempts = 0;
    unsigned max_iterations = ComputeMaxIterations(n, table_size_,
                                                   num_hash_functions_);
    unsigned total_table_size = table_size_ + kStashSize;

    while (num_failures && ++num_attempts < kMaxRestartAttempts) {
        if (num_hash_functions_ == 2)
            constants_2_.Generate(n, d_keys, table_size_);
        else if (num_hash_functions_ == 3)
            constants_3_.Generate(n, d_keys, table_size_);
        else if (num_hash_functions_ == 4)
            constants_4_.Generate(n, d_keys, table_size_);
        else
            constants_5_.Generate(n, d_keys, table_size_);

        // Initialize the cuckoo hash table.
        CUDAWrapper::ClearTable(total_table_size,
                                kKeyEmpty,
                                d_scratch_cuckoo_keys_);

        num_failures = 0;
        cudaMemcpy(d_failures_,
                   &num_failures,
                   sizeof(unsigned),
                   cudaMemcpyHostToDevice);

        unsigned *d_stash_count = NULL;
        cudaMalloc((void**)&d_stash_count, sizeof(unsigned));
        cudaMemset(d_stash_count, 0, sizeof(unsigned));

        CUDAWrapper::CallHashBuildCompacting(n, 
                                             num_hash_functions_,
                                             d_keys,
                                             table_size_,
                                             constants_2_,
                                             constants_3_,
                                             constants_4_,
                                             constants_5_,
                                             stash_constants_,
                                             max_iterations,
                                             d_scratch_cuckoo_keys_,
                                             d_stash_count,
                                             d_failures_);

        CUDA_SAFE_CALL(cudaMemcpy(&stash_count_, d_stash_count, 
                                  sizeof(unsigned), cudaMemcpyDeviceToHost));
        if (stash_count_) {
            char buffer[256];
            sprintf(buffer, "Stash count: %u", stash_count_);
            PrintMessage(buffer, true);
        }
        CUDA_SAFE_CALL(cudaFree(d_stash_count));

        CUDA_CHECK_ERROR("!!! Failed to cuckoo hash.\n");

        CUDA_SAFE_CALL(cudaMemcpy(&num_failures,
                                  d_failures_,
                                  sizeof(unsigned),
                                  cudaMemcpyDeviceToHost));

#ifdef COUNT_UNINSERTED
        if (num_failures > 0) {
            char buffer[256];
            sprintf(buffer, "Num failures: %u", num_failures);
            PrintMessage(buffer, true);
        }
#endif
    }

    if (num_attempts >= kMaxRestartAttempts) {
        PrintMessage("Completely failed to build.", true);
        return false;
    } else if (num_attempts > 1) {
        char buffer[256];
        sprintf(buffer, "Needed %u attempts", num_attempts);
        PrintMessage(buffer);
    }

    if (num_failures == 0) {
        CUDAWrapper::CallHashRemoveDuplicates(num_hash_functions_,
                                              table_size_,
                                              total_table_size,
                                              constants_2_,
                                              constants_3_,
                                              constants_4_,
                                              constants_5_,
                                              stash_constants_,
                                              d_scratch_cuckoo_keys_,
                                              d_scratch_counts_);

        // Do a prefix-sum over the values to assign each key a unique index.
        CUDPPConfiguration config;
        config.op        = CUDPP_ADD;
        config.datatype  = CUDPP_UINT;
        config.algorithm = CUDPP_SCAN;
        config.options   = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;

        CUDPPResult result = cudppPlan(theCudpp, &scanplan_, config, 
                                       total_table_size, 1, 0);
        if (CUDPP_SUCCESS == result) {
            cudppScan(scanplan_, d_scratch_unique_ids_, d_scratch_counts_,
                      total_table_size);
        } else {
            PrintMessage("!!! Failed to create plan.", true);
        }
        CUDA_CHECK_ERROR("!!! Scan failed.\n");

        // Determine how many unique values there are.
        CUDA_SAFE_CALL(cudaMemcpy(&unique_keys_size_,
                                  d_scratch_unique_ids_ + total_table_size - 1,
                                  sizeof(unsigned),
                                  cudaMemcpyDeviceToHost));

        // Copy the unique indices back and compact down the neighbors.
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_unique_keys_,
                                  sizeof(unsigned) * unique_keys_size_));
        CUDA_SAFE_CALL(cudaMemset(d_unique_keys_,
                                  0xff,
                                  sizeof(unsigned) * unique_keys_size_));
        CUDAWrapper::CallHashCompactDown(total_table_size, 
                                         d_contents_, 
                                         d_unique_keys_, 
                                         d_scratch_cuckoo_keys_, 
                                         d_scratch_unique_ids_);
    }

    CUDA_CHECK_ERROR("Error occurred during hash table build.\n");

    return true;
}

CompactingHashTable::CompactingHashTable() :
    d_unique_keys_(NULL),
    d_scratch_cuckoo_keys_(NULL),
    d_scratch_counts_(NULL),
    d_scratch_unique_ids_(NULL),
    scanplan_(0)
{
}

bool CompactingHashTable::Initialize(const unsigned   max_table_entries,
                                     const float      space_usage,
                                     const unsigned   num_functions)
{                                    
    bool success = HashTable::Initialize(max_table_entries, space_usage, 
                                         num_functions);

    unsigned slots_to_allocate = table_size_ + kStashSize;
    CUDA_SAFE_CALL(cudaMalloc( (void**)&d_scratch_cuckoo_keys_, 
                               sizeof(unsigned) * slots_to_allocate ));
    CUDA_SAFE_CALL(cudaMalloc( (void**)&d_scratch_counts_,      
                               sizeof(unsigned) * slots_to_allocate ));
    CUDA_SAFE_CALL(cudaMalloc( (void**)&d_scratch_unique_ids_,  
                               sizeof(unsigned) * slots_to_allocate ));

    success &= d_scratch_cuckoo_keys_ != NULL;
    success &= d_scratch_counts_      != NULL;
    success &= d_scratch_unique_ids_  != NULL;

    return success;
}

CompactingHashTable::~CompactingHashTable() {
    Release();
}

void CompactingHashTable::Release() {
    HashTable::Release();

    CUDA_SAFE_CALL(cudaFree(d_unique_keys_));
    CUDA_SAFE_CALL(cudaFree(d_scratch_cuckoo_keys_));
    CUDA_SAFE_CALL(cudaFree(d_scratch_counts_));
    CUDA_SAFE_CALL(cudaFree(d_scratch_unique_ids_));

    d_unique_keys_         = NULL;
    d_scratch_cuckoo_keys_ = NULL;
    d_scratch_counts_      = NULL;
    d_scratch_unique_ids_  = NULL;

    if (scanplan_) {
      cudppDestroyPlan(scanplan_);
    }
    scanplan_         = 0;
    unique_keys_size_ = 0;
}

void CompactingHashTable::Retrieve(const unsigned  n_queries,
                                   const unsigned *d_keys,
                                   unsigned       *d_values) {
    CUDAWrapper::CallHashRetrieveCompacting(n_queries,
                                            num_hash_functions_,
                                            d_keys,
                                            table_size_,
                                            d_contents_,
                                            constants_2_,
                                            constants_3_,
                                            constants_4_,
                                            constants_5_,
                                            stash_constants_,
                                            stash_count_,
                                            d_values);
}

};  // namespace CuckooHashing
};  // namespace CudaHT


// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

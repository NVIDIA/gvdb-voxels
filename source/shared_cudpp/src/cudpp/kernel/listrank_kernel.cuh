// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt 
// in the root directory of this source distribution.
// ------------------------------------------------------------- 

#include <cudpp_globals.h>
#include "sharedmem.h"
#include <stdio.h>

#define LISTRANK_TOTAL 1024*1024
#define LISTRANK_MAX 2048    // max number of threads dispatched
#define LISTRANK_CTA_BLOCK 128       // threads per block

/**
 * @file
 * listrank_kernel.cu
 * 
 * @brief CUDPP kernel-level listrank routines
 */

/** \addtogroup cudpp_kernel
 * @{
 */

/** @name ListRank Functions
 * @{
 */

typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;

/**
 * @brief Use pointer jumping to rank values. After ranking
 * the values, calculate the next set of indices. The number
 * of values ranked doubles at each kernel call. Called by
 * listRank().
 *
 * @param[out] d_ranked_values Ranked values array
 * @param[in]  d_unranked_values Unranked values array
 * @param[in]  d_ping Next indices array for the current kernel call
 * @param[in]  d_pong Next indices array for the next kernel call
 * @param[in]  d_start_indices Holds the starting node indices
 *             for "ranking" threads. The number of "ranking" threads
 *             doubles at each stage.
 * @param[in]  step The number of "ranking" threads.
 * @param[in]  head Head node index of the linked-list.
 * @param[in]  numElts Number of nodes to rank
 *
 */
template <typename T>
__global__ void list_rank_kernel_soa_1(T*               d_ranked_values,
                                       const T*         d_unranked_values,
                                       const int*       d_ping,
                                       int*             d_pong,
                                       int*             d_start_indices,
                                       int              step,
                                       int              head,
                                       int              numElts)
{
    // Global, Local IDs
    uint idx = threadIdx.x + (blockIdx.x * blockDim.x);

    if(idx==0 && step==1) d_ranked_values[0] = d_unranked_values[head];

    if(idx < step && (idx+step) < numElts)
    {
        // Only threads which are alive/dispatched compute here
        int myhead = (idx==0) ? head : d_start_indices[idx-1];
        int next_index = d_ping[myhead];

        d_start_indices[idx+(step-1)] = next_index;
        d_ranked_values[idx+step] = d_unranked_values[next_index];
    }

    // All threads work to calculate next neighbors
    // (1, 2, 4, 8, etc. neigbbors away)
    for(int i = (int)idx; i < numElts; i += LISTRANK_TOTAL)
    {
        int my_next_index = d_ping[i];
        int val_to_store = -1;
        if(my_next_index >= 0){
            val_to_store = d_ping[my_next_index];
        }
        d_pong[i] = val_to_store;
    }
}

/**
 * @brief After pointer jumping is finished and all threads are
 * able to rank values, ranking continues serially. Each thread
 * ranks values until all values are ranked. Called by listRank().
 *
 * @param[out] d_ranked_values Ranked values array
 * @param[in]  d_unranked_values Unranked values array
 * @param[in]  d_pong Next indices array for the current kernel call
 * @param[in]  d_start_indices Holds the starting node indices
 *             for "ranking" threads. The number of "ranking" threads
 *             doubles at each stage.
 * @param[in]  head Head node index of the linked-list.
 * @param[in]  numElts Number of nodes to rank
 *
 */
template <typename T>
__global__ void list_rank_kernel_soa_2(T*               d_ranked_values,
                                       const T*         d_unranked_values,
                                       const int*       d_pong,
                                       const int*       d_start_indices,
                                       int              head,
                                       int              numElts)
{
    // Global, Local IDs
    uint idx = threadIdx.x + (blockIdx.x * blockDim.x);

    int myIndex = (idx==0) ? head : d_start_indices[idx-1];

    // No more threads to dispatch, each thread keeps chasing
    // until end of string (start where we left off from prev kernel)
    for(int j=((int)idx+LISTRANK_MAX); j<numElts; j += LISTRANK_MAX)
    {
        int val = d_pong[myIndex];
        d_ranked_values[j] = d_unranked_values[val];
        myIndex = val;
    }
}

/** @} */ // end listrank functions
/** @} */ // end cudpp_kernel
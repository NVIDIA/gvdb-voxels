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
 * @file
 * cudpp_hash.cpp
 *
 * @brief Main hash table library source file. Implements wrappers for
 * public interface.
 * 
 * Main hash table library source file. Implements wrappers for public
 * interface. These wrappers call application-level operators and
 * internal data structures.
 */

/**
 * @page hash_overview Overview of CUDPP hash tables
 * 
 * Hash tables are useful for efficiently storing and retrieving
 * sparse data sets. Unlike dense representations, the size of a hash
 * table is generally proportional to the number of elements stored in
 * it rather than the size of the space of possible elements. Hash
 * tables should also have a small cost to insert items and a small
 * cost to retrieve items. CUDPP hash tables have these properties.
 * 
 * CUDPP includes three different hash table types:
 * <table style="width:90%; margin: auto; border: 1px solid #dddddd;">
 *   <tr>
 *     <th class="classes" style="padding-right: 1em;">
 *       \ref CUDPP_BASIC_HASH_TABLE
 *     </th>
 *     <td>
 *       Stores a single value per key. The input is expected to be a
 *       set of key-value pairs, where the keys are all unique.
 *     </td>
 *   </tr>
 *   <tr>
 *     <th class="classes" style="padding-right: 1em;">
 *       \ref CUDPP_COMPACTING_HASH_TABLE
 *     </th>
 *     <td>
 *       Assigns each key a unique identifier and allows O(1)
 *       translation between the key and the unique IDs. Input is a
 *       set of keys that may, or may not, be repeated.
 *     </td>
 *   </tr>
 *   <tr>
 *     <th class="classes" style="padding-right: 1em;">
 *       \ref CUDPP_MULTIVALUE_HASH_TABLE
 *     </th>
 *     <td>
 *       Allows you to store multiple values for each key. Multiple
 *       values for the same key are represented by different
 *       key-value pairs in the input.
 *     </td>
 *   </tr>
 * </table>
 *
 * \section hash_using Using CUDPP hash tables
 *
 * CUDPP supports four major routines for hash tables: creating a hash
 * table (\ref cudppHashTable), inserting items (typically key/value
 * pairs) into a hash table (\ref cudppHashInsert), retrieving values
 * from a hash table given their keys (\ref cudppHashRetrieve), and
 * destroying the hash table (\ref cudppDestroyHashTable). Each of
 * these routines works with each of the 3 types of hash tables above.
 *
 * A typical use of a hash table might look like this (see the sample
 * application cudpp_hash_testrig for a complete example):
 *
 * - Create and populate two arrays in GPU device memory, one of keys,
 *   one of values.
 * - Configure a CUDPPHashTableConfig data structure with:
 *   - the type of hash table (\ref CUDPPHashTableType); 
 *   - \a kInputSize, the number of items to be inserted into the hash
 *     table; and
 *   - the space usage multiplier \a space_usage; the hash table will
 *     store kInputSize elements but require kInputSize * space_usage
 *     elements of storage. Smaller space_usage factors use less space
 *     overall but take longer to build. The cudpp_hash_testrig
 *     example tests with five \a space_usage factors from 1.05 to
 *     2.0. 
 * - Initialize the table using \ref cudppHashTable.
 * - Insert the keys and arrays into the hash table using \ref
 *   cudppHashInsert. 
 * - Create two arrays in GPU device memory, one populated with a list
 *   of retrieval keys, and one empty to retrieve the value associated
 *   with those keys.
 * - Retrieve those values with \ref cudppHashRetrieve.
 * - When you're done, destroy the hash table with \ref
 *   cudppDestroyHashTable. 
 *
 * \section hash_other_software Other software used in CUDPP's hash tables
 *
 * CUDPP's hash table library relies on a good random number generator
 * to ensure good hash function generation. CUDPP uses the Mersenne
 * Twister implementation provided by Makoto Matsumoto, <a
 * href=http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/emt19937ar.html>available
 * here</a> (and included in the CUDPP distribution). You may try
 * using your system's rand() function and srand() functions, but keep
 * in mind that Windows' generator produces numbers in a very small
 * range.
 * 
 * The compacting hash table and multivalue hash table implementation
 * use CUDPP's scan and sort functionality.
 * 
 * \section hash_space_limitations Hash table space limitations
 *
 * The maximum size of the hash table implementations are primarily
 * limited by the size of available memory. The figures below indicate
 * the size of the hash table as a function of:
 *
 * - N, the number of elements in the hash table;
 * - K, the number of unique keys in the input (for the multivalue  
 *   and compacting hash table); and
 * - M, the space multiplier for the hash table (for every 1 element
 *   to be stored in the hash table, we allocate space for M
 *   elements). M defaults to 1.25.
 * 
 * \subsection basic_hash_space_limitations Basic hash table:
 * 
 * Other than some 32-bit values (like an error flag and the number of
 * items in the stash), the only memory allocation is for the actual hash
 * table itself:
 * 
 * - Hash table: 2 * (M * N + 101) uint32s
 * 
 * 101 is the size of the stash (in uint32s).
 * We multiply by 2 because we store both keys and values.
 * 
 * \subsection multivalue_hash_space_limitations Multi-value hash table:
 * 
 * In the multivalue hash table, memory is held until the hash table
 * is deleted; it was set up this way to prevent mucking up the
 * timings with memory management. 
 * 
 * - Hash table: 2 * (M * K + 101) uint32s
 * - Values of all of the keys: N uint32s
 * - Information about each key's values: 2 * K uint32s
 * - Compacted list of the keys: K uint32s
 * - Scratch memory, CUDPP scan: Enough to scan N items. Note that
 *   scan has its own limitation on size (~67M elements).
 * - Scratch memory, radix sort: 2N uint32s. Note that
 *   sort has its own limitation on size (~2B elements).
 * - Scratch memory, additional: N uint32s
 * 
 * Each key is capped at UINT_MAX values. Although the code doesn't do
 * it, you could in theory creatively re-use some of the memory or
 * move allocations around to avoid having to hold onto more scratch
 * space than you really need. The CUDPP scan memory, for example, is
 * allocated when the hash table is initialized but only used when
 * it's being built.
 * 
 * \subsection compacting_hash_space_limitations Compacting hash table:
 * 
 * - Hash table: 2 * (M * N + 101) uint32s
 * - Compacted unique keys: K uint32s
 * - Scratch memory: 3 * (M * N + 101) uint32s
 * - Scratch memory, CUDPP scan: Enough to scan N + 101 uint32s. Note that
 *   scan has its own limitation on size (~67M elements).
 */

#include <cuda_runtime.h>
#include "cudpp_hash.h"
#include "cudpp_plan.h"
#include "cudpp_manager.h"

#include "hash_table.h"         // HashTable class
#include "hash_compacting.h"    // CompactingHashTable class
#include "hash_multivalue.h"    // MultivalueHashTable class

typedef CUDPPHashTableInternal<CudaHT::CuckooHashing::HashTable> hti_basic;
typedef CUDPPHashTableInternal<CudaHT::CuckooHashing::CompactingHashTable> hti_compacting;
typedef CUDPPHashTableInternal<CudaHT::CuckooHashing::MultivalueHashTable> hti_multivalue;
typedef CUDPPHashTableInternal<void> hti_void;

/** @addtogroup publicInterface
  * @{
  */

/** @name Hash Table Interface
 * @{
 */

/* @brief unsigned int indicating a not-found value in a hash table */
const unsigned int CUDPP_HASH_KEY_NOT_FOUND = CudaHT::CuckooHashing::kNotFound;

// cudppHashTable will create some sort of internal struct that you
// write. It will then cast the pointer to that struct to a
// CUDPPHandle (just like cudppPlan() does), and return that.

/**
 * @brief Creates a CUDPP hash table in GPU memory given an input hash
 * table configuration; returns the \a plan for that hash table. 
 *
 * Requires a CUDPPHandle for the CUDPP instance (to ensure thread
 * safety); call cudppCreate() to get this handle. 
 * 
 * The hash table implementation requires hardware capability 2.0 or
 * higher (64-bit atomic operations).
 * 
 * Hash table types and input parameters are discussed in
 * CUDPPHashTableType and CUDPPHashTableConfig.
 * 
 * After you are finished with the hash table, clean up with
 * cudppDestroyHashTable().
 * 
 * See \ref hash_overview for an overview of CUDPP's hash table support. 
 *
 * @param[in] cudppHandle Handle to CUDPP instance
 * @param[out] plan Handle to hash table instance
 * @param[in] config Configuration for hash table to be created
 * @returns CUDPPResult indicating if creation was successful
 * 
 * @see cudppCreate, cudppDestroyHashTable, CUDPPHashTableType,
 * CUDPPHashTableConfig, \ref hash_overview
 */
CUDPP_DLL
CUDPPResult
cudppHashTable(CUDPPHandle cudppHandle, CUDPPHandle *plan,
               const CUDPPHashTableConfig *config)
{
    // first check: is this device >= 2.0? if not, return error
    CUDPPManager *mgr = CUDPPManager::getManagerFromHandle(cudppHandle);
    cudaDeviceProp prop;
    mgr->getDeviceProps(prop);
    
    if (prop.major < 2)
    {
        // Hash tables are only supported on devices with compute
        // capability 2.0 or greater
        return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
    }

    switch(config->type)
    {
    case CUDPP_BASIC_HASH_TABLE:
    {
        //printf("Size outside: %lu\n", sizeof(CudaHT::CuckooHashing::HashTable));
        CudaHT::CuckooHashing::HashTable * basic_table = 
            new CudaHT::CuckooHashing::HashTable();
        basic_table->setTheCudpp(cudppHandle);
        basic_table->Initialize(config->kInputSize, config->space_usage);
        hti_basic * hti = new hti_basic(config, basic_table);
        if (!hti)
        {
            return CUDPP_ERROR_UNKNOWN;
        }
        else
        {
            *plan = hti->getHandle();
            return CUDPP_SUCCESS;
        }
        break;
    }
    case CUDPP_COMPACTING_HASH_TABLE:
    {
        CudaHT::CuckooHashing::CompactingHashTable * compacting_table = 
            new CudaHT::CuckooHashing::CompactingHashTable();
        compacting_table->setTheCudpp(cudppHandle);
        compacting_table->Initialize(config->kInputSize, config->space_usage);
        hti_compacting * hti = new hti_compacting(config, compacting_table);
        if (!hti)
        {
            return CUDPP_ERROR_UNKNOWN;
        }
        else
        {
            *plan = hti->getHandle();
            return CUDPP_SUCCESS;
        }
        break;
    }
    case CUDPP_MULTIVALUE_HASH_TABLE:
    {
        CudaHT::CuckooHashing::MultivalueHashTable * multivalue_table = 
            new CudaHT::CuckooHashing::MultivalueHashTable();
        multivalue_table->setTheCudpp(cudppHandle);
        multivalue_table->Initialize(config->kInputSize, config->space_usage);
        hti_multivalue * hti = new hti_multivalue(config, multivalue_table);
        if (!hti)
        {
            return CUDPP_ERROR_UNKNOWN;
        }
        else
        {
            *plan = hti->getHandle();
            return CUDPP_SUCCESS;
        }
        break;
    }
    case CUDPP_INVALID_HASH_TABLE:
        return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
        break;
    }
    return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
}

// Then cudppHashTableInsert/Retrieve, or any other functions that
// operate on it, take the CUDPPHandle as input, and call
// getPlanPtrFromHandle<T>(handle), where T is the type of the
// internal struct you define, to get back the pointer to the struct.
/**
 * @brief Inserts keys and values into a CUDPP hash table
 * 
 * Requires a CUDPPHandle for the hash table instance; call
 * cudppHashTable() to create the hash table and get this handle.
 *
 * \a d_keys and \a d_values should be in GPU memory. These should be
 * pointers to arrays of unsigned ints.
 *
 * Calls HashTable::Build internally.
 * 
 * See \ref hash_overview for an overview of CUDPP's hash table support. 
 *
 * @param[in] plan Handle to hash table instance
 * @param[in] d_keys GPU pointer to keys to be inserted
 * @param[in] d_vals GPU pointer to values to be inserted
 * @param[in] num Number of keys/values to be inserted
 * @returns CUDPPResult indicating if insertion was successful
 * 
 * @see cudppHashTable, cudppHashRetrieve,
 * HashTable::Build, CompactingHashTable::Build,
 * MultivalueHashTable::Build, \ref hash_overview
 */

CUDPP_DLL
CUDPPResult 
cudppHashInsert(CUDPPHandle plan, const void* d_keys, const void* d_vals,
                size_t num)
{
    // the other way to do this hacky thing is to have inherited classes
    // from CUDPPHashTableInternal maybe?
    hti_void * hti_init = (hti_void *) getPlanPtrFromHandle<hti_void>(plan);
    switch(hti_init->config.type)
    {
    case CUDPP_BASIC_HASH_TABLE:
    {
        hti_basic * hti = (hti_basic *) getPlanPtrFromHandle<hti_basic>(plan);
        bool s = hti->hash_table->Build(num, (const unsigned int *) d_keys, 
                                        (const unsigned int *) d_vals);
        return s ? CUDPP_SUCCESS : CUDPP_ERROR_UNKNOWN;
        break;
    }
    case CUDPP_COMPACTING_HASH_TABLE:
    {
        hti_compacting * hti =
            (hti_compacting *) getPlanPtrFromHandle<hti_compacting>(plan);
        bool s = hti->hash_table->Build(num, (const unsigned int *) d_keys, 
                                        (const unsigned int *) d_vals);
        return s ? CUDPP_SUCCESS : CUDPP_ERROR_UNKNOWN;
        break;
    } 
    case CUDPP_MULTIVALUE_HASH_TABLE:
    {
        hti_multivalue * hti =
            (hti_multivalue *) getPlanPtrFromHandle<hti_multivalue>(plan);
        bool s = hti->hash_table->Build(num, (const unsigned int *) d_keys, 
                                        (const unsigned int *) d_vals);
        return s ? CUDPP_SUCCESS : CUDPP_ERROR_UNKNOWN;
        break;
    } 
    case CUDPP_INVALID_HASH_TABLE:
        return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
        break;
    }
    return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
}

/**
 * @brief Retrieves values, given keys, from a CUDPP hash table
 * 
 * Requires a CUDPPHandle for the hash table instance; call
 * cudppHashTable() to create the hash table and get this handle.
 *
 * \a d_keys and \a d_values should be in GPU memory. These should be
 * pointers to arrays of unsigned ints.
 *
 * Calls HashTable::Retrieve internally.
 * 
 * See \ref hash_overview for an overview of CUDPP's hash table support. 
 *
 * @param[in] plan Handle to hash table instance
 * @param[in] d_keys GPU pointer to keys to be retrieved
 * @param[out] d_vals GPU pointer to values to be retrieved
 * @param[in] num Number of keys/values to be retrieved
 * @returns CUDPPResult indicating if retrieval was successful
 * 
 * @see cudppHashTable, cudppHashBuild, HashTable::Retrieve,
 * CompactingHashTable::Retrieve, MultivalueHashTable::Retrieve, \ref
 * hash_overview
 */
CUDPP_DLL
CUDPPResult
cudppHashRetrieve(CUDPPHandle plan, const void* d_keys, void* d_vals, 
                  size_t num)
{
    hti_void * hti_init = (hti_void *) getPlanPtrFromHandle<hti_void>(plan);
    switch(hti_init->config.type)
    {
    case CUDPP_BASIC_HASH_TABLE:
    {
        hti_basic * hti = (hti_basic *) getPlanPtrFromHandle<hti_basic>(plan);
        hti->hash_table->Retrieve(num, (const unsigned int *) d_keys, 
                                           (unsigned int *) d_vals);
        return CUDPP_SUCCESS;
        break;
    }
    case CUDPP_COMPACTING_HASH_TABLE:
    {
        hti_compacting * hti = 
            (hti_compacting *) getPlanPtrFromHandle<hti_compacting>(plan);
        hti->hash_table->Retrieve(num, (const unsigned int *) d_keys, 
                                  (unsigned int *) d_vals);
        return CUDPP_SUCCESS;
        break;
    }
    case CUDPP_MULTIVALUE_HASH_TABLE:
    {
        hti_multivalue * hti = 
            (hti_multivalue *) getPlanPtrFromHandle<hti_multivalue>(plan);
        hti->hash_table->Retrieve(num, (const unsigned int *) d_keys, 
                                  (uint2 *) d_vals);
        return CUDPP_SUCCESS;
        break;
    }
    case CUDPP_INVALID_HASH_TABLE:
        return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
        break;
    }
    return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
}

/**
 * @brief Destroys a hash table given its handle.
 * 
 * Requires a CUDPPHandle for the CUDPP instance (to ensure thread
 * safety); call cudppCreate() to get this handle. 
 * 
 * Requires a CUDPPHandle for the hash table instance; call
 * cudppHashTable() to get this handle.
 * 
 * See \ref hash_overview for an overview of CUDPP's hash table support. 
 *
 * @param[in] cudppHandle Handle to CUDPP instance
 * @param[in] plan Handle to hash table instance
 * @returns CUDPPResult indicating if destruction was successful
 * 
 * @see cudppHashTable, \ref hash_overview
 */
CUDPP_DLL
CUDPPResult
cudppDestroyHashTable(CUDPPHandle cudppHandle, CUDPPHandle plan)
{
    (void) cudppHandle;         // eliminates doxygen (!) warning
    hti_void * hti_init = (hti_void *) getPlanPtrFromHandle<hti_void>(plan);
    switch(hti_init->config.type)
    {
    case CUDPP_BASIC_HASH_TABLE:
    {
        hti_basic * hti = (hti_basic *) getPlanPtrFromHandle<hti_basic>(plan);
        delete hti;
        return CUDPP_SUCCESS;
    }
    case CUDPP_COMPACTING_HASH_TABLE:
    {
        hti_compacting * hti = 
            (hti_compacting *) getPlanPtrFromHandle<hti_compacting>(plan);
        delete hti;
        return CUDPP_SUCCESS;
    }
    case CUDPP_MULTIVALUE_HASH_TABLE:
    {
        hti_multivalue * hti = 
            (hti_multivalue *) getPlanPtrFromHandle<hti_multivalue>(plan);
        delete hti;
        return CUDPP_SUCCESS;
    }
    case CUDPP_INVALID_HASH_TABLE:
        return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
        break;
    }
    return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
}

/**
 * @brief Retrieves the size of the values array in a multivalue hash table
 * 
 * Only relevant for multivalue hash tables.
 * 
 * Requires a CUDPPHandle for the hash table instance; call
 * cudppHashTable() to get this handle.
 * 
 * See \ref hash_overview for an overview of CUDPP's hash table support. 
 *
 * @param[in] plan Handle to hash table instance
 * @param[out] size Pointer to size of multivalue hash table
 * @returns CUDPPResult indicating if operation was successful
 * 
 * @see cudppHashTable, cudppMultivalueHashGetAllValues, \ref
 * hash_overview
 */
CUDPP_DLL
CUDPPResult
cudppMultivalueHashGetValuesSize(CUDPPHandle plan, unsigned int * size)
{
    hti_void * hti_init = (hti_void *) getPlanPtrFromHandle<hti_void>(plan);
    if (hti_init->config.type != CUDPP_MULTIVALUE_HASH_TABLE)
    {
        // better be a MULTIVALUE
        return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
    }
    hti_multivalue * hti = 
        (hti_multivalue *) getPlanPtrFromHandle<hti_multivalue>(plan);
    *size = hti->hash_table->get_values_size();
    return CUDPP_SUCCESS;
}

/**
 * @brief Retrieves a pointer to the values array in a multivalue hash table
 * 
 * Only relevant for multivalue hash tables.
 * 
 * Requires a CUDPPHandle for the hash table instance; call
 * cudppHashTable() to get this handle.
 * 
 * See \ref hash_overview for an overview of CUDPP's hash table support. 
 *
 * @param[in] plan Handle to hash table instance
 * @param[out] d_vals Pointer to pointer of values (in GPU memory)
 * @returns CUDPPResult indicating if operation was successful
 * 
 * @see cudppHashTable, cudppMultivalueHashGetValuesSize, \ref
 * hash_overview
 */
CUDPP_DLL
CUDPPResult
cudppMultivalueHashGetAllValues(CUDPPHandle plan, unsigned int ** d_vals)
{
    hti_void * hti_init = (hti_void *) getPlanPtrFromHandle<hti_void>(plan);
    if (hti_init->config.type != CUDPP_MULTIVALUE_HASH_TABLE)
    {
        // better be a MULTIVALUE
        return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
    }
    hti_multivalue * hti = 
        (hti_multivalue *) getPlanPtrFromHandle<hti_multivalue>(plan);
    // @TODO fix up constness
    *d_vals = (unsigned*) (hti->hash_table->get_all_values());
    return CUDPP_SUCCESS;
}

/** @} */ // end Plan Interface
/** @} */ // end publicInterface

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

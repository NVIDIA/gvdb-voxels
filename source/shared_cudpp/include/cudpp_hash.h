#include "cudpp.h"
#include "cudpp_config.h"

/**
 * @file
 * cudpp_hash.h
 * 
 * @brief Main library header file for CUDPP hash tables. Defines public 
 * interface.
 *
 * The CUDPP public interface is a C-only interface to enable 
 * linking with code written in other languages (e.g. C, C++, 
 * and Fortran).  While the internals of CUDPP are not limited 
 * to C (C++ features are used), the public interface is 
 * entirely C (thus it is declared "extern C").
 */


/**
 * @brief Supported types of hash tables
 *
 * @see CUDPPHashTableConfig
 */
enum CUDPPHashTableType
{
    CUDPP_BASIC_HASH_TABLE,     /**< Stores a single value per key.
                                 * Input is expected to be a set of
                                 * key-value pairs, where the keys are
                                 * all unique. */
    CUDPP_COMPACTING_HASH_TABLE,/**< Assigns each key a unique
                                 * identifier and allows O(1)
                                 * translation between the key and the
                                 * unique IDs. Input is a set of keys
                                 * that may, or may not, be
                                 * repeated. */
    CUDPP_MULTIVALUE_HASH_TABLE,/**< Can store multiple values for
                                 * each key. Multiple values for the
                                 * same key are represented by
                                 * different key-value pairs in the
                                 * input. */
    CUDPP_INVALID_HASH_TABLE,   /**< Invalid hash table; flags error
                                 * if used. */
};

/**
 * @brief Configuration struct for creating a hash table (CUDPPHashTable())
 * 
 * @see CUDPPHashTable, CUDDPHashTableType
 */
struct CUDPPHashTableConfig
{
    CUDPPHashTableType type;    /**< see CUDPPHashTableType */
    unsigned int kInputSize;    /**< number of elements to be stored
                                 * in hash table */
    float space_usage;          /**< space factor multiple for the
                                 * hash table; multiply space_usage by
                                 * kInputSize to get the actual space
                                 * allocation in GPU memory. 1.05 is
                                 * about the minimum possible to get a
                                 * working hash table. Larger values
                                 * use more space but take less time
                                 * to construct. */
};

/**
 * \defgroup cudpp_hash_data_structures Hash Table Data Structures and Constants
 * Internal hash table data structures and constants used by CUDPP.
 *
 * @{
 */

/** @brief Internal structure used to store a generic CUDPP hash table
 * 
 * @see CUDPPHashTableConfig, CudaHT::CuckooHashing::HashTable,
 * CudaHT::CuckooHashing::CompactingHashTable,
 * CudaHT::CuckooHashing::MultivalueHashTable
 */
template<class T>
class CUDPPHashTableInternal
{
public:
    //! @brief Constructor for CUDPPHashTableInternal
    //! @param [in] c Pointer to configuration structure
    //! @param [in] t Hash table pointer
    CUDPPHashTableInternal(const CUDPPHashTableConfig * c, T * t) : 
        config(*c), hash_table(t) {}
    CUDPPHashTableConfig config;
    T * hash_table;
    // template<typename T> T getHashTablePtr()
    // {
    // return reinterpret_cast<T>(hash_table);
    // }
    //! @brief Convert this pointer to an opaque handle
    //! @returns Opaque handle for this structure
    CUDPPHandle getHandle()
    {
        return reinterpret_cast<CUDPPHandle>(this);
    }
    //! @brief Destructor for CUDPPHashTableInternal
    ~CUDPPHashTableInternal() 
    {
        delete hash_table;
    }
};

/** @} */ // end cudpp_hash_data_structures

extern CUDPP_DLL const unsigned int CUDPP_HASH_KEY_NOT_FOUND;

CUDPP_DLL CUDPPResult 
cudppHashTable(CUDPPHandle cudppHandle, 
               CUDPPHandle *plan,
               const CUDPPHashTableConfig *config);

CUDPP_DLL CUDPPResult
cudppDestroyHashTable(CUDPPHandle cudppHandle, 
                      CUDPPHandle plan);

CUDPP_DLL CUDPPResult
cudppHashInsert(CUDPPHandle plan, 
                const void* d_keys, 
                const void* d_vals,
                size_t num);

CUDPP_DLL CUDPPResult
cudppHashRetrieve(CUDPPHandle plan, 
                  const void* d_keys, 
                  void* d_vals, 
                  size_t num);

CUDPP_DLL CUDPPResult
cudppMultivalueHashGetValuesSize(CUDPPHandle plan, 
                                 unsigned int * size);

CUDPP_DLL CUDPPResult
cudppMultivalueHashGetAllValues(CUDPPHandle plan, 
                                unsigned int ** d_vals);

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:


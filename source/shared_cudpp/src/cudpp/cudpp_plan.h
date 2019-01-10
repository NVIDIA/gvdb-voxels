// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: 3572$
// $Date: 2007-11-19 13:58:06 +0000 (Mon, 19 Nov 2007) $
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// ------------------------------------------------------------- 
#ifndef __CUDPP_PLAN_H__
#define __CUDPP_PLAN_H__

typedef void* KernelPointer;
class CUDPPPlan;
class CUDPPManager;

#include "cudpp.h"

//! @internal Convert an opaque handle to a pointer to a plan
template <typename T>
T* getPlanPtrFromHandle(CUDPPHandle handle)
{
    return reinterpret_cast<T*>(handle);
}


/** @brief Base class for CUDPP Plan data structures
  *
  * CUDPPPlan and its subclasses provide the internal (i.e. not visible to the
  * library user) infrastructure for planning algorithm execution.  They 
  * own intermediate storage for CUDPP algorithms as well as, in some cases,
  * information about optimal execution configuration for the present hardware.
  * 
  */
class CUDPPPlan
{
public:
    CUDPPPlan(CUDPPManager *mgr, CUDPPConfiguration config, 
              size_t numElements, size_t numRows, size_t rowPitch);
    virtual ~CUDPPPlan() {}

    // Note anything passed to functions compiled by NVCC must be public
    CUDPPConfiguration m_config;        //!< @internal Options structure
    size_t             m_numElements;   //!< @internal Maximum number of input elements
    size_t             m_numRows;       //!< @internal Maximum number of input rows
    size_t             m_rowPitch;      //!< @internal Pitch of input rows in elements
    CUDPPManager      *m_planManager;  //!< @internal pointer to the manager of this plan
   
    //! @internal Convert this pointer to an opaque handle
    //! @returns Handle to a CUDPP plan
    CUDPPHandle getHandle()
    {
        return reinterpret_cast<CUDPPHandle>(this);
    }
};

/** @brief Plan class for scan algorithm
  *
  */
class CUDPPScanPlan : public CUDPPPlan
{
public:
    CUDPPScanPlan(CUDPPManager *mgr, CUDPPConfiguration config, size_t numElements, size_t numRows, size_t rowPitch);
    virtual ~CUDPPScanPlan();

    void  **m_blockSums;          //!< @internal Intermediate block sums array
    size_t *m_rowPitches;         //!< @internal Pitch of each row in elements (for cudppMultiScan())
    size_t  m_numEltsAllocated;   //!< @internal Number of elements allocated (maximum scan size)
    size_t  m_numRowsAllocated;   //!< @internal Number of rows allocated (for cudppMultiScan())
    size_t  m_numLevelsAllocated; //!< @internal Number of levels allocaed (in _scanBlockSums)
};

/** @brief Plan class for segmented scan algorithm
*
*/
class CUDPPSegmentedScanPlan : public CUDPPPlan
{
public:
    CUDPPSegmentedScanPlan(CUDPPManager *mgr, CUDPPConfiguration config, size_t numElements);
    virtual ~CUDPPSegmentedScanPlan();

    void          **m_blockSums;          //!< @internal Intermediate block sums array
    unsigned int  **m_blockFlags;         //!< @internal Intermediate block flags array
    unsigned int  **m_blockIndices;       //!< @internal Intermediate block indices array
    size_t        m_numEltsAllocated;     //!< @internal Number of elements allocated (maximum scan size)
    size_t        m_numLevelsAllocated;   //!< @internal Number of levels allocaed (in _scanBlockSums)
};

/** @brief Plan class for compact algorithm
*
*/
class CUDPPCompactPlan : public CUDPPPlan
{
public:
    CUDPPCompactPlan(CUDPPManager *mgr, CUDPPConfiguration config, size_t numElements, size_t numRows, size_t rowPitch);
    virtual ~CUDPPCompactPlan();

    CUDPPScanPlan *m_scanPlan;         //!< @internal Compact performs a scan of type unsigned int using this plan
    unsigned int* m_d_outputIndices; //!< @internal Output address of compacted elements; this is the result of scan
    
};

/** @brief Plan class for reduce algorithm
*
*/
class CUDPPReducePlan : public CUDPPPlan
{
public:
    CUDPPReducePlan(CUDPPManager *mgr, CUDPPConfiguration config, size_t numElements);
    virtual ~CUDPPReducePlan();

    unsigned int m_threadsPerBlock;     //!< @internal number of threads to launch per block
    unsigned int m_maxBlocks;           //!< @internal maximum number of blocks to launch
    void         *m_blockSums;          //!< @internal Intermediate block sums array
};  

/** @brief Plan class for mergesort algorithm
*
*/

class CUDPPMergeSortPlan : public CUDPPPlan
{
public:
    CUDPPMergeSortPlan(CUDPPManager *mgr, CUDPPConfiguration config, size_t numElements);
    virtual ~CUDPPMergeSortPlan();

    mutable void *m_tempKeys;
    unsigned int *m_tempValues;
    int *m_partitionBeginA;
    int *m_partitionSizeA;

    unsigned int m_numElements;
    unsigned int m_subPartitions, m_swapPoint;
};

/** @brief Plan class for stringsort algorithm
*
*/

class CUDPPStringSortPlan : public CUDPPPlan
{
public:
    CUDPPStringSortPlan(CUDPPManager *mgr, CUDPPConfiguration config, size_t numElements, size_t stringArrayLength);
    virtual ~CUDPPStringSortPlan();

    unsigned int m_stringArrayLength;

	CUDPPScanPlan *m_scanPlan;
	unsigned int m_numElements;
	unsigned int *m_keys;
        unsigned int *m_tempKeys;
        unsigned int *m_tempAddress;
	unsigned int *m_packedAddress;
	unsigned int *m_packedAddressRef;
	unsigned int *m_addressRef;
	unsigned int *m_numSpaces;
	unsigned int *m_spaceScan;

	unsigned int m_subPartitions, m_swapPoint;
	unsigned int *m_partitionSizeA, *m_partitionSizeB, *m_partitionStartA, *m_partitionStartB;


	
};

/** @brief Plan class for radixsort algorithm
*
*/

class CUDPPRadixSortPlan : public CUDPPPlan
{
public:
    CUDPPRadixSortPlan(CUDPPManager *mgr, CUDPPConfiguration config, size_t numElements);
    virtual ~CUDPPRadixSortPlan();
        
    bool           m_bKeysOnly;
    bool           m_bManualCoalesce;
    bool           m_bUsePersistentCTAs;
    unsigned int   m_persistentCTAThreshold[2];
    unsigned int   m_persistentCTAThresholdFullBlocks[2];
    unsigned int   m_keyBits;
    bool           m_bBackward;       //!< Designates reverse-order sort
    CUDPPScanPlan *m_scanPlan;        //!< @internal Sort performs a scan of type unsigned int using this plan

    mutable void  *m_tempKeys;        //!< @internal Intermediate storage for keys
    mutable void  *m_tempValues;      //!< @internal Intermediate storage for values
    unsigned int  *m_counters;        //!< @internal Counter for each radix
    unsigned int  *m_countersSum;     //!< @internal Prefix sum of radix counters
    unsigned int  *m_blockOffsets;    //!< @internal Global offsets of each radix in each block

    enum RadixSortKernels
    {
        KERNEL_RSB_4_0_F_F_T,
        KERNEL_RSB_4_0_F_T_T,
        KERNEL_RSB_4_0_T_F_T,
        KERNEL_RSB_4_0_T_T_T,
        KERNEL_RSBKO_4_0_F_F_T,
        KERNEL_RSBKO_4_0_F_T_T,
        KERNEL_RSBKO_4_0_T_F_T,
        KERNEL_RSBKO_4_0_T_T_T,
        KERNEL_FRO_0_F_T,
        KERNEL_FRO_0_T_T,
        KERNEL_RD_0_F_F_F_T,
        KERNEL_RD_0_F_F_T_T,
        KERNEL_RD_0_F_T_F_T,
        KERNEL_RD_0_F_T_T_T,
        KERNEL_RD_0_T_F_F_T,
        KERNEL_RD_0_T_F_T_T,
        KERNEL_RD_0_T_T_F_T,
        KERNEL_RD_0_T_T_T_T,
        KERNEL_RDKO_0_F_F_F_T,
        KERNEL_RDKO_0_F_F_T_T,
        KERNEL_RDKO_0_F_T_F_T,
        KERNEL_RDKO_0_F_T_T_T,
        KERNEL_RDKO_0_T_F_F_T,
        KERNEL_RDKO_0_T_F_T_T,
        KERNEL_RDKO_0_T_T_F_T,
        KERNEL_RDKO_0_T_T_T_T,
        KERNEL_EK,
        NUM_KERNELS
    };
    unsigned int m_numCTAs[NUM_KERNELS];

};

/** @brief Plan class for sparse-matrix dense-vector multiply
*
*/
class CUDPPSparseMatrixVectorMultiplyPlan : public CUDPPPlan
{
public:
    CUDPPSparseMatrixVectorMultiplyPlan(CUDPPManager *mgr, 
                                        CUDPPConfiguration config, size_t numNZElts,
                                        const void         *A,
                                        const unsigned int *rowindx, 
                                        const unsigned int *indx, size_t numRows);
    virtual ~CUDPPSparseMatrixVectorMultiplyPlan();

    CUDPPSegmentedScanPlan *m_segmentedScanPlan; //!< @internal Performs a segmented scan of type T using this plan
    void             *m_d_prod;  //!< @internal Vector of products (of an element in A and its corresponding (thats is
                                 //!            belongs to the same row) element in x; this is the input and output of 
                                 //!            segmented scan
    unsigned int     *m_d_flags; //!< @internal Vector of flags where a flag is set if an element of A is the first element
                                 //!            of its row; this is the flags vector for segmented scan
    unsigned int     *m_d_rowFinalIndex; //!< @internal Vector of row end indices, which for each row specifies an index in A
                                         //!            which is the last element of that row. Resides in GPU memory. 
    unsigned int     *m_d_rowIndex; //!< @internal Vector of row end indices, which for each row specifies an index in A
                                    //!            which is the first element of that row. Resides in GPU memory. 
    unsigned int     *m_d_index;    //!<@internal Vector of column numbers one for each element in A 
    void             *m_d_A;        //!<@internal The A matrix 
    unsigned int     *m_rowFinalIndex; //!< @internal Vector of row end indices, which for each row specifies an index in A
                                       //!            which is the last element of that row. Resides in CPU memory.
    size_t           m_numRows; //!< Number of rows
    size_t           m_numNonZeroElements; //!<Number of non-zero elements
};

/** @brief Plan class for random number generator
*
*/
class CUDPPRandPlan : public CUDPPPlan
{
public:
    CUDPPRandPlan(CUDPPManager *mgr, CUDPPConfiguration config, size_t num_elements);

    unsigned int m_seed; //!< @internal the seed for the random number generator
};

/** @brief Plan class for tridiagonal solver
*
*/
class CUDPPTridiagonalPlan : public CUDPPPlan
{
public:
    CUDPPTridiagonalPlan(CUDPPManager *mgr, CUDPPConfiguration config);
};

// Vector struct for merging SA12 and SA3
struct Vector
{
    unsigned int a;
    unsigned int b;
    unsigned int c;
    unsigned int d;  
};

/** @brief Plan class for suffix array
*
*/
class CUDPPSaPlan : public CUDPPPlan
{
public:
    CUDPPSaPlan(CUDPPManager *mgr, CUDPPConfiguration config, size_t str_length);
    virtual ~CUDPPSaPlan();
   
    // Intermediate buffers and variables during suffix array construction 
    bool *m_d_unique;                    //!< @internal the flag to mark if SA12 is fully sorted
    unsigned int* d_str_value;           //!< @internal the input unsigned int array
    unsigned int* m_d_keys_srt_12;       //!< @internal SA12
    unsigned int* m_d_keys_srt_3;        //!< @internal SA3

    Vector* m_d_aKeys;                   //!< @internal SA12 keys for final merge
    Vector* m_d_bKeys;                   //!< @internal SA3 keys for final merge
    Vector* m_d_cKeys;                   //!< @internal merging result

    unsigned int* m_d_new_str;           //!< @internal new string for recursion
    unsigned int* m_d_isa_12;	         //!< @internal ISA12
};


/** @brief Plan class for compressor
*
*/
struct encoded;
class CUDPPCompressPlan : public CUDPPPlan
{
public:
    CUDPPCompressPlan(CUDPPManager *mgr, CUDPPConfiguration config, size_t numElements);
    virtual ~CUDPPCompressPlan();

    // BWT
    unsigned int *m_d_values;
    unsigned char *m_d_bwtOut;
    CUDPPSaPlan *m_saPlan;         //!< @internal Suffix Array performs sorting permutations of the string using this plan

    // MTF
    unsigned char *m_d_mtfIn;
    unsigned char *m_d_mtfOut;
    unsigned char *m_d_lists;
    unsigned short *m_d_list_sizes;
    unsigned int npad;

    // Huffman
    unsigned char *m_d_huffCodesPacked;   // tightly pack together all huffman codes
    unsigned int *m_d_huffCodeLocations;  // keep track of where each huffman code starts
    unsigned char *m_d_huffCodeLengths;   // lengths of each huffman codes (in bits)
    unsigned int *m_d_histograms;         // histogram used to build huffman tree
    unsigned int *m_d_nCodesPacked;       // Size of all Huffman codes packed together (in bytes)
    encoded *m_d_encoded;

};

/** @brief Plan class for BWT
*
*/
class CUDPPBwtPlan : public CUDPPPlan
{
public:
    CUDPPBwtPlan(CUDPPManager *mgr, CUDPPConfiguration config, size_t numElements);
    virtual ~CUDPPBwtPlan();

    // BWT
    unsigned int *m_d_values;
    CUDPPSaPlan *m_saPlan;         //!< @internal Suffix Array performs sorting permutations of the string using this plan
};

/** @brief Plan class for MTF
*
*/
class CUDPPMtfPlan : public CUDPPPlan
{
public:
    CUDPPMtfPlan(CUDPPManager *mgr, CUDPPConfiguration config, size_t numElements);
    virtual ~CUDPPMtfPlan();

    // MTF
    unsigned char   *m_d_lists;
    unsigned short  *m_d_list_sizes;
    unsigned int    npad;
};

/** @brief Plan class for ListRank
*
*/
class CUDPPListRankPlan : public CUDPPPlan
{
public:
    CUDPPListRankPlan(CUDPPManager *mgr, CUDPPConfiguration config, size_t numElements);
    virtual ~CUDPPListRankPlan();

    // Intermediate buffers used during list ranking
    int *m_d_tmp1; //!< @internal temporary next indices array
    int *m_d_tmp2; //!< @internal temporary start indices array
    int *m_d_tmp3; //!< @internal temporary next indices array
};


/** @brief Plan class for MultiSplit
*
*/
class CUDPPMultiSplitPlan : public CUDPPPlan
{
public:
  CUDPPMultiSplitPlan(CUDPPManager *mgr, CUDPPConfiguration config,
      size_t numElements, size_t numBuckets);
    virtual ~CUDPPMultiSplitPlan();

    unsigned int m_numElements;
    unsigned int m_numBuckets;
    unsigned int *m_d_mask;
    unsigned int *m_d_out;
    unsigned int *m_d_fin;
    unsigned int *m_d_temp_keys;
    unsigned int *m_d_temp_values;
    unsigned long long int *m_d_key_value_pairs;
};

#endif // __CUDPP_PLAN_H__

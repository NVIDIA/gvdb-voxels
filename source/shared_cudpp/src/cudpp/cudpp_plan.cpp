// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: 3572$
// $Date: 2007-11-19 13:58:06 +0000 (Mon, 19 Nov 2007) $
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// ------------------------------------------------------------- 

#include "cudpp.h"
#include "cudpp_manager.h"
#include "cudpp_scan.h"
#include "cudpp_segscan.h"
#include "cudpp_compact.h"
#include "cudpp_spmvmult.h"
#include "cudpp_stringsort.h"
#include "cudpp_mergesort.h"
#include "cudpp_radixsort.h"
#include "cudpp_reduce.h"
#include "cudpp_compress.h"
#include "cudpp_listrank.h"
#include "cudpp_sa.h"
#include "cudpp_multisplit.h"
#include "cuda_util.h"
#include <cuda_runtime_api.h>
#include <assert.h>

CUDPPResult validateOptions(CUDPPConfiguration config, size_t numElements, size_t numRows, size_t /*rowPitch*/)
{
    CUDPPResult ret = CUDPP_SUCCESS;
    if ((config.options & CUDPP_OPTION_BACKWARD) && (config.options & CUDPP_OPTION_FORWARD))
        ret = CUDPP_ERROR_ILLEGAL_CONFIGURATION;
    if ((config.options & CUDPP_OPTION_EXCLUSIVE) && (config.options & CUDPP_OPTION_INCLUSIVE))
        ret = CUDPP_ERROR_ILLEGAL_CONFIGURATION;

    if (config.algorithm == CUDPP_COMPACT && numRows > 1)
        ret = CUDPP_ERROR_ILLEGAL_CONFIGURATION; //!< @todo: add support for multi-row cudppCompact

    if (config.algorithm == CUDPP_TRIDIAGONAL) {
        if (config.datatype != CUDPP_FLOAT && config.datatype != CUDPP_DOUBLE) 
            ret = CUDPP_ERROR_ILLEGAL_CONFIGURATION;
    }

    return ret;
}

/** @addtogroup publicInterface
  * @{
  */

/** @name Plan Interface
 * @{
 */


/** @brief Create a CUDPP plan 
  * 
  * A plan is a data structure containing state and intermediate storage space
  * that CUDPP uses to execute algorithms on data.  A plan is created by 
  * passing to cudppPlan() a CUDPPConfiguration that specifies the algorithm,
  * operator, datatype, and options.  The size of the data must also be passed
  * to cudppPlan(), in the \a numElements, \a numRows, and \a rowPitch 
  * arguments.  These sizes are used to allocate internal storage space at the
  * time the plan is created.  The CUDPP planner may use the sizes, options,
  * and information about the present hardware to choose optimal settings.
  *
  * Note that \a numElements is the maximum size of the array to be processed
  * with this plan.  That means that a plan may be re-used to process (for 
  * example, to sort or scan) smaller arrays.  
  * 
  * @param[out] planHandle A pointer to an opaque handle to the internal plan
  * @param[in]  cudppHandle A handle to an instance of the CUDPP library used for resource management
  * @param[in]  config The configuration struct specifying algorithm and options
  * @param[in]  numElements The maximum number of elements to be processed
  * @param[in]  numRows The number of rows (for 2D operations) to be processed
  * @param[in]  rowPitch The pitch of the rows of input data, in elements
  * @returns CUDPPResult indicating success or error condition
  */
CUDPP_DLL
CUDPPResult cudppPlan(const CUDPPHandle  cudppHandle,
                      CUDPPHandle        *planHandle,
                      CUDPPConfiguration config, 
                      size_t             numElements, 
                      size_t             numRows, 
                      size_t             rowPitch)
{
    CUDPPResult result = CUDPP_SUCCESS;

    CUDPPPlan *plan;
    CUDPPManager *mgr = CUDPPManager::getManagerFromHandle(cudppHandle);

    result = validateOptions(config, numElements, numRows, rowPitch);
    if (result != CUDPP_SUCCESS)
    {
        *planHandle = CUDPP_INVALID_HANDLE;
        return result;
    }

    switch (config.algorithm)
    {
    case CUDPP_SCAN:
        {
            plan = new CUDPPScanPlan(mgr, config, numElements, numRows, rowPitch);
            break;
        }
    case CUDPP_COMPACT:
        {
            plan = new CUDPPCompactPlan(mgr, config, numElements, numRows, rowPitch);
            break;
        }
    case CUDPP_SORT_RADIX:
        {
            plan = new CUDPPRadixSortPlan(mgr, config, numElements);
            break;
        }
    case CUDPP_SORT_MERGE:
        {
            plan = new CUDPPMergeSortPlan(mgr, config, numElements);
            break;
        }
    case CUDPP_SORT_STRING:
        {
            plan = new CUDPPStringSortPlan(mgr, config, numElements, rowPitch);
            break;
        }	
    case CUDPP_SEGMENTED_SCAN:
        {
            plan = new CUDPPSegmentedScanPlan(mgr, config, numElements);
            break;
        }
    case CUDPP_RAND_MD5:
        {
            plan = new CUDPPRandPlan(mgr, config, numElements);
            break;
        }
    case (CUDPP_TRIDIAGONAL):
        {
            plan = new CUDPPTridiagonalPlan(mgr, config);
            break;
        }
    case CUDPP_REDUCE:
        {
            plan = new CUDPPReducePlan(mgr, config, numElements);
            break;
        }
    case CUDPP_COMPRESS:
        {
            plan = new CUDPPCompressPlan(mgr, config, numElements);
            break;
        }
    case CUDPP_BWT:
        {
            plan = new CUDPPBwtPlan(mgr, config, numElements);
            break;
        }
    case CUDPP_MTF:
        {
            plan = new CUDPPMtfPlan(mgr, config, numElements);
            break;
        }
    case CUDPP_LISTRANK:
        {
            plan = new CUDPPListRankPlan(mgr, config, numElements);
            break;
        }
    case CUDPP_SA:
        {
            plan = new CUDPPSaPlan(mgr, config, numElements);
            break;
        }
    case CUDPP_MULTISPLIT:
        {
            plan = new CUDPPMultiSplitPlan(mgr, config, numElements, numRows);
            break;
        }

    default:
        return CUDPP_ERROR_ILLEGAL_CONFIGURATION; 
        break;
    }

    if (!plan)
        return CUDPP_ERROR_UNKNOWN;
    else
    {
        *planHandle = plan->getHandle();
        return CUDPP_SUCCESS;
    }
}

/** @brief Destroy a CUDPP Plan
  *
  * Deletes the plan referred to by \a planHandle and all associated internal
  * storage.
  * 
  * @param[in] planHandle The CUDPPHandle to the plan to be destroyed
  * @returns CUDPPResult indicating success or error condition
  */
CUDPP_DLL
CUDPPResult cudppDestroyPlan(CUDPPHandle planHandle)
{
    if (planHandle == CUDPP_INVALID_HANDLE)
        return CUDPP_ERROR_INVALID_HANDLE;

    CUDPPPlan* plan = getPlanPtrFromHandle<CUDPPPlan>(planHandle);

    switch (plan->m_config.algorithm)
    {
    case CUDPP_SCAN:
        {
            delete static_cast<CUDPPScanPlan*>(plan);
            break;
        }
    case CUDPP_COMPACT:
        {
            delete static_cast<CUDPPCompactPlan*>(plan);
            break;
        }
    case CUDPP_SORT_RADIX:
        {
            delete static_cast<CUDPPRadixSortPlan*>(plan);
            break;
        }
    case CUDPP_SORT_MERGE:
        {
            delete static_cast<CUDPPMergeSortPlan*>(plan);
            break;
        }
    case CUDPP_SORT_STRING:
        {
            delete static_cast<CUDPPStringSortPlan*>(plan);
            break;
        }	
    case CUDPP_SEGMENTED_SCAN:
        {
            delete static_cast<CUDPPSegmentedScanPlan*>(plan);
            break;
        }
    case CUDPP_RAND_MD5:
        {
            delete static_cast<CUDPPRandPlan*>(plan);
            break;
        }
    case (CUDPP_TRIDIAGONAL):
        {
            delete static_cast<CUDPPTridiagonalPlan*>(plan);
            break;
        }
    case CUDPP_REDUCE:
        {
            delete static_cast<CUDPPReducePlan*>(plan);
            break;
        }
    case CUDPP_COMPRESS:
        {
            delete static_cast<CUDPPCompressPlan*>(plan);
            break;
        }
    case CUDPP_BWT:
        {
            delete static_cast<CUDPPBwtPlan*>(plan);
            break;
        }
    case CUDPP_MTF:
        {
            delete static_cast<CUDPPMtfPlan*>(plan);
            break;
        }
    case CUDPP_LISTRANK:
        {
            delete static_cast<CUDPPListRankPlan*>(plan);
            break;
        }
    case CUDPP_SA:
        {
            delete static_cast<CUDPPSaPlan*>(plan);
            break;
        }
    case CUDPP_MULTISPLIT:
        {
            delete static_cast<CUDPPMultiSplitPlan*>(plan);
            break;
        }
    default:
        return CUDPP_ERROR_ILLEGAL_CONFIGURATION; 
        break;
    }

    plan = 0;
    return CUDPP_SUCCESS;
}

/** @brief Create a CUDPP Sparse Matrix Object 
  *
  * The sparse matrix plan is a data structure containing state and
  * intermediate storage space that CUDPP uses to perform sparse
  * matrix dense vector multiply. This plan is created by passing to
  * CUDPPSparseMatrixVectorMultiplyPlan() a CUDPPConfiguration that
  * specifies the algorithm (sprarse matrix-dense vector multiply) and
  * datatype, along with the sparse matrix itself in CSR format. The
  * number of non-zero elements in the sparse matrix must also be
  * passed as \a numNonZeroElements. This is used to allocate internal
  * storage space at the time the sparse matrix plan is created.
  *
  * @param[out] sparseMatrixHandle A pointer to an opaque handle to the sparse matrix object
  * @param[in]  cudppHandle A handle to an instance of the CUDPP library used for resource management
  * @param[in]  config The configuration struct specifying algorithm and options
  * @param[in]  numNonZeroElements The number of non zero elements in the sparse matrix 
  * @param[in]  numRows This is the number of rows in y, x and A for y = A * x
  * @param[in]  A The matrix data
  * @param[in]  h_rowIndices An array containing the index of the start of each row in \a A
  * @param[in]  h_indices An array containing the index of each nonzero element in \a A
  * @returns CUDPPResult indicating success or error condition
  */
CUDPP_DLL
CUDPPResult cudppSparseMatrix(const CUDPPHandle  cudppHandle,
                              CUDPPHandle        *sparseMatrixHandle, 
                              CUDPPConfiguration config, 
                              size_t             numNonZeroElements, 
                              size_t             numRows, 
                              const void         *A,
                              const unsigned int *h_rowIndices,
                              const unsigned int *h_indices)
{
    CUDPPResult result = CUDPP_SUCCESS;

    CUDPPPlan *sparseMatrix;
    CUDPPManager *mgr = CUDPPManager::getManagerFromHandle(cudppHandle);

    if ((config.algorithm != CUDPP_SPMVMULT) || 
        (numNonZeroElements <= 0) || (numRows <= 0))
    {
        result = CUDPP_ERROR_ILLEGAL_CONFIGURATION;
    }

    if (result != CUDPP_SUCCESS)
    {
        *sparseMatrixHandle = CUDPP_INVALID_HANDLE;
        return result;
    }

    sparseMatrix = 
        new CUDPPSparseMatrixVectorMultiplyPlan(mgr, config, numNonZeroElements, A, 
                                                h_rowIndices, h_indices, numRows);

    if (sparseMatrix)
    {
        *sparseMatrixHandle = sparseMatrix->getHandle();
        return CUDPP_SUCCESS;
    }
    else
    {
        return CUDPP_ERROR_UNKNOWN;
    }

}

/** @brief Destroy a CUDPP Sparse Matrix Object
  *
  * Deletes the sparse matrix data and plan referred to by \a
  * sparseMatrixHandle and all associated internal storage.
  * 
  * @param[in] sparseMatrixHandle The CUDPPHandle to the matrix object to be 
  *                               destroyed
  * @returns CUDPPResult indicating success or error condition
  */
CUDPP_DLL
CUDPPResult cudppDestroySparseMatrix(CUDPPHandle sparseMatrixHandle)
{
    if (sparseMatrixHandle == CUDPP_INVALID_HANDLE)
        return CUDPP_ERROR_INVALID_HANDLE;

    CUDPPSparseMatrixVectorMultiplyPlan* plan = 
        getPlanPtrFromHandle<CUDPPSparseMatrixVectorMultiplyPlan>(sparseMatrixHandle);
    delete plan;
    plan = 0;
    return CUDPP_SUCCESS;
}

/** @} */ // end Plan Interface
/** @} */ // end publicInterface


/** @brief Plan base class constructor
  * 
  * @param[in]  mgr pointer to the CUDPPManager
  * @param[in]  config The configuration struct specifying algorithm and options
  * @param[in]  numElements The maximum number of elements to be processed
  * @param[in]  numRows The number of rows (for 2D operations) to be processed
  * @param[in]  rowPitch The pitch of the rows of input data, in elements
  */
CUDPPPlan::CUDPPPlan(CUDPPManager *mgr,
                     CUDPPConfiguration config, 
                     size_t numElements, 
                     size_t numRows, 
                     size_t rowPitch)
: m_config(config),
  m_numElements(numElements),
  m_numRows(numRows),
  m_rowPitch(rowPitch),
  m_planManager(mgr)
{
}

/** @brief Scan Plan constructor
* 
* @param[in]  mgr pointer to the CUDPPManager
* @param[in]  config The configuration struct specifying algorithm and options
* @param[in]  numElements The maximum number of elements to be scanned
* @param[in]  numRows The maximum number of rows (for 2D operations) to be scanned
* @param[in]  rowPitch The pitch of the rows of input data, in elements
*/
CUDPPScanPlan::CUDPPScanPlan(CUDPPManager *mgr,
                             CUDPPConfiguration config, 
                             size_t numElements, 
                             size_t numRows, 
                             size_t rowPitch)
: CUDPPPlan(mgr, config, numElements, numRows, rowPitch),
  m_blockSums(0),
  m_rowPitches(0),
  m_numEltsAllocated(0),
  m_numRowsAllocated(0),
  m_numLevelsAllocated(0)
{
    allocScanStorage(this);
}

/** @brief CUDPP scan plan destructor */
CUDPPScanPlan::~CUDPPScanPlan()
{
    freeScanStorage(this);
}

/** @brief SegmentedScan Plan constructor
* 
* @param[in]  mgr pointer to the CUDPPManager
* @param[in]  config The configuration struct specifying options
* @param[in]  numElements The maximum number of elements to be scanned
*/
CUDPPSegmentedScanPlan::CUDPPSegmentedScanPlan(CUDPPManager *mgr,
                                               CUDPPConfiguration config, 
                                               size_t numElements)
: CUDPPPlan(mgr, config, numElements, 1, 0),
  m_blockSums(0),
  m_blockFlags(0),
  m_blockIndices(0),
  m_numEltsAllocated(0),
  m_numLevelsAllocated(0)
{
    allocSegmentedScanStorage(this);
}

/** @brief SegmentedScan plan destructor */
CUDPPSegmentedScanPlan::~CUDPPSegmentedScanPlan()
{
    freeSegmentedScanStorage(this);
}

/** @brief Compact Plan constructor
* 
* @param[in]  mgr pointer to the CUDPPManager
* @param[in]  config The configuration struct specifying options
* @param[in]  numElements The maximum number of elements to be compacted
* @param[in]  numRows The number of rows (for 2D operations) to be compacted
* @param[in]  rowPitch The pitch of the rows of input data, in elements
*/
CUDPPCompactPlan::CUDPPCompactPlan(CUDPPManager *mgr,
                                   CUDPPConfiguration config, 
                                   size_t numElements, 
                                   size_t numRows, 
                                   size_t rowPitch)
: CUDPPPlan(mgr, config, numElements, numRows, rowPitch),
  m_d_outputIndices(0)
{
    assert(numRows == 1); //!< @todo Add support for multirow compaction

    CUDPPConfiguration scanConfig = 
    { 
      CUDPP_SCAN, 
      CUDPP_ADD, 
      CUDPP_UINT, 
      (config.options & CUDPP_OPTION_BACKWARD) ? 
        CUDPP_OPTION_BACKWARD | CUDPP_OPTION_EXCLUSIVE : 
        CUDPP_OPTION_FORWARD  | CUDPP_OPTION_EXCLUSIVE 
    };
    m_scanPlan = new CUDPPScanPlan(mgr, scanConfig, numElements, numRows, rowPitch);

    allocCompactStorage(this);
}

/** @brief Compact plan destructor */
CUDPPCompactPlan::~CUDPPCompactPlan()
{
    delete m_scanPlan;
    freeCompactStorage(this);
}

/** @brief Reduce Plan constructor
* 
* @param[in]  mgr pointer to the CUDPPManager
* @param[in]  config The configuration struct specifying options
* @param[in]  numElements The maximum number of elements to be reduced
*/
CUDPPReducePlan::CUDPPReducePlan(CUDPPManager *mgr,
                                 CUDPPConfiguration config, 
                                 size_t numElements)
: CUDPPPlan(mgr, config, numElements, 1, 0),
  m_threadsPerBlock(REDUCE_CTA_SIZE),
  m_maxBlocks(64)
{
    allocReduceStorage(this);
}

/** @brief Reduce plan destructor */
CUDPPReducePlan::~CUDPPReducePlan()
{
    freeReduceStorage(this);
}

/** @brief Merge Sort Plan consturctor
* @param[in]  mgr pointer to the CUDPPManager
* @param[in]  config The configuration struct specifying options
* @param[in]  numElements The maximum number of elements to be sorted
*/
CUDPPMergeSortPlan::CUDPPMergeSortPlan(CUDPPManager *mgr,
                                       CUDPPConfiguration config,
				       size_t numElements)
: CUDPPPlan(mgr, config, numElements, 1, 0), m_tempKeys(0), m_tempValues(0)
{
	m_subPartitions = 4;
	m_swapPoint = 64;
	m_numElements = numElements;
	allocMergeSortStorage(this);

}

/** @brief Merge sort plan destructor */
CUDPPMergeSortPlan::~CUDPPMergeSortPlan()
{
    freeMergeSortStorage(this);
}



/** @brief String Sort Plan consturctor
* @param[in]  mgr pointer to the CUDPPManager
* @param[in]  config The configuration struct specifying options
* @param[in]  numElements The maximum number of elements to be sorted
* @param[in]  stringArrayLength The length of our input string (in uint)
*/
CUDPPStringSortPlan::CUDPPStringSortPlan(CUDPPManager *mgr,
										 CUDPPConfiguration config,
										 size_t numElements, 
										 size_t stringArrayLength)
: CUDPPPlan(mgr, config, numElements, stringArrayLength, 0)
{ 
	m_subPartitions = 4;
	m_swapPoint = 64;
	
	CUDPPConfiguration scanConfig = 
    { 
      CUDPP_SCAN, 
      CUDPP_ADD, 
      CUDPP_UINT, 
      CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE 
    };    
	

	m_scanPlan = new CUDPPScanPlan(mgr, scanConfig, numElements+1, 1, 0);	
	m_numElements = numElements;
	allocStringSortStorage(this);	
}

/** @brief String sort plan destructor */
CUDPPStringSortPlan::~CUDPPStringSortPlan()
{
    freeStringSortStorage(this);
}
/** @brief Radix Sort Plan constructor
* 
* @param[in]  mgr pointer to the CUDPPManager
* @param[in]  config The configuration struct specifying options
* @param[in]  numElements The maximum number of elements to be sorted

*/
CUDPPRadixSortPlan::CUDPPRadixSortPlan(CUDPPManager *mgr, 
                                       CUDPPConfiguration config, 
                                       size_t numElements)
: CUDPPPlan(mgr, config, numElements, 1, 0),
  m_bKeysOnly((m_config.options & CUDPP_OPTION_KEYS_ONLY) != 0),
  m_bBackward((m_config.options & CUDPP_OPTION_BACKWARD) != 0),
  m_scanPlan(0),
  m_tempKeys(0),    
  m_tempValues(0),
  m_counters(0),
  m_countersSum(0),
  m_blockOffsets(0) 
{
    size_t numBlocks2 = ((numElements % (SORT_CTA_SIZE * 2)) == 0) ?
            (numElements / (SORT_CTA_SIZE * 2)) : (numElements / (SORT_CTA_SIZE * 2) + 1);

    CUDPPConfiguration scanConfig = 
    { 
      CUDPP_SCAN, 
      CUDPP_ADD, 
      CUDPP_UINT, 
      CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE 
    };    

    m_scanPlan = new CUDPPScanPlan(mgr, scanConfig, numBlocks2*16, 1, 0);    
        
    allocRadixSortStorage(this); 
}

/** @brief Radix sort plan destructor */
CUDPPRadixSortPlan::~CUDPPRadixSortPlan()
{
    delete m_scanPlan;
    freeRadixSortStorage(this);
}

/** @brief SparseMatrixVectorMultiply Plan constructor
* 
* @param[in]  mgr pointer to the CUDPPManager
* @param[in]  config The configuration struct specifying options
* @param[in]  numNonZeroElements The number of non-zero elements in sparse matrix
* @param[in]  A Array of non-zero matrix elements
* @param[in]  rowIndex Array of indices of the first element of each row 
*                     in the "flattened" version of the sparse matrix
* @param[in]  index Array of indices of non-zero elements in the matrix
* @param[in]  numRows The number of rows in the sparse matrix
*/
CUDPPSparseMatrixVectorMultiplyPlan::CUDPPSparseMatrixVectorMultiplyPlan(CUDPPManager   *mgr,
                                                                         CUDPPConfiguration config,
                                                                         size_t             numNonZeroElements,
                                                                         const void         *A,
                                                                         const unsigned int *rowIndex,
                                                                         const unsigned int *index,
                                                                         size_t             numRows
                                                                         )
: CUDPPPlan(mgr, config, numNonZeroElements, 1, 0),
  m_segmentedScanPlan(0),
  m_d_prod(0),
  m_d_flags(0),
  m_d_rowFinalIndex(0),
  m_rowFinalIndex(0),
  m_numRows(numRows),
  m_numNonZeroElements(numNonZeroElements)  
{
    CUDPPConfiguration segScanConfig = 
    { 
      CUDPP_SEGMENTED_SCAN, 
      CUDPP_ADD, 
      config.datatype, 
      (CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE) 
    };
    m_segmentedScanPlan = new CUDPPSegmentedScanPlan(mgr, segScanConfig, m_numNonZeroElements);

    // Generate an array of the indices of the last element of each row
    // in the "flattened" version of the sparse matrix
    m_rowFinalIndex = new unsigned int [m_numRows];
    for (unsigned int i=0; i < m_numRows; ++i)
    {
        if (i < m_numRows-1)
            m_rowFinalIndex[i] = rowIndex[i+1];
        else
            m_rowFinalIndex[i] = (unsigned int)numNonZeroElements;
    }

    allocSparseMatrixVectorMultiplyStorage(this, A, rowIndex, index);
}

/** @brief Sparse matrix-vector plan destructor */
CUDPPSparseMatrixVectorMultiplyPlan::~CUDPPSparseMatrixVectorMultiplyPlan()
{
    freeSparseMatrixVectorMultiplyStorage(this);
    delete m_segmentedScanPlan;
    delete [] m_rowFinalIndex;
}

/** @brief CUDPP Rand Plan Constructor
  *
  * @param[in]  mgr pointer to the CUDPPManager
  * @param[in] config The configuration struct specifying options
  * @param[in] num_elements The number of elements to generate random bits for
  */
CUDPPRandPlan::CUDPPRandPlan(CUDPPManager *mgr, CUDPPConfiguration config, size_t num_elements) 
 : CUDPPPlan(mgr, config, num_elements, 1, 0),
   m_seed(0)
{
    
}

/** @brief CUDPP Tridiagonal Plan Constructor
  *
  * @param[in]  mgr pointer to the CUDPPManager
  * @param[in] config The configuration struct specifying options
  */
CUDPPTridiagonalPlan::CUDPPTridiagonalPlan(CUDPPManager *mgr, CUDPPConfiguration config) 
 : CUDPPPlan(mgr, config, 0, 0, 0)
{
    
}

/** @brief CUDPP Compress Plan Constructor
  *
  * @param[in] mgr pointer to the CUDPPManager
  * @param[in] config The configuration struct specifying options
  * @param[in] numElements The maximum number of elements to be compressed
  */
CUDPPCompressPlan::CUDPPCompressPlan(CUDPPManager *mgr, CUDPPConfiguration config, size_t numElements) 
 : CUDPPPlan(mgr, config, numElements, 1, 0)
{
    m_saPlan = new CUDPPSaPlan(mgr, config, numElements);
    allocCompressStorage(this);
}

/** @brief Compress plan destructor */
CUDPPCompressPlan::~CUDPPCompressPlan()
{
    delete m_saPlan;
    freeCompressStorage(this);
}

/** @brief CUDPP BWT Plan Constructor
  *
  * @param[in] mgr pointer to the CUDPPManager
  * @param[in] config The configuration struct specifying options
  * @param[in] numElements The maximum number of elements to be transformed
  */
CUDPPBwtPlan::CUDPPBwtPlan(CUDPPManager *mgr, CUDPPConfiguration config, size_t numElements) 
 : CUDPPPlan(mgr, config, numElements, 1, 0)
{
    m_saPlan = new CUDPPSaPlan(mgr, config, numElements);
    allocBwtStorage(this);
}

/** @brief BWT plan destructor */
CUDPPBwtPlan::~CUDPPBwtPlan()
{
    //delete m_saPlan;
    freeBwtStorage(this);
}

/** @brief CUDPP MTF Plan Constructor
  *
  * @param[in] mgr pointer to the CUDPPManager
  * @param[in] config The configuration struct specifying options
  * @param[in] numElements The maximum number of elements to be moved to front
  */
CUDPPMtfPlan::CUDPPMtfPlan(CUDPPManager *mgr, CUDPPConfiguration config, size_t numElements) 
 : CUDPPPlan(mgr, config, numElements, 1, 0)
{
    allocMtfStorage(this);
}

/** @brief MTF plan destructor */
CUDPPMtfPlan::~CUDPPMtfPlan()
{
    freeMtfStorage(this);
}

/** @brief CUDPP ListRank Plan Constructor
  *
  * @param[in] mgr pointer to the CUDPPManager
  * @param[in] config The configuration struct specifying options
  * @param[in] numElements The maximum number of elements to be ranked
  */
CUDPPListRankPlan::CUDPPListRankPlan(CUDPPManager *mgr, CUDPPConfiguration config, size_t numElements) 
 : CUDPPPlan(mgr, config, numElements, 1, 0)
{
    allocListRankStorage(this);
}

/** @brief ListRank plan destructor */
CUDPPListRankPlan::~CUDPPListRankPlan()
{
    freeListRankStorage(this);
}

/** @brief CUDPP Suffix Array Plan Constructor
  *
  * @param[in] mgr pointer to the CUDPPManager
  * @param[in] config The configuration struct specifying options
  * @param[in] str_length the length of the string including $
  */
CUDPPSaPlan::CUDPPSaPlan(CUDPPManager *mgr, CUDPPConfiguration config, size_t str_length)
 : CUDPPPlan(mgr, config, str_length, 1, 0)
{
    allocSaStorage(this);
}

/** brief SA plan destructor*/
CUDPPSaPlan::~CUDPPSaPlan()
{
    freeSaStorage(this);
}

/** @brief CUDPP MultiSplit Plan Constructor
  *
  * @param[in] mgr pointer to the CUDPPManager
  * @param[in] config The configuration struct specifying options
  * @param[in] The number of elements to be split
  * @param[in] The number of buckets
  *
  */
CUDPPMultiSplitPlan::CUDPPMultiSplitPlan(CUDPPManager *mgr,
    CUDPPConfiguration config, size_t numElements, size_t numBuckets)
 : CUDPPPlan(mgr, config, numElements, 1, 0)
{
    m_numElements = numElements;
    m_numBuckets = numBuckets;

    allocMultiSplitStorage(this);

    // use the allocated array for temporary storage of keys and values
    m_d_temp_keys = (unsigned int *) m_d_key_value_pairs;
    m_d_temp_values = (unsigned int *) m_d_key_value_pairs + numElements;
}

/** brief MultiSplit Plan Destructor*/
CUDPPMultiSplitPlan::~CUDPPMultiSplitPlan()
{
    freeMultiSplitStorage(this);
}

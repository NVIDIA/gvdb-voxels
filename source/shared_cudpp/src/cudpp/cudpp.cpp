// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: 5636 $
// $Date: 2009-07-02 13:39:38 +1000 (Thu, 02 Jul 2009) $
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// -------------------------------------------------------------

/**
 * @file
 * cudpp.cpp
 *
 * @brief Main library source file.  Implements wrappers for public
 * interface.
 *
 * Main library source file.  Implements wrappers for public
 * interface.  These wrappers call application-level operators.
 * As this grows we may decide to partition into multiple source
 * files.
 */

/**
 * \defgroup publicInterface CUDPP Public Interface
 * The CUDA public interface comprises the functions, structs, and enums
 * defined in cudpp.h.  Public interface functions call functions in the
 * \link cudpp_app Application-Level\endlink interface. The public
 * interface functions include Plan Interface functions and Algorithm
 * Interface functions.  Plan Interface functions are used for creating
 * CUDPP Plan objects that contain configuration details, intermediate
 * storage space, and in the case of cudppSparseMatrix(), data.  The
 * Algorithm Interface is the set of functions that do the real work
 * of CUDPP, such as cudppScan() and cudppSparseMatrixVectorMultiply().
 *
 * @{
 */

/** @name Algorithm Interface
 * @{
 */

#include "cudpp.h"
#include "cudpp_manager.h"
#include "cudpp_scan.h"
#include "cudpp_segscan.h"
#include "cudpp_compact.h"
#include "cudpp_spmvmult.h"
#include "cudpp_mergesort.h"
#include "cudpp_multisplit.h"
#include "cudpp_radixsort.h"
#include "cudpp_rand.h"
#include "cudpp_reduce.h"
#include "cudpp_stringsort.h"
#include "cudpp_tridiagonal.h"
#include "cudpp_compress.h"
#include "cudpp_sa.h"
#include "cudpp_listrank.h"
#include <stdio.h>

/**
 * @brief Performs a scan operation of numElements on its input in
 * GPU memory (d_in) and places the output in GPU memory
 * (d_out), with the scan parameters specified in the plan pointed to by
 * planHandle.

 * The input to a scan operation is an input array, a binary associative
 * operator (like + or max), and an identity element for that operator
 * (+'s identity is 0). The output of scan is the same size as its input.
 * Informally, the output at each element is the result of operator
 * applied to each input that comes before it. For instance, the
 * output of sum-scan at each element is the sum of all the input
 * elements before that input.
 *
 * More formally, for associative operator
 * @htmlonly&oplus;@endhtmlonly@latexonly$\oplus$@endlatexonly,
 * <var>out<sub>i</sub></var> = <var>in<sub>0</sub></var>
 * @htmlonly&oplus;@endhtmlonly@latexonly$\oplus$@endlatexonly
 * <var>in<sub>1</sub></var>
 * @htmlonly&oplus;@endhtmlonly@latexonly$\oplus$@endlatexonly ...
 * @htmlonly&oplus;@endhtmlonly@latexonly$\oplus$@endlatexonly
 * <var>in<sub>i-1</sub></var>.
 *
 * CUDPP supports "exclusive" and "inclusive" scans. For the ADD operator,
 * an exclusive scan computes the sum of all input elements before the
 * current element, while an inclusive scan computes the sum of all input
 * elements up to and including the current element.
 *
 * Before calling scan, create an internal plan using cudppPlan().
 *
 * After you are finished with the scan plan, clean up with cudppDestroyPlan().
 *
 * @param[in] planHandle Handle to plan for this scan
 * @param[out] d_out output of scan, in GPU memory
 * @param[in] d_in input to scan, in GPU memory
 * @param[in] numElements number of elements to scan
 * @returns CUDPPResult indicating success or error condition
 *
 * @see cudppPlan, cudppDestroyPlan
 */
CUDPP_DLL
CUDPPResult cudppScan(const CUDPPHandle planHandle,
                      void              *d_out,
                      const void        *d_in,
                      size_t            numElements)
{
    CUDPPScanPlan *plan =
        (CUDPPScanPlan*)getPlanPtrFromHandle<CUDPPScanPlan>(planHandle);

    if (plan != NULL)
    {
        if (plan->m_config.algorithm != CUDPP_SCAN)
            return CUDPP_ERROR_INVALID_PLAN;

        cudppScanDispatch(d_out, d_in, numElements, 1, plan);
        return CUDPP_SUCCESS;
    }
    else
        return CUDPP_ERROR_INVALID_HANDLE;
}

/**
 * @brief Performs a segmented scan operation of numElements on its input in
 * GPU memory (d_idata) and places the output in GPU memory
 * (d_out), with the scan parameters specified in the plan pointed to by
 * planHandle.

 * The input to a segmented scan operation is an input array of data,
 * an input array of flags which demarcate segments, a binary associative
 * operator (like + or max), and an identity element for that operator
 * (+'s identity is 0). The array of flags is the same length as the input
 * with 1 marking the the first element of a segment and 0 otherwise. The
 * output of segmented scan is the same size as its input. Informally, the
 * output at each element is the result of operator applied to each input
 * that comes before it in that segment. For instance, the output of
 * segmented sum-scan at each element is the sum of all the input elements
 * before that input in that segment.
 *
 * More formally, for associative operator
 * @htmlonly&oplus;@endhtmlonly@latexonly$\oplus$@endlatexonly,
 * <var>out<sub>i</sub></var> = <var>in<sub>k</sub></var>
 * @htmlonly&oplus;@endhtmlonly@latexonly$\oplus$@endlatexonly
 * <var>in<sub>k+1</sub></var>
 * @htmlonly&oplus;@endhtmlonly@latexonly$\oplus$@endlatexonly ...
 * @htmlonly&oplus;@endhtmlonly@latexonly$\oplus$@endlatexonly
 * <var>in<sub>i-1</sub></var>. <i>k</i> is the index of the first
 * element of the segment in which <i>i</i> lies.
 *
 * We support both "exclusive" and "inclusive" variants. For a
 * segmented sum-scan, the exclusive variant computes the sum of all
 * input elements before the current element in that segment, while
 * the inclusive variant computes the sum of all input elements up to
 * and including the current element, in that segment.
 *
 * Before calling segmented scan, create an internal plan using cudppPlan().
 *
 * After you are finished with the scan plan, clean up with cudppDestroyPlan().
 * @param[in] planHandle Handle to plan for this scan
 * @param[out] d_out output of segmented scan, in GPU memory
 * @param[in] d_idata input data to segmented scan, in GPU memory
 * @param[in] d_iflags input flags to segmented scan, in GPU memory
 * @param[in] numElements number of elements to perform segmented scan on
 * @returns CUDPPResult indicating success or error condition
 *
 * @see cudppPlan, cudppDestroyPlan
 */
CUDPP_DLL
CUDPPResult cudppSegmentedScan(const CUDPPHandle  planHandle,
                               void               *d_out,
                               const void         *d_idata,
                               const unsigned int *d_iflags,
                               size_t             numElements)
{
    CUDPPSegmentedScanPlan *plan =
        (CUDPPSegmentedScanPlan*)getPlanPtrFromHandle<CUDPPSegmentedScanPlan>(planHandle);

    if (plan != NULL)
    {
        if (plan->m_config.algorithm != CUDPP_SEGMENTED_SCAN)
            return CUDPP_ERROR_INVALID_PLAN;

        cudppSegmentedScanDispatch(d_out, d_idata, d_iflags, numElements, plan);
        return CUDPP_SUCCESS;
    }
    else
        return CUDPP_ERROR_INVALID_HANDLE;
}

/**
 * @brief Performs numRows parallel scan operations of numElements
 * each on its input (d_in) and places the output in d_out,
 * with the scan parameters set by config. Exactly like cudppScan
 * except that it runs on multiple rows in parallel.
 *
 * Note that to achieve good performance with cudppMultiScan one should
 * allocate the device arrays passed to it so that all rows are aligned
 * to the correct boundaries for the architecture the app is running on.
 * The easy way to do this is to use cudaMallocPitch() to allocate a
 * 2D array on the device.  Use the \a rowPitch parameter to cudppPlan()
 * to specify this pitch. The easiest way is to pass the device pitch
 * returned by cudaMallocPitch to cudppPlan() via \a rowPitch.
 *
 * @param[in] planHandle handle to CUDPPScanPlan
 * @param[out] d_out output of scan, in GPU memory
 * @param[in] d_in input to scan, in GPU memory
 * @param[in] numElements number of elements (per row) to scan
 * @param[in] numRows number of rows to scan in parallel
 * @returns CUDPPResult indicating success or error condition
 *
 * @see cudppScan, cudppPlan
 */
CUDPP_DLL
CUDPPResult cudppMultiScan(const CUDPPHandle planHandle,
                           void              *d_out,
                           const void        *d_in,
                           size_t            numElements,
                           size_t            numRows)
{
    CUDPPScanPlan *plan =
        (CUDPPScanPlan*)getPlanPtrFromHandle<CUDPPScanPlan>(planHandle);
    if (plan != NULL)
    {
        if (plan->m_config.algorithm != CUDPP_SCAN)
            return CUDPP_ERROR_INVALID_PLAN;

        cudppScanDispatch(d_out, d_in, numElements, numRows, plan);
        return CUDPP_SUCCESS;
    }
    else
        return CUDPP_ERROR_INVALID_HANDLE;
}


/**
 * @brief Given an array \a d_in and an array of 1/0 flags in \a
 * deviceValid, returns a compacted array in \a d_out of corresponding
 * only the "valid" values from \a d_in.
 *
 * Takes as input an array of elements in GPU memory
 * (\a d_in) and an equal-sized unsigned int array in GPU memory
 * (\a deviceValid) that indicate which of those input elements are
 * valid. The output is a packed array, in GPU memory, of only those
 * elements marked as valid.
 *
 * Internally, uses cudppScan.
 *
 * Example:
 * \code
 * d_in    = [ a b c d e f ]
 * deviceValid = [ 1 0 1 1 0 1 ]
 * d_out   = [ a c d f ]
 * \endcode
 *
 * @todo [MJH] We need to evaluate whether cudppCompact should be a
 * core member of the public interface. It's not clear to me that what
 * the user always wants is a final compacted array. Often one just
 * wants the array of indices to which each input element should go in
 * the output. The split() routine used in radix sort might make more
 * sense to expose.
 *
 * @param[in] planHandle handle to CUDPPCompactPlan
 * @param[out] d_out compacted output
 * @param[out] d_numValidElements set during cudppCompact; is set with the
 * number of elements valid flags in the d_isValid input array
 * @param[in] d_in input to compact
 * @param[in] d_isValid which elements in d_in are valid
 * @param[in] numElements number of elements in d_in
 * @returns CUDPPResult indicating success or error condition
 */
CUDPP_DLL
CUDPPResult cudppCompact(const CUDPPHandle  planHandle,
                         void               *d_out,
                         size_t             *d_numValidElements,
                         const void         *d_in,
                         const unsigned int *d_isValid,
                         size_t             numElements)
{
    CUDPPCompactPlan *plan =
        (CUDPPCompactPlan*)getPlanPtrFromHandle<CUDPPCompactPlan>(planHandle);

    if (plan != NULL)
    {
        if (plan->m_config.algorithm != CUDPP_COMPACT)
            return CUDPP_ERROR_INVALID_PLAN;

        cudppCompactDispatch(d_out, d_numValidElements, d_in, d_isValid,
            numElements, plan);
        return CUDPP_SUCCESS;
    }
    else
        return CUDPP_ERROR_INVALID_HANDLE;
}

/**
 * @brief Reduces an array to a single element using a binary associative operator
 *
 * For example, if the operator is CUDPP_ADD, then:
 * \code
 * d_in    = [ 3 2 0 1 -4 5 0 -1 ]
 * d_out   = [ 6 ]
 * \endcode
 *
 * If the operator is CUDPP_MIN, then:
 * \code
 * d_in    = [ 3 2 0 1 -4 5 0 -1 ]
 * d_out   = [ -4 ]
 * \endcode
 *
 * Limits: \a numElements must be at least 1, and is currently limited
 * only by the addressable memory in CUDA (and the output accuracy is
 * limited by numerical precision).
 *
 * @param[in] planHandle handle to CUDPPReducePlan
 * @param[out] d_out Output of reduce (a single element) in GPU memory.
 *                   Must be a pointer to an array of at least a single element.
 * @param[in] d_in Input array to reduce in GPU memory.
 *                 Must be a pointer to an array of at least \a numElements elements.
 * @param[in] numElements the number of elements to reduce.
 * @returns CUDPPResult indicating success or error condition
 *
 * @see cudppPlan
 */
CUDPP_DLL
CUDPPResult cudppReduce(const CUDPPHandle planHandle,
                        void              *d_out,
                        const void        *d_in,
                        size_t            numElements)
{
    CUDPPReducePlan *plan =
        (CUDPPReducePlan*)getPlanPtrFromHandle<CUDPPReducePlan>(planHandle);

    if (plan != NULL)
    {
        if (plan->m_config.algorithm != CUDPP_REDUCE)
            return CUDPP_ERROR_INVALID_PLAN;

        cudppReduceDispatch(d_out, d_in, numElements, plan);
        return CUDPP_SUCCESS;
    }
    else
        return CUDPP_ERROR_INVALID_HANDLE;
}

/**
 * @brief Sorts key-value pairs or keys only
 *
 * Takes as input an array of keys in GPU memory
 * (d_keys) and an optional array of corresponding values,
 * and outputs sorted arrays of keys and (optionally) values in place.
 * Radix sort or Merge sort is selected through the configuration (.algorithm)
 * Key-value and key-only sort is selected through the configuration of
 * the plan, using the options CUDPP_OPTION_KEYS_ONLY and
 * CUDPP_OPTION_KEY_VALUE_PAIRS.
 *
 * Supported key types are CUDPP_FLOAT and CUDPP_UINT.  Values can be
 * any 32-bit type (internally, values are treated only as a payload
 * and cast to unsigned int).
 *
 * @todo Determine if we need to provide an "out of place" sort interface.
 *
 * @param[in] planHandle handle to CUDPPSortPlan
 * @param[out] d_keys keys by which key-value pairs will be sorted
 * @param[in] d_values values to be sorted
 * @param[in] numElements number of elements in d_keys and d_values
 * @returns CUDPPResult indicating success or error condition
 *
 * @see cudppPlan, CUDPPConfiguration, CUDPPAlgorithm
 */
CUDPP_DLL
CUDPPResult cudppRadixSort(const CUDPPHandle planHandle,
                      void              *d_keys,
                      void              *d_values,
                      size_t            numElements)
{



    CUDPPRadixSortPlan *plan =
        (CUDPPRadixSortPlan*)getPlanPtrFromHandle<CUDPPRadixSortPlan>(planHandle);

    if (plan != NULL)
    {
        if (plan->m_config.algorithm != CUDPP_SORT_RADIX)
            return CUDPP_ERROR_INVALID_PLAN;

        if(plan->m_config.algorithm == CUDPP_SORT_RADIX)
            cudppRadixSortDispatch(d_keys, d_values, numElements, plan);

        return CUDPP_SUCCESS;
    }
    else
        return CUDPP_ERROR_INVALID_HANDLE;
}
/**
 * @brief Sorts key-value pairs or keys only
 *
 * Takes as input an array of keys in GPU memory
 * (d_keys) and an optional array of corresponding values,
 * and outputs sorted arrays of keys and (optionally) values in place.
 * Radix sort or Merge sort is selected through the configuration (.algorithm)
 * Key-value and key-only sort is selected through the configuration of
 * the plan, using the options CUDPP_OPTION_KEYS_ONLY and
 * CUDPP_OPTION_KEY_VALUE_PAIRS.
 *
 * Supported key types are CUDPP_FLOAT and CUDPP_UINT.  Values can be
 * any 32-bit type (internally, values are treated only as a payload
 * and cast to unsigned int).
 *
 * @todo Determine if we need to provide an "out of place" sort interface.
 *
 * @param[in] planHandle handle to CUDPPSortPlan
 * @param[out] d_keys keys by which key-value pairs will be sorted
 * @param[in] d_values values to be sorted
 * @param[in] numElements number of elements in d_keys and d_values
 * @returns CUDPPResult indicating success or error condition
 *
 * @see cudppPlan, CUDPPConfiguration, CUDPPAlgorithm
 */
CUDPP_DLL
CUDPPResult cudppMergeSort(const CUDPPHandle planHandle,
                      void              *d_keys,
                      void              *d_values,
                      size_t            numElements)
{
    CUDPPMergeSortPlan *plan =
        (CUDPPMergeSortPlan*)getPlanPtrFromHandle<CUDPPMergeSortPlan>(planHandle);

    if (plan != NULL)
    {
        if ((plan->m_config.algorithm != CUDPP_SORT_MERGE) ||
            ((plan->m_config.datatype != CUDPP_INT) &&
             (plan->m_config.datatype != CUDPP_UINT) &&
             (plan->m_config.datatype != CUDPP_FLOAT)))
        {
            return CUDPP_ERROR_INVALID_PLAN;
        }
        else
        {
            cudppMergeSortDispatch(d_keys, d_values, numElements, plan);
            return CUDPP_SUCCESS;
        }
    }
    else
    {
        return CUDPP_ERROR_INVALID_HANDLE;
    }
}

/**
 * @brief Sorts strings. Keys are the first four characters of the string,
 * and values are the addresses where the strings reside in memory (stringVals)
 *
 * Takes as input an array of strings (broken up as first four chars
 * (key), addresses (values), and the strings themselves (stringVals)
 * aligned by 4 character and packed into a uint)
 *
 *
 * @todo Determine if we need to provide an "out of place" sort interface.
 *
 * @param[in] planHandle handle to CUDPPSortPlan
 * @param[in,out] d_keys keys (first four chars of string to be sorted)
 * @param[in,out] d_values addresses where the strings reside
 * @param[in] stringVals Packed String input, series of characters each terminated by a null
 * @param[in] numElements number of elements in d_keys and d_values
 * @param[in] stringArrayLength Length in uint of the size of stromgVals
 * @returns CUDPPResult indicating success or error condition
 *
 * @see cudppPlan, CUDPPConfiguration, CUDPPAlgorithm
 */
CUDPP_DLL
CUDPPResult cudppStringSortAligned(const CUDPPHandle planHandle,
                      unsigned int              *d_keys,
                      unsigned int              *d_values,
                      unsigned int              *stringVals,
                      size_t            numElements,
                      size_t            stringArrayLength)
{
    CUDPPStringSortPlan *plan =
        (CUDPPStringSortPlan*)getPlanPtrFromHandle<CUDPPStringSortPlan>(planHandle);

    if (plan != NULL)
    {
        if (plan->m_config.algorithm != CUDPP_SORT_STRING)
            return CUDPP_ERROR_INVALID_PLAN;
                cudppStringSortDispatch(d_keys, d_values, stringVals, numElements, stringArrayLength, 0, plan);
            return CUDPP_SUCCESS;
    }
    else
        return CUDPP_ERROR_INVALID_HANDLE;
}

/**
 * @brief Sorts strings. Keys are the first four characters of the string,
 * and values are the addresses where the strings reside in memory (stringVals)
 *
 * Takes as input an array of strings arranged as a char* array with
 * NULL terminating characters. This function will reformat this info
 * into keys (first four chars) values(pointers to string array
 * addresses) and aligned string value array.
 *
 *
 *
 * @param[in] planHandle handle to CUDPPSortPlan
 * @param[in] d_stringVals Original string input, no need for alignment or offsets.
 * @param[in] d_address Pointers (in order) to each strings starting location in the stringVals array
 * @param[in] termC Termination character used to separate strings
 * @param[in] numElements number of strings
 * @param[in] stringArrayLength Length in uint of the size of all strings
 * @returns CUDPPResult indicating success or error condition
 *
 * @see cudppPlan, CUDPPConfiguration, CUDPPAlgorithm
 */
CUDPP_DLL
CUDPPResult cudppStringSort(const CUDPPHandle planHandle,
                      unsigned char     *d_stringVals,
                                          unsigned int      *d_address,
                                          unsigned char     termC,
                      size_t            numElements,
                      size_t            stringArrayLength)
{
    CUDPPStringSortPlan *plan =
        (CUDPPStringSortPlan*)getPlanPtrFromHandle<CUDPPStringSortPlan>(planHandle);


    if (plan != NULL)
    {
        if (plan->m_config.algorithm != CUDPP_SORT_STRING)
            return CUDPP_ERROR_INVALID_PLAN;

        unsigned int* packedStringVals;
        unsigned int *packedStringLength = (unsigned int*)malloc(sizeof(unsigned int));;

        calculateAlignedOffsets(d_address, plan->m_numSpaces, d_stringVals, termC, numElements, stringArrayLength);
        cudppScanDispatch(plan->m_spaceScan, plan->m_numSpaces, numElements+1, 1, plan->m_scanPlan);
        dotAdd(d_address, plan->m_spaceScan, plan->m_packedAddress, numElements+1, stringArrayLength);

        cudaMemcpy(packedStringLength, (plan->m_packedAddress)+numElements, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(plan->m_packedAddressRef, plan->m_packedAddress, sizeof(unsigned int)*numElements, cudaMemcpyDeviceToDevice);
        cudaMemcpy(plan->m_addressRef, d_address, sizeof(unsigned int)*numElements, cudaMemcpyDeviceToDevice);

        //system("PAUSE");
        cudaMalloc((void**)&packedStringVals, sizeof(unsigned int)*packedStringLength[0]);

        packStrings(packedStringVals, d_stringVals, plan->m_keys, plan->m_packedAddress, d_address, numElements, stringArrayLength, termC);

        cudppStringSortDispatch(plan->m_keys, plan->m_packedAddress, packedStringVals, numElements, packedStringLength[0], termC, plan);
        unpackStrings(plan->m_packedAddress, plan->m_packedAddressRef, d_address, plan->m_addressRef, numElements);


        free(packedStringLength);
        cudaFree(packedStringVals);
        return CUDPP_SUCCESS;
    }
    else
        return CUDPP_ERROR_INVALID_HANDLE;
}

/** @brief Perform matrix-vector multiply y = A*x for arbitrary sparse matrix A and vector x
  *
  * Given a matrix object handle (which has been initialized using
  * cudppSparseMatrix()), This function multiplies the input vector \a
  * d_x by the matrix referred to by \a sparseMatrixHandle, returning
  * the result in \a d_y.
  *
  * @param sparseMatrixHandle Handle to a sparse matrix object created with cudppSparseMatrix()
  * @param d_y The output vector, y
  * @param d_x The input vector, x
  * @returns CUDPPResult indicating success or error condition
  *
  * @see cudppSparseMatrix, cudppDestroySparseMatrix
  */
CUDPP_DLL
CUDPPResult cudppSparseMatrixVectorMultiply(const CUDPPHandle  sparseMatrixHandle,
                                            void               *d_y,
                                            const void         *d_x)
{
    CUDPPSparseMatrixVectorMultiplyPlan *plan =
        (CUDPPSparseMatrixVectorMultiplyPlan*)
        getPlanPtrFromHandle<CUDPPSparseMatrixVectorMultiplyPlan>(sparseMatrixHandle);

    if (plan != NULL)
    {
        if (plan->m_config.algorithm != CUDPP_SPMVMULT)
            return CUDPP_ERROR_INVALID_PLAN;

        cudppSparseMatrixVectorMultiplyDispatch(d_y, d_x, plan);
        return CUDPP_SUCCESS;
    }
    else
        return CUDPP_ERROR_INVALID_HANDLE;
}

/**
 * @brief Rand puts \a numElements random 32-bit elements into \a d_out
 *

 * Outputs \a numElements random values to \a d_out. \a d_out must be of
 * type unsigned int, allocated in device memory.
 *
 * The algorithm used for the random number generation is stored in \a planHandle.
 * Depending on the specification of the pseudo random number generator(PRNG),
 * the generator may have one or more seeds.  To set the seed, use cudppRandSeed().
 *
 * @todo Currently only MD5 PRNG is supported.  We may provide more rand routines in
 * the future.
 *
 * @param[in] planHandle Handle to plan for rand
 * @param[in] numElements number of elements in d_out.
 * @param[out] d_out output of rand, in GPU memory.  Should be an array of unsigned integers.
 * @returns CUDPPResult indicating success or error condition
 *
 * @see cudppPlan, CUDPPConfiguration, CUDPPAlgorithm
 */
CUDPP_DLL
CUDPPResult cudppRand(const CUDPPHandle planHandle,
                      void *            d_out,
                      size_t            numElements)
{
    CUDPPRandPlan * plan =
        (CUDPPRandPlan *) getPlanPtrFromHandle<CUDPPRandPlan>(planHandle);

    if(plan != NULL)
    {
        if (plan->m_config.algorithm != CUDPP_RAND_MD5)
            return CUDPP_ERROR_INVALID_PLAN;

        //dispatch the rand algorithm here
        cudppRandDispatch(d_out, numElements, plan);
        return CUDPP_SUCCESS;
    }
    else
        return CUDPP_ERROR_INVALID_HANDLE;
}


/**@brief Sets the seed used for rand
 *
 * The seed is crucial to any random number generator as it allows a
 * sequence of random numbers to be replicated.  Since there may be
 * multiple different rand algorithms in CUDPP, cudppRandSeed
 * uses \a planHandle to determine which seed to set.  Each rand
 * algorithm has its own  unique set of seeds depending on what
 * the algorithm needs.
 *
 * @param[in] planHandle the handle to the plan which specifies which rand seed to set
 * @param[in] seed the value which the internal cudpp seed will be set to
 * @returns CUDPPResult indicating success or error condition
 */
CUDPP_DLL
CUDPPResult cudppRandSeed(const CUDPPHandle planHandle,
                          unsigned int      seed)
{
    CUDPPRandPlan * plan =
        (CUDPPRandPlan *) getPlanPtrFromHandle<CUDPPRandPlan>(planHandle);

    if (plan != NULL)
    {
        if (plan->m_config.algorithm != CUDPP_RAND_MD5)
            return CUDPP_ERROR_INVALID_PLAN;
        plan->m_seed = seed;
    }
    else
        return CUDPP_ERROR_INVALID_HANDLE;

    return CUDPP_SUCCESS;
}//end cudppRandSeed

/**
 * @brief Solves tridiagonal linear systems
 *
 * The solver uses a hybrid CR-PCR algorithm described in our papers "Fast
 * Fast Tridiagonal Solvers on the GPU" and "A Hybrid Method for Solving
 * Tridiagonal Systems on the GPU". (See the \ref references bibliography).
 * Please refer to the papers for a complete description of the basic CR
 * (Cyclic Reduction) and PCR (Parallel Cyclic Reduction) algorithms and their
 * hybrid variants.
 *
 * - Both float and double data types are supported.
 * - Both power-of-two and non-power-of-two system sizes are supported.
 * - The maximum system size could be limited by the maximum number of threads
 * of a CUDA block, the number of registers per multiprocessor, and the
 * amount of shared memory available. For example, on the GTX 280 GPU, the
 * maximum system size is 512 for the float datatype, and 256 for the double
 * datatype, which is limited by the size of shared memory in this case.
 * - The maximum number of systems is 65535, that is the maximum number of
 * one-dimensional blocks that could be launched in a kernel call. Users could
 * launch the kernel multiple times to solve more systems if required.
 *
 * @param[out] d_x Solution vector
 * @param[in] planHandle Handle to plan for tridiagonal solver
 * @param[in] d_a Lower diagonal
 * @param[in] d_b Main diagonal
 * @param[in] d_c Upper diagonal
 * @param[in] d_d Right hand side
 * @param[in] systemSize The size of the linear system
 * @param[in] numSystems The number of systems to be solved
 * @returns CUDPPResult indicating success or error condition
 *
 * @see cudppPlan, CUDPPConfiguration, CUDPPAlgorithm
 */
CUDPP_DLL
CUDPPResult cudppTridiagonal(CUDPPHandle planHandle,
                             void *d_a,
                             void *d_b,
                             void *d_c,
                             void *d_d,
                             void *d_x,
                             int systemSize,
                             int numSystems)
{
    CUDPPTridiagonalPlan * plan =
        (CUDPPTridiagonalPlan *) getPlanPtrFromHandle<CUDPPTridiagonalPlan>(planHandle);

    if(plan != NULL)
    {
        //dispatch the tridiagonal solver here
        return cudppTridiagonalDispatch(d_a, d_b, d_c, d_d, d_x,
                                        systemSize, numSystems, plan);
    }
    else
        return CUDPP_ERROR_INVALID_HANDLE;
}

/**
 * @brief Compresses data stream
 *
 * Performs compression using a three stage pipeline consisting of the
 * Burrows-Wheeler transform, the move-to-front transform, and Huffman
 * encoding. The compression algorithms are described in our paper
 * "Parallel Lossless Data Compression on the GPU". (See the \ref
 * references bibliography).
 *
 * - Only unsigned char type is supported.
 * - Currently, the input stream (d_uncompressed) must be a buffer of 1,048,576 (uchar) elements (~1MB).
 * - The BWT Index (d_bwtIndex) is an integer number (int). This is used during the reverse-BWT stage.
 * - The Histogram size pointer (d_histSize) can be ignored and can be passed a null pointer.
 * - The Histrogram (d_hist) is a 256-entry (unsigned int) buffer. The histogram is used to
 * construct the Huffman tree during decoding.
 * - The Encoded offset table (d_encodeOffset) is a 256-entry (unsigned int) buffer. Since the input
 * stream is compressed in blocks of 4096 characters, the offset table gives the starting offset of
 * where each block starts in the compressed data (d_compressedSize). The very first uint at each starting offset
 * gives the size (in words) of that corresponding compressed block. This allows us to decompress each 4096
 * character-block in parallel.
 * - The size of compressed data (d_compressedSize) is a uint and gives the final size (in words)
 * of the compressed data.
 * - The compress data stream (d_compressed) is a uint buffer. The user should allocate enough
 * memory for worst-case (no compression occurs).
 * - \a numElements is a uint and must be set to 1048576.
 *
 * @param[out] d_bwtIndex BWT Index (int)
 * @param[out] d_histSize Histogram size (ignored, null ptr)
 * @param[out] d_hist Histogram (256-entry, uint)
 * @param[out] d_encodeOffset Encoded offset table (256-entry, uint)
 * @param[out] d_compressedSize Size of compressed data (uint)
 * @param[out] d_compressed Compressed data
 * @param[in] planHandle Handle to plan for compressor
 * @param[in] d_uncompressed Uncompressed data
 * @param[in] numElements Number of elements to compress
 * @returns CUDPPResult indicating success or error condition
 *
 * @see cudppPlan, CUDPPConfiguration, CUDPPAlgorithm
 */
CUDPP_DLL
CUDPPResult cudppCompress(CUDPPHandle planHandle,
                          unsigned char *d_uncompressed,
                          int *d_bwtIndex,
                          unsigned int *d_histSize,
                          unsigned int *d_hist,
                          unsigned int *d_encodeOffset,
                          unsigned int *d_compressedSize,
                          unsigned int *d_compressed,
                          size_t numElements)
{
    // first check: is this device >= 2.0? if not, return error

    int dev;
    cudaGetDevice(&dev);

    cudaDeviceProp devProps;
    cudaGetDeviceProperties(&devProps, dev);

    if((int)devProps.major < 2) {
        // Only supported on devices with compute
        // capability 2.0 or greater
        return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
    }

    CUDPPCompressPlan * plan =
        (CUDPPCompressPlan *) getPlanPtrFromHandle<CUDPPCompressPlan>(planHandle);

    if(plan != NULL)
    {
        if (plan->m_config.algorithm != CUDPP_COMPRESS)
            return CUDPP_ERROR_INVALID_PLAN;
        if (plan->m_config.datatype != CUDPP_UCHAR)
            return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
        //if (numElements != 1048576)
        //    return CUDPP_ERROR_ILLEGAL_CONFIGURATION;

        cudppCompressDispatch(d_uncompressed, d_bwtIndex, d_histSize, d_hist, d_encodeOffset,
            d_compressedSize, d_compressed, numElements, plan);
        return CUDPP_SUCCESS;
    }
    else
        return CUDPP_ERROR_INVALID_HANDLE;
}

/**
 * @brief Performs the Burrows-Wheeler Transform
 *
 * Performs a parallel Burrows-Wheeler transform on 1,048,576 elements.
 * The BWT leverages a string-sort algorithm based on merge-sort.
 *
 * - Currently, the BWT can only be performed on 1,048,576 (uchar) elements.
 * - The transformed string is written to \a d_x.
 * - The BWT index (used during the reverse-BWT) is recorded as an int
 * in \a d_index.
 *
 * @param[in] planHandle Handle to plan for BWT
 * @param[out] d_in BWT Index
 * @param[out] d_out Output data
 * @param[in] d_index Input data
 * @param[in] numElements Number of elements
 * @returns CUDPPResult indicating success or error condition
 *
 * @see cudppPlan, CUDPPConfiguration, CUDPPAlgorithm
 */
CUDPP_DLL
CUDPPResult cudppBurrowsWheelerTransform(CUDPPHandle planHandle,
                                         unsigned char *d_in,
                                         unsigned char *d_out,
                                         int *d_index,
                                         size_t numElements)
{
    // first check: is this device >= 2.0? if not, return error
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp devProps;
    cudaGetDeviceProperties(&devProps, dev);

    if((int)devProps.major < 2) {
        // Only supported on devices with compute
        // capability 2.0 or greater
        return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
    }

    CUDPPBwtPlan * plan =
        (CUDPPBwtPlan *) getPlanPtrFromHandle<CUDPPBwtPlan>(planHandle);

    if(plan != NULL)
    {
        if (plan->m_config.algorithm != CUDPP_BWT)
            return CUDPP_ERROR_INVALID_PLAN;
        if (plan->m_config.datatype != CUDPP_UCHAR)
            return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
        //if (numElements != 1048576)
        //    return CUDPP_ERROR_ILLEGAL_CONFIGURATION;

        cudppBwtDispatch(d_in, d_out, d_index, numElements, plan);
        return CUDPP_SUCCESS;
    }
    else
        return CUDPP_ERROR_INVALID_HANDLE;
}

/**
 * @brief Performs the Move-to-Front Transform
 *
 * Performs a parallel move-to-front transform on 1,048,576 elements.
 * The MTF uses a scan-based algorithm to parallelize the computation.
 * The MTF uses a scan-based algorithm described in our paper "Parallel
 * Lossless Data Compression on the GPU". (See the \ref references bibliography).
 *
 * - Currently, the MTF can only be performed on 1,048,576 (uchar) elements.
 * - The transformed string is written to \a d_mtfOut.
 *
 * @param[in] planHandle Handle to plan for MTF
 * @param[out] d_out Output data
 * @param[in] d_in Input data
 * @param[in] numElements Number of elements
 * @returns CUDPPResult indicating success or error condition
 *
 * @see cudppPlan, CUDPPConfiguration, CUDPPAlgorithm
 */
CUDPP_DLL
CUDPPResult cudppMoveToFrontTransform(CUDPPHandle planHandle,
                                      unsigned char *d_in,
                                      unsigned char *d_out,
                                      size_t numElements)
{
    // first check: is this device >= 2.0? if not, return error
    int dev;
    cudaGetDevice(&dev);

    cudaDeviceProp devProps;
    cudaGetDeviceProperties(&devProps, dev);

    if((int)devProps.major < 2) {
        // Only supported on devices with compute
        // capability 2.0 or greater
        return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
    }

    CUDPPMtfPlan * plan =
        (CUDPPMtfPlan *) getPlanPtrFromHandle<CUDPPMtfPlan>(planHandle);

    if(plan != NULL)
    {
        if (plan->m_config.algorithm != CUDPP_MTF)
            return CUDPP_ERROR_INVALID_PLAN;
        if (plan->m_config.datatype != CUDPP_UCHAR)
            return CUDPP_ERROR_ILLEGAL_CONFIGURATION;

        cudppMtfDispatch(d_in, d_out, numElements, plan);
        return CUDPP_SUCCESS;
    }
    else
        return CUDPP_ERROR_INVALID_HANDLE;
}

/**
 * @brief Performs list ranking of linked list node values
 *
 * Performs parallel list ranking on values of a linked-list
 * using a pointer-jumping algorithm.
 *
 * Takes as input an array of values in GPU memory
 * (\a d_unranked_values) and an equal-sized int array in GPU memory
 * (\a d_next_indices) that represents the next indices of the linked
 * list. The index of the head node (\a head) is given as an
 * unsigned int. The output (\a d_ranked_values) is an equal-sized array,
 * in GPU memory, that has the values ranked in-order.
 *
 * Example:
 * \code
 * d_a     = [  f a c d b e  ]
 * d_b     = [ -1 4 3 5 2 0  ]
 * head    = 1
 * d_x     = [ a b c d e f ]
 * \endcode
 *
 *
 * @param[in] planHandle Handle to plan for list ranking
 * @param[out] d_ranked_values Output ranked values
 * @param[in] d_unranked_values Input unranked values
 * @param[in] d_next_indices Input next indices
 * @param[in] head Input head node index
 * @param[in] numElements number of nodes
 * @returns CUDPPResult indicating success or error condition
 *
 * @see cudppPlan, CUDPPConfiguration, CUDPPAlgorithm
 */
CUDPP_DLL
CUDPPResult cudppListRank(CUDPPHandle planHandle,
                          void *d_ranked_values,
                          void *d_unranked_values,
                          void *d_next_indices,
                          size_t head,
                          size_t numElements)
{
    CUDPPListRankPlan * plan =
        (CUDPPListRankPlan *) getPlanPtrFromHandle<CUDPPListRankPlan>(planHandle);

    if(plan != NULL)
    {
        if (plan->m_config.algorithm != CUDPP_LISTRANK)
            return CUDPP_ERROR_INVALID_PLAN;

        return cudppListRankDispatch(d_ranked_values, d_unranked_values,
                                     d_next_indices, head, numElements, plan);
    }
    else
        return CUDPP_ERROR_INVALID_HANDLE;
}

/**
 * @brief Performs the Suffix Array
 *
 * Performs a parallel suffix array using linear-time recursive skew algorithm.
 * The SA leverages a suffix-sort algorithm based on divide and conquer.
 *
 * - The SA is GPU memory bounded, it needs about seven times size of input data.
 * - Only unsigned char type is supported.
 *
 * - The input char array is transformed into an unsigned int array storing the
 * key values followed by three 0s for the convinience of building triplets.
 * - The output data is an unsigned int array storing the positions of the
 * lexicographically sorted suffixes not including the last {0,0,0} triplet.
 *
 * @param[in] planHandle Handle to plan for CUDPPSuffixArrayPlan
 * @param[out] d_in  Input data
 * @param[out] d_out Output data
 * @param[in] numElements Number of elements
 * @returns CUDPPResult indicating success or error condition
 *
 * @see cudppPlan, CUDPPConfiguration, CUDPPAlgorithm
 */

CUDPP_DLL
CUDPPResult cudppSuffixArray(CUDPPHandle planHandle,
                             unsigned char *d_in,
                             unsigned int *d_out,
                             size_t numElements)
{

    // first check: is this device >= 2.0? if not, return error
    int dev;
    cudaGetDevice(&dev);

    cudaDeviceProp devProps;
    cudaGetDeviceProperties(&devProps, dev);

    if((int)devProps.major < 2) {
        // Only supported on devices with compute
        // capability 2.0 or greater
        return CUDPP_ERROR_ILLEGAL_CONFIGURATION;
    }

    CUDPPSaPlan * plan =
         (CUDPPSaPlan *) getPlanPtrFromHandle<CUDPPSaPlan>(planHandle);
    if(plan != NULL)
    {
        if (plan->m_config.algorithm != CUDPP_SA)
            return CUDPP_ERROR_INVALID_PLAN;
        if (plan->m_config.datatype != CUDPP_UCHAR)
            return CUDPP_ERROR_ILLEGAL_CONFIGURATION;

        cudppSuffixArrayDispatch(d_in, d_out, numElements, plan);
        return CUDPP_SUCCESS;
    }
    else
        return CUDPP_ERROR_INVALID_HANDLE;

}

/**
 * @brief Splits an array of keys and an optional 
 * array of values into a set of buckets.
 *
 * Takes as input an array of keys in GPU memory
 * (d_keys) and an optional array of corresponding values,
 * and outputs an arrays of keys and (optionally) values in place,
 * where the keys and values have been split into ordered buckets.
 * Key-value or key-only multisplit is selected through the configuration of
 * the plan, using the options CUDPP_OPTION_KEYS_ONLY or
 * CUDPP_OPTION_KEY_VALUE_PAIRS. The function used to map a key to a bucket
 * is selected through the configuration option 'bucket_mapper'. 
 * The current options are:
 *
 * ORDERED_CYCLIC_BUCKET_MAPPER (default):
 * bucket = (key % numElements) / ((numElements + numBuckets - 1) / numBuckets);
 *
 * MSB_BUCKET_MAPPER:
 * bucket = (key >> (32 - ceil(log2(numBuckets)))) % numBuckets;
 *
 * Currently, the only supported key and value type is CUDPP_UINT.
 *
 *
 * @param[in] planHandle Handle to plan for CUDPPMultiSplitPlan
 * @param[in,out] d_keys keys by which key-value pairs will be split
 * @param[in,out] d_values values to be split
 * @param[in] numElements number of elements in d_keys and d_values
 * @param[in] numBuckets Number of buckets
 * @returns CUDPPResult indicating success or error condition
 *
 * @see cudppPlan, CUDPPConfiguration, CUDPPAlgorithm
 */
CUDPP_DLL
CUDPPResult cudppMultiSplit(const CUDPPHandle planHandle,
                            unsigned int      *d_keys,
                            unsigned int      *d_values,
                            size_t            numElements,
                            size_t            numBuckets)
{
    CUDPPMultiSplitPlan *plan =
        (CUDPPMultiSplitPlan*)getPlanPtrFromHandle<CUDPPMultiSplitPlan>(planHandle);

    if (plan != NULL)
    {
      cudppMultiSplitDispatch(d_keys, d_values, numElements, numBuckets, NULL, plan);
          return CUDPP_SUCCESS;
    }
    else
    {
        return CUDPP_ERROR_INVALID_HANDLE;
    }
}

/**
 * @brief Splits an array of keys and an optional array of values into 
 * a set of buckets using a custom function to map elements to buckets.
 *
 * Takes as input an array of keys in GPU memory
 * (d_keys) and an optional array of corresponding values,
 * and outputs an arrays of keys and (optionally) values in place,
 * where the keys and values have been split into ordered buckets.
 * Key-value or key-only multisplit is selected through the configuration of
 * the plan, using the options CUDPP_OPTION_KEYS_ONLY or
 * CUDPP_OPTION_KEY_VALUE_PAIRS. To use this function, the 
 * configuration option 'bucket_mapper' must be set to CUSTOM_BUCKET_MAPPER.
 * This option lets the library know to use the custom function pointer,
 * specified in the last argument, when assigning an element to a bucket.
 * The user specified bucket mapper must be a function pointer to a device
 * function that takes one unsigned int argument (the element) and returns 
 * an unsigned int (the bucket). 
 *
 *
 * Currently, the only supported key and value type is CUDPP_UINT.
 *
 * @param[in] planHandle Handle to plan for BWT
 * @param[in,out] d_keys  Input data
 * @param[in,out] d_values Output data
 * @param[in] numElements Number of elements
 * @param[in] numBuckets Number of buckets
 * @param[in] bucketMappingFunc function that maps an element to a bucket
 * @returns CUDPPResult indicating success or error condition
 *
 * @see cudppPlan, CUDPPConfiguration, CUDPPAlgorithm
 */
CUDPP_DLL
CUDPPResult cudppMultiSplitCustomBucketMapper(const CUDPPHandle planHandle,
                                              unsigned int      *d_keys,
                                              unsigned int      *d_values,
                                              size_t            numElements,
                                              size_t            numBuckets,
                                              BucketMappingFunc bucketMappingFunc)
{
    CUDPPMultiSplitPlan *plan =
        (CUDPPMultiSplitPlan*)getPlanPtrFromHandle<CUDPPMultiSplitPlan>(planHandle);

    if (plan != NULL)
    {
      cudppMultiSplitDispatch(d_keys, d_values, numElements, numBuckets,
        bucketMappingFunc, plan);
          return CUDPP_SUCCESS;
    }
    else
    {
        return CUDPP_ERROR_INVALID_HANDLE;
    }
}

/** @} */ // end Algorithm Interface
/** @} */ // end of publicInterface group

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

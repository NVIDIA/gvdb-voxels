// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// ------------------------------------------------------------- 
#include <cudpp.h>
#include <stdio.h>
#include <algorithm>
#include "cudpp_testrig_utils.h"

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set for sum-scan
//! Each element is the sum of the elements before it in the array.
//! @param reference  reference data, computed but preallocated
//! @param idata      const input data as provided to device
//! @param len        number of elements in reference / idata
//! @param config     Options for the scan
////////////////////////////////////////////////////////////////////////////////
template<typename T>
void
computeSumScanGold( T *reference, const T *idata, 
                    const unsigned int len,
                    const CUDPPConfiguration &config) 
{
    int startIdx = 1, stopIdx = len;
    int increment = 1;
    reference[0] = 0;    
    
    if (config.options & CUDPP_OPTION_BACKWARD)
    {
        reference[len-1] = 0;
        startIdx = len-2;
        stopIdx = -1;
        increment = -1;
    }
    
    double total_sum = 0;
    
    for( int i = startIdx; i != stopIdx; i = i + increment)
    {
        total_sum += idata[i-increment];
        reference[i] = idata[i-increment] + reference[i-increment];
    }

    if (config.options & CUDPP_OPTION_INCLUSIVE) 
    {
        total_sum += idata[stopIdx - increment];

        for (int i = startIdx - increment; i != stopIdx; i = i + increment) 
        {
            reference[i] += idata[i];
        }      
    }

    if (total_sum != reference[stopIdx - increment])
    {
        printf("Warning: exceeding single-precision accuracy. "
               "Scan will be inaccurate.\n");
        printf("total sum ");
        printItem(total_sum, "\n");
        printf("reference ");
        printItem(reference[stopIdx - increment], "\n");
    }   
}

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set for multiply-scan
//! Each element is the sum of the elements before it in the array.
//! @param reference  reference data, computed but preallocated
//! @param idata      const input data as provided to device
//! @param len        number of elements in reference / idata
//! @param config     Options for the scan
////////////////////////////////////////////////////////////////////////////////

template<typename T>
void
computeMultiplyScanGold( T *reference, const T *idata, 
                         const unsigned int len,
                         const CUDPPConfiguration &config) 
{
    int startIdx = 1, stopIdx = len;
    int increment = 1;
    reference[0] = 1;    

    if (config.options & CUDPP_OPTION_BACKWARD)
    {
        reference[len-1] = 1;
        startIdx = len-2;
        stopIdx = -1;
        increment = -1;
    }

    double totalProduct = 1;

    for( int i = startIdx; i != stopIdx; i = i + increment)
    {
        totalProduct *= idata[i-increment];
        reference[i] = idata[i-increment] * reference[i-increment];
    }

    if (config.options & CUDPP_OPTION_INCLUSIVE) 
    {
        totalProduct *= idata[stopIdx - increment];

        for (int i = startIdx - increment; i != stopIdx; i = i + increment) 
        {
            reference[i] *= idata[i];
        }      
    }

    if (totalProduct != reference[stopIdx - increment])
    {
        printf("Warning: exceeding single-precision accuracy. "
               "Scan will be inaccurate.\n");
        printf("total product ");
        printItem(totalProduct, "\n");
        printf("reference ");
        printItem(reference[stopIdx - increment], "\n");
    }   
}


template<typename T>
void
computeSegmentedSumScanGold(T* reference, const T* idata, 
                            const unsigned int *iflag,
                            const unsigned int len,
                            const CUDPPConfiguration & config) 
{
    int startIdx=1, stopIdx=len;
    int increment=1;
    reference[0] = 0;
    
    if (config.options & CUDPP_OPTION_BACKWARD)
    {
        reference[len-1] = 0;
        startIdx = len-2;
        stopIdx = -1;
        increment = -1;    
    }
 
    double total_sum = 0;

    for( int i = startIdx; i != stopIdx; i = i + increment)
    {
        if (config.options & CUDPP_OPTION_FORWARD)
        {
            total_sum = 
                (iflag[i] == 1) ? 0 : total_sum + idata[i-increment];

            reference[i] = 
                (iflag[i] == 1) ? 
                    0 : idata[i-increment] + reference[i-increment];
        }
        else if (config.options & CUDPP_OPTION_BACKWARD)
        {        
            total_sum = 
                (iflag[i-increment] == 1) ? 0 : total_sum + idata[i-increment];

            reference[i] = 
                (iflag[i-increment] == 1) ? 
                    0 : idata[i-increment] + reference[i-increment];
        }
    }

    if (config.options & CUDPP_OPTION_INCLUSIVE) 
    {
        total_sum += idata[stopIdx - increment];

        for (int i = startIdx - increment; i != stopIdx; i = i + increment) 
        {
            reference[i] += idata[i];
        }
    }

    if (total_sum != reference[stopIdx - increment])
    {
        printf("Warning: exceeding single-precision accuracy. "
               "Scan will be inaccurate.\n");
        printf("total sum ");
        printItem(total_sum, "\n");
        printf("reference ");
        printItem(reference[stopIdx - increment], "\n");
    }   
}

template<typename T>
void
computeSegmentedMaxScanGold(T* reference, const T* idata, 
                            const unsigned int *iflag,
                            const unsigned int len,
                            const CUDPPConfiguration & config) 
{
    int startIdx=1, stopIdx=len;
    int increment=1;
    
    OperatorMax<T> mx;
    reference[0] = mx.identity();
    
    if (config.options & CUDPP_OPTION_BACKWARD)
    {
        reference[len-1] = mx.identity();
        startIdx = len-2;
        stopIdx = -1;
        increment = -1;
    }
    
    T total_max = mx.identity();

    for( int i = startIdx; i != stopIdx; i = i + increment)
    {
        if (config.options & CUDPP_OPTION_FORWARD)
        {
            total_max = 
                (iflag[i] == 1) ? mx.identity() : std::max(total_max, idata[i-increment]);

            reference[i] = 
                (iflag[i] == 1) ? 
                mx.identity() : std::max(idata[i-increment], reference[i-increment]);
        }
        else if (config.options & CUDPP_OPTION_BACKWARD)
        {
            total_max = 
                (iflag[i-increment] == 1) ? mx.identity() : std::max(total_max, idata[i-increment]);

            reference[i] = 
                (iflag[i-increment] == 1) ? 
                mx.identity() : std::max(idata[i-increment], reference[i-increment]);
        }

    }

    if (config.options & CUDPP_OPTION_INCLUSIVE) 
    {
        total_max = std::max(total_max, idata[stopIdx - increment]);

        for (int i = startIdx - increment; i != stopIdx; i = i + increment) 
        {
            reference[i] = std::max(reference[i], idata[i]);
        }
    }

    if (total_max != reference[stopIdx - increment])
    {
        printf("Warning: exceeding single-precision accuracy. "
               "Scan will be inaccurate.\n");
        printf("total max ");
        printItem(total_max, "\n");
        printf("reference ");
        printItem(reference[stopIdx - increment], "\n");
    }   
}

template<typename T>
void
computeSegmentedMultiplyScanGold(T* reference, const T* idata, 
                                 const unsigned int *iflag,
                                 const unsigned int len,
                                 const CUDPPConfiguration & config) 
{
    int startIdx=1, stopIdx=len;
    int increment=1;
    reference[0] = T(1.0);
    
    if (config.options & CUDPP_OPTION_BACKWARD)
    {
        reference[len-1] = T(1.0);
        startIdx = len-2;
        stopIdx = -1;
        increment = -1;
    }

    double total_multi = 1.0;

    for( int i = startIdx; i != stopIdx; i = i + increment)
    {
        if (config.options & CUDPP_OPTION_FORWARD)
        {
            total_multi = 
                (iflag[i] == 1) ? 1.0 : total_multi * idata[i-increment];

            reference[i] = 
                (iflag[i] == 1) ? 
                T(1.0) : idata[i-increment] * reference[i-increment];
        }
        else
        {
            total_multi = 
                (iflag[i-increment] == 1) ? 1.0 : total_multi * idata[i-increment];

            reference[i] = 
                (iflag[i-increment] == 1) ? 
                T(1.0) : idata[i-increment] * reference[i-increment];
        }
    }

    if (config.options & CUDPP_OPTION_INCLUSIVE) 
    {
        total_multi = total_multi * idata[stopIdx - increment];

        for (int i = startIdx - increment; i != stopIdx; i = i + increment) 
        {
            reference[i] = reference[i] * idata[i];
        }
    }

    if (total_multi != reference[stopIdx - increment])
    {
        printf("Warning: exceeding single-precision accuracy. "
               "Scan will be inaccurate.\n");
        printf("total multi ");
        printItem(total_multi, "\n");
        printf("reference ");
        printItem(reference[stopIdx - increment], "\n");
    }   
}

template<typename T>
void
computeSegmentedMinScanGold(T* reference, const T* idata, 
                            const unsigned int *iflag,
                            const unsigned int len,
                            const CUDPPConfiguration & config) 
{
    OperatorMin<T> mn;
    int startIdx=1, stopIdx=len;
    int increment=1;
    reference[0] = mn.identity();
    
    if (config.options & CUDPP_OPTION_BACKWARD)
    {
        reference[len-1] = mn.identity();
        startIdx = len-2;
        stopIdx = -1;
        increment = -1;
    }

    T total_min = mn.identity();

    for( int i = startIdx; i != stopIdx; i = i + increment)
    {
        if (config.options & CUDPP_OPTION_FORWARD)
        {
            total_min = 
                (iflag[i] == 1) ? mn.identity() : std::min(total_min, idata[i-increment]);

            reference[i] = 
                (iflag[i] == 1) ? 
                mn.identity() : std::min(idata[i-increment], reference[i-increment]);
        }
        else if (config.options & CUDPP_OPTION_BACKWARD)
        {
            total_min = 
                (iflag[i-increment] == 1) ? mn.identity() : std::min(total_min, idata[i-increment]);

            reference[i] = 
                (iflag[i-increment] == 1) ? 
                mn.identity() : std::min(idata[i-increment], reference[i-increment]);
        }
    }

    if (config.options & CUDPP_OPTION_INCLUSIVE) 
    {
        total_min = std::min(total_min, idata[stopIdx - increment]);

        for (int i = startIdx - increment; i != stopIdx; i = i + increment) 
        {
            reference[i] = std::min(reference[i], idata[i]);
        }
    }

    if (total_min != reference[stopIdx - increment])
    {
        printf("Warning: exceeding single-precision accuracy. "
               "Scan will be inaccurate.\n");
        printf("total min ");
        printItem(total_min, "\n");
        printf("reference ");
        printItem(reference[stopIdx - increment], "\n");
    }   
}

template <typename T>
struct LargestSimilarType { typedef long long type; };

template <> struct LargestSimilarType<float>  { typedef double type; };
template <> struct LargestSimilarType<double> { typedef double type; };

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set for exclusive sum-scan
//! Each element is the sum of the elements before it in the array.
//! @param reference  reference data, computed but preallocated
//! @param idata      const input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
template<typename T, class Operator>
void
computeMultiRowScanGold( T *reference, const T *idata, 
                         const unsigned int len,
                         const unsigned int rows,
                         const CUDPPConfiguration &config) 
{
    Operator op;
    
    int startIdx = 1, stopIdx = len;
    int increment = 1;
    
    if (config.options & CUDPP_OPTION_FORWARD)
    {
        for (unsigned int r = 0; r < rows; ++r)
        {
            reference[r*len] = op.identity();
        }
    }
    else if (config.options & CUDPP_OPTION_BACKWARD)
    {
        startIdx = len-2;
        stopIdx = -1;
        increment = -1;
        for (unsigned int r = 1; r <= rows; ++r)
        {
            reference[r*len-1] = op.identity();
        }
    }
    
    typedef typename LargestSimilarType<T>::type bigtype;
    bigtype *total_sum = new bigtype[rows];
    
    for (unsigned int r = 0; r < rows; ++r)
    {       
        total_sum[r] = (bigtype)op.identity();
        
        for( int i = startIdx; i != stopIdx; i = i + increment)    
        {   
            total_sum[r] = (bigtype)op((T)total_sum[r], idata[r*len+i-increment]);
            reference[r*len+i] = op(idata[r*len+i-increment], reference[r*len+i-increment]);
        }
        
        if (config.options & CUDPP_OPTION_INCLUSIVE) 
        {
            total_sum[r] = (bigtype)op((T)total_sum[r], idata[r*len+stopIdx-increment]);

            for (int i = startIdx - increment; i != stopIdx; i = i + increment) 
            {
                reference[r*len+i] = op(reference[r*len+i], idata[r*len+i]);
            }      
        }
        
        if (((T)total_sum[r]) != reference[r*len+stopIdx-increment])
        {
            printf("Warning: exceeding single-precision accuracy. "
                   "Scan will be inaccurate.\n");
            printf("total sum ");
            printItem(total_sum[r], "\n");
        }
    }

    delete [] total_sum;
}

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set for exclusive max-scan
//! Each element is the max of the elements before it in the array.
//! First element set to INT_MIN
//! @param reference  reference data, computed but preallocated
//! @param idata      const input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////

template<typename T>
void
computeMaxScanGold(T *reference, const T *idata,   
                   const unsigned int len, const CUDPPConfiguration & config) 
{
    OperatorMax<T> mx;
    if(config.options & CUDPP_OPTION_FORWARD)
    {
        int j = 0;
        if (config.options & CUDPP_OPTION_EXCLUSIVE)
        {
            reference[0] = mx.identity();
        }
        else
        {
            reference[0] = idata[0];
            j++;
        }
        for( unsigned int i = 1; i < len; ++i,++j) 
        {
            reference[i] = std::max(idata[j], reference[i-1]);
        }
    }
    else
    {
        int j = len - 1;
        if (config.options & CUDPP_OPTION_EXCLUSIVE)
        {   
            reference[len-1] = mx.identity();
        }
        else
        {
            reference[len-1] = idata[len-1];
            j--;
        }
        for( int i = len - 2; i >=0; i--,j--)
        {
            reference[i] = std::max(idata[j], reference[i+1]);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set for exclusive max-scan
//! Each element is the max of the elements before it in the array.
//! First element set to INT_MIN
//! @param reference  reference data, computed but preallocated
//! @param idata      const input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
template<typename T>
void
computeMinScanGold(T *reference, const T *idata,   
                   const unsigned int len, const CUDPPConfiguration & config) 
{
    OperatorMin<T> mn;
    if(config.options & CUDPP_OPTION_FORWARD)
    {
        int j = 0;
        if (config.options & CUDPP_OPTION_EXCLUSIVE)
        {
            reference[0] = mn.identity();
        }
        else
        {
            reference[0] = idata[0];
            j++;
        }
        for( unsigned int i = 1; i < len; ++i,++j) 
        {
            reference[i] = std::min(idata[j], reference[i-1]);
        }
    }
    else
    {
        int j = len - 1;
        if (config.options & CUDPP_OPTION_EXCLUSIVE)
        {   
            reference[len-1] = mn.identity();
        }
        else
        {
            reference[len-1] = idata[len-1];
            j--;            
        }
        for( int i = len - 2; i >=0; i--,j--)
        {
            reference[i] = std::min(idata[j], reference[i+1]);
        }
    }
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

#include <cudpp.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>
#include <algorithm>

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C" 
void computeSumScanGold( float* reference, const float* idata, 
                         const unsigned int len,
                         const CUDPPConfiguration &config);

extern "C" 
void computeMultiplyScanGold( float* reference, const float* idata, 
                              const unsigned int len,
                              const CUDPPConfiguration &config);

extern "C"
void
computeSumSegmentedScanGold(float* reference, const float* idata, 
                            const unsigned int *iflag,
                            const unsigned int len,
                            const CUDPPConfiguration & config); 

extern "C" 
void computeMultiRowSumScanGold( float* reference, const float* idata, 
                                 const unsigned int len,
                                 const unsigned int rows,
                                 const CUDPPConfiguration &config);

extern "C" 
void computeMaxScanGold( float *reference, const float *idata, 
                         const unsigned int len, const CUDPPConfiguration &config);

extern "C" 
void computeMinScanGold( float *reference, const float *idata, 
                        const unsigned int len, const CUDPPConfiguration &config);

extern "C"
void
computeMaxSegmentedScanGold(float* reference, const float* idata, 
                            const unsigned int *iflag,
                            const unsigned int len,
                            const CUDPPConfiguration & config); 

extern "C"
void
computeMultiplySegmentedScanGold(float* reference, const float* idata, 
                                 const unsigned int *iflag,
                                 const unsigned int len,
                                 const CUDPPConfiguration & config); 

extern "C"
void
computeMinSegmentedScanGold(float* reference, const float* idata, 
                            const unsigned int *iflag,
                            const unsigned int len,
                            const CUDPPConfiguration & config);

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set for sum-scan
//! Each element is the sum of the elements before it in the array.
//! @param reference  reference data, computed but preallocated
//! @param idata      const input data as provided to device
//! @param len        number of elements in reference / idata
//! @param config     Options for the scan
////////////////////////////////////////////////////////////////////////////////
void
computeSumScanGold( float *reference, const float *idata, 
                    const unsigned int len,
                    const CUDPPConfiguration &config) 
{
    if (config.options & CUDPP_OPTION_FORWARD)
    {
        reference[0] = 0;
    }
    else if (config.options & CUDPP_OPTION_BACKWARD)
    {
        reference[len-1] = 0;
    }

    int startIdx = 0, stopIdx = 0;
    int increment = 0;

    if (config.options & CUDPP_OPTION_FORWARD)
    {
        startIdx = 1;
        stopIdx = len ;
        increment = 1 ;
    }
    else if (config.options & CUDPP_OPTION_BACKWARD)
    {
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
        printf("Warning: exceeding single-precision accuracy.  Scan will be inaccurate.\n");
        printf("total sum %f\n", total_sum);
        printf("reference %f\n", reference[stopIdx - increment]);
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
void
computeMultiplyScanGold( float *reference, const float *idata, 
                         const unsigned int len,
                         const CUDPPConfiguration &config) 
{
    if (config.options & CUDPP_OPTION_FORWARD)
    {
        reference[0] = 1;
    }
    else if (config.options & CUDPP_OPTION_BACKWARD)
    {
        reference[len-1] = 1;
    }

    int startIdx = 0, stopIdx = 0;
    int increment = 0;

    if (config.options & CUDPP_OPTION_FORWARD)
    {
        startIdx = 1;
        stopIdx = len ;
        increment = 1 ;
    }
    else if (config.options & CUDPP_OPTION_BACKWARD)
    {
        startIdx = len-2;
        stopIdx = -1;
        increment = -1;
    }

    double totalProduct = 0;

    for( int i = startIdx; i != stopIdx; i = i + increment)
    {
        totalProduct *= idata[i-increment];
        reference[i] = idata[i-increment] * reference[i-increment];
    }

    if (config.options & CUDPP_OPTION_INCLUSIVE) 
    {
        totalProduct *= idata[stopIdx - increment];

        for (int i = startIdx - increment; i < stopIdx; i = i + increment) 
        {
            reference[i] *= idata[i];
        }      
    }

    if (totalProduct != reference[stopIdx - increment])
    {
        printf("Warning: exceeding single-precision accuracy.  Scan will be inaccurate.\n");
        printf("total product %f\n", totalProduct);
        printf("reference %f\n", reference[stopIdx - increment]);
    }   
}


void
computeSumSegmentedScanGold(float* reference, const float* idata, 
                            const unsigned int *iflag,
                            const unsigned int len,
                            const CUDPPConfiguration & config) 
{
    reference[0] = 0;
 
    int startIdx, stopIdx;
    int increment;

    startIdx = 1;
    stopIdx = len ;
    increment = 1 ;
    
    double total_sum = 0;

    for( int i = startIdx; i != stopIdx; i = i + increment)
    {
        total_sum = 
                (iflag[i] == 1) ? 0 : total_sum + idata[i-increment];

        reference[i] = 
                (iflag[i] == 1) ? 
                    0 : idata[i-increment] + reference[i-increment];
    }

    if (config.options & CUDPP_OPTION_INCLUSIVE) 
    {
        total_sum += idata[stopIdx - increment];

        for (unsigned int i = 0; i < len; i++) 
        {
            reference[i] += idata[i];
        }
    }

    if (total_sum != reference[stopIdx - increment])
    {
        printf("Warning: exceeding single-precision accuracy.  Scan will be inaccurate.\n");
        printf("total sum %f\n", total_sum);
        printf("reference %f\n", reference[stopIdx - increment]);
    }   
}


void
computeMaxSegmentedScanGold(float* reference, const float* idata, 
                            const unsigned int *iflag,
                            const unsigned int len,
                            const CUDPPConfiguration & config) 
{
    reference[0] = -FLT_MAX;
 
    int startIdx, stopIdx;
    int increment;

    startIdx = 1;
    stopIdx = len ;
    increment = 1 ;
    
    float total_max = -FLT_MAX;

    for( int i = startIdx; i != stopIdx; i = i + increment)
    {
        total_max = 
            (iflag[i] == 1) ? -FLT_MAX : std::max(total_max, idata[i-increment]);

        reference[i] = 
                (iflag[i] == 1) ? 
                -FLT_MAX : std::max(idata[i-increment], reference[i-increment]);
    }

    if (config.options & CUDPP_OPTION_INCLUSIVE) 
    {
        total_max = std::max(total_max, idata[stopIdx - increment]);

        for (unsigned i = 0; i < len; i++) 
        {
            reference[i] = std::max(reference[i], idata[i]);
        }
    }

    if (total_max != reference[stopIdx - increment])
    {
        printf("Warning: exceeding single-precision accuracy.  Scan will be inaccurate.\n");
        printf("total max %f\n", total_max);
        printf("reference %f\n", reference[stopIdx - increment]);
    }   
}

void
computeMultiplySegmentedScanGold(float* reference, const float* idata, 
                                 const unsigned int *iflag,
                                 const unsigned int len,
                                 const CUDPPConfiguration & config) 
{
    reference[0] = 1.0;
 
    int startIdx, stopIdx;
    int increment;

    startIdx = 1;
    stopIdx = len ;
    increment = 1 ;
    
    double total_multi = 1.0;

    for( int i = startIdx; i != stopIdx; i = i + increment)
    {
        total_multi = 
            (iflag[i] == 1) ? 1.0 : total_multi * idata[i-increment];

        reference[i] = 
                (iflag[i] == 1) ? 
                1.0f : idata[i-increment] * reference[i-increment];
    }

    if (config.options & CUDPP_OPTION_INCLUSIVE) 
    {
        total_multi = total_multi * idata[stopIdx - increment];

        for (unsigned i = 0; i < len; i++) 
        {
            reference[i] = reference[i] * idata[i];
        }
    }

    if (total_multi != reference[stopIdx - increment])
    {
        printf("Warning: exceeding single-precision accuracy.  Scan will be inaccurate.\n");
        printf("total multi %f\n", total_multi);
        printf("reference %f\n", reference[stopIdx - increment]);
    }   
}

void
computeMinSegmentedScanGold(float* reference, const float* idata, 
                            const unsigned int *iflag,
                            const unsigned int len,
                            const CUDPPConfiguration & config) 
{
    reference[0] = FLT_MAX;
 
    int startIdx, stopIdx;
    int increment;

    startIdx = 1;
    stopIdx = len ;
    increment = 1 ;
    
    float total_min = FLT_MAX;

    for( int i = startIdx; i != stopIdx; i = i + increment)
    {
        total_min = 
            (iflag[i] == 1) ? FLT_MAX : std::min(total_min, idata[i-increment]);

        reference[i] = 
                (iflag[i] == 1) ? 
                FLT_MAX : std::min(idata[i-increment], reference[i-increment]);
    }

    if (config.options & CUDPP_OPTION_INCLUSIVE) 
    {
        total_min = std::min(total_min, idata[stopIdx - increment]);

        for (unsigned i = 0; i < len; i++) 
        {
            reference[i] = std::min(reference[i], idata[i]);
        }
    }

    if (total_min != reference[stopIdx - increment])
    {
        printf("Warning: exceeding single-precision accuracy.  Scan will be inaccurate.\n");
        printf("total min %f\n", total_min);
        printf("reference %f\n", reference[stopIdx - increment]);
    }   
}

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set for exclusive sum-scan
//! Each element is the sum of the elements before it in the array.
//! @param reference  reference data, computed but preallocated
//! @param idata      const input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
void
computeMultiRowSumScanGold( float *reference, const float *idata, 
                            const unsigned int len,
                            const unsigned int rows,
                            const CUDPPConfiguration &config) 
{
    if (config.options & CUDPP_OPTION_FORWARD)
    {
        for (unsigned int r = 0; r < rows; ++r)
        {
            reference[r*len] = 0;
        }
    }
    else if (config.options & CUDPP_OPTION_BACKWARD)
    {
        for (unsigned int r = 1; r <= rows; ++r)
        {
            reference[r*len-1] = 0;
        }
    }

    int startIdx = 0, stopIdx = 0;
    int increment = 0;

    if (config.options & CUDPP_OPTION_FORWARD)
    {
        startIdx = 1;
        stopIdx = len ;
        increment = 1 ;
    }
    else if (config.options & CUDPP_OPTION_BACKWARD)
    {
        startIdx = len-2;
        stopIdx = -1;
        increment = -1;
    }
    
    double *total_sum;
    total_sum = (double*) malloc(rows * sizeof(double));
    for (unsigned int r = 0; r < rows; ++r)
        total_sum[r] = 0;

    for( int i = startIdx; i != stopIdx; i = i + increment)
    {       
        for (unsigned int r = 0; r < rows; ++r)
        {   
            total_sum[r] += idata[i-increment];
            reference[r*len+i] = idata[r*len+i-increment] + reference[r*len+i-increment];
        }
    }

    for (unsigned int r = 0; r < rows; ++r)
    {
        if (total_sum[r] != reference[stopIdx - increment])
        {
            printf("Warning: exceeding single-precision accuracy.  Scan will be inaccurate.\n");
            printf("total sum %f\n", total_sum[r]);
        }  
    }
    free((void*)total_sum);
}

int maxi(int a, int b)
{
    return (a > b) ? a : b;
}

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set for exclusive max-scan
//! Each element is the max of the elements before it in the array.
//! First element set to INT_MIN
//! @param reference  reference data, computed but preallocated
//! @param idata      const input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
void
computeMaxScanGold(float *reference, const float *idata,   
                   const unsigned int len, const CUDPPConfiguration & config) 
{
    if(config.options & CUDPP_OPTION_FORWARD)
    {
        int j = 0;
        if (config.options & CUDPP_OPTION_EXCLUSIVE)
        {
            reference[0] = -FLT_MAX;
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
            reference[len-1] = -FLT_MAX;
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
void
computeMinScanGold(float *reference, const float *idata,   
                   const unsigned int len, const CUDPPConfiguration & config) 
{
    if(config.options & CUDPP_OPTION_FORWARD)
    {
        int j = 0;
        if (config.options & CUDPP_OPTION_EXCLUSIVE)
        {
            reference[0] = FLT_MAX;
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
            reference[len-1] = FLT_MAX;
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

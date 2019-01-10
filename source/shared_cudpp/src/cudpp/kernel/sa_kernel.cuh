// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
//  $Revision$
//  $Date$
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// -------------------------------------------------------------

#include <cudpp_globals.h>
#include <stdio.h>
#include "cudpp_util.h"
#define IDX (threadIdx.x + (blockIdx.x * blockDim.x))

/**
 * @file
 * sa_kernel.cuh
 *
 * @brief CUDPP kernel-level suffix array routines
 */

/** \addtogroup cudpp_kernel
 * @{
 */

/** @name Suffix Array Functions
 * @{
 */

typedef unsigned int uint;
typedef unsigned char uchar;

/** @brief Construct the input array
 *
 * This is the first stage in the SA. This stage construct the
 * input array composed of values of the input char array
 * followed by three 0s.
 *
 *
 * @param[in]   d_str         Input char array to perform the SA on.
 * @param[out]  d_str_value   Output unsigned int array prepared for SA.
 * @param[in]   str_length    The number of elements we are performing the SA on.
 *
 **/

__global__ void
strConstruct(uchar* d_str,
             uint* d_str_value,
             size_t str_length)
{
#if (__CUDA_ARCH__ >= 200)
   const int STRIDE = gridDim.x * blockDim.x;
   #pragma unroll
   for(int i = IDX; i < str_length; i += STRIDE)
      d_str_value[i] = (uint) d_str[i] +1 ;

   if (IDX > str_length-1 && IDX < str_length + 3) d_str_value[IDX] = 0;
#endif
}

/** @brief Reconstruct the output
 *
 * This is the final stage in the SA. This stage reconstruct the
 * output array by reducing each value by one.
 *
 *  @param[in,out] d_keys_sa   Final output of the suffix array which stores the
                               positions of sorted suffixes.
 *  @param[in]     str_length  Size of the array.
 *
 **/
__global__ void
resultConstruct(uint* d_keys_sa,
                size_t str_length)
{
#if (__CUDA_ARCH__ >= 200)
   const int STRIDE = gridDim.x * blockDim.x;
   #pragma unroll
   for(int i = IDX; i < str_length; i += STRIDE)
      d_keys_sa[i] = d_keys_sa[i] - 1;
#endif
}

/** @brief Initialize the SA12 triplets
 *  @param[in]   d_str           Initial array of character values.
 *  @param[out]  d_keys_uint_12  The keys of righ-most char in SA12 triplets.
 *  @param[out]  d_keys_srt_12   SA12 triplets positions.
 *  @param[in]   mod_1           The number of elements whose positions mod3 = 1 (SA1)
 *  @param[in]   tThreads        The number of elements whose positions mod3 = 1,2 (SA12)
 *
 **/
__global__ void
sa12_keys_construct(uint* d_str,
                    uint* d_keys_uint_12,
                    uint* d_keys_srt_12,
                    int mod_1,
                    size_t tThreads)
{
#if (__CUDA_ARCH__ >= 200)
    if(IDX < mod_1)
    {
      d_keys_srt_12[IDX] = IDX*3+1;
      d_keys_uint_12[IDX] = d_str[IDX*3+2];

    }
    else if(IDX < tThreads)
    {
      d_keys_srt_12[IDX] = (IDX-mod_1)*3+2;
      d_keys_uint_12[IDX] = d_str[(IDX-mod_1)*3+3];
    }

#endif
}

/** @brief Construct SA12 for the second radix sort
 *  @param[in]     d_str            Initial array of character values.
 *  @param[out]    d_keys_uint_12   The keys of second char in SA12 triplets.
 *  @param[in]     d_keys_srt_12    SA12 triplets positions.
 *  @param[in]     tThreads         The number of elements in SA12.
 *
 **/
__global__ void
sa12_keys_construct_0(uint* d_str,
                      uint* d_keys_uint_12,
                      uint* d_keys_srt_12,
                      size_t tThreads)
{
#if (__CUDA_ARCH__ >= 200)
    if (IDX < tThreads)
        d_keys_uint_12[IDX] = d_str[d_keys_srt_12[IDX]];

#endif
}

/** @brief Construct SA12 for the third radix sort
 *  @param[in]     d_str            Initial array of character values.
 *  @param[out]    d_keys_uint_12   The keys of third char in SA12 triplets.
 *  @param[in]     d_keys_srt_12    SA12 triplets positions.
 *  @param[in]     tThreads         The number of elements in SA12.
 *
 **/
__global__ void
sa12_keys_construct_1(uint* d_str,
                      uint* d_keys_uint_12,
                      uint* d_keys_srt_12,
                      size_t tThreads)
{
#if (__CUDA_ARCH__ >= 200)
    if (IDX < tThreads)
        d_keys_uint_12[IDX] = d_str[d_keys_srt_12[IDX]-1];

#endif
}

/** @brief Turn on flags for sorted SA12 triplets
 *  @param[in]    d_str             Initial array of character values.
 *  @param[in]    d_keys_srt_12     SA12 triplets positions.
 *  @param[out]   d_flag            Marking the sorted triplets.
 *  @param[out]   result            0 if SA12 is not fully sorted.
 *  @param[in]    tThreads          The number of elements in SA12.
 *  @param[in]    str_length        The number of elements in original string.
 *
 **/
__global__ void
compute_rank(uint* d_str,
             uint* d_keys_srt_12,
             uint* d_flag,
             bool* result,
             size_t tThreads,
             int str_length)
{
#if (__CUDA_ARCH__ >= 200)
    if(IDX==0) d_flag[IDX]=1;
    else if(IDX < tThreads)
    {
        int i=d_keys_srt_12[IDX], j=d_keys_srt_12[IDX-1];
        if(i < str_length+2 && j < str_length+2)
        {
            if((d_str[i-1]==d_str[j-1]) && (d_str[i]==d_str[j]) &&
               (d_str[i+1]==d_str[j+1])) {
                d_flag[IDX] = 0; result[0]=0;
            } else {
                d_flag[IDX] = 1;
            }
        }
    }
#endif
}

/** @brief Construct new array for recursion
 *  @param[out]    d_new_str         The new string to be sent to recursion.
 *  @param[in]     d_keys_srt_12     SA12 triplets positions.
 *  @param[in]     d_rank            Ranks of SA12 from compute_rank kernel.
 *  @param[in]     mod_1             The number of elements of SA1.
 *  @param[in]     tThreads          The number of elements of SA12.
 *
 **/
__global__ void
new_str_construct(uint* d_new_str,
                  uint* d_keys_srt_12,
                  uint* d_rank,
                  int mod_1,
                  size_t tThreads)
{
#if (__CUDA_ARCH__ >= 200)
    if(IDX<tThreads)
    {
        uint pos = d_keys_srt_12[IDX];
        uint rank = d_rank[IDX];
        if(pos%3 == 1) d_new_str[(pos-1)/3] = rank;
        else d_new_str[mod_1+(pos-2)/3] = rank;
    }
    else if(IDX == tThreads || IDX == tThreads+1) d_new_str[IDX]=0;
#endif
}

/** @brief Translate SA12 from recursion
 *  @param[in,out]  d_keys_srt_12   Sorted SA12.
 *  @param[in]      d_isa_12        ISA12.
 *  @param[in]      d_flag          Flags to mark SA1.
 *  @param[in]      mod_1           The number of elements in SA1.
 *  @param[in]      tThreads        The number of elements in SA12.
 *
 **/
__global__ void
reconstruct(uint* d_keys_srt_12,
            uint* d_isa_12,
            uint* d_flag,
            int mod_1,
            size_t tThreads)
{
#if (__CUDA_ARCH__ >= 200)
    if(IDX<tThreads)
    {
        uint pos=d_keys_srt_12[IDX];
        if(pos<tThreads+1){
            d_isa_12[pos-1]=IDX+1;

            if(pos > mod_1)
            {
              d_keys_srt_12[IDX] = 3*(pos-mod_1-1)+2;
              d_flag[IDX]=0;
            }
            else
            {
              d_keys_srt_12[IDX] = 3*(pos-1)+1;
              if(pos>1) d_flag[IDX] =1;
              else d_flag[IDX]=0;
            }
        }
    }

#endif
}

/** @brief Construct ISA12
 *  @param[in]      d_keys_srt_12  Fully sorted SA12 in global position.
 *  @param[out]     d_isa_12       ISA12 to store the ranks in local position.
 *  @param[out]     d_flag         Flags to mark SA1.
 *  @param[in]      mod_1          The number of elements in SA1.
 *  @param[in]      tThreads       The number of elements in SA12.
 *
 **/
__global__ void
isa12_construct(uint* d_keys_srt_12,
                uint* d_isa_12,
                uint* d_flag,
                int mod_1,
                size_t tThreads)
{
#if (__CUDA_ARCH__ >= 200)
  uint pos;
  if(IDX<tThreads)
  {
    pos = d_keys_srt_12[IDX];
    if(pos%3==1) {
       pos = (pos-1)/3;
       if(d_keys_srt_12[IDX]>3) d_flag[IDX]=1;
       else d_flag[IDX]=0;
    }
    else if(pos%3==2) {
        pos = mod_1+ (pos-2)/3;
        d_flag[IDX]=0;
    }
  }
  __syncthreads();

  if(pos<tThreads && IDX<tThreads)
    d_isa_12[pos] = IDX+1;


#endif
}

/** @brief Contruct SA3 triplets positions
 *  @param[out]     d_keys_srt_3   SA3 generated from SA1.
 *  @param[in]      d_str          Original input array.
 *  @param[in]      d_keys_srt_12  Fully sorted SA12.
 *  @param[in]      d_keys_sa      Positions of SA1.
 *  @param[in]      tThreads1      The number of elements of SA12.
 *  @param[in]      tThreads2      The number of elements of SA3.
 *  @param[in]      str_length     The number of elements in original string.
 *
 **/
__global__ void
sa3_srt_construct(uint* d_keys_srt_3,
                  uint* d_str,
                  uint* d_keys_srt_12,
                  uint* d_keys_sa,
                  size_t tThreads1,
                  size_t tThreads2,
                  int str_length)
{
#if (__CUDA_ARCH__ >= 200)
    if(IDX<tThreads1)
    {
        uint pos=d_keys_sa[IDX];
        if((str_length+1)%3==0)
        {
            if(IDX == 0) d_keys_srt_3[IDX] = str_length+1;
            if(d_keys_srt_12[IDX] > 3 && d_keys_srt_12[IDX] % 3 == 1 &&
               pos<tThreads2-1)
                d_keys_srt_3[pos+1]=d_keys_srt_12[IDX]-1;
        }
        else
        {
            if(d_keys_srt_12[IDX]>3 && d_keys_srt_12[IDX]%3==1 && pos<tThreads2)
               d_keys_srt_3[pos]=d_keys_srt_12[IDX]-1;
        }
    }
#endif
}

/** @brief Construct SA3 triplets keys
 *  @param[in]     d_keys_srt_3   SA3 triplets positions.
 *  @param[out]    d_keys_sa      SA3 keys.
 *  @param[in]     d_str          Original input string.
 *  @param[in]     tThreads       The number of elements in SA12.
 *  @param[in]     str_length     The number of elements in original string.
 *
 **/
__global__ void
sa3_keys_construct(uint* d_keys_srt_3,
                   uint* d_keys_sa,
                   uint* d_str,
                   size_t tThreads,
                   int str_length)
{
#if (__CUDA_ARCH__ >= 200)
    if(IDX<tThreads)
    {
       if(d_keys_srt_3[IDX] < str_length+4)
          d_keys_sa[IDX] = d_str[d_keys_srt_3[IDX]-1];
    }
#endif
}

/** @brief Construct SA12 keys in terms of Vector
 *  @param[in]    d_str          Original input data stream
 *  @param[in]    d_keys_srt_12  The order of aKeys.
 *  @param[in]    d_isa_12       The ranks in SA12 orders.
 *  @param[out]   d_aKeys        SA12 keys in Vectors.
 *  @param[in]    tThreads       The number elements in SA12
 *  @param[in]    mod_1          The number of elements in SA1.
 *  @param[in]    bound          The number of elements in SA12 plus SA3.
 *  @param[in]    str_length     The number of elements in original string.
 *
 **/
__global__ void
merge_akeys_construct(uint* d_str,
                      uint* d_keys_srt_12,
                      uint* d_isa_12,
                      Vector* d_aKeys,
                      size_t tThreads,
                      int mod_1,
                      int bound,
                      int str_length)
{
#if (__CUDA_ARCH__ >= 200)
    if(IDX < tThreads)
    {
        int i = d_keys_srt_12[IDX];
        if(i < str_length+3)
        {
            if(i%3==1)
            {
                d_aKeys[IDX].a = d_str[i-1];
                d_aKeys[IDX].b = (bound-i>0) ? d_isa_12[mod_1+(i-1)/3] : 0;
                d_aKeys[IDX].c = 0;
                d_aKeys[IDX].d = 1;
            }
            else if(i%3==2)
            {
                d_aKeys[IDX].a = d_str[i-1];
                d_aKeys[IDX].b = (bound-i>0) ? d_str[i] : 0;
                d_aKeys[IDX].c = (bound-i>1) ? d_isa_12[(i-2)/3+1] : 0;
                d_aKeys[IDX].d = 0;
            }
        }
    }
#endif
}

/** @brief Construct SA3 keys in Vector
 *
 *  @param[in]     d_str         Original input data stream.
 *  @param[in]     d_keys_srt_3  The order of bKeys
 *  @param[in]     d_isa_12      ISA12.
 *  @param[out]    d_bKeys       SA3 keys in Vectors.
 *  @param[in]     tThreads      The number of total threads.
 *  @param[in]     mod_1         The number of elements in SA1.
 *  @param[in]     bound         The number of elements in SA12 and SA3.
 *  @param[in]     str_length    The number of elements in original str.
 *
 **/
__global__ void
merge_bkeys_construct(uint* d_str,
                      uint* d_keys_srt_3,
                      uint* d_isa_12,
                      Vector* d_bKeys,
                      size_t tThreads,
                      int mod_1,
                      int bound,
                      int str_length)
{
#if (__CUDA_ARCH__ >= 200)
    if(IDX<tThreads){
        int i = d_keys_srt_3[IDX];
        if(i < str_length+3){
            d_bKeys[IDX].a = d_str[i-1];
            d_bKeys[IDX].b = (bound-i>0) ? d_str[i] : 0;
            d_bKeys[IDX].c = (bound-i>0) ? d_isa_12[i/3] : 0;
            d_bKeys[IDX].d = (bound-i>1) ? d_isa_12[i/3+mod_1] : 0;
        }
    }
#endif
}

/** @} */ // end suffix array functions
/** @} */ // end cudpp_kernel


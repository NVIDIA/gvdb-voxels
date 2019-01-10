// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
//  $Revision: 4400 $
//  $Date: 2008-08-04 10:58:14 -0700 (Mon, 04 Aug 2008) $
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt 
// in the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
 * @file
 * rand_kernel.cu
 *
 * @brief CUDPP kernel-level rand routines
 */

/** \addtogroup cudpp_kernel
  * @{
  */
/** @name Rand Functions
 * @{
 */

/**
 * @brief The main MD5 generation algorithm.
 *
 * This function runs the MD5 hashing random number generator.  It generates
 * MD5 hashes, and uses the output as randomized bits.  To repeatedly call this
 * function, always call cudppRandSeed() first to set a new seed or else the output
 * may be the same due to the deterministic nature of hashes.  gen_randMD5 generates
 * 128 random bits per thread.  Therefore, the parameter \a d_out is expected to be
 * an array of type uint4 with \a numElements indicies.
 *
 * @param[out] d_out the output array of type uint4.
 * @param[in] numElements the number of elements in \a d_out
 * @param[in] seed the random seed used to vary the output
 *
 * @see launchRandMD5Kernel()
 */
__global__ void gen_randMD5(uint4 *d_out, size_t numElements, unsigned int seed)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    unsigned int data[16];
    setupInput(data, seed);
    
    unsigned int h0 = 0x67452301;
    unsigned int h1 = 0xEFCDAB89;
    unsigned int h2 = 0x98BADCFE;
    unsigned int h3 = 0x10325476;

    uint4 result = make_uint4(h0,h1,h2,h3);
    uint4 td = result;

    float p = pow(2.0,32.0);
    
    uint4 Fr = make_uint4(7,12,17,22);
    uint4 Gr = make_uint4(5,9,14,20);
    uint4 Hr = make_uint4(4,11,16,23);
    uint4 Ir = make_uint4(6,10,15,21);    
    
    //for optimization, this is loop unrolled
    FF(&td, 0, &Fr,p,data);
    FF(&td, 1, &Fr,p,data);
    FF(&td, 2, &Fr,p,data);
    FF(&td, 3, &Fr,p,data);
    FF(&td, 4, &Fr,p,data);
    FF(&td, 5, &Fr,p,data);
    FF(&td, 6, &Fr,p,data);
    FF(&td, 7, &Fr,p,data);
    FF(&td, 8, &Fr,p,data);
    FF(&td, 9, &Fr,p,data);
    FF(&td,10, &Fr,p,data);
    FF(&td,11, &Fr,p,data);
    FF(&td,12, &Fr,p,data);
    FF(&td,13, &Fr,p,data);
    FF(&td,14, &Fr,p,data);
    FF(&td,15, &Fr,p,data);

    GG(&td,16, &Gr,p,data);
    GG(&td,17, &Gr,p,data);
    GG(&td,18, &Gr,p,data);
    GG(&td,19, &Gr,p,data);
    GG(&td,20, &Gr,p,data);
    GG(&td,21, &Gr,p,data);
    GG(&td,22, &Gr,p,data);
    GG(&td,23, &Gr,p,data);
    GG(&td,24, &Gr,p,data);
    GG(&td,25, &Gr,p,data);
    GG(&td,26, &Gr,p,data);
    GG(&td,27, &Gr,p,data);
    GG(&td,28, &Gr,p,data);
    GG(&td,29, &Gr,p,data);
    GG(&td,30, &Gr,p,data);
    GG(&td,31, &Gr,p,data);

    HH(&td,32, &Hr,p,data);
    HH(&td,33, &Hr,p,data);
    HH(&td,34, &Hr,p,data);
    HH(&td,35, &Hr,p,data);
    HH(&td,36, &Hr,p,data);
    HH(&td,37, &Hr,p,data);
    HH(&td,38, &Hr,p,data);
    HH(&td,39, &Hr,p,data);
    HH(&td,40, &Hr,p,data);
    HH(&td,41, &Hr,p,data);
    HH(&td,42, &Hr,p,data);
    HH(&td,43, &Hr,p,data);
    HH(&td,44, &Hr,p,data);
    HH(&td,45, &Hr,p,data);
    HH(&td,46, &Hr,p,data);
    HH(&td,47, &Hr,p,data);

    II(&td,48, &Ir,p,data);
    II(&td,49, &Ir,p,data);
    II(&td,50, &Ir,p,data);
    II(&td,51, &Ir,p,data);
    II(&td,52, &Ir,p,data);
    II(&td,53, &Ir,p,data);
    II(&td,54, &Ir,p,data);
    II(&td,55, &Ir,p,data);
    II(&td,56, &Ir,p,data);
    II(&td,57, &Ir,p,data);
    II(&td,58, &Ir,p,data);
    II(&td,59, &Ir,p,data);
    II(&td,60, &Ir,p,data);
    II(&td,61, &Ir,p,data);
    II(&td,62, &Ir,p,data);
    II(&td,63, &Ir,p,data);
/*    */        
    result.x = result.x + td.x;
    result.y = result.y + td.y;
    result.z = result.z + td.z;
    result.w = result.w + td.w;

    __syncthreads();

    if (idx < numElements)
    {
        d_out[idx].x = result.x;
        d_out[idx].y = result.y;
        d_out[idx].z = result.z;
        d_out[idx].w = result.w;
    }
}
/** @} */ // end rand functions
/** @} */ // end cudpp_kernel


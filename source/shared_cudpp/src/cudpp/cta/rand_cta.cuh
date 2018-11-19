// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
//  $Revision: 4730 $
//  $Date: 2009-01-14 21:38:38 -0800 (Wed, 14 Jan 2009) $
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
 * @file
 * rand_cta.cu
 * 
 * @brief CUDPP CTA-level rand routines
 */

/** \addtogroup cudpp_cta 
* @{
*/

/** @name Rand Functions
* @{
*/


//------------MD5 ROTATING FUNCTIONS------------------------

/**
 * @brief Does a GLSL-style swizzle assigning f->xyzw = f->yzwx
 * 
 *  It does the equvalent of f->xyzw = f->yzwx since this functionality is 
 *  in shading languages but not exposed in CUDA.
 *  @param[in] f the uint4 data type which will have its elements shifted.  Passed in as pointer.
 * 
 **/
__device__ void swizzleShift(uint4 *f)
{
    unsigned int temp;
    temp = f->x;
    f->x = f->y;
    f->y = f->z;
    f->z = f->w;
    f->w = temp;
}
/**
 * @brief Rotates the bits in \a x over by \a n bits.
 * 
 *  This is the equivalent of the ROTATELEFT operation as described in
 *  the MD5 working memo. It takes the bits in \a x and circular
 *  shifts it over by \a n bits.
 *
 *  For more information see: <a
 *  href="http://tools.ietf.org/html/rfc1321">The MD5 Message-Digest
 *  Algorithm</a>.
 * 
 *  @param[in] x the variable with the bits 
 *  @param[in] n the number of bits to shift left by.
 *  @returns Rotated input
 **/
__device__ unsigned int leftRotate(unsigned int x, unsigned int n)
{
    unsigned int t = ( ((x) << (n)) | ((x) >> (32-n)) ) ;
    return t;
}

/**
 * @brief The F scrambling function.
 * 
 *  The F function in the MD5 technical memo scrambles three variables 
 *  \a x, \a y, and \a z in the following way using bitwise logic:
 *
 *  (x & y) | ((~x) & z)
 *
 *  The resulting value is returned as an unsigned int.  
 *
 *  For more information see: <a href="http://tools.ietf.org/html/rfc1321">The MD5 Message-Digest Algorithm</a>
 * 
 *  @param[in] x See the above formula
 *  @param[in] y See the above formula
 *  @param[in] z See the above formula
 *  @returns F(x, y, z)
 *
 *  @see FF()
 **/
__device__ unsigned int F(unsigned int x, unsigned int y, unsigned int z)
{
    unsigned int t;
    t = ( (x&y) | ((~x) & z) );
    return t;
}

/**
 * @brief The G scrambling function.
 * 
 *  The G function in the MD5 technical memo scrambles three variables 
 *  \a x, \a y, and \a z in the following way using bitwise logic:
 *
 *  (x & z) | ((~z) & y)
 *
 *  The resulting value is returned as an unsigned int.  
 *
 *  For more information see: <a href="http://tools.ietf.org/html/rfc1321">The MD5 Message-Digest Algorithm</a>
 * 
 *  @param[in] x See the above formula
 *  @param[in] y See the above formula
 *  @param[in] z See the above formula
 *  @returns G(x, y, z)
 *
 *  @see GG()
**/
__device__ unsigned int G(unsigned int x, unsigned int y, unsigned int z)
{
    unsigned int t;
    t = ( (x&z) | ((~z) & y) );
    return t;
}

/**
 * @brief The H scrambling function.
 * 
 *  The H function in the MD5 technical memo scrambles three variables 
 *  \a x, \a y, and \a z in the following way using bitwise logic:
 *
 *  (x ^ y ^ z)
 *
 *  The resulting value is returned as an unsigned int.  
 *
 *  For more information see: <a href="http://tools.ietf.org/html/rfc1321">The MD5 Message-Digest Algorithm</a>
 * 
 *  @param[in] x See the above formula
 *  @param[in] y See the above formula
 *  @param[in] z See the above formula
 *  @returns H(x, y, z)
 *
 *  @see HH()
 **/
__device__ unsigned int H(unsigned int x, unsigned int y, unsigned int z)
{
    unsigned int t;
    t = (x ^ y ^ z );
    return t;
}

/**
 * @brief The I scrambling function.
 * 
 *  The I function in the MD5 technical memo scrambles three variables 
 *  \a x, \a y, and \a z in the following way using bitwise logic:
 *
 *  (y ^ (x | ~z))
 *
 *  The resulting value is returned as an unsigned int.  
 *
 *  For more information see: <a href="http://tools.ietf.org/html/rfc1321">The MD5 Message-Digest Algorithm</a>
 * 
 *  @param[in] x See the above formula
 *  @param[in] y See the above formula
 *  @param[in] z See the above formula
 *  @returns I(x, y, z)
 *
 *  @see II()
 **/
__device__ unsigned int I(unsigned int x, unsigned int y, unsigned int z)
{
    unsigned int t;
    t = ( y ^ (x | ~z) );
    return t;
}

/**
 * @brief The FF scrambling function
 * 
 *  The FF function in the MD5 technical memo is a wrapper for the F
 *  scrambling function as well as performing its own rotations using
 *  LeftRotate and swizzleShift. The variable \a td is the current
 *  scrambled digest which is passed along and scrambled using the
 *  current iteration \a i, the rotation information \a Fr, and the
 *  starting input \a data. \a p is kept as a constant of 2^32. The
 *  resulting value is stored in \a td.
 *
 *  For more information see: <a
 *  href="http://tools.ietf.org/html/rfc1321">The MD5 Message-Digest
 *  Algorithm</a>.
 * 
 *  @param[in,out] td The current value of the digest stored as an uint4.
 *  @param[in] i  The current iteration of the algorithm.  This affects the values in \a data.
 *  @param[in] Fr The current rotation order.
 *  @param[in] p The constant 2^32.
 *  @param[in] data The starting input to MD5.  Padded from setupInput().
 *  @returns FF(input)
 * 
 *  @see F()
 *  @see swizzleShift()
 *  @see leftRotate()
 *  @see setupInput()
**/
__device__ void FF(uint4 * td, int i, uint4 * Fr, float p, unsigned int * data)
{
    unsigned int Ft = F(td->y, td->z, td->w);
    unsigned int r = Fr->x;
    swizzleShift(Fr);
    
    float t = sin(__int_as_float(i)) * p;
    unsigned int trigFunc = __float2uint_rd(t);
    td->x = td->y + leftRotate(td->x + Ft + trigFunc + data[i], r);
    swizzleShift(td);
}

/**
 * @brief The GG scrambling function
 * 
 *  The GG function in the MD5 technical memo is a wrapper for the G
 *  scrambling function as well as performing its own rotations using
 *  LeftRotate() and swizzleShift(). The variable \a td is the current
 *  scrambled digest which is passed along and scrambled using the
 *  current iteration \a i, the rotation information \a Gr, and the
 *  starting input \a data. \a p is kept as a constant of 2^32. The
 *  resulting value is stored in \a td.
 *
 *  For more information see: <a
 *  href="http://tools.ietf.org/html/rfc1321">The MD5 Message-Digest
 *  Algorithm</a>.
 * 
 *  @param[in,out] td The current value of the digest stored as an uint4.
 *  @param[in] i  The current iteration of the algorithm.  
 *                This affects the values in \a data.
 *  @param[in] Gr The current rotation order.
 *  @param[in] p The constant 2^32.
 *  @param[in] data The starting input to MD5.  Padded from setupInput().
 *  @returns GG(input)
 *
 *  @see G()
 *  @see swizzleShift()
 *  @see leftRotate()
 *  @see setupInput()
**/
__device__ void GG(uint4 * td, int i, uint4 * Gr, float p, unsigned int * data)
{
    unsigned int Ft = G(td->y, td->z, td->w);
    i = (5*i+1) %16;
    unsigned int r = Gr->x;
    swizzleShift(Gr);
    
    float t = sin(__int_as_float(i)) * p;
    unsigned int trigFunc = __float2uint_rd(t);
    td->x = td->y + leftRotate(td->x + Ft + trigFunc + data[i], r);
    swizzleShift(td);
}

/**
 * @brief The HH scrambling function
 * 
 *  The HH function in the MD5 technical memo is a wrapper for the H
 *  scrambling function as well as performing its own rotations using
 *  LeftRotate() and swizzleShift(). The variable \a td is the current
 *  scrambled digest which is passed along and scrambled using the
 *  current iteration \a i, the rotation information \a Hr, and the
 *  starting input \a data. \a p is kept as a constant of 2^32. The
 *  resulting value is stored in \a td.
 *
 *  For more information see: <a
 *  href="http://tools.ietf.org/html/rfc1321">The MD5 Message-Digest
 *  Algorithm</a>.
 * 
 *  @param[in,out] td The current value of the digest stored as an uint4.
 *  @param[in] i  The current iteration of the algorithm.  
 *                This affects the values in \a data.
 *  @param[in] Hr The current rotation order.
 *  @param[in] p The constant 2^32.
 *  @param[in] data The starting input to MD5.  Padded from setupInput().
 *  @returns HH(input)
 *
 *  @see H()
 *  @see swizzleShift()
 *  @see leftRotate()
 *  @see setupInput()
**/
__device__ void HH(uint4 * td, int i, uint4 * Hr, float p, unsigned int * data)
{
    unsigned int Ft = H(td->y, td->z, td->w);
    i = (3*i+5) %16;
    unsigned int r = Hr->x;
    swizzleShift(Hr);
    
    float t = sin(__int_as_float(i)) * p;
    unsigned int trigFunc = __float2uint_rd(t);
    td->x = td->y + leftRotate(td->x + Ft + trigFunc + data[i], r);
    swizzleShift(td);
}

/**
 * @brief The II scrambling function
 * 
 *  The II function in the MD5 technical memo is a wrapper for the I
 *  scrambling function as well as performing its own rotations using
 *  LeftRotate() and swizzleShift(). The variable \a td is the current
 *  scrambled digest which is passed along and scrambled using the
 *  current iteration \a i, the rotation information \a Ir, and the
 *  starting input \a data. \a p is kept as a constant of 2^32. The
 *  resulting value is stored in \a td.
 *
 *  For more information see: <a
 *  href="http://tools.ietf.org/html/rfc1321">The MD5 Message-Digest
 *  Algorithm</a>.
 * 
 *  @param[in,out] td The current value of the digest stored as an uint4.
 *  @param[in] i  The current iteration of the algorithm.  
 *                This affects the values in \a data.
 *  @param[in] Ir The current rotation order.
 *  @param[in] p The constant 2^32.
 *  @param[in] data The starting input to MD5.  Padded from setupInput().
 *  @returns II(input)
 *
 *  @see I()
 *  @see swizzleShift()
 *  @see leftRotate()
 *  @see setupInput()
**/
__device__ void II(uint4 * td, int i, uint4 * Ir, float p, unsigned int * data)
{
    unsigned int Ft = G(td->y, td->z, td->w);
    i = (7*i) %16;
    unsigned int r = Ir->x;
    swizzleShift(Ir);
    
    float t = sin(__int_as_float(i)) * p;
    unsigned int trigFunc = __float2uint_rd(t);
    td->x = td->y + leftRotate(td->x + Ft + trigFunc + data[i], r);
    swizzleShift(td);
}

/**
 * @brief Sets up the \a input array using information of \a seed, and \a threadIdx
 * 
 *  This function sets up the \a input array using a combination of
 *  the current thread's id and the user supplied \a seed.
 *
 *  For more information see: <a
 *  href="http://tools.ietf.org/html/rfc1321">The MD5 Message-Digest
 *  Algorithm</a>.
 * 
 *  @param[out] input The array which will contain the initial values for 
 *                    all the scrambling functions.
 *  @param[in] seed The user supplied seed as an unsigned int.
 *
 *  @see FF()
 *  @see GG()
 *  @see HH()
 *  @see II()
 *  @see gen_randMD5()
**/
__device__ void setupInput(unsigned int * input, unsigned int seed)
{    
    //loop unroll, also do this more intelligently
    input[0] = threadIdx.x ^ seed;
    input[1] = threadIdx.y ^ seed;
    input[2] = threadIdx.z ^ seed;
    input[3] = 0x80000000 ^ seed;
    input[4] = blockIdx.x ^ seed;
    input[5] = seed;
    input[6] = seed;
    input[7] = blockDim.x ^ seed;
    input[8] = seed;
    input[9] = seed;
    input[10] = seed;
    input[11] = seed;
    input[12] = seed;
    input[13] = seed;
    input[14] = seed;
    input[15] = 128 ^ seed;
}

//-------------------END MD5 FUNCTIONS--------------------------------------

/** @} */ // end rand functions
/** @} */ // end cudpp_cta

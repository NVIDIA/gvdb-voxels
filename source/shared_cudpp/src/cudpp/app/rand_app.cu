// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: 4400 $
// $Date: 2008-08-04 10:58:14 -0700 (Mon, 04 Aug 2008) $
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt 
// in the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
 * @file
 * rand_md5_app.cu
 *
 * @brief CUDPP application-level rand routine for MD5
 */

#include "cuda_util.h"
#include "cudpp.h"
#include "cudpp_util.h"
#include "cudpp_plan.h"

#include <cstdlib>
#include <cstdio>
#include <assert.h>

#include "cta/rand_cta.cuh"
#include "kernel/rand_kernel.cuh"

#define RAND_CTA_SIZE 128 //128 chosen, may be changed later

/** \addtogroup cudpp_app
  *
  */

/** @name Rand Functions
 * @{
 */

/**@brief Launches the MD5 Random number generator kernel
 *
 * The MD5 Random number generator works by generating 128 bit digests which 
 * are then broken down into 32 bit chunks and stored inside \a d_out.  
 * \a d_out is expected to  be of type unsigned int and can hold \a numElements 
 * elements.
 *
 * An analysis of the stastical distribution of the MD5 random number generator
 * can be found in the original paper 
 * <a href="http://portal.acm.org/citation.cfm?id=1342263">
 * Parallel white noise generation on a GPU via cryptographic hash</a>.
 * The optimizations mentioned in the paper are also present in the CUDPP
 * version of the MD5 Random number generator.
 *
 * It is also worth pointing out that the GPU version will \b not generate 
 * the same output * as the CPU version.  This is due to the difference in the 
 * floating point accuracy and several optimizations that have been used 
 * (i.e. calculating sin using device hardware  rather than storing it in 
 * an array that the original implementation does).  However, the distribution 
 * of the numbers is well suited for random number generation, even without
 * the CPU-GPU invariance.
 *
 * @param[out] d_out the array of unsigned integers allocated on device memory
 * @param[in] seed the random seed used to vary the output
 * @param[in] numElements the number of elements in \a d_out
 * @see gen_randMD5()
 * @see cudppRand()
 * @todo: chose a better block size, perhaps a multiple of two is optimal
 */
void launchRandMD5Kernel(unsigned int * d_out, unsigned int seed, 
                         size_t numElements)
{
    //first, we need a temporary array of uints
    uint4 * dev_output;

    //figure out how many elements are needed in this array
    unsigned int devOutputsize = (unsigned int)(numElements / 4);
    devOutputsize += (numElements %4 == 0) ? 0 : 1; //used for overflow
    unsigned int memSize = devOutputsize * sizeof(uint4);


    //now figure out block size
    unsigned int blockSize = RAND_CTA_SIZE;
    if(devOutputsize < RAND_CTA_SIZE) blockSize = devOutputsize;

    unsigned int n_blocks = 
            devOutputsize/blockSize + (devOutputsize%blockSize == 0 ? 0:1);  

    //printf("Generating %u random numbers using %u blocks and %u threads per block\n", numElements, n_blocks, blockSize);
/*  old debug code now removed.
    printf("\nnumber of elements: %u, devOutputSize: %u\n", 
            numElements, devOutputsize);
    printf("number of blocks: %u blocksize: %u devOutputsize = %u\n", 
            n_blocks, blockSize, devOutputsize);
    printf("number of threads: %u\n", n_blocks * blockSize);
    printf("seed value: %u\n", seed);
*/
    //now create the memory on the device
    CUDA_SAFE_CALL( cudaMalloc((void **) &dev_output, memSize));
    CUDA_SAFE_CALL( cudaMemset(dev_output, 0, memSize)); 
    gen_randMD5<<<n_blocks, blockSize>>>(dev_output, devOutputsize, seed);

    //here the GPU computation is done
    //here we have all the data on the device, we copy it over into host memory


    //calculate final memSize
    //@TODO: write a template version of this which calls two different version 
    // depending if numElements %4 == 0
    size_t finalMemSize = sizeof(unsigned int) * numElements;
    CUDA_SAFE_CALL( cudaMemcpy(d_out, dev_output, finalMemSize, 
                               cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL( cudaFree(dev_output));
}//end launchRandMD5Kernel

#ifdef __cplusplus
extern "C"
{
#endif

/**@brief Dispatches the rand function based on the plan
 *
 * This is the dispatch call which looks at the algorithm specified in \a plan 
 * and calls the appropriate random number generation algorithm.  
 *
 * @param[out] d_out the array allocated on device memory where the random 
 * numbers will be stored
 * must be of type unsigned int
 * @param[in] numElements the number of elements in the array d_out
 * @param[in] plan pointer to CUDPPRandPlan which contains the algorithm to run
 */
void cudppRandDispatch(void * d_out, size_t numElements, 
                       const CUDPPRandPlan * plan)
{
    //switch to figure out which algorithm to run
    switch(plan->m_config.algorithm)
    {
    case CUDPP_RAND_MD5:
        //run the md5 algorithm here
        launchRandMD5Kernel( (unsigned int *) d_out, plan->m_seed, numElements);
        break;
    default:
        break;
    }//end switch

}


#ifdef __cplusplus
}
#endif
/** @} */ // end rand_app
/** @} */ // end cudpp_app

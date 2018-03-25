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
#include "cudpp_plan.h"
#include "cudpp_manager.h"
#include "cudpp_maximal_launch.h"
#include "cuda_util.h"

typedef void* KernelPointer;


/** @addtogroup publicInterface
  * @{
  */

/** @name Library Management Interface
 * @{
 */

/**
 * @brief Creates an instance of the CUDPP library, and returns a handle.
 *
 * cudppCreate() must be called before any other CUDPP function.  In a 
 * multi-GPU application that uses multiple CUDA context, cudppCreate() must
 * be called once for each CUDA context.  Each call returns a different handle,
 * because each CUDA context (and the host thread that owns it) must use a 
 * separate instance of the CUDPP library.  
 *
 * @param[in,out] theCudpp a pointer to the CUDPPHandle for the created CUDPP instance.
 * @returns CUDPPResult indicating success or error condition
 */
CUDPP_DLL
CUDPPResult cudppCreate(CUDPPHandle* theCudpp)
{
    CUDPPManager *mgr = new CUDPPManager();
    *theCudpp = mgr->getHandle();
    return CUDPP_SUCCESS;
}

/**
 * @brief Destroys an instance of the CUDPP library given its handle.
 *
 * cudppDestroy() should be called once for each handle created using cudppCreate(),
 * to ensure proper resource cleanup of all library instances.
 *
 * @param[in] theCudpp the handle to the CUDPP instance to destroy.
 * @returns CUDPPResult indicating success or error condition
 */
CUDPP_DLL
CUDPPResult cudppDestroy(CUDPPHandle theCudpp)
{
    CUDPPManager *mgr = CUDPPManager::getManagerFromHandle(theCudpp);
    delete mgr;
    mgr = 0;
    return CUDPP_SUCCESS;
}

/** @} */ // end Library Management Interface

/** @} */ // end publicInterface

//! @brief CUDPP Manager constructor
CUDPPManager::CUDPPManager()
{
    int device = -1;
    CUDA_SAFE_CALL(cudaGetDevice(&device));
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&m_deviceProps, device));
}

/** @brief CUDPP Manager destructor 
*/
CUDPPManager::~CUDPPManager()
{
}

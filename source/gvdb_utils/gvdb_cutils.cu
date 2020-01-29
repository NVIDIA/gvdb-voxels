#include"gvdb_cutils.cuh"

void gvdbDeviceRadixSort(CUdeviceptr inSource, const uint32_t inCount){
    thrust::device_ptr<uint64_t> d_keys = thrust::device_pointer_cast((uint64_t*)inSource);
    thrust::device_vector<uint64_t> d_keysVec(d_keys, d_keys + inCount);
	thrust::sort(d_keysVec.begin(), d_keysVec.end(), thrust::less<uint64_t>());
}

void gvdbDeviceMaxElementF(CUdeviceptr inSource, CUdeviceptr inDest, const uint32_t inCount){

	thrust::device_ptr<float> d_dst = thrust::device_pointer_cast((float*)inSource);
    thrust::device_ptr<float> d_src = thrust::device_pointer_cast((float*)inDest);
    thrust::device_vector<float> d_srcVec(d_src, d_src + inCount);
    thrust::device_vector<float>::iterator iter = thrust::max_element(d_srcVec.begin(), d_srcVec.end());
    d_dst[0] = *iter;

}

void gvdbDeviceMinElementF(CUdeviceptr inSource, CUdeviceptr inDest, const uint32_t inCount){
	thrust::device_ptr<float> d_dst = thrust::device_pointer_cast((float*)inSource);
    thrust::device_ptr<float> d_src = thrust::device_pointer_cast((float*)inDest);
    thrust::device_vector<float> d_srcVec(d_src, d_src + inCount);
    thrust::device_vector<float>::iterator iter = thrust::min_element(d_srcVec.begin(), d_srcVec.end());
    d_dst[0] = *iter;
}



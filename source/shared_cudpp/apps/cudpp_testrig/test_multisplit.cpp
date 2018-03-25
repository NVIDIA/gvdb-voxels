// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision$
// $Date$
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt
// in the root directory of this source distribution.
// -------------------------------------------------------------

#include <stdio.h>
#include <math.h>
#include <cuda_runtime_api.h>

#include "cudpp.h"
#include "cudpp_testrig_options.h"
#include "cudpp_testrig_utils.h"
#include "cuda_util.h"
#include "commandline.h"

typedef unsigned int uint;

#ifdef WIN32
#undef min
#undef max

typedef _int32 int32_t;
typedef unsigned _int32 uint32_t;
typedef _int64 int64_t;
typedef unsigned _int64 uint64_t;
typedef unsigned _int64 uint64;

#else

#include <stdint.h>
typedef unsigned long long int uint64;

#endif

#include <limits>


using namespace cudpp_app;

void randomPermute(unsigned int *elements, unsigned int numElements)
{
  //Uses knuth's method to randomly permute
//  for(unsigned int i = 0; i < numElements; i++)
//    input[i] = i; //rand() + rand() << 15;

  for(unsigned int i = 0; i < numElements; i++)
  {
    unsigned int rand1 = rand();
    unsigned int rand2 = (rand() << 15) + rand1;
    unsigned int swap = i + (rand2%(numElements-i));

    unsigned int temp = elements[i];
    elements[i] = elements[swap];
    elements[swap] = temp;
  }
}

class LSBBucketMapperHost {
public:
  LSBBucketMapperHost(unsigned int numBuckets) {
    lsbBitMask = 0xFFFFFFFF >>
        (32 - (unsigned int) ceil(log2((float)numBuckets)));
    this->numBuckets = numBuckets;
  }

  unsigned int operator()(unsigned int element) {
    return (element & lsbBitMask) % numBuckets;
  }

private:
  unsigned int numBuckets;
  unsigned int lsbBitMask;
};

class MSBBucketMapperHost {
public:
  MSBBucketMapperHost(unsigned int numBuckets) {
    msbShift = 32 - ceil(log2((float)numBuckets));
    this->numBuckets = numBuckets;
  }

  unsigned int operator()(unsigned int element) {
    return (element >> msbShift) % numBuckets;
  }

private:
  unsigned int numBuckets;
  unsigned int msbShift;
};

class OrderedCyclicBucketMapperHost {
public:
  OrderedCyclicBucketMapperHost(unsigned int elements, unsigned int buckets) {
    numElements = elements;
    numBuckets = buckets;
    elementsPerBucket = (elements + buckets - 1) / buckets;
  }

  unsigned int operator()(unsigned int element) {
    return (element % numElements) / elementsPerBucket;
  }

private:
  unsigned int numBuckets;
  unsigned int numElements;
  unsigned int elementsPerBucket;
};

template<class T>
void cpuMultisplit(uint *input, uint *output, uint numElements, uint numBuckets,
    T bucketMapper) {
  // Performs the mutlisplit with arbitrary bucket distribution on cpu:
  // bucket_mode == 0: equal number of elements per bucket
  // bucket_mode == 1: most significant bits of input represent bucket ID
  // n: number of elements
  if (numBuckets == 1) {
    for (unsigned int k = 0; k < numElements; k++)
      output[k] = input[k];
    return;
  }

  uint logBuckets = ceil(log2(numBuckets));
  uint *bins = new uint[numBuckets]; // histogram results holder
  uint *scanBins = new uint[numBuckets];
  uint *currentIdx = new uint[numBuckets];
  // Computing histograms:
  uint bucketId;

  uint elsPerBucket = (numElements + numBuckets - 1)/numBuckets;
  uint msbShift = 32 - logBuckets;

  for(unsigned int k = 0; k<numBuckets; k++)
    bins[k] = 0;

  for(unsigned int i = 0; i<numElements ; i++)
  {
    //bucketId = ((bucket_mode == 0)?(input[i]/elsPerBucket):(input[i]>>msbShift));
    bucketId = bucketMapper(input[i]);
    bins[bucketId]++;
  }

  // computing exclusive scan operation on the inputs:
  scanBins[0] = 0;
  for(unsigned int j = 1; j<numBuckets; j++)
    scanBins[j] = scanBins[j-1] + bins[j-1];

  // Placing items in their new positions:
  for(unsigned int k = 0; k<numBuckets; k++)
    currentIdx[k] = 0;

  for(unsigned int i = 0; i<numElements; i++)
  {
    bucketId = bucketMapper(input[i]);
    output[scanBins[bucketId] + currentIdx[bucketId]] = input[i];
    currentIdx[bucketId]++;
  }
  // releasing memory:
  delete[] bins;
  delete[] scanBins;
  delete[] currentIdx;
}

template<class T>
void cpuMultiSplitPairs(uint* keys_input, uint* keys_output, uint* values_input,
    uint* values_output, uint numElements, uint numBuckets, T bucketMapper) {

  if (numBuckets == 1) {
    for (unsigned int k = 0; k < numElements; k++) {
      keys_output[k] = keys_input[k];
      values_output[k] = values_input[k];
    }
    return;
  }

  uint logBuckets = ceil(log2(numBuckets));
  uint *bins = new uint[numBuckets]; // histogram results holder
  uint *scan_bins = new uint[numBuckets];
  uint *current_idx = new uint[numBuckets];
  // Computing histograms:
  uint bucketId;
  uint elsPerBucket = (numElements + numBuckets - 1)/numBuckets;
  uint msb_shift = 32 - logBuckets;

  for(int k = 0; k<numBuckets; k++)
    bins[k] = 0;

  for(int i = 0; i<numElements ; i++)
  {
    bucketId = bucketMapper(keys_input[i]);
    bins[bucketId]++;
  }

  // computing exclusive scan operation on the inputs:
  scan_bins[0] = 0;
  for(int j = 1; j<numBuckets; j++)
    scan_bins[j] = scan_bins[j-1] + bins[j-1];
  // Placing items in their new positions:
  for(int k = 0; k<numBuckets; k++)
    current_idx[k] = 0;

  for(int i = 0; i<numElements; i++)
  {
    bucketId = bucketMapper(keys_input[i]);
    keys_output[scan_bins[bucketId] + current_idx[bucketId]] = keys_input[i];
    values_output[scan_bins[bucketId] + current_idx[bucketId]] = values_input[i];
    current_idx[bucketId]++;
  }

  // releasing memory:
  delete[] bins;
  delete[] scan_bins;
  delete[] current_idx;
}

int verifyMultiSplit(const unsigned int *correct_keys,
    const unsigned int *computed_keys, const unsigned int *correct_values,
    const unsigned int *computed_values, size_t numElements,
    bool keyValuesOption) {
  int retVal = 0;
  unsigned int count = 0;

  for (unsigned int i = 0; i < numElements; i++) {
    if (correct_keys[i] != computed_keys[i]) {
      count++;
      printf(" ##### index %d, correct key = %d, computed key = %d\n", i,
          correct_keys[i], computed_keys[i]);
      if (count == 10) {
        printf("...\n");
        break;
      }
    }

    if (keyValuesOption) {
      if (correct_values[i] != computed_values[i]) {
        count++;
        printf(" ##### index %d, correct value = %d, computed value = %d\n", i,
            correct_values[i], computed_values[i]);
        if (count == 10) {
          printf("...\n");
          break;
        }
      }
    }
  }

  if (count)
    retVal = 1;

  return retVal;
}

int multiSplitKeysOnlyTest(CUDPPHandle theCudpp, CUDPPConfiguration config,
    size_t *elementTests, size_t *bucketTests, unsigned int numElementTests,
    unsigned int numBucketTests, size_t maxNumElements, size_t maxNumBuckets,
    testrigOptions testOptions, bool quiet) {
  int retVal = 0;
  srand(44);

  unsigned int* keys = (unsigned int*) malloc(
      sizeof(unsigned int) * maxNumElements);
  unsigned int* gpu_result_keys = (unsigned int*) malloc(
      sizeof(unsigned int) * maxNumElements);

  // an arbitrary initialization
  //for(int i = 0; i<maxNumElements; i++)
  //  elements[i] = i;
  //randomPermute(elements, maxNumElements);
  unsigned int *d_keys = NULL;
  CUDA_SAFE_CALL(
      cudaMalloc((void** ) &d_keys, maxNumElements * sizeof(unsigned int))); // gpu input (keys)

  CUDPPHandle plan;
  // allocate memory once for the maximum number of elements and buckets
  CUDPPResult result = cudppPlan(theCudpp, &plan, config, maxNumElements,
      maxNumBuckets, 0);

  if (result != CUDPP_SUCCESS) {
    printf("Error in plan creation\n");
    retVal = numElementTests;
    cudppDestroyPlan(plan);
    return retVal;
  }

  cudaEvent_t startEvent, stopEvent;
  CUDA_SAFE_CALL(cudaEventCreate(&startEvent));
  CUDA_SAFE_CALL(cudaEventCreate(&stopEvent));

  if (numElementTests == 1)
    elementTests[0] = maxNumElements;
  if (numBucketTests == 1)
    bucketTests[0] = maxNumBuckets;

  printf("Performing keys-only multisplit tests.\n");
  for (unsigned int k = 0; k < numElementTests; ++k) {
    // an arbitrary initialization
    for (unsigned int j = 0; j < maxNumElements; ++j)
      keys[j] = rand();

    for (unsigned int b = 0; b < numBucketTests; ++b) {
      int testFailed = 0;

      if (!quiet) {

        printf("Running a multi-split on %ld keys and %ld buckets\n",
            elementTests[k], bucketTests[b]);
        fflush(stdout);
      }

      float totalTime = 0;
      for (unsigned int i = 0; i < testOptions.numIterations; ++i) {
        CUDA_SAFE_CALL(
            cudaMemcpy(d_keys, keys, elementTests[k] * sizeof(unsigned int),
                cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaEventRecord(startEvent, 0));

        cudppMultiSplit(plan, d_keys, NULL, elementTests[k], bucketTests[b]);

        CUDA_SAFE_CALL(cudaEventRecord(stopEvent, 0));
        CUDA_SAFE_CALL(cudaEventSynchronize(stopEvent));

        float time = 0;
        CUDA_SAFE_CALL(cudaEventElapsedTime(&time, startEvent, stopEvent));
        totalTime += time;
      }

      CUDA_CHECK_ERROR("testMultiSplit - cudppMultiSplit");

      // copy results
      CUDA_SAFE_CALL(
          cudaMemcpy(gpu_result_keys, d_keys,
              elementTests[k] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
      // === Sanity check:
      uint count = 0;
      uint *cpu_result_keys = new uint[elementTests[k]];
      switch(config.bucket_mapper)
      {
      case CUDPP_DEFAULT_BUCKET_MAPPER:
        cpuMultisplit(keys, cpu_result_keys, elementTests[k], bucketTests[b],
            OrderedCyclicBucketMapperHost(elementTests[k], bucketTests[b]));
        break;
      case CUDPP_MSB_BUCKET_MAPPER:
        cpuMultisplit(keys, cpu_result_keys, elementTests[k], bucketTests[b],
            MSBBucketMapperHost(bucketTests[b]));
        break;
      case CUDPP_LSB_BUCKET_MAPPER:
        cpuMultisplit(keys, cpu_result_keys, elementTests[k], bucketTests[b],
            LSBBucketMapperHost(bucketTests[b]));
        break;
      default:
        cpuMultisplit(keys, cpu_result_keys, elementTests[k], bucketTests[b],
            OrderedCyclicBucketMapperHost(elementTests[k], bucketTests[b]));
        break;
      }
      testFailed = verifyMultiSplit(cpu_result_keys, gpu_result_keys, NULL,
          NULL, elementTests[k], false);
      retVal += testFailed;
      delete[] cpu_result_keys;

      if (!quiet) {
        printf("test %s\n", (testFailed == 0) ? "PASSED" : "FAILED");
        printf("Average execution time: %f ms\n",
            totalTime / testOptions.numIterations);
      } else {
        printf("\t%10ld\t%0.4f\n", elementTests[k],
            totalTime / testOptions.numIterations);
      }
    }
  }
  printf("\n");

  CUDA_CHECK_ERROR("after multi-split");

  result = cudppDestroyPlan(plan);

  if (result != CUDPP_SUCCESS) {
    printf("Error destroying CUDPPPlan for multi-split\n");
    retVal = numElementTests;
  }

  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);

  cudaFree(d_keys);
  free(keys);
  free(gpu_result_keys);

  return retVal;
}

int multiSplitKeyValueTest(CUDPPHandle theCudpp, CUDPPConfiguration config,
    size_t *elementTests, size_t *bucketTests, unsigned int numElementTests,
    unsigned int numBucketTests, size_t maxNumElements, size_t maxNumBuckets,
    testrigOptions testOptions, bool quiet) {
  int retVal = 0;
  srand(44);

  unsigned int* keys = (unsigned int*) malloc(
      sizeof(unsigned int) * maxNumElements);
  unsigned int* values = (unsigned int*) malloc(
      sizeof(unsigned int) * maxNumElements);
  unsigned int* gpu_result_keys = (unsigned int*) malloc(
      sizeof(unsigned int) * maxNumElements);
  unsigned int* gpu_result_values = (unsigned int*) malloc(
      sizeof(unsigned int) * maxNumElements);

  // an arbitrary initialization
  //for(int i = 0; i<maxNumElements; i++)
  //  elements[i] = i;
  //randomPermute(elements, maxNumElements);
  unsigned int *d_keys = NULL;
  unsigned int *d_values = NULL;
  CUDA_SAFE_CALL(
      cudaMalloc((void** ) &d_keys, maxNumElements * sizeof(unsigned int))); // gpu input (keys)
  CUDA_SAFE_CALL(
      cudaMalloc((void** ) &d_values, maxNumElements * sizeof(unsigned int))); // gpu input (keys)

  CUDPPHandle plan;
  // allocate memory once for the maximum number of elements and buckets
  CUDPPResult result = cudppPlan(theCudpp, &plan, config, maxNumElements,
      maxNumBuckets, 0);

  if (result != CUDPP_SUCCESS) {
    printf("Error in plan creation\n");
    retVal = numElementTests;
    cudppDestroyPlan(plan);
    return retVal;
  }

  cudaEvent_t startEvent, stopEvent;
  CUDA_SAFE_CALL(cudaEventCreate(&startEvent));
  CUDA_SAFE_CALL(cudaEventCreate(&stopEvent));

  if (numElementTests == 1)
    elementTests[0] = maxNumElements;
  if (numBucketTests == 1)
    bucketTests[0] = maxNumBuckets;

  printf("Performing key-value multisplit tests.\n");
  for (unsigned int k = 0; k < numElementTests; ++k) {
    // an arbitrary initialization
    for (unsigned int j = 0; j < maxNumElements; j++) {
      keys[j] = rand();
      values[j] = rand();
    }

    for (unsigned int b = 0; b < numBucketTests; ++b) {
      int testFailed = 0;

      if (!quiet) {
        printf("Running a multi-split on %ld keys and values and %ld buckets\n",
            elementTests[k], bucketTests[b]);
        fflush(stdout);
      }

      float totalTime = 0;

      for (unsigned int i = 0; i < testOptions.numIterations; ++i) {
        CUDA_SAFE_CALL(
            cudaMemcpy(d_keys, keys, elementTests[k] * sizeof(unsigned int),
                cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(
            cudaMemcpy(d_values, values, elementTests[k] * sizeof(unsigned int),
                cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaEventRecord(startEvent, 0));

        cudppMultiSplit(plan, d_keys, d_values, elementTests[k],
            bucketTests[b]);

        CUDA_SAFE_CALL(cudaEventRecord(stopEvent, 0));
        CUDA_SAFE_CALL(cudaEventSynchronize(stopEvent));

        float time = 0;
        CUDA_SAFE_CALL(cudaEventElapsedTime(&time, startEvent, stopEvent));
        totalTime += time;
      }

      CUDA_CHECK_ERROR("testMultiSplit - cudppMultiSplit");

      // copy results
      CUDA_SAFE_CALL(
          cudaMemcpy(gpu_result_keys, d_keys,
              elementTests[k] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
      CUDA_SAFE_CALL(
          cudaMemcpy(gpu_result_values, d_values,
              elementTests[k] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
      // === Sanity check:
      uint count = 0;
      uint *cpu_result_keys = new uint[elementTests[k]];
      uint *cpu_result_values = new uint[elementTests[k]];

      switch(config.bucket_mapper)
      {
      case CUDPP_DEFAULT_BUCKET_MAPPER:
        cpuMultiSplitPairs(keys, cpu_result_keys, values, cpu_result_values,
            elementTests[k], bucketTests[b],
            OrderedCyclicBucketMapperHost(elementTests[k], bucketTests[b]));
        break;
      case CUDPP_MSB_BUCKET_MAPPER:
        cpuMultiSplitPairs(keys, cpu_result_keys, values, cpu_result_values,
            elementTests[k], bucketTests[b], MSBBucketMapperHost(bucketTests[b]));
        break;
      case CUDPP_LSB_BUCKET_MAPPER:
        cpuMultiSplitPairs(keys, cpu_result_keys, values, cpu_result_values,
            elementTests[k], bucketTests[b], LSBBucketMapperHost(bucketTests[b]));
        break;
      default:
        cpuMultiSplitPairs(keys, cpu_result_keys, values, cpu_result_values,
            elementTests[k], bucketTests[b],
            OrderedCyclicBucketMapperHost(elementTests[k], bucketTests[b]));
        break;
      }
      testFailed = verifyMultiSplit(cpu_result_keys, gpu_result_keys,
          cpu_result_values, gpu_result_values, elementTests[k], true);
      retVal += testFailed;
      delete[] cpu_result_keys;
      delete[] cpu_result_values;

      if (!quiet) {
        printf("test %s\n", (testFailed == 0) ? "PASSED" : "FAILED");
        printf("Average execution time: %f ms\n",
            totalTime / testOptions.numIterations);
      } else {
        printf("\t%10ld\t%0.4f\n", elementTests[k],
            totalTime / testOptions.numIterations);
      }
    }
  }
  printf("\n");

  CUDA_CHECK_ERROR("after multi-split");

  result = cudppDestroyPlan(plan);

  if (result != CUDPP_SUCCESS) {
    printf("Error destroying CUDPPPlan for multi-split\n");
    retVal = numElementTests;
  }

  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);

  cudaFree(d_keys);
  cudaFree(d_values);
  free(keys);
  free(values);
  free(gpu_result_keys);
  free(gpu_result_values);

  return retVal;
}

/**
 * testMultiSplit tests cudpp's multisplit
 *
 * @param argc Number of arguments on the command line, passed
 * directly from main
 * @param argv Array of arguments on the command line, passed directly
 * from main
 * @param configPtr Configuration for multisplit, set by caller
 * @return Number of tests that failed regression (0 for all pass)
 * @see cudppMultiSplit
 */
int testMultiSplit(int argc, const char **argv,
    const CUDPPConfiguration *configPtr) {

  int cmdVal;
  int retVal = 0;

  bool quiet = checkCommandLineFlag(argc, argv, "quiet");
  testrigOptions testOptions;
  setOptions(argc, argv, testOptions);

  CUDPPConfiguration config;
  config.algorithm = CUDPP_MULTISPLIT;
  config.datatype = CUDPP_UINT;
  config.options = CUDPP_OPTION_KEYS_ONLY;
  config.bucket_mapper = CUDPP_DEFAULT_BUCKET_MAPPER;
  if (configPtr != NULL) {
    config = *configPtr;
  }

  // The last test size should be the largest
  size_t elementTests[] = {262144, 2097152, 4194304, 8388608, 16777216 };
  size_t bucketTests[] =
      { 1, 2, 3, 13, 32, 33, 63, 83, 97, 100, 112, 129, 145 };

  int numElementTests = sizeof(elementTests) / sizeof(elementTests[0]);
  int numBucketTests = sizeof(bucketTests) / sizeof(bucketTests[0]);

  // small GPUs are susceptible to running out of memory,
  // restrict the tests to only those where we have enough
  size_t freeMem, totalMem;
  CUDA_SAFE_CALL(cudaMemGetInfo(&freeMem, &totalMem));
  printf("freeMem: %lu, totalMem: %lu\n", freeMem, totalMem);

  while (freeMem < 175 * elementTests[numElementTests - 1]) // 175B/item appears to be enough
  {
    numElementTests--;
    if (numElementTests <= 0) {
      // something has gone very wrong
      printf("Not enough free memory to run any multisplit tests.\n");
      return -1;
    }
  }

  size_t maxNumElements = elementTests[numElementTests - 1];
  size_t maxNumBuckets = bucketTests[numBucketTests - 1];

  if (commandLineArg(cmdVal, argc, (const char**) argv, "n")) {
    maxNumElements = cmdVal;
    numElementTests = 1;
  }

  if (commandLineArg(cmdVal, argc, (const char**) argv, "b")) {
    maxNumBuckets = cmdVal;
    numBucketTests = 1;
  }

  CUDPPResult result = CUDPP_SUCCESS;
  CUDPPHandle theCudpp;
  result = cudppCreate(&theCudpp);

  if (result != CUDPP_SUCCESS) {
    printf("Error initializing CUDPP Library.\n");
    retVal = numElementTests;
    return retVal;
  }

  retVal += multiSplitKeysOnlyTest(theCudpp, config, elementTests, bucketTests,
      numElementTests, numBucketTests, maxNumElements, maxNumBuckets, testOptions,
      quiet);

  config.options = CUDPP_OPTION_KEY_VALUE_PAIRS;
  retVal += multiSplitKeyValueTest(theCudpp, config, elementTests, bucketTests,
      numElementTests, numBucketTests, maxNumElements, maxNumBuckets, testOptions,
      quiet);
  result = cudppDestroy(theCudpp);

  return retVal;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

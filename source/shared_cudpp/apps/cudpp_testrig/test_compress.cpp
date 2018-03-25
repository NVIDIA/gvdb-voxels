// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: $
// $Date: $
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// -------------------------------------------------------------

/**
 * @file
 * test_compress.cpp
 *
 * @brief Host testrig routines to exercise cudpp's compress functionality.
 */

#include <cstring>
#include <iostream>
#include <cuda_runtime_api.h>
#include <time.h>

#include "cudpp.h"
#include "sparse.h"
#include "cudpp_testrig_options.h"
#include "cudpp_testrig_utils.h"
#include "cuda_util.h"
#include "stopwatch.h"
#include "commandline.h"
#include "comparearrays.h"

#define NONE    -1
#define NUM_CHARS           257     /* 256 bytes + EOF */
#define EOF_CHAR    (NUM_CHARS - 1) /* index used for EOF */
#define COMPOSITE_NODE      -1      /* node represents multiple characters */
#define THREADS_PER_BLOCK 128
#define WORK_PER_THREAD 32
#define BLOCK_CHARS (THREADS_PER_BLOCK*WORK_PER_THREAD)

typedef struct my_huffman_node_t
{
    int value;          /* character(s) represented by this entry */
    unsigned int count;      /* number of occurrences of value (probability) */

    char ignore;        /* TRUE -> already handled or no need to handle */
    int level;          /* depth in tree (root is 0) */
    unsigned int iter;
    int left, right, parent;
} my_huffman_node_t;

using namespace cudpp_app;

#define Wrap(value, limit) (((value) < (limit)) ? (value) : ((value) - (limit)))

int FindMinimumCountTest(my_huffman_node_t* ht, int elements)
{
    int i;                          // array index
    int currentIndex = NONE;        // index with lowest count seen so far
    int currentCount = INT_MAX;     // lowest count seen so far
    int currentLevel = INT_MAX;     // level of lowest count seen so far

    // sequentially search array
    for (i = 0; i < elements; i++)
    {
        // check for lowest count (or equally as low, but not as deep)
        if( (!ht[i].ignore) &&
            (ht[i].count < (unsigned int)currentCount ||
             (ht[i].count == currentCount && ht[i].level < currentLevel)) )
        {
            currentIndex = i;
            currentCount = ht[i].count;
            currentLevel = ht[i].level;
        }
    }

    return currentIndex;
}

void computeBwtGold(unsigned char *i_data, unsigned char *reference,
                    int &ref_index, unsigned int numElements)
{
    unsigned int* sa = new unsigned int[numElements+3];
    computeSaGold(i_data, sa, numElements);
    for(int i=0; i<numElements; i++)
    {
        unsigned int val=sa[i];
        if(val==0) ref_index = i;
        reference[i] = (val==0) ? i_data[numElements-1] : i_data[val-1];
    }
    delete [] sa;
}

void computeMtfGold( unsigned char* out, const unsigned char* idata,
                     const unsigned int len)
{
    unsigned char* list = new unsigned char[256];
    unsigned int j = 0;

    // init mtf list
    for(unsigned int i=0; i<256; i++)
        list[i] = i;

    for (unsigned int i = 0; i < len; i++)
    {
        // Find the character in the list of characters
        for (j = 0; j < 256; j++)
        {
            if (list[j] == idata[i])
            {
                // Found the character
                out[i] = j;
                break;
            }
        }

        // Move the current character to the front of the list
        for (; j > 0; j--)
        {
            list[j] = list[j - 1];
        }
        list[0] = idata[i];
    }

    delete [] list;
}

void huffman_build_tree_cpu(my_huffman_node_t* tree, unsigned int nNodes,
                            int &head)
{
    int min1, min2;     // two nodes with the lowest count

    // keep looking until no more nodes can be found
    for (;;)
    {
        // find node with lowest count
        min1 = FindMinimumCountTest(&tree[0], nNodes);
        if (min1 == NONE) break; // No more nodes to combine

        tree[min1].ignore = 1;    // remove from consideration

        // find node with second lowest count
        min2 = FindMinimumCountTest(&tree[0], nNodes);
        if (min2 == NONE) break; // No more nodes to combine

        // Move min1 to the next available slots
        tree[min1].ignore = 0;
        unsigned char min1_replacement = 0;

        for(int i = (int)nNodes; i<(NUM_CHARS*2-1); i++)
        {
            if(tree[i].count==0)
            {
                tree[i] = tree[min1];
                tree[i].iter = (unsigned int)i;
                tree[i].ignore = 1;
                tree[i].parent = tree[min1].iter;

                if(tree[i].left >= 0)
                    tree[tree[i].left].parent = i;
                if(tree[i].right >= 0)
                    tree[tree[i].right].parent = i;

                tree[min1].left = i;

                min1_replacement = 1;
                break;
            }
        }

        if(min1_replacement == 0)
        {
            printf("ERROR: Tree size too small\n");
            break;
        }

        tree[min2].ignore = 1;

        // Combines both nodes into composite node
        tree[min1].value = COMPOSITE_NODE;
        tree[min1].ignore = 0;
        tree[min1].count = tree[min1].count + tree[min2].count;
        tree[min1].level = max(tree[min1].level, tree[min2].level) + 1;

        tree[min1].right = tree[min2].iter;
        tree[min2].parent =  tree[min1].iter;
        tree[min1].parent = -1;
    }

    head = min1;
}

void computeCompressGold(unsigned char* reference,
                         int h_bwtIndex,
                         unsigned int* h_hist,
                         unsigned int* h_encodeOffset,
                         size_t h_compressedSize,
                         unsigned int* h_compressed,
                         size_t numElements)
{
    // Host tree
    my_huffman_node_t* h_huffmanArray = new my_huffman_node_t[NUM_CHARS*2-1];
    int head;

    // Set all iterations of the tree
    unsigned int nNodes = 0;
    for(int j=0; j<NUM_CHARS; j++) {
        h_huffmanArray[j].iter = (unsigned int)j;
        h_huffmanArray[j].value = j;
        h_huffmanArray[j].ignore = true;      // will be FALSE if one is found
        h_huffmanArray[j].count = 0;
        h_huffmanArray[j].level = 0;
        h_huffmanArray[j].left = -1;
        h_huffmanArray[j].right = -1;
        h_huffmanArray[j].parent = -1;
    }
    for(int j=NUM_CHARS; j<NUM_CHARS*2-1; j++) {
        h_huffmanArray[j].iter = (unsigned int)j;
        h_huffmanArray[j].value = 0;
        h_huffmanArray[j].ignore = true;      // will be FALSE if one is found
        h_huffmanArray[j].count = 0;
        h_huffmanArray[j].level = 0;
        h_huffmanArray[j].left = -1;
        h_huffmanArray[j].right = -1;
        h_huffmanArray[j].parent = -1;
    }

    h_hist[EOF_CHAR] = 1;
    for(int j=0; j<(NUM_CHARS); j++)
    {
        if(h_hist[j] > 0)
        {
            h_huffmanArray[nNodes].count = h_hist[j];
            h_huffmanArray[nNodes].ignore = 0;
            h_huffmanArray[nNodes].value = j;
            nNodes++;
        }
    }
    huffman_build_tree_cpu(h_huffmanArray, nNodes, head);

    for(int i=0; i<256; i++)
    {
        unsigned int my_offset = h_encodeOffset[i];
        unsigned char buffer = 32;
        unsigned int section = 1+my_offset;
        unsigned int n_found_chars = 0;
        unsigned char c=0;
        unsigned int encoded_val = h_compressed[section];

        int currentNode = head;

        while(n_found_chars < BLOCK_CHARS)
        {
            buffer--;
            c = (encoded_val >> buffer) & 0x00000001;

            if(buffer == 0) {
                buffer = 32;
                section++;
                encoded_val = h_compressed[section];
            }

            // traverse the tree finding matches for our characters
            if (c != 0)
            {
                currentNode = h_huffmanArray[currentNode].right;
            }
            else
            {
                currentNode = h_huffmanArray[currentNode].left;
            }

            if (h_huffmanArray[currentNode].value != COMPOSITE_NODE)
            {
                // we've found a character
                if (h_huffmanArray[currentNode].value == EOF_CHAR)
                {
                    // we've just read the EOF
                    currentNode = head;
                    break;
                }

                // write out character
                reference[i*BLOCK_CHARS+n_found_chars] =
                    (unsigned char)h_huffmanArray[currentNode].value;
                n_found_chars++;

                // back to top of tree
                currentNode = head;
            }
        }

    }

    //Reverse  MTF
    unsigned char* mtfOut = new unsigned char[numElements];
    unsigned char* mtfList = new unsigned char[256];
    for(unsigned int i=0; i<256; i++) {
        mtfList[i] = (unsigned char)i;
    }
    for (unsigned int i = 0; i < numElements; i++)
    {
        // decode the character
        unsigned char tmp = reference[i];
        mtfOut[i] = mtfList[tmp];
        // now move the current character to the front of the list
        for (unsigned char j = tmp; j > 0; j--)
        {
            mtfList[j] = mtfList[j - 1];
        }
        mtfList[0] = mtfOut[i];
    }

    // BWT
    CUDPPHandle theCudpp;
    CUDPPConfiguration config;
    CUDPPHandle plan;
    cudppCreate(&theCudpp);
    config.algorithm = CUDPP_SORT_RADIX;
    config.datatype = CUDPP_UCHAR;
    config.options = CUDPP_OPTION_KEY_VALUE_PAIRS;
    cudppPlan(theCudpp, &plan, config, numElements, 1, 0);

    unsigned char* d_keys;
    unsigned int* d_values;
    unsigned int* h_values = new unsigned int[numElements];
    for(unsigned int i=0; i<numElements; i++) {
        h_values[i] = i;
    }


    CUDA_SAFE_CALL( cudaMalloc( (void **) &d_keys,
                                numElements*sizeof(unsigned char)));
    CUDA_SAFE_CALL( cudaMalloc( (void **) &d_values,
                                numElements*sizeof(unsigned int)));
    CUDA_SAFE_CALL( cudaMemcpy(d_keys, mtfOut,
                               numElements*sizeof(unsigned char),
                               cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(d_values, h_values,
                               numElements*sizeof(unsigned int),
                               cudaMemcpyHostToDevice) );


    // sort
    cudppRadixSort(plan, (void*)d_keys, (void*)d_values, numElements);

    // Decode final BWT
    cudaMemcpy( h_values, d_values, numElements*sizeof(unsigned int),
                cudaMemcpyDeviceToHost);


    for(unsigned int i=0; i<numElements; i++) {
        h_bwtIndex = h_values[h_bwtIndex];
        reference[i] = mtfOut[h_bwtIndex];
    }
    // Free
    delete [] h_huffmanArray;
    delete [] mtfOut;
    delete [] mtfList;
    delete [] h_values;
    cudaFree(d_keys);
    cudaFree(d_values);
    cudppDestroyPlan(plan);
    cudppDestroy(theCudpp);
}

int mtfTest(int argc, const char **argv, const CUDPPConfiguration &config,
            const testrigOptions &testOptions)
{
    int retval = 0;

    cudpp_app::StopWatch timer;

    bool quiet = checkCommandLineFlag(argc, (const char**)argv, "quiet");

    unsigned int test[] = {39, 128, 256, 512, 1000, 1024, 1025, 32768, 45537,
                           65536, 131072, 262144, 500001, 524288, 1048577,
                           1048576, 1048581};
    int numTests = sizeof(test) / sizeof(test[0]);
    if (testOptions.skiplongtests)
    {
        numTests -= 3;          // leave out last 3 tests, they may time out
    }
    int numElements = test[numTests-1]; // maximum test size

    bool oneTest = false;
    if (commandLineArg(numElements, argc, (const char**) argv, "n"))
    {
        oneTest = true;
        numTests = 1;
        test[0] = numElements;
    }

    CUDPPResult result = CUDPP_SUCCESS;
    CUDPPHandle theCudpp;
    result = cudppCreate(&theCudpp);
    if (result != CUDPP_SUCCESS)
    {
        if (!quiet)
            fprintf(stderr, "Error initializing CUDPP library.\n");
        retval = (oneTest) ? 1 : numTests;
        return retval;
    }

    CUDPPHandle plan;
    result = cudppPlan(theCudpp, &plan, config, numElements, 1, 0);

    if (result != CUDPP_SUCCESS)
    {
        if (!quiet)
            fprintf(stderr, "Error creating plan for MTF\n");
        retval = (oneTest) ? 1 : numTests;
        return retval;
    }

    unsigned int memSize = sizeof(unsigned char) * numElements;

    // allocate host memory to store the input data
    unsigned char* i_data = new unsigned char[numElements];
    unsigned char* reference = new unsigned char[numElements];

    // allocate device memory input and output arrays
    unsigned char* d_idata     = (unsigned char *) NULL;
    unsigned char* d_odata     = (unsigned char *) NULL;

    CUDA_SAFE_CALL( cudaMalloc( (void **) &d_idata, memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void **) &d_odata, memSize));

    for (int k = 0; k < numTests; ++k)
    {
        if (!quiet)
        {
            printf("Running a MTF test of %u %s nodes\n",
                test[k],
                datatypeToString(config.datatype));
            fflush(stdout);
        }

        // initialize the input data on the host
         srand(95835);
         for(int j=0; j<test[k]; ++j)
                     i_data[j] = (unsigned char)(rand()%255+1);

        memset(reference, 0, sizeof(unsigned char) * test[k]);
        computeMtfGold( reference, i_data, test[k]);

        CUDA_SAFE_CALL( cudaMemcpy(d_idata, i_data,
                                   sizeof(unsigned char) * test[k],
                                   cudaMemcpyHostToDevice) );
        CUDA_SAFE_CALL( cudaMemset(d_odata, 0,
                                   sizeof(unsigned char) * test[k]) );

        // run once to avoid timing startup overhead.
        cudppMoveToFrontTransform(plan, d_idata, d_odata, test[k]);

        timer.reset();
        timer.start();
        for (int i = 0; i < testOptions.numIterations; i++)
        {
            cudppMoveToFrontTransform(plan, d_idata, d_odata, test[k]);
        }
        cudaThreadSynchronize();
        timer.stop();

        // allocate host memory to store the output data
        unsigned char* o_data = (unsigned char*) malloc( sizeof(unsigned char) *
                                                         test[k]);
        CUDA_SAFE_CALL(cudaMemcpy( o_data, d_odata,
                                   sizeof(unsigned char) * test[k],
                                   cudaMemcpyDeviceToHost));

        bool result = compareArrays<unsigned char>( reference, o_data, test[k]);

        free(o_data);

        retval += result ? 0 : 1;
        if (!quiet)
        {
            printf("MTF test %s\n", result ? "PASSED" : "FAILED");
        }
        if (!quiet)
        {
            printf("Average execution time: %f ms\n",
                timer.getTime() / testOptions.numIterations);
        }
        else
            printf("\t%10d\t%0.4f\n", test[k],
                   timer.getTime() / testOptions.numIterations);
    }

    result = cudppDestroyPlan(plan);
    if (result != CUDPP_SUCCESS)
    {
        if (!quiet)
            printf("Error destroying CUDPPPlan for MTF\n");
    }

    result = cudppDestroy(theCudpp);
    if (result != CUDPP_SUCCESS)
    {
        if (!quiet)
            printf("Error shutting down CUDPP Library.\n");
    }

    delete [] reference;
    delete [] i_data;
    cudaFree(d_odata);
    cudaFree(d_idata);
    return retval;
}

int bwtTest(int argc, const char **argv, const CUDPPConfiguration &config,
            const testrigOptions &testOptions)
{
    int retval = 0;
    int numElements = 1048576; // test size
    bool quiet = checkCommandLineFlag(argc, argv, "quiet");
    int numTests = 1;
    bool oneTest = true;

    // Initialize CUDPP
    CUDPPHandle plan;
    CUDPPResult result = CUDPP_SUCCESS;
    CUDPPHandle theCudpp;
    result = cudppCreate(&theCudpp);
    if (result != CUDPP_SUCCESS)
    {
        fprintf(stderr, "Error initializing CUDPP library\n");
        retval = 1;
        return retval;
    }

    result = cudppPlan(theCudpp, &plan, config, numElements, 1, 0);

    if(result != CUDPP_SUCCESS)
    {
        printf("Error in plan creation\n");
        retval = numTests;
        cudppDestroyPlan(plan);
        cudppDestroy(theCudpp);
        return retval;
    }

    unsigned int memSize = sizeof(unsigned char) * numElements;

    // allocate host memory to store the input data
    unsigned char* i_data = new unsigned char[numElements];

    // initialize the input data on the host
    float range = (float)(sizeof(unsigned char)*8);

    //VectorSupport<unsigned char>::fillVector(i_data, numElements, range);
    srand(95835);
    for(int j = 0; j < numElements; j++)
    {
        i_data[j] = (unsigned char)(rand()%255+1);
    }

    unsigned char* reference = new unsigned char[numElements];
    int ref_index;

    // allocate device memory input and output arrays
    unsigned char* d_idata      = (unsigned char *) NULL;
    unsigned char* d_odata      = (unsigned char *) NULL;
    int* d_oindex               = (int *) NULL;

    CUDA_SAFE_CALL( cudaMalloc( (void **) &d_idata, memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void **) &d_odata, memSize));
    CUDA_SAFE_CALL( cudaMalloc( (void **) &d_oindex, sizeof(int)));

    CUDA_SAFE_CALL( cudaMemcpy(d_idata, i_data, memSize,
                               cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemset(d_odata, 0, memSize) );

    char dt[10];
    strcpy(dt, "uchar");

    if (!quiet)
    {
        printf("Running a BWT of %d %s elements\n",
               numElements, dt);
        fflush(stdout);
    }

    computeBwtGold(i_data,reference, ref_index, numElements);

    // Run the BWT
    // run once to avoid timing startup overhead.
    result = cudppBurrowsWheelerTransform(plan, d_idata, d_odata, d_oindex,
                                          (unsigned int)numElements);

    if (result != CUDPP_SUCCESS)
    {
        if(!quiet)
            printf("Error destroying cudppBurrowsWheelerTransform for BWT\n");
        retval = numTests;
    }

    // copy result from device to host
    unsigned char* o_data = new unsigned char[numElements];
    int o_index;
    CUDA_SAFE_CALL(cudaMemcpy( o_data, d_odata, memSize,
                               cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy( &o_index, d_oindex, sizeof(int),
                               cudaMemcpyDeviceToHost));

    // check results
    bool error = false;
    for(int i=0; i<numElements; i++)
    {
        if(o_data[i] != reference[i])
        {
            error = true;
            retval = 1;
            break;
        }
    }
    if(o_index != ref_index) {
        error = true;
        retval = 1;
    }

    printf("BWT test %s\n", (error) ? "FAILED" : "PASSED");

    result = cudppDestroyPlan(plan);

    if (result != CUDPP_SUCCESS)
    {
        printf("Error destroying CUDPPPlan for BWT\n");
        retval = numTests;
    }

    result = cudppDestroy(theCudpp);

    if (result != CUDPP_SUCCESS)
    {
        printf("Error shutting down CUDPP library.\n");
        retval = numTests;
    }

    delete [] reference;
    delete [] o_data;
    delete [] i_data;
    cudaFree(d_odata);
    cudaFree(d_idata);
    cudaFree(d_oindex);

    return retval;
}

int compressTest(int argc, const char **argv, const CUDPPConfiguration &config,
                 const testrigOptions &testOptions)
{
    int retval = 0;
    int numElements = 1048576; // test size

    bool quiet = checkCommandLineFlag(argc, argv, "quiet");
    int numTests = 1;
    bool oneTest = true;

    // Initialize CUDPP
    CUDPPHandle plan;
    CUDPPResult result = CUDPP_SUCCESS;
    CUDPPHandle theCudpp;
    result = cudppCreate(&theCudpp);
    if (result != CUDPP_SUCCESS)
    {
        fprintf(stderr, "Error initializing CUDPP library\n");
        retval = 1;
        return retval;
    }

    result = cudppPlan(theCudpp, &plan, config, numElements, 1, 0);

    if(result != CUDPP_SUCCESS)
    {
        printf("Error in plan creation\n");
        retval = numTests;
        cudppDestroyPlan(plan);
        cudppDestroy(theCudpp);
        return retval;
    }

    // allocate host memory to store the input data
    unsigned char* i_data = new unsigned char[numElements];

    // initialize the input data on the host
    srand(95835);
    for(int j = 0; j < numElements-1; j++)
    {
        i_data[j] = (unsigned char)(rand()%255+1);
    }
    i_data[numElements-1]=(unsigned char)0;
    // host ptrs
    int h_bwtIndex;
    unsigned int* h_hist = new unsigned int[NUM_CHARS];
    unsigned int* h_encodeOffset = new unsigned int[256];
    size_t        h_compressedSize = 0;
    unsigned char* reference = new unsigned char[numElements];

    // allocate device memory input and output arrays
    unsigned char  *d_uncompressed;         // user provides
    int            *d_bwtIndex;             // sizeof(int)
    unsigned int   *d_histSize;             // ignored
    unsigned int   *d_hist;                 // 256*sizeof(uint)
    unsigned int   *d_encodeOffset;         // 256*sizeof(uint)
    unsigned int   *d_compressedSize;       // sizeof(uint)
    unsigned int   *d_compressed;           // d_compressedSize*sizeof(uint)

    CUDA_SAFE_CALL(cudaMalloc( (void**)&d_uncompressed,
                               numElements*sizeof(unsigned char) ));
    CUDA_SAFE_CALL(cudaMalloc( (void**)&d_bwtIndex, sizeof(int) ));
    CUDA_SAFE_CALL(cudaMalloc( (void**)&d_hist, 256*sizeof(unsigned int) ));
    CUDA_SAFE_CALL(cudaMalloc( (void**)&d_encodeOffset,
                               256*sizeof(unsigned int) ));
    CUDA_SAFE_CALL(cudaMalloc( (void**)&d_compressedSize,
                               sizeof(unsigned int) ));
    CUDA_SAFE_CALL(cudaMalloc( (void**)&d_compressed,
                               (1536+1)*256*sizeof(unsigned int) ));
    d_histSize = (unsigned int*)NULL;

    CUDA_SAFE_CALL(cudaMemcpy(d_uncompressed, i_data,
                              numElements*sizeof(unsigned char),
                              cudaMemcpyHostToDevice));

    char dt[10];
    strcpy(dt, "uchar");

    if (!quiet)
    {
        printf("Running a compress of %d %s elements\n",
               numElements, dt);
        fflush(stdout);
    }

    // clear buffers
    CUDA_SAFE_CALL(cudaMemset( (void*)d_encodeOffset, 0,
                               256*sizeof(unsigned int) ));
    CUDA_SAFE_CALL(cudaMemset( (void*)d_compressedSize, 0,
                               sizeof(unsigned int) ));
    CUDA_SAFE_CALL(cudaMemset( (void*)d_compressed,  0,
                               (1536+1)*256*sizeof(unsigned int) ));
    // Run the compression
    // run once to avoid timing startup overhead.
    result = cudppCompress(plan, d_uncompressed, d_bwtIndex,
                           d_histSize, d_hist,
                           d_encodeOffset, d_compressedSize,
                           d_compressed, numElements);

    if (result != CUDPP_SUCCESS)
    {
        if (!quiet)
            printf("Error calling cudppCompress for compression\n");
        retval = numTests;
    } else {
        // Copy from device back to host
        CUDA_SAFE_CALL(cudaMemcpy(&h_bwtIndex, d_bwtIndex, sizeof(int),
                                  cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(h_hist, d_hist, 256*sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(h_encodeOffset, d_encodeOffset,
                                  256*sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(&h_compressedSize, d_compressedSize,
                                  sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost));

        unsigned int* h_compressed = new unsigned int[h_compressedSize];

        CUDA_SAFE_CALL(cudaMemcpy(h_compressed, d_compressed,
                                  h_compressedSize*sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost));

        // Decompress on the CPU
        computeCompressGold( reference, h_bwtIndex, h_hist, h_encodeOffset,
                             h_compressedSize, h_compressed, numElements);

        delete [] h_compressed;
    }

    // check results
    int errorCount = 0;

    for (int i = 0; i < h_compressedSize; i++)
    {
        if (i_data[i] != reference[i])
        {
            errorCount++;
            printf("Found compress error on char %d (%c, should be %c)\n",
                   i, i_data[i], reference[i]);
            retval = 1;
            if (errorCount > 5)
            {
                printf("Stopping after 5 errors.\n");
                break;
            }
        }
    }
    printf("Compress test %s\n", errorCount ? "FAILED" : "PASSED");


    result = cudppDestroyPlan(plan);

    if (result != CUDPP_SUCCESS)
    {
        printf("Error destroying CUDPPPlan for compress\n");
        retval = numTests;
    }

    result = cudppDestroy(theCudpp);

    if (result != CUDPP_SUCCESS)
    {
        printf("Error shutting down CUDPP library.\n");
        retval = numTests;
    }

    delete [] reference;
    delete [] h_hist;
    delete [] h_encodeOffset;
    delete [] i_data;
    CUDA_SAFE_CALL(cudaFree(d_hist));
    CUDA_SAFE_CALL(cudaFree(d_encodeOffset));
    CUDA_SAFE_CALL(cudaFree(d_compressedSize));
    CUDA_SAFE_CALL(cudaFree(d_bwtIndex));
    CUDA_SAFE_CALL(cudaFree(d_compressed));
    return retval;
}

int testMtf(int argc, const char **argv, const CUDPPConfiguration *configPtr)
{
    testrigOptions testOptions;
    setOptions(argc, argv, testOptions);

    CUDPPConfiguration config;
    config.algorithm = CUDPP_MTF;
    config.options = 0;

    if (configPtr != NULL)
    {
        config = *configPtr;
    }
    else
    {
        config.datatype = CUDPP_UCHAR;
    }

    return mtfTest(argc, argv, config, testOptions);
}


int testBwt(int argc, const char **argv, const CUDPPConfiguration *configPtr)
{
    testrigOptions testOptions;
    setOptions(argc, argv, testOptions);

    CUDPPConfiguration config;
    config.algorithm = CUDPP_BWT;
    config.options = 0;

    if (configPtr != NULL)
    {
        config = *configPtr;
    }
    else
    {
        config.datatype = CUDPP_UCHAR;
    }

    return bwtTest(argc, argv, config, testOptions);
}

int testCompress(int argc, const char **argv,
                 const CUDPPConfiguration *configPtr)
{
    testrigOptions testOptions;
    setOptions(argc, argv, testOptions);

    CUDPPConfiguration config;
    config.algorithm = CUDPP_COMPRESS;
    config.options = 0;

    if (configPtr != NULL)
    {
        config = *configPtr;
    }
    else
    {
        config.datatype = CUDPP_UCHAR;
    }

    return compressTest(argc, argv, config, testOptions);
}


// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: $
// $Date: $
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// -------------------------------------------------------------

/*! @file hash_testrig.cu
 *  @brief This file demonstrates how to use all three hash tables in
 *  the CUDPP hash table distribution.
 */

#include <cudpp_hash.h>
#include <cuda_util.h>
#include <mt19937ar.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdio>
#include <map>
#include <set>
#include <vector>
#include <string.h>             // memcpy

#include "random_numbers.h"
#define CUDPP_APP_COMMON_IMPL
#include "stopwatch.h"
#include "commandline.h"


using namespace cudpp_app;

int CheckResults_basic(const unsigned            kInputSize,
                       const std::map<unsigned, unsigned> &pairs,
                       const unsigned           *query_keys,
                       const unsigned           *query_vals)
{
    int errors = 0;
    for (unsigned i = 0; i < kInputSize; ++i)
    {
        // @TODO is this right?
        unsigned actual_value = CUDPP_HASH_KEY_NOT_FOUND;
        std::map<unsigned, unsigned>::const_iterator it =
            pairs.find(query_keys[i]);
        if (it != pairs.end())
        {
            actual_value = it->second;
        }

        if (actual_value != query_vals[i])
        {
            errors++;
#ifdef _DEBUG
            printf("\t\t\tError for key %10u: Actual value is "
                   "%10u, but hash returned %10u.\n",
                   query_keys[i], actual_value, query_vals[i]);
#endif
        }
    }
    return errors;
}

int CheckResults_compacting(const unsigned            kInputSize,
                            const std::set<unsigned> &pairs,
                            const unsigned           *query_keys,
                            const unsigned           *query_vals)
{
    int errors = 0;
    std::map<unsigned, unsigned> id_to_key_map;
    std::map<unsigned, unsigned> key_to_id_map;
    std::vector<unsigned> ids;
    ids.reserve(kInputSize);

    for (unsigned j = 0; j < kInputSize; ++j)
    {
        // Confirm that all valid queries get a valid ID back, and
        // that bad queries get an invalid one.
        if (pairs.find(query_keys[j]) != pairs.end())
        {
            if (query_vals[j] >= kInputSize)
            {
#ifdef _DEBUG
                fprintf(stderr,
                        "\t\t\t\t!!! Valid query returned bad ID: %10u %10u\n",
                        query_keys[j], query_vals[j]);
#endif
                errors++;
            }
        }
        else
        {
            if (query_vals[j] < kInputSize)
            {
#ifdef _DEBUG
                fprintf(stderr,
                        "\t\t\t\t!!! Invalid query returned good ID: "
                        "%10u %10u\n", query_keys[j], query_vals[j]);
#endif
                errors++;
            }
        }

        // Track which unique IDs were returned.
        if (query_vals[j] != CUDPP_HASH_KEY_NOT_FOUND)
        {
            ids.push_back(query_vals[j]);
        }

        // Track which keys mapped to which unique IDs.
        if (pairs.find(query_keys[j]) != pairs.end())
        {
            // Make sure all copies of a key get the same ID back.
            if (key_to_id_map.find(query_keys[j]) != key_to_id_map.end())
            {
                if (key_to_id_map[query_keys[j]] != query_vals[j])
                {
#ifdef _DEBUG
                    fprintf(stderr,
                            "\t\t\t\t!!! Key %10u had two IDs: %10u %10u\n",
                            query_keys[j],
                            key_to_id_map[query_keys[j]],
                            query_vals[j]);
#endif
                    errors++;
                }
            }

            // Make sure all copies of the same ID have the same key.
            if (id_to_key_map.find(query_vals[j]) != id_to_key_map.end())
            {
                if (id_to_key_map[query_vals[j]] != query_keys[j])
                {
#ifdef _DEBUG
                    fprintf(stderr,
                            "\t\t\t\t!!! ID %10u had two keys: %10u %10u\n",
                            query_vals[j],
                            id_to_key_map[query_vals[j]],
                            query_keys[j]);
#endif
                    errors++;
                }
            }

            key_to_id_map[query_keys[j]] = query_vals[j];
            id_to_key_map[query_vals[j]] = query_keys[j];
        }
    }

    std::sort(ids.begin(), ids.end());
    if (ids.size()>0 && ids.back() >= pairs.size())
    {
        fprintf(stderr, "\t\t\t\t!!! Biggest ID >= number of input items\n");
        errors++;
    }

    if (key_to_id_map.size() != id_to_key_map.size())
    {
        fprintf(stderr, "\t\t\t\t!!! Number of unique IDs doesn't match the "
                "number of input items in the query set\n");
        errors++;
    }

    for (std::map<unsigned, unsigned>::iterator itr = key_to_id_map.begin();
         itr != key_to_id_map.end(); ++itr)
    {
        unsigned current_key    = itr->first;
        unsigned expected_value = itr->second;
        if (id_to_key_map[expected_value] != current_key)
        {
            fprintf(stderr,
                    "\t\t\t\t!!! Translation mismatch: %u has ID %u, but ID "
                    "is mapped to %u\n",
                    current_key, expected_value,
                    id_to_key_map[expected_value]);
            errors++;
        }
    }
    return errors;
}

int CheckResults_multivalue(const unsigned            kInputSize,
                            const std::map<unsigned, std::vector<unsigned> >
                            &pairs,
                            const unsigned           *sorted_values,
                            const unsigned           *query_keys,
                            const uint2              *query_vals_multivalue)
{
    int errors = 0;
    for (unsigned i = 0; i < kInputSize; ++i) {
        std::map<unsigned, std::vector<unsigned> >::const_iterator itr =
            pairs.find(query_keys[i]);
        if (itr != pairs.end()) {
            // The query key was part of the input. Confirm that
            // the number of values is right.
            if (query_vals_multivalue[i].y != itr->second.size()) {
                fprintf(stderr,
                        "\t\t\t\t!!! Input query key %10u returned %u "
                        "values instead of %u.\n",
                        query_keys[i],
                        query_vals_multivalue[i].y,
                        (unsigned int) itr->second.size());
                errors++;
            }

            // Confirm that all of the values can be found.
            std::vector<unsigned> hash_table_values;
            for (unsigned j = 0; j < query_vals_multivalue[i].y; ++j) {
                hash_table_values.push_back(
                    sorted_values[query_vals_multivalue[i].x + j]);
            }
            std::sort(hash_table_values.begin(), hash_table_values.end());

            for (unsigned j = 0; j < query_vals_multivalue[i].y; ++j) {
                if (hash_table_values[j] != itr->second[j]) {
                    fprintf(stderr,
                            "\t\t\t\t!!! Values didn't match: %10u != %10u\n",
                            hash_table_values[j],
                            itr->second[j]);
                    errors++;
                }
            }
        } else {
            // The query key was not part of the input. Confirm
            // that there are 0 values.
            if (query_vals_multivalue[i].y != 0) {
                fprintf(stderr,
                        "\t\t\t\t!!! Invalid query key %10u has %u values "
                        "instead of 0.\n",
                        query_keys[i],
                        query_vals_multivalue[i].y);
                errors++;
            }
        }
    }
    return errors;
}

/*
 * @brief Tests hash table implementations.
 *
 * <b>Simple hash table sample</b>
 *
 * This program builds a basic hash table using N random key-value
 * pairs with unique keys, then queries it for N unique keys, where
 * the queries are comprised of keys both inside the hash table and
 * not in the hash table. Multiple copies of the hash table are built
 * for each trial, where each hash table has a different number of
 * slots.
 *
 * After the construction of each hash table, it is queried multiple
 * times with a different set of keys. Each query key set is composed
 * of a portion of the original input keys (which can be found in the
 * hash table), and keys that were not part of the original input
 * (which cause the queries to fail).
 *
 * <b>Compacting hash table example</b>
 *
 * This builds a compacting hash table, using N random keys. Multiple
 * copies of a key in the input are all given the same unique ID by
 * the hash table. In addition to the trials performed by the simple
 * hash table sample, it also performs multiple trials with an
 * increasing average number of copies for each key: the compacting
 * hash table is always handed N keys, but it is possible that many
 * keys will have a large number of copies.
 *
 * <b>Multi-value hash table example</b>
 *
 * This builds a multi-value hash table, using N random key-value
 * pairs. A key with multiple values is represented by multiple
 * key-value pairs in the input with the same key.
 */
int testHashTable(CUDPPHandle theCudpp,
                  CUDPPHashTableType htt,
                  unsigned int kMaxIterations,
                  unsigned int kInputSize,
                  unsigned int * number_pool,
                  unsigned int pool_size,
                  unsigned int * input_vals,
                  unsigned int * input_keys,
                  unsigned int * d_test_vals,
                  uint2 *        d_test_vals_multivalue,
                  unsigned int * d_test_keys,
                  unsigned int * query_vals,
                  uint2 *        query_vals_multivalue,
                  unsigned int * query_keys,
                  unsigned int   skipEveryNTests,
                  bool           skipHighMultiplicities,
                  unsigned int & testNumber)
{
    int total_errors = 0;
    for (unsigned iteration = 0; iteration < kMaxIterations; ++iteration)
    {
        switch(htt)
        {
        case CUDPP_BASIC_HASH_TABLE:
            printf("Basic hash table: ");
            break;
        case CUDPP_COMPACTING_HASH_TABLE:
            printf("Compacting hash table: ");
            break;
        case CUDPP_MULTIVALUE_HASH_TABLE:
            printf("Multivalue hash table: ");
            break;
        default:
            printf("INVALID hash table: ");
            break;
        }

        printf("Iteration %u\n", iteration);

        unsigned int multiplicity_max = 1; // loops only once by default
        switch(htt)
        {
        case CUDPP_COMPACTING_HASH_TABLE:
        case CUDPP_MULTIVALUE_HASH_TABLE:
            if (skipHighMultiplicities)
            {
                multiplicity_max = 256;
            }
            else
            {
                multiplicity_max = 2048;
            }
            break;
        default:
            break;
        }

        for (unsigned multiplicity = 1;
             multiplicity <= multiplicity_max;
             multiplicity *= 2)
        {
            float chance_of_repeating = 1.0f - 1.0f/multiplicity;
            if (multiplicity_max != 1)
            {
                printf("\tAverage multiplicity of keys: %u\n", multiplicity);
            }

            // Generate random data.
            GenerateUniqueRandomNumbers(number_pool, pool_size);
            switch(htt)
            {
            case CUDPP_BASIC_HASH_TABLE:
            case CUDPP_MULTIVALUE_HASH_TABLE:
                for (unsigned i = 0; i < kInputSize; ++i)
                {
                    input_vals[i] = genrand_int32();
                }
                if (htt == CUDPP_BASIC_HASH_TABLE)
                {
                    break;
                }
                // otherwise fall through (MULTIVALUE)
            case CUDPP_COMPACTING_HASH_TABLE:
                GenerateMultiples(kInputSize, chance_of_repeating, number_pool);
                GenerateMultiples(kInputSize, chance_of_repeating,
                                  number_pool + kInputSize);
                Shuffle(kInputSize, number_pool);
                Shuffle(kInputSize, number_pool + kInputSize);
                break;
                break;
            default:
                fprintf(stderr, "Bad CUDPPHashTableType (htt) in "
                        "testHashTable\n");
                break;
            }

            // The unique numbers are pre-shuffled by the generator.
            // Take the first half as the input keys.
            memcpy(input_keys, number_pool, sizeof(unsigned) * kInputSize);

            // Save the original input for checking the results.
            std::map<unsigned, unsigned> pairs_basic;
            std::set<unsigned> pairs_compacting;
            std::map<unsigned, std::vector<unsigned> > pairs_multivalue;
            switch(htt)
            {
            case CUDPP_BASIC_HASH_TABLE:
                for (unsigned i = 0; i < kInputSize; ++i)
                {
                    pairs_basic[input_keys[i]] = input_vals[i];
                }
                break;
            case CUDPP_COMPACTING_HASH_TABLE:
                for (unsigned i = 0; i < kInputSize; ++i)
                {
                    pairs_compacting.insert(input_keys[i]);
                }
                break;
            case CUDPP_MULTIVALUE_HASH_TABLE:
                for (unsigned i = 0; i < kInputSize; ++i) {
                    pairs_multivalue[input_keys[i]].push_back(input_vals[i]);
                }
                for (std::map<unsigned, std::vector<unsigned> >::iterator itr =
                         pairs_multivalue.begin();
                     itr != pairs_multivalue.end();
                     itr++)
                {
                    std::sort(itr->second.begin(), itr->second.end());
                }
                break;
            default:
                fprintf(stderr, "Bad CUDPPHashTableType (htt) in "
                        "testHashTable\n");
                break;
            }


            const float kSpaceUsagesToTest[] = {1.05f, 1.15f, 1.25f, 1.5f,
                                                2.0f};
            const unsigned kNumSpaceUsagesToTest = 5;

            for (unsigned i = 0; i < kNumSpaceUsagesToTest; ++i)
            {

                if ((testNumber++ % skipEveryNTests) != 0)
                {
                    continue;
                }

                float space_usage = kSpaceUsagesToTest[i];
                printf("\tSpace usage: %f\n", space_usage);

                CUDA_SAFE_CALL(cudaMemcpy(d_test_keys, input_keys,
                                          sizeof(unsigned) * kInputSize,
                                          cudaMemcpyHostToDevice));
                switch(htt)
                {
                case CUDPP_BASIC_HASH_TABLE:
                case CUDPP_MULTIVALUE_HASH_TABLE:
                    CUDA_SAFE_CALL(cudaMemcpy(d_test_vals, input_vals,
                                              sizeof(unsigned) * kInputSize,
                                              cudaMemcpyHostToDevice));
                    break;
                case CUDPP_COMPACTING_HASH_TABLE:
                    break;
                default:
                    fprintf(stderr, "Bad CUDPPHashTableType (htt) in "
                            "testHashTable\n");
                    break;
                }



                /// -------------------- Create and build the basic hash table.
                CUDPPHashTableConfig config;
                config.type = htt;
                config.kInputSize = kInputSize;
                config.space_usage = space_usage;
                CUDPPHandle hash_table_handle;
                CUDPPResult result;
                result = cudppHashTable(theCudpp, &hash_table_handle, &config);
                if (result != CUDPP_SUCCESS)
                {
                    fprintf(stderr, "Error in cudppHashTable call in"
                            "testHashTable (make sure your device is at"
                            "least compute version 2.0\n");
                }

                cudpp_app::StopWatch timer;
                timer.reset();
                timer.start();

                result = cudppHashInsert(hash_table_handle, d_test_keys,
                                         d_test_vals, kInputSize);
                cudaThreadSynchronize();
                timer.stop();
                if (result != CUDPP_SUCCESS)
                {
                    fprintf(stderr, "Error in cudppHashInsert call in"
                            "testHashTable\n");
                }
                printf("\tHash table build: %f ms\n", timer.getTime());
                /// -----------------------------------------------------------

                unsigned *sorted_values = NULL;
                if (htt == CUDPP_MULTIVALUE_HASH_TABLE)
                {
                    unsigned int values_size;
                    if (cudppMultivalueHashGetValuesSize(hash_table_handle,
                                                         &values_size) !=
                        CUDPP_SUCCESS)
                    {
                        fprintf(stderr, "Error: "
                                "cudppMultivalueHashGetValuesSize()\n");
                    }
                    sorted_values = new unsigned[values_size];
                    unsigned int * d_all_values = NULL;
                    if (cudppMultivalueHashGetAllValues(hash_table_handle,
                                                        &d_all_values) !=
                        CUDPP_SUCCESS)
                    {
                        fprintf(stderr, "Error: "
                                "cudppMultivalueHashGetAllValues()\n");
                    }

                    CUDA_SAFE_CALL(cudaMemcpy(sorted_values,
                                              d_all_values,
                                              sizeof(unsigned) * values_size,
                                              cudaMemcpyDeviceToHost));
                }

                unsigned int failure_trials = 10;
                for (unsigned failure = 0; failure <= failure_trials; ++failure)
                {
                    // Generate a set of queries comprised of keys both
                    // from and not from the input.
                    float failure_rate = failure / (float) failure_trials;
                    GenerateQueries(kInputSize, failure_rate, number_pool,
                                    query_keys);
                    CUDA_SAFE_CALL(cudaMemcpy(d_test_keys, query_keys,
                                              sizeof(unsigned) * kInputSize,
                                              cudaMemcpyHostToDevice));
                    CUDA_SAFE_CALL(cudaMemset(d_test_vals, 0,
                                              sizeof(unsigned) * kInputSize));

                    /// --------------------------------------- Query the table.
                    timer.reset();
                    timer.start();
                    /// hash_table.Retrieve(kInputSize, d_test_keys,
                    //                      d_test_vals);

                    unsigned int errors = 0;
                    switch(htt)
                    {
                    case CUDPP_BASIC_HASH_TABLE:
                    case CUDPP_COMPACTING_HASH_TABLE:
                        result = cudppHashRetrieve(hash_table_handle,
                                                   d_test_keys, d_test_vals,
                                                   kInputSize);
                        break;
                    case CUDPP_MULTIVALUE_HASH_TABLE:
                        result = cudppHashRetrieve(hash_table_handle,
                                                   d_test_keys,
                                                   d_test_vals_multivalue,
                                                   kInputSize);
                        break;
                    default:
                        errors++;
                        fprintf(stderr, "Bad CUDPPHashTableType (htt) in "
                                "testHashTable\n");
                        break;
                    }
                    cudaThreadSynchronize();
                    timer.stop();
                    if (result != CUDPP_SUCCESS)
                    {
                        fprintf(stderr, "Error in cudppHashRetrieve call in"
                                "testHashTable\n");
                    }
                    printf("\tHash table retrieve with %3u%% chance of "
                           "failed queries: %f ms\n", failure * failure_trials,
                           timer.getTime());
                    /// --------------------------------------------------------

                    // Check the results.
                    switch(htt)
                    {
                    case CUDPP_BASIC_HASH_TABLE:
                        CUDA_SAFE_CALL(cudaMemcpy(query_vals, d_test_vals,
                                                  sizeof(unsigned) * kInputSize,
                                                  cudaMemcpyDeviceToHost));
                        errors +=
                            CheckResults_basic(kInputSize,
                                               pairs_basic,
                                               query_keys,
                                               query_vals);
                        break;
                    case CUDPP_COMPACTING_HASH_TABLE:
                        CUDA_SAFE_CALL(cudaMemcpy(query_vals, d_test_vals,
                                                  sizeof(unsigned) * kInputSize,
                                                  cudaMemcpyDeviceToHost));
                        errors +=
                            CheckResults_compacting(kInputSize,
                                                    pairs_compacting,
                                                    query_keys,
                                                    query_vals);
                        break;
                    case CUDPP_MULTIVALUE_HASH_TABLE:
                        CUDA_SAFE_CALL(cudaMemcpy(query_vals_multivalue,
                                                  d_test_vals_multivalue,
                                                  sizeof(uint2) * kInputSize,
                                                  cudaMemcpyDeviceToHost));
                        errors +=
                            CheckResults_multivalue(kInputSize,
                                                    pairs_multivalue,
                                                    sorted_values,
                                                    query_keys,
                                                    query_vals_multivalue);
                        break;
                    default:
                        errors++;
                        fprintf(stderr, "Bad CUDPPHashTableType (htt) in "
                                "testHashTable\n");
                        break;
                    }

                    if (errors > 0)
                    {
                        printf("%d errors found\n", errors);
                    }
                    else
                    {
                        printf("No errors found, test passes\n");
                    }
                    total_errors += errors;
                }

                delete [] sorted_values;
                /// -------------------------------------------- Free the table.
                result = cudppDestroyHashTable(theCudpp, hash_table_handle);
                if (result != CUDPP_SUCCESS)
                {
                    fprintf(stderr, "Error in cudppDestroyHashTable call in"
                            "testHashTable\n");
                }

                /// hash_table.Release();
                /// ------------------------------------------------------------
            }
        }
    }
    return total_errors;
}


/**
 * main in hash_testrig is a dispatch routine to exercise cudpp hash
 * table functionality.
 *
 * - -all calls every regression routine (-basic, -compacting, -multivalue)
 * - -basic calls basic_hash_table
 * - -compacting calls compacting_hash_table
 * - -multivalue calls multivalue_hash_table
 * - -n=# sets the size of the dataset (default 1M)
 * - -iterations=# sets the number of iterations to run (default 1)
 * - -skip=# runs only every #th test (default 7)
 * - -skiphighx=<true,false> skips the tests with high multiplicity
 *      (since they take a *long* time to test) (default true)
 */
int main(int argc, const char **argv)
{
    bool quiet = checkCommandLineFlag(argc, argv, "quiet");

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0)
    {
        fprintf(stderr, "Error (main): no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    commandLineArg(dev, argc, argv, "device");
    if (dev < 0) dev = 0;
    if (dev > deviceCount-1) dev = deviceCount - 1;
    cudaSetDevice(dev);

    cudaDeviceProp prop;
    if (!quiet && cudaGetDeviceProperties(&prop, dev) == 0)
    {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %uB; compute v%d.%d; clock: %d kHz\n",
               prop.name, (unsigned int)prop.totalGlobalMem, (int)prop.major,
               (int)prop.minor, (int)prop.clockRate);
    }

    if (prop.major < 2)
    {
        fprintf(stderr, "ERROR: CUDPP hash tables are only supported on "
                "devices with compute\n  capability 2.0 or greater; "
                "exiting.\n");
        exit(1);
    }

    int retval = 0;

    init_genrand(543289423);
    //init_genrand(time(NULL));

    bool runAll = checkCommandLineFlag(argc, argv, "all");
    bool runBasicHash = runAll || checkCommandLineFlag(argc, argv, "basic");
    bool runCompactingHash =
        runAll || checkCommandLineFlag(argc, argv, "compacting");
    bool runMultivalueHash =
        runAll || checkCommandLineFlag(argc, argv, "multivalue");

    unsigned kInputSize = 1000000;
    commandLineArg(kInputSize, argc, argv, "n");

    unsigned kMaxIterations = 1;
    commandLineArg(kMaxIterations, argc, argv, "iterations");

    unsigned skipEveryNTests = 7;
    commandLineArg(skipEveryNTests, argc, argv, "skip");

    bool skipHighMultiplicities = true;
    commandLineArg(skipHighMultiplicities, argc, argv, "skiphighx");

    /// Allocate memory.
    /* We will need a pool of random numbers to create test input and queries
     *   from:
     * we need N input keys and N query keys to produce failed queries.
     */
    const unsigned pool_size = kInputSize * 2;
    unsigned *number_pool = new unsigned[pool_size];

    // Set aside memory for input keys and values.
    unsigned *input_keys = new unsigned[kInputSize];
    unsigned *input_vals = new unsigned[kInputSize];
    unsigned *query_keys = new unsigned[kInputSize];
    unsigned *query_vals = new unsigned[kInputSize];
    uint2 *query_vals_multivalue = new uint2[kInputSize];

    // Allocate the GPU memory.
    unsigned *d_test_keys = NULL, *d_test_vals = NULL;
    uint2 *d_test_vals_multivalue = NULL;
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_test_keys,
                              sizeof(unsigned) * kInputSize));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_test_vals,
                              sizeof(unsigned) * kInputSize));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_test_vals_multivalue,
                              sizeof(uint2) * kInputSize));

    CUDPPHandle theCudpp;
    CUDPPResult result = cudppCreate(&theCudpp);
    if (result != CUDPP_SUCCESS)
    {
        if (!quiet)
            fprintf(stderr, "Error initializing CUDPP Library.\n");
        retval = 1;
        return retval;
    }

    int total_errors = 0;
    unsigned int testNumber = 0;
    for (CUDPPHashTableType htt = CUDPP_BASIC_HASH_TABLE;
         htt != CUDPP_INVALID_HASH_TABLE;
         htt = (CUDPPHashTableType)((unsigned)htt+1))
    {
        if (runAll ||
            ((htt == CUDPP_BASIC_HASH_TABLE) && runBasicHash) ||
            ((htt == CUDPP_COMPACTING_HASH_TABLE) && runCompactingHash) ||
            ((htt == CUDPP_MULTIVALUE_HASH_TABLE) && runMultivalueHash))
        {
            total_errors +=
                testHashTable(theCudpp,
                              htt,
                              kMaxIterations,
                              kInputSize,
                              number_pool,
                              pool_size,
                              input_vals,
                              input_keys,
                              d_test_vals,
                              d_test_vals_multivalue,
                              d_test_keys,
                              query_vals,
                              query_vals_multivalue,
                              query_keys,
                              skipEveryNTests,
                              skipHighMultiplicities,
                              testNumber);
        }
    }
    if (total_errors == 0)
    {
        printf("All tests pass.\n");
    }
    else
    {
        printf("Some tests failed.\n");
    }


    CUDA_SAFE_CALL(cudaFree(d_test_keys));
    CUDA_SAFE_CALL(cudaFree(d_test_vals));
    CUDA_SAFE_CALL(cudaFree(d_test_vals_multivalue));
    delete [] number_pool;
    delete [] input_keys;
    delete [] input_vals;
    delete [] query_keys;
    delete [] query_vals;
    delete [] query_vals_multivalue;

    result = cudppDestroy(theCudpp);
    if (result != CUDPP_SUCCESS)
    {
        if (!quiet)
        {
            printf("Error shutting down CUDPP Library.\n");
        }
    }

    return retval;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

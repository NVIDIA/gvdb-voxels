// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: $
// $Date: $
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// ------------------------------------------------------------- 

#include <algorithm>
#include <cstdio>
#include <string.h>             // memcpy
#include <mt19937ar.h>

void Shuffle(const unsigned  num_random_numbers,
                   unsigned *random_numbers) {
  // Fisher-Yates shuffle the unique numbers.
  for (unsigned index_1 = 0; index_1 < num_random_numbers; ++index_1) {
    unsigned num_left = num_random_numbers - index_1;
    unsigned index_2  = index_1 + (genrand_int32() % num_left);
    std::swap(random_numbers[index_1], random_numbers[index_2]);
  }
}

bool GenerateUniqueRandomNumbers(unsigned       *random_numbers,
                                 const unsigned  num_random_numbers,
                                 const unsigned  max_number) {
  if (random_numbers == NULL) {
    return false;
  }

  // Generate a certain percentage extra of random numbers as a cushion.
  unsigned  num_numbers  = (unsigned)(num_random_numbers * 1.1);
  unsigned *temp_numbers = new unsigned[num_numbers];
  if (temp_numbers == NULL) {
    fprintf(stderr, "Failed to allocate space.\n");
    return false;
  }

  unsigned num_unique = 0;
  while (num_unique < num_random_numbers) {
    // Generate numbers.
    for (unsigned i = num_unique; i < num_numbers; ++i) {
      do {
        temp_numbers[i] = genrand_int32();
      } while (temp_numbers[i] > max_number);
    }

    // Sort to put all copies of the same number next to each other.
    // TODO(dfalcantara): A faster sort would speed this up considerably, but I don't want to introduce more dependencies.
    std::sort(temp_numbers, temp_numbers + num_numbers);

    // Determine which are unique & replace with new numbers.
    num_unique = 1;
    for (unsigned i = 1; i < num_numbers; ++i) {
      if (temp_numbers[i-1] == temp_numbers[i]) {
        temp_numbers[i-1] = max_number;
      } else {
        num_unique++;
      }
    }

    // Move all of the non-unique numbers to the end.
    std::sort(temp_numbers, temp_numbers + num_numbers);
  }

  // Shuffle all of the unique keys.
  Shuffle(num_unique, temp_numbers);
  
  // Copy the number of keys requested & toss the rest.
  memcpy(random_numbers, temp_numbers, sizeof(unsigned) * num_random_numbers);
  delete [] temp_numbers;

  return true;
}                           


unsigned GenerateMultiples(const unsigned  num_random_numbers,
                                 float     chance_of_repeating,
                                 unsigned *random_numbers) {
  unsigned num_unique = 1;                           
  for (unsigned i = 1; i < num_random_numbers; ++i) {
    if (genrand_real1() < chance_of_repeating) {
      random_numbers[i] = random_numbers[i-1];
    } else {
      num_unique++;
    }
  }
  printf("Unique keys: %u / %u\n", num_unique, num_random_numbers);
  return num_unique;
}


void GenerateQueries(const unsigned  size,
                     const float     failure_rate,
                           unsigned *number_pool,
                           unsigned *queries) {
  unsigned num_failed_queries = (unsigned)(failure_rate * size);
  unsigned num_good_queries   = size - num_failed_queries;

 /// Pick some of the input keys as queries.
  if (num_good_queries) {
    Shuffle(size, number_pool);
    memcpy(queries, number_pool, sizeof(unsigned) * num_good_queries);
  }

 /// Pick some of the non-input keys as queries.
  if (num_failed_queries) {
    Shuffle(size, number_pool + size);
    memcpy(queries + num_good_queries, number_pool + size, sizeof(unsigned) * num_failed_queries);
  }

 /// Shuffle them all together.
  Shuffle(size, queries);
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

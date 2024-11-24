#include <cuda_runtime_api.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <tuple>
#include <utility>
#include <numeric>
#include <iomanip>

// Declare a GPU-visible unsigned long long variable in global memory.
__device__ unsigned long long dResult;

/*
   The simplest reduction operation: every threads atomically adds to the global value.
   atomicAdd is a CUDA built-in function
*/
__global__ void reduceAtomicGlobal(const unsigned long long* __restrict input, unsigned long long N)
{
    const unsigned long long id = threadIdx.x + blockIdx.x * blockDim.x;
    /* 
    Since all blocks must have the same number of threads,
    we may have to launch more threads than there are 
    inputs. Superfluous threads should not try to read 
    from the input (out of bounds access!)
    */
    if (id < N)
        atomicAdd(&dResult, input[id]);
}

/*
   Suggested improvement #1: use shared memory. 
  
   Try to use a __shared__ variable (shared only within the block) to accumulate updates from each thread block.
   Only update the global variable (dResult) at the end of each block's run.
*/
__global__ void reduceAtomicShared(const unsigned long long* __restrict input, unsigned long long N)
{
    const unsigned long long id = threadIdx.x + blockIdx.x * blockDim.x;

    // Declare a shared var for each block
    __shared__ unsigned long long x;

    // Only one thread should initialize this shared value
    if (threadIdx.x == 0) 
        x = 0.0f;
    
    /*
    Before we continue, we must ensure that all threads
    can see this update (initialization) by thread 0
    */
    __syncthreads();

    /*
    Every thread in the block adds its input to the
    shared variable of the block.
    */
    if (id < N) 
        atomicAdd(&x, input[id]);

    // Wait until all threads have done their part
    __syncthreads();

    /*
    Once they are all done, only one thread must add
    the block's partial result to the global variable. 
    */
    if (threadIdx.x == 0) 
        atomicAdd(&dResult, x);
}

/*
 Suggested improvement #2 (May be challenging, but doable!): Using a better algorithm.
 
 Notice that in improvement #1, all threads in a block now contend for the shared variable.
 While this is better than a global variable, it is still not ideal.

 We can use a much better parallel algorithm to reduce contention. 
 See the idea from Lecture 1, Slide 28 ("Better Parallel Algorithm")

 Consider using a shared __array__ of long long values, one for each thread in the block.
 We can compute the parallel result over multiple iterations 
 	(since compute instructions are much faster than memory accesses).

 In each iteration, each thread should accumulate a partial result from the previous iteration. 
 	Not all threads have to be involved in each iteration, the number of threads involved will reduce each time.
 This will make the most sense if you see the appropriate slide in the lecture notes 
 	Instead of each node being a core, now each node is a GPU thread.

 Similarly, in the end, some thread in the block should atomically update dResult

 With this improvement, you should get an algorithm that is closer to O(log N) instead of O(N).

*/
template <int BLOCK_SIZE>
__global__ void reduceShared(const unsigned long long* __restrict input, unsigned long long N)
{
    const unsigned long long id = threadIdx.x + blockIdx.x * blockDim.x;

    /*
       One shared variable for each thread to accumulate a partial sum with less contention.
    */
    __shared__ unsigned long long data[BLOCK_SIZE];

    /*
    Use a new strategy to handle superfluous threads.
    To make sure they stay alive and can help with
    the reduction, threads without an input simply
    produce a '0', which has no effect on the result.
    */
    data[threadIdx.x] = (id < N ? input[id] : 0);

    /*
    log N iterations to complete. In each step, a thread
    accumulates two partial values to form the input for
    the next iteration. The sum of all partial results 
    eventually yields the full result of the reduction. 
    */
    for (unsigned long long s = blockDim.x / 2; s > 0; s /= 2)
    {
        /*
        In each iteration, we must make sure that all
        threads are done writing the updates of the
        previous iteration / the initialization.
        */
        __syncthreads();
        if (threadIdx.x < s)
            data[threadIdx.x] += data[threadIdx.x + s];
    }

    /*
    Note: thread 0 is the last thread to combine two
    partial results, and the one who writes to global
    memory, therefore no synchronization is required
    after the last iteration.
    */
    if (threadIdx.x == 0)
        atomicAdd(&dResult, data[0]);
}

// You do not need to change this
__host__ void prepareArrayCPUGPU(unsigned long long N, std::vector<unsigned long long>& vals, unsigned long long** dValsPtr)
{
    constexpr unsigned long long target = 42;
    std::cout << "\nExpected value: " << target * N << "\n" << std::endl;

    // Generate
    vals.resize(N);
    // There are better ways to do this but this allows for random numbers in the future 
    std::for_each(vals.begin(), vals.end(), [](unsigned long long& f) { f = target; });

    // Allocate some global GPU memory to write the inputs to
    cudaMalloc((void**)dValsPtr, sizeof(unsigned long long) * N);
    // Expliclity copy the inputs from the CPU to the GPU
    cudaMemcpy(*dValsPtr, vals.data(), sizeof(unsigned long long) * N, cudaMemcpyHostToDevice);
}

int main()
{
    /*
     Expected output: Accumulated results from CPU and GPU that equals 42 * NUM_ITEMS 
    */

    constexpr unsigned long long BLOCK_SIZE = 256;
    constexpr unsigned long long WARMUP_ITERATIONS = 10;
    constexpr unsigned long long TIMING_ITERATIONS = 20;
    constexpr unsigned long long N = 10'000'000;

    // Create input arrays in CPU and GPU
    std::cout << "Producing input array...\n\n";
    std::vector<unsigned long long> vals;
    unsigned long long* dValsPtr;
    prepareArrayCPUGPU(N, vals, &dValsPtr);

    std::cout << "==== CPU Reduction ====\n" << std::endl;
    // A reference value is computed by sequential reduction
    unsigned long long referenceResult = std::accumulate(vals.cbegin(), vals.cend(), 0ll);
    std::cout << "Computed CPU value: " << referenceResult << std::endl;

    std::cout << "\n==== GPU Reductions ====\n" << std::endl;
    /*
     Set up a collection of reductions to evaluate for performance. 
     Each entry gives a technique's name, the kernel to call, and
     the number of threads required for each individual technique.
    */
    const std::tuple<const char*, void(*)(const unsigned long long*, unsigned long long), unsigned long long> reductionTechniques[]
    {
        {"Atomic Global", reduceAtomicGlobal, N},
        {"Atomic Shared", reduceAtomicShared, N},
        {"Reduce Shared", reduceShared<BLOCK_SIZE>, N},
	// TODO: add any new functions you want to test here
    };

    // Evaluate each technique separately
    for (const auto& [name, func, numThreads] : reductionTechniques)
    {
        // Compute the smallest grid to start required threads with a given block size
        const dim3 blockDim = { BLOCK_SIZE, 1, 1 };
        const dim3 gridDim = { (numThreads + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1 };

        // Run several reductions for GPU to warm up
        for (unsigned long long i = 0; i < WARMUP_ITERATIONS; i++)
            func<<<gridDim, blockDim>>>(dValsPtr, N);

        // Synchronize to ensure CPU only records time after warmup is done
        cudaDeviceSynchronize();
        const auto before = std::chrono::system_clock::now();

        unsigned long long result = 0.0f;
        // Run several iterations to get an average measurement
        for (unsigned long long i = 0; i < TIMING_ITERATIONS; i++)
        {
            // Reset acummulated result to 0 in each run
            cudaMemcpyToSymbol(dResult, &result, sizeof(unsigned long long));
            func<<<gridDim, blockDim>>>(dValsPtr, N);
        }

        // cudaMemcpyFromSymbol will implicitly synchronize CPU and GPU
        cudaMemcpyFromSymbol(&result, dResult, sizeof(unsigned long long));

        // Can measure time without an extra synchronization
        const auto after = std::chrono::system_clock::now();
        const auto elapsed = 1000.f * std::chrono::duration_cast<std::chrono::duration<float>>(after - before).count();
	const auto status = result == referenceResult ? "OK" : "FAILED";
        std::cout << std::setw(20) << name << "\t" << elapsed / TIMING_ITERATIONS << "ms \t" << std::setw(10) << result  << "\t" << status << std::endl;
    }

    // Free the allocated memory for input
    cudaFree(dValsPtr);
    return 0;
}


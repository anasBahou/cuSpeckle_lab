#include <string>
#include <chrono>
#include <ctime>
#include <cstring>
#include <cstdio>
#include <sys/time.h>
#include <iostream>
#include <unistd.h>
#include <assert.h>
#include <stdlib.h>
#include "speckle_gen.h"

#include <helper_cuda.h>
#include <cuda_runtime.h>

#include <cuda.h>
#include <curand_kernel.h>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CUDA_CALL(x) do { if((x) != cudaSuccess) {printf("Error at %s:%d\n",__FILE__,__LINE__); return EXIT_FAILURE;}} while(0)

/*
delta estimation function to calculate a uniform upper bound of the Frobenius norm of the Jacobian matrix J of the inverse displacement field U
@dims : image dimensions
*/
float estimate_delta_mesh(vec2D<int> dims, float *disp_map_x, float *disp_map_y)
{
    const int width = dims.x;
    const int height = dims.y;
    const int size = width * height;
    float *dX, *dY;
    dX = (float *)malloc(size * sizeof(float));
    dY = (float *)malloc(size * sizeof(float));

    float cur_disp_x, cur_disp_y; //current displacements for each region
    float fx = 0, fy = 0;         //actual displacements for each pixel
    for (int x = 1; x < height + 1; ++x)
    {
        for (int y = 1; y < width + 1; ++y)
        {
            cur_disp_x = disp_map_x[(y - 1) + (x - 1) * width];
            cur_disp_y = disp_map_y[(y - 1) + (x - 1) * width];
            fx = x + cur_disp_x;
            fy = y + cur_disp_y;
            dX[(x - 1) + (y - 1) * width] = fx - x;
            dY[(x - 1) + (y - 1) * width] = fy - y;
        }
    }

    float *sqF;
    sqF = (float *)malloc((width - 1) * (height - 1) * sizeof(float));
    float max = -INFINITY;
    float sqF_r_c = 0;
    // Matlab equivalent ==>  diff(dX(:,2:end),1,1).^2 + diff(dY(:,2:end),1,1).^2 + diff(dX(2:end,:),1,2).^2 + diff(dY(2:end,:),1,2).^2;
    for (int row = 0; row < height - 1; ++row)
    {
        for (int column = 0; column < width - 1; ++column)
        {
            sqF_r_c = pow(dX[(column + 1) + row * width] - dX[(column + 1) + (row + 1) * width], 2);
            sqF_r_c += pow(dY[(column + 1) + row * width] - dY[(column + 1) + (row + 1) * width], 2);
            sqF_r_c += pow(dX[column + (row + 1) * width] - dX[(column + 1) + (row + 1) * width], 2);
            sqF_r_c += pow(dY[column + (row + 1) * width] - dY[(column + 1) + (row + 1) * width], 2);
            sqF[column + row * (width - 1)] = sqF_r_c;
            max = (sqF[column + row * (width - 1)] > max) ? sqF[column + row * (width - 1)] : max; // calculate max
        }
    }

    free(dX);
    free(dY);
    free(sqF);

    return (float)(sqrt(max));
}

/** From cuda_samples/MC_EstimatePiP
 * @brief Calculate the sum within the block
 * 
 * @param in 
 * @param cta 
 * @return __device__ 
 */

__device__ unsigned int reduce_sum_2(unsigned int in, cg::thread_block cta)
{
    extern __shared__ unsigned int sdata[];

    // Perform first level of reduction:
    // - Write to shared memory
    unsigned int ltid = threadIdx.x;

    sdata[ltid] = in;
    cg::sync(cta);

    // Do reduction in shared mem
    for (unsigned int s = blockDim.x / 2 ; s > 0 ; s >>= 1)
    {
        if (ltid < s)
        {
            sdata[ltid] += sdata[ltid + s];
        }

        cg::sync(cta);
    }

    return sdata[0];
}

__global__ void setup_kernel_2(unsigned int seed, curandStatePhilox4_32_10_t *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(seed, id, 0, &state[id]);
}


__global__ void compute_intensity_kernel_float(curandStatePhilox4_32_10_t *state,
                                        int samples,
                                        unsigned int *result, int x, int y, Random_disk* disk_arr, int rc_size, float cur_disp_x, float cur_disp_y)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    unsigned int bid = blockIdx.x;

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    const int max_itr_per_thread = samples/(blockDim.x * gridDim.x);
    const int r_itr_per_thread = samples%(blockDim.x * gridDim.x);
    unsigned int intensity = 0;
    float2 rand_var; // OR double2
    Random_disk disk; // tmp disk
    /* Copy state to local memory for efficiency */
    curandStatePhilox4_32_10_t localState = state[id];
    /* Run MC simulation */
    for(int i = 0; i < max_itr_per_thread; ++i) {
        rand_var = curand_normal2(&localState);
        float fx = 0, fy = 0;
        fx = x + cur_disp_x + rand_var.x;
        fy = y + cur_disp_y + rand_var.y;

        // check if phi(x+Xm) belongs to one of the disks
        for (unsigned int k = 0; k < rc_size; ++k)
        {
            disk = disk_arr[k];
            float d1 =disk.x - fx;
            float d2 = disk.y - fy;
            float dist = d1 * d1 + d2 * d2;
            if (dist < disk.r)
            {
                ++intensity;
                break;
            }
        }
    }

    // carry on the ramaining simulations on the first 'r_itr_per_thread' threads (in case (float)'max_itr_per_thread'=not int)
    if (id < r_itr_per_thread){                
        rand_var = curand_normal2(&localState);
        float fx = 0, fy = 0;
        fx = x + cur_disp_x + rand_var.x;
        fy = y + cur_disp_y + rand_var.y;

        // check if phi(x+Xm) belongs to one of the disks
        for (unsigned int k = 0; k < rc_size; ++k)
        {
            disk = disk_arr[k];
            float d1 =disk.x - fx;
            float d2 = disk.y - fy;
            float dist = d1 * d1 + d2 * d2;
            if (dist < disk.r)
            {
                ++intensity;
                break;
            }
        }
    
    }

    /* Copy state back to global memory */
    state[id] = localState;
    // /* Store results */
    // result[id] += intensity;

    // Reduce within the block
    intensity = reduce_sum_2(intensity, cta);

    // Store the result
    if (threadIdx.x == 0)
    {
        result[bid] = intensity;
    }
}


int monte_carlo_estimation_mesh(float *speckle_matrix, float *Random_centers, float *Random_radius, float *RBound, int number, unsigned int seed, int width, int height, float alpha, int nbit, float gamma, int N0, float* disp_map_x, float* disp_map_y)
{

    //----- cuda Threads/Blocks setup variables(preparation) ----- ///
    struct cudaDeviceProp     deviceProperties;
    unsigned int device = gpuGetMaxGflopsDeviceId();

    // Get device properties
    cudaGetDeviceProperties(&deviceProperties, device);

    // Determine how to divide the work between cores
    dim3 block;
    dim3 grid;
    block.x = (unsigned int)deviceProperties.maxThreadsDim[0];
    grid.x  = (unsigned int)deviceProperties.maxGridSize[0];

    // Aim to launch around ten or more times as many blocks as there
    // are multiprocessors on the target device.
    unsigned int blocksPerSM = 4; // check ==> https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html
    unsigned int numSMs      = deviceProperties.multiProcessorCount;

    // make sure to use maximum grid size(nb blocks per SM) 
    while (grid.x > 2 * blocksPerSM * numSMs)
    {
        grid.x >>= 1;
    }

    // define nb of Threads per block staticly
    block.x = 64;

    const unsigned int totalThreads = block.x * grid.x;

    // Print Threads info
    printf("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n");
    printf("Nb Blocks = %d\n", grid.x);
    printf("Nb Threads Per Block = %d\n", block.x);
    printf("Total nb_threads = %d\n", totalThreads);
    // ------------------------------- ///

    // calculation variables
    unsigned int total;
    curandStatePhilox4_32_10_t *devPHILOXStates;
    unsigned int *devResults, *hostResults;

    // define a constant variable needed to calculate NMC
    const float cst_var = (float)2 / pi * gamma * gamma * pow(2, 2 * nbit) / (alpha * alpha);

    /* Allocate space for results on host */
    hostResults = (unsigned int *)calloc(grid.x, sizeof(unsigned int));

    /* Allocate space for results on device */
    CUDA_CALL(cudaMalloc((void **)&devResults, grid.x * sizeof(unsigned int)));

    CUDA_CALL(cudaMalloc((void **)&devPHILOXStates, totalThreads * sizeof(curandStatePhilox4_32_10_t)));

    /* Allocate memory for RC & RR */
    Random_disk *disk_arr;
    CUDA_CALL(cudaMallocManaged((void **)&disk_arr, number * sizeof(*disk_arr)));

    /* Compacting boolean model parameters into one array of structures */
    Boolean_model_disk *BM_disk_arr;
    BM_disk_arr = (Boolean_model_disk *)calloc(number, sizeof(Boolean_model_disk));

    Boolean_model_disk tmp;
    for (int i = 0; i < number; ++i) // do the compaction
    {
        tmp.disk.x = Random_centers[2 * i];
        tmp.disk.y = Random_centers[2 * i + 1];
        tmp.disk.r = Random_radius[i] * Random_radius[i];
        tmp.rbound = RBound[i];
        BM_disk_arr[i] = tmp;
    }
    
    /* Setup prng states */
    setup_kernel_2<<<grid, block>>>(seed, devPHILOXStates);

    // utility var 
    int count = 0;
    float dist;

    // //define regions and displacement per region
    // const unsigned int region_size_x = width/nb_regions_x;
    // const unsigned int region_size_y = height/nb_regions_y;

    float cur_disp_x, cur_disp_y; //current displacements for each pixel

    // Monte Carlo estimation
    for (int x = 1; x < width + 1; ++x)
    {
        for (int y = 1; y < height + 1; ++y)
        {
            cur_disp_x = disp_map_x[(x - 1) + (y - 1) * width];
            cur_disp_y = disp_map_y[(x - 1) + (y - 1) * width];
            float fx = 0, fy = 0;
            float d1, d2;
            // Disp<float>()(x, y, &fx, &fy);
            fx = x+cur_disp_x;
            fy = y+cur_disp_y;
            // calculate L(x,y) = Ind
            count = 0; // size of RR
            Random_disk disk;
            Boolean_model_disk bm_disk;
            for (int i = 0; i < number; ++i)
            {
                bm_disk = BM_disk_arr[i];
                disk = bm_disk.disk;
                d1 = disk.x - fx;
                d2 = disk.y - fy;
                dist = d1 * d1 + d2 * d2;
                if (dist <= bm_disk.rbound)
                {
                    disk_arr[count] = disk;

                    count++;
                }
            }

            /* Set results to 0 */
            CUDA_CALL(cudaMemset(devResults, 0, grid.x * sizeof(unsigned int)));

            //Monte Carlo estimation with sample size = N0
            compute_intensity_kernel_float<<<grid, block, block.x *sizeof(unsigned int)>>>(devPHILOXStates, N0, devResults, x, y, disk_arr, count, cur_disp_x, cur_disp_y);
            /* Copy device memory to host */
            CUDA_CALL(cudaMemcpy(hostResults, devResults, grid.x *
                sizeof(unsigned int), cudaMemcpyDeviceToHost));

            /* Finish sum on host */
            total = 0;
            for(int i = 0; i < grid.x; i++) {
                total += hostResults[i];
            }
            
            float intensity = (float)total/N0;
            // Estimation of Monte Carlo sample size NMC
            int NMC;
            NMC = floor((float)2 / pi * gamma * gamma * (intensity - intensity * intensity) * pow(2, 2 * nbit) / (alpha * alpha)) - N0;

            if (NMC < 1)
            {
                speckle_matrix[(x - 1) + (y - 1) * width] = 1 - intensity; // x & y start at 1 instead of 0
            }
            else
            {
                float res = (1 - intensity) * ((float)N0 / (N0 + NMC));

                /* Set results to 0 */
                CUDA_CALL(cudaMemset(devResults, 0, grid.x * sizeof(unsigned int)));

                //Monte Carlo estimation with sample size = NMC
                compute_intensity_kernel_float<<<grid, block, block.x *sizeof(unsigned int)>>>(devPHILOXStates, NMC, devResults, x, y, disk_arr, count, cur_disp_x, cur_disp_y);
                /* Copy device memory to host */
                CUDA_CALL(cudaMemcpy(hostResults, devResults, grid.x *
                    sizeof(unsigned int), cudaMemcpyDeviceToHost));

                /* Finish sum on host */
                total = 0;
                for(int i = 0; i < grid.x; i++) {
                    total += hostResults[i];
                }
                
                float intensity = (float)total/NMC;
                speckle_matrix[(x - 1) + (y - 1) * width] = res + (1 - intensity) * ((float)NMC / (N0 + NMC));
            }
        }
    }

    // Cleanup
    CUDA_CALL(cudaFree(disk_arr));
    CUDA_CALL(cudaFree(devPHILOXStates));
    CUDA_CALL(cudaFree(devResults));
    free(hostResults);
    free(BM_disk_arr);


    for (int i = 0; i < width * height; ++i)
        speckle_matrix[i] = pow(2, nbit - 1) + (gamma * pow(2, nbit) * (speckle_matrix[i] - 0.5));

    return EXIT_SUCCESS;
}

// template <typename T>
// void monte_carlo_estimation_cuda(T *speckle_matrix, T *Random_centers, T *Random_radius, T *RBound, int number, unsigned int seed, int width, int height, T alpha, int nbit, T gamma, int N0)
// {
//     int test_success;
//     if (typeid(T) == typeid(float)){
//         test_success = MC_estimation_cuda_float((float *)speckle_matrix, (float *)Random_centers, (float *)Random_radius, (float *)RBound, number, seed, width, height, (float)alpha, nbit, (float)gamma, N0);
//     }
//     else if (typeid(T) == typeid(double)){
//         test_success = MC_estimation_cuda_double((double *)speckle_matrix, (double *)Random_centers, (double *)Random_radius, (double *)RBound, number, seed, width, height, (double)alpha, nbit, (double)gamma, N0);
//     }
//     else{
//         printf("Could not run MC estimation CUDA\n");
//     }

//     if (test_success==EXIT_SUCCESS)
//         printf("^^^^ MC estimation CUDA test PASSED\n");
//     else
//         printf("Could not run MC estimation CUDA\n");

// }

// template void monte_carlo_estimation_cuda<float>(float *, float *, float *, float *, int, unsigned int, int, int, float, int, float, int);
// template void monte_carlo_estimation_cuda<double>(double *, double *, double *, double *, int, unsigned int, int, int, double, int, double, int);

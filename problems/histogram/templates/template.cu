#include <cuda_runtime.h>

// CUDA kernel for multi-channel histogram computation
__global__ void histogram_cuda_kernel(
    const int* __restrict__ data,    // Input data [length, num_channels]
    int* __restrict__ histogram,     // Output histogram [num_channels, num_bins]
    const int length,
    const int num_channels,
    const int num_bins
) {
    // Each block handles one channel
    const int channel = blockIdx.x;
    if (channel >= num_channels) return;
    
    // Shared memory for local histogram (one per block)
    extern __shared__ int shared_hist[];
    
    // Initialize shared memory histogram
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        shared_hist[i] = 0;
    }
    __syncthreads();
    
    // Each thread processes multiple elements in this channel
    const int tid = threadIdx.x;
    const int threads_per_block = blockDim.x;
    
    // Process elements in this channel
    for (int idx = tid; idx < length; idx += threads_per_block) {
        int value = data[idx * num_channels + channel];
        if (value >= 0 && value < num_bins) {
            atomicAdd(&shared_hist[value], 1);
        }
    }
    __syncthreads();
    
    // Write shared histogram to global memory
    for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
        histogram[channel * num_bins + bin] = shared_hist[bin];
    }
}


// Host function to launch kernel
torch::Tensor histogram_kernel(
    torch::Tensor data,  // [length, num_channels]
    int num_bins
) {
    TORCH_CHECK(data.device().is_cuda(), "Tensor data must be a CUDA tensor");

    const int length = data.size(0);
    const int num_channels = data.size(1);
    
    // Allocate output tensor
    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(data.device());
    torch::Tensor histogram = torch::zeros({num_channels, num_bins}, options);
    

    ////
    // Launch your kernel here
    
    // Kernel configuration
    const int threads_per_block = 256;
    const int num_blocks = num_channels;  // One block per channel
    const int shared_mem_size = num_bins * sizeof(int);
    
    // Launch kernel
    histogram_cuda_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        data.data_ptr<int>(),
        histogram.data_ptr<int>(),
        length,
        num_channels,
        num_bins
    );
    
    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();

    ////


    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return histogram;
}
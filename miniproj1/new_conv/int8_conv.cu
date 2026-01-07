#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace std;

#define DTYPE int8_t // Change from __half to int8_t

__global__ void conv2d_kernel(
    const DTYPE* __restrict__ input,  // [height][width][in_channels], quantized to INT8
    const DTYPE* __restrict__ kernel, // [out_channels][in_channels][kernel_height][kernel_width], quantized to INT8
    DTYPE* __restrict__ output,       // [height][width][out_channels], INT8 output
    int height, int width, int in_channels, int out_channels,
    int kernel_size, int stride, int padding,
    float input_scale, float kernel_scale // Scale factors for quantization
) {
    // Compute output coordinates
    int ox = blockIdx.x * blockDim.x + threadIdx.x; // Output x (width)
    int oy = blockIdx.y * blockDim.y + threadIdx.y; // Output y (height)
    int oc = blockIdx.z;                            // Output channel

    // Check if within output bounds
    int out_width = (width + 2 * padding - kernel_size) / stride + 1;
    int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    if (ox >= out_width || oy >= out_height || oc >= out_channels) return;

    // Shared memory for input tile (36x36 for 3x3 kernel, stride=1, padding=1)
    __shared__ DTYPE s_input[36][36];
    __shared__ DTYPE s_kernel[64][3][3]; // [in_channels][ky][kx] for current oc

    int32_t sum = 0; // Accumulate in int32_t to avoid overflow

    // Load kernel for current output channel into shared memory
    if (threadIdx.x < 3 && threadIdx.y < 3) {
        for (int ic = 0; ic < in_channels; ic += blockDim.x * blockDim.y) {
            int tid = threadIdx.y * blockDim.x + threadIdx.x;
            int ic_idx = ic + tid;
            if (ic_idx < in_channels) {
                int kernel_idx = ((oc * in_channels + ic_idx) * kernel_size + threadIdx.y) * kernel_size + threadIdx.x;
                s_kernel[ic_idx][threadIdx.y][threadIdx.x] = kernel[kernel_idx];
            }
        }
    }

    // Convolution loop over input channels
    for (int ic = 0; ic < in_channels; ++ic) {
        // Load input tile for current input channel
        s_input[threadIdx.y][threadIdx.x] = 0; // Initialize
        if (threadIdx.y < 36 && threadIdx.x < 36) {
            int ix = blockIdx.x * blockDim.x + threadIdx.x - padding;
            int iy = blockIdx.y * blockDim.y + threadIdx.y - padding;
            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                int input_idx = (iy * width + ix) * in_channels + ic;
                s_input[threadIdx.y][threadIdx.x] = input[input_idx];
            }
        }
        __syncthreads();

        // Compute convolution for this input channel using integer arithmetic
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int ix = ox * stride + kx - padding;
                int iy = oy * stride + ky - padding;
                if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                    int sy = threadIdx.y * stride + ky;
                    int sx = threadIdx.x * stride + kx;
                    sum += static_cast<int32_t>(s_input[sy][sx]) * static_cast<int32_t>(s_kernel[ic][ky][kx]);
                }
            }
        }
        __syncthreads();
    }

    // Scale output to INT8 range
    // Output = (sum * input_scale * kernel_scale) / output_scale, quantized to INT8
    // For simplicity, assume output_scale = input_scale * kernel_scale
    float output_float = static_cast<float>(sum) * input_scale * kernel_scale;
    float output_scale = input_scale * kernel_scale; // Adjust based on desired output range
    output_float = output_float / output_scale; // Normalize
    output_float = min(max(output_float, -128.0f), 127.0f); // Clamp to INT8 range
    int out_idx = (oy * out_width + ox) * out_channels + oc;
    output[out_idx] = static_cast<DTYPE>(round(output_float));
}

int main() {
    // Convolution parameters
    const int height = 224;         // Ny
    const int width = 224;          // Nx
    const int in_channels = 64;     // Ni
    const int out_channels = 64;    // Nn
    const int kernel_size = 3;      // Kx, Ky
    const int stride = 1;
    const int padding = 1;

    // Compute output dimensions
    const int out_width = (width + 2 * padding - kernel_size) / stride + 1;   // 224
    const int out_height = (height + 2 * padding - kernel_size) / stride + 1; // 224

    // Host memory allocations (FP32 for input/kernel, INT8 for quantized data)
    float *h_input_fp32 = (float *)malloc(height * width * in_channels * sizeof(float));
    float *h_kernel_fp32 = (float *)malloc(out_channels * in_channels * kernel_size * kernel_size * sizeof(float));
    DTYPE *h_input = (DTYPE *)malloc(height * width * in_channels * sizeof(DTYPE));
    DTYPE *h_kernel = (DTYPE *)malloc(out_channels * in_channels * kernel_size * kernel_size * sizeof(DTYPE));
    DTYPE *h_output = (DTYPE *)malloc(out_height * out_width * out_channels * sizeof(DTYPE));

    // Initialize input and kernel in FP32 (fill with 1.0f for testing)
    cout << "Initializing FP32 arrays\n";
    for (int i = 0; i < height * width * in_channels; ++i) {
        h_input_fp32[i] = 1.0f; // Replace with actual data
    }
    for (int i = 0; i < out_channels * in_channels * kernel_size * kernel_size; ++i) {
        h_kernel_fp32[i] = 1.0f; // Replace with actual kernel weights
    }

    // Quantize input and kernel to INT8
    cout << "Quantizing arrays to INT8\n";
    float input_max = 0.0f;
    float kernel_max = 0.0f;
    for (int i = 0; i < height * width * in_channels; ++i) {
        input_max = max(input_max, fabs(h_input_fp32[i]));
    }
    for (int i = 0; i < out_channels * in_channels * kernel_size * kernel_size; ++i) {
        kernel_max = max(kernel_max, fabs(h_kernel_fp32[i]));
    }
    float input_scale = 127.0f / input_max; // Scale to [-128, 127]
    float kernel_scale = 127.0f / kernel_max;
    for (int i = 0; i < height * width * in_channels; ++i) {
        float val = h_input_fp32[i] * input_scale;
        h_input[i] = static_cast<DTYPE>(min(max(round(val), -128.0f), 127.0f));
    }
    for (int i = 0; i < out_channels * in_channels * kernel_size * kernel_size; ++i) {
        float val = h_kernel_fp32[i] * kernel_scale;
        h_kernel[i] = static_cast<DTYPE>(min(max(round(val), -128.0f), 127.0f));
    }

    // Device memory allocations
    DTYPE *d_input, *d_kernel, *d_output;
    cudaMalloc((void **)&d_input, height * width * in_channels * sizeof(DTYPE));
    cudaMalloc((void **)&d_kernel, out_channels * in_channels * kernel_size * kernel_size * sizeof(DTYPE));
    cudaMalloc((void **)&d_output, out_height * out_width * out_channels * sizeof(DTYPE));

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, height * width * in_channels * sizeof(DTYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, out_channels * in_channels * kernel_size * kernel_size * sizeof(DTYPE), cudaMemcpyHostToDevice);

    cout << "Starting computation\n";

    // Define block and grid dimensions
    dim3 block(8, 8); // 32x8 threads per block
    dim3 grid(
        (out_width + block.x - 1) / block.x,   // ceil(224 / 32) = 7
        (out_height + block.y - 1) / block.y,  // ceil(224 / 8) = 28
        out_channels                           // Nn = 64
    );

    // Launch kernel with scale factors
    conv2d_kernel<<<grid, block>>>(
        d_input, d_kernel, d_output,
        height, width, in_channels, out_channels,
        kernel_size, stride, padding,
        input_scale, kernel_scale
    );

    // Synchronize to ensure kernel execution is complete
    cudaDeviceSynchronize();

    // Copy output from device to host
    cudaMemcpy(h_output, d_output, out_height * out_width * out_channels * sizeof(DTYPE), cudaMemcpyDeviceToHost);

    // Optional: Dequantize output for verification (FP32)
    float *h_output_fp32 = (float *)malloc(out_height * out_width * out_channels * sizeof(float));
    float output_scale = input_scale * kernel_scale;
    for (int i = 0; i < out_height * out_width * out_channels; ++i) {
        h_output_fp32[i] = static_cast<float>(h_output[i]) * output_scale;
    }

    // Clean up
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    free(h_input_fp32);
    free(h_kernel_fp32);
    free(h_input);
    free(h_kernel);
    free(h_output);
    free(h_output_fp32);

    return 0;
}
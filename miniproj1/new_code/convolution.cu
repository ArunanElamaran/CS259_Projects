#include <iostream>
#include <string>
#include <cudnn.h>
#include <assert.h>
#include "dnn.hpp"

using namespace std;

//Define the parameters if not defined externally
#ifndef Sy
  #define Sy 1
  #define Sx 1
#endif

#ifndef Tnn
  //Tiling Sizes
  #define Tnn 32
  #define Tn  16
  #define Ti  16
  
  #define Ty  8
  #define Tx  8
#endif

#define NYPAD (Ny+Ky)
#define NXPAD (Nx+Kx)

#define NYSCL (Ny/Sy)
#define NXSCL (Nx/Sx)

#define SYNAPSE_SIZE (1L*Ky*Kx*Nn*Ni)
#define NEURON_I_SIZE (1L*NYPAD*NXPAD*Ni)
#define NEURON_N_SIZE (1L*NYSCL*NXSCL*Nn)

// Host side Memory Structures
VTYPE (*synapse);

VTYPE  (*neuron_i);
VTYPE  (*neuron_n);
VTYPE (*neuron_n2);
VTYPE (*neuron_n3);

// Device side Memory Structures
VTYPE (*synapse_D);
VTYPE (*neuron_i_D);
VTYPE (*neuron_n_D);

// --------------------- CONVOLUTION FUNCTIONS ----------------------------

void convolution_layer_original(VTYPE *synapse, VTYPE *neuron_i, VTYPE *neuron_n) {

  VTYPE sum[Nn]={0};

  // — Original code — (excluding nn, ii loops)
  int yout = 0;
  for (int y = 0; y < Ny; y += Sy) { // tiling for y;
    int xout = 0;
    for (int x = 0; x < Ny; x += Sx) { // tiling for x;
      for (int nn = 0; nn < Nn; nn += Tn) {
        for (int n = nn; n < nn + Tn; n++) {
          sum[n]=0;
        }

        // sliding window;
        for (int ky = 0; ky < Ky; ky++)
          for (int kx = 0; kx < Kx; kx++)
            for (int n = nn; n < nn + Tn; n++)
              for (int i = 0; i < Ni; i++) {
                int synapse_idx = (((ky * Kx) + kx) * Nn + n) * Ni + i;
                int neuron_i_idx = ((y + ky) * NXPAD + (x + kx)) * Ni + i;
                sum[n]+= synapse[synapse_idx] * neuron_i[neuron_i_idx];
              }
        for (int n = nn; n < nn + Tn; n++) {
          int neuron_n_idx = (yout * NXSCL) * Nn + (xout * Nn) + n;
          neuron_n[neuron_n_idx] = transfer(sum[n]);
        }
      }
      xout++; 
    }
    yout++;
  }
}

__global__
void convolution_layer(VTYPE *synapse, VTYPE *neuron_i, VTYPE *neuron_n) {

  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.z;

  if(y >= Ny || x >= Nx || n >= Nn) { return; }

  VTYPE sum = 0;

  // For each kernel position (ky, kx)
  for (int ky = 0; ky < Ky; ky++) {
    for (int kx = 0; kx < Kx; kx++) {
      // For each input channel (i)
      for (int i = 0; i < Ni; i++) {
        int synapse_idx = (((ky * Kx) + kx) * Nn + n) * Ni + i;
        int input_idx   = ((y + ky) * NXPAD + (x + kx)) * Ni + i;
        sum += synapse[synapse_idx] * neuron_i[input_idx];
      }
    }
  }

  // Apply activation function
  int output_idx = (y * Nx + x) * Nn + n;
  neuron_n[output_idx] = transfer_d(sum);
}

void cudnn_convolution_graph(VTYPE* synapse_D, VTYPE* neuron_i_D, VTYPE* neuron_n_D) {
  // DOES NOT WORK DOES NOT WORK DOES NOT WORK DOES NOT WORK DOES NOT WORK DOES NOT WORK
  cudnnHandle_t handle;
  cudnnCreate(&handle);

  // Create descriptors
  cudnnTensorDescriptor_t input_desc, output_desc;
  cudnnFilterDescriptor_t filter_desc;
  cudnnConvolutionDescriptor_t conv_desc;

  cudnnCreateTensorDescriptor(&input_desc);
  cudnnCreateTensorDescriptor(&output_desc);
  cudnnCreateFilterDescriptor(&filter_desc);
  cudnnCreateConvolutionDescriptor(&conv_desc);

  // --- Describe the INPUT tensor ---
  // Now using unpadded Ny x Nx input size
  cudnnSetTensor4dDescriptor(
      input_desc,
      CUDNN_TENSOR_NCHW,  // input is NCHW
      CUDNN_DATA_FLOAT,
      1,                  // batch size
      Ni,                 // input channels
      Ny,                 // input height (NO padding)
      Nx                  // input width (NO padding)
  );

  // --- Describe the OUTPUT tensor ---
  cudnnSetTensor4dDescriptor(
      output_desc,
      CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT,
      1,                  // batch size
      Nn,                 // output channels
      NYSCL,              // output height
      NXSCL               // output width
  );

  // --- Describe the FILTER tensor ---
  cudnnSetFilter4dDescriptor(
      filter_desc,
      CUDNN_DATA_FLOAT,
      CUDNN_TENSOR_NCHW,  // filter layout
      Nn,                 // output channels
      Ni,                 // input channels
      Ky,                 // kernel height
      Kx                  // kernel width
  );

  // --- Describe the CONVOLUTION operation ---
  cudnnSetConvolution2dDescriptor(
      conv_desc,
      Ky/2, Kx/2,         // pad_h, pad_w → padding HALF of kernel size (same padding)
      Sy, Sx,             // stride_h, stride_w
      1, 1,               // dilation_h, dilation_w
      CUDNN_CROSS_CORRELATION,
      CUDNN_DATA_FLOAT
  );

  // --- Choose best algorithm ---
  cudnnConvolutionFwdAlgoPerf_t perf;
  int returnedAlgoCount = 0;
  cudnnFindConvolutionForwardAlgorithm(
      handle,
      input_desc, filter_desc, conv_desc, output_desc,
      1, &returnedAlgoCount, &perf
  );
  cudnnConvolutionFwdAlgo_t algo = perf.algo;

  // --- Allocate workspace ---
  size_t workspace_size = 0;
  cudnnGetConvolutionForwardWorkspaceSize(
      handle,
      input_desc, filter_desc, conv_desc, output_desc,
      algo, &workspace_size
  );

  void* workspace = nullptr;
  if (workspace_size > 0) {
      cudaMalloc(&workspace, workspace_size);
  }

  // --- Perform the convolution ---
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cudnnConvolutionForward(
      handle,
      &alpha,
      input_desc, neuron_i_D,
      filter_desc, synapse_D,
      conv_desc,
      algo,
      workspace, workspace_size,
      &beta,
      output_desc, neuron_n_D
  );

  // --- Cleanup ---
  if (workspace) {
      cudaFree(workspace);
  }
  cudnnDestroyTensorDescriptor(input_desc);
  cudnnDestroyTensorDescriptor(output_desc);
  cudnnDestroyFilterDescriptor(filter_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  cudnnDestroy(handle);
}




int main(const int argc, const char** argv) {
  /*
  Quick Notes:
  - <# of blocks, # of threads> are the args
  */

  // --------------------- HOST ALLOCATIONS and INITIALIZATION ----------------------------
  // Host Side Memory Allocations
  synapse   = (VTYPE *)malloc(SYNAPSE_SIZE*sizeof(VTYPE));
  neuron_i   = (VTYPE *)malloc(NEURON_I_SIZE*sizeof(VTYPE));
  neuron_n   = (VTYPE *)malloc(NEURON_N_SIZE*sizeof(VTYPE));
  neuron_n2   = (VTYPE *)malloc(NEURON_N_SIZE*sizeof(VTYPE));
  neuron_n3   = (VTYPE *)malloc(NEURON_N_SIZE*sizeof(VTYPE));

  cout << "initializing arrays\n";
  for (int i = 0; i < SYNAPSE_SIZE; i++) {
    synapse[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
  }
  for (int i = 0; i < NEURON_I_SIZE; i++) {
    neuron_i[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
  }
  
  // --------------------- DEVICE ALLOCATIONS and DATA TRANSFERS ----------------------------
  // Device Side Memory Allocations
  cudaMalloc((void **) &synapse_D,  SYNAPSE_SIZE*sizeof(VTYPE));
  cudaMalloc((void **) &neuron_i_D, NEURON_I_SIZE*sizeof(VTYPE));
  cudaMalloc((void **) &neuron_n_D, NEURON_N_SIZE*sizeof(VTYPE));

  // Copy data from Host to Device
  cudaMemcpy(synapse_D, synapse, SYNAPSE_SIZE*sizeof(VTYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(neuron_i_D, neuron_i, NEURON_I_SIZE*sizeof(VTYPE), cudaMemcpyHostToDevice);
  
  // --------------------- PERFORMING CONVOLUTION ----------------------------
  cout << "starting computation\n";

  // Define block and grid dimensions
  dim3 block(32, 32); // 16x16 threads per block
  dim3 grid(
      (NXSCL + block.x - 1) / block.x,  
      (NYSCL + block.y - 1) / block.y,
      Nn
  );

  //CUDA Simple Version
  begin_roi();
  convolution_layer<<<grid, block>>>(synapse_D,neuron_i_D,neuron_n_D);
  cudaDeviceSynchronize();
  end_roi();
  cudaMemcpy(neuron_n2, neuron_n_D, NEURON_N_SIZE*sizeof(VTYPE),cudaMemcpyDeviceToHost);

  cout << "simple CUDA version complete!\n";  

  // CUDNN Version
  // DOES NOT WORK DOES NOT WORK DOES NOT WORK DOES NOT WORK DOES NOT WORK DOES NOT WORK
  // cout << "starting cuDNN Graph computation\n";

  // begin_roi();
  // cudnn_convolution_graph(synapse_D, neuron_i_D, neuron_n_D);
  // cudaDeviceSynchronize();
  // end_roi();
  // cudaMemcpy(neuron_n3, neuron_n_D, NEURON_N_SIZE*sizeof(VTYPE), cudaMemcpyDeviceToHost);
  // cout << "cudnn Graph version complete!\n";

  // Original Version
  begin_roi();
  convolution_layer_original(synapse,neuron_i,neuron_n);
  cudaDeviceSynchronize();
  end_roi();

  cout << "original simple computation complete!\n"; 

  // --------------------- COPY BACK AND COMPARE ----------------------------
  // Copy output data from Device to Host
  // Check for kernel errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
      exit(1);
  }
  else { cout << "Kernel Success!" << endl; }

  cout<<"Comparing Original and Custom CUDA..."<<endl;
  compare((VTYPE*)neuron_n,(VTYPE*)neuron_n2,NEURON_N_SIZE);

  // cout<<"Comparing Original and CUDNN..."<<endl;
  // compare((VTYPE*)neuron_n,(VTYPE*)neuron_n3,NEURON_N_SIZE);
  
  // --------------------- CLEAN UP ----------------------------
  // Deallocate Host and Device memory structures
  cudaFree(synapse_D); free(synapse);
  cudaFree(neuron_i_D); free(neuron_i);
  cudaFree(neuron_n_D); free(neuron_n);
  free(neuron_n2); free(neuron_n3);
}
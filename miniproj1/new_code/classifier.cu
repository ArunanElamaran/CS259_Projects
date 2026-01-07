#include <iostream>
#include <cudnn.h>
#include <assert.h>
#include "dnn.hpp"

using namespace std;

//Define the parameters if not defined externally
#ifndef Nn
  #define Nn 128  // Number of Output Layers
  #define Ni 224  // Number of Input  Layers
#endif

#ifndef Tii
  // Tiling Sizes
  #define Tnn 32  
  #define Tii 32
  //#define Tn 5
  //#define Ti 25
  #define Tn 16
  #define Ti 16
#endif

// Sizes
#define SYNAPSE_SIZE (1L*Nn*Ni)
#define NEURON_I_SIZE (1L*Ni)
#define NEURON_N_SIZE (1L*Nn)

#define MAX(a, b) ((a) > (b) ? (a) : (b))

// --- Host side Memory Structures
// For normal storage of values
VTYPE *synapseM, *neuron_iM;

// For CPU algorithm
VTYPE synapse[Nn][Ni] __attribute__((aligned(64)));
VTYPE neuron_i[Ni] __attribute__((aligned(64)));
VTYPE neuron_n[Nn] __attribute__((aligned(64))),    neuron_n2[Nn] __attribute__((aligned(64))), neuron_n3[Nn] __attribute__((aligned(64))), neuron_n4[Nn] __attribute__((aligned(64)));

// --- Device side Memory Structures
VTYPE *synapse_D;
VTYPE *neuron_i_D;
VTYPE *neuron_n_D, *neuron_n2_D, *neuron_n3_D;

// --------------------- CLASSIFIER FUNCTIONS ----------------------------

void fill_classifier(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[Ni], 
    VTYPE (&neuron_n)[Nn],   VTYPE (&neuron_n2)[Nn],
    VTYPE *synapseM, VTYPE *neuron_iM) 
{
  for(int n = 0; n < Nn; ++n) {
    for(int i = 0; i < Ni; ++i) {
      VTYPE num = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
      synapse[n][i] = num; synapseM[n*Ni + i] = num; 
    }
  }
  for(int i = 0; i < Ni; ++i) {
    VTYPE num = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
    neuron_i[i] = num; neuron_iM[i] = num;
  }
  for(int n = 0; n < Nn; ++n) {
    neuron_n[n] = 0; //i;
    neuron_n2[n] = 0; //i;
  }
}

void classifier_layer(VTYPE (&synapse)[Nn][Ni], VTYPE (&neuron_i)[Ni], VTYPE (&neuron_n)[Nn]) {
  // int total_calc=0;
  for (int n = 0; n < Nn; n++) {
    VTYPE temp=0;
    for (int i = 0; i < Ni; i++) {
      temp += synapse[n][i] * neuron_i[i];
    }
    neuron_n[n] = transfer(temp);
  }
}

__global__
void simple_classifier_layer_cuda(VTYPE *synapse, VTYPE *neuron_i, VTYPE *neuron_n) {
  int n = blockIdx.x * blockDim.x + threadIdx.x; // each thread handles one output neuron

  if (n < Nn) {
      VTYPE temp = 0;
      for (int i = 0; i < Ni; i++) {
          temp += synapse[n * Ni + i] * neuron_i[i];
      }
      neuron_n[n] = transfer_d(temp);
  }
}

__global__
void classifier_layer_cuda(
  const VTYPE* __restrict__ synapse,
  const VTYPE* __restrict__ neuron_i,
  VTYPE* neuron_n
) {
  int n = blockIdx.x * MAX(blockDim.x, Tii) + threadIdx.x;
  if (n >= Nn || threadIdx.x >= Tii) return;

  VTYPE sum = 0;
  __shared__ VTYPE shared_neuron_i[Tii];

  for (int tile_i = 0; tile_i < NEURON_I_SIZE; tile_i += Tii) {
      int i = tile_i + threadIdx.x;
      if (i < Ni) {
          shared_neuron_i[threadIdx.x] = neuron_i[i];
      }
      __syncthreads();

      for (int i_local = 0; i_local < Tii; i_local++) {
          int i_global = tile_i + i_local;
          if (i_global < Ni) {
              sum += synapse[n * Ni + i_global] * shared_neuron_i[i_local];
          }
      }
      __syncthreads();
  }

  neuron_n[n] = transfer_d(sum);
}

void transpose_synapse(VTYPE* dest, VTYPE* src, int rows, int cols) {
  for (int n = 0; n < rows; ++n) {
      for (int i = 0; i < cols; ++i) {
          dest[i * rows + n] = src[n * cols + i];
      }
  }
}

void cudnn_classifier_layer(VTYPE* synapse_D, VTYPE* neuron_i_D, VTYPE* neuron_n_D) {
  // DOES NOT WORK DOES NOT WORK DOES NOT WORK DOES NOT WORK DOES NOT WORK DOES NOT WORK
  cudnnHandle_t handle;
  cudnnCreate(&handle);

  cudnnTensorDescriptor_t input_desc, output_desc;
  cudnnFilterDescriptor_t filter_desc;
  cudnnConvolutionDescriptor_t conv_desc;

  cudnnCreateTensorDescriptor(&input_desc);
  cudnnCreateTensorDescriptor(&output_desc);
  cudnnCreateFilterDescriptor(&filter_desc);
  cudnnCreateConvolutionDescriptor(&conv_desc);

  // --- Setup Tensor Shapes
  cudnnSetTensor4dDescriptor(
      input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
      1, Ni, 1, 1
  );
  cudnnSetTensor4dDescriptor(
      output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
      1, Nn, 1, 1
  );
  cudnnSetFilter4dDescriptor(
      filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
      Nn, Ni, 1, 1
  );
  cudnnSetConvolution2dDescriptor(
      conv_desc,
      0, 0,    // pad height, width
      1, 1,    // stride height, width
      1, 1,    // dilation height, width
      CUDNN_CROSS_CORRELATION,
      CUDNN_DATA_FLOAT
  );

  // --- Algorithm Selection (modern way)
  cudnnConvolutionFwdAlgoPerf_t perf;
  int returnedAlgoCount = 0;
  cudnnFindConvolutionForwardAlgorithm(
      handle,
      input_desc, filter_desc, conv_desc, output_desc,
      1, &returnedAlgoCount, &perf
  );
  cudnnConvolutionFwdAlgo_t algo = perf.algo;

  // --- Workspace
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

  const float alpha = 1.0f;
  const float beta = 0.0f;

  // --- Perform the convolution (which acts like GEMV)
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

  if (workspace_size > 0) {
      cudaFree(workspace);
  }

  // --- Cleanup
  cudnnDestroyTensorDescriptor(input_desc);
  cudnnDestroyTensorDescriptor(output_desc);
  cudnnDestroyFilterDescriptor(filter_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  cudnnDestroy(handle);
}

int main(int argc, char** argv) {

  // --------------------- HOST ALLOCATIONS and INITIALIZATION ----------------------------
  // Host Side Memory Allocations
  synapseM   = (VTYPE *)malloc(SYNAPSE_SIZE*sizeof(VTYPE));
  neuron_iM   = (VTYPE *)malloc(NEURON_I_SIZE*sizeof(VTYPE));

  cout << "initializing arrays\n";

  fill_classifier(synapse,neuron_i,neuron_n,neuron_n2,synapseM,neuron_iM);

  // --------------------- DEVICE ALLOCATIONS and DATA TRANSFERS ----------------------------
  // Device Side Memory Allocations
  cudaMalloc((void **) &synapse_D,  SYNAPSE_SIZE*sizeof(VTYPE));
  cudaMalloc((void **) &neuron_i_D, NEURON_I_SIZE*sizeof(VTYPE));
  cudaMalloc((void **) &neuron_n_D, NEURON_N_SIZE*sizeof(VTYPE));

  // Copy data from Host to Device
  cudaMemcpy(synapse_D, synapse, SYNAPSE_SIZE*sizeof(VTYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(neuron_i_D, neuron_i, NEURON_I_SIZE*sizeof(VTYPE), cudaMemcpyHostToDevice);

  // --------------------- PERFORMING COMPUTATION ----------------------------
  cout << "starting computation\n\n";

  //CUDA Simple Version
  int tpb = Tnn;  // launch one thread per neuron in tile
  int numBlocks = (Nn + tpb - 1) / tpb;

  begin_roi();
  simple_classifier_layer_cuda<<<numBlocks, tpb>>>(synapse_D, neuron_i_D, neuron_n_D);  
  cudaDeviceSynchronize();
  end_roi();
  cudaMemcpy(neuron_n2, neuron_n_D, NEURON_N_SIZE*sizeof(VTYPE),cudaMemcpyDeviceToHost);

  cout << "simple CUDA version complete!\n\n";
  
  //CUDA Shared Memory Version
  tpb = Tii;
  numBlocks = (Nn + tpb - 1) / tpb;

  begin_roi();
  classifier_layer_cuda<<<numBlocks, tpb>>>(synapse_D, neuron_i_D, neuron_n_D);  
  cudaDeviceSynchronize();
  end_roi();
  cudaMemcpy(neuron_n3, neuron_n_D, NEURON_N_SIZE*sizeof(VTYPE),cudaMemcpyDeviceToHost);

  cout << "Complex CUDA version complete!\n\n";

  // cuDNN classifier version
  // DOES NOT WORK DOES NOT WORK DOES NOT WORK DOES NOT WORK DOES NOT WORK DOES NOT WORK
  // // Create a temporary transposed buffer
  // VTYPE* synapse_transposed = (VTYPE*)malloc(SYNAPSE_SIZE*sizeof(VTYPE));
  // transpose_synapse(synapse_transposed, synapseM, Nn, Ni);

  // // Copy transposed synapse
  // cudaMemcpy(synapse_D, synapse_transposed, SYNAPSE_SIZE*sizeof(VTYPE), cudaMemcpyHostToDevice);

  // free(synapse_transposed);
  
  // begin_roi();
  // cudnn_classifier_layer(synapse_D, neuron_i_D, neuron_n_D);
  // cudaDeviceSynchronize();
  // end_roi();
  // cudaMemcpy(neuron_n4, neuron_n_D, NEURON_N_SIZE*sizeof(VTYPE), cudaMemcpyDeviceToHost);

  // cout << "cuDNN classifier version complete!\n\n";

  // Original Version
  begin_roi();
  classifier_layer(synapse,neuron_i,neuron_n);  
  end_roi();

  cout << "original simple computation complete!\n\n";  

  // --------------------- COPY BACK AND COMPARE ----------------------------
  // Copy output data from Device to Host
  // Check for kernel errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
      exit(1);
  }
  else { cout << "Kernel Success!" << endl; }

  cout<<"Comparing Original and CUDA Simple ..."<<endl;
  compare(neuron_n,neuron_n2,Nn);
  cout<<"Comparing Original and CUDA Shared Memory ..."<<endl;
  compare(neuron_n,neuron_n3,Nn);
  // cout<<"Comparing Original and CUDNN Version ..."<<endl;
  // compare(neuron_n,neuron_n4,Nn);

  // --------------------- CLEAN UP ----------------------------
  // Deallocate Device memory structures
  cudaFree(synapse_D); free(synapseM);
  cudaFree(neuron_i_D); free(neuron_iM);
  cudaFree(neuron_n_D);
}
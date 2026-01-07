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

// ------------ Tiling Sizes ------------
#ifndef Tn
  #define Tn  16
#endif

#define KX Kx
#define KY Ky

#ifndef BLOCK_X
  #define BLOCK_X 32
#endif

#ifndef BLOCK_Y
  #define BLOCK_Y 32
#endif
// ------------------------

#define SHMEM_TILE_X (BLOCK_X + KX - 1)  // 32 + 3 - 1 = 34
#define SHMEM_TILE_Y (BLOCK_Y + KY - 1)  // 32 + 3 - 1 = 34

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

// For rearranging for device purposes
VTYPE (*synapse_CD); 
VTYPE  (*neuron_i_CD);
VTYPE  (*neuron_n_CD);

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

  // Thread & block indices
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x * blockDim.x;
  int by = blockIdx.y * blockDim.y;
  int n = blockIdx.z;

  int x = bx + tx;
  int y = by + ty;

  // Shared memory per input channel
  __shared__ float shmem_input[SHMEM_TILE_Y][SHMEM_TILE_X];

  VTYPE sum = 0;

  // Iterate over input channels
  for (int i = 0; i < Ni; i++) {
    // Each thread will cooperatively load the padded input tile into shared memory
    for (int dy = ty; dy < SHMEM_TILE_Y; dy += blockDim.y) {
      for (int dx = tx; dx < SHMEM_TILE_X; dx += blockDim.x) {
        int global_y = by + dy;
        int global_x = bx + dx;

        if (global_y < NYPAD && global_x < NXPAD) {
          int input_idx = (i * NYPAD + global_y) * NXPAD + global_x;
          shmem_input[dy][dx] = neuron_i[input_idx];
        }
      }
    }

    __syncthreads();  // Ensure all data is loaded

    // Compute convolution only if output thread is within bounds
    if (x < NXSCL && y < NYSCL && n < Nn) {
      for (int ky = 0; ky < Ky; ky++) {
        for (int kx = 0; kx < Kx; kx++) {
          int sh_y = ty + ky;
          int sh_x = tx + kx;
          VTYPE val = shmem_input[sh_y][sh_x];

          int synapse_idx = ((n * Ni + i) * Ky + ky) * Kx + kx;
          sum += synapse[synapse_idx] * val;
        }
      }
    }

    __syncthreads();  // Wait before loading next input channel
  }

  // Write result if output index is valid
  if (x < NXSCL && y < NYSCL && n < Nn) {
    int output_idx = (n * NYSCL + y) * NXSCL + x;
    neuron_n[output_idx] = transfer_d(sum);
  }
}


int main(const int argc, const char** argv) {
  /*
  Quick Notes:
  - <# of blocks, # of threads> are the args
  */

  // --------------------- HOST ALLOCATIONS and INITIALIZATION ----------------------------
  // Host Side Memory Allocations
  synapse   = (VTYPE *)malloc(SYNAPSE_SIZE*sizeof(VTYPE)); synapse_CD = (VTYPE *)malloc(SYNAPSE_SIZE*sizeof(VTYPE));
  neuron_i   = (VTYPE *)malloc(NEURON_I_SIZE*sizeof(VTYPE)); neuron_i_CD = (VTYPE *)malloc(NEURON_I_SIZE*sizeof(VTYPE));
  neuron_n   = (VTYPE *)malloc(NEURON_N_SIZE*sizeof(VTYPE)); neuron_n_CD = (VTYPE *)malloc(NEURON_N_SIZE*sizeof(VTYPE));
  neuron_n2   = (VTYPE *)malloc(NEURON_N_SIZE*sizeof(VTYPE));

  cout << "initializing arrays\n";
  for (int i = 0; i < SYNAPSE_SIZE; i++) {
    synapse[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
  }
  for (int i = 0; i < NEURON_I_SIZE; i++) {
    neuron_i[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
  }

  /*
  For device side accesses, rearrange the parameters from their initial layout to device friendly layout:
  VTYPE (*synapse)[Ky][Kx][Nn][Ni]; --> VTYPE (*synapse_CD)[Nn][Ni][Ky][Kx];
  VTYPE  (*neuron_i)[NYPAD][NXPAD][Ni]; --> VTYPE  (*neuron_i_CD)[Ni][NYPAD][NXPAD];
  VTYPE  (*neuron_n_CD)[Nn][NYSCL][NXSCL]; --> VTYPE  (*neuron_n)[NYSCL][NXSCL][Nn];
  */

  // Rearrange Synapse
  for (int ky = 0; ky < Ky; ky++) {
    for (int kx = 0; kx < Kx; kx++) {
      for (int n = 0; n < Nn; n++) {
        for (int i = 0; i < Ni; i++) {
          int src_idx = ((ky * Kx + kx) * Nn + n) * Ni + i;
          int dst_idx = ((n * Ni + i) * Ky + ky) * Kx + kx;
          synapse_CD[dst_idx] = synapse[src_idx];
        }
      }
    }
  }

  // Rearrange Input
  for (int y = 0; y < NYPAD; y++) {
    for (int x = 0; x < NXPAD; x++) {
      for (int i = 0; i < Ni; i++) {
        int src_idx = (y * NXPAD + x) * Ni + i;
        int dst_idx = (i * NYPAD + y) * NXPAD + x;
        neuron_i_CD[dst_idx] = neuron_i[src_idx];
      }
    }
  }
  
  // --------------------- DEVICE ALLOCATIONS and DATA TRANSFERS ----------------------------
  // Device Side Memory Allocations
  cudaMalloc((void **) &synapse_D,  SYNAPSE_SIZE*sizeof(VTYPE));
  cudaMalloc((void **) &neuron_i_D, NEURON_I_SIZE*sizeof(VTYPE));
  cudaMalloc((void **) &neuron_n_D, NEURON_N_SIZE*sizeof(VTYPE));

  // Copy data from Host to Device
  cudaMemcpy(synapse_D, synapse_CD, SYNAPSE_SIZE*sizeof(VTYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(neuron_i_D, neuron_i_CD, NEURON_I_SIZE*sizeof(VTYPE), cudaMemcpyHostToDevice);
  
  // --------------------- PERFORMING CONVOLUTION ----------------------------
  cout << "starting computation\n";

  // Define block and grid dimensions
  dim3 block(BLOCK_X, BLOCK_Y); // 16x16 threads per block
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
  cudaMemcpy(neuron_n_CD, neuron_n_D, NEURON_N_SIZE*sizeof(VTYPE),cudaMemcpyDeviceToHost);

  cout << "simple CUDA version complete!\n";  

  // Rearrange the Output
  for (int n = 0; n < Nn; n++) {
    for (int y = 0; y < NYSCL; y++) {
      for (int x = 0; x < NXSCL; x++) {
        int src_idx = (n * NYSCL + y) * NXSCL + x;     // [n][y][x]
        int dst_idx = (y * NXSCL + x) * Nn + n;        // [y][x][n]
        neuron_n2[dst_idx] = neuron_n_CD[src_idx];
      }
    }
  }

  // // Original Version
  // begin_roi();
  // convolution_layer_original(synapse,neuron_i,neuron_n);
  // cudaDeviceSynchronize();
  // end_roi();

  // cout << "original simple computation complete!\n"; 

  // --------------------- COPY BACK AND COMPARE ----------------------------
  // Copy output data from Device to Host
  // Check for kernel errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
      exit(1);
  }
  else { cout << "Kernel Success!" << endl; }

  // cout<<"Comparing Original and Custom CUDA..."<<endl;
  // compare((VTYPE*)neuron_n,(VTYPE*)neuron_n2,NEURON_N_SIZE);
  
  // --------------------- CLEAN UP ----------------------------
  // Deallocate Host and Device memory structures
  cudaFree(synapse_D); free(synapse); free(synapse_CD);
  cudaFree(neuron_i_D); free(neuron_i); free(neuron_i_CD);
  cudaFree(neuron_n_D); free(neuron_n); free(neuron_n_CD);
  free(neuron_n2);
}
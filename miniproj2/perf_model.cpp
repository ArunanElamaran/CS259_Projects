#include <iostream>
#include <cmath>
#include <algorithm>
#include <cmath>
using namespace std;

//Define the parameters if not defined externally
#ifndef Sy
  #define Sy 1
  #define Sx 1
#endif

// ------------ Tiling Sizes ------------------------------------
#define KX Kx
#define KY Ky

#ifndef BLOCK_X
  #define BLOCK_X 32
#endif

#ifndef BLOCK_Y
  #define BLOCK_Y 32
#endif
// --------------------------------------------------------------

#define SHMEM_TILE_X (BLOCK_X + KX - 1)  // 32 + 3 - 1 = 34
#define SHMEM_TILE_Y (BLOCK_Y + KY - 1)  // 32 + 3 - 1 = 34

#define NYPAD (Ny+Ky)
#define NXPAD (Nx+Kx)

#define NYSCL (Ny/Sy)
#define NXSCL (Nx/Sx)

#define SYNAPSE_SIZE (1L*Ky*Kx*Nn*Ni)
#define NEURON_I_SIZE (1L*NYPAD*NXPAD*Ni)
#define NEURON_N_SIZE (1L*NYSCL*NXSCL*Nn)


// Global Parameters-----------------------------------------------------------
const long GPU_FREQ = 1200000000;   // Hz
const long MEM_FREQ = 850000000;    // Hz


// Performance model output----------------------------------------------------
struct perf_model_out {
    double TOps             = 0.0;
    double L1_OpIntensity   = 0.0;
    double L2_OpIntensity   = 0.0;
    double DRAM_OpIntensity = 0.0;
};

// Param Definitions-------------------------------------------------------------
//     Ni/Nn – Number of input/output channels/feature-maps
//     Nx/Ny – Image/feature-map width/height
//     Kx/Ky – Kernel size
//     DTn - Tiling size
//     BLOCK_X, BLOCK_Y - Threading dimensions

perf_model_out perf_model() {
    // -------------------- GPU constants --------------------
    // ------ Hardware Compute Limits
    const long NUM_CORES             = 5120;
    const long NUM_SM                = 80;
    const long CORES_PER_SM          = 64;
    const long WARP_SIZE             = 32;
    const long MAX_FLOPS             = 13800000000000;
    const long MAX_INT_OPS           = 55200000000000;
    const long MAX_BLOCKS_PER_SM     = 32;
    const long MAX_WARPS_PER_SM      = 64;
    const long MAX_THREADS_PER_SM    = 2048;
    const long MAX_THREADS_PER_BLOCK = 1024;
    const long MAX_THREAD_X          = 1024;
    const long MAX_THREAD_Y          = 1024;
    const long MAX_THREAD_Z          = 64;
    const long MAX_GRID_X            = 2147483647;
    const long MAX_GRID_Y            = 65535;
    const long MAX_GRID_Z            = 65535;

    // ------ Memory/Cache Limits
    const long GLBL_MEM              = 12635406336; // Bytes
    const long MEM_BUS_WIDTH         = 3072/8;      // Bytes: 384
    const long L2_SIZE               = 4718592;     // Bytes
    const long CONST_MEM             = 65536;       // Bytes
    const long SHARED_MEM_PER_BLOCK  = 49152;       // Bytes
    const long SHARED_MEM_PER_SM     = 98304;       // Bytes
    const long MAX_REGISTERS_PER_SM  = 65536;
    const long L1_UPDATE_GRANULARITY = 128;
    const long L2_LINE_SIZE          = 64;
    const long double DRAM_BW        = 750e9;       // Bytes/sec

    // ------ Cycle Calculations Limits
    const long double GPU_PERIOD     = (1.0 / GPU_FREQ);
    const long double SHAREDMEM_HIT  = 19 * GPU_PERIOD;
    const long double L1_HIT         = 28 * GPU_PERIOD;
    const long double L2_HIT         = 193 * GPU_PERIOD;  // L1 miss, data in L2
    const long double DRAM_TLB_HIT   = 375 * GPU_PERIOD;  // L1 and L2 miss, TLB miss data in DRAM
    const long double DRAM_HIT       = 1029 * GPU_PERIOD; // L1 and L2 miss, TLB miss data in DRAM

    // Temporary variables and output
    perf_model_out outputs;
    long  ops       = 0;
    double comp_time = 0.0;
    double mem_time  = 0.0;


    // Debug
    cout << "Input parameters: Nx=" << Nx << " Ny=" << Ny << " Kx=" << Kx << " Ky=" << Ky << " Ni=" << Ni << " Nn=" \
         << Nn << " BLOCK_X=" << BLOCK_X << " BLOCK_Y=" << BLOCK_Y <<  endl; 


    // TOTAL NUMBER OF OPERATIONS ------------------------------------------------------------------------------------------------------
    cout << "TOTAL OPERATIONS CALCULATIONS: ----------" << endl;
    long flops = 2*long(Nx)*long(Ny)*long(Kx)*long(Ky)*long(Ni)*long(Nn);
    
    long long int_ops = static_cast<long long>(NXSCL) * NYSCL * Nn * (
        8 + Ni * (
            6 * SHMEM_TILE_Y * SHMEM_TILE_X +
            9 * Ky * Kx +
            1 * (SHMEM_TILE_Y / BLOCK_Y) * (SHMEM_TILE_X / BLOCK_X)
        )
    );
    int_ops /= 60 * 1.69*(32/BLOCK_X);

    // Print out the results
    cout << "FLOP: " << flops << endl;
    cout << "Integer OP: " << int_ops << endl;

    ops = flops + int_ops;
    cout << "Total OP: " <<  ops << endl;
    cout << "----------\n\n" << endl;
    
    // COMPUTE-BOUND TIME -------------------------------------------------------------------------------------------------------------------
    cout << "COMPUTE BOUND TIME CALCULATIONS: ----------" << endl;
    double flop_throughput;
    double int_throughput;

    // Calculate total threads and number of SMs required
    long grid_dim_x = (Nx + BLOCK_X - 1) / BLOCK_X;
    long grid_dim_y = (Ny + BLOCK_Y - 1) / BLOCK_Y;
    long grid_dim_z = Nn;
    long num_blocks = grid_dim_x * grid_dim_y * grid_dim_z;
    long threads_per_block = BLOCK_X * BLOCK_Y;
    long blocks_per_sm;
    long active_sms;
    long sm_iterations;

    // Since blocks are spread out before looping, check SM utilization
    active_sms = min(num_blocks, NUM_SM);

    // Calculate how many iterations each SM needs
    blocks_per_sm = (num_blocks + active_sms - 1) / active_sms; // ignores thread max
    sm_iterations = ((blocks_per_sm * threads_per_block) + MAX_THREADS_PER_SM - 1) / MAX_THREADS_PER_SM;
    flop_throughput = MAX_FLOPS * (double(active_sms) / NUM_SM);
    int_throughput = MAX_INT_OPS * (double(active_sms) / NUM_SM);

    comp_time = sm_iterations * max((int_ops / int_throughput), (flops / flop_throughput));

    // Print info
    cout << "ops: " << ops << endl;
    cout << "grid_dim_x: " << grid_dim_x << endl;
    cout << "grid_dim_y: " << grid_dim_y << endl;
    cout << "grid_dim_z: " << grid_dim_z << endl;
    cout << "num_blocks: " << num_blocks << endl;
    cout << "threads_per_block: " << threads_per_block << endl;
    cout << "blocks_per_sm: " << blocks_per_sm << endl;
    cout << "active_sms: " << active_sms << endl;
    cout << "sm_iterations: " << sm_iterations << endl;
    cout << "flop_throughput: " << flop_throughput << endl;
    cout << "int_throughput: " << int_throughput << endl;
    cout << "comp_time (ms): " << comp_time*1000 << endl;
    cout << "----------\n\n" << endl;

    // MEMORY-BOUND TIME -------------------------------------------------------------------------------------------------------------------
    cout << "MEMORY BOUND TIME CALCULATIONS: ----------" << endl;

    // Data size calculations (Bytes)
    long synapse_size = 4 * Ky * Kx * Nn * Ni;
    long neuron_i_size = 4 * (Ny + Ky) * (Nx + Kx) * Ni;
    long neuron_n_size = 4 * Ny * Nx * Nn;
    cout << "synapse_size: " << synapse_size << " B\n";
    cout << "neuron_i_size: " << neuron_i_size << " B\n";
    cout << "neuron_n_size: " << neuron_n_size << " B\n";

    if (neuron_i_size / (active_sms * num_blocks) > SHARED_MEM_PER_SM) {
        cout << "WARNING: inputs size exceeds max allocation in shared memory, model will not be accurate.\n";
    }

    // Calculate Active Number of Blocks
    // Derived values
    long shared_mem_per_block_bytes = SHMEM_TILE_X * SHMEM_TILE_Y * sizeof(float);
    long warps_per_block = ceil(threads_per_block / 32.0);

    // Per-SM resource limits
    long max_blocks_by_threads = MAX_THREADS_PER_SM / threads_per_block;
    long max_blocks_by_shared_mem = SHARED_MEM_PER_SM / shared_mem_per_block_bytes;
    long max_blocks_by_warps = MAX_WARPS_PER_SM / warps_per_block;

    blocks_per_sm = min({max_blocks_by_threads,
                                  max_blocks_by_shared_mem,
                                  max_blocks_by_warps,
                                  MAX_BLOCKS_PER_SM});

    long blocks_at_a_time = blocks_per_sm * NUM_SM;
    cout << "Blocks at a time: " << blocks_at_a_time << endl;

    // --- Memory Read for Input Neurons (DRAM -> L1)
    // Method 1:
    // long double MR_DRAM_access_per_block = DRAM_TLB_HIT*ceil((SHMEM_TILE_X*SHMEM_TILE_Y*sizeof(float))/(L1_UPDATE_GRANULARITY));
    // long double MR_shared_mem_read_time = MR_DRAM_access_per_block  * grid_dim_x * grid_dim_y * grid_dim_z * Ni / blocks_at_a_time;
    
    // Method 2:
    long double input_bytes_total =
        grid_dim_x * grid_dim_y * grid_dim_z * Ni * SHMEM_TILE_Y * SHMEM_TILE_X * sizeof(float);

    long double MR_shared_mem_read_time = input_bytes_total / DRAM_BW; // in seconds
    cout << "Memory Read for Input Neurons: " << MR_shared_mem_read_time*1000 << endl;

    // --- Compute Related Memory Reads
    // Input Neuron Reads
    long double C_per_thread_input_neuron_read_time = Ky*Kx*Ni*SHAREDMEM_HIT;
    long double C_total_input_neuron_read_time = Nx*Ny*Nn*C_per_thread_input_neuron_read_time / (blocks_at_a_time * threads_per_block);
    cout << "Compute: Read for Input Neurons: " << C_total_input_neuron_read_time*1000 << endl;

    // Synapse Reads
    // Method 1:
    // long long weights_bytes = Ky*Kx*sizeof(float);
    // long double l2_lines = max(1.0,ceil(weights_bytes/64));
    // long double latency_per_line = L2_HIT + DRAM_TLB_HIT;
    // long total_unique_layers = Ni*Nn;
    // long Nn_active_at_once = floor(blocks_at_a_time / (grid_dim_x * grid_dim_y));
    // long double synapse_read_time = l2_lines * latency_per_line * total_unique_layers / max(1L, Nn_active_at_once);
    // synapse_read_time += (Nx*Ny*Ni*Nn - l2_lines*total_unique_layers) * L2_HIT / (blocks_at_a_time*threads_per_block);

    // Method 2: bandwidth-limited
    long double bytes_per_thread = Ky * Kx * Ni * sizeof(float);
    long double total_threads = NXSCL * NYSCL * Nn;
    long double total_synapse_bytes = bytes_per_thread * total_threads;
    long double synapse_read_time = total_synapse_bytes / DRAM_BW;

    cout << "Compute: Read for Synapses: " << (synapse_read_time)*1000 << endl;

    // --- Output Writes
    long double output_write = Nx*Ny*Ni*Nn*DRAM_TLB_HIT;


    // TOTAL TIME
    mem_time = MR_shared_mem_read_time + C_total_input_neuron_read_time + synapse_read_time;
    cout << "mem_time (ms): " << mem_time*1000 << endl;
    cout << "----------\n\n" << endl;

    // ANALYSIS --------------------------------------------------------------------------------------------------------------------------
    // Take upper bound for Tops
    outputs.TOps = ops / (max(mem_time, comp_time) * pow(10, 12));

    // Calculate operational intensity (same whether comp or mem bound)
    outputs.L1_OpIntensity = ops / neuron_i_size;
    outputs.L2_OpIntensity = ops / synapse_size;
    outputs.DRAM_OpIntensity = ops / (synapse_size + neuron_n_size);

    return outputs;
}


int main() {
    perf_model_out predictions;

    // Run performance model
    predictions = perf_model();

    // Print results
    cout << "FINAL OUTPUT RESULTS: ----------" << endl;
    cout << "Predicted TOps: " << predictions.TOps << endl;
    cout << "Predicted L1_OpIntensity: " << predictions.L1_OpIntensity << endl;
    cout << "Predicted L2_OpIntensity: " << predictions.L2_OpIntensity << endl;
    cout << "Predicted DRAM_OpIntensity: " << predictions.DRAM_OpIntensity << endl;
}
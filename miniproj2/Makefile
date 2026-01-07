CPP=g++
OPT?=-O3
CFLAGS= --std=c++11 -g -ggdb -gdwarf-3 $(OPT) -fsanitize=address
MODULE := conv1
PERF_EXE := perf_model
HEADERS := dnn.hpp

# === Macro Parameters (editable in one place) ===
Nx := 32
Ny := 32
Kx := 3
Ky := 3
Ni := 32
Nn := 32
Tn := 16
BLOCK_X := 32
BLOCK_Y := 32

# === Derived Macro Flags ===
MACROS := -DNx=$(Nx) -DNy=$(Ny) -DKx=$(Kx) -DKy=$(Ky) -DNi=$(Ni) -DNn=$(Nn) -DTn=$(Tn) -DBLOCK_X=$(BLOCK_X) -DBLOCK_Y=$(BLOCK_Y)

.PHONY: all clean perf_test channel_sweep size_sweep kernel_sweep block_sweep

all: $(MODULE) $(PERF_EXE) channel_sweep size_sweep kernel_sweep block_sweep

# === Build and run the perf_model ===
$(PERF_EXE): perf_model.cpp
	$(CPP) $(CFLAGS) -o $(PERF_EXE) perf_model.cpp $(MACROS)

perf_test: $(PERF_EXE)
	./$(PERF_EXE)

# === Build the CUDA convolution program ===
$(MODULE): convolution.cu
	nvcc $^ -o $@ $(MACROS) -lcudnn

benchmark_test: $(PERF_EXE) $(MODULE)
	@echo "Running perf_model (CPU model)"
	./$(PERF_EXE)
	@echo ""
	@echo "Running conv1 with nvprof (GPU)"
	nvprof -m flop_count_sp,inst_integer,sm_efficiency,achieved_occupancy \
	       -m dram_read_throughput,dram_write_throughput,dram_utilization \
	       -m gld_efficiency,stall_inst_fetch,stall_memory_dependency \
	       -m stall_memory_throttle,stall_not_selected,stall_other,stall_pipe_busy,stall_sync \
	       ./$(MODULE)


clean:
	@rm -f $(MODULE) $(PERF_EXE)
	@rm -rf storedOutputs/

# Sweep over input/output channels (powers of 2)
channel_sweep:
	@mkdir -p storedOutputs/channel_sweep
	@for N in 2 4 8 16 32 64 128 256 512; do \
	  echo "Building with Ni = Nn = $$N"; \
		OUT_PREFIX=storedOutputs/channel_sweep/out_N$$N; \
		echo "Compiling perf_model..."; \
		$(CPP) $(CFLAGS) -o $(PERF_EXE) perf_model.cpp -DNx=$(Nx) -DNy=$(Nx) -DKx=$(Kx) -DKy=$(Ky) -DNi=$$N -DNn=$$N -DTn=$(Tn) -DBLOCK_X=$(BLOCK_X) -DBLOCK_Y=$(BLOCK_Y); \
		echo "Compiling conv1..."; \
	  nvcc convolution.cu -o conv1 -DNx=$(Nx) -DNy=$(Ny) -DKx=$(Kx) -DKy=$(Ky) -DNi=$$N -DNn=$$N -DTn=$(Tn) -DBLOCK_X=$(BLOCK_X) -DBLOCK_Y=$(BLOCK_Y) -lcudnn; \
	  echo "Running perf_model (CPU)..."; \
	  ./$(PERF_EXE) > $$OUT_PREFIX\_cpu.txt 2>&1; \
		echo "Running conv1 with nvprof (GPU)..."; \
	  nvprof -m flop_count_sp,inst_integer,sm_efficiency,achieved_occupancy,dram_read_throughput,dram_write_throughput,dram_utilization,gld_efficiency,stall_inst_fetch,stall_memory_dependency,stall_memory_throttle,stall_not_selected,stall_other,stall_pipe_busy,stall_sync ./conv1 > $$OUT_PREFIX\_gpu.txt 2>&1; \
		nvprof ./conv1 >> $$OUT_PREFIX\_gpu.txt 2>&1; \
		echo "Done with Ni = Nn = $$N"; \
	  echo ""; \
	done

# Sweep over Nx = Ny sizes
size_sweep:
	@mkdir -p storedOutputs/size_sweep
	@for NXY in 14 28 56 112 256; do \
	  echo "Building with Nx = Ny = $$NXY"; \
		OUT_PREFIX=storedOutputs/size_sweep/out_NXY$$NXY; \
		echo "Compiling perf_model..."; \
		$(CPP) $(CFLAGS) -o $(PERF_EXE) perf_model.cpp -DNx=$$NXY -DNy=$$NXY -DKx=$(Kx) -DKy=$(Ky) -DNi=$(Ni) -DNn=$(Nn) -DTn=$(Tn) -DBLOCK_X=$(BLOCK_X) -DBLOCK_Y=$(BLOCK_Y); \
		echo "Compiling conv1..."; \
	  nvcc convolution.cu -o conv1 -DNx=$$NXY -DNy=$$NXY -DKx=$(Kx) -DKy=$(Ky) -DNi=$(Ni) -DNn=$(Nn) -DTn=$(Tn) -DBLOCK_X=$(BLOCK_X) -DBLOCK_Y=$(BLOCK_Y) -lcudnn; \
	  echo "Running perf_model (CPU)..."; \
	  ./$(PERF_EXE) > $$OUT_PREFIX\_cpu.txt 2>&1; \
		echo "Running conv1 with nvprof (GPU)..."; \
	  nvprof -m flop_count_sp,inst_integer,sm_efficiency,achieved_occupancy,dram_read_throughput,dram_write_throughput,dram_utilization,gld_efficiency,stall_inst_fetch,stall_memory_dependency,stall_memory_throttle,stall_not_selected,stall_other,stall_pipe_busy,stall_sync ./conv1 > $$OUT_PREFIX\_gpu.txt 2>&1; \
		nvprof ./conv1 >> $$OUT_PREFIX\_gpu.txt 2>&1; \
		echo "Done with Nx = Ny = $$NXY"; \
	  echo ""; \
	done


# Sweep over kernel size Kx = Ky (odd values typically used)
kernel_sweep:
	@mkdir -p storedOutputs/kernel_sweep
	@for K in 3 5 7 9 11; do \
	  echo "Building with Kx = Ky = $$K"; \
		OUT_PREFIX=storedOutputs/kernel_sweep/out_K$$K; \
		echo "Compiling perf_model..."; \
		$(CPP) $(CFLAGS) -o $(PERF_EXE) perf_model.cpp -DNx=$(Nx) -DNy=$(Ny) -DKx=$$K -DKy=$$K -DNi=$(Ni) -DNn=$(Nn) -DTn=$(Tn) -DBLOCK_X=$(BLOCK_X) -DBLOCK_Y=$(BLOCK_Y); \
		echo "Compiling conv1..."; \
	  nvcc convolution.cu -o conv1 -DNx=$(Nx) -DNy=$(Ny) -DKx=$$K -DKy=$$K -DNi=$(Ni) -DNn=$(Nn) -DTn=$(Tn) -DBLOCK_X=$(BLOCK_X) -DBLOCK_Y=$(BLOCK_Y) -lcudnn; \
	  echo "Running perf_model (CPU)..."; \
	  ./$(PERF_EXE) > $$OUT_PREFIX\_cpu.txt 2>&1; \
		echo "Running conv1 with nvprof (GPU)..."; \
	  nvprof -m flop_count_sp,inst_integer,sm_efficiency,achieved_occupancy,dram_read_throughput,dram_write_throughput,dram_utilization,gld_efficiency,stall_inst_fetch,stall_memory_dependency,stall_memory_throttle,stall_not_selected,stall_other,stall_pipe_busy,stall_sync ./conv1 > $$OUT_PREFIX\_gpu.txt 2>&1; \
		nvprof ./conv1 >> $$OUT_PREFIX\_gpu.txt 2>&1; \
		echo "Done with Kx = Ky = $$K"; \
	  echo ""; \
	done

# Sweep over block sizes (BLOCK_X and BLOCK_Y)
block_sweep:
	@mkdir -p storedOutputs/block_sweep
	@for B in 4 8 16 32; do \
	  echo "Building with BLOCK_X = BLOCK_Y = $$B"; \
	  OUT_PREFIX=storedOutputs/block_sweep/out_B$$B; \
	  echo "Compiling perf_model..."; \
	  $(CPP) $(CFLAGS) -o $(PERF_EXE) perf_model.cpp -DNx=$(Nx) -DNy=$(Ny) -DKx=$(Kx) -DKy=$(Ky) -DNi=$(Ni) -DNn=$(Nn) -DTn=$(Tn) -DBLOCK_X=$$B -DBLOCK_Y=$$B; \
	  echo "Compiling conv1..."; \
	  nvcc convolution.cu -o conv1 -DNx=$(Nx) -DNy=$(Ny) -DKx=$(Kx) -DKy=$(Ky) -DNi=$(Ni) -DNn=$(Nn) -DTn=$(Tn) -DBLOCK_X=$$B -DBLOCK_Y=$$B -lcudnn; \
	  echo "Running perf_model (CPU)..."; \
	  ./$(PERF_EXE) > $$OUT_PREFIX\_cpu.txt 2>&1; \
	  echo "Running conv1 with nvprof (GPU)..."; \
	  nvprof -m flop_count_sp,inst_integer,sm_efficiency,achieved_occupancy,dram_read_throughput,dram_write_throughput,dram_utilization,gld_efficiency,stall_inst_fetch,stall_memory_dependency,stall_memory_throttle,stall_not_selected,stall_other,stall_pipe_busy,stall_sync ./conv1 > $$OUT_PREFIX\_gpu.txt 2>&1; \
	  nvprof ./conv1 >> $$OUT_PREFIX\_gpu.txt 2>&1; \
		echo "Done with BLOCK_X = BLOCK_Y = $$B"; \
	  echo ""; \
	done

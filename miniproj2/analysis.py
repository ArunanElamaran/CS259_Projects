import os
import re
import math

def extract_metrics(filepath):
    flop_count = 0
    int_count = 0
    sci_number = r'([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*$'
    with open(filepath, 'r') as f:
        for line in f:
            if 'flop_count_sp' in line:
                match = re.findall(sci_number, line)
                if match:
                    flop_count = float(match[0])
            elif 'inst_integer' in line:
                match = re.findall(sci_number, line)
                if match:
                    int_count = float(match[0])
    return flop_count, int_count

def collect_data(base_dir):
    data_by_folder = {}
    for root, dirs, files in os.walk(base_dir):
        folder = os.path.basename(root)
        for file in files:
            if file.endswith('.txt'):
                path = os.path.join(root, file)
                match = re.search(r'out_(\d+)\.txt', file)
                if match:
                    x_val = int(match.group(1))
                    flops, ints = extract_metrics(path)
                    if folder not in data_by_folder:
                        data_by_folder[folder] = []
                    data_by_folder[folder].append((x_val, flops, ints))
    return data_by_folder

def smart_axis_labels(folder):
    if folder == 'channel_sweep':
        return 'Input/Output Channels (Ni = Nn)'
    elif folder == 'size_sweep':
        return 'Input Width/Height (Nx = Ny)'
    elif folder == 'kernel_sweep':
        return 'Kernel Size (Kx = Ky)'
    else:
        return 'Parameter Value from Filename'

def estimate_int_ops(Nx, Ny, Kx, Ky, Ni, Nn, BLOCK_X, BLOCK_Y, Sx=1, Sy=1):
    SHMEM_TILE_X = BLOCK_X + Kx - 1
    SHMEM_TILE_Y = BLOCK_Y + Ky - 1
    NXSCL = Nx // Sx
    NYSCL = Ny // Sy
    NXPAD = Nx+Kx
    NYPAD = Ny+Ky
    O = NXSCL * NYSCL * Nn
    L_shmem = math.ceil((SHMEM_TILE_X * SHMEM_TILE_Y) / (BLOCK_X * BLOCK_Y))

    int_ops = NXSCL * NYSCL * Nn * (8 + Ni * (6 * SHMEM_TILE_Y * SHMEM_TILE_X + 9 * Ky * Kx + 1 * (SHMEM_TILE_Y / BLOCK_Y) * (SHMEM_TILE_X / BLOCK_X)))
    int_ops /= 60 * 1.69*(32/16)


    # # // Threads and blocks
    # threads = NXSCL * NYSCL * Nn
    # blocks_x = (NXSCL + BLOCK_X - 1) / BLOCK_X
    # blocks_y = (NYSCL + BLOCK_Y - 1) / BLOCK_Y
    # blocks = blocks_x * blocks_y * Nn
    # # // Per-thread cost: convolution, indexing, loop logic
    # per_thread = threads * (8 + Ni * (6 * Ky * Kx + 3 * (SHMEM_TILE_Y / BLOCK_Y) * (SHMEM_TILE_X / BLOCK_X)))
    # # // Per-block cost: shared memory load
    # per_block = blocks * (Ni * 6 * SHMEM_TILE_Y * SHMEM_TILE_X)
    # # // Combined
    # int_ops = per_thread + per_block
    # norm = 1+(Ky - 3)*0.4
    # int_ops /= norm * 1.845*(32/16)


    return int_ops

def plot_data(data_by_folder):
    # Default constants (change as needed)
    defaults = {
        'Nx': 14,
        'Ny': 14,
        'Kx': 3,
        'Ky': 3,
        'Ni': 512,
        'Nn': 512,
        'BLOCK_X': 32,
        'BLOCK_Y': 32
    }

    for folder, data in data_by_folder.items():
        data.sort()
        x_label = smart_axis_labels(folder)

        print(f"\n=== {folder.upper()} ===")
        print(f"{x_label:<35} | INT Ops (Actual) | INT Ops (Est.) | Factor (Est / Actual)")
        print("-" * 95)

        for x, flops, ints in data:
            # Copy defaults
            params = defaults.copy()

            # Vary based on folder
            if folder == 'channel_sweep':
                params['Ni'] = x
                params['Nn'] = x
            elif folder == 'size_sweep':
                params['Nx'] = x
                params['Ny'] = x
            elif folder == 'kernel_sweep':
                params['Kx'] = x
                params['Ky'] = x

            # Estimate
            est_ints = estimate_int_ops(
                params['Nx'], params['Ny'],
                params['Kx'], params['Ky'],
                params['Ni'], params['Nn'],
                params['BLOCK_X'], params['BLOCK_Y']
            )

            factor = est_ints / ints if ints != 0 else float('nan')

            print(f"{x:<35} | {ints:<16.2f} | {est_ints:<15.2f} | {factor:.6f}")

if __name__ == "__main__":
    data = collect_data('storedOutputs')
    plot_data(data)

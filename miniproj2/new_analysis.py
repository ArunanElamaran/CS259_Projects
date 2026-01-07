import os
import re

def extract_param_value(filename, param, device):
    """
    Extracts the numeric value for a given parameter from the filename.
    Expected format: out_<param><number>_<device>.txt
    Example: out_N2_cpu.txt with param='N' → 2
    """
    pattern = rf"out_{param}(\d+)_({device})\.txt"
    match = re.match(pattern, filename)
    if match:
        return int(match.group(1))
    return None


def extract_cpu_metrics(directory, keywords, sweep_param="N"):
    N_values = []
    metrics = {key: [] for key in keywords}

    for filename in os.listdir(directory):
        if "_cpu.txt" not in filename:
            continue

        N = extract_param_value(filename, sweep_param, "cpu")
        if N is None:
            continue

        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as f:
            content = f.readlines()

        found_values = {}
        for line in content:
            for key in keywords:
                if line.strip().startswith(f"{key}:"):
                    value_str = line.split(":", 1)[1].strip().split()[0]
                    try:
                        value = float(value_str) if re.search(r'[.eE]', value_str) else int(value_str)
                    except ValueError:
                        value = None
                    found_values[key] = value

        if all(key in found_values for key in keywords):
            N_values.append(N)
            for key in keywords:
                metrics[key].append(found_values[key])

    sorted_indices = sorted(range(len(N_values)), key=lambda i: N_values[i])
    N_values_sorted = [N_values[i] for i in sorted_indices]
    sorted_metrics = {key: [metrics[key][i] for i in sorted_indices] for key in keywords}

    return N_values_sorted, sorted_metrics


def extract_gpu_metrics(directory, keywords, sweep_param="N"):
    N_values = []
    metrics = {key: [] for key in keywords}

    for filename in os.listdir(directory):
        if "_gpu.txt" not in filename:
            continue

        N = extract_param_value(filename, sweep_param, "gpu")
        if N is None:
            continue

        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as f:
            content = f.readlines()

        found_values = {}
        for line in content:
            for key in keywords:
                if key in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        avg_value_str = parts[-1]
                        try:
                            value = float(re.sub(r"[^\d\.eE+-]", "", avg_value_str))
                        except ValueError:
                            value = None
                        found_values[key] = value
        for line in content:
            if 'convolution_layer' in line:
                # Match both us and ms formats
                match = re.search(
                    r'\s+([\d\.]+)(us|ms)\s+\d+\s+[\d\.]+(us|ms)\s+[\d\.]+(us|ms)\s+[\d\.]+(us|ms)\s+convolution_layer',
                    line
                )
                if match:
                    value = float(match.group(1))
                    unit = match.group(2)
                    conv_time_ms = value if unit == 'ms' else value / 1000
                    found_values["convolution_layer_time (ms)"] = conv_time_ms

        if all(key in found_values for key in keywords):
            N_values.append(N)
            for key in keywords:
                metrics[key].append(found_values[key])

    sorted_indices = sorted(range(len(N_values)), key=lambda i: N_values[i])
    N_values_sorted = [N_values[i] for i in sorted_indices]
    sorted_metrics = {key: [metrics[key][i] for i in sorted_indices] for key in keywords}

    return N_values_sorted, sorted_metrics


def print_metrics_csv(N_values, metrics, title):
    header = ["Param"] + [str(n) for n in N_values]
    print(",".join(header))
    for key, values in metrics.items():
        row = [key] + [f"{v:.6f}" if isinstance(v, float) else str(v) for v in values]
        print(",".join(row))




if __name__ == "__main__":
    # dir_path = "storedOutputs/block_sweep"
    # sweep_param = "B"
    # dir_path = "storedOutputs/channel_sweep"
    # sweep_param = "N"
    dir_path = "storedOutputs/kernel_sweep"
    sweep_param = "K"
    # dir_path = "storedOutputs/size_sweep"
    # sweep_param = "NXY"

    cpu_keys = ["FLOP", "Integer OP", "Total OP", "comp_time (ms)", "mem_time (ms)", "Predicted TOps", "Predicted L1_OpIntensity", "Predicted L2_OpIntensity", "Predicted DRAM_OpIntensity"]
    gpu_keys = ["flop_count_sp", "inst_integer", "convolution_layer_time (ms)"]

    N_cpu, cpu_metrics = extract_cpu_metrics(dir_path, cpu_keys, sweep_param=sweep_param)
    print_metrics_csv(N_cpu, cpu_metrics, "CPU Results")

    N_gpu, gpu_metrics = extract_gpu_metrics(dir_path, gpu_keys, sweep_param=sweep_param)
    print("\nGPU Results:")
    print_metrics_csv(N_gpu, gpu_metrics, "GPU Results")
        

# profile_data = {
#     "payload": <payload size in bytes>,
#     "comm_time": [t0, t1, t2, ...],
#     "total_time": <dt in milliseconds>,
# }

import json
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
import re

def parse_payload_size(dir_name):
    """Parse payload size from directory name (e.g., '16mb' -> 16*1000*1000)"""
    dir_name = dir_name.lower()
    if dir_name.endswith('kb'):
        return int(dir_name[:-2]) * 1000
    elif dir_name.endswith('mb'):
        return int(dir_name[:-2]) * 1000 * 1000
    elif dir_name.endswith('gb'):
        return int(dir_name[:-2]) * 1000 * 1000 * 1000
    else:
        return int(dir_name)

def load_profile_data(profile_dir="profile"):
    """Load all profile data from the profile directory"""
    data = {}
    
    if not os.path.exists(profile_dir):
        print(f"Profile directory '{profile_dir}' not found. Please run the tests first.")
        return data
    
    for subdir in os.listdir(profile_dir):
        subdir_path = os.path.join(profile_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
            
        # Parse payload size from directory name
        try:
            payload_size = parse_payload_size(subdir)
        except ValueError:
            print(f"Skipping directory with invalid name: {subdir}")
            continue
            
        data[payload_size] = {
            'reduce_scatter_accumulation': [],      # Your implementation
            'reduce_scatter_accumulation_nccl': []  # NCCL implementation
        }
        
        # Load all JSON files in this subdirectory
        for filename in os.listdir(subdir_path):
            if not filename.endswith('.json'):
                continue
                
            # Parse filename: <func>-<payload>-<rank>.json
            match = re.match(r'(.+)-(\d+)-(\d+)\.json', filename)
            if not match:
                continue
                
            func_name, payload, rank = match.groups()
            payload = int(payload)
            
            if func_name not in data[payload_size]:
                print(f"Unknown function name: {func_name}")
                continue
                
            # Load JSON data
            filepath = os.path.join(subdir_path, filename)
            try:
                with open(filepath, 'r') as f:
                    profile_data = json.load(f)
                    data[payload_size][func_name].append(profile_data)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    
    return data

def analyze_data(data):
    """Analyze the loaded data and compute statistics"""
    results = {}
    
    for payload_size, implementations in data.items():
        results[payload_size] = {}
        
        for impl_name, profiles in implementations.items():
            if not profiles:
                continue
                
            # Extract communication times
            comm_times = []
            total_times = []
            reduce_scatter_payloads = []
            
            for profile in profiles:
                comm_times.extend(profile['comm_time'])
                total_times.append(profile['total_time'])
                reduce_scatter_payloads.append(profile['payload'])
            
            # Compute statistics
            comm_times = np.array(comm_times)
            total_times = np.array(total_times)
            reduce_scatter_payloads = np.array(reduce_scatter_payloads)
            assert np.all(reduce_scatter_payloads == reduce_scatter_payloads[0])
            
            # Convert payload to MB for bandwidth calculation
            payload_mb = reduce_scatter_payloads[0] / (1000 * 1000)
            # payload_mb = payload_size / (1000 * 1000)
            
            # Calculate bandwidth (MB/s)
            # comm_time is in milliseconds, so convert to seconds
            # reduce_scatter_payload = payload_mb // group_size* (group_size - 1)* data[0].dtype.itemsize
            bandwidths = (payload_mb / (comm_times / 1000))
            
            # print(f"Payload size: {payload_mb} MB, Comm times: {comm_times[:10]}, bandwidths: {bandwidths[:10]}")
            # if payload_size == 64 * 1000 * 1000:
                # print(f"Payload size: {payload_mb} MB, total_times: {total_times}, Comm times: {np.mean(comm_times)}, bandwidths: {np.mean(bandwidths)} mean bandwidth: {payload_mb * 1000 / np.mean(comm_times)} MB/s")
            # TODO: add total_bandwidth
            results[payload_size][impl_name] = {
                'comm_times': comm_times,
                'total_times': total_times,
                'payloads': reduce_scatter_payloads,
                'payload_mb': payload_mb,
                'bandwidths': bandwidths,
                'avg_comm_time': np.mean(comm_times),
                'std_comm_time': np.std(comm_times),
                'avg_bandwidth': np.mean(bandwidths),
                'std_bandwidth': np.std(bandwidths),
                'avg_total_time': np.mean(total_times),
                'std_total_time': np.std(total_times)
            }
    
    return results

def plot_results(results):
    """Create comprehensive plots comparing the implementations"""
    if not results:
        print("No data to plot")
        return
    
    # Sort payload sizes for proper x-axis ordering
    payload_sizes = sorted(results.keys())
    payload_labels = [f"{size/(1000*1000):.0f}MB" for size in payload_sizes]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Reduce-Scatte rPerformance Comparison: Custom vs NCCL', fontsize=16)
    
    # Colors for implementations
    colors = {'reduce_scatter_accumulation': 'blue', 'reduce_scatter_accumulation_nccl': 'red'}
    labels = {'reduce_scatter_accumulation': 'Custom Implementation', 'reduce_scatter_accumulation_nccl': 'NCCL'}
    
    # Plot 1: Average Communication Time
    ax1.set_title('Average Communication Time')
    ax1.set_xlabel('Payload Size')
    ax1.set_ylabel('Time (ms)')
    ax1.set_xticks(range(len(payload_sizes)))
    ax1.set_xticklabels(payload_labels)
    
    for impl_name in ['reduce_scatter_accumulation', 'reduce_scatter_accumulation_nccl']:
        comm_times = []
        comm_stds = []
        for payload_size in payload_sizes:
            if impl_name in results[payload_size]:
                comm_times.append(results[payload_size][impl_name]['avg_comm_time'])
                comm_stds.append(results[payload_size][impl_name]['std_comm_time'])
            else:
                comm_times.append(0)
                comm_stds.append(0)
        
        ax1.errorbar(range(len(payload_sizes)), comm_times, yerr=comm_stds, 
                    marker='o', color=colors[impl_name], label=labels[impl_name], capsize=5)
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average Bandwidth
    ax2.set_title('Average Bandwidth')
    ax2.set_xlabel('Payload Size')
    ax2.set_ylabel('Bandwidth (MB/s)')
    ax2.set_xticks(range(len(payload_sizes)))
    ax2.set_xticklabels(payload_labels)
    
    for impl_name in ['reduce_scatter_accumulation', 'reduce_scatter_accumulation_nccl']:
        bandwidths = []
        bandwidth_stds = []
        for payload_size in payload_sizes:
            if impl_name in results[payload_size]:
                bandwidths.append(results[payload_size][impl_name]['avg_bandwidth'])
                bandwidth_stds.append(results[payload_size][impl_name]['std_bandwidth'])
            else:
                bandwidths.append(0)
                bandwidth_stds.append(0)
        
        ax2.errorbar(range(len(payload_sizes)), bandwidths, yerr=bandwidth_stds, 
                    marker='s', color=colors[impl_name], label=labels[impl_name], capsize=5)
    
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Total Time Comparison
    ax3.set_title('Average Total Time')
    ax3.set_xlabel('Payload Size')
    ax3.set_ylabel('Time (ms)')
    ax3.set_xticks(range(len(payload_sizes)))
    ax3.set_xticklabels(payload_labels)
    
    for impl_name in ['reduce_scatter_accumulation', 'reduce_scatter_accumulation_nccl']:
        total_times = []
        total_stds = []
        for payload_size in payload_sizes:
            if impl_name in results[payload_size]:
                total_times.append(results[payload_size][impl_name]['avg_total_time'])
                total_stds.append(results[payload_size][impl_name]['std_total_time'])
            else:
                total_times.append(0)
                total_stds.append(0)
        
        ax3.errorbar(range(len(payload_sizes)), total_times, yerr=total_stds, 
                    marker='^', color=colors[impl_name], label=labels[impl_name], capsize=5)
    
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance Ratio (Custom/NCCL)
    ax4.set_title('Performance Ratio (Custom / NCCL)')
    ax4.set_xlabel('Payload Size')
    ax4.set_ylabel('Ratio (Custom/NCCL)')
    ax4.set_xticks(range(len(payload_sizes)))
    ax4.set_xticklabels(payload_labels)
    ax4.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Equal Performance')
    
    ratios = []
    for payload_size in payload_sizes:
        if ('reduce_scatter_accumulation' in results[payload_size] and 
            'reduce_scatter_accumulation_nccl' in results[payload_size]):
            custom_time = results[payload_size]['reduce_scatter_accumulation']['avg_comm_time']
            nccl_time = results[payload_size]['reduce_scatter_accumulation_nccl']['avg_comm_time']
            if nccl_time > 0:
                ratio = custom_time / nccl_time
                ratios.append(ratio)
            else:
                ratios.append(1.0)
        else:
            ratios.append(1.0)
    
    ax4.plot(range(len(payload_sizes)), ratios, marker='D', color='green', linewidth=2, markersize=8)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def print_summary(results):
    """Print a summary of the results"""
    print("\n" + "="*80)
    print("REDUCE-SCATTER PERFORMANCE ANALYSIS SUMMARY")
    print("="*80)
    
    if not results:
        print("No data available for analysis")
        return
    
    # Sort payload sizes
    payload_sizes = sorted(results.keys())
    
    for payload_size in payload_sizes:
        print(f"\nPayload Size: {payload_size/(1000*1000):.1f} MB ({payload_size:,} bytes)")
        print("-" * 60)
        
        for impl_name in ['reduce_scatter_accumulation', 'reduce_scatter_accumulation_nccl']:
            if impl_name in results[payload_size]:
                data = results[payload_size][impl_name]
                impl_label = "Custom Implementation" if impl_name == "reduce_scatter_accumulation" else "NCCL"
                
                print(f"{impl_label}:")
                print(f"  Avg Comm Time: {data['avg_comm_time']:.3f} ± {data['std_comm_time']:.3f} ms")
                print(f"  Avg Bandwidth: {data['avg_bandwidth']:.1f} ± {data['std_bandwidth']:.1f} MB/s")
                print(f"  Avg Total Time: {data['avg_total_time']:.3f} ± {data['std_total_time']:.3f} ms")
        
        # Compare performance if both implementations exist
        if ('reduce_scatter_accumulation' in results[payload_size] and 
            'reduce_scatter_accumulation_nccl' in results[payload_size]):
            custom_time = results[payload_size]['reduce_scatter_accumulation']['avg_comm_time']
            nccl_time = results[payload_size]['reduce_scatter_accumulation_nccl']['avg_comm_time']
            
            if nccl_time > 0:
                ratio = custom_time / nccl_time
                if ratio < 1.0:
                    print(f"  Performance: Custom is {1/ratio:.2f}x FASTER than NCCL")
                elif ratio > 1.0:
                    print(f"  Performance: Custom is {ratio:.2f}x SLOWER than NCCL")
                else:
                    print(f"  Performance: Custom and NCCL are EQUAL")
            print()

def main(data_dir):
    """Main function to run the analysis"""
    print("Loading profile data...")
    data = load_profile_data(data_dir)
    
    if not data:
        print("No profile data found. Please run the tests first using:")
        print("  bash launch_profile.sh")
        return
    
    print(f"Loaded data for {len(data)} payload sizes")
    
    print("Analyzing data...")
    results = analyze_data(data)
    
    print("Creating plots...")
    fig = plot_results(results)
    
    print("Printing summary...")
    print_summary(results)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    # data_dir = "proc-profile2"
    # data_dir = "multi-buf-profile"
    data_dir = "fxbuf-profile"
    main(data_dir)

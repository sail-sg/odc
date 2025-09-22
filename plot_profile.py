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
import csv
import pandas as pd

def parse_input_size(dir_name):
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

def load_profile_data(profile_dir="profile", operation_type="reduce_scatter"):
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
            input_size = parse_input_size(subdir)
        except ValueError:
            print(f"Skipping directory with invalid name: {subdir}")
            continue
            
        # Initialize data structure based on operation type
        if operation_type == "reduce_scatter":
            data[input_size] = {
                'reduce_scatter_accumulation': [],      # Your implementation
                'reduce_scatter_accumulation_nccl': []  # NCCL implementation
            }
        elif operation_type == "all_gather":
            data[input_size] = {
                'all_gather_into_tensor': [],      # Your implementation
                'all_gather_into_tensor_nccl': []  # NCCL implementation
            }
        
        # Load all JSON files in this subdirectory
        for filename in os.listdir(subdir_path):
            if not filename.endswith('.json'):
                continue
                
            # Parse filename: <func>-<payload>-<num_nodes>-<num_ranks>-<rank>.json
            match = re.match(r'(.+)-(\d+)-(\d+)-(\d+)-(\d+)\.json', filename)
            assert match, f"Invalid filename: {filename}"
            func_name, payload, num_nodes, num_ranks, rank = match.groups()
            payload = int(payload)
            num_nodes = int(num_nodes)
            num_ranks = int(num_ranks)
            
            if func_name not in data[input_size]:
                # print(f"Unknown function name: {func_name}")
                continue
                
            # Load JSON data
            filepath = os.path.join(subdir_path, filename)
            try:
                with open(filepath, 'r') as f:
                    profile_data = json.load(f)
                    # Add num_nodes and num_ranks to profile data
                    profile_data['num_nodes'] = num_nodes
                    profile_data['num_ranks'] = num_ranks
                    data[input_size][func_name].append(profile_data)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    
    return data

def analyze_data(data):
    """Analyze the loaded data and compute statistics"""
    results = {}
    
    for payload_size, implementations in data.items():
        # Group by (num_nodes, num_ranks) first
        grouped_data = {}
        
        for impl_name, profiles in implementations.items():
            if not profiles:
                continue
            
            # Group profiles by (num_nodes, num_ranks)
            for profile in profiles:
                num_nodes = profile.get('num_nodes', 1)
                num_ranks = profile.get('num_ranks', 1)
                key = (num_nodes, num_ranks, payload_size)
                
                if key not in grouped_data:
                    grouped_data[key] = {}
                if impl_name not in grouped_data[key]:
                    grouped_data[key][impl_name] = []
                
                grouped_data[key][impl_name].append(profile)
        
        # Process each group
        for key, impl_data in grouped_data.items():
            if key not in results:
                results[key] = {}
            
            for impl_name, profiles in impl_data.items():
                if not profiles:
                    continue
                    
                # Extract communication times
                comm_times = []
                total_times = []
                operation_payloads = []
                
                total_count = None
                for profile in profiles:
                    if total_count is None:
                        total_count = len(profile['comm_time'])
                    else:
                        assert total_count == len(profile['comm_time'])
                    comm_times.extend(profile['comm_time'])
                    total_times.append(profile['total_time'])
                    operation_payloads.append(profile['payload'])
                
                # Compute statistics
                comm_times = np.array(comm_times)
                total_times = np.array(total_times)
                operation_payloads = np.array(operation_payloads)
                assert np.all(operation_payloads == operation_payloads[0])
                
                # Convert payload to MB for bandwidth calculation
                payload_mb = operation_payloads[0] / (1000 * 1000)
                
                # Calculate bandwidth (MB/s)
                # comm_time is in milliseconds, so convert to seconds
                bandwidths = (payload_mb / (comm_times / 1000))
                
                total_bandwidth = payload_mb * total_count / total_times
                results[key][impl_name] = {
                    'comm_times': comm_times,
                    'total_times': total_times,
                    'payloads': operation_payloads,
                    'payload_mb': payload_mb,
                    'bandwidths': bandwidths,
                    'avg_comm_time': np.mean(comm_times),
                    'std_comm_time': np.std(comm_times),
                    'avg_bandwidth': np.mean(bandwidths),
                    'std_bandwidth': np.std(bandwidths),
                    'avg_total_time': np.mean(total_times),
                    'std_total_time': np.std(total_times),
                    'avg_total_bandwidth': np.mean(total_bandwidth),
                    'std_total_bandwidth': np.std(total_bandwidth),
                    'num_nodes': profiles[0]['num_nodes'],
                    'num_ranks': profiles[0]['num_ranks'],
                }
    
    return results

def plot_results(results, operation_type="reduce_scatter"):
    """Create comprehensive plots comparing the implementations"""
    if not results:
        print("No data to plot")
        return
    
    # Group results by (num_nodes, num_ranks) for separate plots
    grouped_results = {}
    for (num_nodes, num_ranks, payload_size), impl_data in results.items():
        key = (num_nodes, num_ranks)
        if key not in grouped_results:
            grouped_results[key] = {}
        grouped_results[key][payload_size] = impl_data
    
    # Create subplots for each (num_nodes, num_ranks) combination
    num_groups = len(grouped_results)
    fig, axes = plt.subplots(2, num_groups, figsize=(6*num_groups, 10))
    if num_groups == 1:
        axes = axes.reshape(2, 1)
    
    # Set title based on operation type
    operation_label = "Reduce-Scatter" if operation_type == "reduce_scatter" else "All-Gather"
    fig.suptitle(f'{operation_label} Performance Comparison: ODC vs NCCL', fontsize=16)
    
    # Set implementation names and labels based on operation type
    if operation_type == "reduce_scatter":
        impl_names = ['reduce_scatter_accumulation', 'reduce_scatter_accumulation_nccl']
        base_labels = {'reduce_scatter_accumulation': 'ODC', 'reduce_scatter_accumulation_nccl': 'NCCL'}
    else:  # all_gather
        impl_names = ['all_gather_into_tensor', 'all_gather_into_tensor_nccl']
        base_labels = {'all_gather_into_tensor': 'ODC', 'all_gather_into_tensor_nccl': 'NCCL'}
    
    # Colors for implementations
    colors = {impl_names[0]: 'blue', impl_names[1]: 'red'}
    
    for idx, ((num_nodes, num_ranks), group_results) in enumerate(grouped_results.items()):
        # Sort payload sizes for proper x-axis ordering
        payload_sizes = sorted(group_results.keys())
        payload_labels = [f"{size/(1000*1000):.0f}MB" for size in payload_sizes]
        
        # Create labels with node and rank info
        labels = {}
        for impl_name in impl_names:
            base_label = base_labels[impl_name]
            if impl_name.startswith('reduce_scatter_accumulation') and not impl_name.endswith('_nccl'):
                labels[impl_name] = f"{base_label}-n{num_nodes}-r{num_ranks}"
            else:
                labels[impl_name] = f"{base_label}-n{num_nodes}-r{num_ranks}"
        
        # Plot 1: Average Communication Time
        ax1 = axes[0, idx]
        ax1.set_title(f'Comm Time (n{num_nodes}-r{num_ranks})')
        ax1.set_xlabel('Payload Size')
        ax1.set_ylabel('Time (ms)')
        ax1.set_xticks(range(len(payload_sizes)))
        ax1.set_xticklabels(payload_labels)
        
        for impl_name in impl_names:
            comm_times = []
            comm_stds = []
            for payload_size in payload_sizes:
                if impl_name in group_results[payload_size]:
                    comm_times.append(group_results[payload_size][impl_name]['avg_comm_time'])
                    comm_stds.append(group_results[payload_size][impl_name]['std_comm_time'])
                else:
                    comm_times.append(0)
                    comm_stds.append(0)
            
            ax1.errorbar(range(len(payload_sizes)), comm_times, yerr=comm_stds, 
                        marker='o', color=colors[impl_name], label=labels[impl_name], capsize=5)
        
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Average Bandwidth
        ax2 = axes[1, idx]
        ax2.set_title(f'Bandwidth (n{num_nodes}-r{num_ranks})')
        ax2.set_xlabel('Payload Size')
        ax2.set_ylabel('Bandwidth (MB/s)')
        ax2.set_xticks(range(len(payload_sizes)))
        ax2.set_xticklabels(payload_labels)
        
        for impl_name in impl_names:
            bandwidths = []
            bandwidth_stds = []
            for payload_size in payload_sizes:
                if impl_name in group_results[payload_size]:
                    bandwidths.append(group_results[payload_size][impl_name]['avg_bandwidth'])
                    bandwidth_stds.append(group_results[payload_size][impl_name]['std_bandwidth'])
                else:
                    bandwidths.append(0)
                    bandwidth_stds.append(0)
            
            ax2.errorbar(range(len(payload_sizes)), bandwidths, yerr=bandwidth_stds, 
                        marker='s', color=colors[impl_name], label=labels[impl_name], capsize=5)
        
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_multi_bandwidth(data_dirs, field, operation_type="reduce_scatter"):
    """Plot the field from multiple data directories, one chart per directory"""
    if not data_dirs:
        print("No data directories provided")
        return
    
    # Load and analyze data for each directory
    all_results = {}
    for data_dir in data_dirs:
        print(f"Loading data from {data_dir}...")
        data = load_profile_data(data_dir, operation_type)
        if data:
            results = analyze_data(data)
            all_results[data_dir] = results
            print(f"Loaded data for {len(data)} payload sizes from {data_dir}")
        else:
            print(f"No data found in {data_dir}")
    
    if not all_results:
        print("No data found in any directory")
        return
    
    # Group results by (num_nodes, num_ranks) for each directory
    grouped_all_results = {}
    for data_dir, results in all_results.items():
        grouped_results = {}
        for (num_nodes, num_ranks, payload_size), impl_data in results.items():
            key = (num_nodes, num_ranks)
            if key not in grouped_results:
                grouped_results[key] = {}
            grouped_results[key][payload_size] = impl_data
        grouped_all_results[data_dir] = grouped_results
    
    # Create subplots - one for each data directory and (num_nodes, num_ranks) combination
    num_dirs = len(all_results)
    max_groups = max(len(groups) for groups in grouped_all_results.values()) if grouped_all_results else 1
    # print(f"num_dirs: {num_dirs}, max_groups: {max_groups}")
    fig, axes = plt.subplots(num_dirs, max_groups, figsize=(6*max_groups, 5*num_dirs))
    if num_dirs == 1 and max_groups == 1:
        axes = [[axes]]
    elif num_dirs == 1:
        axes = [axes]
    elif max_groups == 1:
        axes = [[ax] for ax in axes]
    print(f"num_dirs: {num_dirs}, max_groups: {max_groups} axes: {axes}")
    
    # Set title based on operation type
    operation_label = "Reduce-Scatter" if operation_type == "reduce_scatter" else "All-Gather"
    fig.suptitle(f'{operation_label} Bus Bandwidth', fontsize=16)
    
    # Set implementation names and labels based on operation type
    if operation_type == "reduce_scatter":
        impl_names = ['reduce_scatter_accumulation', 'reduce_scatter_accumulation_nccl']
        base_labels = {'reduce_scatter_accumulation': 'ODC', 'reduce_scatter_accumulation_nccl': 'NCCL'}
    else:  # all_gather
        impl_names = ['all_gather_into_tensor', 'all_gather_into_tensor_nccl']
        base_labels = {'all_gather_into_tensor': 'ODC', 'all_gather_into_tensor_nccl': 'NCCL'}
    
    # Colors and markers for implementations
    colors = {impl_names[0]: 'blue', impl_names[1]: 'red'}
    markers = {impl_names[0]: 'o', impl_names[1]: 's'}
    
    for dir_idx, (data_dir, grouped_results) in enumerate(grouped_all_results.items()):
        # Sort groups by (num_nodes, num_ranks) for consistent ordering
        sorted_groups = sorted(grouped_results.items(), key=lambda x: (x[0][0], x[0][1]))
        for group_idx, ((num_nodes, num_ranks), group_data) in enumerate(sorted_groups):
            ax = axes[dir_idx][group_idx]  # if num_dirs > 1 else axes[group_idx]
            ax.set_title(f'{data_dir} (n{num_nodes}-r{num_ranks})')
            ax.set_xlabel('Payload Size')
            ax.set_ylabel('Total Bandwidth (GB/s)')
            
            # Sort payload sizes for proper x-axis ordering
            payload_sizes = sorted(group_data.keys())
            payload_labels = [f"{size/(1000*1000):.0f}MB" for size in payload_sizes]
            
            ax.set_xticks(range(len(payload_sizes)))
            ax.set_xticklabels(payload_labels)
            
            # Create labels with node and rank info
            labels = {}
            for impl_name in impl_names:
                base_label = base_labels[impl_name]
                labels[impl_name] = f"{base_label}-n{num_nodes}-r{num_ranks}"
            
            # Plot field for each implementation
            for impl_name in impl_names:
                bandwidths = []
                for payload_size in payload_sizes:
                    if impl_name in group_data[payload_size]:
                        # Convert from MB/s to GB/s
                        bandwidths.append(group_data[payload_size][impl_name][field] / 1000)
                    else:
                        bandwidths.append(0)
                
                ax.errorbar(range(len(payload_sizes)), bandwidths,
                           marker=markers[impl_name], color=colors[impl_name], label=labels[impl_name], capsize=5)
            
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def export_multi_bandwidth_to_csv(data_dirs, field, operation_type="reduce_scatter", filename=None):
    """Export multi-directory bandwidth comparison data to CSV"""
    if not data_dirs:
        print("No data directories provided")
        return
    
    # Load and analyze data for each directory
    all_results = {}
    for data_dir in data_dirs:
        print(f"Loading data from {data_dir}...")
        data = load_profile_data(data_dir, operation_type)
        if data:
            results = analyze_data(data)
            all_results[data_dir] = results
            print(f"Loaded data for {len(data)} payload sizes from {data_dir}")
        else:
            print(f"No data found in {data_dir}")
    
    if not all_results:
        print("No data found in any directory")
        return
    
    # Set default filename if not provided
    if filename is None:
        operation_label = "reduce_scatter" if operation_type == "reduce_scatter" else "all_gather"
        filename = f"{operation_label}_multi_bandwidth_comparison.csv"
    
    # Set implementation names and labels based on operation type
    if operation_type == "reduce_scatter":
        impl_names = ['reduce_scatter_accumulation', 'reduce_scatter_accumulation_nccl']
        impl_labels = {'reduce_scatter_accumulation': 'ODC', 'reduce_scatter_accumulation_nccl': 'NCCL'}
    else:  # all_gather
        impl_names = ['all_gather_into_tensor', 'all_gather_into_tensor_nccl']
        impl_labels = {'all_gather_into_tensor': 'ODC', 'all_gather_into_tensor_nccl': 'NCCL'}
    
    # Prepare data for CSV export
    csv_data = []
    
    for data_dir, results in all_results.items():
        # Group results by (num_nodes, num_ranks)
        grouped_results = {}
        for (num_nodes, num_ranks, payload_size), impl_data in results.items():
            key = (num_nodes, num_ranks)
            if key not in grouped_results:
                grouped_results[key] = {}
            grouped_results[key][payload_size] = impl_data
        
        # Sort groups by (num_nodes, num_ranks) for consistent ordering
        sorted_groups = sorted(grouped_results.items(), key=lambda x: (x[0][0], x[0][1]))
        
        for (num_nodes, num_ranks), group_data in sorted_groups:
            # Sort payload sizes for proper ordering
            payload_sizes = sorted(group_data.keys())
            
            for payload_size in payload_sizes:
                
                for impl_name in impl_names:
                    if impl_name in group_data[payload_size]:
                        # Convert from MB/s to GB/s for bandwidth field
                        bandwidth_value = group_data[payload_size][impl_name][field] / 1000
                        
                        row = {
                            'operation_type': operation_type,
                            'implementation': impl_labels[impl_name],
                            'num_nodes': num_nodes,
                            'num_ranks': num_ranks,
                            'payload_size': payload_size,
                            'bandwidth_gbps': bandwidth_value,
                        }
                        csv_data.append(row)
                    else:
                        # Add row with zeros for missing implementation
                        row = {
                            'operation_type': operation_type,
                            'implementation': impl_labels[impl_name],
                            'num_nodes': num_nodes,
                            'num_ranks': num_ranks,
                            'payload_size': payload_size,
                            'bandwidth_gbps': 0,
                        }
                        csv_data.append(row)
    
    # Write to CSV
    if csv_data:
        df = pd.DataFrame(csv_data)
        df.to_csv(filename, index=False)
        print(f"Multi-directory bandwidth comparison exported to {filename}")
        print(f"Exported {len(csv_data)} rows of data")
    else:
        print("No data to export")

def main(data_dir, operation_type="reduce_scatter", export_csv=False, csv_filename=None):
    """Main function to run the analysis"""
    print("Loading profile data...")
    data = load_profile_data(data_dir, operation_type)
    
    if not data:
        print("No profile data found. Please run the tests first using:")
        print("  bash launch_profile.sh")
        return
    
    print(f"Loaded data for {len(data)} payload sizes")
    
    print("Analyzing data...")
    results = analyze_data(data)
    
    print("Creating plots...")
    fig = plot_results(results, operation_type)
    print("\nAnalysis complete!")


if __name__ == "__main__":
    # Example usage for all_gather data
    
    # For all_gather analysis with CSV export
    # main(data_dir, operation_type="all_gather", export_csv=True)
    
    # For reduce_scatter analysis (original functionality) with CSV export
    # main(data_dir, operation_type="reduce_scatter", export_csv=True)
    
    # For multi-directory bandwidth comparison with CSV export
    # data_dirs = [
    #     # "rs-profile",
    #     "ag-profile",
    # ]
    
    # Plot and export multi-directory bandwidth comparison
    plot_multi_bandwidth(["ag-profile"], "avg_bandwidth", operation_type="all_gather")
    # export_multi_bandwidth_to_csv(["ag-profile"], "avg_bandwidth", operation_type="all_gather")
    
    # plot_multi_bandwidth(["rs-profile"], "avg_bandwidth", operation_type="reduce_scatter")
    # export_multi_bandwidth_to_csv(["rs-profile"], "avg_bandwidth", operation_type="reduce_scatter")

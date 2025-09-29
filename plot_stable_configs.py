import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.patches import Circle
import re

# Define custom colors
colors = ['#0072b2',  # blue
         '#009e74',  # bluish green
         '#cc79a7']  # reddish purple

# Set the absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory where the script is located
data_dir = 'stable_configurations'
output_dir = os.path.join(script_dir, "visualizations")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

print(f"Script directory: {script_dir}")
print(f"Data directory: {data_dir}")
print(f"Output directory: {output_dir}")

# Extract epsilon values from the histogram files
histogram_files = glob.glob(os.path.join(data_dir, "h_histogram_epsilon_*"))
epsilon_values = []

for file in histogram_files:
    # Flexible regex pattern that extracts the numeric part in the filename
    match = re.search(r"epsilon_(\d+\.\d{2})", file)
    if match:
        epsilon_str = match.group(1)
        try:
            epsilon_values.append(float(epsilon_str))
        except ValueError:
            print(f"Warning: Could not convert '{epsilon_str}' to float in file {os.path.basename(file)}")
epsilon_values = sorted(list(set(epsilon_values)))  # Remove duplicates and sort

print(f"Found {len(epsilon_values)} epsilon values: {epsilon_values}")

# If no epsilon values were found, try to extract them from config files
if not epsilon_values:
    config_files = glob.glob(os.path.join(data_dir, "config_h_*_epsilon_{epsilon:.2f}*"))
    for file in config_files:
        match = re.search(r"epsilon_([0-9.]+)(?:\.|$)", file)
        if match:
            epsilon_str = match.group(1)
            try:
                epsilon_values.append(float(epsilon_str))
            except ValueError:
                print(f"Warning: Could not convert '{epsilon_str}' to float in file {os.path.basename(file)}")

if not epsilon_values:
    print("Error: Could not find any epsilon values in the files!")
    exit(1)

# For each epsilon value
for i, epsilon in enumerate(epsilon_values):
    print(f"Processing epsilon = {epsilon}")
    
    # 1. Create a figure for this epsilon's histogram
    fig_hist = plt.figure(figsize=(8, 4))
    
    # Load and plot histogram
    hist_file = os.path.join(data_dir, f"h_histogram_epsilon_{epsilon:.2f}.txt")
    if not os.path.exists(hist_file):
        print(f"Warning: Histogram file not found: {hist_file}")
        
        # Try to extract h-values from config files for this epsilon
        h_values = []
        h_counts = {}
        
        # Find configuration files for this epsilon
        config_files = glob.glob(os.path.join(data_dir, f"visual_config_h_*_epsilon_{epsilon:.2f}*"))
        for file in config_files:
            match = re.search(r"h_([0-9.]+)_epsilon", file)
            if match:
                h_value = float(match.group(1))
                h_values.append(h_value)
                if h_value in h_counts:
                    h_counts[h_value] += 1
                else:
                    h_counts[h_value] = 1
        
        if h_values:
            print(f"Found {len(h_values)} h-values from config files")
            plt.bar(list(h_counts.keys()), list(h_counts.values()), width=0.05)
            plt.title(rf"$|h|$ distribution for $\epsilon$ = {epsilon} (from config files)")
            plt.xlabel(r"$|h|$")
            plt.ylabel("Count")
        else:
            print(f"No h-values found for epsilon = {epsilon}")
            plt.close(fig_hist)
            continue
    else:
        try:
            hist_data = pd.read_csv(hist_file, comment='#', sep=' ', header=None, names=['h_value', 'count'])
            print(hist_data)
            plt.bar(hist_data['h_value'], hist_data['count'], width=0.05)
            plt.title(rf"$|h|$ distribution for $\epsilon$ = {epsilon}")
            plt.xlabel(r"$|h|$")
            plt.ylabel("Count")
        except Exception as e:
            print(f"Error reading histogram file {hist_file}: {e}")
            plt.close(fig_hist)
            continue
    
    # Save the histogram
    hist_output = os.path.join(output_dir, f"histogram_epsilon_{epsilon}.png")
    plt.tight_layout()
    plt.savefig(hist_output, dpi=150)
    plt.close(fig_hist)
    print(f"Saved histogram to {hist_output}")
    
    # Find configuration files for this epsilon
    config_files = glob.glob(os.path.join(data_dir, f"visual_config_h_*_epsilon_{epsilon:.2f}*"))
    
    if not config_files:
        print(f"No configuration files found for epsilon = {epsilon}")
        continue
    
    # Extract h-values from the config files
    h_values = []
    for file in config_files:
        match = re.search(r"h_([0-9.]+)_epsilon", file)
        if match:
            h_str = match.group(1)
            try:
                h_value = float(h_str)
                h_values.append((h_value, file))
            except ValueError:
                print(f"Warning: Could not convert '{h_str}' to float in file {os.path.basename(file)}")
    
    # Sort by h-value
    h_values.sort()
    
    # Keep only top 5 if there are more
    if len(h_values) > 5:
        h_values = h_values[:5]
    
    # Create a figure for this epsilon's configurations
    fig_configs = plt.figure(figsize=(15, 3))
    fig_configs.suptitle(rf"Most common configurations for $\epsilon$ = {epsilon}", fontsize=16)
    
    # For each configuration file
    for j, (h_value, config_file) in enumerate(h_values):
        # Check if file exists
        if not os.path.exists(config_file):
            print(f"Warning: Configuration file not found: {config_file}")
            continue
            
        # Extract disk positions
        disk_positions = []
        phi_values = None
        exact_h = h_value  # Default to the h-value from filename
        
        with open(config_file, 'r') as f:
            for line in f:
                if line.startswith('# h_value:'):
                    try:
                        exact_h = float(line.split(':')[1].strip())
                    except:
                        pass
                elif line.startswith('# Initial phi values:'):
                    try:
                        phi_values = [float(x) for x in line.split(':')[1].strip().split()]
                    except:
                        pass
                elif not line.startswith('#'):
                    try:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            x, y = float(parts[0]), float(parts[1])
                            disk_positions.append((x, y))
                    except Exception as e:
                        print(f"Error parsing line '{line}' in {config_file}: {e}")
        
        # Plot the configuration
        ax = fig_configs.add_subplot(1, len(h_values), j+1)
        
        # Draw the circular boundary
        enclosure = Circle((0, 0), 3 + epsilon, fill=False, color='black', linestyle='--')
        ax.add_patch(enclosure)
        
        # Draw the disks
        for k, (x, y) in enumerate(disk_positions):
            disk_color = colors[k % len(colors)]  # Cycle through custom colors
            disk = Circle((x, y), 1, fill=True, color=disk_color, alpha=0.3)
            ax.add_patch(disk)
            ax.text(x, y, str(k+1), ha='center', va='center')

        # Add vectors to show phi directions if available
        if phi_values and len(phi_values) == len(disk_positions):
            for k, (x, y) in enumerate(disk_positions):
                phi = phi_values[k]
                dx, dy = np.cos(phi), np.sin(phi)
                disk_color = colors[k % len(colors)]  # Use same color as the disk
                ax.arrow(x, y, 0.5*dx, 0.5*dy, head_width=0.15, head_length=0.15, 
                        fc=disk_color, ec=disk_color, linewidth=1.5)
        
        ax.set_xlim(-4-epsilon, 4+epsilon)
        ax.set_ylim(-4-epsilon, 4+epsilon)
        ax.set_aspect('equal')
        ax.set_title(rf"$|h|$ = {np.abs(exact_h):.4f}")
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    config_output = os.path.join(output_dir, f"configurations_epsilon_{epsilon}.png")
    plt.savefig(config_output, dpi=150)
    plt.close(fig_configs)
    print(f"Saved configurations to {config_output}")

print("Visualization complete. Check the output PNG files in:", output_dir)
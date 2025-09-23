import numpy as np
import numpy.random
import pandas as pd
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob

#################### Simulation parameters #########################################
N = 3
r = 1/(2**(1/6))
active_f = [1]
T = 0
epsilon_values = [0.01,0.014,0.021,0.03,0.043,0.062,0.089,0.13,0.18,0.26,0.38,0.55,0.78,1.1,1.6,2.3,3.4,4.8,7,10,14.8] # evenly log-spaced values between 0.01 and 10
taup_vals = [0.2, 0.5, 1, 2, 3, 5] # persistence times
thresh_params = [0.5, 1, 1.25] #[0.25, 0.5, 0.75, 1, 1.25]  # Threshold parameters to test

# Processing parameters
chunk_size = int(1e8)      # Number of elements per chunk for trajectory processing
overlap_size = int(2e6)    # Overlap between chunks to catch boundary transitions

#################### File paths ####################################################
data_path = 'Z:/Jared/Data/weeks_model_desktop/ABP caging time h trajectories, vary epsilon and taup'
output_file = 'chunksize=1e8_Athermal_ABP_cage_break_times_analysis.pkl'  # Output file for the dictionary

#################### Functions for processing trajectory data ######################
## extract active_f from filename
def extract_parameters_from_filename(filename):
    """Extract epsilon, f, and taup parameters from a trajectory filename."""
    basename = os.path.basename(filename)
    params = {}
    
    # Extract values using string parsing
    parts = basename.split('_')
    for part in parts:
        if part.startswith('epsilon='):
            params['epsilon'] = float(part.split('=')[1])
        elif part.startswith('f='):
            # Extract the raw string value first
            f_str = part.split('=')[1]
            # Convert to float - this handles both integer and decimal representations
            params['active_f'] = float(f_str)
        elif part.startswith('taup='):
            # Remove .bin extension if present
            taup_str = part.split('=')[1]
            if taup_str.endswith('.bin'):
                taup_str = taup_str[:-4]
            params['taup'] = float(taup_str)
    
    return params

## Main function that calculates time intervals between cage breaking events
def calc_jump_times(h, thresh_param=1):
    """Calculate the time intervals and indices of verified zero crossings in h, over the trajectory provided to the function.
    If this is used on chunked datasets (like we do here), this may miss zero crossings that are in progress but have not yet completed
    by the end of a chunk. This is why we allow for a substantial overlap_size of far larger than the typical maximum transition time interval,
    to catch any transitions that may have been in progress between chunks."""
    h = np.array(h)
    # Find all zero crossings
    zero_crossings = np.where(h[:-1] * h[1:] <= 0)[0]
   
    # Return empty arrays if there are no zero crossings
    if len(zero_crossings) == 0:
        return np.array([]), np.array([])
   
    validated_crossings = []
    i = 0
   
    # First determine the earliest stable state (h beyond threshold positive or negative) at start of data
    initial_stable_found = False
    for j in range(len(h)):
        if h[j] > thresh_param:
            current_state = 1
            initial_stable_found = True
            break
        elif h[j] < -thresh_param:
            current_state = -1
            initial_stable_found = True
            break
           
    if not initial_stable_found:
        return np.array([]), np.array([])
   
    while i < len(zero_crossings):
        crossing = zero_crossings[i]
       
        # Only consider crossing if we're currently in a stable state
        if current_state == 0:
            i += 1
            continue
           
        # Look forward to verify transition (starting from after the crossing)
        reached_opposite_state = False
        next_state = 0
       
        for j in range(crossing+1, len(h)):
            if current_state > 0:  # Coming from positive
                if h[j] < -thresh_param:
                    reached_opposite_state = True
                    next_state = -1
                    break
                # Reset if we go back to original state
                if h[j] > thresh_param:
                    next_state = 1
                    break
            else:  # Coming from negative
                if h[j] > thresh_param:
                    reached_opposite_state = True
                    next_state = 1
                    break
                # Reset if we go back to original state
                if h[j] < -thresh_param:
                    next_state = -1
                    break
       
        if reached_opposite_state:
            validated_crossings.append(crossing)
            current_state = next_state
            # Skip forward past this transition
            while i < len(zero_crossings) and zero_crossings[i] <= j:
                i += 1
        else:
            current_state = next_state
            i += 1
   
    validated_crossings = np.array(validated_crossings)
   
    # Always return the crossing times, but time intervals only make sense with 2+ crossings
    if len(validated_crossings) > 1:
        time_intervals = np.diff(validated_crossings)
    else:
        time_intervals = np.array([])  # No intervals possible with single crossing
       
    return time_intervals, validated_crossings # Note: validated_crossings are the indices of the verified transitions

# 
def process_large_signal_overlapping(h, chunk_size=int(1e7), overlap_size=int(2e6), threshold=1.0):
    """
    Process large signal by splitting it into overlapping chunks and feeding those into the original algorithm of calc_jump_times()
    
    Parameters:
    -----------
    h : array_like
        Complete signal to process (or handle to read from file)
    chunk_size : int
        Size of each processing chunk
    overlap_size : int
        Size of overlap between adjacent chunks (should be larger than typical transition span)
    threshold : float
        Threshold parameter for original algorithm
        
    Returns:
    --------
    all_intervals : numpy array
        Time intervals between validated transitions
    all_crossings : numpy array  
        Global indices of all validated crossings (duplicates removed)
    """
    
    h = np.array(h)
    total_length = len(h)
    
    if total_length <= chunk_size:
        # Signal fits in single chunk - just use original algorithm
        return calc_jump_times(h, thresh_param=threshold)
    
    all_crossings_with_duplicates = []
    chunk_info = []  # For debugging
    
    # Process overlapping chunks
    start_idx = 0
    chunk_number = 0
    
    while start_idx < total_length:
        # Define chunk boundaries
        end_idx = min(start_idx + chunk_size, total_length)
        chunk = h[start_idx:end_idx]
        
        # print(f"Processing chunk {chunk_number}: indices {start_idx} to {end_idx-1} (size: {len(chunk)})")
        
        # Apply original algorithm to this chunk
        intervals_chunk, crossings_local = calc_jump_times(chunk, thresh_param=threshold)
        
        # Convert local indices to global indices
        crossings_global = crossings_local + start_idx
        
        # Store results
        all_crossings_with_duplicates.extend(crossings_global)
        chunk_info.append({
            'chunk_num': chunk_number,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'n_crossings': len(crossings_local),
            'crossings_global': crossings_global.tolist() if len(crossings_global) > 0 else []
        })
        
        # print(f"  Found {len(crossings_local)} crossings in chunk {chunk_number}")
        
        # Move to next chunk with overlap
        if end_idx >= total_length:
            break
            
        # Calculate next start position (subtract overlap)
        start_idx = end_idx - overlap_size
        
        # Ensure we don't go backwards
        if start_idx <= chunk_info[-1]['start_idx'] if chunk_info else 0:
            start_idx = end_idx
            
        chunk_number += 1
    
    # Remove duplicates - crossings within overlap_size of each other
    all_crossings_with_duplicates = np.array(all_crossings_with_duplicates)
    
    if len(all_crossings_with_duplicates) == 0:
        return np.array([]), np.array([])
    
    # Sort crossings
    all_crossings_with_duplicates = np.sort(all_crossings_with_duplicates)
    
    print(f"\nDuplicate removal:")
    print(f"  Total crossings before deduplication: {len(all_crossings_with_duplicates)}")
    
    # Remove duplicates: if two crossings have the same global index, only keep the first one
    unique_crossings = np.unique(all_crossings_with_duplicates)
    print(f"  Unique crossings after deduplication: {len(unique_crossings)}")
    print(f"  Removed {len(all_crossings_with_duplicates) - len(unique_crossings)} duplicates")
    
    # Calculate intervals
    if len(unique_crossings) > 1:
        intervals = np.diff(unique_crossings)
    else:
        intervals = np.array([])
    
    # # For debugging: print chunk-by-chunk breakdown
    # print(f"\nChunk breakdown:")
    # for info in chunk_info:
    #     print(f"  Chunk {info['chunk_num']}: {info['n_crossings']} crossings at global indices {info['crossings_global']}")
    
    return intervals, unique_crossings

def process_trajectory_file(filename, thresh_params, epsilon=None):
    """
    Process a trajectory file for multiple threshold parameters using overlapping chunks.
    
    Args:
        filename: Path to binary file with float32 values
        thresh_params: List of threshold parameters
        epsilon: Epsilon value for calculating h_max
    
    Returns:
        all_results: Dict mapping threshold param to list of intervals
        entropic_barrier: Calculated entropic barrier
    """
    if not os.path.exists(filename):
        print(f"Warning: File {filename} not found.")
        return {thresh: [] for thresh in thresh_params}, np.nan
    
    # Calculate theoretical h_max based on epsilon
    h_max_theory = 2 + epsilon + np.sqrt(3 + 4*epsilon + epsilon**2)
    buffer = 0.07  # Add 7% buffer for soft interactions
    h_max = h_max_theory * (1 + buffer)
    h_min = 0.0  # Starting from 0 since we're taking abs(h)
    
    # Define histogram parameters
    n_bins = 100
    bin_width = (h_max - h_min) / n_bins
    
    file_size = os.path.getsize(filename)
    total_elements = file_size // 4  # Each float32 is 4 bytes
    
    print(f"Processing file {filename} ({file_size/1e9:.2f} GB, {total_elements} points)")
    print(f"  Histogram: {n_bins} bins, bin width {bin_width:.6f}, range: [{h_min}, {h_max:.6f})")
    
    # Initialize histogram for entropic barrier calculations
    bin_edges = np.linspace(h_min, h_max, n_bins + 1)
    histogram = np.zeros(n_bins)
    
    # Read the entire file for histogram calculation (in chunks to manage memory)
    print("  Calculating histogram...")
    with open(filename, 'rb') as f:
        offset = 0
        while offset < total_elements:
            # Calculate how many elements to read in this chunk
            elements_to_read = min(chunk_size, total_elements - offset)
            
            # Read chunk
            f.seek(offset * 4)  # Each float32 is 4 bytes
            chunk = np.fromfile(f, dtype=np.float32, count=elements_to_read)
            
            if len(chunk) == 0:
                break
                
            # Update histogram with |h| values
            h_abs = np.abs(chunk)
            hist, _ = np.histogram(h_abs, bins=bin_edges)
            histogram += hist
            
            offset += elements_to_read
    
    # Now read the entire file into memory for transition analysis
    print("  Loading full trajectory for transition analysis...")
    h_full = np.fromfile(filename, dtype=np.float32)
    
    # Process for all thresholds using the overlapping chunk method
    all_intervals = {}
    
    print(f"  Processing transitions with chunk_size={chunk_size}, overlap_size={overlap_size}")
    
    for thresh in thresh_params:
        print(f"  Processing threshold {thresh}...")
        intervals, crossings = process_large_signal_overlapping(
            h_full, 
            chunk_size=chunk_size, 
            overlap_size=overlap_size, 
            threshold=thresh
        )
        all_intervals[thresh] = intervals
        print(f"    Found {len(intervals)} intervals from {len(crossings)} crossings")
    
    # Calculate entropic barrier
    n_max = np.max(histogram)
    n_0 = histogram[0]  # Value at bin containing h=0
    max_idx = np.argmax(histogram)
    
    entropic_barrier = -np.log(n_0/n_max) if n_0 > 0 else np.inf
    print(f"  Entropic barrier: {entropic_barrier:.4f} (n_0={n_0} at bin 0 [{bin_edges[0]:.6f}, {bin_edges[1]:.6f}), "
          f"n_max={n_max} at bin {max_idx} [{bin_edges[max_idx]:.6f}, {bin_edges[max_idx+1]:.6f}))")
    
    return all_intervals, entropic_barrier

def format_filename_parameter(value):
    """Format a numeric parameter to match how it appears in filenames."""
    # For whole numbers, remove the decimal point and trailing zeros
    if value == int(value):
        return str(int(value))
    # For decimal numbers, use as is
    return str(value)

# Check for REPEAT files in data
def find_repeat_file(data_path, eps, active_f, taup):
    """Check if a REPEAT file exists for the given parameters."""
    # Format parameters consistently with how they appear in filenames
    eps_str = format_filename_parameter(eps)
    f_str = format_filename_parameter(active_f)
    taup_str = format_filename_parameter(taup)
    
    repeat_pattern = os.path.join(data_path, f'REPEAT_h_trajectories_epsilon={eps_str}_f={f_str}_taup={taup_str}.bin')
    matches = glob.glob(repeat_pattern)
    
    if matches:
        return matches[0]
    return None

# Process data for all parameters
def process_parameters(taup, active_f, eps_values, all_data_current, data_path, thresh_params):
    """
    Process all epsilon values for a given taup and active_f value.
    
    Args:
        taup: The persistence time to process
        active_f: The active force parameter
        eps_values: List of epsilon values to process
        all_data_current: The current data for this taup and active_f
        data_path: Path to data directory
        thresh_params: List of threshold parameters
        
    Returns:
        param_results: Dictionary of results for these parameters
    """
    param_results = all_data_current.copy() if all_data_current else {}
    
    # Ensure all threshold entries have the required fields
    for thresh_param in thresh_params:
        thresh_key = f"thresh_{thresh_param}"
        if thresh_key in param_results:
            if 'intervals' not in param_results[thresh_key]:
                param_results[thresh_key]['intervals'] = [[] for _ in range(len(param_results[thresh_key]['epsilon']))]
            
            if 'num_transitions' not in param_results[thresh_key]:
                param_results[thresh_key]['num_transitions'] = [-1] * len(param_results[thresh_key]['epsilon'])
    
    for eps in eps_values:
        # First check if a REPEAT file exists
        repeat_file = find_repeat_file(data_path, eps, active_f, taup)
        
        if repeat_file:
            full_path = repeat_file
            is_repeat = True
            print(f"Found REPEAT file: {os.path.basename(full_path)}")
        else:
            # Use original file if no REPEAT file exists
            eps_str = format_filename_parameter(eps)
            f_str = format_filename_parameter(active_f)
            taup_str = format_filename_parameter(taup)
            file_h = f'h_trajectories_epsilon={eps_str}_f={f_str}_taup={taup_str}.bin'
            full_path = os.path.join(data_path, file_h)
            is_repeat = False
        
        # Check if file exists
        if not os.path.exists(full_path):
            print(f"Skipping missing file: {full_path}")
            continue
        
        # Process the trajectory file once for all thresholds and calculate entropic barrier
        all_intervals, entropic_barrier = process_trajectory_file(full_path, thresh_params, epsilon=eps)
        
        # Skip adding entries if entropic_barrier is nan
        if np.isnan(entropic_barrier):
            print(f"Skipping results for taup={taup}, f={active_f}, epsilon={eps} due to invalid entropic barrier")
            continue
            
        # Update entropic barrier result
        if 'entropic_barrier' not in param_results:
            param_results['entropic_barrier'] = {'epsilon': [], 'value': []}
        
        # Handle entropic barrier data
        if is_repeat and eps in param_results['entropic_barrier']['epsilon']:
            idx = param_results['entropic_barrier']['epsilon'].index(eps)
            param_results['entropic_barrier']['value'][idx] = entropic_barrier
        else:
            param_results['entropic_barrier']['epsilon'].append(eps)
            param_results['entropic_barrier']['value'].append(entropic_barrier)
        
        # Store results for each threshold
        for thresh_param in thresh_params:
            thresh_key = f"thresh_{thresh_param}"
            
            # Initialize threshold entry if it doesn't exist
            if thresh_key not in param_results:
                param_results[thresh_key] = {
                    'epsilon': [], 
                    'tau': [], 
                    'tau_std': [], 
                    'num_transitions': [],
                    'intervals': []
                }
            
            # For REPEAT files, replace existing entries; otherwise skip existing entries
            if not is_repeat and eps in param_results[thresh_key]['epsilon']:
                print(f"Skipping existing entry: taup={taup}, f={active_f}, threshold={thresh_param}, epsilon={eps}")
                continue
            
            intervals = all_intervals[thresh_param]
            num_transitions = len(intervals)
            
            if num_transitions > 0:
                avg_cage_time = np.mean(intervals)
                std_cage_time = np.std(intervals)
                std_error = std_cage_time / np.sqrt(num_transitions)
                print(f"Results: taup={taup}, f={active_f}, eps={eps}, thresh={thresh_param}, avg_time={avg_cage_time:.2f} ± {std_error:.2f}, transitions={num_transitions}")
            else:
                avg_cage_time = np.nan
                std_error = np.nan
                print(f"Warning: No transitions found for epsilon={eps}, taup={taup}, f={active_f}, thresh={thresh_param}")
            
            # Replace or add data
            if is_repeat and eps in param_results[thresh_key]['epsilon']:
                idx = param_results[thresh_key]['epsilon'].index(eps)
                param_results[thresh_key]['tau'][idx] = avg_cage_time
                param_results[thresh_key]['tau_std'][idx] = std_error
                param_results[thresh_key]['num_transitions'][idx] = num_transitions
                param_results[thresh_key]['intervals'][idx] = intervals.tolist()
            else:
                param_results[thresh_key]['epsilon'].append(eps)
                param_results[thresh_key]['tau'].append(avg_cage_time)
                param_results[thresh_key]['tau_std'].append(std_error)
                param_results[thresh_key]['num_transitions'].append(num_transitions)
                param_results[thresh_key]['intervals'].append(intervals.tolist())
    
    return (taup, active_f), param_results

def load_or_create_dictionary():
    """Load existing dictionary or create a new one with metadata."""
    if os.path.exists(output_file):
        try:
            with open(output_file, 'rb') as f:
                all_data = pickle.load(f)
            print(f"Loaded existing data from {output_file}")
            return all_data
        except Exception as e:
            print(f"Error loading existing file: {e}")
            print("Creating new dictionary...")
    
    # Create new dictionary with metadata
    all_data = {
        'metadata': {
            'N': N,
            'r': r,
            'active_f': active_f,
            'T': T,
            'epsilon_values': epsilon_values,
            'taup_vals': taup_vals,
            'thresh_params': thresh_params,
            'description': 'Cage breaking times analysis for hard disk model'
        },
        'results': {}
    }
    return all_data

def save_dictionary(all_data):
    """Save the dictionary to a pickle file."""
    with open(output_file, 'wb') as f:
        pickle.dump(all_data, f)
    print(f"Data saved to {output_file}")

################## Main processing loop ##############################
def main():
    global active_f, taup_vals, epsilon_values, thresh_params
    
    # Load or create the dictionary
    all_data = load_or_create_dictionary()
    
    # Initialize results structure if it doesn't exist
    if 'results' not in all_data:
        all_data['results'] = {}
    
    # Set up multiprocessing
    max_workers = os.cpu_count() - 1
    print(f"Using {max_workers} worker processes")
    
    # Use only the specified combinations instead of discovering all files
    parameter_combinations = []
    for taup in taup_vals:
        for f in active_f:
            parameter_combinations.append((taup, f))
    
    print(f"Processing {len(parameter_combinations)} specified parameter combinations")
    
    # Ensure all existing entries have active_f=1 if not already specified
    migrated_structure = {}
    for taup in all_data['results']:
        if isinstance(all_data['results'][taup], dict) and not any(isinstance(key, tuple) for key in all_data['results']):
            # This is old structure - migrate to new structure with active_f=1
            if taup not in migrated_structure:
                migrated_structure[taup] = {}
            migrated_structure[taup][1.0] = all_data['results'][taup]
            print(f"Migrated existing data for taup={taup} to include active_f=1")
        else:
            # Already in new structure or empty
            migrated_structure[taup] = all_data['results'][taup]
    
    all_data['results'] = migrated_structure
    
    # Use a context manager to ensure clean shutdown of processes
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit each (taup, active_f) combination to the process pool
        futures = {}
        for taup, f in parameter_combinations:
            # Make sure the nested dictionary structure exists
            if taup not in all_data['results']:
                all_data['results'][taup] = {}
            if f not in all_data['results'][taup]:
                all_data['results'][taup][f] = {}
            
            # Create a future for this parameter combination
            future = executor.submit(
                process_parameters, 
                taup, 
                f,
                epsilon_values, 
                all_data['results'][taup][f].copy(), 
                data_path,
                thresh_params
            )
            futures[future] = (taup, f)
        
        # Process results as they complete
        for future in as_completed(futures):
            params = futures[future]
            taup, active_f = params
            try:
                result_key, param_results = future.result()
                all_data['results'][taup][active_f] = param_results
                print(f"Completed processing for taup={taup}, active_f={active_f}")
                
                # Save intermediate results
                save_dictionary(all_data)
                print(f"Saved data after processing taup={taup}, active_f={active_f}")
            except Exception as e:
                print(f"Error processing taup={taup}, active_f={active_f}: {e}")
    
    # Final save
    save_dictionary(all_data)
    print("Analysis complete!")

if __name__ == "__main__":
    main()
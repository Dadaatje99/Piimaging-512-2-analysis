# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:45:06 2024

@author: admin_local
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def rebin_traces(traces, step=2):
    """
    Rebins the traces by summing over the specified step sizes.

    Parameters:
    traces (np.ndarray): The 2D array of traces to be rebinned (columns rebinned).
                         Shape: (n_rows, n_columns)
    step (int): The step size for rebinning.

    Returns:
    np.ndarray: The rebinned traces.
    """
    # Ensure the number of columns is divisible by step
    rows, cols = traces.shape
    if cols % step != 0:
        raise ValueError("The number of columns in traces must be divisible by step.")
    
    # Reshape and sum across the step axis (rebinned columns)
    rebinned_traces = traces.reshape(rows, cols // step, step).sum(axis=2)
    return rebinned_traces



def reverse_sigma_clipping(data, sigma):
    """
    Perform sigma clipping to remove lower outliers (spikes).
    Keeps the main signal by iteratively excluding values below a dynamic lower bound.
    """
    iterations = 0
    while True:
        mean = np.mean(data)
        std = np.std(data)
        clipped_data = data[data > mean - sigma * std]
        if len(clipped_data) == len(data):
            break
        data = clipped_data
        iterations += 1
    return data, iterations

def refine_lower_threshold(data, sigma=2):
    """
    Refine the lower threshold for separating the ON state by treating spikes as outliers.
    """
    max_val = np.max(data)
    min_val = np.min(data)
    middle_value = (max_val + min_val) / 2
    clipped_data, _ = reverse_sigma_clipping(data, sigma)
    mean_signal = np.mean(clipped_data)
    std_signal = np.std(clipped_data)
    
    # Calculate the threshold based on the middle of max and min values


    # Define a threshold that is around the middle value but considers the spread
    threshold = max(middle_value, mean_signal - sigma * std_signal)


    return mean_signal, threshold



def analyze_traces_robust(intensity_traces, sigma=3, plot=False):
    """
    Analyzes traces to calculate a robust lower threshold and optionally plots results.
    """
    thresholds = []
    num_traces = len(intensity_traces)
    num_figures = (num_traces + 15) // 16  # Number of figures needed for 16 traces per figure

    for fig_idx in range(num_figures):
        start_idx = fig_idx * 16
        end_idx = min(start_idx + 16, num_traces)

        if plot:
            fig, axs = plt.subplots(4, 4, figsize=(16, 10))
            axs = axs.flatten()

        for i in range(start_idx, end_idx):
            data = intensity_traces[i]
            if data.size == 0:
                thresholds.append(None)
                if plot:
                    axs[i - start_idx].set_title(f'Trace {i}')
                    axs[i - start_idx].axis('off')
                continue

            # Calculate the robust lower threshold
            clipped_mean, threshold = refine_lower_threshold(data, sigma)
            thresholds.append([clipped_mean, threshold])

            # Plot if required
            if plot:
                ax = axs[i - start_idx]
                ax.plot(data, color='gray', alpha=0.8, linewidth=0.5, label="Intensity Trace")
                ax.axhline(y=threshold, color='red', linestyle='--', label='Lower Threshold')
                ax.set_title(f'Trace {i}')
                ax.set_xlabel('Frame')
                ax.set_ylabel('Intensity')

        if plot: 
            # Remove unused subplots
            for j in range(end_idx - start_idx, 16):
                fig.delaxes(axs[j])
            plt.tight_layout()
            plt.show()

    return thresholds

frame_intensity2 = rebin_traces(frame_intensity1,step=4)

mask = np.max(frame_intensity2, axis=1) >= 50

# Step 3: Apply the mask to filter the data
A_kept_indices = np.where(mask)[0]  # Indices of rows that are kept
A_discarded_indices = np.where(~mask)[0]  # Indices of rows that are discarded

frame_intensity2 = frame_intensity2[mask]

info = analyze_traces_robust(frame_intensity2)
thresholds= np.array(info).T[1]
#info = analyze_traces_robust(frame_intensity2,plot=True)
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from scipy.ndimage import uniform_filter1d
# Assuming frame_intensity2 is your original array of shape (N, 12000)
max_values = np.max(uniform_filter1d(frame_intensity2,size=5), axis=1)

# Get the sorted indices
sorted_indices = np.argsort(max_values)

# Sort the max_values and frame_intensity2 accordingly
sorted_max_values = max_values[sorted_indices]
sorted_frame_intensity2 = frame_intensity2[sorted_indices]
sorted_thresholds = thresholds[sorted_indices]

# Define the number of bins (groups)
num_bins = 10  # You can change this to the desired number of binsu

# Compute the bin edges using numpy.histogram
bin_edges = np.histogram_bin_edges(sorted_max_values, bins=num_bins)

# Split sorted_frame_intensity2 into groups based on the bin edges
groups = []
for i in range(num_bins):
    # Get the indices of the elements within each bin
    bin_mask = (sorted_max_values >= bin_edges[i]) & (sorted_max_values < bin_edges[i+1])
    
    # Collect the corresponding rows from sorted_frame_intensity2
    groups.append(sorted_frame_intensity2[bin_mask])

#%%
# Calculate the maximum values for each row
max_values = np.max(sorted_frame_intensity2, axis=1)

# Filter only the values satisfying the mask
valid_max_values = max_values[max_values >= 100]
print(valid_max_values)
#%%
sorted_values = np.sort(valid_max_values)
# Define the number of bins and calculate their range
num_bins = 25
bin_edges = np.interp(
    np.linspace(0, len(sorted_values), num_bins + 1),
    np.arange(len(sorted_values)),
    sorted_values,
)

# Digitize the data into bins based on the calculated edges
bin_indices = np.digitize(valid_max_values, bin_edges, right=True)

# Extract the corresponding rows from sorted_frame_intensity2 for each bin
bin_data = [
    sorted_frame_intensity2[bin_indices == i] for i in range(1, num_bins + 1)
]

bin_mid_values = (bin_edges[:-1] + bin_edges[1:]) / 2
# Plot the histogram with equal-frequency bins
plt.hist(valid_max_values, bins=bin_edges, color='blue', alpha=0.7, edgecolor='black')
plt.title('Histogram of Maximum Values (>= 100)')
plt.xlabel('Maximum Value')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Collect thresholds for each bin
bin_thresholds = []
idx = 0
for i, bin_array in enumerate(bin_data):
    x = bin_array.shape[0]
    print(f"Bin {i + 1} Range: {bin_edges[i]} to {bin_edges[i + 1]}, Count: {x}")
    bin_thresholds.append(sorted_thresholds[idx:idx + x])  # Update slicing
    idx += x  # Increment idx

#%%
plt.plot(sorted_frame_intensity2[-3])


#%%
def ON_OFF(thresholds, frame_intensity2, tint, max_time=0.9, max_time2=0.2):
    
    def power_law(x, C, alpha):
        """
        Defines the power-law function.
        """
        return C * x ** (-alpha)
    
    def power_exp_law(x, C, alpha,tau):
        """
        Defines the power-law function.
        """
        return C * x ** (-alpha) * np.exp(-x/tau)
    
    def power_exp_law_log_space(x, C, alpha, tau):
        """
        Defines the transformed power-law with exponential decay in log space.
        This is the linearized form of the original equation.
        """
        # Calculate log of x and y
        x_log = np.log(x)
        
        # The fitted model in log space
        return np.log(C) - alpha * x_log - np.exp(x_log) / tau
    
    def double_power_exp_law(x, C, alpha,tau, C1, alpha1):
        """
        Defines the power-law function.
        """
        return C * x ** (-alpha) * np.exp(-x/tau) * C1 * x ** (-alpha1)
    
    # Lists to store all consecutive ON and OFF durations
    all_consecutive_on = []
    all_consecutive_off = []
    
    # Calculate ON and OFF durations based on thresholds
    for trace_index, threshold in enumerate(thresholds):
        # Skip if threshold is None
        if threshold is None:
            continue
    
        # Extract the ON threshold
        high_threshold = threshold
    
        # Get the intensity trace
        data = frame_intensity2[trace_index]
    
        # Generate ON and OFF masks
        mask_on = data > high_threshold
        mask_off = ~mask_on  # OFF state is the complement of ON
    
        # Initialize counters
        consecutive_on = []
        consecutive_off = []
    
        current_on_count = 0
        current_off_count = 0
    
        # Loop through the trace to calculate consecutive ON and OFF durations
        for i in range(len(data)):
            if mask_on[i]:  # ON state
                if current_off_count > 0:
                    consecutive_off.append(current_off_count)
                    current_off_count = 0
                current_on_count += 1
            else:  # OFF state
                if current_on_count > 0:
                    consecutive_on.append(current_on_count)
                    current_on_count = 0
                current_off_count += 1
    
        # Handle remaining counts at the end of the trace
        if current_on_count > 0:
            consecutive_on.append(current_on_count)
        if current_off_count > 0:
            consecutive_off.append(current_off_count)
    
        # Append to the global lists
        all_consecutive_on.append(consecutive_on)
        all_consecutive_off.append(consecutive_off)
    
    # Flatten lists of consecutive durations
    all_consecutive_on_flat = np.concatenate(all_consecutive_on)
    all_consecutive_off_flat = np.concatenate(all_consecutive_off)
    
    # Unique durations and their frequencies
    # Calculate unique counts and frequencies
    unique_on_counts, on_frequencies = np.unique(all_consecutive_on_flat, return_counts=True)
    unique_off_counts, off_frequencies = np.unique(all_consecutive_off_flat, return_counts=True)
    
    # Step 2: Calculate average time between nearest neighbor events
    def calculate_weights(times, frequencies):
        weights = []
        for i, t in enumerate(times):
            if i == 0:
                # First bin, use distance to the next event
                nearest_neighbor_distance = abs(times[i + 1] - t)
            elif i == len(times) - 1:
                # Last bin, use distance to the previous event
                nearest_neighbor_distance = abs(t - times[i - 1])
            else:
                # Middle bins, use average distance to nearest neighbors
                nearest_neighbor_distance = (abs(t - times[i - 1]) + abs(times[i + 1] - t)) / 2
            weights.append(nearest_neighbor_distance)
        return np.array(weights)
    
    # Calculate weights for on and off times
    on_weights = calculate_weights(unique_on_counts, on_frequencies)
    off_weights = calculate_weights(unique_off_counts, off_frequencies)
    
    # Step 3: Weight the histogram counts
    weighted_on_frequencies = on_frequencies / on_weights
    weighted_off_frequencies = off_frequencies / off_weights
    
    # Step 4: Normalize to form a probability density
    on_probability_densities = weighted_on_frequencies / np.sum(weighted_on_frequencies)
    off_probability_densities = weighted_off_frequencies / np.sum(weighted_off_frequencies)
    
    on_time_values = unique_on_counts * tint * 10**-3 * 4
    off_time_values = unique_off_counts * tint * 10**-3 * 4
    
    # Filter by maximum time values
    on_mask = on_time_values <= 20
    off_mask = off_time_values <= 20 #max_time
    
    on_time_values_filtered = on_time_values[on_mask]
    off_time_values_filtered = off_time_values[off_mask]
    
    on_probability_densities_filtered = on_probability_densities[on_mask]
    off_probability_densities_filtered = off_probability_densities[off_mask]
    print(len(on_time_values_filtered))
    #wortel(N)/on_weights
    # Fit power-law functions
    #popt_on, pcov_on  = curve_fit(power_exp_law, on_time_values_filtered[100:], on_probability_densities_filtered[100:],
    #                       sigma=np.sqrt(on_probability_densities_filtered[100:]),bounds=([-np.inf,-np.inf,0.1],[np.inf,np.inf,np.inf]), maxfev=12000)
    popt_on, pcov_on  = curve_fit(power_exp_law, on_time_values_filtered[1:], on_probability_densities_filtered[1:],
                           sigma=np.sqrt(on_probability_densities_filtered[1:]),bounds=([-np.inf,-np.inf,0.1],[np.inf,np.inf,np.inf]), maxfev=12000, absolute_sigma=True)
    
    #popt_on, pcov_on  = curve_fit(power_exp_law, on_time_values_filtered, on_probability_densities_filtered,
    #                       sigma=np.sqrt(on_probability_densities_filtered)/on_weights,bounds=([-np.inf,-np.inf,0.1],[np.inf,np.inf,np.inf]), maxfev=12000, absolute_sigma=True)
    popt_on, pcov_on = curve_fit(power_exp_law_log_space, on_time_values_filtered[:90], np.log(on_probability_densities_filtered[:90]),
                           sigma=(np.log(on_probability_densities_filtered[:90])),bounds=([0.005,1,0.1],[np.inf,1.01,np.inf]), maxfev=12000, absolute_sigma=True)

    print(popt_on)
    popt_off, pcov_off  = curve_fit(power_law, off_time_values_filtered[2:], off_probability_densities_filtered[2:],
                            sigma=np.sqrt(off_probability_densities_filtered[2:]), maxfev=12000)
    
    alpha_on_error = np.sqrt(np.diag(pcov_on))[1]  # Error in alpha (on)
    alpha_off_error = np.sqrt(np.diag(pcov_off))[1]  # Error in alpha (off)
    
    # Print the results
    print(f"On fit: alpha = {popt_on[1]:.4f} ± {alpha_on_error:.4f}")
    print(f"Off fit: alpha = {popt_off[1]:.4f} ± {alpha_off_error:.4f}")
    
    
    fig, ax = plt.subplots(figsize=(2.2, 3))
    ax.plot(off_time_values, off_probability_densities, '.', color='red', alpha=0.6)
    ax.plot(off_time_values_filtered, power_law(off_time_values_filtered, *popt_off),
            label=f'$\\alpha={popt_off[1]:.2f}$', color='black', alpha=1)
    # Plot OFF durations and their power-law fit
    #ax.plot(on_time_values_filtered, power_exp_law(on_time_values_filtered, *popt_on),
    #        label=f'$\\alpha={popt_on[1]:.2f}$', color='black', alpha=0.4)
    ax.plot(on_time_values, on_probability_densities, '.', color='green', alpha=0.6)
    ax.plot(on_time_values_filtered, power_exp_law(on_time_values_filtered, *popt_on), color='black', alpha=0.8)
    ax.plot(on_time_values_filtered, power_law(on_time_values_filtered, *popt_on[:2]),
            label=f'$\\alpha={popt_on[1]:.2f}$', color='black', alpha=1,linestyle='--')
    

    
    # Set x and y limits
    ax.set_xlim([0.01, 60])
    ax.set_ylim([0.0000000001, 99.99999])
    
    # Add a text annotation for the power-law equation
    ax.text(0.1, 3, '$C x^{-\\alpha}$', fontsize=16, color='black')
    
    # Set logarithmic scales
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Labels and legend
    ax.set_xlabel('OFF time (s)', fontproperties=font_prop)
    ax.set_ylabel('Probability density (1/s)', fontproperties=font_prop)
    ax.legend()
    
    # Customize tick parameters
    ax.tick_params(axis='x', which='major', labelsize=10)
    ax.tick_params(axis='x', which='minor', labelsize=8)
    
    # Customize tick parameters
    ax.tick_params(axis='y', which='major', labelsize=10)
    ax.tick_params(axis='y', which='minor', labelsize=8)
    # Adjust the layout
    plt.tight_layout()
    
    # Display the plot
    plt.show()
    return all_consecutive_off , all_consecutive_on , [popt_on, pcov_on],[popt_off, pcov_off]

all_consecutive_off , all_consecutive_on,fit_on,fit_off = ON_OFF(sorted_thresholds,sorted_frame_intensity2,tint)
#%%
fits_on=[]
fits_off=[]
for i in range(len(bin_data)):

    all_consecutive_off , all_consecutive_on,fit_on,fit_off= ON_OFF(bin_thresholds[i],bin_data[i],tint)
    fits_on.append(fit_on)
    fits_off.append(fit_off)
#%%
def power_fit2(k_exc,n,k_x):
    return n*(k_exc)**2/((k_exc)+k_x)

def power_fit(kP,n,kBX,kX):
    return n*kBX*kP**2/(kBX*kP + kBX*kX + kP**2)


y = [1/fits_on[i][0][2] for i in range(len(fits_on))]
plt.plot(bin_mid_values,y,'.')    
plt.yscale('log')
plt.xlim([0,7000])
#plt.ylim([0,3])

print(bin_mid_values,y)

popt, pcov = curve_fit(power_fit, bin_mid_values, y,sigma=np.sqrt(y), bounds=[(0,0,0),(np.inf,np.inf,np.inf)])

print(popt)
plt.plot(bin_mid_values,power_fit(bin_mid_values, *popt))
#%%
def power_fit(k_exc,k_x):
    return (k_exc)**2/((k_exc)+k_x)




y = [1/fits_on[i][0][2] for i in range(len(fits_on))]
plt.plot(bin_mid_values,y,'.')    
plt.yscale('log')
plt.xlim([0,7000])
#plt.ylim([0,3])

popt, pcov = curve_fit(power_fit, bin_mid_values, y,sigma=np.sqrt(y), bounds=[(0),(np.inf)])

print(1/(popt/1e9))
plt.plot(bin_mid_values,power_fit(bin_mid_values, *popt))



#%%
from sympy import symbols, Eq, solve

def compute_transition_timescale(kP_val, kX_val, kBX_val):
    # Define variables
    P0, P1, P2 = symbols('P0 P1 P2')
    kP, kX, kBX = symbols('kP kX kBX')

    # Define equations
    eq1 = Eq(P0 + P1 + P2, 1)
    eq2 = Eq(P1 * kX - P0 * kP, 0)
    eq3 = Eq(P0 * kP - P1 * kP + P2 * kBX - P1 * kX, 0)
    eq4 = Eq(P1 * kP - P2 * kBX, 0)

    # Solve equations
    solution = solve((eq1, eq2, eq3, eq4), (P0, P1, P2))
    
    # Substitute numerical values
    numerical_solution = {key: value.subs({kP: kP_val, kX: kX_val, kBX: kBX_val}) for key, value in solution.items()}
    
    # Compute transition frequency
    P1_val = numerical_solution[P1]
    transition_frequency = P1_val * kP_val
    
    # Compute transition timescale
    transition_timescale = 1 / transition_frequency if transition_frequency != 0 else float('inf')
    
    return transition_timescale, numerical_solution,solution

# Example usage:
kP_value = 0.002
kX_value = 1/40
kBX_value = 1/2

timescale, result , solution= compute_transition_timescale(kP_value, kX_value, kBX_value)
print("Steady-state probabilities:", result)
print("Timescale for transitions from P1 to P2:", timescale, "seconds")
print(solution)
#%%
all_consecutive_off , all_consecutive_on = ON_OFF(sorted_thresholds,sorted_frame_intensity2,tint)
#%%

ratios=[]
for i in range(len(all_consecutive_off)):
    ratio = np.sum(all_consecutive_off[i])/(np.sum(all_consecutive_on[i])+np.sum(all_consecutive_off[i]))
    ratios.append(ratio)


# Create a figure and axis
fig, ax = plt.subplots(figsize=(3, 3))

# Create a histogram
ax.hist(ratios, bins=np.arange(0, 1, 0.05), edgecolor='black', color='lightblue', alpha=0.7)

# Set the x-axis and y-axis labels with LaTeX formatting
font_prop.set_size(12)  # Customize font size as needed

ax.set_xlabel(r'$\text{Ratio} = \; \frac{\text{t}_{\text{off}}}{\text{t}_{\text{tot}}}$', fontproperties=font_prop)
ax.set_ylabel('Occurence', fontproperties=font_prop)
ax.set_xlim([0,1])
ax.set_ylim([0,190])
# Remove y-ticks
# Set major and minor locators for x-axis
ax.yaxis.set_major_locator(MultipleLocator(10))  # Major ticks every 0.2 units
ax.yaxis.set_minor_locator(MultipleLocator(5))  # Minor ticks every 0.05 units

# Customize tick parameters
ax.tick_params(axis='y', which='major', labelsize=10)
ax.tick_params(axis='y', which='minor', labelsize=8)

# Set major and minor locators for x-axis
ax.xaxis.set_major_locator(MultipleLocator(0.2))  # Major ticks every 0.2 units
ax.xaxis.set_minor_locator(MultipleLocator(0.05))  # Minor ticks every 0.05 units

# Customize tick parameters
ax.tick_params(axis='x', which='major', labelsize=10)
ax.tick_params(axis='x', which='minor',  labelsize=8)

# Display the plot
plt.show()



#%%
def calculate_on_off_durations(intensity_trace, threshold):
    # Initialize counters
    on_points = 0
    off_points = 0
    
    # Iterate through the intensity trace
    for intensity in intensity_trace:
        if intensity >= threshold:
            on_points += 1
        else:
            off_points += 1
    
    return on_points, off_points

# Provided data
background = bkg  # Background image
segmentation_map = segmentation_map  # Segmentation map
print(np.max(segmentation_map))
framesintensity = sorted_frame_intensity2  # Intensity traces for particles
thresholds = sorted_thresholds  # Thresholds for particles

# Define contour levels and process regions
levels = np.linspace(np.min(background), np.max(background), 11)  # 155 levels


# Initialize data structures
labeled_background = np.digitize(background, levels) - 1  # Adjusted to start from 0

# Initialize data structures
region_ratios = {region: [] for region in range(len(levels) - 1)}

# Map particles to regions and calculate ratios
for particle_id,seg_id in enumerate(A_kept_indices):  # Sequential index: 0, 1, 2, ...
    # Find regions overlapping with this particle
    particle_mask = segmentation_map == seg_id + 1  # Adjust for non-zero particle IDs
    overlapping_regions = np.unique(labeled_background[particle_mask])

    # Retrieve intensity trace and threshold for this particle
    intensity_trace = framesintensity[particle_id]
    threshold = thresholds[particle_id]

    # Assign particle to each overlapping region
    for region in overlapping_regions:
        if 0 <= region < len(levels) - 1:  # Ensure valid region index
            on, off = calculate_on_off_durations(intensity_trace, threshold)
            ratio = off / (on + off) if (on + off) > 0 else np.nan
            region_ratios[region].append(ratio)

# Debugging: Verify that all regions are processed
print(f"Total regions: {len(levels) - 1}, Processed regions: {len(region_ratios)}")

# Compute mean ratios and uncertainties
mean_ratios = []
sigma_values = []

for region in range(len(levels) - 1):
    ratios = region_ratios[region]
    if ratios:
        mean = np.nanmean(ratios)
        sigma = (
            np.nanstd(ratios, ddof=1) / np.sqrt(len(ratios)) * 
            t.ppf(1 - 0.05 / 2, df=len(ratios) - 1)
        )
    else:
        mean = np.nan  # No data in this region
        sigma = np.nan
    mean_ratios.append(mean)
    sigma_values.append(sigma)

# Plot the results
regions = 0.5 * (levels[:-1] + levels[1:])
plt.errorbar(regions, mean_ratios, yerr=sigma_values, fmt='o', label='Mean Ratios')
plt.xlabel('background')
plt.ylabel('Mean ON/OFF Ratio')
plt.ylim([0, 0.5])
plt.title('Mean ON/OFF Ratios per Region')
plt.legend()
plt.show()

# Save results to file
output_file = r'C:/David/ratios/ratios_and_levels1.txt'
with open(output_file, 'w') as f:
    f.write("Region\tMean_Ratio\tSigma\n")
    for region, mean, sigma in zip(regions, mean_ratios, sigma_values):
        f.write(f"{region}\t{mean}\t{sigma}\n")

# Debugging: Verify that all regions are processed
print(f"Total regions: {len(levels) - 1}, Processed regions: {len(mean_ratios)}")

#%%
import os
import glob


# Base directory containing levels0, levels1, ..., levels4
base_path = r'C:/David/ratios/'

# Define the directories to search (levels0 to levels4)
levels = [os.path.join(base_path, f'ratios_and_levels{i}.txt') for i in range(5)]

# Container to store the content of each file
file_contents = []

# Iterate through directories
for level_path in levels:
    if not os.path.exists(level_path):  # Skip if directory does not exist
        print(f"Directory not found: {level_path}")
        continue

    # Find all files in the current directory
    files = glob.glob(os.path.join(level_path))  # Adjust extension if needed

    for file in files:
        # Extract the region name from the filename (or adjust as necessary)
        region_name = os.path.splitext(os.path.basename(file))[0]

        # Read file content
        try:
            with open(file, 'r') as f:
                content = f.read()  # Read the whole file as a string
                file_contents.append(content)

        except Exception as e:
            print(f"Error reading file {file}: {e}")

import matplotlib.pyplot as plt
import numpy as np

# Assuming 'file_contents' is a list of strings, with each string containing data similar to your example.


# Define a set of colors for each dataset
colors = plt.cm.viridis(np.linspace(0, 1, len(file_contents)))
plt.figure(figsize=(4,4))
# Loop over each file and plot its data
combined_x = []
combined_y = []

for idx, content in enumerate(file_contents[:3]):  # Process only the first 3 files
    # Extract the relevant portion of the file content
    input_string = content[24:]

    # Replace newline characters with tabs to maintain the tabular format
    input_string = input_string.replace('\n', '\t')

    # Split the string into parts and clean up empty entries
    parts = [x.strip() for x in input_string.split('\t') if x.strip()]

    # Group the cleaned parts into sublists of 3 columns (x, y, uncertainty)
    column_groups = [parts[i:i + 3] for i in range(0, len(parts), 3)]

    # Convert the string values in each group to float
    data = [[float(x) for x in group] for group in column_groups]

    # Separate the x, y, and uncertainty values
    x_values = [group[0]/60 for group in data]
    y_values = [group[1] for group in data]
    uncertainties = [group[2] for group in data]

    filtered_indices = [i for i, unc in enumerate(uncertainties) if (unc < 0.09) & (unc >0.001)]
    
    # Filter x, y, and uncertainties based on the condition
    filtered_x = [x_values[i] for i in filtered_indices]
    filtered_y = [y_values[i] for i in filtered_indices]
    filtered_uncertainties = [uncertainties[i] for i in filtered_indices]
    
    # Append the current file's filtered x and y values
    combined_x.extend(filtered_x)
    combined_y.extend(filtered_y)
    
    # Plot the data with error bars for the filtered values
    plt.errorbar(
        filtered_x, filtered_y, yerr=filtered_uncertainties, 
        fmt='o', capsize=5, color=colors[idx]
    )

# Convert combined_x and combined_y to numpy arrays
combined_x = np.array(combined_x)
combined_y = np.array(combined_y)
import scipy
# Fit a single line to the combined data points
slope,intercept, r_value, p_value, std_err = scipy.stats.linregress(np.log10(combined_x),combined_y)
print(r_value**2,p_value,std_err)

# Generate the fitted line
x_line = np.linspace(min(combined_x), max(combined_x), 100)
y_line = slope * np.log10(x_line) + intercept

# Plot the single fitted line
plt.plot(
    x_line, y_line, color='black', linestyle='--', 
    label=f'{slope:.2f}log(x) + {intercept:.2f}'
)

# Set axis labels and title
plt.xlabel('Background intensity')
plt.ylabel(r'$\frac{t_{OFF}}{t_{tot}}$')
plt.title('Plot with Error Bars and Single Fitted Line')

plt.xlabel('Background intensity')
plt.ylabel(r'$\frac{t_{OFF}}{t_{tot}}$')
plt.title('Plot with Error Bars from Multiple Files')

# Show the legend to differentiate the datasets
plt.legend()
plt.xscale('log',base=10)
plt.xlim([10,1000])
plt.ylim([0,0.5])


# Display the plot
plt.show()
#%%


#%%

#%%
for b in range(400,1000):
    plt.plot(sorted_frame_intensity2[b])
    plt.hlines(sorted_thresholds[b],xmin=0,xmax=3000,color='black')
    plt.show()
#%%

b= 21
plt.plot(groups[0][b][0:1000])
plt.hlines(y, xmin, xmax, kwargs)
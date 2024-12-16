# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 15:57:06 2024
Version: ALPHA
Code to make results from piimagine_512^2.
prerequistions:
Reduce_size_512SPAD - The code uses processed data and should be in the form of
meta_acq##### and movie_arr_acq#####.npy in the same map.
SEP_D - Code to detect particles and get there properties.

@author: David van Houten
"""
import sys
# Add the directory containing SEP_D.py
sys.path.append('C:/David/SEP_D/') # Replace with your actual path
from SEP_D import *  # Import all functions
import matplotlib.gridspec as gridspec
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import os
import numba
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import curve_fit
import json
import dask.array as da
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import ScalarFormatter

plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

def intensity_traces(movie_array,segmentation_map,objects, background_map=None):
    """Neemt de locaties van quantum dots gedetecteerd door sep en 
    bepaalt de intensity traces van de gevonden quantum dots
    dit gebeurt door het optellen van intensity traces van de losse pixels"""
    
    int_traces = np.zeros((len(objects),movie_array.shape[0]))
    if background_map is not None:
        norm_bkg_map  = background_map/np.max(background_map)
        for i ,key in enumerate(objects): 
            loc = np.array(np.where(segmentation_map == key)).T
            for x,y in loc:
                int_traces[i] += movie_array[:,x,y]-(background_map[x,y]/movie_array.shape[0]) 
    else:
        for i ,key in enumerate(objects): 
            loc = np.array(np.where(segmentation_map == key)).T
            for x,y in loc:
                int_traces[i] += movie_array[:,x,y]
    return int_traces


def sum_columns(array, n):  
    """
    Sums every n columns of a 2D numpy array.  
    Bijvoorbeeld herbinnen van intensity traces of decaycurves
    """
    if array.shape[1] % n != 0:
        raise ValueError("The number of columns must be divisible by n.")
    
    reshaped_array = array.reshape(array.shape[0], -1, n)
    
    summed_array = reshaped_array.sum(axis=2)
    
    return summed_array
  
def mono_exp(t_array,k,I):
    """Monoexponentieel verval zonder achtergrond"""
    return I*np.exp(-k*t_array)# + bg

def allign_dec_arr(dec_arr,objects,total_image,gate_steps):
    """Zorgt dat onderling delay tussen pixels die bij een qd horen opgelost
    wordt door onderling te verschuiven en de maximale overlap te vinden"""
    argdiff = np.array([])
    for key in objects: #loop over alle gevonden qd's 
        pixel_qd = np.where(segmentation_map == key) #alle pixels die bij een qd horen vinden
        idx_max_int = int(np.argmax(total_image[pixel_qd])) #pixel met hoogste intensiteit
        ref_dec = dec_arr[:,pixel_qd[0][idx_max_int],pixel_qd[1][idx_max_int]]
        # decaycurve met meeste counts vinden per quantum dot
        for j in range(len(pixel_qd[0])): #loopen over alle pixels die bij qd horen
            dec = dec_arr[:,pixel_qd[0][j],pixel_qd[1][j]]
            correlations = np.zeros(gate_steps)
            # alle verschillende delays tussen de qd's vinden en overlap
            for k in range(gate_steps):
                correlations[k] = np.dot(ref_dec,np.roll(dec,k))
            dec_arr[:,pixel_qd[0][j],pixel_qd[1][j]] = np.roll(dec,correlations.argmax())
            argdiff = np.append(argdiff,correlations.argmax())
    return dec_arr,argdiff

def decay_curves(movie_arr,seg_map,nstep,nframes,image,objects,background_map=None):
    """Maakt de decaycurve voor iedere qd, die gedetecteerd is in de segmentation map, 
    heeft movie_arr, seg_map, nstep(aantal gate steps), nframes(aantal frames) en image 
    (opgetelde movie_arr) nodig, output is een 2D array (aanatal qd's * nstep)"""
    dec_arr = np.zeros((nstep,movie_arr.shape[1],movie_arr.shape[2]))
    for i in range(nframes):
        dec_arr += movie_arr[i*nstep:(i+1)*nstep] # sommeeert alles wat in dezelfde gate zit
    total_decay = np.sum(dec_arr,axis=(1,2))
    dec_arr, argdiff = allign_dec_arr(dec_arr, objects, image, nstep)
    if background_map:
        dec_curves = intensity_traces(dec_arr, seg_map, objects)
    else:
        dec_curves = intensity_traces(dec_arr, seg_map, objects,background_map=background_map)     
    
    return dec_curves   

@numba.njit(nogil=True,error_model='numpy',cache=True, parallel = True) #voor de snelheid
def model_snel(t_array,dt,k,i): 
    """De oplossing van de integraal onder alle condities, uitgewerkt met Freddy's mathematica tovenarij"""
    t_na_delay = (t_array - dt) % tlaser #compenseert voor piek niet op t = 0, maar op t = 0 + dt, met dt vertraging van detector
    t0 = np.mod(t_na_delay,tlaser) - tlaser #omdat het model de goede waarde tussen -tlaser en 0 geeft en de fit van 0 naar tlaser moet
    y = np.zeros(t0.shape)
    for idx,t in enumerate(t0):
        if -tgate <= t <= 0:
            y[idx] = (1 - np.exp(k*tlaser) - np.exp(-k*t) + np.exp(-k*(tgate - tlaser + t)))/(1 - np.exp(tlaser*k))
        else:
            y[idx] = np.exp(-k*(tgate + t))*(-1 + np.exp(tgate*k))/(-1 + np.exp(tlaser*k))
    return i*y




#%%
def get_index(path):
    # List all files in the directory
    files = os.listdir(path)

    # Filter meta and movie files
    meta_files = [f for f in files if f.startswith('meta_acq')]
    movie_files = [f for f in files if f.startswith('movie_arr_acq')]

    # Check if there are any files available
    if not meta_files or not movie_files:
        print("No meta or movie files found in the directory.")
        return None

    # Print the list of files for user selection
    print("Available meta files:")
    for i, file in enumerate(meta_files):
        print(f"{i}: {file}")
    print("\nAvailable movie files:")
    for i, file in enumerate(movie_files):
        print(f"{i}: {file}")

    # Validate the selected index
    while True:
        try:
            index = int(input("Select the index of the files: "))
            if index < 0 or index >= len(meta_files) or index >= len(movie_files):
                raise ValueError
            break
        except ValueError:
            print("Invalid index. Please enter a valid index within the range of available files.")
            
    return index

def process_meta(meta, mode='intensity'):
    """
    Process metadata for laser and timing parameters.
    
    Parameters:
        meta (dict): Dictionary containing metadata with keys such as 'Laser frequency', 'Gate width', 'Gate step size', 'Gate steps', 'Frames', 'Integration time'.
        mode (str): Mode of operation, either 'gated' or 'intensity'. Default is 'gated'.
        version (str): Version of the code to use, either 'new' or 'old'. Default is 'new'.
    
    Returns:
        tuple: Processed parameters including frequency, time between laser pulses, gate width, gate step size, number of gate steps, number of frames, and integration time.
    """
    if mode not in ['gated', 'intensity']:
        raise ValueError("Invalid mode. Use 'gated' or 'intensity'.")

    freq_laser = float(meta['Laser frequency'].removesuffix('MHz')) if mode == 'gated' else 10e6
    tgate = float(meta['Gate width'].removesuffix('ns')) if mode == 'gated' else None
    tstep = float(meta['Gate step size'].removesuffix('ps')) / 1000 if mode == 'gated' else 1

    tlaser = 1 / freq_laser * 1000  # time between laser pulses in ns
    nstep = int(meta['Gate steps']) if mode == 'gated' else 1
    nframes = int(meta['Frames'])
    tint = float(meta['Integration time'].removesuffix('ms'))

    return [freq_laser, tlaser, tgate, tstep, nstep, nframes, tint]

def select_and_load_files(path, fraction=1):
    # List all files in the directory
    # Infer mode from the path
    if 'intensity' in path:
        mode = 'intensity'
    elif 'gated' in path:
        mode = 'gated'
    else:
        print("Cannot determine mode from the path. Please ensure the path contains 'intensity' or 'gated'.")
        return
    index = get_index(path)
    files = os.listdir(path)

    # Filter meta and movie files
    meta_files = [f for f in files if f.startswith('meta_acq')]
    movie_files = [f for f in files if f.startswith('movie_arr_acq')]

    # Check if there are any files available
    if not meta_files or not movie_files:
        print("No meta or movie files found in the directory.")
        return

    # Validate the selected index
    if index < 0 or index >= len(meta_files) or index >= len(movie_files):
        print("Invalid index. Please enter a valid index within the range of available files.")
        return

    # Load the selected meta file
    with open(os.path.join(path, meta_files[index]), 'r') as data_json:
        meta_raw = json.load(data_json)

    # Load the movie file lazily with memory mapping
    file_path = os.path.join(path, movie_files[index])
    movie_data = np.load(file_path, allow_pickle=True, mmap_mode='r')

    # Determine the slicing range to load only a fraction of the file
    total_frames = movie_data.shape[0]  # Assuming the first axis corresponds to frames
    num_frames_to_load = int(total_frames * fraction)

    # Slice the data to load only the specified fraction
    movie_data_partial = movie_data[:num_frames_to_load]

    # Use Dask to process the partial data
    test = da.from_array(movie_data_partial, chunks='auto')
    movie_arr = test.compute()

    # Compute the total image (sum over frames)
    total_image = np.sum(movie_arr, axis=0)
    print(total_image)
    meta = process_meta(meta_raw, mode=mode)

    # Display the image
    plt.figure()
    plt.imshow(total_image, cmap='jet')
    plt.title("Summed Image")
    plt.show()
    
    return meta_raw,meta, movie_arr, total_image




path = r'C:/SPAD512S/20241128/Jaco/data/intensity_images/'
path = r'C:/David/24-10-24 gQD/data/intensity_images/'
path = r'C:/David/24-10-22 gQD int-traces/data/intensity_images/'

meta_raw,meta,movie_arr,total_image = select_and_load_files(path)

#'important' meta data
freq_laser, tlaser, tgate, tstep, nstep, nframes, tint = meta

#%% SEP detectie van QDs 
from matplotlib import font_manager
font_path = r'C:/Windows/Fonts/framd.ttf'  # Update this with the path to your font file
font_prop = font_manager.FontProperties(fname=font_path)

def slice_and_bkg_image(movie_arr, total_image, xlim=None, ylim=None, mesh_size=20, sigma=2):
    # Plot the full image and full background first
    fig, axes = plt.subplots(1, 3, figsize=(8,5))  # 1 row, 2 columns

    axes[0].imshow(total_image, origin='lower', cmap='jet', vmax=np.max(total_image))
    axes[0].set_title('Image')
    axes[0].set_xlabel('Pixels', fontproperties=font_prop)
    axes[0].set_ylabel('Pixels', fontproperties=font_prop)
    axes[0].xaxis.set_major_locator(MultipleLocator(100)) 
    axes[0].xaxis.set_minor_locator(MultipleLocator(50)) 
    axes[0].tick_params(axis='x', which='major', labelsize=10)
    axes[0].tick_params(axis='x', which='minor',  labelsize=8)
    axes[0].yaxis.set_major_locator(MultipleLocator(100))
    axes[0].yaxis.set_minor_locator(MultipleLocator(50))  
    axes[0].tick_params(axis='y', which='major', labelsize=10)
    axes[0].tick_params(axis='y', which='minor',  labelsize=8)
    
    image, bkg = background_subtraction(total_image, mesh_size=mesh_size, sigma=sigma, background_map=True)

    axes[1].imshow(bkg, origin='lower', cmap='jet', vmax=np.max(total_image))
    axes[1].set_title('Background')
    axes[1].set_xlabel('Pixels', fontproperties=font_prop)
    axes[1].set_ylabel('Pixels', fontproperties=font_prop)
    axes[1].xaxis.set_major_locator(MultipleLocator(100)) 
    axes[1].xaxis.set_minor_locator(MultipleLocator(50)) 
    axes[1].tick_params(axis='x', which='major', labelsize=10)
    axes[1].tick_params(axis='x', which='minor',  labelsize=8)
    axes[1].yaxis.set_major_locator(MultipleLocator(100))
    axes[1].yaxis.set_minor_locator(MultipleLocator(50))  
    axes[1].tick_params(axis='y', which='major', labelsize=10)
    axes[1].tick_params(axis='y', which='minor',  labelsize=8)
    

    # Slice the image based on xlim and ylim if provided
    if xlim:
        if isinstance(xlim, int):
            xlim = (xlim, xlim + 1)
        movie_arr = movie_arr[:, xlim[0]:xlim[1], :]
        image = image[xlim[0]:xlim[1], :]
        bkg = bkg[xlim[0]:xlim[1], :]

    if ylim:
        if isinstance(ylim, int):
            ylim = (ylim, ylim + 1
                    )
        movie_arr = movie_arr[:, :, ylim[0]:ylim[1]]
        image = image[:, ylim[0]:ylim[1]]
        bkg = bkg[:, ylim[0]:ylim[1]]

    axes[2].imshow(image, origin='lower', cmap='jet')
    axes[2].set_title('Image - Background')
    axes[2].set_xlabel('Pixels', fontproperties=font_prop)
    axes[2].set_ylabel('Pixels', fontproperties=font_prop)
    axes[2].xaxis.set_major_locator(MultipleLocator(100)) 
    axes[2].xaxis.set_minor_locator(MultipleLocator(50)) 
    axes[2].tick_params(axis='x', which='major', labelsize=10)
    axes[2].tick_params(axis='x', which='minor',  labelsize=8)
    axes[2].yaxis.set_major_locator(MultipleLocator(100))
    axes[2].yaxis.set_minor_locator(MultipleLocator(50))  
    axes[2].tick_params(axis='y', which='major', labelsize=10)
    axes[2].tick_params(axis='y', which='minor',  labelsize=8)

    plt.tight_layout()
    plt.show()
    return movie_arr, image,bkg

# Example usage:
movie_arr_cut, total_image_cut,bkg = slice_and_bkg_image(movie_arr, total_image)
#movie_arr_cut, total_image_cut,bkg = slice_and_bkg_image(movie_arr, total_image,xlim=[100,200],ylim=[100,200])
#%%
filters = [
    {'key': 'npix', 'lower': 4},
    {'key': 'npix', 'upper': 20000},
    {'key': 'mean_int', 'lower': 10},
    {'key': 'mean_int', 'upper': 5000000}
] 

threshold = 0.5
#objects, segmentation_map, deblend_info = extract_objects(total_image_cut, threshold,filters=filters, 
#                                        deblending= True)
objects, segmentation_map = extract_objects(movie_arr_cut, threshold,filters=filters)
#objects, segmentation_map  = extract_objects(image, threshold,filters=filters)
#de threshold die gebruikt worddt in gated imaging is lager dan in normale imaging
#dit komt omdat er relatief meer achtergrond is, hier nog niet de detectie met twee verschillende thresholds 
#die kan gekopieerd worden uit de code voor intensity traces mocht dat nodig zijn
print(fr'Aantal gevonden quantum dots door algoritme {len(objects)}')
x = [objects[i]['x'] for i in objects]
y = [objects[i]['y'] for i in objects]


plot_objects(total_image_cut, objects,radius=3)
#%%
sum_ = np.mean(total_image_cut[segmentation_map == 4]/len(total_image_cut))
print(sum_)
#%%
plot_segmap(segmentation_map)
#%%
plot_branches(total_image_cut, deblend_info[0], deblend_info[1]) # shows also the filtered objects so you can see what is filtered



#%%
def intensity_traces(movie_array,segmentation_map,objects, background_map=None):
    """Neemt de locaties van quantum dots gedetecteerd door sep en 
    bepaalt de intensity traces van de gevonden quantum dots
    dit gebeurt door het optellen van intensity traces van de losse pixels"""
    
    int_traces = np.zeros((len(objects),movie_array.shape[0]))
    if background_map is not None:
        norm_bkg_map  = background_map/np.max(background_map)
        for i ,key in enumerate(objects): 
            loc = np.array(np.where(segmentation_map == key)).T
            for x,y in loc:
                int_traces[i] += movie_array[:,x,y]-(background_map[x,y]/movie_array.shape[0]) 
    else:
        for i ,key in enumerate(objects): 
            loc = np.array(np.where(segmentation_map == key)).T
            for x,y in loc:
                int_traces[i] += movie_array[:,x,y]
    return int_traces

frame_intensity1 = intensity_traces(movie_arr_cut, segmentation_map, objects)



#%%


def intensity_plots(frame_intensity, start_idx, num_plots, return_hist=False, summed=1):
    col = 4  # Fixed number of columns for 16 plots
    row = 4  # Fixed number of rows for 16 plots
    bins = 20  # Define the number of bins for the histogram, adjust as needed

    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(row, 3 * col, width_ratios=[8, 1, 2.2] * col, wspace=0, hspace=0.0)
    hists = np.zeros((frame_intensity.shape[0], bins))
    thresholds = np.zeros((frame_intensity.shape[0], 2))

    for idx in range(num_plots):
        current_row = idx // col
        current_col = idx % col
        if start_idx + idx >= frame_intensity.shape[0]:
            break

        if current_row < row:
            # Image plot
            img_ax = fig.add_subplot(gs[current_row, 3 * current_col])
            img_ax.plot(np.arange(0,nframes*nstep/summed)*int(tint)*10**-3*summed, frame_intensity[start_idx + idx], c='#600000ff', alpha=0.95 ,linewidth=0.05)
            #img_ax.set_xlim([0, nframes/summed*nstep*int(tint)*10**-3*summed])  # Adjust x-axis to cover the entire frame
            #img_ax.set_title(f'QD {start_idx + idx}')
            
            #img_ax.set_ylabel('Intensity (counts)')

            min_val = np.min(frame_intensity[start_idx + idx])
            max_val = np.max(frame_intensity[start_idx + idx])
            img_ax.set_ylim(bottom=0)
            #img_ax.set_yticks([max_val * 0.25, max_val * 0.5, max_val * 0.75])
            #img_ax.set_yticklabels([''] * 3)   
            #img_ax.set_xticks([])
            img_ax.set_ylim([0, max_val])
            #img_ax.set_xlim([0, (nframes/summed*nstep*int(tint)*10**-3*summed)/30])
            img_ax.tick_params(axis='y', direction='in')
            bin_edges = np.linspace(0, max_val, bins + 1)
            hist, _ = np.histogram(frame_intensity[start_idx + idx], bins=bin_edges)
            hists[start_idx + idx] += hist


            hist_ax = fig.add_subplot(gs[current_row, 3 * current_col + 1])
            hist_ax.barh(bin_edges[:-1], hist / max(hist), height=np.diff(bin_edges), color='#600000ff', alpha=0.5, align='edge', edgecolor='black', linewidth=0.5)
            hist_ax.set_ylim([0, max_val])
            hist_ax.set_yticks([]) 
            hist_ax.set_xticks([])  
            hist_ax.set_xlim([0, 1.2])
            maxi = tint * frame_intensity.shape[1]*10**-3*summed
            interval = 10
            xticks = list(range(0, int(maxi) + interval, interval))
            xticklabels = [str(tick) for tick in xticks]
            img_ax.set_xlim([0,np.max(xticks)])
            img_ax.set_xticklabels(' ')
    
        if current_row == row-1:
            img_ax.set_xlabel('time (sec)', fontproperties=font_prop)
            img_ax.set_xticks(xticks)
            img_ax.set_xticklabels(xticklabels, fontproperties=font_prop)  # Set the labels with the custom font

            #img_ax.xaxis.set_major_locator(MultipleLocator(20))
            #img_ax.xaxis.set_minor_locator(MultipleLocator(10))
    plt.tight_layout()
    plt.show()


    if return_hist == True:
        return hists, thresholds

hists = np.zeros((frame_intensity1.shape[0], 20))
thresholds = np.zeros((frame_intensity1.shape[0], 2))
num_plots=len(frame_intensity1)
num_figures = (num_plots + 15) // 16  # Calculate the number of figures needed
for i in range(num_figures):
    start_idx = i * 16
    # Determine how many plots to draw for this figure
    plots_in_current_figure = min(16, num_plots - start_idx)
    
    hist,threshold = intensity_plots(frame_intensity1, start_idx, num_plots, 
                                     return_hist=True, summed = 1)
    hists += hist


#%%




#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Assuming frame_intensity1 is defined and contains your intensity data
# Replace frame_intensity1 with your actual data variable

for j in range(frame_intensity1.shape[0]):
    
    # Define your x and y arrays
    x = np.arange(0, len(frame_intensity1[j]) * 0.001, 0.001)  # Assuming each frame represents 1 ms
    y = np.array(frame_intensity1[j])
    
    for i in np.arange(0, np.max(x), 0.2):
        plt.plot(x, y)
        plt.ylim([0, np.max(y)])
        plt.xlim([0 + i, 0.20 + i])
        
        # Find the indices of the local maxima
        max_ind, _ = find_peaks(y, distance=7)
        
        # Initialize a list to store the last minima corresponding to each maximum
        last_minima = []
        
        for idx in max_ind:
            # Start checking at least 2 points before the maximum
            if idx <= 2:
                continue  # Skip if the maximum is too close to the start
            
            current_min_idx = idx - 2
            while current_min_idx > 0 and y[current_min_idx - 1] < y[current_min_idx]:
                current_min_idx -= 1
            last_minima.append(current_min_idx)
        
        # Optionally, plot the maxima and the last minima found
        plt.scatter(x[max_ind], y[max_ind], label='Maxima', color='red')
        if last_minima:  # Check if we have any minima to plot
            plt.scatter(x[last_minima], y[last_minima], label='Last Minima', color='blue')
        
        plt.title(f'Frame {j}')
        plt.legend()
        plt.show()


#%%
x=[3,0,6,8,9,10,11,12,14,19,21]

sum_trace=np.zeros(frame_intensity1.shape[1])
for i in x:
    sum_trace += frame_intensity1[i]
    fig, ax = plt.subplots(figsize=(8, 2))

    # Plotting the data
    ax.plot(np.linspace(0, 60, 12000), sum_trace, color='blue', linestyle='-', linewidth=1, label='Signal')

    # Enhancing the aesthetics

    ax.set_xlabel('Time (s)',fontproperties=font_prop)
    ax.set_ylabel('Counts / 10 ms',  fontproperties=font_prop)
    ax.set_xlim([0, 20])
    #ax.set_ylim([0, 3000])
    ax.set_ylim(bottom=0)

    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    # Show the plot
    plt.tight_layout()
    plt.show()




# Create the plot
fig, ax = plt.subplots(figsize=(8, 2))

# Plotting the data
ax.plot(np.linspace(0, 60, 12000), sum_trace, color='blue', linestyle='-', linewidth=1, label='Signal')

# Enhancing the aesthetics

ax.set_xlabel('Time (s)',fontproperties=font_prop)
ax.set_ylabel('Counts / 10 ms',  fontproperties=font_prop)
ax.set_xlim([0, 20])
ax.set_ylim([0, 3000])

ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
# Show the plot
plt.tight_layout()
plt.show()

        
  

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def compute_segment_properties(intensity, nframes, nstep, tstep):
    x = np.arange(0, nframes * nstep) * 10**-3
    # Find the indices of the local maxima
    max_indices, _ = find_peaks(y, distance=7)
    
    # Initialize a list to store the last minima corresponding to each maximum
    last_minima = []
    
    for idx in max_indices:
        # Start checking at least 2 points before the maximum
        if idx <= 2:
            continue  # Skip if the maximum is too close to the start
        
        current_min_idx = idx - 2
        while current_min_idx > 0 and y[current_min_idx - 1] < y[current_min_idx]:
            current_min_idx -= 1
        last_minima.append(current_min_idx)
    
    min_indices = np.array(last_minima)

    segment_sums = []
    weighted_sums = []
    
    for i in range(len(max_indices) - 1):
        start = max_indices[i]
        end = min_indices[min_indices > max_indices[i]][0]
        segment_length = end - start
        
        new_x = np.linspace(0.5, segment_length - 0.5, segment_length) * tstep
        segment_sum = np.sum(intensity[start:end])
        weighted_sum = np.sum(new_x * intensity[start:start + len(new_x)])
        
        segment_sums.append(segment_sum)
        weighted_sums.append(weighted_sum)
    
    average_lifetimes = [weighted_sums[i] / segment_sums[i] for i in range(len(weighted_sums))]
    return average_lifetimes, segment_sums

def plot_histogram2D(average_lifetimes, segment_sums, xstep, ystep,  num_plots,title=None):
    col = 4  # Fixed number of columns for 16 plots
    row = 4  # Fixed number of rows for 16 plots
    
    fig = plt.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(row, col, wspace=0.4, hspace=0.4)

    plt.suptitle(title)
    plt.ylabel('PL intensity (cts / 200ms)')
    plt.gca().yaxis.set_label_coords(-0.1, 0.5)  # (x, y) coordinates
    for spine in ['right', 'top', 'left', 'bottom']:
        plt.gca().spines[spine].set_visible(False)
    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks
    # Flatten axes array for easy iteration
    for idx in range(num_plots):
        current_row = idx // col
        current_col = idx % col
        
        ax = fig.add_subplot(gs[current_row, current_col])

        hist, xedges, yedges = np.histogram2d(average_lifetimes[idx], segment_sums[idx], bins=(np.arange(0, max(average_lifetimes[idx]), xstep),
                                                                                        np.arange(0, max(segment_sums[idx]), ystep)))
        
    
        im = ax.imshow(hist.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', origin='lower', cmap='coolwarm')
        #cbar = fig.colorbar(im, ax=ax)
        
        #cbar.set_ticks([])
        ax.set_xlim([0,40])
        
        
    
        if current_row == row-1:
            ax.set_xlabel('Average Lifetime (ns)')

    plt.show()

def FLIDS(frame_intensity1, nframes, nstep, tstep):
    FLID = []
    for index, intensity in enumerate(frame_intensity1):
        average_lifetimes, segment_sums = compute_segment_properties(intensity, nframes, nstep, tstep)
        FLID.append([average_lifetimes, segment_sums])
    
    num_plots = len(FLID)
    average_lifetimes = [item[0] for item in FLID]
    segment_sums      = [item[1] for item in FLID]

    
    num_figures = (num_plots + 15) // 16  # Calculate the number of figures needed
    for i in range(num_figures):
        start_idx = i * 16
        plots_in_current_figure = min(16, num_plots - start_idx)
        # plot_histogram2D(average_lifetimes[start_idx:start_idx + plots_in_current_figure], 
        #                  segment_sums[start_idx:start_idx + plots_in_current_figure], 
        #                  np.max([max(al) for al in average_lifetimes]) / 100, 
        #                  np.max([max(ss) for ss in segment_sums]) / 100, 
        #                  plots_in_current_figure,title = f'Decay Curves {i+1}')
        plot_histogram2D(average_lifetimes[start_idx:start_idx + plots_in_current_figure], 
                         segment_sums[start_idx:start_idx + plots_in_current_figure], 
                         np.max(average_lifetimes[i]) / 50, 
                         np.max(segment_sums[i]) / 25, 
                         plots_in_current_figure,title = f'FLIDs page:{i+1}')


FLIDS(frame_intensity1, nframes, nstep, tstep)
#%%
def compute_flid(intensity, tstep, nstep):
    # Compute the total time for each gate
    time_bins = np.arange(0, nstep) * tstep
    # Sum intensity over the gates
    intensity_sum = np.sum(intensity)  
    # Compute the weighted sum for each gate step
    weighted_sum = np.dot(time_bins, intensity.T)
    # Compute the average lifetime for each gate step
    average_lifetimes = weighted_sum / intensity_sum
    return average_lifetimes, intensity_sum

def plot_histogram2D(average_lifetimes, segment_sums, xstep, ystep, num_plots, title=None):
    col = 4  # Fixed number of columns for 16 plots
    row = 4  # Fixed number of rows for 16 plots
    
    fig = plt.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(row, col, wspace=0.4, hspace=0.4)

    plt.suptitle(title)
    for idx in range(num_plots):
        current_row = idx // col
        current_col = idx % col
        
        ax = fig.add_subplot(gs[current_row, current_col])

        hist, xedges, yedges = np.histogram2d(
            average_lifetimes[idx], segment_sums[idx],
            bins=(np.arange(0, np.max(average_lifetimes[idx]) + xstep, xstep),
                  np.arange(0, np.max(segment_sums[idx]) + ystep, ystep))
        )
        
        im = ax.imshow(hist.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                       aspect='auto', origin='lower', cmap='coolwarm')
        
        ax.set_xlim([0, np.max(average_lifetimes[idx])])
        ax.set_ylim([0, np.max(segment_sums[idx])])
        
        if current_row == row - 1:
            ax.set_xlabel('Average Lifetime (ns)')
        if current_col == 0:
            ax.set_ylabel('Intensity (cts / 200ms)')
    
    plt.show()

def FLIDS(frame_intensity1):
    # Extract parameters from metadata
    FLID = []
    for intensity in frame_intensity1:
        all_avg_lifetimes = []
        all_segment_sums = []
        
        for frame_idx in range(nframes):
            frame_start = frame_idx * nstep
            frame_end = frame_start + nstep
            frame_intensity = intensity[frame_start:frame_end]
            
            average_lifetimes, segment_sums = compute_flid(frame_intensity, tstep, nstep)
            all_avg_lifetimes.append(average_lifetimes)
            all_segment_sums.append(segment_sums)
        
        FLID.append([all_avg_lifetimes, all_segment_sums])
    
    num_plots = len(FLID)
    average_lifetimes = [item[0] for item in FLID]
    segment_sums = [item[1] for item in FLID]
    
    num_figures = (num_plots + 15) // 16  # Calculate the number of figures needed
    for i in range(num_figures):
        start_idx = i * 16
        plots_in_current_figure = min(16, num_plots - start_idx)
        plot_histogram2D(
            average_lifetimes[start_idx:start_idx + plots_in_current_figure],
            segment_sums[start_idx:start_idx + plots_in_current_figure],
            xstep=np.max([np.max(al) for al in average_lifetimes]) / 50,
            ystep=np.max([np.max(ss) for ss in segment_sums]) / 25,
            num_plots=plots_in_current_figure,
            title=f'FLIDs page:{i+1}'
        )
FLIDS(frame_intensity1)
#%%   
    plot_histogram2D([np.concatenate(average_lifetimes)],[np.concatenate(segment_sums)], 
    np.max([max(al) for al in average_lifetimes]) / 100, 
    np.max([max(ss) for ss in segment_sums]) / 100, 1)

    plot_histogram2D(average_lifetimes, segment_sums, np.max(average_lifetimes) / 30, np.max(segment_sums) / 30, f'Frame {index}')
#%%


#%%
av_life = np.array([item[0] for item in max_values] ) # Extracts [a1, a2, a3]
segment_sums = np.array([item[1] for item in max_values] ) # Extracts [b1, b2, b3]
# After processing all frames, create a new FLID based on the maximum values collected
FLID(av_life, segment_sums, 0.5, 25)

#%%
#CODE FOR DECAY CURVES
from matplotlib import font_manager
font_path = r'C:/Windows/Fonts/FRABK.ttf'  # Update this with the path to your font file
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['font.weight'] = font_prop.get_weight()
plt.rcParams['font.size'] = font_prop.get_size()
# Calculate grid size
# Function to plot decay curves
def plot_decay_curves(dec_curves, start_idx, num_plots,fit=None,title=None):
    col = 4  # Fixed number of columns for 16 plots
    row = 4  # Fixed number of rows for 16 plots
    
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(row, 2 * col, width_ratios=[8, 3] * col, wspace=0, hspace=0.0)

    plt.suptitle(title)
    plt.ylabel(r'Intensity (normalized)')
    
    plt.gca().yaxis.set_label_coords(-0.05, 0.5)  # (x, y) coordinates
    for spine in ['right', 'top', 'left', 'bottom']:
        plt.gca().spines[spine].set_visible(False)
    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks
    #ax.set_title(f'QD {start_idx + idx}')
    # Flatten axes array for easy iteration

    for idx in range(num_plots):
        current_row = idx // col
        current_col = idx % col
        if start_idx + idx >= dec_curves.shape[0]:
            break
        ax = fig.add_subplot(gs[current_row, 2 * current_col])
        ax.plot(tstep * np.arange(0, dec_curves.shape[1]), dec_curves[start_idx + idx], c='royalblue')


        if fit is not None:
            fit_list = fit if isinstance(fit, list) else [fit]
            for f in fit_list:
                if f.shape[1] == 3:
                    ax.plot(tstep * np.arange(0, dec_curves.shape[1]), 
                            model_snel(datat, *f[start_idx + idx]), '.', c='orangered')
                elif f.shape[1] == 2:
                    ax.plot(np.roll(tstep * np.arange(0, dec_curves.shape[1]), 
                                    -np.where(dec_curves[start_idx + idx] == np.max(dec_curves[start_idx + idx]))[0][0])[:-2], 
                            mono_exp(datat, *f[start_idx + idx])[:-2], '.', c='green')
                elif f.shape[1] == 5:
                    ax.plot(np.roll(tstep * np.arange(0, dec_curves.shape[1]), 
                                    -np.where(dec_curves[start_idx + idx] == np.max(dec_curves[start_idx + idx]))[0][0])[:-2], 
                            double_exp(datat, *f[start_idx + idx])[:-2], '.', c='yellow')        


        if current_row < row:    

            ax.set_yscale('log')
            #ax.set_xlabel(r'$\tau$ (ns)')
    
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-1, 1))
    
            ax.yaxis.set_major_formatter(formatter)
            ax.yaxis.set_minor_formatter(formatter)
    
            max_value = np.max(dec_curves[start_idx + idx])
            min_value = np.min(dec_curves[start_idx + idx])
            range_value = max_value - min_value
            power = np.floor(np.log10(max_value)).astype(int)
    
            # Check the range and set tick intervals accordingly
            if range_value < 10**power:
                major_tick_interval = 10 ** power * 0.2
                minor_tick_interval = major_tick_interval / 2
            else: 
                major_tick_interval = 10 ** power
                minor_tick_interval = major_tick_interval / 2
            ax.set_xticks([])
            
            ax.set_yticks([])
            ax.set_yticks([round(max_value,1)])
            ax.set_yticklabels([f'{round(max_value / 1000,1)}'], fontproperties=font_prop)
            
            ax.tick_params(axis='y', which='minor', labelleft=False)

            maxi = tstep * dec_curves.shape[1]
            interval = 10
            xticks = list(range(0, int(maxi) + interval, interval))
            xticklabels = [str(tick) for tick in xticks]
            ax.set_xlim([0,np.max(xticks)])


            #ax.ticklabel_format(style='scientific', axis='y', scilimits=(-1, 1))
        if current_row == 0:
            ax.set_title('$\\mathdefault{{10^3}}$',loc='left', fontproperties=font_prop)

        if current_row == row - 1:
            ax.set_xlabel(r'$\tau$ (ns)', fontproperties=font_prop)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, fontproperties=font_prop)  # Set the labels with the custom font
            ax.xaxis.set_major_locator(MultipleLocator(20))
            ax.xaxis.set_minor_locator(MultipleLocator(10))


    plt.tight_layout()
    plt.show()

dec_curves = decay_curves(movie_arr_cut, segmentation_map, nstep, nframes, total_image_cut,objects)

num_plots=len(dec_curves)
# Check the number of plots and call the function accordingly
num_figures = (num_plots + 15) // 16  # Calculate the number of figures needed
for i in range(num_figures):
    start_idx = i * 16
    # Determine how many plots to draw for this figure
    plots_in_current_figure = min(16, num_plots - start_idx)
    plot_decay_curves(dec_curves, start_idx, plots_in_current_figure,title=f'Decay Curves page:{i+1}')


#%%fitten aan model met convolutie

a =time.perf_counter()
datat = np.linspace(0, tlaser - tstep, num=int(nstep))

model_snel_popt = np.zeros((len(objects),3))

for i, datai in enumerate(dec_curves):
    try:
        model_snel_popt[i],_ = curve_fit(model_snel, datat, datai,p0=[85,1/60,100],bounds=([0,0,0],[100000,0.4,10_000_000]))
    except RuntimeError as e:
        print(f"Fout bij fitten decaycurve {i}: {e}")
        continue

print(f'Tijd nodig voor fit {time.perf_counter() - a}')
 

num_figures = (num_plots + 15) // 16  # Calculate the number of figures needed
for i in range(num_figures):
    start_idx = i * 16
    # Determine how many plots to draw for this figure
    plots_in_current_figure = min(16, num_plots - start_idx)
    plot_decay_curves(dec_curves, start_idx, plots_in_current_figure,fit=model_snel_popt)


#%% fitten aan simpel monoexponentieel verval om lifetimes te vergelijken met convolutie fit
def mono_exp(t_array,k,I):
    """Monoexponentieel verval zonder achtergrond"""
    return I*np.exp(-k*t_array) # + bg


mono_popt = np.zeros((len(objects),2))

for i, datai in enumerate(dec_curves):
    try:
        datai = np.roll(datai,-np.where(datai==np.max(datai))[0][0])
        mono_popt[i],_ = curve_fit(mono_exp, datat[:-1], datai[:-1],p0=[1/100,1_000_000],bounds=([0,0],[0.2,100_000_000]))
        #mono_popt[i],_ = curve_fit(mono_exp, datat[:-1], datai[:-1],sigma=np.sqrt(datai[:-1]))
    except RuntimeError as e:
        print(f"Fout bij fitten decaycurve {i}: {e}")
        continue


num_figures = (num_plots + 15) // 16  # Calculate the number of figures needed
for i in range(num_figures):
    start_idx = i * 16
    # Determine how many plots to draw for this figure
    plots_in_current_figure = min(16, num_plots - start_idx)
    plot_decay_curves(dec_curves, start_idx, plots_in_current_figure,fit=mono_popt)

plt.tight_layout()

#%%
plt.hist(1/mono_popt.T[0],range(0,100,10))
#%%
plt.hist(1/model_snel_popt.T[1],range(0,100,10))
#%%

def double_exp(t_array,k1,I1,k2,I2,bk):
    """Monoexponentieel verval zonder achtergrond"""
    return I1*np.exp(-k1*t_array) + I2*np.exp(-k2*t_array) + bk


double_popt = np.zeros((len(objects),5))

for i, datai in enumerate(dec_curves):
    try:
        datai = np.roll(datai,-np.where(datai==np.max(datai))[0][0])
        double_popt[i],_ = curve_fit(double_exp, datat[:-1], datai[:-1],sigma=np.sqrt(datai[:-1]),p0=[1/100,1_000_000,1/100,1_000_000,10000],bounds=([0,0,0,0,0],[0.2,100_000_000,0.2,100_000_000,np.inf]))
        #mono_popt[i],_ = curve_fit(mono_exp, datat[:-1], datai[:-1],sigma=np.sqrt(datai[:-1]))
    except RuntimeError as e:
        print(f"Fout bij fitten decaycurve {i}: {e}")
        continue


num_figures = (num_plots + 15) // 16  # Calculate the number of figures needed
for i in range(num_figures):
    start_idx = i * 16
    # Determine how many plots to draw for this figure
    plots_in_current_figure = min(16, num_plots - start_idx)
    plot_decay_curves(dec_curves, start_idx, plots_in_current_figure,fit=double_popt)

#%%
#combines
num_figures = (num_plots + 15) // 16  # Calculate the number of figures needed
for i in range(num_figures):
    start_idx = i * 16
    # Determine how many plots to draw for this figure
    plots_in_current_figure = min(16, num_plots - start_idx)
    plot_decay_curves(dec_curves, start_idx, plots_in_current_figure,fit=[model_snel_popt,mono_popt])



#%%
double_popt= np.sort(double_popt)

plt.figure() #decay rates gevonden volgens beide modellen met elkaar vergelijken 
plt.plot(1/model_snel_popt[:,1],1/double_popt[:,0],'.',c='royalblue',alpha = 0.5)
plt.plot(1/model_snel_popt[:,1],1/double_popt[:,1],'.',c='red',alpha=0.5)
plt.plot([5,45],[5,45],'black',alpha=0.5)
plt.xlabel('$\Gamma_{cov}$',size=15)
plt.ylabel('$\Gamma_{double}$',size=15)
plt.xlim([0,50])
plt.ylim([0,50])
#%%

plt.figure() #decay rates gevonden volgens beide modellen met elkaar vergelijken 
plt.plot(1/model_snel_popt[:,1],1/mono_popt[:,0],'.',c='royalblue',alpha = 0.5)
plt.plot([5,155],[5,155],'black',alpha=0.5)
plt.xlabel('$\Gamma_{cov}$',size=15)
plt.ylabel('$\Gamma_{mono}$',size=15)
plt.xlim([0,50])
plt.ylim([0,50])
#%%
plt.figure() 
plt.plot(model_snel_popt[:,1],'.',label='box_functie')
plt.plot(mono_popt[:,0],'.',label='mono_exponential')
plt.ylim([0,0.05])
plt.legend()

#%% plotten van een specifieke decaycurve
idx=1
plt.figure()
plt.plot(datat,dec_curves[idx],c='royalblue',label='measured')
plt.plot(datat, model_snel(datat, *model_snel_popt[idx]),'.',c='orangered',label='fit convolution')
plt.plot(np.roll(datat,-np.where(dec_curves[ idx] == np.max(dec_curves[ idx]))[0][0])[:-2],
             mono_exp(datat, *mono_popt[idx])[:-2],'.',c='green',label='mono exponential fit')
plt.xlabel('Time (ns)',size=15)
plt.ylabel('Intensity',size=15)
plt.yscale('log')


# %% histogram van gevonden lifetimes
lifetimes = (mono_popt[:,0])**(-1) #lifetime is inverse van 
bins_hist = np.arange(0,400,40)
plt.figure()
plt.hist(lifetimes,bins=bins_hist,rwidth=.9)
plt.xlabel('Lifetime (ns)')
plt.ylabel('Frequency')

# %%

# Assuming mono_popt is already defined
lifetimes = (mono_popt[:,0])**(-1)  # Lifetime is inverse

# Create logarithmic bins
bins_hist = np.logspace(np.log10(10), np.log10(1000), num=20)  # Create 20 bins between 10 and 400 on a log scale

plt.figure()
plt.hist(lifetimes, bins=bins_hist, rwidth=1)
plt.xscale('log')
plt.xlim([10,1000])
plt.xlabel('Lifetime (ns)')
plt.ylabel('Frequency')
plt.show()
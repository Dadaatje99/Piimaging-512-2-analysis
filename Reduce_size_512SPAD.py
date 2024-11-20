# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 08:35:00 2024

Code om 3D-array te maken van data, deze bij te snijden en metadata op te slaan
Hierbij kan je zelf het bereik invullen wat je wil zien
De png's worden op dit moment nog niet verwijderd dus dat moet handmatig gebeuren

David
+ Can run whole maps of data/intensity_images or data/gated_images and gives 
progress updates/warnings.
+ Code in functions

@author: Willem Vlasblom & David van Houten
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2 as cv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

def metadata(frame_file):
    """Retrieve metadata from a PNG file."""
    im = Image.open(frame_file)
    im.load()
    return im.info

def load_image(file_path):
    """Load a grayscale image from the given file path."""
    img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
    return img

def number_of_images(meta):
    """Calculate the total number of images based on metadata."""
    frames = meta.get('Frames', None)
    gate_steps = meta.get('Gate steps', None)
    if frames is None:
        raise ValueError("Metadata 'Frames' not found in the image")
    if gate_steps is not None:
        return int(gate_steps) * int(frames)
    else:
        return int(frames)

def process_folder(path_read, path_out, bereik_x, bereik_y):
    """Process all images in the given folder."""
    frames = os.listdir(path_read)
    frame_files = [os.path.join(path_read, x) for x in frames if x.endswith('.png')]

    if not frame_files:
        raise ValueError(f"No PNG files found in the directory: {path_read}")

    meta = metadata(frame_files[0])
    picture_amount = number_of_images(meta)
    
    if picture_amount > len(frame_files):
        print(f"Warning: Metadata expects {picture_amount} images, but only {len(frame_files)} files are available.")
        picture_amount = len(frame_files)

    # 3D array for all images
    movie_arr = np.zeros((picture_amount, 512, 256), dtype=np.uint16)

    # Number of available threads for reading images
    num_workers = os.cpu_count()

    # Load images in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {executor.submit(load_image, frame_files[i]): i for i in range(picture_amount)}

        # Process completed futures as they finish
        for idx, future in enumerate(as_completed(future_to_idx)):
            movie_arr[future_to_idx[future]] = future.result()
            if (idx + 1) % 200 == 0 or idx == picture_amount - 1:
                print(f"Processed {idx + 1}/{picture_amount} images in {os.path.basename(path_read)}")

    total_image = np.sum(movie_arr[:], axis=0)
    total_image_cut = total_image[bereik_x[0]:bereik_x[1], bereik_y[0]:bereik_y[1]]
    movie_arr_cut = movie_arr[:, bereik_x[0]:bereik_x[1], bereik_y[0]:bereik_y[1]]

    #plt.figure()
    #plt.imshow(total_image, cmap='gray')
    #plt.xlim(0, 512)
    #plt.ylim(0, 512)
    #plt.axhline(y=bereik_y[0], color='red', linestyle='-')
    #plt.axhline(y=bereik_y[1], color='red', linestyle='-')
    #plt.axvline(x=bereik_x[0], color='red', linestyle='-')
    #plt.axvline(x=bereik_x[1], color='red', linestyle='-')
    #plt.tight_layout()
    #plt.show()

    # Save the processed arrays and metadata
    np.save(os.path.join(path_out, f'movie_arr_{os.path.basename(path_read)}.npy'), movie_arr_cut)
    meta_json = json.dumps(meta)
    with open(os.path.join(path_out, f'meta_{os.path.basename(path_read)}.json'), 'w') as f:
        f.write(meta_json)

    print(f"Processing complete for {os.path.basename(path_read)}. Data saved to {path_out}")


# Specify the main directory path and parameters
main_directory = r'D:/Universiteit/5.1-6.2 Master Thesis/Experiments/24-10-22 gCdSe CdS decay curves/data/gated_images/' #end with /
path_out = main_directory  # Change if you want to save in a different folder
bereik_x = [128, 384]  # Range of x pixels to be saved (half-open interval)
bereik_y = [0, -1]  # Range of y pixels to be saved (half-open interval) 

# List all subdirectories in the main directory
subdirectories = [os.path.join(main_directory, d) for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d))]

# Process each subdirectory
for subdirectory in subdirectories:
    print(f"Starting processing for {os.path.basename(subdirectory)}...")
    try:
        process_folder(subdirectory, path_out, bereik_x, bereik_y)
        print(f"Finished processing for {os.path.basename(subdirectory)}.\n")
    except Exception as e:
        print(f"Error processing {os.path.basename(subdirectory)}: {e}\n")

# Load the saved metadata and array for verification (example for the first subdirectory)
first_subdirectory = os.path.basename(subdirectories[0])
with open(os.path.join(path_out, f'meta_{first_subdirectory}.json'), 'r') as data_json:
    meta2 = json.load(data_json)

movie_arr_out = np.load(os.path.join(path_out, f'movie_arr_{first_subdirectory}.npy'))

#%%
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2 as cv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

def metadata(frame_file):
    """Retrieve metadata from a PNG file."""
    im = Image.open(frame_file)
    im.load()
    return im.info

def load_image(file_path):
    """Load a 16-bit grayscale image from the given file path."""
    img = cv.imread(file_path, cv.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Image {file_path} could not be loaded.")
    return img

def number_of_images(meta):
    """Calculate the total number of images based on metadata."""
    frames = meta.get('Frames', None)
    gate_steps = meta.get('Gate steps', None)
    if frames is None:
        raise ValueError("Metadata 'Frames' not found in the image")
    if gate_steps is not None:
        return int(gate_steps) * int(frames)
    else:
        return int(frames)

def process_folder(path_read, path_out, bereik_x, bereik_y):
    """Process all images in the given folder."""
    frames = os.listdir(path_read)
    frame_files = [os.path.join(path_read, x) for x in frames if x.endswith('.png')]

    if not frame_files:
        raise ValueError(f"No PNG files found in the directory: {path_read}")

    meta = metadata(frame_files[0])
    picture_amount = number_of_images(meta)
    
    if picture_amount > len(frame_files):
        print(f"Warning: Metadata expects {picture_amount} images, but only {len(frame_files)} files are available.")
        picture_amount = len(frame_files)

    # 3D array for all images
    movie_arr = np.zeros((picture_amount, 512, 256), dtype=np.uint16)

    # Number of available threads for reading images
    num_workers = os.cpu_count()

    # Load images in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {executor.submit(load_image, frame_files[i]): i for i in range(picture_amount)}

        # Process completed futures as they finish
        for idx, future in enumerate(as_completed(future_to_idx)):
            movie_arr[future_to_idx[future]] = future.result()
            if (idx + 1) % 200 == 0 or idx == picture_amount - 1:
                print(f"Processed {idx + 1}/{picture_amount} images in {os.path.basename(path_read)}")

    total_image = np.sum(movie_arr[:], axis=0, dtype=np.uint32)  # Use uint32 for summation to avoid overflow
    total_image_cut = total_image[bereik_x[0]:bereik_x[1], bereik_y[0]:bereik_y[1]]
    movie_arr_cut = movie_arr[:, bereik_x[0]:bereik_x[1], bereik_y[0]:bereik_y[1]]

    # Save the processed arrays and metadata
    np.save(os.path.join(path_out, f'movie_arr_{os.path.basename(path_read)}.npy'), movie_arr_cut)
    meta_json = json.dumps(meta)
    with open(os.path.join(path_out, f'meta_{os.path.basename(path_read)}.json'), 'w') as f:
        f.write(meta_json)

    print(f"Processing complete for {os.path.basename(path_read)}. Data saved to {path_out}")


# Specify the main directory path and parameters
main_directory = r'D:/Universiteit/5.1-6.2 Master Thesis/Experiments/24-10-22 gCdSe CdS decay curves/data/gated_images/' #end with /
path_out = main_directory  # Change if you want to save in a different folder
bereik_x = [128, 384]  # Range of x pixels to be saved (half-open interval)
bereik_y = [0, -1]  # Range of y pixels to be saved (half-open interval) 

# List all subdirectories in the main directory
subdirectories = [os.path.join(main_directory, d) for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d))]

# Process each subdirectory
for subdirectory in subdirectories:
    print(f"Starting processing for {os.path.basename(subdirectory)}...")
    try:
        process_folder(subdirectory, path_out, bereik_x, bereik_y)
        print(f"Finished processing for {os.path.basename(subdirectory)}.\n")
    except Exception as e:
        print(f"Error processing {os.path.basename(subdirectory)}: {e}\n")

# Load the saved metadata and array for verification (example for the first subdirectory)
first_subdirectory = os.path.basename(subdirectories[0])
with open(os.path.join(path_out, f'meta_{first_subdirectory}.json'), 'r') as data_json:
    meta2 = json.load(data_json)

movie_arr_out = np.load(os.path.join(path_out, f'movie_arr_{first_subdirectory}.npy'))

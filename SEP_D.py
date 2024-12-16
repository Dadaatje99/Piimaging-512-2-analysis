"""
Created on Tue Sep 24 10:34:00 2024

@author: David van Houten
"""

from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

###############################################################################
#background correction code

def sigma_clipping(data, sigma):
    """Perform sigma clipping on the data until all values are within the specified sigma range."""
    iterations = 0
    while True:
        mean = np.mean(data)
        std = np.std(data)
        clipped_data = data[(data > mean - sigma * std) & (data < mean + sigma * std)]
        if len(clipped_data) == len(data):
            break
        data = clipped_data
        iterations += 1
    return data, iterations

def estimate_background(data, sigma):
    """Estimate the background using a combination of σ-clipping and mode estimation."""
    clipped_data, iterations = sigma_clipping(data, sigma)
    median = np.median(clipped_data)
    mean = np.mean(clipped_data)
    mode = 2.5 * median - 1.5 * mean
    return mode, iterations

def background_subtraction(image, mesh_size=32, sigma=2, background_map=False, plot_sigma_clipping=False):
    """Compute the background map for the image."""
    ny, nx = image.shape
    background = np.zeros_like(image)
    total_iterations = []

    # Estimate background in each mesh
    for i in range(0, nx, mesh_size):
        for j in range(0, ny, mesh_size):
            mesh = image[j:j+mesh_size, i:i+mesh_size]
            mode, iterations = estimate_background(mesh.flatten(), sigma)
            background[j:j+mesh_size, i:i+mesh_size] = mode
            total_iterations.append(iterations)
    
    # Interpolate between the meshes
    background = gaussian_filter(background, sigma=mesh_size)

    corrected_image = image - background
    corrected_image[corrected_image > 1e9] = 0
    if plot_sigma_clipping:
        plt.hist(total_iterations, bins=range(max(total_iterations)+1), edgecolor='black')
        plt.xlabel('Iterations')
        plt.ylabel('Frequency')
        plt.title('Sigma Clipping Iterations per Mesh')
        plt.show()
    
    if background_map:
        return corrected_image, background
    else:
        return corrected_image


###############################################################################
#Source extraction code

def threshold_image(image, sigma_factor):
    """Apply a sigma-based threshold to the image."""
    mean = np.mean(image)
    std = np.std(image)
    threshold = mean + sigma_factor * std
    return (image > threshold).astype(int)


def connected_components(binary_image):
    """Label connected components in the binary image."""
    labels = np.zeros_like(binary_image)
    label = 1
    for i in range(binary_image.shape[0]):
        for j in range(binary_image.shape[1]):
            if binary_image[i, j] == 1 and labels[i, j] == 0:
                flood_fill(binary_image, labels, i, j, label)
                label += 1
    return labels


def flood_fill(image, labels, x, y, label):
    """Recursively fill connected component with a unique label."""
    if x < 0 or x >= image.shape[0] or y < 0 or y >= image.shape[1]:
        return
    if image[x, y] == 0 or labels[x, y] != 0:
        return
    labels[x, y] = label
    flood_fill(image, labels, x + 1, y, label)
    flood_fill(image, labels, x - 1, y, label)
    flood_fill(image, labels, x, y + 1, label)
    flood_fill(image, labels, x, y - 1, label)


def calculate_centroid(labels, label):
    """Calculate the centroid of a labeled component."""
    indices = np.where(labels == label)
    if len(indices[0]) == 0:
        return None
    x_centroid = np.mean(indices[0])
    y_centroid = np.mean(indices[1])
    return x_centroid, y_centroid


def calculate_bounding_box(labels, label):
    """Calculate the bounding box of a labeled component."""
    indices = np.where(labels == label)
    if len(indices[0]) == 0:
        return None
    x_min = np.min(indices[0])
    x_max = np.max(indices[0])
    y_min = np.min(indices[1])
    y_max = np.max(indices[1])
    return x_min, x_max, y_min, y_max

def calculate_npix(labels,label):
    """Calculate the pixelsize of a labeled component."""
    indices = np.where(labels == label)
    return len(indices[0])

def calculate_mean_int(labels,label,image,len_arr):
    """Calculate the pixelsize of a labeled component."""
    mean = np.sum(image[labels == label])/len_arr
    return mean

###############################################################################
#branches splitting

def closest_branch(point, branches, intensity_map):
    """Find the closest branch to a given point based on the highest intensity."""
    neighbors = get_neighbors(point)
    max_intensity = -float('inf')
    closest = None
    
    for neighbor in neighbors:
        if is_within_bounds(neighbor, intensity_map.shape):
            neighbor_intensity = intensity_map[neighbor[0], neighbor[1]]
            for branch in branches:
                if neighbor in branch:
                    if neighbor_intensity > max_intensity:
                        max_intensity = neighbor_intensity
                        closest = branch
                    break
    return closest

def get_neighbors_branches(point):
    
    """Get neighboring points (n-connectivity) of the given point."""
    x, y = point
    return [                    
                    (x-1, y-1), (x, y-1), (x+1, y-1), 
                    (x-1, y  ),           (x+1, y  ), 
                    (x-1, y+1), (x, y+1), (x+1, y+1),                          
    ]

def get_neighbors(point):
    
    """Get neighboring points (n-connectivity) of the given point."""
    x, y = point
    return [                    
                    (x-1, y-1), (x, y-1), (x+1, y-1), 
                    (x-1, y  ),           (x+1, y  ), 
                    (x-1, y+1), (x, y+1), (x+1, y+1),                          
    ]

def is_within_bounds(point, shape):
    """Check if the point is within the bounds of the intensity map."""
    x, y = point
    return 0 <= x < shape[0] and 0 <= y < shape[1]

def add_to_branch(point, branches, intensity_map):
    """Add a point to the closest branch, or create a new branch if no close branch exists,
    with the condition that the new branch must have a peak height at least min_peak_ratio times
    the maximum peak height in the branches."""
    neighbors = get_neighbors_branches(point)
    neighboring_branches = set()

    for neighbor in neighbors:
        for branch in branches:
            if neighbor in branch:
                neighboring_branches.add(tuple(map(tuple, branch)))

    if len(neighboring_branches) > 1:
        # Point is connected to multiple branches, do not add to any branch
        return None

    closest = closest_branch(point, branches, intensity_map)
    if closest is None or not get_neighbors(point):
        # Create a new branch and check the peak height condition
        new_branch = [point]
        branches.append(new_branch)
    else:
        # Add to the closest branch
        closest.append(point)

def extract_branches(total_image, objects, segmap, cleaning=None):
    all_branches = []

    for i in objects:
        bbox = objects[i]['bbox']
        x = total_image[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]
        seg_region = segmap[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]  # Segmentation map for the current region
        x_copy = np.where(seg_region == i, x, 0)  # Masked image region

        local_branches = []  # Local branches for the current object region
        while np.max(x_copy) > 0:
            max_pos = np.where(x_copy == np.max(x_copy))
            row = max_pos[0][0] + bbox[0]
            col = max_pos[1][0] + bbox[2]
            point = (row, col)
            x_copy[row - bbox[0], col - bbox[2]] = 0
            add_to_branch(point, local_branches, total_image)
        
        if cleaning:
            local_branches = [branch for branch in local_branches if len(branch) > cleaning]
        all_branches.append(local_branches)
        

    return all_branches

def update_objects(total_image, objects, branches):
    """Update the objects based on the new segmentation map and branches."""
    updated_objects = {}
    label = 1
    for obj_branches in branches:
        for branch in obj_branches:
            branch_points = np.array(branch)
            branch_labels = np.zeros_like(total_image)
            
            branch_labels[branch_points[:, 0], branch_points[:, 1]] = label
            
            centroid = calculate_centroid(branch_labels, label)
            bbox = calculate_bounding_box(branch_labels, label)
            npix = calculate_npix(branch_labels, label)
            
            updated_objects[label] = {'x': centroid[0], 'y': centroid[1], 'bbox': bbox, 'npix': npix}
            label += 1
    return updated_objects

def update_segmentation_map(segmap, objects, all_branches):
    """Update the segmentation map with branches, considering new objects while preserving zeros."""
    updated_segmap = np.zeros_like(segmap)  # Start with a copy of the original segmentation map
    current_label = 0 + 1  # Start new labels after the maximum existing label

    for obj_id, branches in zip(objects.keys(), all_branches):
        bbox = objects[obj_id]['bbox']
        y_min, y_max, x_min, x_max = bbox

        for branch in branches:
            new_branch_label = current_label
            for point in branch:
                row, col = point
                full_image_row = row
                full_image_col = col
                updated_segmap[full_image_row, full_image_col] = new_branch_label

            # Move to the next label for the next branch
            current_label += 1

    return updated_segmap


###############################################################################
#Objects extraction

def extract_objects(movie_arr, threshold, deblending=None, filters=None, cleaning=None):
    """Detect and extract objects from the image."""
    # Step 1: Initial detection
    image = np.sum(movie_arr, axis=0)
    binary_image = threshold_image(image, threshold)
    segmap = connected_components(binary_image)
    objects = {}
    for label in range(1, np.max(segmap) + 1):
        centroid = calculate_centroid(segmap, label)
        bbox = calculate_bounding_box(segmap, label)
        npix = calculate_npix(segmap, label)
        mean_int = calculate_mean_int(segmap, label,image,movie_arr.shape[0])
        
        objects[label] = {'x': centroid[0], 'y': centroid[1], 'bbox': bbox, 
                          'npix': npix,'mean_int': mean_int}

    # Step 2: Deblending (if specified)
    if deblending:
        all_branches = extract_branches(image, objects, segmap, cleaning)
        updated_objects = update_objects(image, objects, all_branches)
        updated_segmap = update_segmentation_map(segmap, objects, all_branches)
    else:
        updated_objects = objects
        updated_segmap = segmap
        all_branches = None  # No branches if deblending isn't applied

    # Step 3: Apply filtering (after deblending)
    if filters:
        filtered_objects = updated_objects
        for f in filters:
            key = f['key']
            lower = f.get('lower', float('-inf'))
            upper = f.get('upper', float('inf'))
            filtered_objects = {k: v for k, v in filtered_objects.items() if lower <= v[key] <= upper}
        
        # Update the segmentation map based on filtered objects
        filtered_segmap = updated_segmap.copy()
        for obj_id in updated_objects.keys():
            if obj_id not in filtered_objects:
                filtered_segmap[filtered_segmap == obj_id] = 0  # Assuming 0 is the background value
    else:
        filtered_objects = updated_objects
        filtered_segmap = updated_segmap

    # Return outputs
    if deblending:
        return filtered_objects, filtered_segmap, [objects, all_branches, segmap]
    
    return filtered_objects, filtered_segmap



from matplotlib import font_manager
font_path = r'C:/Windows/Fonts/framd.ttf'  # Update this with the path to your font file
font_prop = font_manager.FontProperties(fname=font_path)
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import ScalarFormatter
###############################################################################
#Objects plotting

def plot_objects(frames, objects,pix_sizes = 13,pix_sum = 1,mag = 100, radius=2):
    
    pix_size = pix_sizes / mag * pix_sum
    fig, ax = plt.subplots(figsize=(8/2,5/2))
    # ax.imshow(frames, origin='lower', cmap='jet', vmin=0, vmax=np.max(frames),
    #           extent=[0, frames.shape[1] * pix_size, 0, frames.shape[0] * pix_size])
    ax.imshow(frames, origin='lower', cmap='jet', vmin=0, vmax=np.max(frames))
    
    for idx, key in enumerate(objects):
        # x = objects[key]['x'] * pix_size
        # y = objects[key]['y'] * pix_size
        x = objects[key]['x']
        y = objects[key]['y']

        circle = Circle((y, x), radius, edgecolor='yellow', facecolor='none', lw=0.5)
        ax.add_artist(circle)
    
    ax.set_xlabel('Pixels', fontproperties=font_prop)
    ax.set_ylabel('Pixels', fontproperties=font_prop)
    ax.xaxis.set_major_locator(MultipleLocator(100)) 
    ax.xaxis.set_minor_locator(MultipleLocator(50)) 
    ax.tick_params(axis='x', which='major', labelsize=10)
    ax.tick_params(axis='x', which='minor',  labelsize=8)
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_minor_locator(MultipleLocator(50))  
    ax.tick_params(axis='y', which='major', labelsize=10)
    ax.tick_params(axis='y', which='minor',  labelsize=8)
    plt.show()


###############################################################################
#branch plotting per pixel

def extract_branches2(total_image, objects, segmap):
    all_branches = []
    assignment_log = []

    for obj_id, obj_info in objects.items():
        bbox = obj_info['bbox']
        x = total_image[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]
        seg_region = segmap[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]  # Segmentation map for the current region
        x_copy = np.where(seg_region > 0, x, 0)  # Masked image region

        local_branches = []  # Local branches for the current object region
        while np.max(x_copy) > 0:
            max_pos = np.where(x_copy == np.max(x_copy))
            row = max_pos[0][0] + bbox[0]
            col = max_pos[1][0] + bbox[2]
            point = (row, col)
            x_copy[row - bbox[0], col - bbox[2]] = 0
            add_to_branch(point, local_branches,total_image)
            
            # Log the assignment
            assignment_log.append((row, col, obj_id, len(local_branches)))

        all_branches.append(local_branches)

    return all_branches, assignment_log


def plot_pixel_order(total_image, objects, old_branches, step=5):
    all_branches, assignment_log = extract_branches2(total_image, objects, old_branches)
    # Ensure all_branches is correctly indexed
    all_branches_dict = {i: branches for i, branches in zip(objects.keys(), all_branches)}
    max_int = np.max(total_image)*0.5

    # Dictionary to accumulate points
    accumulated_points = {obj_id: [] for obj_id in objects.keys()}
    
    # Plot the branch assignments over time
    frame=0
    for row, col, obj_id, branch_idx in assignment_log:
        frame += 1

            
        bbox = objects[obj_id]['bbox']
        x = total_image[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]

        # Add the new point to the accumulated points
        accumulated_points[obj_id].append((row, col))
        if frame%step:
            continue
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the sub-region image
        axs[0].imshow(x, cmap='jet', vmax=max_int)
        axs[0].set_title(f'Object {obj_id + 1} - Image')
        axs[0].axis('off')

        # Plot the branches
        axs[1].imshow(x, cmap='jet', vmax=max_int)  # Background image for reference

        for branch in all_branches_dict[obj_id]:
            branch = np.array(branch)
            axs[1].plot(branch[:, 1] - bbox[2], branch[:, 0] - bbox[0],'.')
        
        # Plot all accumulated points
        for point in accumulated_points[obj_id]:
            axs[1].plot(point[1] - bbox[2], point[0] - bbox[0], '.',color='red', marker='x')
        
        axs[1].axis('off')

        plt.show()


###############################################################################
#branch plotting      



def plot_branches(total_image, objects, all_branches):
    # Ensure all_branches is correctly indexed
    all_branches_dict = {i: branches for i, branches in zip(objects.keys(), all_branches)}
    max_int = np.max(total_image) * 0.8
    idx = -1
    for obj_idx, obj_key in enumerate(objects):
        obj = objects[obj_key]
        # Extract the region for the current object
        x = total_image[obj['bbox'][0]:obj['bbox'][1] + 1, obj['bbox'][2]:obj['bbox'][3] + 1]
        # Retrieve the branches for the current object
        branches = all_branches_dict.get(obj_key, [])

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the sub-region image with colorbar
        im = axs[0].imshow(x, cmap='jet', vmax=max_int)
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        axs[0].axis('off')

        # Plot the branches on the second subplot
        axs[1].imshow(x, cmap='jet', vmax=max_int)
        colors = plt.cm.get_cmap('gist_ncar')
        branch_colors = colors(np.linspace(0.15, 0.9, len(branches)))
        
        # Title text initialization

        y_offset = 0.0  # Slightly above the plot
        x_offset = 0.1  # Slightly right of the plot edge
        
        for branch_idx, branch in enumerate(branches):
            idx = idx +1
            if not branch:
                continue
            branch = np.array(branch)
            axs[1].plot(branch[:, 1] - obj['bbox'][2], branch[:, 0] - obj['bbox'][0],
                        '.', color=branch_colors[branch_idx], markersize=5)
            plt.figtext(x_offset, y_offset, f'Objects {idx}', fontsize=12, color=branch_colors[branch_idx])
            x_offset += 0.1  # Adjust for the next branch annotation
        
        
        
        # Display title with custom colors

        axs[1].axis('off')
        # Add custom colored annotations for each branch



        plt.tight_layout()
        plt.show()

###############################################################################
#intensity traces     

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


from matplotlib.colors import Normalize, LinearSegmentedColormap
def plot_segmap(image, threshold=1, below_color='lightblue', above_color='darkblue'):
    # Create a colormap with two colors
    colors = [below_color, above_color]
    cmap = LinearSegmentedColormap.from_list('threshold_cmap', colors, N=2)
    
    # Normalize the data with a specified threshold
    norm = Normalize(vmin=threshold, vmax=threshold + 1)
    
    # Plot the image with the custom colormap and normalization
    plt.imshow(image, cmap=cmap, norm=norm,origin='lower')

    plt.show()
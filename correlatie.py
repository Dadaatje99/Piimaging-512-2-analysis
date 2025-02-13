# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:41:01 2024

@author: dadav
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d

def intensity_traces(movie_array, segmentation_map, objects, background_map=None):
    """Computes the intensity traces of detected quantum dots."""
    pixel_traces = {}
    int_traces = np.zeros((len(objects), movie_array.shape[0]))
    
    if background_map is not None:
        norm_bkg_map = background_map / np.max(background_map)
        for i, key in enumerate(objects):
            loc = np.array(np.where(segmentation_map == key)).T
            for x, y in loc:
                intensity_trace = movie_array[:, x, y] - (background_map[x, y] / movie_array.shape[0])
                int_traces[i] += intensity_trace
                if key not in pixel_traces:
                    pixel_traces[key] = []
                pixel_traces[key].append((intensity_trace, x, y))
    else:
        for i, key in enumerate(objects):
            loc = np.array(np.where(segmentation_map == key)).T
            for x, y in loc:
                intensity_trace = movie_array[:, x, y]
                int_traces[i] += intensity_trace
                if key not in pixel_traces:
                    pixel_traces[key] = []
                pixel_traces[key].append((intensity_trace, x, y))
    
    # Convert the pixel traces list to a numpy array for each object
    for key in pixel_traces:
        pixel_traces[key] = np.array(pixel_traces[key], dtype=object)
    
    return int_traces, pixel_traces

# Example usage:
frame_intensity1, pixel_traces = intensity_traces(movie_arr_cut, segmentation_map, objects)

#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import distance_matrix
from scipy.ndimage import uniform_filter1d
from scipy.stats import zscore

def correlation_matrix_ncc(traces):
    # Extract the intensity traces over time
    intensity_traces = np.array([trace[0] for trace in traces])
    n = len(intensity_traces)
    corr_matrix = np.zeros((n, n))
    
    
    for i in range(n):
        for j in range(i, n):
            f = intensity_traces[i]
            t = intensity_traces[j]
            #print(f.shape)
            f2 = np.zeros(int(f.shape[0]/10))
            t2 = np.zeros(int(f.shape[0]/10))
            for x in range(10):
                f2 += f[x::10]
                t2 += t[x::10]
            
            # Compute the dot product
            corr =  np.dot(f2,t2)/(np.sqrt(np.sum(f2**2))*np.sqrt(np.sum(t2**2)))
            corr = np.sum((f2 - np.mean(f2)) * (t2 - np.mean(t2))) / (np.std(f2) * np.std(t2) * len(f2))

            #print(f2[0:10],t2[0:10],corr)
            
            # Fill the matrix (symmetric)
            corr_matrix[i, j] = corr_matrix[j, i] = corr
    
    return corr_matrix





traces = pixel_traces[1]
intensity_traces = np.array([trace[0] for trace in traces])
                
                
        # Compute the original correlation matrix
corr_mat = correlation_matrix_ncc(traces)
print(corr_mat)
        
sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap='coolwarm', cbar=False, xticklabels=False,yticklabels=False)

#%%

all_traces = []
all_loc = []
correlation = []
for key in objects:
    traces = pixel_traces[key]
    
    if len(traces) > 1:
        # Extract and normalize intensity traces
        intensity_traces = np.array([trace[0] for trace in traces])
                
                
        # Compute the original correlation matrix
        corr_mat = correlation_matrix_ncc(traces)
        
        norm_traces = zscore(intensity_traces, axis=1)

        # Set the threshold for correlation
        threshold = 0.9
    
        # Only consider the lower triangle of the matrix (excluding the diagonal)
        lower_triangle = np.tril(corr_mat, k=-1)

    
        # Extract the positions where correlation is above the threshold
        high_corr_positions = np.argwhere(lower_triangle >= threshold)
    
        # Sort the high correlation positions by the correlation values
        sorted_corr_positions = sorted(high_corr_positions, key=lambda pos: lower_triangle[pos[0], pos[1]], reverse=True)
    
        # Initialize branches with the first correlated pairs
        branches = []
        pixel_to_branch = {}  # To track which branch a pixel belongs to
    
        def average_correlation(branch, new_pixel):
            """Calculate the average correlation of a new pixel with an existing branch."""
            correlations = [corr_mat[new_pixel, pixel] for pixel in branch]
            return np.mean(correlations)
    
        # Check each pair of correlated positions
        for i, j in sorted_corr_positions:
            # Try to find if i or j is already in any branch
            branch_i = pixel_to_branch.get(i)
            branch_j = pixel_to_branch.get(j)
    
            if branch_i is not None and branch_j is not None:
                # Both pixels are already in different branches, merge them
                if branch_i != branch_j:
                    # Merge the two branches
                    branches[branch_i].extend(branches[branch_j])
                    for pixel in branches[branch_j]:
                        pixel_to_branch[pixel] = branch_i
                    branches[branch_j] = []  # Clear the merged branch
            elif branch_i is not None:
                # Pixel i is already in a branch, check if j can be added to the same branch
                if average_correlation(branches[branch_i], j) >= threshold:
                    branches[branch_i].append(j)
                    pixel_to_branch[j] = branch_i
            elif branch_j is not None:
                # Pixel j is already in a branch, check if i can be added to the same branch
                if average_correlation(branches[branch_j], i) >= threshold:
                    branches[branch_j].append(i)
                    pixel_to_branch[i] = branch_j
            else:
                # Both pixels are in new branches, create a new branch
                new_branch_index = len(branches)
                branches.append([i, j])
                pixel_to_branch[i] = new_branch_index
                pixel_to_branch[j] = new_branch_index
    
        # Filter out empty branches
        branches = [branch for branch in branches if len(branch) > 0]
    
    
        # Flatten the branches into a single list of indices
        branch_indices = []
        for branch in branches:
            branch_indices.extend(branch)
        
        # Identify pixels that don't belong to any branch (if any)
        all_indices = set(range(corr_mat.shape[0]))
        branch_indices_set = set(branch_indices)
        remaining_indices = list(all_indices - branch_indices_set)
        # Combine branch indices with the remaining indices (those not in any branch)
        final_order = branch_indices + remaining_indices
        
        # Get the bbox for the object
        bbox = objects[key]['bbox']  # Example: bbox = [y_min, y_max, x_min, x_max]
        x_min, x_max, y_min, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
        
        # Extract the region of interest (ROI) from the total image using the bbox
        roi = total_image_cut[x_min:x_max + 1, y_min:y_max + 1]
        reorganized_corr_mat = corr_mat[final_order, :][:, final_order]
        traces1 = np.array([list(traces[i][0]) for i in range(len(traces))])
        loc = np.array([[traces[i][1],traces[i][2]] for i in range(len(traces))])
        coords = np.array([[trace[1], trace[2]] for trace in traces])
        dist_mat = distance_matrix(coords, coords)[final_order, :][:, final_order]
        norm_intensities = np.array([np.sum(trace[0]) for trace in traces]) / np.max(np.array([np.sum(trace[0]) for trace in traces]))
        norm_intensities = norm_intensities[final_order]
        
        # Determine the group correlation matrix, distance matrix, and normalized intensities based on branches
        if len(branches) > 10:
            concatenated_branches = np.concatenate(branches)
            group_corr_mat = reorganized_corr_mat[:len(concatenated_branches), :len(concatenated_branches)]
            group_dist_mat = dist_mat[:len(concatenated_branches), :len(concatenated_branches)]
            group_norm_intensities = norm_intensities[:len(concatenated_branches)]
        else:
            group_corr_mat = reorganized_corr_mat
            group_dist_mat = dist_mat
            group_norm_intensities = norm_intensities
        
        # Initialize the list to hold masks
        masks = []
        
        # Iterate over each group of branches to plot and analyze
        for idx, group in enumerate(branches):
            # Create the subplots
            fig, axs = plt.subplots(2, 2, figsize=(8, 8))  # 2x2 grid of subplots
        
            # Reorganize the correlation matrix (reorganized_corr_mat2 should be defined somewhere)
            #sns.heatmap(reorganized_corr_mat, cmap='coolwarm', ax=axs[0, 1], cbar=False, vmin=0)
            #axs[0, 1].set_title('Reorganized Correlation Matrix')
            group_corr_mat2 = np.copy(group_corr_mat)
            group_dist_mat2 = np.copy(group_dist_mat)
            # Create the combined matrix
            combined_matrix = np.zeros_like(group_corr_mat2)
            combined_matrix[np.tril_indices_from(combined_matrix, k=-1)] = group_corr_mat2[np.tril_indices_from(group_corr_mat2, k=-1)]
            combined_matrix[np.triu_indices_from(combined_matrix, k=1)] = group_dist_mat2[np.triu_indices_from(group_dist_mat2, k=1)]
            np.fill_diagonal(combined_matrix, group_norm_intensities)
        
            # Plotting the combined matrix
            mask_upper = np.triu(np.ones_like(combined_matrix, dtype=bool), k=1)
            mask_lower = np.tril(np.ones_like(combined_matrix, dtype=bool), k=-1)
            mask_diag = np.eye(combined_matrix.shape[0], dtype=bool)
        
            sns.heatmap(combined_matrix, cmap='coolwarm', mask=mask_upper, ax=axs[0, 1], cbar=False, xticklabels=False,yticklabels=False,vmin=0,vmax=1)
            sns.heatmap(combined_matrix, cmap='viridis', mask=mask_lower, ax=axs[0, 1], cbar=False, xticklabels=False,yticklabels=False)
            sns.heatmap(combined_matrix, cmap='jet', mask=~mask_diag, ax=axs[0, 1], cbar=False, xticklabels=False,yticklabels=False)
        
            # Plot the segmentation map (ROI)
            axs[0, 0].imshow(roi.T, origin='lower', cmap='jet', alpha=0.7)
            axs[0, 0].set_title(f'Segmentation Map for Group {idx + 1}')

            # Plot branches on the segmentation map
            for branch_index, branch in enumerate(branches):
                colors = plt.cm.get_cmap('nipy_spectral')
                colors = colors(np.linspace(0.1, .9, len(branches)))
                branch_x_coords = [coords[i][0] - x_min for i in branch]
                branch_y_coords = [coords[i][1] - y_min for i in branch]
                axs[0, 0].scatter(branch_x_coords, branch_y_coords, color=colors[branch_index], s=15)
        
        
            # Initialize masks for the first iteration
            if idx == 0:
                for sublist in branches:
                    mask = [False] * reorganized_corr_mat.shape[0]
                    for index in sublist:
                        mask[index] = True
                    masks.append(mask)
        
            # Plotting the traces for each branch
            x = np.sum(traces1[masks[idx]], axis=0)
            final_loc=loc[masks[idx]]
            all_loc.append(final_loc)
            all_traces.append(x)
            axs[1, 0].plot(x, color=colors[idx])

            # Create the combined matrix
            group_traces = norm_traces[group, :]  # Extract the traces for this group
            group_corr_mat1 = np.corrcoef(group_traces)  # Correlation matrix for the group
            coords = np.array([[trace[1], trace[2]] for trace in traces])
            dist_mat = distance_matrix(coords, coords)
            norm_intensities = np.array([np.sum(trace[0]) for trace in traces]) / np.max(np.array([np.sum(trace[0]) for trace in traces]))

            combined_matrix = np.zeros_like(group_corr_mat1)
            combined_matrix[np.tril_indices_from(combined_matrix, k=-1)] = group_corr_mat1[np.tril_indices_from(group_corr_mat1, k=-1)]
            combined_matrix[np.triu_indices_from(combined_matrix, k=1)] = dist_mat[np.triu_indices_from(dist_mat[group, :][:, group], k=1)]
            np.fill_diagonal(combined_matrix, norm_intensities[group])

        
            # Plotting the combined matrix
            mask_upper = np.triu(np.ones_like(combined_matrix, dtype=bool), k=1)
            mask_lower = np.tril(np.ones_like(combined_matrix, dtype=bool), k=-1)
            mask_diag = np.eye(combined_matrix.shape[0], dtype=bool)
        
            sns.heatmap(combined_matrix, annot=True, fmt=".2f", cmap='coolwarm', mask=mask_upper, ax=axs[1, 1], cbar=False,vmin=0,vmax=1, xticklabels=False,yticklabels=False)
            sns.heatmap(combined_matrix, annot=True, fmt=".2f", cmap='viridis', mask=mask_lower, ax=axs[1, 1], cbar=False,vmin=1,vmax=np.max(group_dist_mat), xticklabels=False,yticklabels=False)
            sns.heatmap(combined_matrix, annot=True, fmt=".2f", cmap='jet', mask=~mask_diag, ax=axs[1, 1], cbar=False,vmin=0,vmax=1, xticklabels=False,yticklabels=False)
        
            # Display the combined plot
            plt.tight_layout()
            plt.show()

#%%
a_all_branches = [all_loc]
objects = update_objects(total_image_cut, objects, a_all_branches)
segmentation_map = update_segmentation_map(segmentation_map, objects, a_all_branches)
#%%
plot_segmap(segmentation_map)    


#%%
frame_intensity1 = np.array(all_traces)
#%%


traces = pixel_traces[6]
    

traces_int = np.array([gaussian_filter1d(uniform_filter1d(trace[0],15), 5) for trace in traces])
traces_int1 = np.array([trace[0]for trace in traces])    
plt.plot(traces_int1[0][:1000],alpha=0.3)
plt.plot(traces_int[0][:1000],alpha=0.3)

#%%

traces_int = np.array([gaussian_filter1d(uniform_filter1d(trace[0],15), 5) for trace in traces])
traces_int1 = np.array([trace[0]for trace in traces])    
plt.plot(traces_int1[1][:1000],alpha=0.3)
plt.plot(traces_int[1][:1000],alpha=0.3)



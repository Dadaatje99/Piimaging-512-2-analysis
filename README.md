# Introduction
For my master’s thesis, I worked with a SPAD512² detector from Pi-Imaging and developed code to process its data more efficiently. The goal was to make the data easier to transfer and reduce its size, while also allowing users to crop the data into smaller, custom shapes if needed. 

The original output of the detector is a large number of images saved in a single folder. Transferring this many files to another computer can be very time-consuming, as a single measurement can amount to several gigabytes of data, depending on the settings. To address this, I created the script `Reduce_size_512SPAD.py`. This script takes the original data as input and generates two output files: `meta_acqxxxxx.json` and `movie_arr_acqxxxxx.npy`.

- The **meta file** (`meta_acqxxxxx.json`) contains all the key information about the measurement. It can be opened with any basic notepad application.  
- The **movie file** (`movie_arr_acqxxxxx.npy`) contains the data in a compact 3D array format.

By reducing the total size of the data (if cropping is applied) and reduce it into just two files, the script makes data transfer significantly faster.

To access and work with the files mentioned above, I created the script `512^2_####.py`, which can read the files in what I found to be the fastest way possible.
The script outputs a 3D array along with all the 'important' settings already imported. It’s designed to be easily expandable, so you can add more information if needed. You only need to specify the location of the data and indicate whether it was an intensity or gated measurement.

In my data analysis work, I developed a custom particle detection script called `SEP_D[adaatje99].py`, inspired by the Source Extraction and Photometry (SEP) method used for astronomical object detection and analysis [1]. While there already exists a Python package based on this method [2], I decided to create my own version of the code, tailored specifically for detecting nanoparticles rather than stars and galaxies. The aim was to enhance my understanding of particle detection processes and develop a more specialized tool for nanoparticle analysis. As part of this effort, I have made the script freely available for anyone who wishes to use it.
The primary functions of `SEP_D.py` include background subtraction, object detection, deblending of overlapping particle clusters, and visualizing the detected objects.
Additionally, for those who want to perform analyses such as creating intensity traces, generating fluorescence decay traces, fitting decay traces, and examining results, the rest of the `512^2_####.py` script can be used.

Please note that this project is still in the early stages and is a work in progress.
[The code is written during the 1.51 version of the SPAD512² software and used for 8-bit intensity and 8-bit gated data]


## `Reduce_size_512SPAD.py`
This code should be installed on the computer used for performing the measurements.

### How to Use the Code
1. Ensure the code is installed on the measurement computer.  
2. To run the code, simply provide the correct path to the directory where the data is stored.  
3. Always use the **single measurement** option to ensure that all images are placed into one folder.  

Multiple measurements can be performed, and the data will be automatically organized as follows:
```
project-directory/
├── data/
│   ├── intensity_images/
│   │   ├── acq00000
│   │   └── acq00001
│   └── gated_images/
│       ├── acq00000
│       └── acq00001
```
To process all acquisitions at once, set the path to:  `path= r'project-directory/data/intensity_images/' ` All output files will be stored in the same folder, using the following naming conventions: `meta_acqxxxxx.json` & `movie_arr_acqxxxxx.npy`

## `512^2_####.py`
This code should be installed on the computer used for performing data analysis.

### How to Use the Code
1. Ensure the code is installed on the computer.
2. Provide the Data Path: Set the correct path to the directory where the data is stored. For example: `path= r'project-directory/data/intensity_images/' `
3. Run the code and specify the index you wish to access.
4. Now we get the following 4 in return:
   - **meta_raw** contains the raw metadata information.
   - **meta** contains the 'most important' values extracted from the metadata.
   - **movie_arr**  is a 3D array of shape (nframes, xsize, ysize) containing all the data.
   - **image** is a 2D array of shape (xsize, ysize) representing the sum of the frames in the 3D array.

You can write your own code to further analyze the data from here.
To create intensity traces or fluorescence decay traces, the rest of the code can be used.

## `SEP_D.py`
`SEP_D.py` is a standalone script designed for object detection on various types of measurements. It has been successfully tested on 2D images from different devices, including:
- Single-Photon Avalanche Diode array (SPAD512)
- Electron-Multiplying Charge-Coupled Device (EMCCD)
- Atomic Force Microscopy (AFM)
- Transmission Electron Microscopy (TEM)

The script can detect multiple objects and determine the pixels belonging to each object. It is based on the Source Extraction and Photometry (SEP) framework, with several enhancements such as plotting closeups of the objects and different systems for object detection (currently, only one system is implemented).

`SEP_D.py` is neccesary to create create intensity traces and fluorescence decay traces in the `512^2_####.py` script.

### How to Install the Code
1. Ensure the code is installed on the computer.
2. Import the `sys` module.

    ```python
    import sys
    ```

3. Add the directory containing `SEP_D.py` to the system path.
   Replace the path with the actual path to your `SEP_D.py` file.

    ```python
    # Add the directory containing SEP_D.py
    sys.path.append('D:/project-directory/SEP_D_map/')  # Replace with your actual path
    ```

4. Import all functions from `SEP_D.py`.

    ```python
    from SEP_D import *  # Import all functions
    ```
### How to Use the Code
The rest of the `512^2_####.py` script shows the basic usage.



### references
> [1] (1996). SExtractor: Software for source extraction. Astronomy and astrophysics supplement series, 117(2), 393-404.
> [2] https://github.com/kbarbary/sep

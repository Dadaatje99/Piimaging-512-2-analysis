For my master’s thesis, I worked with a SPAD512² detector from Pi-Imaging and developed code to process its data more efficiently. The goal was to make the data easier to transfer and reduce its size, while also allowing users to crop the data into smaller, custom shapes if needed. 

The original output of the detector is a large number of images saved in a single folder. Transferring this many files to another computer can be very time-consuming, as a single measurement can amount to several gigabytes of data, depending on the settings. To address this, I created the script `Reduce_size_512SPAD.py`. This script takes the original data as input and generates two output files: `meta_acqxxxxx.json` and `movie_arr_acqxxxxx.npy`.

- The **meta file** (`meta_acqxxxxx.json`) contains all the key information about the measurement. It can be opened with any basic notepad application.  
- The **movie file** (`movie_arr_acqxxxxx.npy`) contains the data in a compact 3D array format.

By reducing the total size of the data (if cropping is applied) and reduce it into just two files, the script makes data transfer significantly faster.

To access and work with the files mentioned above, I created the script `512^2_####.py`, which can read the files in what I found to be the fastest way possible.
The script outputs a 3D array along with all the 'important' settings already imported. It’s designed to be easily expandable, so you can add more information if needed. You only need to specify the location of the data and indicate whether it was an intensity or gated measurement.

For my data analysis, I developed a custom particle detection script called SEP_D[adaatje99].py, inspired by the Source Extraction and Photometry method [CITE]. I created this script to deepen my understanding of particle detection processes, and it is freely available for use.
The primary functions of SEP_D.py include background subtraction, object detection, deblending particle clusters, and visualizing detected objects.
Additionally, for those who want to perform analyses such as creating intensity traces, generating fluorescence decay traces, fitting decay traces, and examining results, the rest of the 512^2_####.py script can be used.

Please note that this project is still in the early stages and is a work in progress.
[The code is written during the 1.51 version of the SPAD512² software and used for 8-bit intensity and 8-bit gated data]

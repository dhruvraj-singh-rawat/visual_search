import os
import numpy as np
import cv2
import scipy.io as sio
from extractRandom import extractRandom
from cvpr_computehist import calculate_histogram_bins

# DATASET_FOLDER = 'MSRC_ObjCategImageDatabase_v2'
# OUT_FOLDER = 'descriptors'
# OUT_SUBFOLDER = 'globalRGBhisto'
# Q = 4 # Number of Quantization 

def create_global_color_hist(Q,DATASET_FOLDER,OUT_FOLDER,OUT_SUBFOLDER):
    # Ensure the output directory exists
    os.makedirs(os.path.join(OUT_FOLDER, OUT_SUBFOLDER), exist_ok=True)

    # Iterate through all BMP files in the dataset folder
    for filename in os.listdir(os.path.join(DATASET_FOLDER, 'Images')):
        if filename.endswith(".bmp"):
            #print(f"Processing file {filename}")
            img_path = os.path.join(DATASET_FOLDER, 'Images', filename)
            img = cv2.imread(img_path).astype(np.float64) / 255.0  # Normalize the image
            fout = os.path.join(OUT_FOLDER, OUT_SUBFOLDER, filename.replace('.bmp', '.mat'))
            
            # Call extractRandom (or another feature extraction function) to get the descriptor
            F = calculate_histogram_bins(img,Q) ##  Implemented Histogram Bins 
            
            # Save the descriptor to a .mat file
            sio.savemat(fout, {'F': F})

    print("Successfully Created Global Color Descriptors")

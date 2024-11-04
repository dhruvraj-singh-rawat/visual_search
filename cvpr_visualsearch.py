import os
import numpy as np
import scipy.io as sio
import cv2

# DESCRIPTOR_FOLDER = 'descriptors'
# DESCRIPTOR_SUBFOLDER = 'globalRGBhisto'
# IMAGE_FOLDER = 'MSRC_ObjCategImageDatabase_v2'

def load_descriptors(IMAGE_FOLDER,DESCRIPTOR_FOLDER,DESCRIPTOR_SUBFOLDER):
    # Load all descriptors
    ALLFEAT = []
    ALLFILES = []
    for filename in os.listdir(os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER)):
        if filename.endswith('.mat'):
            img_path = os.path.join(DESCRIPTOR_FOLDER, DESCRIPTOR_SUBFOLDER, filename)
            img_actual_path = os.path.join(IMAGE_FOLDER,'Images',filename).replace(".mat",".bmp")
            # ipdb.set_trace()
            img_data = sio.loadmat(img_path)
            ALLFILES.append(img_actual_path)
            ALLFEAT.append(img_data['F'][0])  # Assuming F is a 1D array

    # Convert ALLFEAT to a numpy array
    ALLFEAT = np.array(ALLFEAT)

    return ALLFILES,ALLFEAT

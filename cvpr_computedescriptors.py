import os
import numpy as np
import cv2
import scipy.io as sio


def colour_histogram(image, Q):
    binned_image = np.clip(np.floor(Q * image).astype(int), 0, Q - 1)

    # Split the channels
    red_channel = binned_image[:, :, 0]
    green_channel = binned_image[:, :, 1]
    blue_channel = binned_image[:, :, 2]

    # Calculate the histogram for each channel, with a bin count of Q
    red_hist = np.bincount(red_channel.flatten(), minlength=Q)
    green_hist = np.bincount(green_channel.flatten(), minlength=Q)
    blue_hist = np.bincount(blue_channel.flatten(), minlength=Q)

    overall_bin = red_hist * (Q**2) + green_hist * Q + blue_hist
    return overall_bin

def sobel_quantization(image, num_bins=8):
    # Apply Sobel filter to get gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x-direction
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y-direction

    # Calculate magnitude and angle of gradient
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    angle = np.arctan2(sobely, sobelx)  # Angle in radians

    # Convert angle to degrees for easier quantization (range -180 to 180 degrees)
    angle = np.degrees(angle)
    angle[angle < 0] += 360  # Normalize to 0-360 degrees

    # Quantize angles into `num_bins` angular bins
    bin_width = 360 / num_bins
    quantized_angles = np.floor(angle / bin_width).astype(int) % num_bins

    # Quantize magnitudes into `num_bins` bins
    # Normalize the magnitude to a range [0, 1] for quantization
    magnitude_normalized = magnitude / np.max(magnitude) if np.max(magnitude) > 0 else magnitude
    quantized_magnitudes = np.floor(magnitude_normalized * num_bins).astype(int)  # Scale and quantize

    # Ensure the quantized magnitudes fit within the bin range
    quantized_magnitudes[quantized_magnitudes >= num_bins] = num_bins - 1  # Cap at max bin

    return quantized_angles, quantized_magnitudes, magnitude


def color_texture_grid_descriptor(image, grid_size=(4, 4), color_bins=8, sobel_bins=8):
    # Get image dimensions and calculate grid cell size
    height, width, _ = image.shape
    cell_height, cell_width = height // grid_size[0], width // grid_size[1]
    
    # Initialize lists to store descriptors for color histograms and Sobel features
    grid_overall_descriptor = []
    overall_color_angle_hist = [] 
    overall_color_mag_hist = []

    # Iterate over grid cells
    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            # Extract grid cell
            start_y, start_x = row * cell_height, col * cell_width
            end_y, end_x = (row + 1) * cell_height, (col + 1) * cell_width
            grid_cell = image[start_y:end_y, start_x:end_x]
            
            # Calculate color histogram descriptor for the grid cell
            color_hist_descriptor = colour_histogram(grid_cell, color_bins)

            # Calculate Sobel descriptors for the grid cell
            gray_grid_cell = cv2.cvtColor(grid_cell, cv2.COLOR_RGB2GRAY)
            quantized_angles, quantized_magnitudes, _ = sobel_quantization(gray_grid_cell, sobel_bins)
            
            # Create histograms for Sobel angle and magnitude descriptors
            angle_hist = np.bincount(quantized_angles.flatten(), minlength=sobel_bins)
            magnitude_hist = np.bincount(quantized_magnitudes.flatten(), minlength=sobel_bins)
            
            # Concatenate angle and magnitude histograms to form Sobel descriptor
            sobel_descriptor = np.concatenate((angle_hist, magnitude_hist))
            # Concatenate color and Sobel descriptors
            grid_overall_descriptor.append(np.concatenate([color_hist_descriptor, sobel_descriptor]))
            overall_color_angle_hist.append(np.concatenate([color_hist_descriptor, angle_hist]))
            overall_color_mag_hist.append(np.concatenate([color_hist_descriptor, magnitude_hist]))
   
    # Final descriptor: Concatenate color and Sobel descriptors
    overall_descriptor = np.array(grid_overall_descriptor) 
    overall_descriptor.reshape(overall_descriptor.shape[0], -1) # Flatten 2nd & 3rd Dimension

    overall_color_angle_hist = np.array(overall_color_angle_hist) 
    overall_color_angle_hist.reshape(overall_color_angle_hist.shape[0], -1) # Flatten 2nd & 3rd Dimension

    overall_color_mag_hist = np.array(overall_color_mag_hist) 
    overall_color_mag_hist.reshape(overall_color_mag_hist.shape[0], -1) # Flatten 2nd & 3rd Dimension

    return overall_descriptor,overall_color_angle_hist,overall_color_mag_hist

def gabor_descriptor(image):
    # Apply Gabor filters with different orientations and frequencies
    gabor_features = []
    for theta in range(4):  # Four orientations
        theta_rad = theta * np.pi / 4
        kernel = cv2.getGaborKernel((21, 21), 8.0, theta_rad, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered_img = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        gabor_features.append(np.mean(filtered_img))
    return np.array(gabor_features)

from skimage.feature import graycomatrix, graycoprops
def haralick_features(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray_img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')
    correlation = graycoprops(glcm, 'correlation')
    energy = graycoprops(glcm, 'energy')
    return np.concatenate((contrast.flatten(), correlation.flatten(), energy.flatten()))


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
            F = colour_histogram(img,Q) ##  Implemented Histogram Bins 
            
            # Save the descriptor to a .mat file
            sio.savemat(fout, {'F': F})

    print("Successfully Created Global Color Descriptors")

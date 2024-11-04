import numpy as np

# Takes normalised image between 0.0 - 1.0 
# Q is number of Bin 

def compute_colour_histogram(image, Q):
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
import numpy as np

def cvpr_compare(F1, F2,type):
    # This function compares F1 to F2 - i.e. compute the distance
    # between the two descriptors
    # extended the capacity of this function to calc both L1 and L2 distance 
    
    if type == 'L2':
        F_diff = F1-F2
        F_diff_sqr = np.square(F_diff)
        l2_norm = np.sqrt(np.sum(F_diff_sqr))
        return l2_norm
    
    if type == 'L1':
        l1_norm = np.sum(np.absolute(F1 - F2))
        return l1_norm

    return None

import numpy as np
import cv2
from scipy.spatial import distance

def compute_all_distances(descriptor1, descriptor2, cov_matrix=None):
    
    # Calculate distances and store them in a dictionary
    distances = {}

    # L1 distance (Manhattan)
    distances["L1"] = np.sum(np.abs(descriptor1 - descriptor2))
    
    # L2 distance (Euclidean)
    distances["L2"] = np.sqrt(np.sum((descriptor1 - descriptor2) ** 2))
    
    # Mahalanobis distance
    if cov_matrix is not None:
        diff = descriptor1 - descriptor2
        distances["Mahalanobis"] = np.sqrt(np.dot(np.dot(diff.T, np.linalg.inv(cov_matrix)), diff))
    else:
        distances["Mahalanobis"] = None  # or could raise a warning if desired
    
    # Earth Mover's Distance (for histograms)
    descriptor1_float = descriptor1.astype(np.float32)
    descriptor2_float = descriptor2.astype(np.float32)
    emd_distance, _ = cv2.EMD(descriptor1_float, descriptor2_float, cv2.DIST_L2)
    distances["EMD"] = emd_distance
    
    # Bhattacharyya distance (for histograms)
    bhattacharyya_distance = cv2.compareHist(descriptor1_float, descriptor2_float, cv2.HISTCMP_BHATTACHARYYA)
    distances["Bhattacharyya"] = bhattacharyya_distance
    
    # Histogram Intersection (for histograms)
    histogram_intersection = np.sum(np.minimum(descriptor1, descriptor2)) / np.sum(descriptor1)
    distances["Histogram_Intersection"] = histogram_intersection
    
    # Hamming distance (for binary descriptors)
    distances["Hamming"] = distance.hamming(descriptor1, descriptor2) * len(descriptor1)
    
    # Chi-Squared distance (for histograms)
    chi_squared = 0.5 * np.sum(((descriptor1 - descriptor2) ** 2) / (descriptor1 + descriptor2 + 1e-10))
    distances["Chi-Squared"] = chi_squared
    
    return distances



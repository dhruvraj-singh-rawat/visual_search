import numpy as np
from cvpr_toolset import extract_class_and_file
from sklearn.metrics import auc
import pandas as pd 

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

    distances = {}
    
    # Ensure the descriptors are 1-D arrays (flatten if necessary)
    descriptor1 = descriptor1.flatten()
    descriptor2 = descriptor2.flatten()


    # L1 Distance
    distances["L1"] = np.sum(np.abs(descriptor1 - descriptor2))

    # L2 Distance
    distances["L2"] = np.sqrt(np.sum((descriptor1 - descriptor2) ** 2))

    # Chi-Squared Distance
    distances["Chi-Squared"] = 0.5 * np.sum(((descriptor1 - descriptor2) ** 2) / (descriptor1 + descriptor2 + 1e-10))

    # Cosine Distance
    distances["Cosine"] = 1 - (np.dot(descriptor1, descriptor2) / (np.linalg.norm(descriptor1) * np.linalg.norm(descriptor2) + 1e-10))


    return distances


def calculate_auc_for_descriptors(query_img_index, descriptors, distance_measures, all_files):
    auc_results = []

    for descriptor_name, descriptor_data in descriptors.items():
        query = descriptor_data[query_img_index]
        distances = {measure: [] for measure in distance_measures}

        # Compute distances for all images
        for i in range(len(descriptor_data)):
            candidate = descriptor_data[i]
            for measure in distance_measures:
                distance = compute_all_distances(query, candidate)[measure]
                distances[measure].append((distance, i))

        # Calculate AUC for each distance measure
        for measure, distance_list in distances.items():
            # Sort distances
            distance_list.sort(key=lambda x: x[0])
            response_classes = []
            
            for dist, img_no in distance_list:
                temp_class, temp_file = extract_class_and_file(all_files[img_no])
                response_classes.append(temp_class)

            # Extract query class
            query_class = response_classes[0]
            response_classes = response_classes[1:]  # Remove the query class itself

            # Calculate precision and recall
            precision, recall = [], []
            relevant_retrieved = 0
            total_relevant = sum(1 for cls in response_classes if cls == query_class)

            for i, cls in enumerate(response_classes):
                if cls == query_class:
                    relevant_retrieved += 1

                current_precision = relevant_retrieved / (i + 1)
                current_recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0

                precision.append(current_precision)
                recall.append(current_recall)

            # Calculate AUC for the precision-recall curve
            pr_auc = auc(recall, precision)
            auc_results.append({'Descriptor': descriptor_name, 'Distance Measure': measure, 'AUC': pr_auc})

    # Create a DataFrame from the results
    auc_df = pd.DataFrame(auc_results)
    return auc_df
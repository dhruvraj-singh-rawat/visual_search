import numpy as np
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, precision_score, recall_score


def extract_class_and_file(file_path):
    # Regular expression to match the class name and file number
    match = re.search(r'/(\d+)_(\d+)_s\.bmp$', file_path)
    
    if match:
        class_name = match.group(1)
        file_number = match.group(2)
        return int(class_name), int(file_number)
    else:
        return None, None
    

def calculate_precision_recall(query_class, response_class_arr):
    # Convert to numpy arrays
    query_class = np.array(query_class)
    response_class_arr = np.array(response_class_arr)
    
    # Convert to binary (1 if class matches, 0 if not)
    y_true = (response_class_arr == query_class).astype(int)
    
    # Assuming a score system where class matches have higher relevance
    y_scores = np.ones_like(y_true)  # Assign score of 1 for all responses
    
    # Calculate precision and recall for plotting the curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
    
    # Calculate overall precision and recall values
    overall_precision = precision_score(y_true, y_scores)
    overall_recall = recall_score(y_true, y_scores)
    
    return precision_curve, recall_curve, overall_precision, overall_recall
   
def plot_precision_recall(precision, recall):
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.show()

def evalute_performance(query_image_name, arr_response_names):

    query_class , query_class_file_num =  extract_class_and_file(query_image_name)
    
    response_class_arr = []

    for response in arr_response_names: 
        temp_response_class, _ = extract_class_and_file(response)
        response_class_arr.append(temp_response_class)

    # Calculate precision and recall
    precision_curve, recall_curve, overall_precision, overall_recall = calculate_precision_recall(query_class, response_class_arr)
    return precision_curve, recall_curve, overall_precision, overall_recall 

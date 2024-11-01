import os
import numpy as np
import scipy.io as sio
import cv2
from random import randint
from cvpr_compare import cvpr_compare
from cvpr_evaluation import evalute_performance,plot_precision_recall,extract_class_and_file
#import ipdb

DESCRIPTOR_FOLDER = 'descriptors'
DESCRIPTOR_SUBFOLDER = 'globalRGBhisto'
IMAGE_FOLDER = 'MSRC_ObjCategImageDatabase_v2'

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

# Pick a random image as the query
NIMG = ALLFEAT.shape[0]
queryimg = randint(0, NIMG - 1)

# Compute the distance between the query and all other descriptors
dst = []
query = ALLFEAT[queryimg]
for i in range(NIMG):
    candidate = ALLFEAT[i]
    distance = cvpr_compare(query, candidate)
    dst.append((distance, i))

# Sort the distances
dst.sort(key=lambda x: x[0])

# Show the top 15 results
SHOW = 15
for i in range(SHOW):
    img = cv2.imread(ALLFILES[dst[i][1]])
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))  # Make image quarter size
    cv2.imshow(f"Result {i+1}", img)
    cv2.waitKey(0)
cv2.destroyAllWindows()

query_image_name = ALLFILES[queryimg]
response_image_arr = [] 
for i in range(SHOW):
    response_image_arr.append(ALLFILES[dst[i][1]])

precision_curve, recall_curve, overall_precision, overall_recall  = evalute_performance(query_image_name, response_image_arr)

query_class,_ = extract_class_and_file(query_image_name)
response_class = [] 
for res_img in response_image_arr:
    temp_class , temp_file = extract_class_and_file(res_img)
    response_class.append(temp_class)


print("Query name: ",query_class)
print("Response arr: ",response_class)

print("Precision: ",round(overall_precision,2))
print("Recall: ",round(overall_recall,2))


plot_precision_recall(precision_curve, recall_curve)

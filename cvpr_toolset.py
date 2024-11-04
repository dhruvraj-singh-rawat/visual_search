import numpy as np
import re
import numpy as np
import matplotlib.pyplot as plt

def extract_class_and_file(file_path):
    # Regular expression to match the class name and file number
    match = re.search(r'/(\d+)_(\d+)_s\.bmp$', file_path)
    
    if match:
        class_name = match.group(1)
        file_number = match.group(2)
        return int(class_name), int(file_number)
    else:
        return None, None

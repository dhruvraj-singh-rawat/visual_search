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



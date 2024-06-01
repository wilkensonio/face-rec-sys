import numpy as np
import os 
import glob
from sklearn.model_selection import train_test_split

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)




def filterdata():
# Defines folder with datasets
    base_path = r'B:\Github Repos\csc481A\AR_DB\points_22'
    data_array = []
    # Collect all file paths
    all_files = glob.glob(os.path.join(base_path, 'm-*', '*.pts')) + glob.glob(os.path.join(base_path, 'w-*', '*.pts'))
    for file in all_files:
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines[3:25] :
                xx, yy = line.split()
                data_array.append((xx, yy))
                
                #data_array 
    print(data_array)           
filterdata()
    
                
        




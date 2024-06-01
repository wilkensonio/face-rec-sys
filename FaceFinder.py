import numpy as np
import os 
import glob
from sklearn.model_selection import train_test_split

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)





# Defines folder with datasets
base_path = r'B:\Github Repos\csc481A\Face Markup AR Database\points_22'

# Collect all file paths
all_files = glob.glob(os.path.join(base_path, 'm-*', '*.pts')) + glob.glob(os.path.join(base_path, 'w-*', '*.pts'))

print (all_files)


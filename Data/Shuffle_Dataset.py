import GPyOpt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import math
#import sys
import os
#import shutil
from datetime import datetime
#import random
import matplotlib.pyplot as plt

dataset_file = r"Data_Set.txt";
initial_dataset = pd.read_csv(dataset_file, sep='\t',header=0)
initial_dataset['RandomNumber'] = 0


arr = np.arange(len(initial_dataset)/3)
np.random.shuffle(arr)


#shuffle the data set consisting of 3 measurements per experiment.
for i in range(int(len(initial_dataset)/3)):
    initial_dataset['RandomNumber'][i*3] = arr[i]
    initial_dataset['RandomNumber'][i*3+1] = arr[i]
    initial_dataset['RandomNumber'][i*3+2] = arr[i]
    
initial_dataset = initial_dataset.sort_values(['RandomNumber'],ascending=False)
initial_dataset = initial_dataset[["HEA","Pt","Ru","Pd","Rh","Au","Onset"]]
initial_dataset.to_csv('Data_Set_Shuffled.txt', sep='\t', index=False)    
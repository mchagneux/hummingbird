import shutil 
import pandas as pd 
import os 
filenames = pd.read_csv('train.csv').iloc[:,0]

for filename in filenames:
    os.remove(filename)
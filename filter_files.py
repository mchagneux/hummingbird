import shutil 
import pandas as pd 
import os 
valid_filenames = list(pd.read_csv('train.csv').iloc[:,0])
for filename in os.listdir('train'):
    if filename not in valid_filenames:
        os.remove(os.path.join('train',filename))
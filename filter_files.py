import shutil 
import pandas as pd 
import os 
filenames = pd.read_csv('train.csv').iloc[:,0]

for filename in filenames:
    image_name = filename.split('train/')[1]
    os.makedirs('train', exist_ok=True)
    shutil.copy(filename, os.path.join('train', image_name))
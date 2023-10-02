import pandas as pd
import os

full_annotation_file = pd.read_csv("data_full_annotations.csv")

def get_annotation_file(name, annotation_file, write = False):
  files = os.listdir(name)
  annotations = annotation_file[annotation_file['filename'].isin(files)]
  annotations.loc[:, "filename"] =  name + '/' + annotations.loc[:, "filename"]
  if write:
    annotations.to_csv(name + '.csv', index=False, header=False)
  else:
    return annotations

get_annotation_file("train", full_annotation_file, write = True)
get_annotation_file("test", full_annotation_file, write = True)
train_files = get_annotation_file("train", full_annotation_file, write = False)

class_names = pd.Categorical(pd.Series(train_files["class"].unique()), 
              categories=["empty", "hummingbird", "rodent", "frog", "other_bird", "bat", "oppossum"])
classes = pd.DataFrame({"class_name": class_names}).sort_values("class_name").reset_index(drop=True)
classes["class_number"] = classes.index
classes.to_csv('class.csv', index=False, header=False)


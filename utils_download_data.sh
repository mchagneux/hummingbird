#!/bin/sh

rm -rf train test
rm train.csv test.csv class.csv

# wget télécharge
# -0- le met sur la sortie standard de wget le contenu du fichier
# tar x détare 
wget -O- "https://seafile.agroparistech.fr/f/6f479e65efb04a00b48c/?dl=1" \
  | tar x 
python3 utils_create_train_test_class_files.py

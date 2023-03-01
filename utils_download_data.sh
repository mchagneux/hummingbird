#!/bin/sh

rm -rf train test
rm train.csv test.csv class.csv

# wget télécharge
# -0- le met sur la sortie standard de wget le contenu du fichier
# tar x détare 
wget -O- "https://seafile.agroparistech.fr/f/8a94b7c9ec3e44c7a3b7/?dl=1" | tar x
python3 utils_create_train_test_class_files.py

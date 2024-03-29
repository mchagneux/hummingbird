#!/bin/bash                                                                                                        
       
# Les lignes commançant par #SBATCH sont interprétées par SLURM.
# Elles permettent demander les ressources nécessaires au job.

# Nombre de Noeud
#SBATCH --nodes=1

# Nombre de processeur par noeud
#SBATCH --ntasks-per-node=4

# Nom du job
#SBATCH --job-name=Train_RetinaColibri_200epoch

# Temps max d'execution0

#SBATCH --time=100:00:00

# Quantité de RAM par noeud
#SBATCH --mem=10G

# Quel type de machine demander (type_1 ou type_2)
#SBATCH --partition=type_1

#SBATCH --output=sortie_job_200epoch.out
#SBATCH --error=erreur_job_200epoch.err

# Chargement des modules
module load userspace/tr17.10
module load python/conda

# Activer env conda ET lire le script python
source activate retinanet_detection
python3 learning_from_colibris.py

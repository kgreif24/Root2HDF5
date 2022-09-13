#!/bin/bash

# Job sumission script for data generation on gpatlas

#SBATCH --job-name=ms
#SBATCH --time=10:00:00 # hh:mm:ss

#SBATCH --partition=atlas

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=20G

#SBATCH --output=ms.out 
#SBATCH --error=ms.err

#SBATCH --mail-user=kgreif@uci.edu
#SBATCH --mail-type=ALL

python make_set.py

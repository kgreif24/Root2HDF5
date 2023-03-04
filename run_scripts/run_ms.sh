#!/bin/bash

# Job sumission script for data generation on gpatlas

#SBATCH --job-name=ms_tst_nominal
#SBATCH --time=10:00:00 # hh:mm:ss

#SBATCH --partition=atlas

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=10G

#SBATCH --output=ms_tst_nominal.out 
#SBATCH --error=ms_tst_nominal.err

#SBATCH --mail-user=kgreif@uci.edu
#SBATCH --mail-type=ALL

python make_set.py

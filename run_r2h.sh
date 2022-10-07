#!/bin/bash

# Job sumission script for data generation on gpatlas

#SBATCH --job-name=data_gen
#SBATCH --time=20:00:00 # hh:mm:ss

#SBATCH --partition=atlas

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=20G

#SBATCH --output=data_gen_ln.out 
#SBATCH --error=data_gen_ln.err

#SBATCH --mail-user=kgreif@uci.edu
#SBATCH --mail-type=ALL

python r2h_lognorm.py

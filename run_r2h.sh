#!/bin/bash

# Job sumission script for data generation on gpatlas

#SBATCH --job-name=data_gen
#SBATCH --time=20:00:00 # hh:mm:ss

#SBATCH --partition=atlas

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=10G

#SBATCH --output=data_gen.out 
#SBATCH --error=data_gen.err

#SBATCH --mail-user=kgreif@uci.edu
#SBATCH --mail-type=ALL

python r2h_syst.py

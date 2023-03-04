#!/bin/bash

# Job sumission script for data generation on gpatlas

#SBATCH --job-name=r2h_zp_tst_nominal
#SBATCH --time=20:00:00 # hh:mm:ss

#SBATCH --partition=atlas

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --mem=10G

#SBATCH --output=r2h_zp_tst_nominal.out 
#SBATCH --error=r2h_zp_tst_nominal.err

#SBATCH --mail-user=kgreif@uci.edu
#SBATCH --mail-type=ALL

python r2h_lognorm.py

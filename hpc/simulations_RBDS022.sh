#!/bin/bash
#SBATCH --job-name=simulations_RBDS022
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --output=simulations_RBDS022.log
#SBATCH --mail-user=aaron.f@deakin.edu.au
#SBATCH --mail-type=ALL

python run_simulations.py -p RBDS022 -s T35
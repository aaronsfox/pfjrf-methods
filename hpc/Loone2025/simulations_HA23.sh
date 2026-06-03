#!/bin/bash
#SBATCH --job-name=simulations_HA23
#SBATCH --ntasks=1
#SBATCH --partition=normal
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --output=simulations_HA23.log
#SBATCH --mail-user=aaron.f@deakin.edu.au
#SBATCH --mail-type=ALL

python run_simulations.py -p HA23 -c SRRun
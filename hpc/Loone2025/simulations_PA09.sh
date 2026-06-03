#!/bin/bash
#SBATCH --job-name=simulations_PA09
#SBATCH --ntasks=1
#SBATCH --partition=normal
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --output=simulations_PA09.log
#SBATCH --mail-user=aaron.f@deakin.edu.au
#SBATCH --mail-type=ALL

python run_simulations.py -p PA09 -c SRRun
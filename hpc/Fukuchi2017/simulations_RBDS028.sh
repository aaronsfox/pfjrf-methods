#!/bin/bash
#SBATCH --job-name=simulations_RBDS028
#SBATCH --ntasks=1
#SBATCH --partition=normal
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --output=simulations_RBDS028.log
#SBATCH --mail-user=aaron.f@deakin.edu.au
#SBATCH --mail-type=ALL

python run_simulations.py -p RBDS028 -s T25
python run_simulations.py -p RBDS028 -s T35
python run_simulations.py -p RBDS028 -s T45
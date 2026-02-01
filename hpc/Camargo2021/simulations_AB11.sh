#!/bin/bash
#SBATCH --job-name=simulations_AB11
#SBATCH --ntasks=1
#SBATCH --partition=normal
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --output=simulations_AB11.log
#SBATCH --mail-user=aaron.f@deakin.edu.au
#SBATCH --mail-type=ALL

python run_simulations.py --participant AB11 --direction ascent --height 4 --leg rl
python run_simulations.py --participant AB11 --direction descent --height 4 --leg rl
python run_simulations.py --participant AB11 --direction ascent --height 7 --leg rl
python run_simulations.py --participant AB11 --direction descent --height 7 --leg rl
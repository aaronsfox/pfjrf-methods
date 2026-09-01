# -*- coding: utf-8 -*-
"""

@author:

    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au

    Quick script to write sbatch files for HPC runs.

"""

# =========================================================================
# Import packages
# =========================================================================

import os

# =========================================================================
# Set-up
# =========================================================================

# Identify participants based on who is in the data folder
participant_list = [
    ii for ii in os.listdir(
        os.path.join('..', '..', 'data', 'Scherpereel2023')
    ) if os.path.isdir(os.path.join('..', '..', 'data', 'Scherpereel2023', ii))]
# participant_list = ['AB05','AB12']

# =========================================================================
# Define functions
# =========================================================================

# Create sbatch file for participant for the squat
# -------------------------------------------------------------------------
def create_sbatch_squat(participant):

    # Define the sbatch directives
    job_name = f'simulations_{participant}_squat'
    ntasks = 1
    mem_per_cpu = '8G'
    cpus_per_task = 12
    timeout = '12:00:00'
    output_file = f'simulations_{participant}_squat.log'
    mail_user = 'aaron.f@deakin.edu.au'
    mail_type = 'ALL'

    # Create the sbatch output
    sbatch_directives = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --ntasks={ntasks}
#SBATCH --partition=normal
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time={timeout}
#SBATCH --output={output_file}
#SBATCH --mail-user={mail_user}
#SBATCH --mail-type={mail_type}

python run_simulations_squat.py -p {participant}"""

    # Define the filename for the .sh script
    filename = f'simulations_{participant}_squat.sh'

    # Write the content to the file with unix line endings
    with open(filename, 'w', newline='') as f:
        f.write(sbatch_directives)

    # Print confirmation
    print(f'{"*" * 10} Created sbatch script for {participant} squat {"*" * 10}')

# =========================================================================
# Create sbatch scripts
# =========================================================================

if __name__ == '__main__':

    # Create sbatch scripts
    # -------------------------------------------------------------------------

    # Loop through participants to create sbatch scripts for all speeds
    for participant in participant_list:
        create_sbatch_squat(participant)

    # Finalise and exit kernel
    # -------------------------------------------------------------------------

    # Doing this seems to avoid an error code when completing the script run
    os._exit(00)

# %% ---------- end of gen_sbatch.py ---------- %% #
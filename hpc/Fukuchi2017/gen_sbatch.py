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
        os.path.join('..', '..', 'data', 'Fukuchi2017')
    ) if os.path.isdir(os.path.join('..', '..', 'data', 'Fukuchi2017', ii))]

# =========================================================================
# Define functions
# =========================================================================

# Create sbatch file for participant for a single speed
# -------------------------------------------------------------------------
def create_sbatch(participant, speed):

    # Define the sbatch directives
    job_name = f'simulations_{participant}_{speed}'
    ntasks = 1
    mem_per_cpu = '8G'
    cpus_per_task = 12
    timeout = '12:00:00'
    output_file = f'simulations_{participant}_{speed}.log'
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

python run_simulations.py -p {participant} -s {speed}"""

    # Define the filename for the .sh script
    filename = f'simulations_{participant}_{speed}.sh'

    # Write the content to the file with unix line endings
    with open(filename, 'w', newline='') as f:
        f.write(sbatch_directives)

    # Print confirmation
    print(f'{"*" * 10} Created sbatch script for {participant} {speed} {"*" * 10}')

# Create sbatch file for participant that includes all speeds
# -------------------------------------------------------------------------
def create_sbatch_all(participant):

    # Define the sbatch directives
    job_name = f'simulations_{participant}'
    ntasks = 1
    mem_per_cpu = '8G'
    cpus_per_task = 12
    timeout = '12:00:00'
    output_file = f'simulations_{participant}.log'
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

python run_simulations.py -p {participant} -s T25
python run_simulations.py -p {participant} -s T35
python run_simulations.py -p {participant} -s T45"""

    # Define the filename for the .sh script
    filename = f'simulations_{participant}.sh'

    # Write the content to the file with unix line endings
    with open(filename, 'w', newline='') as f:
        f.write(sbatch_directives)

    # Print confirmation
    print(f'{"*" * 10} Created sbatch script for {participant} {"*" * 10}')


# =========================================================================
# Create sbatch scripts
# =========================================================================

if __name__ == '__main__':

    # Create sbatch scripts
    # -------------------------------------------------------------------------

    # # Loop through participants to create sbatch scripts for individual speeds
    # for participant in participant_list:
    #     for speed in ['T25','T35','T45']:
    #         create_sbatch(participant, speed)

    # Loop through participants to create sbatch scripts for all speeds
    for participant in participant_list:
        create_sbatch_all(participant)

    # Finalise and exit kernel
    # -------------------------------------------------------------------------

    # Doing this seems to avoid an error code when completing the script run
    os._exit(00)

# %% ---------- end of gen_sbatch.py ---------- %% #
# -*- coding: utf-8 -*-
"""

@author:

    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au

    This script creates a file to refer to in performing a pseudo randomised
    selection of trials for each participant in the Camargo2021 dataset. It
    randomly selects one trial from each stair height, ascent/descent direction,
    and transition leg order. Not all trials are used in later simulations,
    but this process gives a platform to select from for any simulation approach
    desired.

"""

# =========================================================================
# Import packages
# =========================================================================

import pandas as pd
import numpy as np
import os
import shutil

# =========================================================================
# Select trials
# =========================================================================

# Read in the trial info dataset
trial_data = pd.read_csv('all-participants_stair_trial-info.csv')

# Define columns to group by
group_cols = ['participant', 'stair_height', 'direction', 'trans_leg']

# Group the data by columns and sample
random_seed = np.random.seed(12345)
selected_trials = trial_data.groupby(group_cols, dropna=True).sample(
    n=1, random_state=random_seed).reset_index(drop=True)

# Write trials to file
selected_trials.to_csv('select-participants_stair_trial-info.csv', index=False)

# %% ---------- end of trial_selection.py ---------- %% #

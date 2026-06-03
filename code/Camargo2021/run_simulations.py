# -*- coding: utf-8 -*-
"""

@author:

    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au

    This script runs the various torque and muscle driven simulations of
    the stair climbing tasks to generate the outputs for calculating
    patellofemoral joint reaction forces with different modelling approaches.

    With the approach of using state tracking for the dynamic optimisation,
    this simulation needs to successfully run first to capture the comparative
    kinematics to use in the static optimisation, muscle analysis and inverse
    dynamics approaches.

    STATUS:
        > Reviewed as tentatively complete to check simulations on 20/01/2026
        > Updated for better trial start time to check simulations on 21/01/2026
        > Updated with initial guess code 23/01/2026 - checked with participant AB12
        > Copied AB12 simulations across after working 24/01/2026
        > Tested on AB11 26/01/2026
        > Pelvis kinematic tracking not great with initial weights. Upped translations by e2 and rotations e1
        > Re-tested on AB11 27/0/2026
        > Upped mesh interval, removed marker tracking for simplicity, re-tested on original AB12 guess creation
            >> Doesn't seem to fix pelvis drift problem
            >> Higher marker tracking weight?
        > Mesh interval of 50 is probably too slow and impractical to use
        > Updated with marker only and higher effort weight goal as per Fukuchi code
        > High frequency noise in solution and potential drift at end
        > Add in 50 millisecond buffer to avoid erroneous forces at start/end of simulation

    TODO:
        > Add final marker goal to have model in position appropriately at end?
        > Add smoothing criteria for bodies or potentially penalise torque controls a little more?
        > Not sure added 50 milliseconds helps in this instance - causes issues at beginning and end
        > Minimising accelerations avoids jitteriness, but might also cause poorer tracking?
        > Consider polynomial muscle paths given improvements with OpenSim 4.6

        

"""

# =========================================================================
# Import packages
# =========================================================================

import opensim as osim
import os
import numpy as np
import pandas as pd
import shutil
from scipy.signal import find_peaks, butter, sosfiltfilt
import matplotlib.pyplot as plt
import pickle
import argparse
import random
import re
import time

# =========================================================================
# Flags for running analyses
# =========================================================================

# Set participant ID to run
# participant = 'AB12'
# direction = 'ascent'
# height = 4
# leg = 'rl'
# trial = 'stairs_1_3'
parser = argparse.ArgumentParser()
parser.add_argument('--participant', action = 'store', type = str, help = 'Enter the participant ID')
parser.add_argument('--direction', action = 'store', type = str, help = 'Enter the stair direction (ascent or decent)')
parser.add_argument('--height', action = 'store', type = int, help = 'Enter the stair height (4 or 7)')
parser.add_argument('--leg', action = 'store', type = str, help = 'Enter the leg transition order (rl or lr). Note that rl only is used in this code...')
args = parser.parse_args()
participant = args.participant
direction = args.direction
height = args.height
leg = args.leg

# Settings for running specific sections of code
runDynamicOpt = True
runStaticOpt = True
runInverseDyn = True

# =========================================================================
# Set-up
# =========================================================================

# General settings
# -------------------------------------------------------------------------

# Set dataset name
dataset = 'Camargo2021'

# Read in participant info
participant_info = pd.read_csv(os.path.join('..', '..', 'data', dataset, 'participant_info.csv'))

# Get participant list from folder
participant_list = [ii for ii in os.listdir(
    os.path.join('..', '..', 'data', dataset)) if os.path.isdir(
    os.path.join(os.path.join('..', '..', 'data', dataset, ii)))]

# Check if input participant is in list
if participant not in participant_list:
    raise ValueError(f'No data found for participant ID {participant}. Check input for error...')

# Read in participant trial list
selected_trials = pd.read_csv(os.path.join('..', '..', 'data', dataset, 'select-participants_stair_trial-info.csv'))

# Set trial label
trial_label = f'{direction}_{height}_{leg}'

# Create the general folder for the participant and trial
os.makedirs(os.path.join('..','..','simulations',dataset,participant,trial_label), exist_ok=True)

# Plot settings
# -------------------------------------------------------------------------

# Set matplotlib parameters
from matplotlib import rcParams
import matplotlib
# matplotlib.use('TkAgg')
# plt.ion()

# rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Arial'
rcParams['font.weight'] = 'bold'
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 16
rcParams['axes.linewidth'] = 1.5
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.spines.right'] = False
rcParams['axes.spines.top'] = False
rcParams['legend.fontsize'] = 10
rcParams['xtick.major.width'] = 1.5
rcParams['ytick.major.width'] = 1.5
rcParams['legend.framealpha'] = 0.0
rcParams['savefig.dpi'] = 300
rcParams['savefig.format'] = 'pdf'

# OpenSim settings
# -------------------------------------------------------------------------

# Add the utility geometry path for model visualisation
osim.ModelVisualizer.addDirToGeometrySearchPaths(os.path.join(os.getcwd(), '..', '..', 'model', 'Geometry'))

# Set weights for optimisations
# Nitschke et al. (2023) used higher value of 1e0 for muscular effort weight
# Nitschke et al. (2023) used lower weight of 10e-1 on joint tracking, but was in degrees squared (rather than radians?)
# Nitschke et al. (2023) used lower weight of 10e-2 on joint tracking, but was in mm squared?
globalMarkerTrackingWeight = 1e1  # TODO: dramatically increased this from 1e-1 to get better tracking?
# globalMarkerFinalWeight = 1e2  # TODO: needed? too high?
# globalStateTrackingWeight = 1e-1  # TODO: reduced this to check if data follows markers better
globalAccelMinWeight = 1e-6  # TODO: useful?
# globalTorqueControlWeight = 1e-3
# globalMuscleControlWeight = 1e-3
globalControlEffortWeight = 1e0  # TODO: originally 1e-3 - is this therefore causing high activations? yes, but need to balance state tracking...

# Set mesh interval for dynamic optimisation
# Note this is somewhat generic
# mesh_interval_dyn = 25
mesh_interval_dyn = 50  # used for creating more refined consistent guess

# Set kinematics filter frequency
# This matches filter from associated paper
kinematic_filt_freq = 6

# =========================================================================
# Define functions
# =========================================================================

# Define simple low-pass butterworth filter
# -------------------------------------------------------------------------
def butter_lowpass_filter(data, cutoff, fs, order=4):
    # Nyquist Frequency is half the sampling rate
    nyq = 0.5 * fs
    # Normalize the cutoff frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients (using 'sos' for numerical stability)
    sos = butter(order, normal_cutoff, btype='low', analog=False, output='sos')
    # Apply the filter
    y = sosfiltfilt(sos, data)
    return y


# Run a marker tracking dynamic optimisation
# -------------------------------------------------------------------------
def run_dynamic_optimisation(model_type):

    """

    This function runs the muscle-driven tracking simulations, with the goal
    here to generate a muscle-driven simulation of the step climb that minimises
    the state and marker tracking error and muscle activations. These simulations will
    generate the kinematic data to be consistently used across other approaches.

    """

    # Do an initial input check as the function won't work with an incorrect model type
    if model_type != 'complex' and model_type != 'simple':
        raise ValueError('model_type variable must be string of "complex" or "simple".')

    # =========================================================================
    # Set-up files and parameters for simulation
    # =========================================================================

    # Files and settings
    # -------------------------------------------------------------------------

    # Create the folder for simulation
    os.makedirs(os.path.join('..','..','simulations',dataset,participant,trial_label,'dynamic_optimisation'),
                exist_ok=True)

    # Get trial name from selected trials
    # Order of trial label is direction, stair height, transition leg
    trial_split = trial_label.split('_')
    trial_data = selected_trials.loc[
        (selected_trials['participant'] == participant) &
        (selected_trials['direction'] == trial_split[0]) &
        (selected_trials['stair_height'] == int(trial_split[1])) &
        (selected_trials['trans_leg'] == trial_split[2]),]
    trial_name = trial_data['trial_name'].values[0]

    # Navigate to simulation folder for ease of use
    home_dir = os.getcwd()
    os.chdir(os.path.join('..','..','simulations',dataset,participant,trial_label,'dynamic_optimisation'))

    # Copy GRF file to simulation directory
    shutil.copyfile(
        os.path.join('..', '..', '..', '..', '..', 'data', dataset, participant, f'{trial_name}_grf.mot'),
        f'{trial_name}_grf.mot')

    # Copy marker file to simulation directory
    shutil.copyfile(
        os.path.join('..', '..', '..', '..', '..', 'data', dataset, participant, f'{trial_name}.trc'),
        f'{trial_name}.trc')

    # Copy IK data to simulation directory
    shutil.copyfile(
        os.path.join('..', '..', '..', '..', '..', 'data', dataset, participant, 'ik', trial_label,
                     f'{participant}_{trial_label}_ik_{model_type}_filt.mot'),
        f'{participant}_{trial_label}_ik_{model_type}_filt.mot')

    # Create associated external loads file for use in simulation
    ext_loads = osim.ExternalLoads()
    # This is a simplified external loads file that only allocates force plate data over the period of interest
    # It also assumes that the 'rl' transition leg trials are being used
    # For ascent trials
    if trial_split[0] == 'ascent':
        # FP5 gets allocated to left leg on initial step
        left_force = osim.ExternalForce()
        left_force.setName('LeftGRF')
        left_force.setAppliedToBodyName('calcn_l')
        left_force.set_force_expressed_in_body('ground')
        left_force.set_point_expressed_in_body('ground')
        left_force.set_force_identifier('FP5_v')
        left_force.set_point_identifier('FP5_p')
        left_force.set_torque_identifier('FP5_moment_')
        ext_loads.cloneAndAppend(left_force)
        # FP4 gets allocated to right leg on second step
        right_force = osim.ExternalForce()
        right_force.setName('RightGRF')
        right_force.setAppliedToBodyName('calcn_r')
        right_force.set_force_expressed_in_body('ground')
        right_force.set_point_expressed_in_body('ground')
        right_force.set_force_identifier('FP4_v')
        right_force.set_point_identifier('FP4_p')
        right_force.set_torque_identifier('FP4_moment_')
        ext_loads.cloneAndAppend(right_force)
    elif trial_split[0] == 'descent':
        # FP1 gets allocated to left leg on initial step
        left_force = osim.ExternalForce()
        left_force.setName('LeftGRF')
        left_force.setAppliedToBodyName('calcn_l')
        left_force.set_force_expressed_in_body('ground')
        left_force.set_point_expressed_in_body('ground')
        left_force.set_force_identifier('FP1_v')
        left_force.set_point_identifier('FP1_p')
        left_force.set_torque_identifier('FP1_moment_')
        ext_loads.cloneAndAppend(left_force)
        # FP2 gets allocated to right leg on second step
        right_force = osim.ExternalForce()
        right_force.setName('RightGRF')
        right_force.setAppliedToBodyName('calcn_r')
        right_force.set_force_expressed_in_body('ground')
        right_force.set_point_expressed_in_body('ground')
        right_force.set_force_identifier('FP2_v')
        right_force.set_point_identifier('FP2_p')
        right_force.set_torque_identifier('FP2_moment_')
        ext_loads.cloneAndAppend(right_force)
    else:
        # Throw error for not identifying trial correctly
        raise ValueError('Trial not identified as ascent or descent, check labels...')
    # Set datafile
    ext_loads.setDataFileName(f'{trial_name}_grf.mot')
    # Print to file
    ext_loads.printToXML(f'{trial_name}_grf.xml')

    # Set marker tracking weights
    marker_weights = {
        # Pelvis
        'R_ASIS': {'weight': 5.0}, 'L_ASIS': {'weight': 5.0}, 'R_PSIS': {'weight': 5.0}, 'L_PSIS': {'weight': 5.0},
        # Right thigh
        'R_Thigh_Upper': {'weight': 2.5}, 'R_Thigh_Front': {'weight': 2.5}, 'R_Thigh_Rear': {'weight': 2.5},
        'R_Knee_Lat': {'weight': 0.0},
        # Right shank
        'R_Shank_Upper': {'weight': 2.5}, 'R_Shank_Front': {'weight': 2.5}, 'R_Shank_Rear': {'weight': 2.5},
        'R_Ankle_Lat': {'weight': 0.0},
        # Right foot
        'R_Heel': {'weight': 10.0},
        'R_Toe_Tip': {'weight': 5.0}, 'R_Toe_Med': {'weight': 5.0}, 'R_Toe_Lat': {'weight': 5.0},
        # Left thigh
        'L_Thigh_Upper': {'weight': 2.5}, 'L_Thigh_Front': {'weight': 2.5}, 'L_Thigh_Rear': {'weight': 2.5},
        'L_Knee_Lat': {'weight': 0.0},
        # Left shank
        'L_Shank_Upper': {'weight': 2.5}, 'L_Shank_Front': {'weight': 2.5}, 'L_Shank_Rear': {'weight': 2.5},
        'L_Ankle_Lat': {'weight': 0.0},
        # Left foot
        'L_Heel': {'weight': 10.0},
        'L_Toe_Tip': {'weight': 5.0}, 'L_Toe_Med': {'weight': 5.0}, 'L_Toe_Lat': {'weight': 5.0},
    }

    # # Set state tracking weights
    # state_weights = {'pelvis_tx': 2.5e2, 'pelvis_ty': 1.0e2, 'pelvis_tz': 2.5e2,
    #         'pelvis_tilt': 5.0e1, 'pelvis_list': 10.0e1, 'pelvis_rotation': 5.0e1,
    #         'hip_flexion_r': 15.0, 'hip_adduction_r': 7.5, 'hip_rotation_r': 5.0,
    #         'knee_angle_r': 25.0, 'ankle_angle_r': 15.0,
    #         'hip_flexion_l': 15.0, 'hip_adduction_l': 7.5, 'hip_rotation_l': 5.0,
    #         'knee_angle_l': 25.0, 'ankle_angle_l': 15.0,
    #         }
    # speeds_scale = 0.01

    # Set actuator forces to support simulation
    act_forces = {'pelvis_tx': {'actuatorType': 'residual', 'optForce': 5},
                  'pelvis_ty': {'actuatorType': 'residual', 'optForce': 5},
                  'pelvis_tz': {'actuatorType': 'residual', 'optForce': 5},
                  'pelvis_tilt': {'actuatorType': 'residual', 'optForce': 2.5},
                  'pelvis_list': {'actuatorType': 'residual', 'optForce': 2.5},
                  'pelvis_rotation': {'actuatorType': 'residual', 'optForce': 2.5},
                  'hip_flexion_r': {'actuatorType': 'reserve', 'optForce': 3.0},
                  'hip_adduction_r': {'actuatorType': 'reserve', 'optForce': 2.0},
                  'hip_rotation_r': {'actuatorType': 'reserve', 'optForce': 1.0},
                  'knee_angle_r': {'actuatorType': 'reserve', 'optForce': 3.0},
                  'ankle_angle_r': {'actuatorType': 'reserve', 'optForce': 2.0},
                  'subtalar_angle_r': {'actuatorType': 'reserve', 'optForce': 1.0},
                  'hip_flexion_l': {'actuatorType': 'torque', 'optForce': 300.0},
                  'hip_adduction_l': {'actuatorType': 'torque', 'optForce': 200.0},
                  'hip_rotation_l': {'actuatorType': 'torque', 'optForce': 100.0},
                  'knee_angle_l': {'actuatorType': 'torque', 'optForce': 300.0},
                  'ankle_angle_l': {'actuatorType': 'torque', 'optForce': 200.0},
                  'subtalar_angle_l': {'actuatorType': 'torque', 'optForce': 100.0},
                  }

    # Identify the time period to run the simulation over
    # TODO: this could probably be optimised around repetition of code
    # -------------------------------------------------------------------------

    # Set force and frame thresholds to identify FP contacts
    force_threshold = 20.0
    frame_threshold = 200
    if trial_split[0] == 'ascent':
        frame_threshold_v = 50
    elif trial_split[0] == 'descent':
        frame_threshold_v = 20

    # Read in marker, IK and GRF data for use in algorithms
    # Trim marker and GRF data to match IK
    marker_trial = osim.TimeSeriesTableVec3(f'{trial_name}.trc').flatten()
    grf_trial = osim.TimeSeriesTable(f'{trial_name}_grf.mot')
    ik_trial = osim.TimeSeriesTable(f'{participant}_{trial_label}_ik_{model_type}_filt.mot')
    marker_trial.trim(ik_trial.getIndependentColumn()[0], ik_trial.getIndependentColumn()[-1])
    grf_trial.trim(ik_trial.getIndependentColumn()[0], ik_trial.getIndependentColumn()[-1])

    # Get start time based on direction
    if trial_split[0] == 'ascent':

        # For ascent trials on the right leg START is where the right trailing leg leaves the ground at the start of the trial
        # The vertical velocity of the stepping limb toe marker is checked for when this crosses a determined
        # threshold (0.10) for at least 50 frames
        # -------------------------------------------------------------------------
        # Get vertical toe tip marker position, plus the time, for the stepping leg
        vert_toe_tip = marker_trial.getDependentColumn('R_Toe_Tip_2').to_numpy() / 1000  # convert from mm to m
        t = np.array(marker_trial.getIndependentColumn())
        # Calculate velocity using gradient function to take derivative with respect to time
        # Filter data to smooth before peak identification
        vert_toe_tip_v = np.gradient(vert_toe_tip, t)
        vert_toe_tip_v_filt = butter_lowpass_filter(
            vert_toe_tip_v, kinematic_filt_freq, 1 / np.diff(t).mean(), order=4)
        # Find where data meets vertical velocity threshold
        above = vert_toe_tip_v_filt >= 0.10
        padded = np.pad(above, (1, 1), 'constant', constant_values=(False,))
        diff = np.diff(padded.astype(int))
        start_idx_v = np.where(diff == 1)[0]
        end_idx_v = np.where(diff == -1)[0]
        run_lengths = end_idx_v - start_idx_v
        valid_starts = start_idx_v[run_lengths >= frame_threshold_v]
        valid_ends = end_idx_v[run_lengths >= frame_threshold_v]
        # Identify foot-off time as the first occurrence in the trial
        start_time = t[valid_starts[0]]

        # For ascent trials on the right leg the END of the trial is the left leg touchdown after the first right leg touchdown
        # This is done using the algorithm from Foster et al. (https://doi.org/10.1016/j.gaitpost.2013.11.005)
        # This uses the local minima in leading limb (i.e. left limb) toe vertical velocity to determine touchdown
        # The L_Toe_Tip marker is used in determining toe vertical velocity
        # -------------------------------------------------------------------------
        # Identify times where right leg force plate data (FP4) is above threshold
        above = grf_trial.getDependentColumn('FP4_vy').to_numpy() >= force_threshold
        padded = np.pad(above, (1, 1), 'constant', constant_values=(False,))
        diff = np.diff(padded.astype(int))
        # Identify start and ends of periods of contact
        start_idx = np.where(diff == 1)[0]
        end_idx = np.where(diff == -1)[0]
        run_lengths = end_idx - start_idx
        valid_starts = start_idx[run_lengths >= frame_threshold]
        valid_ends = end_idx[run_lengths >= frame_threshold]
        # As the trial includes the ascent and descent the first valid start is the ascent
        right_contact_time = grf_trial.getIndependentColumn()[valid_starts[0]]
        if right_contact_time < start_time:
            raise ValueError('First right leg contact time after start time. Check timing data...')
        # Get vertical toe tip marker position of trailing leg, plus the time
        vert_toe_tip = marker_trial.getDependentColumn('L_Toe_Tip_2').to_numpy() / 1000  # convert from mm to m
        t = np.array(marker_trial.getIndependentColumn())
        # Calculate velocity using gradient function to take derivative with respect to time
        # Filter data to smooth before peak identification
        vert_toe_tip_v = np.gradient(vert_toe_tip, t)
        vert_toe_tip_v_filt = butter_lowpass_filter(
            vert_toe_tip_v, kinematic_filt_freq, 1 / np.diff(t).mean(), order=4)
        # Identify local minima peaks in vertical velocity (inverse to get peaks)
        peaks = find_peaks(vert_toe_tip_v_filt * -1,
                           height = 0.15,  # seems reasonable based on data
                           distance=100)[0]  # distance seems reasonable based on sampling rate and step freq
        # plt.plot(vert_toe_tip_v_filt)
        # plt.scatter(peaks, vert_toe_tip_v_filt[peaks], color='green', s=25, zorder=3)
        # Identify times when peaks occur relative to start_time
        peak_times = [t[ii] for ii in peaks]
        peak_time_diffs = [ii - (right_contact_time+0.10) for ii in peak_times]  # add a small buffer here to account for potential errors
        peak_index = peak_time_diffs.index(min(vv for vv in peak_time_diffs if vv > 0))
        # Identify touchdown time as first following start time
        end_time = peak_times[peak_index]

    elif trial_split[0] == 'descent':

        # # For descent trials on the right leg the start of the trial is the initial contact on FP2
        # # -------------------------------------------------------------------------
        # # Identify times where force plate data is above threshold
        # above = grf_trial.getDependentColumn('FP2_vy').to_numpy() >= force_threshold
        # padded = np.pad(above, (1, 1), 'constant', constant_values=(False,))
        # diff = np.diff(padded.astype(int))
        # # Identify start and ends of periods of contact
        # start_idx = np.where(diff == 1)[0]
        # end_idx = np.where(diff == -1)[0]
        # run_lengths = end_idx - start_idx
        # valid_starts = start_idx[run_lengths >= frame_threshold]
        # valid_ends = end_idx[run_lengths >= frame_threshold]
        # # As the trial includes the ascent and descent the second valid start is the ascent
        # start_time = grf_trial.getIndependentColumn()[valid_starts[1]]

        # For descent trials on the right leg START is where the right trailing leg leaves the ground at the start of the trial
        # The vertical velocity of the stepping limb toe marker is checked for when this crosses a determined
        # threshold (0.10) for at least 50 frames
        # -------------------------------------------------------------------------
        # Get toe marker and associated time data
        vert_toe_tip = marker_trial.getDependentColumn('R_Toe_Tip_2').to_numpy() / 1000  # convert from mm to m
        t = np.array(marker_trial.getIndependentColumn())
        # Calculate velocity using gradient function to take derivative with respect to time
        # Filter data to smooth before peak identification
        vert_toe_tip_v = np.gradient(vert_toe_tip, t)
        vert_toe_tip_v_filt = butter_lowpass_filter(
            vert_toe_tip_v, kinematic_filt_freq, 1 / np.diff(t).mean(), order=4)
        # Find where data meets vertical velocity threshold
        above = vert_toe_tip_v_filt >= 0.10
        padded = np.pad(above, (1, 1), 'constant', constant_values=(False,))
        diff = np.diff(padded.astype(int))
        start_idx_v = np.where(diff == 1)[0]
        end_idx_v = np.where(diff == -1)[0]
        run_lengths = end_idx_v - start_idx_v
        valid_starts = start_idx_v[run_lengths >= frame_threshold_v]
        valid_ends = end_idx_v[run_lengths >= frame_threshold_v]
        # Identify foot-off time as the first occurrence in the trial
        # Try to find this event, otherwise just use start of IK trial time
        try:
            start_time = t[valid_starts[0]]
        except:
            start_time = ik_trial.getIndependentColumn()[0]

        # For ascent trials on the right leg the END of the trial is the left leg touchdown after the first right leg touchdown
        # This is done using the algorithm from Foster et al. (https://doi.org/10.1016/j.gaitpost.2013.11.005)
        # This uses the local minima in centre of mass vertical velocity
        # The pelvis_y position is used as a proxy of COM position
        # -------------------------------------------------------------------------
        # Identify times where right leg force plate data (FP2) is above threshold
        above = grf_trial.getDependentColumn('FP2_vy').to_numpy() >= force_threshold
        padded = np.pad(above, (1, 1), 'constant', constant_values=(False,))
        diff = np.diff(padded.astype(int))
        # Identify start and ends of periods of contact
        start_idx = np.where(diff == 1)[0]
        end_idx = np.where(diff == -1)[0]
        run_lengths = end_idx - start_idx
        valid_starts = start_idx[run_lengths >= frame_threshold]
        valid_ends = end_idx[run_lengths >= frame_threshold]
        # As the trial includes just the descent the first valid start is the contact
        right_contact_time = grf_trial.getIndependentColumn()[valid_starts[0]]
        if right_contact_time < start_time:
            raise ValueError('First right leg contact time after start time. Check timing data...')
        # Get the vertical pelvis position as a proxy of vertical COM position
        pelvis_ty_pos = ik_trial.getDependentColumn('/jointset/ground_pelvis/pelvis_ty/value').to_numpy()
        t = np.array(ik_trial.getIndependentColumn())
        # Calculate velocity using gradient function to take derivative with respect to time
        # Filter data to smooth before peak identification
        com_vert_v = np.gradient(pelvis_ty_pos, t)
        # com_vert_v_filt = butter_lowpass_filter(
        #     com_vert_v, kinematic_filt_freq, 1 / np.diff(t).mean(), order=4)
        # Identify local minima peaks in vertical velocity (inverse to get peaks)
        peaks = find_peaks(com_vert_v * -1,
                           distance=50)[0]  # distance seems reasonable based on sampling rate and step freq
        # plt.plot(com_vert_v_filt)
        # plt.scatter(peaks, com_vert_v_filt[peaks], color='green', s=25, zorder=3)
        # Identify times when peaks occur relative to start_time
        peak_times = [t[ii] for ii in peaks]
        peak_time_diffs = [ii - (right_contact_time + 0.10) for ii in
                           peak_times]  # add a small buffer here to account for potential errors
        peak_index = peak_time_diffs.index(min(vv for vv in peak_time_diffs if vv > 0))
        # Identify touchdown time as first following start time
        end_time = peak_times[peak_index]

    # Check start time vs. end time
    if start_time > end_time:
        raise ValueError('Start time identified after end time. Some sort of error in step detection...')

    # Check if end time is much greater than start time
    if end_time - start_time > 2.0:
        raise ValueError('End time greater than 2 seconds after start. Check event identification...')

    # Add 50 millisecond buffer on start and end time to avoid potentially erroneous initial and final states
    # See De Groote et al. (2016)
    start_time -= 0.05
    end_time += 0.05

    # =========================================================================
    # Set-up and run the tracking simulation
    # =========================================================================

    # Set-up the model for the tracking simulation
    # -------------------------------------------------------------------------

    # TODO: tendon compliance is currently ignored in the scaled model --- switch back on for some?

    # Construct a model processor to use with the tool
    model_proc = osim.ModelProcessor(os.path.join('..', '..', '..', '..', '..', 'data', dataset, participant, 'scaling',
                                                  f'{participant}_{model_type}.osim'))

    # Append external loads
    model_proc.append(osim.ModOpAddExternalLoads(f'{trial_name}_grf.xml'))

    # Increase muscle isometric force by a scaling factor to deal with potentially higher muscle forces
    model_proc.append(osim.ModOpScaleMaxIsometricForce(1.5))

    # Scale active force curve width
    model_proc.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))

    # Process model for further edits
    opt_model = model_proc.process()

    # Add coordinate actuators to model
    # Get model coordinates to check against actuators being added
    model_coordinates = [
        opt_model.getCoordinateSet().get(ii).getName() for ii in range(opt_model.getCoordinateSet().getSize())]
    # Add actuators
    for coordinate in act_forces:
        #Check if in model
        if coordinate in model_coordinates:
            # Create actuator
            actu = osim.CoordinateActuator()
            # Set name
            actu.setName(f'{coordinate}_{act_forces[coordinate]["actuatorType"]}')
            # Set coordinate
            actu.setCoordinate(opt_model.updCoordinateSet().get(coordinate))
            # Set optimal force
            actu.setOptimalForce(act_forces[coordinate]['optForce'])
            # Set min and max control
            actu.setMinControl(np.inf * -1)
            actu.setMaxControl(np.inf * 1)
            # Append to model force set
            opt_model.updForceSet().cloneAndAppend(actu)

    # # Adjust limits on muscle activations to produce necessary force if needed
    # for muscle_ind in range(opt_model.getMuscles().getSize()):
    #     musc = opt_model.getMuscles().get(muscle_ind)
    #     musc.setMaxControl(np.inf)
    #     osim.DeGrooteFregly2016Muscle().safeDownCast(musc).get_fiber_damping()
    #     # # Option for elastic tendons on plantarflexor muscles
    #     # # TODO: does this work?
    #     # if musc.getName() in ['gaslat_r', 'gasmed_r', 'soleus_r']:
    #     #     musc.set_ignore_tendon_compliance(False)
    #     #     osim.DeGrooteFregly2016Muscle().safeDownCast(musc).set_tendon_compliance_dynamics_mode('implicit')
    #     # Option to reduce fiber damping
    #     osim.DeGrooteFregly2016Muscle().safeDownCast(musc).set_fiber_damping(1.0e-3)

    # Finalise model connections
    opt_model.finalizeConnections()
    opt_model.initSystem()

    # Print model to file in tracking directory
    opt_model.printToXML(f'{participant}_{trial_label}_dynamic-optimisation_{model_type}.osim')

    # Clean up kinematic data for tracking guess
    # -------------------------------------------------------------------------

    # Load in kinematic data to table processor
    ik_proc = osim.TableProcessor(f'{participant}_{trial_label}_ik_{model_type}_filt.mot')

    # Append operators to derive speeds
    ik_proc.append(osim.TabOpAppendCoordinateValueDerivativesAsSpeeds())

    # Process table to get data
    ik_data = ik_proc.process(opt_model)

    # Trim kinematic data to start and end times
    ik_data.trim(start_time, end_time)

    # Write to file
    osim.STOFileAdapter().write(ik_data, f'{participant}_{trial_label}_ik-initial-guess_{model_type}.sto')

    # Set up tracking simulation
    # -------------------------------------------------------------------------

    # Create tracking tool
    track = osim.MocoTrack()
    track.setName(f'{participant}_{trial_label}_dynamic-optimisation_{model_type}')

    # Set model
    track_model_proc = osim.ModelProcessor(f'{participant}_{trial_label}_dynamic-optimisation_{model_type}.osim')
    track.setModel(track_model_proc)
    track_model = track_model_proc.process()
    track_model.initSystem()

    # Set the marker reference file and settings
    track.setMarkersReferenceFromTRC(f'{trial_name}.trc')
    track.set_markers_global_tracking_weight(globalMarkerTrackingWeight)

    # Set individual marker weights
    marker_weight_set = osim.MocoWeightSet()
    for marker in marker_weights.keys():
        marker_weight_set.cloneAndAppend(osim.MocoWeight(marker, marker_weights[marker]['weight']))
    track.set_markers_weight_set(marker_weight_set)

    # # Set state tracking reference
    # states_table_proc = osim.TableProcessor(ik_data)
    # track.setStatesReference(states_table_proc)

    # Set to ignore unused columns
    track.set_allow_unused_references(True)

    # Set the timings
    # Use times from IK file to avoid rounding issues as this has been cropped to data region of interest
    track.set_initial_time(ik_data.getIndependentColumn()[0])
    track.set_final_time(ik_data.getIndependentColumn()[-1])

    # Initialise to a Moco study and problem to finalise
    # -------------------------------------------------------------------------

    # Get study and problem
    study = track.initialize()
    problem = study.updProblem()

    # Update control effort goal
    # -------------------------------------------------------------------------

    # Get a reference to the MocoControlCost goal and set parameters
    effort = osim.MocoControlGoal.safeDownCast(problem.updGoal('control_effort'))
    effort.setWeight(globalControlEffortWeight)
    effort.setExponent(2)

    # Update individual weights in control effort goal
    # Put higher weight on residual use
    effort.setWeightForControlPattern('/forceset/.*_residual', 10.0)
    # Put heavy weight on the reserve actuators
    effort.setWeightForControlPattern('/forceset/.*_reserve', 5.0)
    # Put low weight on torque actuators
    # Reduced this further after increasing overall control effort weight
    effort.setWeightForControlPattern('/forceset/.*_torque', 1e-02)
    # Use standard weight for muscle controls
    # This probably doesn't change default but provides an option to set
    effort.setWeightForControlPattern('/forceset/.*_r', 1.0)

    # Get and modify state weight tracking goal
    # -------------------------------------------------------------------------

    # # Get a reference to the states tracking goal
    # tracking = osim.MocoStateTrackingGoal.safeDownCast(problem.updGoal('state_tracking'))
    #
    # # Set state weights from dictionary values
    # for coord_ind in range(track_model.updCoordinateSet().getSize()):
    #     # Get the name and absolute path of the coordinate
    #     coord_name = track_model.updCoordinateSet().get(coord_ind).getName()
    #     coord_path = track_model.updCoordinateSet().get(coord_ind).getAbsolutePathString()
    #     # Check if state is in tracking dictionary
    #     if coord_name in state_weights.keys():
    #         # Set weight for state value
    #         tracking.setWeightForState(f'{coord_path}/value', state_weights[coord_name])
    #         # Set weight for state speed
    #         tracking.setWeightForState(f'{coord_path}/speed', state_weights[coord_name] * speeds_scale)

    # Add initial activation goal to avoid muscles from cheating at start of simulation
    # -------------------------------------------------------------------------

    # For all muscles with activation dynamics, the initial activation and initial excitation should be the same.
    # Without this goal, muscle activation may undesirably start at its maximum possible value as only excitation
    # is penalised
    initial_act = osim.MocoInitialActivationGoal('initial_activation')
    problem.addGoal(initial_act)

    # Set wider bounds on muscle activations and controls to assist with convergence
    # -------------------------------------------------------------------------

    # Muscle bounds
    # problem.setStateInfoPattern('/forceset/.*/normalized_tendon_force', [0, 1.5], [], [])  # TODO: reactivate if using elastic tendons
    problem.setStateInfoPattern('/forceset/.*/activation', [0.01, 1.0], [], [])  # allow muscles to over-activate rather than increasing force?
    problem.setControlInfoPattern('/forceset/.*_r', [1e-3, 1.0], [], [])  # TODO: shifted this from 2.0 to 1.0 max limit; allow control signals to over-activate rather than increasing force

    # Set kinematic bounds
    # -------------------------------------------------------------------------

    # Set bounds on joint coordinate values using the states tracking data as a reference
    # Initial bounds are set to be within 20% of the range for their tracking values
    # Final bounds are not set as they are more so dictated by the dynamics of the problem (see: https://simtk.org/plugins/phpBB/viewtopicPhpbb.php?f=1815&t=19781&p=0&start=0&view=&sid=97392e0361ee4ab8a89d4d6cd7134afe)
    # Maximum and minimum bounds are set to be +/- 20% of the range for the coordinate

    # Create a dictionary to store initial bounds for coordinates
    initial_coord_bounds = {}
    coord_bounds = {}

    # Loop through coordinates
    for coord_ind in range(track_model.updCoordinateSet().getSize()):
        # Get the name and absolute path of the coordinate
        coord_name = track_model.updCoordinateSet().get(coord_ind).getName()
        coord_path = track_model.updCoordinateSet().get(coord_ind).getAbsolutePathString()
        # Check to skip constrained joints
        if not coord_name.endswith('_beta'):
            # Get initial, min, max and range values
            initial_val = ik_data.getDependentColumn(f'{coord_path}/value').to_numpy()[0]
            min_val = ik_data.getDependentColumn(f'{coord_path}/value').to_numpy().min()
            max_val = ik_data.getDependentColumn(f'{coord_path}/value').to_numpy().max()
            val_range = np.ptp(ik_data.getDependentColumn(f'{coord_path}/value').to_numpy())
            # Set bounds in dictionary
            initial_coord_bounds[coord_name] = [initial_val - (val_range * 0.20), initial_val + (val_range * 0.20)]
            coord_bounds[coord_name] = [min_val - (val_range * 0.20), max_val + (val_range * 0.20)]

    # Check that any bounds do not exceed model ranges
    for coord_name in coord_bounds.keys():
        if len(coord_bounds[coord_name]) > 0 and len(initial_coord_bounds[coord_name]) > 0:
            # Replace lower initial or overall value if less than model range min bounds
            if initial_coord_bounds[coord_name][0] < opt_model.getCoordinateSet().get(coord_name).getRangeMin():
                initial_coord_bounds[coord_name][0] = opt_model.getCoordinateSet().get(coord_name).getRangeMin()
            if coord_bounds[coord_name][0] < opt_model.getCoordinateSet().get(coord_name).getRangeMin():
                coord_bounds[coord_name][0] = opt_model.getCoordinateSet().get(coord_name).getRangeMin()
            # Replace upper initial or overall value if less than model range min bounds
            if initial_coord_bounds[coord_name][1] > opt_model.getCoordinateSet().get(coord_name).getRangeMax():
                initial_coord_bounds[coord_name][1] = opt_model.getCoordinateSet().get(coord_name).getRangeMax()
            if coord_bounds[coord_name][1] > opt_model.getCoordinateSet().get(coord_name).getRangeMax():
                coord_bounds[coord_name][1] = opt_model.getCoordinateSet().get(coord_name).getRangeMax()

    # Check that any initial bounds do not exceed total coordinate bounds
    for coord_name in coord_bounds.keys():
        if len(coord_bounds[coord_name]) > 0 and len(initial_coord_bounds[coord_name]) > 0:
            # Replace lower initial value if less than total bounds
            if initial_coord_bounds[coord_name][0] < coord_bounds[coord_name][0]:
                initial_coord_bounds[coord_name][0] = coord_bounds[coord_name][0]
            # Replace upper initial value if greater than total bounds
            if initial_coord_bounds[coord_name][1] > coord_bounds[coord_name][1]:
                initial_coord_bounds[coord_name][1] = coord_bounds[coord_name][1]

    # Set coordinate value bounds in problem
    for coord_name in coord_bounds.keys():
        # Get coordinate path
        coord_path = track_model.updCoordinateSet().get(coord_name).getAbsolutePathString()
        # Set in problem
        problem.setStateInfo(f'{coord_path}/value',
                             coord_bounds[coord_name],
                             initial_coord_bounds[coord_name],
                             [])

    # Define and configure the solver
    # -------------------------------------------------------------------------

    # Get the solver
    solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())

    # Solver settings
    solver.set_optim_max_iterations(3000)
    solver.set_num_mesh_intervals(mesh_interval_dyn)
    solver.set_optim_constraint_tolerance(1.0e-0)  # TODO: higher than desirable, but helps with convergence
    solver.set_optim_convergence_tolerance(1.0e-3) # TODO: higher than desirable, but helps with convergence

    # TODO: useful addition from OpenSim 4.6
    solver.set_kinematic_constraint_method('Bordalba2023')

    # Smoothness criterion
    solver.set_multibody_dynamics_mode('implicit')
    solver.set_minimize_implicit_multibody_accelerations(True)
    solver.set_implicit_multibody_accelerations_weight(globalAccelMinWeight)

    # Reset problem
    solver.resetProblem(problem)

    # Get the initial guess
    guess = solver.getGuess()

    # Get and resample the guess to match IK
    guess.resampleWithNumTimes(ik_data.getNumRows())

    # Insert the desired values from IK
    for col in guess.getStateNames():
        if col in ik_data.getColumnLabels():
            guess.setState(col, ik_data.getDependentColumn(col).to_numpy())

    # Generate accelerations from coordinate speeds for implicit mode
    guess.generateAccelerationsFromSpeeds()

    # Set generic zero guess for forceset values
    # This is if we want to use a default generic guess for forces rather than the consistent guess
    for state_name in guess.getStateNames():
        if state_name.startswith('/forceset/'):
            guess.setState(state_name, np.zeros(guess.getNumTimes()))
    for control_name in guess.getControlNames():
        if control_name.startswith('/forceset/'):
            guess.setControl(control_name, np.zeros(guess.getNumTimes()))

    # # Set relevant elements in guess using consistent pre-solved simulation
    # # Read in consistent initial guess for speed
    # consistent_guess = osim.MocoTrajectory(os.path.join('..', '..', '..', '..', '..',
                                                        # 'guess', dataset, f'{trial_label}_consistent-guess.sto'))
    # # Resample current guess to the consistent guess (should match mesh interval)
    # guess.resampleWithNumTimes(consistent_guess.getNumTimes())
    # # Look for relevant states in consistent guess to fill
    # for state_name in guess.getStateNames():
        # if state_name.endswith('/activation'):
            # guess.setState(state_name, consistent_guess.getState(state_name).to_numpy())
    # # Look for relevant controls in consistent guess to fill
    # for control_name in guess.getControlNames():
        # if control_name.endswith('_r') or control_name.endswith('_torque'):
            # guess.setControl(control_name, consistent_guess.getControl(control_name).to_numpy())
        # # Otherwise set to zero for reserve and residual actuators
        # else:
            # guess.setControl(control_name, np.zeros(guess.getNumTimes()))

    # Write to file for reference
    guess.write(f'{participant}_{trial_label}_initial-guess_{model_type}.sto')

    # Set guess in solver
    solver.setGuessFile(f'{participant}_{trial_label}_initial-guess_{model_type}.sto')

    # Reset problem to check any issues
    solver.resetProblem(problem)

    # Solve the problem
    # -------------------------------------------------------------------------

    # Set-up timer to track computation time
    computation_start = time.time()

    # Solve!
    tracking_solution = study.solve()

    # End computation timer and record
    computation_run_time = round(time.time() - computation_start, 2)

    # # Option to visualise solution
    # study.visualize(tracking_solution)

    # Save files and finalize
    # -------------------------------------------------------------------------

    # Write solution to file
    if tracking_solution.isSealed():
        tracking_solution.unseal()
    tracking_solution.write(f'{participant}_{trial_label}_dynamic-optimisation_{model_type}_solution.sto')

    # Save a dictionary storing computational time
    computation = {'time_s': computation_run_time,
                   'note': f'Dynamic optimisation computation time for {participant} {trial_label}'}
    with open(f'{participant}_{trial_label}_dynamic-optimisation_computation-time.pkl', 'wb') as pkl_file:
        pickle.dump(computation, pkl_file)

    # # Remove initial tracked states and markers file
    os.remove(f'{participant}_{trial_label}_dynamic-optimisation_{model_type}_tracked_markers.sto')
    # os.remove(f'{participant}_{trial_label}_dynamic-optimisation_{model_type}_tracked_states.sto')

    # Extract muscle forces from solution
    output_paths = osim.StdVectorString()
    output_paths.append('.*tendon_force')
    output_paths.append('.*fiber_force')
    muscle_force_table = osim.analyze(opt_model,
                                      tracking_solution.exportToStatesTable(),
                                      tracking_solution.exportToControlsTable(),
                                      output_paths)
    # Write to file
    osim.STOFileAdapter().write(muscle_force_table,
                                f'{participant}_{trial_label}_dynamic-optimisation_{model_type}_muscle-forces.sto')

    # Extract PF joint reaction forces if using complex model
    if model_type == 'complex':
        # Set outputs
        output_paths = osim.StdVectorString()
        output_paths.append('.*patellofemoral_r.*reaction_on_child')
        jrf_table = osim.analyzeSpatialVec(opt_model,
                                           tracking_solution.exportToStatesTable(),
                                           tracking_solution.exportToControlsTable(),
                                           output_paths).flatten()
        # Write to file
        osim.STOFileAdapter().write(jrf_table,
                                    f'{participant}_{trial_label}_dynamic-optimisation_{model_type}_pfjrf.sto')

    # Return to home directory
    os.chdir(home_dir)

    # Print out to console as a bookmark in any log file
    print(f'{"*"*10} FINISHED DYNAMIC OPTIMISATION FOR {participant} {trial_label} {"*"*10}')


# Run a static optimisation
# -------------------------------------------------------------------------
def run_static_optimisation(model_type):

    """

    This function runs the muscle-driven static optimisation simulations, with the goal
    here to generate a muscle-driven simulation of the gait cycle that minimises
    squared muscle activations (i.e. standard static optimisations). It uses the kinematics
    from the prior dynamic optimisation pipeline.

    """

    # Do an initial input check as the function won't work with an incorrect model type
    if model_type != 'complex' and model_type != 'simple':
        raise ValueError('model_type variable must be string of "complex" or "simple".')

    # =========================================================================
    # Set-up files and parameters for simulation
    # =========================================================================

    # Files and settings
    # -------------------------------------------------------------------------

    # Create the folder for simulation
    os.makedirs(os.path.join('..','..','simulations',dataset,participant,trial_label,'static_optimisation'), exist_ok=True)

    # Get trial name from selected trials
    # Order of trial label is direction, stair height, transition leg
    trial_split = trial_label.split('_')
    trial_data = selected_trials.loc[
        (selected_trials['participant'] == participant) &
        (selected_trials['direction'] == trial_split[0]) &
        (selected_trials['stair_height'] == int(trial_split[1])) &
        (selected_trials['trans_leg'] == trial_split[2]),]
    trial_name = trial_data['trial_name'].values[0]

    # Navigate to simulation folder for ease of use
    home_dir = os.getcwd()
    os.chdir(os.path.join('..','..','simulations',dataset,participant,trial_label,'static_optimisation'))

    # Copy GRF data to simulation directory
    shutil.copyfile(
        os.path.join('..', '..', '..', '..', '..', 'data', dataset, participant, f'{trial_name}_grf.mot'),
        f'{trial_name}_grf.mot')

    # Create associated external loads file for use in simulation
    ext_loads = osim.ExternalLoads()
    # This is a simplified external loads file that only allocates force plate data over the period of interest
    # It also assumes that the 'rl' transition leg trials are being used
    # For ascent trials
    if trial_split[0] == 'ascent':
        # FP5 gets allocated to left leg on initial step
        left_force = osim.ExternalForce()
        left_force.setName('LeftGRF')
        left_force.setAppliedToBodyName('calcn_l')
        left_force.set_force_expressed_in_body('ground')
        left_force.set_point_expressed_in_body('ground')
        left_force.set_force_identifier('FP5_v')
        left_force.set_point_identifier('FP5_p')
        left_force.set_torque_identifier('FP5_moment_')
        ext_loads.cloneAndAppend(left_force)
        # FP4 gets allocated to right leg on second step
        right_force = osim.ExternalForce()
        right_force.setName('RightGRF')
        right_force.setAppliedToBodyName('calcn_r')
        right_force.set_force_expressed_in_body('ground')
        right_force.set_point_expressed_in_body('ground')
        right_force.set_force_identifier('FP4_v')
        right_force.set_point_identifier('FP4_p')
        right_force.set_torque_identifier('FP4_moment_')
        ext_loads.cloneAndAppend(right_force)
    elif trial_split[0] == 'descent':
        # FP1 gets allocated to left leg on initial step
        left_force = osim.ExternalForce()
        left_force.setName('LeftGRF')
        left_force.setAppliedToBodyName('calcn_l')
        left_force.set_force_expressed_in_body('ground')
        left_force.set_point_expressed_in_body('ground')
        left_force.set_force_identifier('FP1_v')
        left_force.set_point_identifier('FP1_p')
        left_force.set_torque_identifier('FP1_moment_')
        ext_loads.cloneAndAppend(left_force)
        # FP2 gets allocated to right leg on second step
        right_force = osim.ExternalForce()
        right_force.setName('RightGRF')
        right_force.setAppliedToBodyName('calcn_r')
        right_force.set_force_expressed_in_body('ground')
        right_force.set_point_expressed_in_body('ground')
        right_force.set_force_identifier('FP2_v')
        right_force.set_point_identifier('FP2_p')
        right_force.set_torque_identifier('FP2_moment_')
        ext_loads.cloneAndAppend(right_force)
    else:
        # Throw error for not identifying trial correctly
        raise ValueError('Trial not identified as ascent or descent, check labels...')
    # Set datafile
    ext_loads.setDataFileName(f'{trial_name}_grf.mot')
    # Print to file
    ext_loads.printToXML(f'{trial_name}_grf.xml')

    # Copy states from the dynamic optimisation
    dynamic_opt_traj = osim.MocoTrajectory(os.path.join(
        '..', 'dynamic_optimisation', f'{participant}_{trial_label}_dynamic-optimisation_{model_type}_solution.sto'))
    states_table_proc = osim.TableProcessor(dynamic_opt_traj.exportToStatesTable())
    states_table_proc.append(osim.TabOpLowPassFilter(kinematic_filt_freq))
    states_table = states_table_proc.process()
    states_table.trim(dynamic_opt_traj.getInitialTime(), dynamic_opt_traj.getFinalTime())
    osim.STOFileAdapter().write(states_table, f'{participant}_{trial_label}_states.sto')

    # Check for simple model and need to invert knee angle in states
    if model_type == 'simple':
        # Read in data
        states_data = osim.TimeSeriesTable(f'{participant}run{speed}_states.sto')
        # Create new columns for values and speeds. Remove the existing columns
        adjust_cols = ['/jointset/walker_knee_l/knee_angle_l/value',
                       '/jointset/walker_knee_l/knee_angle_l/speed',
                       '/jointset/walker_knee_r/knee_angle_r/value',
                       '/jointset/walker_knee_r/knee_angle_r/speed',
                       ]
        for col in adjust_cols:
            # Create and append the new column
            states_data.appendColumn(col.replace('/walker_knee_', '/knee_'),
                                     osim.Vector().createFromMat(states_data.getDependentColumn(col).to_numpy() * -1))
            # Remove the old one
            states_data.removeColumn(col)
        # Remove the patellofemoral joint columns
        for col in states_data.getColumnLabels():
            if 'patellofemoral_' in col:
                states_data.removeColumn(col)
        # Save new states data to file
        osim.STOFileAdapter().write(states_data, f'{participant}_{trial_label}_states.sto')

    # Set actuator forces to support simulation
    act_forces = {'pelvis_tx': {'actuatorType': 'residual', 'optForce': 5},
                  'pelvis_ty': {'actuatorType': 'residual', 'optForce': 5},
                  'pelvis_tz': {'actuatorType': 'residual', 'optForce': 5},
                  'pelvis_tilt': {'actuatorType': 'residual', 'optForce': 2.5},
                  'pelvis_list': {'actuatorType': 'residual', 'optForce': 2.5},
                  'pelvis_rotation': {'actuatorType': 'residual', 'optForce': 2.5},
                  'hip_flexion_r': {'actuatorType': 'reserve', 'optForce': 3.0},
                  'hip_adduction_r': {'actuatorType': 'reserve', 'optForce': 2.0},
                  'hip_rotation_r': {'actuatorType': 'reserve', 'optForce': 1.0},
                  'knee_angle_r': {'actuatorType': 'reserve', 'optForce': 3.0},
                  'ankle_angle_r': {'actuatorType': 'reserve', 'optForce': 2.0},
                  'subtalar_angle_r': {'actuatorType': 'reserve', 'optForce': 1.0},
                  'hip_flexion_l': {'actuatorType': 'torque', 'optForce': 300.0},
                  'hip_adduction_l': {'actuatorType': 'torque', 'optForce': 200.0},
                  'hip_rotation_l': {'actuatorType': 'torque', 'optForce': 100.0},
                  'knee_angle_l': {'actuatorType': 'torque', 'optForce': 300.0},
                  'ankle_angle_l': {'actuatorType': 'torque', 'optForce': 200.0},
                  'subtalar_angle_l': {'actuatorType': 'torque', 'optForce': 100.0},
                  }

    # Prepare model for static optimisation
    # -------------------------------------------------------------------------

    # Construct a model processor to use with the tool
    model_proc = osim.ModelProcessor(
        os.path.join('..', '..', '..', '..', '..', 'data', dataset, participant, 'scaling',
                     f'{participant}_{model_type}.osim'))

    # Increase muscle isometric force by a scaling factor to deal with potentially higher muscle forces
    model_proc.append(osim.ModOpScaleMaxIsometricForce(1.5))

    # Scale active force curve width
    model_proc.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))

    # Process model for further edits
    opt_model = model_proc.process()

    # Add coordinate actuators to model
    for coordinate in act_forces:
        # Create actuator
        actu = osim.CoordinateActuator()
        # Set name
        actu.setName(f'{coordinate}_{act_forces[coordinate]["actuatorType"]}')
        # Set coordinate
        actu.setCoordinate(opt_model.updCoordinateSet().get(coordinate))
        # Set optimal force
        actu.setOptimalForce(act_forces[coordinate]['optForce'])
        # Set min and max control
        actu.setMinControl(np.inf * -1)
        actu.setMaxControl(np.inf * 1)
        # Append to model force set
        opt_model.updForceSet().cloneAndAppend(actu)

    # Adjust limits on muscle activations to produce necessary force if needed
    for muscle_ind in range(opt_model.getMuscles().getSize()):
        musc = opt_model.getMuscles().get(muscle_ind)
        musc.setMaxControl(np.inf)

    # Finalise model connections
    opt_model.finalizeConnections()

    # Print model to file in tracking directory
    opt_model.printToXML(f'{participant}_{trial_label}_static_optimisation_{model_type}.osim')

    # Set-up static optimisation
    # -------------------------------------------------------------------------

    # Create the analyze tool by reading in the pre-created utility
    analyzeTool = osim.AnalyzeTool(
        os.path.join(os.path.join('..', '..', '..', '..', '..', 'utilities',
                                  f'static_optimisation_{model_type}.xml')), False)

    # Set tool name
    analyzeTool.setName(f'{participant}_{trial_label}')

    # Set the model file
    analyzeTool.setModelFilename(f'{participant}_{trial_label}_static_optimisation_{model_type}.osim')

    # Set times for analysis
    analyzeTool.setStartTime(osim.TimeSeriesTable(f'{participant}_{trial_label}_states.sto').getIndependentColumn()[0])
    analyzeTool.setFinalTime(osim.TimeSeriesTable(f'{participant}_{trial_label}_states.sto').getIndependentColumn()[-1])

    # Set states file
    analyzeTool.setStatesFileName(f'{participant}_{trial_label}_states.sto')

    # Set external loads
    analyzeTool.setExternalLoadsFileName(f'{trial_name}_grf.xml')

    # Save tool
    analyzeTool.printToXML(f'{participant}_{trial_label}_setup-static-optimisation_{model_type}.xml')

    # Run static optimisation
    # -------------------------------------------------------------------------

    # Set-up timer to track computation time
    computation_start = time.time()

    # Read the tool back in as this sometimes helps avoid Python crashing
    runAnalysis = osim.AnalyzeTool(f'{participant}_{trial_label}_setup-static-optimisation_{model_type}.xml')

    # Run the tool
    runAnalysis.run()

    # End computation timer and record
    computation_run_time = round(time.time() - computation_start, 2)

    # Save a dictionary storing computational time
    computation = {'time_s': computation_run_time,
                   'note': f'Static optimisation {model_type} model computation time for {participant} {trial_label}'}
    with open(f'{participant}_{trial_label}_static-optimisation_{model_type}_computation-time.pkl', 'wb') as pkl_file:
        pickle.dump(computation, pkl_file)

    # Return to home directory
    os.chdir(home_dir)

    # Print out to console as a bookmark in any log file
    print(f'{"*" * 10} FINISHED STATIC OPTIMISATION FOR {participant} {trial_label} WITH {model_type.upper()} MODEL {"*" * 10}')


# Run inverse dynamics
# -------------------------------------------------------------------------
def run_inverse_dynamics(model_type):

    """

    This function runs the inverse dynamics step to obtain joint moments to
    use in the equation based approach for estimating PFJRF. It uses the kinematics
    from the prior dynamic optimisation pipeline.

    """

    # Do an initial input check as the function won't work with an incorrect model type
    if model_type != 'complex' and model_type != 'simple':
        raise ValueError('model_type variable must be string of "complex" or "simple".')

    # =========================================================================
    # Set-up files and parameters for simulation
    # =========================================================================

    # Files and settings
    # -------------------------------------------------------------------------

    # Create the folder for simulation
    os.makedirs(os.path.join('..', '..', 'simulations', dataset, participant, trial_label, 'inverse_dynamics'),
                exist_ok=True)

    # Get trial name from selected trials
    # Order of trial label is direction, stair height, transition leg
    trial_split = trial_label.split('_')
    trial_data = selected_trials.loc[
        (selected_trials['participant'] == participant) &
        (selected_trials['direction'] == trial_split[0]) &
        (selected_trials['stair_height'] == int(trial_split[1])) &
        (selected_trials['trans_leg'] == trial_split[2]),]
    trial_name = trial_data['trial_name'].values[0]

    # Navigate to simulation folder for ease of use
    home_dir = os.getcwd()
    os.chdir(os.path.join('..','..','simulations',dataset,participant,trial_label,'inverse_dynamics'))

    # Copy GRF data to simulation directory
    shutil.copyfile(
        os.path.join('..', '..', '..', '..', '..', 'data', dataset, participant, f'{trial_name}_grf.mot'),
        f'{trial_name}_grf.mot')

    # Create associated external loads file for use in simulation
    ext_loads = osim.ExternalLoads()
    # This is a simplified external loads file that only allocates force plate data over the period of interest
    # It also assumes that the 'rl' transition leg trials are being used
    # For ascent trials
    if trial_split[0] == 'ascent':
        # FP5 gets allocated to left leg on initial step
        left_force = osim.ExternalForce()
        left_force.setName('LeftGRF')
        left_force.setAppliedToBodyName('calcn_l')
        left_force.set_force_expressed_in_body('ground')
        left_force.set_point_expressed_in_body('ground')
        left_force.set_force_identifier('FP5_v')
        left_force.set_point_identifier('FP5_p')
        left_force.set_torque_identifier('FP5_moment_')
        ext_loads.cloneAndAppend(left_force)
        # FP4 gets allocated to right leg on second step
        right_force = osim.ExternalForce()
        right_force.setName('RightGRF')
        right_force.setAppliedToBodyName('calcn_r')
        right_force.set_force_expressed_in_body('ground')
        right_force.set_point_expressed_in_body('ground')
        right_force.set_force_identifier('FP4_v')
        right_force.set_point_identifier('FP4_p')
        right_force.set_torque_identifier('FP4_moment_')
        ext_loads.cloneAndAppend(right_force)
    elif trial_split[0] == 'descent':
        # FP1 gets allocated to left leg on initial step
        left_force = osim.ExternalForce()
        left_force.setName('LeftGRF')
        left_force.setAppliedToBodyName('calcn_l')
        left_force.set_force_expressed_in_body('ground')
        left_force.set_point_expressed_in_body('ground')
        left_force.set_force_identifier('FP1_v')
        left_force.set_point_identifier('FP1_p')
        left_force.set_torque_identifier('FP1_moment_')
        ext_loads.cloneAndAppend(left_force)
        # FP2 gets allocated to right leg on second step
        right_force = osim.ExternalForce()
        right_force.setName('RightGRF')
        right_force.setAppliedToBodyName('calcn_r')
        right_force.set_force_expressed_in_body('ground')
        right_force.set_point_expressed_in_body('ground')
        right_force.set_force_identifier('FP2_v')
        right_force.set_point_identifier('FP2_p')
        right_force.set_torque_identifier('FP2_moment_')
        ext_loads.cloneAndAppend(right_force)
    else:
        # Throw error for not identifying trial correctly
        raise ValueError('Trial not identified as ascent or descent, check labels...')
    # Set datafile
    ext_loads.setDataFileName(f'{trial_name}_grf.mot')
    # Print to file
    ext_loads.printToXML(f'{trial_name}_grf.xml')

    # Copy states from the dynamic optimisation
    dynamic_opt_traj = osim.MocoTrajectory(os.path.join(
        '..', 'dynamic_optimisation', f'{participant}_{trial_label}_dynamic-optimisation_{model_type}_solution.sto'))
    coord_table_proc = osim.TableProcessor(dynamic_opt_traj.exportToValuesTable())
    coord_table_proc.append(osim.TabOpLowPassFilter(kinematic_filt_freq))
    coord_table = coord_table_proc.process()
    coord_table.trim(dynamic_opt_traj.getInitialTime(), dynamic_opt_traj.getFinalTime())

    # Inverse dynamics requires base coordinate names, so fix this up
    new_cols = []
    for col_label in coord_table.getColumnLabels():
        new_cols.append(col_label.split('/')[-2])
    coord_table.setColumnLabels(new_cols)

    # Write coordinates to file
    osim.STOFileAdapter().write(coord_table, f'{participant}_{trial_label}_coordinates.sto')

    # # Check for simple model and need to invert knee angle in states
    # if model_type == 'simple':
        # TODO: typical replacement approach won't work as column labels are now the same...

    # Prepare model for inverse dynamics
    # -------------------------------------------------------------------------

    # Construct a model processor to use with the tool
    model_proc = osim.ModelProcessor(
        os.path.join('..', '..', '..', '..', '..', 'data', dataset, participant, 'scaling',
                     f'{participant}_{model_type}.osim'))

    # Remove muscles for simplicity
    model_proc.append(osim.ModOpRemoveMuscles())

    # Process model
    opt_model = model_proc.process()

    # Finalise model connections
    opt_model.finalizeConnections()

    # Print model to file in tracking directory
    opt_model.printToXML(f'{participant}_{trial_label}_inverse_dynamics_{model_type}.osim')

    # Set-up inverse dynamics
    # -------------------------------------------------------------------------

    # Create the analyze tool by reading in the pre-created utility
    idTool = osim.InverseDynamicsTool()

    # Set tool name
    idTool.setName(f'{participant}_{trial_label}')

    # Set the model file
    idTool.setModelFileName(f'{participant}_{trial_label}_inverse_dynamics_{model_type}.osim')

    # Set times for analysis
    idTool.setStartTime(osim.TimeSeriesTable(f'{participant}_{trial_label}_coordinates.sto').getIndependentColumn()[0])
    idTool.setEndTime(osim.TimeSeriesTable(f'{participant}_{trial_label}_coordinates.sto').getIndependentColumn()[-1])

    # Set states file
    idTool.setCoordinatesFileName(f'{participant}_{trial_label}_coordinates.sto')

    # Set external loads
    idTool.setExternalLoadsFileName(f'{trial_name}_grf.xml')

    # Set output filename
    idTool.setOutputGenForceFileName(f'{participant}_{trial_label}_inverse_dynamics_results.sto')

    # Set forces to exclude (muscles just in case, even though there are none)
    exclude_forces = osim.ArrayStr()
    exclude_forces.append('muscles')
    idTool.setExcludedForces(exclude_forces)

    # Save tool
    idTool.printToXML(f'{participant}_{trial_label}_setup-inverse-dynamics_{model_type}.xml')

    # Run inverse dynamics
    # -------------------------------------------------------------------------

    # Set-up timer to track computation time
    computation_start = time.time()

    # Read the tool back in as this sometimes helps avoid Python crashing
    runID = osim.InverseDynamicsTool(f'{participant}_{trial_label}_setup-inverse-dynamics_{model_type}.xml')

    # Run the tool
    runID.run()

    # End computation timer and record
    computation_run_time = round(time.time() - computation_start, 2)

    # Save a dictionary storing computational time
    computation = {'time_s': computation_run_time,
                   'note': f'Inverse dynamics {model_type} model computation time for {participant} {speed}'}
    with open(f'{participant}_{trial_label}_inverse-dynamics_{model_type}_computation-time.pkl', 'wb') as pkl_file:
        pickle.dump(computation, pkl_file)

    # Smooth inverse dyanmics forces for later analyses
    id_table_proc = osim.TableProcessor(f'{participant}_{trial_label}_inverse_dynamics_results.sto')
    id_table_proc.append(osim.TabOpLowPassFilter(kinematic_filt_freq))
    id_table = id_table_proc.process()
    id_table.trim(runID.getStartTime(), runID.getEndTime())
    osim.STOFileAdapter().write(id_table, f'{participant}_{trial_label}_inverse_dynamics_results.sto')

    # Return to home directory
    os.chdir(home_dir)

    # Print out to console as a bookmark in any log file
    print(f'{"*" * 10} FINISHED INVERSE DYNAMICS FOR {participant} {trial_label} WITH {model_type.upper()} MODEL {"*" * 10}')


# Run inverse dynamics via a torque-driven inverse simulation
# -------------------------------------------------------------------------
def run_inverse_dynamics_optim(model_type):

    """

    This function runs the inverse dynamics step to obtain joint moments to
    use in the equation based approach for estimating PFJRF. It uses the kinematics
    from the prior dynamic optimisation pipeline. It uses a MocoInverse direct collocation
    simulation to estimate joint torques from the kinematics, as this approach likely
    produces better residual outcomes.

    NOTE:
        Results from this are near identicaal to standard inverse dynamics, so this function
        isn't used (but is here for other uses maybe...).

    """

    # Do an initial input check as the function won't work with an incorrect model type
    if model_type != 'complex' and model_type != 'simple':
        raise ValueError('model_type variable must be string of "complex" or "simple".')

    # =========================================================================
    # Set-up files and parameters for simulation
    # =========================================================================

    # Files and settings
    # -------------------------------------------------------------------------

    # Create the folder for simulation
    os.makedirs(os.path.join('..', '..', 'simulations', dataset, participant, trial_label, 'inverse_dynamics'),
                exist_ok=True)

    # Get trial name from selected trials
    # Order of trial label is direction, stair height, transition leg
    trial_split = trial_label.split('_')
    trial_data = selected_trials.loc[
        (selected_trials['participant'] == participant) &
        (selected_trials['direction'] == trial_split[0]) &
        (selected_trials['stair_height'] == int(trial_split[1])) &
        (selected_trials['trans_leg'] == trial_split[2]),]
    trial_name = trial_data['trial_name'].values[0]

    # Navigate to simulation folder for ease of use
    home_dir = os.getcwd()
    os.chdir(os.path.join('..', '..', 'simulations', dataset, participant, trial_label, 'inverse_dynamics'))

    # Copy GRF data to simulation directory
    shutil.copyfile(
        os.path.join('..', '..', '..', '..', '..', 'data', dataset, participant, f'{trial_name}_grf.mot'),
        f'{trial_name}_grf.mot')

    # Create associated external loads file for use in simulation
    ext_loads = osim.ExternalLoads()
    # This is a simplified external loads file that only allocates force plate data over the period of interest
    # It also assumes that the 'rl' transition leg trials are being used
    # For ascent trials
    if trial_split[0] == 'ascent':
        # FP5 gets allocated to left leg on initial step
        left_force = osim.ExternalForce()
        left_force.setName('LeftGRF')
        left_force.setAppliedToBodyName('calcn_l')
        left_force.set_force_expressed_in_body('ground')
        left_force.set_point_expressed_in_body('ground')
        left_force.set_force_identifier('FP5_v')
        left_force.set_point_identifier('FP5_p')
        left_force.set_torque_identifier('FP5_moment_')
        ext_loads.cloneAndAppend(left_force)
        # FP4 gets allocated to right leg on second step
        right_force = osim.ExternalForce()
        right_force.setName('RightGRF')
        right_force.setAppliedToBodyName('calcn_r')
        right_force.set_force_expressed_in_body('ground')
        right_force.set_point_expressed_in_body('ground')
        right_force.set_force_identifier('FP4_v')
        right_force.set_point_identifier('FP4_p')
        right_force.set_torque_identifier('FP4_moment_')
        ext_loads.cloneAndAppend(right_force)
    elif trial_split[0] == 'descent':
        # FP1 gets allocated to left leg on initial step
        left_force = osim.ExternalForce()
        left_force.setName('LeftGRF')
        left_force.setAppliedToBodyName('calcn_l')
        left_force.set_force_expressed_in_body('ground')
        left_force.set_point_expressed_in_body('ground')
        left_force.set_force_identifier('FP1_v')
        left_force.set_point_identifier('FP1_p')
        left_force.set_torque_identifier('FP1_moment_')
        ext_loads.cloneAndAppend(left_force)
        # FP2 gets allocated to right leg on second step
        right_force = osim.ExternalForce()
        right_force.setName('RightGRF')
        right_force.setAppliedToBodyName('calcn_r')
        right_force.set_force_expressed_in_body('ground')
        right_force.set_point_expressed_in_body('ground')
        right_force.set_force_identifier('FP2_v')
        right_force.set_point_identifier('FP2_p')
        right_force.set_torque_identifier('FP2_moment_')
        ext_loads.cloneAndAppend(right_force)
    else:
        # Throw error for not identifying trial correctly
        raise ValueError('Trial not identified as ascent or descent, check labels...')
    # Set datafile
    ext_loads.setDataFileName(f'{trial_name}_grf.mot')
    # Print to file
    ext_loads.printToXML(f'{trial_name}_grf.xml')

    # Copy states from the dynamic optimisation
    dynamic_opt_traj = osim.MocoTrajectory(os.path.join(
        '..', 'dynamic_optimisation',
        f'{participant}_{trial_label}_dynamic-optimisation_{model_type}_solution.sto'))

    # Load in and convert to a values table
    coord_table_proc = osim.TableProcessor(dynamic_opt_traj.exportToValuesTable())
    coord_table_proc.append(osim.TabOpLowPassFilter(kinematic_filt_freq))
    coord_table = coord_table_proc.process()
    coord_table.trim(dynamic_opt_traj.getInitialTime(), dynamic_opt_traj.getFinalTime())
    coord_table.addTableMetaDataString('inDegrees','no')  # needs to be added to values tables to work with Moco

    # Write coordinates to file
    osim.STOFileAdapter().write(coord_table, f'{participant}_{trial_label}_coordinates.sto')

    # Prepare model for inverse dynamics
    # -------------------------------------------------------------------------

    # Construct a model processor to use with the tool
    model_proc = osim.ModelProcessor(
        os.path.join('..', '..', '..', '..', '..', 'data', dataset, participant, 'scaling',
                     f'{participant}_{model_type}.osim'))

    # Set actuator forces to support simulation
    act_forces = {'pelvis_tx': {'actuatorType': 'residual', 'optForce': 5},
                  'pelvis_ty': {'actuatorType': 'residual', 'optForce': 5},
                  'pelvis_tz': {'actuatorType': 'residual', 'optForce': 5},
                  'pelvis_tilt': {'actuatorType': 'residual', 'optForce': 2.5},
                  'pelvis_list': {'actuatorType': 'residual', 'optForce': 2.5},
                  'pelvis_rotation': {'actuatorType': 'torque', 'optForce': 2.5},
                  'hip_flexion_r': {'actuatorType': 'torque', 'optForce': 300.0},
                  'hip_adduction_r': {'actuatorType': 'torque', 'optForce': 200.0},
                  'hip_rotation_r': {'actuatorType': 'torque', 'optForce': 100.0},
                  'knee_angle_r': {'actuatorType': 'torque', 'optForce': 300.0},
                  'ankle_angle_r': {'actuatorType': 'torque', 'optForce': 200.0},
                  'hip_flexion_l': {'actuatorType': 'torque', 'optForce': 300.0},
                  'hip_adduction_l': {'actuatorType': 'torque', 'optForce': 200.0},
                  'hip_rotation_l': {'actuatorType': 'torque', 'optForce': 100.0},
                  'knee_angle_l': {'actuatorType': 'torque', 'optForce': 300.0},
                  'ankle_angle_l': {'actuatorType': 'torque', 'optForce': 200.0},
                  }

    # Remove muscles for simplicity
    model_proc.append(osim.ModOpRemoveMuscles())

    # Process model
    opt_model = model_proc.process()

    # Add coordinate actuators to model
    # Get model coordinates to check against actuators being added
    model_coordinates = [
        opt_model.getCoordinateSet().get(ii).getName() for ii in range(opt_model.getCoordinateSet().getSize())]
    # Add actuators
    for coordinate in act_forces:
        # Check if in model
        if coordinate in model_coordinates:
            # Create actuator
            actu = osim.CoordinateActuator()
            # Set name
            actu.setName(f'{coordinate}_{act_forces[coordinate]["actuatorType"]}')
            # Set coordinate
            actu.setCoordinate(opt_model.updCoordinateSet().get(coordinate))
            # Set optimal force
            actu.setOptimalForce(act_forces[coordinate]['optForce'])
            # Set min and max control
            actu.setMinControl(np.inf * -1)
            actu.setMaxControl(np.inf * 1)
            # Append to model force set
            opt_model.updForceSet().cloneAndAppend(actu)

    # Finalise model connections
    opt_model.finalizeConnections()

    # Print model to file in tracking directory
    opt_model.printToXML(f'{participant}_{trial_label}_inverse_dynamics_{model_type}.osim')

    # Set up inverse simulation
    # -------------------------------------------------------------------------

    # Create tracking tool
    inverse = osim.MocoInverse()
    inverse.setName(f'{participant}_{trial_label}_inverse-dynamics_{model_type}')

    # Create model processor
    inverse_model_proc = osim.ModelProcessor(f'{participant}_{trial_label}_inverse_dynamics_{model_type}.osim')

    # Append external loads
    inverse_model_proc.append(osim.ModOpAddExternalLoads(f'{trial_name}_grf.xml'))

    # Set model in tool
    inverse.setModel(inverse_model_proc)

    # Set kinematics
    inverse.setKinematics(osim.TableProcessor(f'{participant}_{trial_label}_coordinates.sto'))
    inverse.set_kinematics_allow_extra_columns(True)

    # Set the timings
    # Use times from pre-solved simulation for ease
    inverse.set_initial_time(osim.TimeSeriesTable(
        f'{participant}_{trial_label}_coordinates.sto').getIndependentColumn()[0])
    inverse.set_final_time(osim.TimeSeriesTable(
        f'{participant}_{trial_label}_coordinates.sto').getIndependentColumn()[-1])

    # Initialise to a Moco study and problem to finalise
    # -------------------------------------------------------------------------

    # Get study and problem
    study = inverse.initialize()
    problem = study.updProblem()

    # Update control effort goal
    # -------------------------------------------------------------------------

    # Get a reference to the MocoControlCost goal and set parameters
    # Note that this has a different name to tracking tool goal
    effort = osim.MocoControlGoal.safeDownCast(problem.updGoal('excitation_effort'))
    effort.setWeight(globalControlEffortWeight)
    effort.setExponent(2)

    # Update individual weights in control effort goal
    # Put higher weight on residual use
    effort.setWeightForControlPattern('/forceset/.*_residual', 10.0)
    # Put standard weight on torque actuators
    effort.setWeightForControlPattern('/forceset/.*_torque', 1.0)

    # Define and configure the solver
    # -------------------------------------------------------------------------

    # Get the solver
    solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())

    # Solver settings
    solver.set_optim_max_iterations(3000)
    solver.set_num_mesh_intervals(mesh_interval_dyn)
    solver.set_optim_constraint_tolerance(1.0e-0)  # higher than desirable, but helps with convergence
    solver.set_optim_convergence_tolerance(1.0e-3)  # higher than desirable, but helps with convergence
    solver.resetProblem(problem)

    # Solve the problem
    # -------------------------------------------------------------------------

    # Set-up timer to track computation time
    computation_start = time.time()

    # Solve!
    inverse_solution = study.solve()

    # End computation timer and record
    computation_run_time = round(time.time() - computation_start, 2)

    # # Option to visualise solution
    # study.visualize(tracking_solution)

    # Save files and finalize
    # -------------------------------------------------------------------------

    # Write solution to file
    if inverse_solution.isSealed():
        inverse_solution.unseal()
    inverse_solution.write(f'{participant}_{trial_label}_inverse-dynamics_{model_type}_solution.sto')

    # Save a dictionary storing computational time
    computation = {'time_s': computation_run_time,
                   'note': f'Inverse dynamics computation time for {participant} {trial_label}'}
    with open(f'{participant}_{trial_label}_inverse-dynamics_computation-time.pkl', 'wb') as pkl_file:
        pickle.dump(computation, pkl_file)

    # Smooth inverse dyanmics forces for later analyses
    # TODO: do optimisation results need to be smoothed? probably for most coordinates, yes...
    # id_table_proc = osim.TableProcessor(f'{participant}_{trial_label}_inverse_dynamics_results.sto')
    # id_table_proc.append(osim.TabOpLowPassFilter(kinematic_filt_freq))
    # id_table = id_table_proc.process()
    # id_table.trim(runID.getStartTime(), runID.getEndTime())
    # osim.STOFileAdapter().write(id_table, f'{participant}_{trial_label}_inverse_dynamics_results.sto')

    # Return to home directory
    os.chdir(home_dir)

    # Print out to console as a bookmark in any log file
    print(f'{"*" * 10} FINISHED INVERSE DYNAMICS FOR {participant} {trial_label} WITH {model_type.upper()} MODEL {"*" * 10}')


# =========================================================================
# Run main code
# =========================================================================

if __name__ == '__main__':

    # TODO: simple model? Not working great in dynamic optimisation

    # Run dynamic optimisation
    # -------------------------------------------------------------------------
    if runDynamicOpt:
        run_dynamic_optimisation('complex')

    # Run static optimisation
    # -------------------------------------------------------------------------
    if runStaticOpt:
        run_static_optimisation('complex')

    # Run inverse dynamics
    # -------------------------------------------------------------------------
    if runInverseDyn:
        run_inverse_dynamics('complex')

    # Exit terminal to avoid any funny business
    # -------------------------------------------------------------------------
    os._exit(00)

# %% ---------- end of script_name.py ---------- %% #

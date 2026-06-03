# -*- coding: utf-8 -*-
"""

@author:

    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au

    This script runs the various torque and muscle driven simulations of
    the running gait cycles for to generate the outputs for calculating
    patellofemoral joint reaction forces with different modelling approaches.

    With the approach of using state tracking for the dynamic optimisation,
    this simulation needs to successfully run first to capture the comparative
    kinematics to use in the static optimisation, muscle analysis and inverse
    dynamics approaches.

    STATUS:
        > TODO

"""

# =========================================================================
# Import packages
# =========================================================================

import opensim as osim
import os
import numpy as np
import pandas as pd
import shutil
# import matplotlib.pyplot as plt
import pickle
import argparse
import random
import re
import time
from glob import glob

# =========================================================================
# Flags for running analyses
# =========================================================================

# Set participant ID to run
# participant = 'PA17'
# condition = 'SRRun'
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--participant', action = 'store', type = str, help = 'Enter the participant ID')
parser.add_argument('-c', '--condition', action = 'store', type = str, help = 'Enter the condition label (e.g. SRRun)')
args = parser.parse_args()
participant = args.participant
condition = args.condition

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
dataset = 'Loone2025'

# Read in participant info
participantInfo = pd.read_csv(os.path.join('..','..','data',dataset,'selected_participant_info.csv'))

# Get participant list from folder
participant_list = [ii for ii in os.listdir(
    os.path.join('..', '..', 'data', dataset)) if os.path.isdir(
    os.path.join(os.path.join('..', '..', 'data', dataset, ii)))]

# Check if input participant is in list
if participant not in participant_list:
    raise ValueError(f'No data found for participant ID {participant}. Check input for error...')

# Set the list of conditions to process
# Currently only one anyway, but if more were to be added it could be done
condition_list = [
    'SRRun',   # standard shoe running
    ]

# Check if input speed is in list
if condition not in condition_list:
    raise ValueError(f'Input condition {condition} is not a valid option. Check input for error...')

# Create the general folder for the participant and speed
os.makedirs(os.path.join('..','..','simulations',dataset,participant), exist_ok=True)
os.makedirs(os.path.join('..','..','simulations',dataset,participant,condition), exist_ok=True)

# OpenSim settings
# -------------------------------------------------------------------------

# Add the utility geometry path for model visualisation
osim.ModelVisualizer.addDirToGeometrySearchPaths(os.path.join(os.getcwd(), '..', '..', 'model', 'Geometry'))

# Set weights for optimisations
# Nitschke et al. (2023) used higher value of 1e0 for muscular effort weight
# Nitschke et al. (2023) used lower weight of 10e-1 on joint tracking, but was in degrees squared (rather than radians?)
# Nitschke et al. (2023) used lower weight of 10e-2 on joint tracking, but was in mm squared?
    # TODO: could do this with add scale factor to convert error to degrees squared? Don't think this is what it does though
globalMarkerTrackingWeight = 1e1  # TODO: dramatically increased this from 1e-1 to get better tracking?
# globalStateTrackingWeight = 1e1  # TODO: originally 1e0 - need to balance out increased effort goal
globalAccelMinWeight = 1e-6  # used for minimising high frequency kinematic oscillations
# globalTorqueControlWeight = 1e-3
# globalMuscleControlWeight = 1e-3
globalControlEffortWeight = 1e0  # TODO: originally 1e-3 - is this therefore causing high activations? yes, but need to balance state tracking...
# globalAuxDerivWeight = 1e-3  # based on Denton and Umberger (2023)

# Set mesh interval for dynamic optimisation
# Note this is somewhat generic but follows some rules proposed by Falisse et al. for gait cycle mesh interval
mesh_interval_dyn = 25
# mesh_interval_dyn = 50  # used for creating more refined consistent guess

# Set kinematics filter frequency
# This matches marker data filter from earlier
kinematic_filt_freq = 10

# =========================================================================
# Define functions
# =========================================================================

# Run a state and marker tracking dynamic optimisation
# -------------------------------------------------------------------------
def run_dynamic_optimisation(model_type):

    """

    This function runs the muscle-driven marker tracking simulations, with the goal
    here to generate a muscle-driven simulation of the gait cycle that minimises
    the marker tracking error and muscle activations. These simulations will
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
    os.makedirs(os.path.join('..','..','simulations',dataset,participant,condition,'dynamic_optimisation'), exist_ok=True)

    # Navigate to simulation folder for ease of use
    home_dir = os.getcwd()
    os.chdir(os.path.join('..','..','simulations',dataset,participant,condition,'dynamic_optimisation'))

    # Identify trial label
    # Use the created mot file to do this
    mot_file = glob(os.path.join('..', '..', '..', '..', '..', 'data', dataset, participant,f'{condition}*_grf.mot'))[0]
    trial_label = os.path.split(mot_file)[-1].split('_grf.mot')[0]

    # Copy external loads file to simulation directory
    shutil.copyfile(
        os.path.join('..', '..', '..', '..', '..', 'data', dataset, participant, f'{trial_label}_grf.mot'),
        f'{trial_label}_grf.mot')
    shutil.copyfile(
        os.path.join('..', '..', '..', '..', '..', 'data', dataset, participant, f'{trial_label}_grf.xml'),
        f'{trial_label}_grf.xml')

    # Copy marker file to simulation directory
    shutil.copyfile(
        os.path.join('..', '..', '..', '..', '..', 'data', dataset, participant, f'{trial_label}_filt.trc'),
        f'{trial_label}_filt.trc')

    # Copy IK data to simulation directory
    shutil.copyfile(
        os.path.join('..', '..', '..', '..', '..', 'data', dataset, participant, 'ik', condition,
                     f'{participant}_{condition}_ik_{model_type}_filt.mot'),
        f'{participant}_{condition}_ik_{model_type}_filt.mot')

    # # Copy function based path set for polynomial approximation of muscles
    # shutil.copyfile(
    #     os.path.join('..', '..', '..', '..', '..', 'data', dataset, participant, 'scaling', f'{model_type}_fitter',
    #                  f'{participant}_{model_type}_FunctionBasedPathSet.xml'),
    #     f'{participant}_{model_type}_FunctionBasedPathSet.xml')

    # Set marker tracking weights
    marker_weights = {
        # Pelvis
        'RASIS': {'weight': 5.0}, 'LASIS': {'weight': 5.0}, 'RPSIS': {'weight': 5.0}, 'LPSIS': {'weight': 5.0},
        'RILCR': {'weight': 2.5}, 'LILCR': {'weight': 2.5},
        # Right thigh
        # 'RGTR': {'weight': 0.0},
        'RTHI': {'weight': 5.0}, 'RLTHI': {'weight': 5.0},
        # 'RLEP': {'weight': 0.0}, 'RMEP': {'weight': 0.0},
        # Right shank
        'RPSH': {'weight': 5.0}, 'RLSH': {'weight': 5.0}, 'RDSH': {'weight': 5.0},
        # 'RLMAL': {'weight': 0.0}, 'RMMAL': {'weight': 0.0},
        # Right foot
        'RHEE': {'weight': 5.0}, 'RTOE': {'weight': 10.0}, 'R5TH': {'weight': 10.0},
        # Left thigh
        # 'LGTR': {'weight': 0.0},
        'LTHI': {'weight': 5.0}, 'LLTHI': {'weight': 5.0},
        # 'LLEP': {'weight': 0.0}, 'LMEP': {'weight': 0.0},
        # Left shank
        'LPSH': {'weight': 5.0}, 'LLSH': {'weight': 5.0}, 'LDSH': {'weight': 5.0},
        # 'LLMAL': {'weight': 0.0}, 'LMMAL': {'weight': 0.0},
        # Left foot
        'LHEE': {'weight': 5.0}, 'LTOE': {'weight': 10.0}, 'L5TH': {'weight': 10.0},
    }

    # # Set state tracking weights
    # state_weights = {'pelvis_tx': 2.5e2, 'pelvis_ty': 1.0e2, 'pelvis_tz': 2.5e2,
    #                  'pelvis_tilt': 5.0e1, 'pelvis_list': 10.0e1, 'pelvis_rotation': 5.0e1,
    #                  'hip_flexion_r': 25.0, 'hip_adduction_r': 15.0, 'hip_rotation_r': 7.5,  # increased hip adduction and rotation weights due to errors
    #                  'knee_angle_r': 25.0, 'ankle_angle_r': 15.0,
    #                  'hip_flexion_l': 15.0, 'hip_adduction_l': 15.0, 'hip_rotation_l': 7.5,  # increased hip adduction and rotation weights due to errors
    #                  'knee_angle_l': 25.0, 'ankle_angle_l': 15.0,
    #                  }
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

    # Select the gait cycle to run the simulation for
    # -------------------------------------------------------------------------

    # Read in GRF data to identify stance phase timings
    trial_grf = osim.TimeSeriesTable(f'{trial_label}_grf.mot')
    vgrf = trial_grf.getDependentColumn('ground_force_r_vy').to_numpy()
    grf_time = np.array(trial_grf.getIndependentColumn())
    # Identify right foot contacts and toe offs based on rising and falling edges above threshold of 50N
    force_above = vgrf > 50
    rising_edges = np.where((~force_above[:-1]) & (force_above[1:]))[0] + 1
    falling_edges = np.where((force_above[:-1]) & (~force_above[1:]))[0] + 1

    # # Stance phase option
    # # Take the mid-point of the indices and take 5 strides either side
    # # Get the associated times to run IK over
    # middle_ind = np.where(rising_edges == rising_edges[len(rising_edges) // 2])[0][0]
    # start_val = rising_edges[middle_ind - 5]
    # end_val = rising_edges[middle_ind + 5]
    # select_from = list(rising_edges[middle_ind - 5:middle_ind+4])

    # Toe-off to toe-off option
    # Take the mid-point of the indices and take 3 strides either side
    middle_ind = np.where(falling_edges == falling_edges[len(rising_edges) // 2])[0][0]
    start_val = falling_edges[middle_ind - 3]
    end_val = falling_edges[middle_ind + 3]
    select_from = list(falling_edges[middle_ind - 3:middle_ind + 2])

    # Randomly sample the starting point from the identified foot strikes
    # Set a seed based on participant ID number for consistency
    random.seed(int(re.search(r"\d+", participant).group()) * 5 + 12345)

    # # Stance phase option
    # select_start = random.sample(select_from, 1)[0]
    # # Find the end of the stance phase based on the force data
    # below = np.where(vgrf[select_start:] < 20)[0]
    # select_end = select_start + below[0]

    # Toe-off to toe-off option
    select_start = random.sample(select_from[:-1], 1)[0]
    select_end = select_from[select_from.index(select_start)+1]

    # Set the start and end times based on grf data
    # Add 50 millisecond buffer either side to avoid potentially erroneous data in initial and final states
    # This gets accounted for by removing when PFJ forces are calculated in a later script
    # See De Groote et al. (2016)
    start_time = grf_time[select_start] - 0.05
    end_time = grf_time[select_end] + 0.05

    # =========================================================================
    # Set-up and run the tracking simulation
    # =========================================================================

    # Set-up the model for the tracking simulation
    # -------------------------------------------------------------------------

    # Construct a model processor to use with the tool
    model_proc = osim.ModelProcessor(os.path.join('..', '..', '..', '..', '..', 'data', dataset, participant, 'scaling',
                                                  f'{participant}_{model_type}.osim'))

    # Append external loads
    model_proc.append(osim.ModOpAddExternalLoads(f'{trial_label}_grf.xml'))

    # Increase muscle isometric force by a scaling factor to deal with potentially higher muscle forces
    model_proc.append(osim.ModOpScaleMaxIsometricForce(1.5))

    # Scale active force curve width
    model_proc.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))

    # # Set muscle to implicit dynamics mode
    # # TODO: does this do anything without tendon compliance?
    # # TODO: don't think this works without activation tendon compliance
    # model_proc.append(osim.ModOpUseImplicitTendonComplianceDynamicsDGF())

    # # Append polynomial approximations for muscles
    # # TODO: probably speeds up but inconsistent with Fukuchi dataset simulations
    # model_proc.append(osim.ModOpReplacePathsWithFunctionBasedPaths(
    #     f'{participant}_{model_type}_FunctionBasedPathSet.xml'))

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
        # musc = opt_model.getMuscles().get(muscle_ind)
        # musc.setMaxControl(np.inf)
        # osim.DeGrooteFregly2016Muscle().safeDownCast(musc).get_fiber_damping()
        # Option for elastic tendons on plantarflexor muscles
        # # TODO: does this work?
        # if musc.getName() in ['gaslat_r', 'gasmed_r', 'soleus_r']:
        #     musc.set_ignore_tendon_compliance(False)
        #     osim.DeGrooteFregly2016Muscle().safeDownCast(musc).set_tendon_compliance_dynamics_mode('implicit')
        # # Option to reduce fiber damping
        # osim.DeGrooteFregly2016Muscle().safeDownCast(musc).set_fiber_damping(1.0e-3)

    # Finalise model connections
    opt_model.finalizeConnections()
    opt_model.initSystem()

    # Print model to file in tracking directory
    opt_model.printToXML(f'{participant}_{condition}_dynamic-optimisation_complex.osim')

    # Clean up marker data for tracking simulation
    # I believe this needs to be done to remove nan's from the marker dataset and crashing the optimisation
    # -------------------------------------------------------------------------

    # Load in the marker file
    trc_data = osim.TimeSeriesTableVec3(f'{trial_label}_filt.trc')

    # Remove the calibration markers
    # Add exceptions here as some data doesn't have the calibration markers
    try:
        trc_data.removeColumn('RMEP')
    except:
        print('RMEP marker not found in dataset. Ignoring...')
    try:
        trc_data.removeColumn('LMEP')
    except:
        print('LMEP marker not found in dataset. Ignoring...')
    try:
        trc_data.removeColumn('RMMAL')
    except:
        print('RMMAL marker not found in dataset. Ignoring...')
    try:
        trc_data.removeColumn('LMMAL')
    except:
        print('LMMAL marker not found in dataset. Ignoring...')

    # Save to file
    osim.TRCFileAdapter().write(trc_data, f'{trial_label}_clean.trc')

    # Clean up kinematic data for tracking guess
    # -------------------------------------------------------------------------

    # Load in kinematic data to table processor
    ik_proc = osim.TableProcessor(f'{participant}_{condition}_ik_{model_type}_filt.mot')

    # Append operators to filter data, derive speeds, convert to radians and use full state names
    ik_proc.append(osim.TabOpAppendCoordinateValueDerivativesAsSpeeds())

    # Process table to get data
    ik_data = ik_proc.process(opt_model)

    # Trim kinematic data to start and end times
    ik_data.trim(start_time, end_time)

    # Write to file
    osim.STOFileAdapter().write(ik_data, f'{participant}_{condition}_ik-initial-guess_{model_type}.sto')

    # Set up tracking simulation
    # -------------------------------------------------------------------------

    # Create tracking tool
    track = osim.MocoTrack()
    track.setName(f'{participant}_{condition}_dynamic-optimisation_{model_type}')

    # Set model
    track_model_proc = osim.ModelProcessor(f'{participant}_{condition}_dynamic-optimisation_complex.osim')
    track.setModel(track_model_proc)
    track_model = track_model_proc.process()
    track_model.initSystem()

    # Set the marker reference file and settings
    track.setMarkersReferenceFromTRC(f'{trial_label}_clean.trc')
    track.set_markers_global_tracking_weight(globalMarkerTrackingWeight)

    # Set individual marker weights
    marker_weight_set = osim.MocoWeightSet()
    for marker in marker_weights.keys():
        marker_weight_set.cloneAndAppend(osim.MocoWeight(marker, marker_weights[marker]['weight']))
    track.set_markers_weight_set(marker_weight_set)

    # # Set state tracking reference
    # states_table_proc = osim.TableProcessor(ik_data)
    # track.setStatesReference(states_table_proc)
    # track.set_states_global_tracking_weight(globalStateTrackingWeight)

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
                initial_coord_bounds[coord_name][1] =opt_model.getCoordinateSet().get(coord_name).getRangeMax()
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
    solver.set_optim_constraint_tolerance(1.0e-0)  # higher than desirable, but helps with convergence
    solver.set_optim_convergence_tolerance(1.0e-3)  # higher than desirable, but helps with convergence
    # solver.set_minimize_implicit_auxiliary_derivatives(True)
    # solver.set_implicit_auxiliary_derivatives_weight(globalAuxDerivWeight) 
    
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

    # # Set generic zero guess for forceset values
    # # This is if we want to use a default generic guess for forces rather than the consistent guess
    # for state_name in guess.getStateNames():
    #     if state_name.startswith('/forceset/'):
    #         guess.setState(state_name, np.zeros(guess.getNumTimes()))
    # for control_name in guess.getControlNames():
    #     if control_name.startswith('/forceset/'):
    #         guess.setControl(control_name, np.zeros(guess.getNumTimes()))

    # # Set relevant elements in guess using consistent pre-solved simulation
    # Read in consistent initial guess for speed
    consistent_guess = osim.MocoTrajectory(os.path.join('..', '..', '..', '..', '..',
                                                        'guess', dataset, f'{condition}_consistent-guess.sto'))
    # Resample current guess to the consistent guess (should match mesh interval)
    guess.resampleWithNumTimes(consistent_guess.getNumTimes())
    # Look for relevant states in consistent guess to fill
    for state_name in guess.getStateNames():
        if state_name.endswith('/activation'):
            guess.setState(state_name, consistent_guess.getState(state_name).to_numpy())
    # Look for relevant controls in consistent guess to fill
    for control_name in guess.getControlNames():
        if control_name.endswith('_r') or control_name.endswith('_torque'):
            guess.setControl(control_name, consistent_guess.getControl(control_name).to_numpy())
        # Otherwise set to zero for reserve and residual actuators
        else:
            guess.setControl(control_name, np.zeros(guess.getNumTimes()))

    # Write to file for reference
    guess.write(f'{participant}_{condition}_initial-guess_{model_type}.sto')

    # Set guess in solver
    solver.setGuessFile(f'{participant}_{condition}_initial-guess_{model_type}.sto')

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
    tracking_solution.write(f'{participant}_{condition}_dynamic-optimisation_{model_type}_solution.sto')

    # Save a dictionary storing computational time
    computation = {'time_s': computation_run_time,
                   'note': f'Dynamic optimisation computation time for {participant} {condition}'}
    with open(f'{participant}_{condition}_dynamic-optimisation_computation-time.pkl', 'wb') as pkl_file:
        pickle.dump(computation, pkl_file)

    # Remove initial tracked states and markers file
    os.remove(f'{participant}_{condition}_dynamic-optimisation_{model_type}_tracked_markers.sto')
    # os.remove(f'{participant}_{condition}_dynamic-optimisation_{model_type}_tracked_states.sto')

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
                                f'{participant}_{condition}_dynamic-optimisation_{model_type}_muscle-forces.sto')

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
                                    f'{participant}_{condition}_dynamic-optimisation_{model_type}_pfjrf.sto')

    # Return to home directory
    os.chdir(home_dir)

    # Print out to console as a bookmark in any log file
    print(f'{"*"*10} FINISHED DYNAMIC OPTIMISATION FOR {participant} {condition} {"*"*10}')


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
    os.makedirs(os.path.join('..','..','simulations',dataset,participant,condition,'static_optimisation'), exist_ok=True)

    # Navigate to simulation folder for ease of use
    home_dir = os.getcwd()
    os.chdir(os.path.join('..','..','simulations',dataset,participant,condition,'static_optimisation'))

    # Identify trial label
    # Use the created mot file to do this
    mot_file = glob(os.path.join('..', '..', '..', '..', '..', 'data', dataset, participant, f'{condition}*_grf.mot'))[0]
    trial_label = os.path.split(mot_file)[-1].split('_grf.mot')[0]

    # Copy external loads file to simulation directory
    shutil.copyfile(
        os.path.join('..', '..', '..', '..', '..', 'data', dataset, participant, f'{trial_label}_grf.mot'),
        f'{trial_label}_grf.mot')
    shutil.copyfile(
        os.path.join('..', '..', '..', '..', '..', 'data', dataset, participant, f'{trial_label}_grf.xml'),
        f'{trial_label}_grf.xml')

    # Copy states from the dynamic optimisation
    dynamic_opt_traj = osim.MocoTrajectory(os.path.join(
        '..', 'dynamic_optimisation', f'{participant}_{condition}_dynamic-optimisation_{model_type}_solution.sto'))
    states_table_proc = osim.TableProcessor(dynamic_opt_traj.exportToStatesTable())
    states_table_proc.append(osim.TabOpLowPassFilter(kinematic_filt_freq))
    states_table = states_table_proc.process()
    states_table.trim(dynamic_opt_traj.getInitialTime(), dynamic_opt_traj.getFinalTime())
    osim.STOFileAdapter().write(states_table, f'{participant}_{condition}_states.sto')

    # Check for simple model and need to invert knee angle in states
    if model_type == 'simple':
        # Read in data
        states_data = osim.TimeSeriesTable(f'{participant}_{condition}_states.sto')
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
        osim.STOFileAdapter().write(states_data, f'{participant}_{condition}_states.sto')

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
    opt_model.printToXML(f'{participant}_{condition}_static_optimisation_{model_type}.osim')

    # Set-up static optimisation
    # -------------------------------------------------------------------------

    # Create the analyze tool by reading in the pre-created utility
    analyzeTool = osim.AnalyzeTool(
        os.path.join(os.path.join('..', '..', '..', '..', '..', 'utilities',
                                  f'static_optimisation_{model_type}.xml')), False)

    # Set tool name
    analyzeTool.setName(f'{participant}_{condition}')

    # Set the model file
    analyzeTool.setModelFilename(f'{participant}_{condition}_static_optimisation_{model_type}.osim')

    # Set times for analysis
    analyzeTool.setStartTime(osim.TimeSeriesTable(f'{participant}_{condition}_states.sto').getIndependentColumn()[0])
    analyzeTool.setFinalTime(osim.TimeSeriesTable(f'{participant}_{condition}_states.sto').getIndependentColumn()[-1])

    # Set states file
    analyzeTool.setStatesFileName(f'{participant}_{condition}_states.sto')

    # Set external loads
    analyzeTool.setExternalLoadsFileName(f'{trial_label}_grf.xml')

    # Save tool
    analyzeTool.printToXML(f'{participant}_{condition}_setup-static-optimisation_{model_type}.xml')

    # Run static optimisation
    # -------------------------------------------------------------------------

    # Set-up timer to track computation time
    computation_start = time.time()

    # Read the tool back in as this sometimes helps avoid Python crashing
    runAnalysis = osim.AnalyzeTool(f'{participant}_{condition}_setup-static-optimisation_{model_type}.xml')

    # Run the tool
    runAnalysis.run()

    # End computation timer and record
    computation_run_time = round(time.time() - computation_start, 2)

    # Save a dictionary storing computational time
    computation = {'time_s': computation_run_time,
                   'note': f'Static optimisation {model_type} model computation time for {participant} {condition}'}
    with open(f'{participant}_{condition}_static-optimisation_{model_type}_computation-time.pkl', 'wb') as pkl_file:
        pickle.dump(computation, pkl_file)

    # Return to home directory
    os.chdir(home_dir)

    # Print out to console as a bookmark in any log file
    print(f'{"*" * 10} FINISHED STATIC OPTIMISATION FOR {participant} {condition} WITH {model_type.upper()} MODEL {"*" * 10}')


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
    os.makedirs(os.path.join('..','..','simulations',dataset,participant,condition,'inverse_dynamics'), exist_ok=True)

    # Navigate to simulation folder for ease of use
    home_dir = os.getcwd()
    os.chdir(os.path.join('..','..','simulations',dataset,participant,condition,'inverse_dynamics'))

    # Identify trial label
    # Use the created mot file to do this
    mot_file = glob(os.path.join('..', '..', '..', '..', '..', 'data', dataset, participant, f'{condition}*_grf.mot'))[0]
    trial_label = os.path.split(mot_file)[-1].split('_grf.mot')[0]

    # Copy external loads file to simulation directory
    shutil.copyfile(
        os.path.join('..', '..', '..', '..', '..', 'data', dataset, participant, f'{trial_label}_grf.mot'),
        f'{trial_label}_grf.mot')
    shutil.copyfile(
        os.path.join('..', '..', '..', '..', '..', 'data', dataset, participant, f'{trial_label}_grf.xml'),
        f'{trial_label}_grf.xml')

    # Copy states from the dynamic optimisation
    dynamic_opt_traj = osim.MocoTrajectory(os.path.join(
        '..', 'dynamic_optimisation', f'{participant}_{condition}_dynamic-optimisation_complex_solution.sto'))
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
    osim.STOFileAdapter().write(coord_table, f'{participant}_{condition}_coordinates.sto')

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
    opt_model.printToXML(f'{participant}_{condition}_inverse_dynamics_{model_type}.osim')

    # Set-up inverse dynamics
    # -------------------------------------------------------------------------

    # Create the analyze tool by reading in the pre-created utility
    idTool = osim.InverseDynamicsTool()

    # Set tool name
    idTool.setName(f'{participant}_{condition}')

    # Set the model file
    idTool.setModelFileName(f'{participant}_{condition}_inverse_dynamics_{model_type}.osim')

    # Set times for analysis
    idTool.setStartTime(osim.TimeSeriesTable(f'{participant}_{condition}_coordinates.sto').getIndependentColumn()[0])
    idTool.setEndTime(osim.TimeSeriesTable(f'{participant}_{condition}_coordinates.sto').getIndependentColumn()[-1])

    # Set states file
    idTool.setCoordinatesFileName(f'{participant}_{condition}_coordinates.sto')

    # Set external loads
    idTool.setExternalLoadsFileName(f'{trial_label}_grf.xml')

    # Set output filename
    idTool.setOutputGenForceFileName(f'{participant}_{condition}_inverse_dynamics_results.sto')

    # Set forces to exclude (muscles just in case, even though there are none)
    exclude_forces = osim.ArrayStr()
    exclude_forces.append('muscles')
    idTool.setExcludedForces(exclude_forces)

    # Save tool
    idTool.printToXML(f'{participant}_{condition}_setup-inverse-dynamics_{model_type}.xml')

    # Run inverse dynamics
    # -------------------------------------------------------------------------

    # Set-up timer to track computation time
    computation_start = time.time()

    # Read the tool back in as this sometimes helps avoid Python crashing
    runID = osim.InverseDynamicsTool(f'{participant}_{condition}_setup-inverse-dynamics_{model_type}.xml')

    # Run the tool
    runID.run()

    # End computation timer and record
    computation_run_time = round(time.time() - computation_start, 2)

    # Save a dictionary storing computational time
    computation = {'time_s': computation_run_time,
                   'note': f'Inverse dynamics {model_type} model computation time for {participant} {condition}'}
    with open(f'{participant}_{condition}_inverse-dynamics_{model_type}_computation-time.pkl', 'wb') as pkl_file:
        pickle.dump(computation, pkl_file)

    # Smooth inverse dyanmics forces for later analyses
    id_table_proc = osim.TableProcessor(f'{participant}_{condition}_inverse_dynamics_results.sto')
    id_table_proc.append(osim.TabOpLowPassFilter(kinematic_filt_freq))
    id_table = id_table_proc.process()
    id_table.trim(runID.getStartTime(), runID.getEndTime())
    osim.STOFileAdapter().write(id_table, f'{participant}_{condition}_inverse_dynamics_results.sto')

    # Return to home directory
    os.chdir(home_dir)

    # Print out to console as a bookmark in any log file
    print(f'{"*" * 10} FINISHED INVERSE DYNAMICS FOR {participant} {condition} WITH {model_type.upper()} MODEL {"*" * 10}')

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

# %% ---------- end of run_simulations.py ---------- %% #
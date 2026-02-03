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

    With the approach of using marker tracking for the dynamic optimisation,
    this simulation needs to successfully run first to capture the comparative
    kinematics to use in the static optimisation, muscle analysis and inverse
    dynamics approaches.

    TODO:
         > Consider muscle parameter optimisation?
         > event detection - https://www.sciencedirect.com/science/article/pii/S0966636213006760
         > generic heights used - find participant details in dataset
         > In simulation code
            > Identify ascent/descent based on trial name
            > Identify relevant trials for right leg data
                >> For stair ascent - detect when left limb comes off bottom step while right limb on FP4, through to
                   detection of when right limb contacts step above FP5
                >> For stair descent - detect when right limb comes off top stair when left limb on FP5, through to when
                   left limb contacts step below FP4
                >> Differs though with AB03 - which has a better full step for right on FP5 for ascent...
                    >> For this you can identify when right comes off previous step with event while left on FP4, and then
                       take from this until left limb event is identified on step above
                >> Given differences between participant trials, should it just be single leg support on measured FP?
        > Single limb stance for ascent has highest PFJRF and recfem forces at beginning, need the full step
            >> i.e. toe on with preceeding step support through to toe on of opposite limb
               (bilateral - single - bilateral support)?
            >> This wouldn't work for AB01 ascent, but would for descent
            >> Would work for AB03 ascent, wouldn't work for descent
            >> Check if this is consistent across other participants...

"""

# =========================================================================
# Import packages
# =========================================================================

import opensim as osim
import os
import numpy as np
import pandas as pd
import shutil
from scipy.signal import find_peaks
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
participant = 'AB01'
trial = 'stairs_1_3'
# parser = argparse.ArgumentParser()
# parser.add_argument('-p', '--participant', action = 'store', type = str, help = 'Enter the participant ID')
# parser.add_argument('-s', '--speed', action = 'store', type = str, help = 'Enter the speed label (T25, T35, T45)')
# args = parser.parse_args()
# participant = args.participant
# speed = args.speed

# Settings for running specific sections of code
runDynamicOpt = True
runTorqueSim = False
runStaticOpt = False

# =========================================================================
# Set-up
# =========================================================================

# General settings
# -------------------------------------------------------------------------

# Set dataset name
dataset = 'Scherpereel2023'

# Read in participant info
# TODO: find participant info data

# Get participant list from folder
participant_list = [ii for ii in os.listdir(
    os.path.join('..', '..', 'data', dataset)) if os.path.isdir(
    os.path.join(os.path.join('..', '..', 'data', dataset, ii)))]

# Check if input participant is in list
if participant not in participant_list:
    raise ValueError(f'No data found for participant ID {participant}. Check input for error...')

# Set the list of trials to process per participant
# Modify this if you want to include different trials
trial_list = {
    'AB01': {'ascent': ['stairs_1_3'], 'descent': ['stairs_1_4']},
    'AB03': {'ascent': ['stairs_1_5'], 'descent': ['stairs_1_2']}
}

# Check if input trial is in list
participant_trials = [ff for ff in trial_list[participant]['ascent']] + \
                     [ff for ff in trial_list[participant]['descent']]
if trial not in participant_trials:
    raise ValueError(f'Input trial of {trial} is not a valid option. Check input for error...')

# Identify whether trial is ascent vs. descent
if trial in [ff for ff in trial_list[participant]['ascent']]:
    trial_type = 'ascent'
elif trial in [ff for ff in trial_list[participant]['descent']]:
    trial_type = 'descent'

# Set a trial label
if trial_type == 'ascent':
    trial_label = 'ascent_' + str([ff for ff in trial_list[participant]['ascent']].index(trial))
elif trial_type == 'descent':
    trial_label = 'descent_' + str([ff for ff in trial_list[participant]['descent']].index(trial))

# Create the general folder for the participant and speed
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
# TODO: review these with new approach? marker tracking needs to be high, while control probably needs to be low
globalMarkerTrackingWeight = 1e1
# globalTorqueControlWeight = 1e-3
# globalMuscleControlWeight = 1e-3
globalControlEffortWeight = 1e-1

# Set mesh interval for dynamic optimisation
# Note this is somewhat generic
mesh_interval_dyn = 50

# # Set mesh interval for torque simulation
# meshIntervalTorque = 50

# # Set mesh refinement interval approach for dynamic optimisation
# # TODO: review this with new approach?
# meshIntervalMuscle = [5, 12, 25, 50]

# # Set kinematics filter frequency
# # This matches marker data filter from associated paper
# kinematic_filt_freq = 10

# =========================================================================
# Define functions
# =========================================================================

# Run a marker tracking dynamic optimisation
# -------------------------------------------------------------------------
def run_dynamic_optimisation(model_type):

    """

    This function runs the muscle-driven marker tracking simulations, with the goal
    here to generate a muscle-driven simulation of the step climb that minimises
    the marker tracking error and muscle activations. These simulations will
    generate the kinematic data to be consistently used across other approaches.

    TODO:
        > Muscle parameter optimisation as a part of this?

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

    # Navigate to simulation folder for ease of use
    home_dir = os.getcwd()
    os.chdir(os.path.join('..','..','simulations',dataset,participant,trial_label,'dynamic_optimisation'))

    # Copy external loads file to simulation directory
    shutil.copyfile(
        os.path.join('..', '..', '..', '..', '..', 'data', dataset, participant, f'{trial}_grf.mot'),
        f'{trial}_grf.mot')
    shutil.copyfile(
        os.path.join('..', '..', '..', '..', '..', 'data', dataset, participant, f'{trial}_grf.xml'),
        f'{trial}_grf.xml')

    # Copy marker file to simulation directory
    shutil.copyfile(
        os.path.join('..', '..', '..', '..', '..', 'data', dataset, participant, f'{trial}.trc'),
        f'{trial}.trc')

    # Copy IK data to simulation directory
    shutil.copyfile(
        os.path.join('..', '..', '..', '..', '..', 'data', dataset, participant, 'ik', trial,
                     f'{participant}_{trial}_ik_{model_type}_filt.mot'),
        f'{participant}_{trial}_ik_{model_type}_filt.mot')

    # Set marker tracking weights
    marker_weights = {
        # Pelvis
        'RASI': {'weight': 5.0}, 'LASI': {'weight': 5.0}, 'RPSI': {'weight': 5.0}, 'LPSI': {'weight': 5.0},
        # Right thigh
        'RGTR': {'weight': 0.0},
        'RTHC': {'weight': 5.0}, 'RTHL': {'weight': 5.0}, 'RTHR': {'weight': 5.0},
        'RKNE': {'weight': 0.0}, 'RMKNE': {'weight': 0.0},
        # Right shank
        'RSHC': {'weight': 5.0}, 'RSHL': {'weight': 5.0}, 'RSHR': {'weight': 5.0},
        'RANK': {'weight': 0.0}, 'RMANK': {'weight': 0.0},
        # Right foot
        'RHEEL': {'weight': 10.0}, 'RMT1': {'weight': 5.0}, 'RMT5': {'weight': 5.0},
        # Left thigh
        'LGTR': {'weight': 0.0},
        'LTHC': {'weight': 5.0}, 'LTHL': {'weight': 5.0}, 'LTHR': {'weight': 5.0},
        'LKNE': {'weight': 0.0}, 'LMKNE': {'weight': 0.0},
        # Left shank
        'LSHC': {'weight': 5.0}, 'LSHL': {'weight': 5.0}, 'LSHR': {'weight': 5.0},
        'LANK': {'weight': 0.0}, 'LMANK': {'weight': 0.0},
        # Left foot
        'LHEEL': {'weight': 10.0}, 'LMT1': {'weight': 5.0}, 'LMT5': {'weight': 5.0},
    }

    # Set actuator forces to support simulation
    # TODO: reserves needed?
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
                  'hip_flexion_l': {'actuatorType': 'torque', 'optForce': 300.0},
                  'hip_adduction_l': {'actuatorType': 'torque', 'optForce': 200.0},
                  'hip_rotation_l': {'actuatorType': 'torque', 'optForce': 100.0},
                  'knee_angle_l': {'actuatorType': 'torque', 'optForce': 300.0},
                  'ankle_angle_l': {'actuatorType': 'torque', 'optForce': 200.0},
                  }

    # Identify the time period to run the simulation over
    # The simulation is run over a single-leg support step
    # -------------------------------------------------------------------------

    # Vertical force threshold
    vert_threshold = 20

    # Read in external loads and GRFS to determine force plate timing
    ex_loads = osim.ExternalLoads(f'{trial}_grf.xml', True)
    grf_data = osim.TimeSeriesTable(f'{trial}_grf.mot')
    grf_names = [ex_loads.get(ii).getName() for ii in range(ex_loads.getSize())]
    grf_right_identifier = ex_loads.get(grf_names.index('RightGRF')).getForceIdentifier()
    grf_left_identifier = ex_loads.get(grf_names.index('LeftGRF')).getForceIdentifier()
    right_vgrf = grf_data.getDependentColumn(f'{grf_right_identifier}y').to_numpy()
    left_vgrf = grf_data.getDependentColumn(f'{grf_left_identifier}y').to_numpy()

    # Determine when right limb is in contact with plate
    right_on = np.argmax(right_vgrf > vert_threshold)
    right_on_t = grf_data.getIndependentColumn()[right_on]
    right_off = np.argmax(right_vgrf[right_on::] < vert_threshold) + right_on - 1
    right_off_t = grf_data.getIndependentColumn()[right_off]

    # Determine when left limb is in contact with plate
    left_on = np.argmax(left_vgrf > vert_threshold)
    left_on_t = grf_data.getIndependentColumn()[left_on]
    left_off = np.argmax(left_vgrf[left_on::] < vert_threshold) + left_on - 1
    left_off_t = grf_data.getIndependentColumn()[left_off]

    # Determine end point of simulation
    # This is where the left limb comes in contact with the next stair
    # First we can check if this can be identified via force plate based on left contact being after the right step
    if left_on > right_on:
        end_time = grf_data.getIndependentColumn()[left_on]
    else:
        # If not, we can use appropriate step detection algorithms
        # The appropriate data to use here will depend on whether it is an ascent or descent trial
        print('TODO: use appropriate step detection algorithm for ascent vs. descent')

    # Determine the start point of the simulation
    # This is where the left limb comes off the previous stair
    # First we can check if this can be identified via force plate based on left off being before the right off
    if left_off < right_off:
        start_time = grf_data.getIndependentColumn()[left_off]
    else:
        # If not, we can use appropriate step detection algorithms
        # The appropriate data to use here will depend on whether it is an ascent or descent trial
        if trial_type == 'ascent':
            # Here we use the peak vertical displacement between a toe-marker and pelvis position
            # Load in IK and marker data
            ik_data = osim.TimeSeriesTable(f'{participant}_{trial}_ik_complex_filt.mot')
            marker_data = osim.TimeSeriesTableVec3(f'{trial}.trc').flatten()
            # Get the relevant vertical positioning data
            pelvis_ty = ik_data.getDependentColumn('/jointset/ground_pelvis/pelvis_ty/value').to_numpy()
            toe_ty = marker_data.getDependentColumn('LMT1_2').to_numpy() / 1000  # convert mm to m
            # Calculate displacement difference
            vd = pelvis_ty - toe_ty
            # Identify peaks in vertical displacement differences
            peaks = find_peaks(vd, distance = 100)[0] # distance seems reasonable based on sampling rate and step freq
            peak_times = [ik_data.getIndependentColumn()[ii] for ii in peaks]
            # Find first peak after right on time
            peak_time_diffs = [ii - right_on_t for ii in peak_times]
            peak_index = peak_time_diffs.index(min(vv for vv in peak_time_diffs if vv > 0))
            # Identify toe off time prior to right foot contact as start time
            start_time = peak_times[peak_index]
        elif trial_type == 'descent':
            print('TODO: descent algorithm...')

    # # Test plot options to review
    # plt.plot(grf_data.getIndependentColumn(), right_vgrf, 'g-')
    # plt.plot(grf_data.getIndependentColumn(), left_vgrf, 'r-')
    # plt.plot(ik_data.getIndependentColumn(), vd * 1000, 'k:')

    # Check start time vs. end time
    if start_time > end_time:
        raise ValueError('Start time identified after end time. Some sort of error in step detection...')

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
    model_proc.append(osim.ModOpAddExternalLoads(f'{trial}_grf.xml'))

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
    opt_model.printToXML(f'{participant}_{trial_type}_dynamic-optimisation_{model_type}.osim')

    # Clean up kinematic data for tracking guess
    # -------------------------------------------------------------------------

    # Load in kinematic data to table processor
    ik_proc = osim.TableProcessor(f'{participant}_{trial}_ik_{model_type}_filt.mot')

    # Append operators to derive speeds
    ik_proc.append(osim.TabOpAppendCoordinateValueDerivativesAsSpeeds())

    # Process table to get data
    ik_data = ik_proc.process(opt_model)

    # Trim kinematic data to start and end times
    ik_data.trim(start_time, end_time)

    # Write to file
    osim.STOFileAdapter().write(ik_data, f'{participant}_{trial_type}_ik-initial-guess_{model_type}.sto')

    # Set up tracking simulation
    # -------------------------------------------------------------------------

    # Create tracking tool
    track = osim.MocoTrack()
    track.setName(f'{participant}_{trial_type}_dynamic-optimisation_{model_type}')

    # Set model
    track_model_proc = osim.ModelProcessor(f'{participant}_{trial_type}_dynamic-optimisation_{model_type}.osim')
    track.setModel(track_model_proc)

    # Set the marker reference file and settings
    track.setMarkersReferenceFromTRC(f'{trial}.trc')
    track.set_markers_global_tracking_weight(globalMarkerTrackingWeight)

    # Set individual marker weights
    marker_weight_set = osim.MocoWeightSet()
    for marker in marker_weights.keys():
        marker_weight_set.cloneAndAppend(osim.MocoWeight(marker, marker_weights[marker]['weight']))
    track.set_markers_weight_set(marker_weight_set)

    # Set to ignore unused columns
    track.set_allow_unused_references(True)

    # Set the timings
    track.set_initial_time(start_time)
    track.set_final_time(end_time)

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
    effort.setWeightForControlPattern('/forceset/.*_torque', 0.1)
    # Set standard weights on muscle controls
    # This probably doesn't change default but provides an option to set
    effort.setWeightForControlPattern('/forceset/.*_r', 1.0)

    # Add initial activation goal to avoid muscles from cheating at start of simulation
    # -------------------------------------------------------------------------

    # For all muscles with activation dynamics, the initial activation and initial excitation should be the same.
    # Without this goal, muscle activation may undesirably start at its maximum possible value as only excitation
    # is penalised
    initial_act = osim.MocoInitialActivationGoal('initial_activation')
    problem.addGoal(initial_act)

    # TODO: consider parameter optimisation?
    # -------------------------------------------------------------------------

    # TODO: consider bounds on muscles?
    # -------------------------------------------------------------------------

    # Define and configure the solver
    # -------------------------------------------------------------------------

    # Get the solver
    solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())

    # # Set-up timer to track total computation time
    # computation_start = time.time()

    # TODO: consider need for mesh refinement approach? Probably not necessary...

    # TODO: get initial guess with low constraint tolerance? or low mesh?

    # Solver settings
    solver.set_optim_max_iterations(1000)  # TODO: might need this higher?
    solver.set_num_mesh_intervals(mesh_interval_dyn)  # TODO: decide on approach here...testing coarse...
    solver.set_optim_constraint_tolerance(1.0e-2)  # TODO: too low?
    solver.set_optim_convergence_tolerance(1.0e-3)
    solver.resetProblem(problem)

    # Get the initial guess
    guess = solver.getGuess()

    # Get and resample the guess to match IK
    guess.resampleWithNumTimes(ik_data.getNumRows())

    # Insert the desired values from IK
    for col in guess.getStateNames():
        if col in ik_data.getColumnLabels():
            guess.setState(col, ik_data.getDependentColumn(col).to_numpy())

    # TODO: set initial guess for muscle controls?

    # Write to file for reference
    guess.write(f'{participant}_{trial_type}_initial-guess_{model_type}.sto')

    # Set guess in solver
    solver.setGuessFile(f'{participant}_{trial_type}_initial-guess_{model_type}.sto')

    # Reset problem to check any issues
    solver.resetProblem(problem)

    # Solve the problem
    # -------------------------------------------------------------------------

    # # Set-up timer to track computation time
    # computation_start = time.time()

    # Solve!
    tracking_solution = study.solve()

    # # End computation timer and record
    # computation_run_time = round(time.time() - computation_start, 2)

    # # Option to visualise solution
    # study.visualize(tracking_solution)

    # Save files and finalize
    # -------------------------------------------------------------------------

    # Write solution to file
    if tracking_solution.isSealed():
        tracking_solution.unseal()
    tracking_solution.write(f'{participant}_{trial_type}_dynamic-optimisation_{model_type}_solution.sto')

    # # Save a dictionary storing computational time
    # computation = {'time_s': computation_run_time,
    #                'note': f'Torque driven marker tracking computation time for {participant} {speed}'}
    # with open(f'{participant}run{speed}_marker_tracking_computation_time.pkl', 'wb') as pkl_file:
    #     pickle.dump(computation, pkl_file)

    # Remove initial tracked states and markers file
    os.remove(f'{participant}_{trial_type}_dynamic-optimisation_{model_type}_tracked_markers.sto')

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
                                f'{participant}_{trial_type}_dynamic-optimisation_{model_type}_muscle-forces.sto')

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
                                    f'{participant}_{trial_type}_dynamic-optimisation_{model_type}_pfjrf.sto')

    # Return to home directory
    os.chdir(home_dir)

    # Print out to console as a bookmark in any log file
    print(f'{"*"*10} FINISHED DYNAMIC OPTIMISATION FOR {participant} {trial_type} {"*"*10}')

# =========================================================================
# TODO: write appropriate header for here
# =========================================================================

if __name__ == '__main__':
    # TODO: run any appropriate analyses here
    # -------------------------------------------------------------------------

    # TODO: add functions to run    

    # Finalise and exit kernel
    # -------------------------------------------------------------------------

    # Doing this seems to avoid an error code when completing the script run
    os._exit(00)

# %% ---------- end of script_name.py ---------- %% #

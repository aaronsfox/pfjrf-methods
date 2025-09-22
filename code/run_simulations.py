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

    TODO:
        > Simple model in dynamic optimisation has some weird results...
            >> High muscle forces, negative plantarflexor muscle forces...
            >> Is this due to damping/passive?
                >> See: https://simtk.org/plugins/phpBB/viewtopicPhpbb.php?f=1815&t=11584&p=32430&start=0&view=
                >> Doesn't seem to be --- muscles just don't seem to do anything?
        > Tendon dynamics included?
            >> Need to test if works
            >> Never works well with inverse...

"""

# =========================================================================
# Import packages
# =========================================================================

import opensim as osim
import os
import numpy as np
import pandas as pd
import shutil
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
# participant = 'RBDS001'
# speed = 'T35'
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--participant', action = 'store', type = str, help = 'Enter the participant ID')
parser.add_argument('-s', '--speed', action = 'store', type = str, help = 'Enter the speed label (T25, T35, T45)')
args = parser.parse_args()
participant = args.participant
speed = args.speed

# Settings for running specific sections of code
runTorqueSim = True
runStaticOpt = True
runDynamicOpt = True

# =========================================================================
# Set-up
# =========================================================================

# General settings
# -------------------------------------------------------------------------

# Read in participant info
participantInfo = pd.read_csv(os.path.join('..','data','participantInfo.csv'))

# Get participant list from folder
participant_list = [ii for ii in os.listdir(
    os.path.join('..', 'data')) if os.path.isdir(os.path.join(os.path.join('..', 'data', ii)))]

# Check if input participant is in list
if participant not in participant_list:
    raise ValueError(f'No data found for participant ID {participant}. Check input for error...')

# Set the list of speed conditions to process
# Modify this if you want to include different running speeds
speed_list = [
    'T25',   # 2.5 m/s
    'T35',   # 3.5 m/s
    'T45',   # 4.5 m/s
    ]

# Check if input speed is in list
if speed not in speed_list:
    raise ValueError(f'Input speed of {speed} is not a valid option. Check input for error...')

# Create the general folder for the participant and speed
os.makedirs(os.path.join('..','simulations',participant,speed), exist_ok=True)

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
osim.ModelVisualizer.addDirToGeometrySearchPaths(os.path.join(os.getcwd(), '..', 'model', 'Geometry'))

# Read in participant info
participantInfo = pd.read_csv(os.path.join('..', 'data', 'participantInfo.csv'))

# Set weights for optimisations
globalMarkerTrackingWeight = 1e1
globalTorqueControlWeight = 1e-3
globalMuscleControlWeight = 1e-3

# Set mesh interval for torque simulation
meshIntervalTorque = 50

# Set mesh refinement interval approach for dynamic optimisation
meshIntervalMuscle = [5, 12, 25, 50]

# Set kinematics filter frequency
# This matches marker data filter from associated paper
kinematic_filt_freq = 10

# =========================================================================
# Define functions
# =========================================================================

# Run torque driven marker tracking simulation
# -------------------------------------------------------------------------
def run_marker_tracking():

    """

    This function runs the torque driven marker tracking simulations, with the goal
    here to generate a dynamically consistent motion for the set of gait cycles which
    can be tracked with muscle-driven simulations. These simulations also generate the
    knee angle and joint moment data required in the mechanical PFJRF model.

    """

    # =========================================================================
    # Set-up files and parameters for simulation
    # =========================================================================

    # Files and settings
    # -------------------------------------------------------------------------

    # Create the folder for simulation
    os.makedirs(os.path.join('..','simulations',participant,speed,'marker_tracking'), exist_ok=True)

    # Navigate to simulation folder for ease of use
    home_dir = os.getcwd()
    os.chdir(os.path.join('..', 'simulations', participant, speed, 'marker_tracking'))

    # Copy external loads file to simulation directory
    shutil.copyfile(
        os.path.join('..', '..', '..', '..', 'data', participant, f'{participant}run{speed}_grf.mot'),
        f'{participant}run{speed}_grf.mot')
    shutil.copyfile(
        os.path.join('..', '..', '..', '..', 'data', participant, f'{participant}run{speed}_grf.xml'),
        f'{participant}run{speed}_grf.xml')

    # Copy marker file to simulation directory
    shutil.copyfile(
        os.path.join('..', '..', '..', '..', 'data', participant, f'{participant}run{speed}.trc'),
        f'{participant}run{speed}.trc')

    # Set marker tracking weights
    marker_weights = {
        # Pelvis
        'R.ASIS': {'weight': 5.0}, 'L.ASIS': {'weight': 5.0}, 'R.PSIS': {'weight': 5.0}, 'L.PSIS': {'weight': 5.0},
        'R.Iliac.Crest': {'weight': 2.5}, 'L.Iliac.Crest': {'weight': 2.5},
        # Right thigh
        'R.GTR': {'weight': 0.0},
        'R.Thigh.Top.Lateral': {'weight': 5.0}, 'R.Thigh.Bottom.Lateral': {'weight': 5.0},
        'R.Thigh.Top.Medial': {'weight': 5.0}, 'R.Thigh.Bottom.Medial': {'weight': 5.0},
        'R.Knee': {'weight': 0.0}, 'R.Knee.Medial': {'weight': 0.0},
        # Right shank
        'R.HF': {'weight': 0.0}, 'R.TT': {'weight': 0.0},
        'R.Shank.Top.Lateral': {'weight': 5.0}, 'R.Shank.Bottom.Lateral': {'weight': 5.0},
        'R.Shank.Top.Medial': {'weight': 5.0}, 'R.Shank.Bottom.Medial': {'weight': 5.0},
        'R.Ankle': {'weight': 0.0}, 'R.Ankle.Medial': {'weight': 0.0},
        # Right foot
        'R.Heel.Top': {'weight': 5.0}, 'R.Heel.Bottom': {'weight': 5.0}, 'R.Heel.Lateral': {'weight': 5.0},
        'R.MT1': {'weight': 10.0}, 'R.MT2': {'weight': 0.0}, 'R.MT5': {'weight': 10.0},
        # Left thigh
        'L.GTR': {'weight': 0.0},
        'L.Thigh.Top.Lateral': {'weight': 5.0}, 'L.Thigh.Bottom.Lateral': {'weight': 5.0},
        'L.Thigh.Top.Medial': {'weight': 5.0}, 'L.Thigh.Bottom.Medial': {'weight': 5.0},
        'L.Knee': {'weight': 0.0}, 'L.Knee.Medial': {'weight': 0.0},
        # Left shank
        'L.HF': {'weight': 0.0}, 'L.TT': {'weight': 0.0},
        'L.Shank.Top.Lateral': {'weight': 5.0}, 'L.Shank.Bottom.Lateral': {'weight': 5.0},
        'L.Shank.Top.Medial': {'weight': 5.0}, 'L.Shank.Bottom.Medial': {'weight': 5.0},
        'L.Ankle': {'weight': 0.0}, 'L.Ankle.Medial': {'weight': 0.0},
        # Left foot
        'L.Heel.Top': {'weight': 5.0}, 'L.Heel.Bottom': {'weight': 5.0}, 'L.Heel.Lateral': {'weight': 5.0},
        'L.MT1': {'weight': 10.0}, 'L.MT2': {'weight': 0.0}, 'L.MT5': {'weight': 10.0},
    }

    # Set actuator forces to drive simulation
    act_forces = {'pelvis_tx': {'actuatorType': 'residual', 'optForce': 5},
                  'pelvis_ty': {'actuatorType': 'residual', 'optForce': 5},
                  'pelvis_tz': {'actuatorType': 'residual', 'optForce': 5},
                  'pelvis_tilt': {'actuatorType': 'residual', 'optForce': 2.5},
                  'pelvis_list': {'actuatorType': 'residual', 'optForce': 2.5},
                  'pelvis_rotation': {'actuatorType': 'residual', 'optForce': 2.5},
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

    # Select the stance phase to run the simulation for
    # -------------------------------------------------------------------------

    # Replicate the 10 gat cycles selected for inverse kinematics
    # Read in GRF data to identify stance phase timings
    trial_grf = osim.TimeSeriesTable(f'{participant}run{speed}_grf.mot')
    vgrf = trial_grf.getDependentColumn('ground_force_r_vy').to_numpy()
    grf_time = np.array(trial_grf.getIndependentColumn())
    # Identify right foot contacts and toe offs based on rising and falling edges above threshold of 20N
    force_above = vgrf > 20
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
    # Take the mid-point of the indices and take 5 strides either side
    # Get the associated times to run IK over
    middle_ind = np.where(falling_edges == falling_edges[len(rising_edges) // 2])[0][0]
    start_val = falling_edges[middle_ind - 5]
    end_val = falling_edges[middle_ind + 5]
    select_from = list(falling_edges[middle_ind - 5:middle_ind + 4])

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
    start_time = grf_time[select_start]
    end_time = grf_time[select_end]

    # =========================================================================
    # Set-up and run the tracking simulation
    # =========================================================================

    # Set-up the model for the tracking simulation
    # -------------------------------------------------------------------------

    # Construct a model processor to use with the tool
    model_proc = osim.ModelProcessor(os.path.join('..', '..', '..', '..', 'data', participant, 'scaling',
                                                  f'{participant}_complex.osim'))

    # Append external loads
    model_proc.append(osim.ModOpAddExternalLoads(f'{participant}run{speed}_grf.xml'))

    # Remove muscles from model
    model_proc.append(osim.ModOpRemoveMuscles())

    # Process model for further edits
    track_model = model_proc.process()

    # Add coordinate actuators to model
    for coordinate in act_forces:
        # Create actuator
        actu = osim.CoordinateActuator()
        # Set name
        actu.setName(f'{coordinate}_{act_forces[coordinate]["actuatorType"]}')
        # Set coordinate
        actu.setCoordinate(track_model.updCoordinateSet().get(coordinate))
        # Set optimal force
        actu.setOptimalForce(act_forces[coordinate]['optForce'])
        # Set min and max control
        actu.setMinControl(np.inf * -1)
        actu.setMaxControl(np.inf * 1)
        # Append to model force set
        track_model.updForceSet().cloneAndAppend(actu)

    # Finalise model connections
    track_model.finalizeConnections()
    track_model.initSystem()

    # Print model to file in tracking directory
    track_model.printToXML(f'{participant}run{speed}_marker_tracking_complex.osim')

    # Clean up kinematic data for tracking guess
    # -------------------------------------------------------------------------

    # Load in kinematic data to table processor
    ik_proc = osim.TableProcessor(os.path.join('..', '..', '..', '..', 'data', participant, 'ik', speed,
                                               f'{participant}_{speed}_ik_complex.mot'))

    # Append operators to filter data, derive speeds, convert to radians and use full state names
    ik_proc.append(osim.TabOpLowPassFilter(kinematic_filt_freq))
    ik_proc.append(osim.TabOpConvertDegreesToRadians())
    ik_proc.append(osim.TabOpUseAbsoluteStateNames())
    ik_proc.append(osim.TabOpAppendCoordinateValueDerivativesAsSpeeds())

    # Process table to get data
    ik_data = ik_proc.process(track_model)

    # Trim kinematic data to start and end times
    ik_data.trim(start_time, end_time)

    # Write to file
    osim.STOFileAdapter().write(ik_data, f'{participant}run{speed}_kinematic_data.sto')

    # Set up tracking simulation
    # -------------------------------------------------------------------------

    # Create tracking tool
    track = osim.MocoTrack()
    track.setName(f'{participant}run{speed}_marker_tracking')

    # Set model
    track_model_proc = osim.ModelProcessor(f'{participant}run{speed}_marker_tracking_complex.osim')
    track.setModel(track_model_proc)

    # Set the marker reference file and settings
    track.setMarkersReferenceFromTRC(f'{participant}run{speed}.trc')
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
    effort.setWeight(globalTorqueControlWeight)
    effort.setExponent(2)

    # Update individual weights in control effort goal
    # Put higher weight on residual use
    effort.setWeightForControlPattern('/forceset/.*_residual', 10.0)
    # Put heavy weight on the reserve actuators
    effort.setWeightForControlPattern('/forceset/.*_torque', 0.1)

    # Define and configure the solver
    # -------------------------------------------------------------------------

    # Get the solver
    solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())

    # Solver settings
    solver.set_optim_max_iterations(2000)
    solver.set_num_mesh_intervals(meshIntervalTorque)
    solver.set_optim_constraint_tolerance(1e-3)
    solver.set_optim_convergence_tolerance(1e-3)

    # Get the initial guess
    guess = solver.getGuess()

    # Get and resample the guess to match IK
    guess.resampleWithNumTimes(ik_data.getNumRows())

    # Insert the desired values from IK
    for col in guess.getStateNames():
        if col in ik_data.getColumnLabels():
            guess.setState(col, ik_data.getDependentColumn(col).to_numpy())

    # Write to file for reference
    guess.write(f'{participant}run{speed}_initial_guess.sto')

    # Set guess in solver
    solver.setGuessFile(f'{participant}run{speed}_initial_guess.sto')

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
    tracking_solution.write(f'{participant}run{speed}_marker_tracking_solution.sto')

    # Save a dictionary storing computational time
    computation = {'time_s': computation_run_time,
                   'note': f'Torque driven marker tracking computation time for {participant} {speed}'}
    with open(f'{participant}run{speed}_marker_tracking_computation_time.pkl', 'wb') as pkl_file:
        pickle.dump(computation, pkl_file)

    # Remove initial tracked states and markers file
    os.remove(f'{participant}run{speed}_marker_tracking_tracked_markers.sto')

    # Create an inverse dynamics like file for joint moments from tracking solution
    # Export the controls from the solution
    controls_table = tracking_solution.exportToControlsTable()
    # Create new torque columns from the control signals to have the torque values in Nm
    for col in controls_table.getColumnLabels():
        if col.endswith('_torque'):
            # Calculate torque in Nm
            torque_nm = controls_table.getDependentColumn(
                col).to_numpy() * act_forces[col.replace('/forceset/','').replace('_torque','')]['optForce']
            # Append column to table
            controls_table.appendColumn(col.replace('/forceset/',''), osim.Vector().createFromMat(torque_nm))
            # Remove the controls column
            controls_table.removeColumn(col)
        else:
            controls_table.removeColumn(col)
    # Use table processor to filter torques
    torque_table_proc = osim.TableProcessor(controls_table)
    torque_table_proc.append(osim.TabOpLowPassFilter(kinematic_filt_freq))
    final_torques = torque_table_proc.process()
    final_torques.trim(controls_table.getIndependentColumn()[0],controls_table.getIndependentColumn()[-1])
    # Write to file
    osim.STOFileAdapter().write(final_torques, f'{participant}run{speed}_marker_tracking_torques.sto')

    # Return to home directory
    os.chdir(home_dir)

    # Print out to console as a bookmark in any log file
    print(f'{"*"*10} FINISHED MARKER TRACKING FOR {participant} {speed} {"*"*10}')


# Run static optimisation
# -------------------------------------------------------------------------
def run_static_optimisation(model_type):

    """

    This function runs static optimisation on the stance phase of a gait cycle
    to estimate the muscle forces driving the motion and subsequent quadriceps
    forces to use in the PFJRF mechanical model.
    
    :param model_type: string of "complex" or "simple" to dictate the model to use
    :return:

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
    os.makedirs(os.path.join('..','simulations',participant,speed,'static_optimisation',model_type), exist_ok=True)

    # Navigate to simulation folder for ease of use
    home_dir = os.getcwd()
    os.chdir(os.path.join('..', 'simulations', participant, speed, 'static_optimisation',model_type))

    # Copy external loads file to simulation directory
    shutil.copyfile(
        os.path.join('..','..', '..', '..', '..', 'data', participant, f'{participant}run{speed}_grf.mot'),
        f'{participant}run{speed}_grf.mot')
    shutil.copyfile(
        os.path.join('..','..', '..', '..', '..', 'data', participant, f'{participant}run{speed}_grf.xml'),
        f'{participant}run{speed}_grf.xml')

    # Copy states from marker tracking
    # Noisiness is sometimes introduced in these torque drive tracking simulations so some filtering is done here
    marker_traj = osim.MocoTrajectory(
        os.path.join('..', '..', 'marker_tracking', f'{participant}run{speed}_marker_tracking_solution.sto'))
    states_table_proc = osim.TableProcessor(marker_traj.exportToStatesTable())
    states_table_proc.append(osim.TabOpLowPassFilter(kinematic_filt_freq))
    states_table = states_table_proc.process()
    states_table.trim(marker_traj.getInitialTime(), marker_traj.getFinalTime())
    osim.STOFileAdapter().write(states_table, f'{participant}run{speed}_states.sto')

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
            states_data.appendColumn(col.replace('/walker_knee_','/knee_'),
                                     osim.Vector().createFromMat(states_data.getDependentColumn(col).to_numpy()*-1))
            # Remove the old one
            states_data.removeColumn(col)
        # Remove the patellofemoral joint columns
        for col in states_data.getColumnLabels():
            if 'patellofemoral_' in col:
                states_data.removeColumn(col)
        # Save new states data to file
        osim.STOFileAdapter().write(states_data, f'{participant}run{speed}_states.sto')

    # Set actuator forces to support simulation
    act_forces = {'pelvis_tx': {'actuatorType': 'residual', 'optForce': 5},
                  'pelvis_ty': {'actuatorType': 'residual', 'optForce': 5},
                  'pelvis_tz': {'actuatorType': 'residual', 'optForce': 5},
                  'pelvis_tilt': {'actuatorType': 'residual', 'optForce': 2.5},
                  'pelvis_list': {'actuatorType': 'residual', 'optForce': 2.5},
                  'pelvis_rotation': {'actuatorType': 'residual', 'optForce': 2.5},
                  'hip_flexion_r': {'actuatorType': 'reserve', 'optForce': 5.0},
                  'hip_adduction_r': {'actuatorType': 'reserve', 'optForce': 5.0},
                  'hip_rotation_r': {'actuatorType': 'reserve', 'optForce': 5.0},
                  'knee_angle_r': {'actuatorType': 'reserve', 'optForce': 5.0},
                  'ankle_angle_r': {'actuatorType': 'reserve', 'optForce': 5.0},
                  'hip_flexion_l': {'actuatorType': 'torque', 'optForce': 300.0},
                  'hip_adduction_l': {'actuatorType': 'torque', 'optForce': 200.0},
                  'hip_rotation_l': {'actuatorType': 'torque', 'optForce': 100.0},
                  'knee_angle_l': {'actuatorType': 'torque', 'optForce': 300.0},
                  'ankle_angle_l': {'actuatorType': 'torque', 'optForce': 200.0},
                  }

    # Prepare model for static optimisation
    # -------------------------------------------------------------------------

    # Construct a model processor to use with the tool
    model_proc = osim.ModelProcessor(os.path.join('..','..', '..', '..', '..', 'data', participant, 'scaling',
                                                  f'{participant}_{model_type}.osim'))

    # Increase muscle isometric force by a scaling factor to deal with potentially higher muscle forces
    model_proc.append(osim.ModOpScaleMaxIsometricForce(1.5))

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
    opt_model.printToXML(f'{participant}run{speed}_static_optimisation_{model_type}.osim')

    # Set-up static optimisation
    # -------------------------------------------------------------------------

    # Create the analyze tool by reading in the pre-created utility
    analyzeTool = osim.AnalyzeTool(
        os.path.join(os.path.join(
            '..','..', '..', '..', '..','utilities',f'static_optimisation_{model_type}.xml')),
        False)

    # Set tool name
    analyzeTool.setName(f'{participant}run{speed}')

    # Set the model file
    analyzeTool.setModelFilename(f'{participant}run{speed}_static_optimisation_{model_type}.osim')

    # Set times for analysis
    analyzeTool.setStartTime(osim.TimeSeriesTable(f'{participant}run{speed}_states.sto').getIndependentColumn()[0])
    analyzeTool.setFinalTime(osim.TimeSeriesTable(f'{participant}run{speed}_states.sto').getIndependentColumn()[-1])

    # Set states file
    analyzeTool.setStatesFileName(f'{participant}run{speed}_states.sto')

    # Set external loads
    analyzeTool.setExternalLoadsFileName(f'{participant}run{speed}_grf.xml')

    # Save tool
    analyzeTool.printToXML(f'{participant}run{speed}_setup_{model_type}.xml')

    # Run static optimisation
    # -------------------------------------------------------------------------

    # Set-up timer to track computation time
    computation_start = time.time()

    # Read the tool back in as this sometimes helps avoid Python crashing
    runAnalysis = osim.AnalyzeTool(f'{participant}run{speed}_setup_{model_type}.xml')

    # Run the tool
    runAnalysis.run()

    # End computation timer and record
    computation_run_time = round(time.time() - computation_start, 2)

    # Save a dictionary storing computational time
    computation = {'time_s': computation_run_time,
                   'note': f'Static optimisation {model_type} model computation time for {participant} {speed}'}
    with open(f'{participant}run{speed}_static_optimisation_{model_type}_computation_time.pkl', 'wb') as pkl_file:
        pickle.dump(computation, pkl_file)

    # Return to home directory
    os.chdir(home_dir)

    # Print out to console as a bookmark in any log file
    print(f'{"*" * 10} FINISHED STATIC OPTIMISATION FOR {participant} {speed} WITH {model_type.upper()} MODEL {"*" * 10}')


# Run dynamic optimisation
# -------------------------------------------------------------------------
def run_dynamic_optimisation(model_type):

    """

    This function runs dynamic optimisation on the stance phase of a gait cycle
    to estimate the muscle forces driving the motion and subsequent quadriceps
    forces to use in the PFJRF mechanical model.

    :param model_type: string of "complex" or "simple" to dictate the model to use
    :return:

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
    os.makedirs(os.path.join('..', 'simulations', participant, speed, 'dynamic_optimisation', model_type), exist_ok=True)

    # Navigate to simulation folder for ease of use
    home_dir = os.getcwd()
    os.chdir(os.path.join('..', 'simulations', participant, speed, 'dynamic_optimisation', model_type))

    # Copy external loads file to simulation directory
    shutil.copyfile(
        os.path.join('..', '..', '..', '..', '..', 'data', participant, f'{participant}run{speed}_grf.mot'),
        f'{participant}run{speed}_grf.mot')
    shutil.copyfile(
        os.path.join('..', '..', '..', '..', '..', 'data', participant, f'{participant}run{speed}_grf.xml'),
        f'{participant}run{speed}_grf.xml')

    # Copy states from marker tracking
    # Noisiness is sometimes introduced in these torque drive tracking simulations so some filtering is done here
    marker_traj = osim.MocoTrajectory(
        os.path.join('..', '..', 'marker_tracking', f'{participant}run{speed}_marker_tracking_solution.sto'))
    states_table_proc = osim.TableProcessor(marker_traj.exportToStatesTable())
    states_table_proc.append(osim.TabOpLowPassFilter(kinematic_filt_freq))
    states_table = states_table_proc.process()
    states_table.trim(marker_traj.getInitialTime(), marker_traj.getFinalTime())
    osim.STOFileAdapter().write(states_table, f'{participant}run{speed}_states.sto')

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
        osim.STOFileAdapter().write(states_data, f'{participant}run{speed}_states.sto')

    # # Copy function based path set for muscles
    # shutil.copyfile(
    #     os.path.join('..', '..', '..', '..', '..', 'data', participant, 'scaling', f'{model_type}_fitter',
    #                  f'{participant}_{model_type}_FunctionBasedPathSet.xml'),
    #     f'{participant}_{model_type}_FunctionBasedPathSet.xml')

    # Set actuator forces to support simulation
    act_forces = {'pelvis_tx': {'actuatorType': 'residual', 'optForce': 5},
                  'pelvis_ty': {'actuatorType': 'residual', 'optForce': 5},
                  'pelvis_tz': {'actuatorType': 'residual', 'optForce': 5},
                  'pelvis_tilt': {'actuatorType': 'residual', 'optForce': 2.5},
                  'pelvis_list': {'actuatorType': 'residual', 'optForce': 2.5},
                  'pelvis_rotation': {'actuatorType': 'residual', 'optForce': 2.5},
                  'hip_flexion_r': {'actuatorType': 'reserve', 'optForce': 5.0},
                  'hip_adduction_r': {'actuatorType': 'reserve', 'optForce': 5.0},
                  'hip_rotation_r': {'actuatorType': 'reserve', 'optForce': 5.0},
                  'knee_angle_r': {'actuatorType': 'reserve', 'optForce': 5.0},
                  'ankle_angle_r': {'actuatorType': 'reserve', 'optForce': 5.0},
                  'hip_flexion_l': {'actuatorType': 'torque', 'optForce': 300.0},
                  'hip_adduction_l': {'actuatorType': 'torque', 'optForce': 200.0},
                  'hip_rotation_l': {'actuatorType': 'torque', 'optForce': 100.0},
                  'knee_angle_l': {'actuatorType': 'torque', 'optForce': 300.0},
                  'ankle_angle_l': {'actuatorType': 'torque', 'optForce': 200.0},
                  }

    # Prepare model for dynamic optimisation
    # -------------------------------------------------------------------------

    # Construct a model processor to use with the tool
    model_proc = osim.ModelProcessor(os.path.join('..', '..', '..', '..', '..', 'data', participant, 'scaling',
                                                  f'{participant}_{model_type}.osim'))

    # Increase muscle isometric force by a scaling factor to deal with potentially higher muscle forces
    model_proc.append(osim.ModOpScaleMaxIsometricForce(1.5))

    # # Append muscle path set
    # model_proc.append(osim.ModOpReplacePathsWithFunctionBasedPaths(f'{participant}_{model_type}_FunctionBasedPathSet.xml'))

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
        osim.DeGrooteFregly2016Muscle().safeDownCast(musc).get_fiber_damping()
        # # Option for elastic tendons on plantarflexor muscles
        # # TODO: does this work?
        # if musc.getName() in ['gaslat_r', 'gasmed_r', 'soleus_r']:
        #     musc.set_ignore_tendon_compliance(False)
        #     osim.DeGrooteFregly2016Muscle().safeDownCast(musc).set_tendon_compliance_dynamics_mode('implicit')
        # Option to reduce fiber damping
        osim.DeGrooteFregly2016Muscle().safeDownCast(musc).set_fiber_damping(1.0e-3)

    # Print model to file in tracking directory
    opt_model.printToXML(f'{participant}run{speed}_dynamic_optimisation_{model_type}.osim')

    # Set-up Moco inverse simulation
    # -------------------------------------------------------------------------

    # Create inverse tool
    inverse = osim.MocoInverse()
    inverse.setName(f'{participant}run{speed}_{model_type}')

    # Construct a model processor to use with the tool
    model_proc = osim.ModelProcessor(f'{participant}run{speed}_dynamic_optimisation_{model_type}.osim')

    # Append external loads
    model_proc.append(osim.ModOpAddExternalLoads(f'{participant}run{speed}_grf.xml'))

    # # Ignore passive fibre forces
    # model_proc.append(osim.ModOpIgnorePassiveFiberForcesDGF())

    # Scale active force curve width
    model_proc.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))

    # Set model
    inverse.setModel(model_proc)

    # Set kinematics
    inverse.setKinematics(osim.TableProcessor(f'{participant}run{speed}_states.sto'))

    # Set times
    start_time = osim.TimeSeriesTable(f'{participant}run{speed}_states.sto').getIndependentColumn()[0]
    end_time = osim.TimeSeriesTable(f'{participant}run{speed}_states.sto').getIndependentColumn()[-1]
    inverse.set_initial_time(start_time)
    inverse.set_final_time(end_time)

    # Set kinematics to have extra columns (even though this shouldn't be an issue)
    inverse.set_kinematics_allow_extra_columns(True)

    # Convert to Moco study for added flexibility
    # -------------------------------------------------------------------------

    # Get the study and problem
    study = inverse.initialize()
    problem = study.updProblem()

    # Update control effort goal

    # Get a reference to the MocoControlCost goal and set parameters
    effort = osim.MocoControlGoal.safeDownCast(problem.updGoal('excitation_effort'))
    effort.setWeight(globalMuscleControlWeight)
    effort.setExponent(2)

    # Update individual weights in control effort goal
    # Put higher weight on residual use
    effort.setWeightForControlPattern('/forceset/.*_residual', 10.0)
    # Put heavy weight on the reserve actuators
    effort.setWeightForControlPattern('/forceset/.*_reserve', 5.0)
    # Set standard weights on muscle controls
    # This probably doesn't change default but provides an option to set
    effort.setWeightForControlPattern('/forceset/.*_r', 1.0)

    # Define and configure the solver
    # -------------------------------------------------------------------------

    # Get the solver
    solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())
    solver.resetProblem(problem)

    # Set-up timer to track total computation time
    computation_start = time.time()

    # Loop through mesh intervals using mesh refinement approach
    for mesh_int in meshIntervalMuscle:

        # Set solver options
        solver.set_optim_max_iterations(2000)
        solver.set_num_mesh_intervals(mesh_int)
        solver.set_optim_constraint_tolerance(1e-2)
        solver.set_optim_convergence_tolerance(1e-3)
        solver.set_minimize_implicit_auxiliary_derivatives(True)
        solver.set_implicit_auxiliary_derivatives_weight(1.0e-3)
        solver.resetProblem(problem)

        # Create generic initial guess for coarsest mesh interval
        if meshIntervalMuscle.index(mesh_int) == 0:
            # Create initial guess from solver
            initial_guess = solver.createGuess('bounds')
            # Set muscle states and controls to a reasonable mid-point value in guess
            for state in initial_guess.getStateNames():
                if state.endswith('/activation'):
                    initial_guess.setState(state, np.ones(initial_guess.getNumTimes())*0.5)
                if state.endswith('/normalized_tendon_force'):
                    initial_guess.setState(state, np.ones(initial_guess.getNumTimes()) * 0.1)
            for control in initial_guess.getControlNames():
                if control.endswith('_r'):
                    initial_guess.setControl(control, np.ones(initial_guess.getNumTimes())*0.5)
            # Save initial guess to file
            initial_guess.write(f'{participant}run{speed}_initial_guess_mesh-int-{mesh_int}.sto')
            # Set guess in solver
            solver.setGuess(initial_guess)
        else:
            # Create initial guess from solver
            initial_guess = solver.createGuess('bounds')
            # Load the previous coarse interval to fill guess data
            # This seems to work better than setting the guess file for some reason, perhaps because it doesn't
            # include residual and reserve torque values
            prev_mesh = meshIntervalMuscle[meshIntervalMuscle.index(mesh_int)-1]
            prev_solution = osim.MocoTrajectory(
                f'{participant}run{speed}_dynamic_optimisation_{model_type}_solution_mesh-int-{prev_mesh}.sto')
            # Set guess to match number of times
            initial_guess.resampleWithNumTimes(prev_solution.getNumTimes())
            # Fill the guess with the data from the previous solution
            for state in initial_guess.getStateNames():
                initial_guess.setState(state, prev_solution.getState(state).to_numpy())
            for control in initial_guess.getControlNames():
                if not (control.endswith('_residual') or control.endswith('_reserve')):
                    initial_guess.setControl(control, prev_solution.getControl(control).to_numpy())
            for multiplier in initial_guess.getMultiplierNames():
                initial_guess.setMultiplier(multiplier, prev_solution.getMultiplier(multiplier).to_numpy())
            # Save initial guess to file
            initial_guess.write(f'{participant}run{speed}_initial_guess_mesh-int-{mesh_int}.sto')
            # Set guess in solver
            solver.setGuess(initial_guess)

        # Solve the problem
        inverse_solution = study.solve()

        # Write solution to file
        if inverse_solution.isSealed():
            inverse_solution.unseal()
        inverse_solution.write(
            f'{participant}run{speed}_dynamic_optimisation_{model_type}_solution_mesh-int-{mesh_int}.sto')

    # Store final outputs
    # -------------------------------------------------------------------------

    # End computation timer and record
    computation_run_time = round(time.time() - computation_start, 2)

    # Save a dictionary storing computational time
    computation = {'time_s': computation_run_time,
                   'note': f'Dynamic optimisation {model_type} model computation time for {participant} {speed}'}
    with open(f'{participant}run{speed}_dynamic_optimisation_{model_type}_computation_time.pkl', 'wb') as pkl_file:
        pickle.dump(computation, pkl_file)

    # Insert the joint coordinate states into the inverse solution
    # Load the joint coordinate states and solution as a trajectory
    joint_states = osim.TimeSeriesTable(f'{participant}run{speed}_states.sto')
    inverse_traj = osim.MocoTrajectory(
        f'{participant}run{speed}_dynamic_optimisation_{model_type}_solution_mesh-int-{meshIntervalMuscle[-1]}.sto')
    # Resample inverse solution to match the joint states number of rows
    inverse_traj.resampleWithNumTimes(joint_states.getNumRows())
    # Insert joint coordinate values and states
    joint_traj = osim.StatesTrajectory().createFromStatesTable(opt_model, joint_states, True)
    inverse_traj.insertStatesTrajectory(joint_states)

    # Write the full states to file
    inverse_traj.write(f'{participant}run{speed}_dynamic_optimisation_{model_type}_full.sto')

    # Extract muscle forces from solution
    output_paths = osim.StdVectorString()
    output_paths.append('.*tendon_force')
    output_paths.append('.*fiber_force')
    muscle_force_table = osim.analyze(opt_model,
                                      inverse_traj.exportToStatesTable(),
                                      inverse_traj.exportToControlsTable(),
                                      output_paths)
    # Write to file
    osim.STOFileAdapter().write(muscle_force_table,
                                f'{participant}run{speed}_dynamic_optimisation_{model_type}_muscle_forces.sto')

    # Return to home directory
    os.chdir(home_dir)

    # Print out to console as a bookmark in any log file
    print(f'{"*" * 10} FINISHED DYNAMIC OPTIMISATION FOR {participant} {speed} WITH {model_type.upper()} MODEL {"*" * 10}')


# =========================================================================
# Run simulations
# =========================================================================

if __name__ == '__main__':

    # TODO: simple model? Not working great in dynamic optimisation

    # Run marker tracking simulation
    # -------------------------------------------------------------------------
    if runTorqueSim:
        run_marker_tracking()

    # Run static optimisation
    # -------------------------------------------------------------------------
    if runStaticOpt:
        run_static_optimisation('complex')

    # Run dynamic optimisation
    # -------------------------------------------------------------------------
    if runDynamicOpt:
        run_dynamic_optimisation('complex')

    # Exit terminal to avoid any funny business
    # -------------------------------------------------------------------------
    os._exit(00)

# %% ---------- end of run_simulations.py ---------- %% #

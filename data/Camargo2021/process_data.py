# -*- coding: utf-8 -*-
"""

@author:

    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au

    This script runs some initial processing steps to prepare the Camargo2021
    dataset for the subsequent simulations. It does some basic scaling of models
    via the static trial and runs IK over the randomly selected trials for different
    combinations of stair height, direction and transition leg using the provided
    times with the trials.

"""

# =========================================================================
# Import packages
# =========================================================================

import opensim as osim
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import glob

# =========================================================================
# Modify to run different parts of code
# =========================================================================

# Set which processing steps to run
runScaling = True
runIK = True

# =========================================================================
# Set-up
# =========================================================================

# General settings
# -------------------------------------------------------------------------

# # Get participant list from folder
# participant_list = [ii for ii in os.listdir(os.getcwd()) if os.path.isdir(os.path.join(os.getcwd(), ii))]
# Or set manually when testing
participant_list = [
    'AB11',
    'AB12',
]

# Read in trial selection file
selected_trials = pd.read_csv('select-participants_stair_trial-info.csv')

# Load participant info
participant_info = pd.read_csv('participant_info.csv')

# Add the utility geometry path for model visualisation
osim.ModelVisualizer.addDirToGeometrySearchPaths(os.path.join(os.getcwd(), '..', '..', 'model', 'Geometry'))

# Create dictionaries for tools to avoid over-writing
scaleTool_complex = {participantId: osim.ScaleTool() for participantId in participant_list}
scaleTool_simple = {participantId: osim.ScaleTool() for participantId in participant_list}

# Identify static trial trc files
static_files = {participant: glob.glob(os.path.join(participant, 'static*.trc'))[0] for participant in participant_list}

# # Identify stair trials for each participant and put in ik tool
# stair_trials = {participant: selected_trials.loc[
#     selected_trials['participant'] == participant][['direction','stair_height','trans_leg']].apply(
#     lambda row: f"{row['direction']}_{row['stair_height']}_{row['trans_leg']}", axis=1).to_list() for
#                 participant in participant_list}
# ikTool = {participant: {trial: osim.InverseKinematicsTool() for trial in stair_trials[participant]} for participant in participant_list}

# Identify stair trials in data directory for each participant for IK tool
ikTool = {participant: {
    os.path.splitext(os.path.basename(stairs))[0]: osim.InverseKinematicsTool() for stairs in glob.glob(
        os.path.join(participant, 'stair_*.trc'))} for participant in participant_list}

# Plot settings
# -------------------------------------------------------------------------

# Set matplotlib parameters
from matplotlib import rcParams
import matplotlib
matplotlib.use('TkAgg')
plt.ion()

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

# Create measurement set for scaling

# Set the parameters for the measurement sets
measurementSetParams = {
    # Pelvis
    'pelvis_d': {'markerPairs': [['R_ASIS', 'R_PSIS'], ['L_ASIS', 'L_PSIS'], ], 'bodyScale': ['pelvis'], 'axes': 'X'},
    'pelvis_wh': {'markerPairs': [['R_ASIS', 'L_ASIS'], ['R_PSIS', 'L_PSIS'], ], 'bodyScale': ['pelvis'], 'axes': 'YZ'},
    # Right thigh
    'r_thigh': {'markerPairs': [['R_ASIS', 'R_Knee_Lat'], ['R_PSIS', 'R_Knee_Lat']], 'bodyScale': ['femur_r'], 'axes': 'XYZ'},
    # Right patella
    'r_patella': {'markerPairs': [['R_ASIS', 'R_Knee_Lat'], ['R_PSIS', 'R_Knee_Lat'], ], 'bodyScale': ['patella_r'], 'axes': 'XYZ'},
    # Right shank
    'r_tibia': {'markerPairs': [['R_Knee_Lat', 'R_Ankle_Lat'], ], 'bodyScale': ['tibia_r', 'talus_r'], 'axes': 'XYZ'},
    # Right foot
    'r_foot': {'markerPairs': [['R_Heel', 'R_Toe_Tip'],['R_Heel', 'R_Toe_Lat'], ], 'bodyScale': ['calcn_r', 'toes_r'], 'axes': 'XYZ'},
    # Left thigh
    'l_thigh': {'markerPairs': [['L_ASIS', 'L_Knee_Lat'], ['L_PSIS', 'L_Knee_Lat']], 'bodyScale': ['femur_l'], 'axes': 'XYZ'},
    # Left patella
    'l_patella': {'markerPairs': [['L_ASIS', 'L_Knee_Lat'], ['L_PSIS', 'L_Knee_Lat'], ], 'bodyScale': ['patella_l'], 'axes': 'XYZ'},
    # Left shank
    'l_tibia': {'markerPairs': [['L_Knee_Lat', 'L_Ankle_Lat'], ], 'bodyScale': ['tibia_l', 'talus_l'], 'axes': 'XYZ'},
    # Left foot
    'l_foot': {'markerPairs': [['L_Heel', 'L_Toe_Tip'], ['L_Heel', 'L_Toe_Lat'], ], 'bodyScale': ['calcn_l', 'toes_l'], 'axes': 'XYZ'},
}

# Create the measurement set
scaleMeasurementSet_complex = osim.MeasurementSet()
scaleMeasurementSet_simple = osim.MeasurementSet()

# Append the measurements from parameters
for measureName in measurementSetParams.keys():
    # Create the measurement
    measurement = osim.Measurement()
    measurement.setName(measureName)
    # Append the marker pairs
    for ii in range(len(measurementSetParams[measureName]['markerPairs'])):
        measurement.getMarkerPairSet().cloneAndAppend(
            osim.MarkerPair(measurementSetParams[measureName]['markerPairs'][ii][0],
                            measurementSetParams[measureName]['markerPairs'][ii][1]))
    # Append the body scales
    for ii in range(len(measurementSetParams[measureName]['bodyScale'])):
        # Create body scale
        bodyScale = osim.BodyScale()
        bodyScale.setName(measurementSetParams[measureName]['bodyScale'][ii])
        # Create and set axis names
        axes = osim.ArrayStr()
        for jj in range(len(measurementSetParams[measureName]['axes'])):
            axes.append(measurementSetParams[measureName]['axes'][jj])
        bodyScale.setAxisNames(axes)
        # Apppend to body scale set
        measurement.getBodyScaleSet().cloneAndAppend(bodyScale)
    # Append the measurement to the complex set
    scaleMeasurementSet_complex.cloneAndAppend(measurement)
    # Check if needed in the simple set (i.e. patella remove)
    if 'patella' not in measureName:
        scaleMeasurementSet_simple.cloneAndAppend(measurement)

# Create scale task set

# Set the parameters for the scale marker set
markerParams = {
    # Pelvis
    'R_ASIS': {'weight': 5.0}, 'L_ASIS': {'weight': 5.0}, 'R_PSIS': {'weight': 5.0}, 'L_PSIS': {'weight': 5.0},
    # Right thigh
    'R_Thigh_Upper': {'weight': 0.0}, 'R_Thigh_Front': {'weight': 0.0}, 'R_Thigh_Rear': {'weight': 0.0},
    'R_Knee_Lat': {'weight': 10.0},
    # Right shank
    'R_Shank_Upper': {'weight': 0.0}, 'R_Shank_Front': {'weight': 0.0}, 'R_Shank_Rear': {'weight': 0.0},
    'R_Ankle_Lat': {'weight': 10.0},
    # Right foot
    'R_Heel': {'weight': 10.0},
    'R_Toe_Tip': {'weight': 2.5}, 'R_Toe_Med': {'weight': 0.0}, 'R_Toe_Lat': {'weight': 2.5},
    # Left thigh
    'L_Thigh_Upper': {'weight': 0.0}, 'L_Thigh_Front': {'weight': 0.0}, 'L_Thigh_Rear': {'weight': 0.0},
    'L_Knee_Lat': {'weight': 10.0},
    # Left shank
    'L_Shank_Upper': {'weight': 0.0}, 'L_Shank_Front': {'weight': 0.0}, 'L_Shank_Rear': {'weight': 0.0},
    'L_Ankle_Lat': {'weight': 10.0},
    # Left foot
    'L_Heel': {'weight': 10.0},
    'L_Toe_Tip': {'weight': 2.5}, 'L_Toe_Med': {'weight': 0.0}, 'L_Toe_Lat': {'weight': 2.5},
}

# Set the parameters for the scale joint set
jointParams = {'pelvis_tilt': 0.001, 'pelvis_list': 0.001, 'pelvis_rotation': 0.001,
               'hip_flexion_r': 0.001, 'hip_adduction_r': 0.001, 'hip_rotation_r': 0.001,
               'knee_angle_r': 0.001, 'ankle_angle_r': 0.001, 'subtalar_angle_r': 0.001,
               'hip_flexion_l': 0.001, 'hip_adduction_l': 0.001, 'hip_rotation_l': 0.001,
               'knee_angle_l': 0.001, 'ankle_angle_l': 0.001, 'subtalar_angle_l': 0.001,
               }

# Create the task set
scaleTaskSet_complex = osim.IKTaskSet()
scaleTaskSet_simple = osim.IKTaskSet()

# Append the tasks from the marker parameters
for taskName in markerParams.keys():
    # Create the task and add details
    task = osim.IKMarkerTask()
    task.setName(taskName)
    task.setWeight(markerParams[taskName]['weight'])
    if markerParams[taskName]['weight'] == 0.0:
        task.setApply(False)
    # Append to task set
    scaleTaskSet_complex.cloneAndAppend(task)
    scaleTaskSet_simple.cloneAndAppend(task)

# Append the tasks from the joint parameters
for jointName in jointParams:
    # Create the task and add details
    jointTask = osim.IKCoordinateTask()
    jointTask.setName(jointName)
    jointTask.setWeight(jointParams[jointName])
    # Append to task set
    scaleTaskSet_complex.cloneAndAppend(jointTask)
    if not 'subtalar' in jointName:
        scaleTaskSet_simple.cloneAndAppend(jointTask)

# Create the IK task set for tracking

# Set the parameters for the IK task sets
ikTaskSetParams = {
    # Pelvis
    'R_ASIS': {'weight': 5.0}, 'L_ASIS': {'weight': 5.0}, 'R_PSIS': {'weight': 5.0}, 'L_PSIS': {'weight': 5.0},
    # Right thigh
    'R_Thigh_Upper': {'weight': 5.0}, 'R_Thigh_Front': {'weight': 5.0}, 'R_Thigh_Rear': {'weight': 5.0},
    'R_Knee_Lat': {'weight': 2.5},
    # Right shank
    'R_Shank_Upper': {'weight': 5.0}, 'R_Shank_Front': {'weight': 5.0}, 'R_Shank_Rear': {'weight': 5.0},
    'R_Ankle_Lat': {'weight': 2.5},
    # Right foot
    'R_Heel': {'weight': 10.0},
    'R_Toe_Tip': {'weight': 5.0}, 'R_Toe_Med': {'weight': 5.0}, 'R_Toe_Lat': {'weight': 5.0},
    # Left thigh
    'L_Thigh_Upper': {'weight': 5.0}, 'L_Thigh_Front': {'weight': 5.0}, 'L_Thigh_Rear': {'weight': 5.0},
    'L_Knee_Lat': {'weight': 2.5},
    # Left shank
    'L_Shank_Upper': {'weight': 5.0}, 'L_Shank_Front': {'weight': 5.0}, 'L_Shank_Rear': {'weight': 5.0},
    'L_Ankle_Lat': {'weight': 2.5},
    # Left foot
    'L_Heel': {'weight': 10.0},
    'L_Toe_Tip': {'weight': 5.0}, 'L_Toe_Med': {'weight': 5.0}, 'L_Toe_Lat': {'weight': 5.0},
}

# Create the task set
ikTaskSet = osim.IKTaskSet()

# Append the tasks from the parameters
for taskName in ikTaskSetParams.keys():
    # Create the task and add details
    task = osim.IKMarkerTask()
    task.setName(taskName)
    task.setWeight(ikTaskSetParams[taskName]['weight'])
    if ikTaskSetParams[taskName]['weight'] == 0.0:
        task.setApply(False)
    # Append to task set
    ikTaskSet.cloneAndAppend(task)

# =========================================================================
# Define functions
# =========================================================================

# Scale participant models
# -------------------------------------------------------------------------
def run_scaling(participant_id):

    """
    :param participant_id: participant ID to run scaling for
    :return:
    """

    # =========================================================================
    # Organise files for scaling
    # =========================================================================

    # Create scaling directory for files
    os.makedirs(os.path.join(participant_id, 'scaling'), exist_ok=True)

    # =========================================================================
    # Set-up and run the scale tool for the two models
    # =========================================================================

    # Set participant mass and height
    mass_kg = participant_info.loc[participant_info['Subject']==participant_id]['Weight'].values[0]
    height_m = participant_info.loc[participant_info['Subject'] == participant_id]['Height'].values[0]
    scaleTool_complex[participant_id].setSubjectMass(mass_kg)
    scaleTool_simple[participant_id].setSubjectMass(mass_kg)

    # Set generic model file
    scaleTool_complex[participant_id].getGenericModelMaker().setModelFileName(
        os.path.join('..', '..', 'model', 'Uhlrich2022_LowerLimb_Camargo2021.osim'))
    scaleTool_simple[participant_id].getGenericModelMaker().setModelFileName(
        os.path.join('..', '..', 'model', 'Denton2023_LowerLimb_Camargo2021.osim'))

    # Set measurement set in model scaler
    scaleTool_complex[participant_id].getModelScaler().setMeasurementSet(scaleMeasurementSet_complex)
    scaleTool_simple[participant_id].getModelScaler().setMeasurementSet(scaleMeasurementSet_simple)

    # Set scale tasks in tool
    for ii in range(scaleTaskSet_complex.getSize()):
        scaleTool_complex[participant_id].getMarkerPlacer().getIKTaskSet().cloneAndAppend(scaleTaskSet_complex.get(ii))
    for ii in range(scaleTaskSet_simple.getSize()):
        scaleTool_simple[participant_id].getMarkerPlacer().getIKTaskSet().cloneAndAppend(scaleTaskSet_simple.get(ii))

    # Set marker file
    scaleTool_complex[participant_id].getMarkerPlacer().setMarkerFileName(static_files[participant_id])
    scaleTool_complex[participant_id].getModelScaler().setMarkerFileName(static_files[participant_id])
    scaleTool_simple[participant_id].getMarkerPlacer().setMarkerFileName(static_files[participant_id])
    scaleTool_simple[participant_id].getModelScaler().setMarkerFileName(static_files[participant_id])

    # Set options
    scaleTool_complex[participant_id].getModelScaler().setPreserveMassDist(True)
    scaleTool_simple[participant_id].getModelScaler().setPreserveMassDist(True)
    scaleOrder = osim.ArrayStr()
    scaleOrder.set(0, 'measurements')
    scaleTool_complex[participant_id].getModelScaler().setScalingOrder(scaleOrder)
    scaleTool_simple[participant_id].getModelScaler().setScalingOrder(scaleOrder)

    # Set time ranges
    initial_time = osim.TimeSeriesTableVec3(static_files[participant_id]).getIndependentColumn()[0]
    final_time = osim.TimeSeriesTableVec3(static_files[participant_id]).getIndependentColumn()[-1]
    timeRange = osim.ArrayDouble()
    timeRange.set(0, initial_time)
    timeRange.set(1, final_time)
    scaleTool_complex[participant_id].getMarkerPlacer().setTimeRange(timeRange)
    scaleTool_complex[participant_id].getModelScaler().setTimeRange(timeRange)
    scaleTool_simple[participant_id].getMarkerPlacer().setTimeRange(timeRange)
    scaleTool_simple[participant_id].getModelScaler().setTimeRange(timeRange)

    # Set output files
    scaleTool_complex[participant_id].getModelScaler().setOutputModelFileName(
        os.path.join(participant_id, 'scaling', f'{participant_id}_scaled_complex.osim'))
    scaleTool_complex[participant_id].getModelScaler().setOutputScaleFileName(
        os.path.join(participant_id, 'scaling', f'{participant_id}_scaleSet_complex.xml'))
    scaleTool_simple[participant_id].getModelScaler().setOutputModelFileName(
        os.path.join(participant_id, 'scaling', f'{participant_id}_scaled_simple.osim'))
    scaleTool_simple[participant_id].getModelScaler().setOutputScaleFileName(
        os.path.join(participant_id, 'scaling', f'{participant_id}_scaleSet_simple.xml'))

    # Set marker adjustment parameters
    scaleTool_complex[participant_id].getMarkerPlacer().setOutputMotionFileName(
        os.path.join(participant_id, 'scaling', f'{participant_id}_staticMotion_complex.mot'))
    scaleTool_complex[participant_id].getMarkerPlacer().setOutputModelFileName(
        os.path.join(participant_id, 'scaling', f'{participant_id}_scaledAdjusted_complex.osim'))
    scaleTool_simple[participant_id].getMarkerPlacer().setOutputMotionFileName(
        os.path.join(participant_id, 'scaling', f'{participant_id}_staticMotion_simple.mot'))
    scaleTool_simple[participant_id].getMarkerPlacer().setOutputModelFileName(
        os.path.join(participant_id, 'scaling', f'{participant_id}_scaledAdjusted_simple.osim'))

    # Save and run scale tool
    scaleTool_complex[participant_id].printToXML(
        os.path.join(participant_id, 'scaling', f'{participant_id}_scaleSetup_complex.xml'))
    scaleTool_simple[participant_id].printToXML(
        os.path.join(participant_id, 'scaling', f'{participant_id}_scaleSetup_simple.xml'))
    scaleTool_complex[participant_id].run()
    scaleTool_simple[participant_id].run()

    # =========================================================================
    # Adjust the models
    # =========================================================================

    # Load the scaled models back in
    scaledModel_complex = osim.Model(
        os.path.join(participant_id, 'scaling', f'{participant_id}_scaledAdjusted_complex.osim'))
    scaledModel_simple = osim.Model(
        os.path.join(participant_id, 'scaling', f'{participant_id}_scaledAdjusted_simple.osim'))

    # Set model name
    scaledModel_complex.setName(participant_id + '_complex')
    scaledModel_simple.setName(participant_id + '_simple')

    # Scale model muscle forces according to height-mass relationship
    # -------------------------------------------------------------------------

    # Get generic model mass and set generic height for 3D model (this is the same as 2D model)
    genModel_complex = osim.Model(os.path.join('..', '..', 'model', 'Uhlrich2022_LowerLimb_Camargo2021.osim'))
    genModel_simple = osim.Model(os.path.join('..', '..', 'model', 'Denton2023_LowerLimb_Camargo2021.osim'))
    genModelMass_complex = np.sum(
        [genModel_complex.getBodySet().get(bodyInd).getMass() for bodyInd in range(genModel_complex.getBodySet().getSize())])
    genModelMass_simple = np.sum(
        [genModel_simple.getBodySet().get(bodyInd).getMass() for bodyInd in range(genModel_simple.getBodySet().getSize())])
    genModelHeight_complex = 1.70
    genModelHeight_simple = 1.70

    # Get muscle volume totals based on mass and heights with linear equation
    genericMuscVol_complex = 47.05 * genModelMass_complex * genModelHeight_complex + 1289.6
    genericMuscVol_simple = 47.05 * genModelMass_simple * genModelHeight_simple + 1289.6
    scaledMuscVol = 47.05 * mass_kg * height_m + 1289.6

    # Loop through all muscles and scale according to volume and muscle parameters
    # Use this opportunity to also update contraction velocity as well

    # Set in complex model
    for muscInd in range(scaledModel_complex.getMuscles().getSize()):
        # Get current muscle name
        muscName = scaledModel_complex.getMuscles().get(muscInd).getName()
        # Get optimal fibre length for muscle from each model
        genericL0 = genModel_complex.getMuscles().get(muscName).getOptimalFiberLength()
        scaledL0 = scaledModel_complex.getMuscles().get(muscName).getOptimalFiberLength()
        # Set force scale factor
        forceScaleFactor = (scaledMuscVol / genericMuscVol_complex) / (scaledL0 / genericL0)
        # Scale current muscle strength
        scaledModel_complex.getMuscles().get(muscInd).setMaxIsometricForce(
            forceScaleFactor * scaledModel_complex.getMuscles().get(muscInd).getMaxIsometricForce())
        # Update max contraction velocity
        scaledModel_complex.getMuscles().get(muscInd).setMaxContractionVelocity(30.0)

    # Set in simple model
    for muscInd in range(scaledModel_simple.getMuscles().getSize()):
        # Get current muscle name
        muscName = scaledModel_simple.getMuscles().get(muscInd).getName()
        # Get optimal fibre length for muscle from each model
        genericL0 = genModel_simple.getMuscles().get(muscName).getOptimalFiberLength()
        scaledL0 = scaledModel_simple.getMuscles().get(muscName).getOptimalFiberLength()
        # Set force scale factor
        forceScaleFactor = (scaledMuscVol / genericMuscVol_simple) / (scaledL0 / genericL0)
        # Scale current muscle strength
        scaledModel_simple.getMuscles().get(muscInd).setMaxIsometricForce(
            forceScaleFactor * scaledModel_simple.getMuscles().get(muscInd).getMaxIsometricForce())
        # Update max contraction velocity
        scaledModel_simple.getMuscles().get(muscInd).setMaxContractionVelocity(30.0)

    # Remove the left side muscles from the models
    # They won't be necessary and this will speed up the optimisations later
    # -------------------------------------------------------------------------

    # Complex model

    # Loop through force set to identify muscles to remove
    remove_ind = []
    for ii in range(scaledModel_complex.getForceSet().getSize()):
        # Check for left side muscle
        if 'Muscle' in scaledModel_complex.getForceSet().get(ii).getConcreteClassName() and \
                scaledModel_complex.getForceSet().get(ii).getName().endswith('_l'):
            # Remove this index
            remove_ind.append(ii)

    # Remove muscles from model
    # Each time a force is removed the index related to the force set is dropped by one each time
    remove_counter = 0
    for ii in remove_ind:
        scaledModel_complex.updForceSet().remove(ii-remove_counter)
        remove_counter += 1

    # Simple model

    # Loop through force set to identify muscles to remove
    remove_ind = []
    for ii in range(scaledModel_simple.getForceSet().getSize()):
        # Check for left side muscle
        if 'Muscle' in scaledModel_simple.getForceSet().get(ii).getConcreteClassName() and \
                scaledModel_simple.getForceSet().get(ii).getName().endswith('_l'):
            # Remove this index
            remove_ind.append(ii)

    # Remove muscles from model
    # Each time a force is removed the index related to the force set is dropped by one each time
    remove_counter = 0
    for ii in remove_ind:
        scaledModel_simple.updForceSet().remove(ii - remove_counter)
        remove_counter += 1

    # Update the muscle parameters in the models
    # -------------------------------------------------------------------------

    # Put into a model processor and append operators for complex model
    model_proc_complex = osim.ModelProcessor(scaledModel_complex)
    model_proc_complex.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
    model_proc_complex.append(osim.ModOpIgnoreTendonCompliance())
    weld_vector = osim.StdVectorString()
    [weld_vector.append(joint) for joint in ['mtp_r', 'mtp_l',
                                             # 'subtalar_r', 'subtalar_l',
                                             ]]
    model_proc_complex.append(osim.ModOpReplaceJointsWithWelds(weld_vector))

    # Put into a model processor and append operators for simple model
    model_proc_simple = osim.ModelProcessor(scaledModel_simple)
    model_proc_simple.append(osim.ModOpIgnoreTendonCompliance())
    model_proc_simple.append(osim.ModOpReplaceJointsWithWelds(weld_vector))

    # Process and overwrite the original scaled model
    scaledModel_complex = model_proc_complex.process()
    scaledModel_simple = model_proc_simple.process()

    # Update colouring of the quadriceps muscles for presentation
    # Complex model
    for muscInd in range(scaledModel_complex.getMuscles().getSize()):
        if scaledModel_complex.getMuscles().get(muscInd).getName() in \
            ['recfem_r', 'vasint_r', 'vaslat_r', 'vasmed_r']:
            # Colour muscles blue
            scaledModel_complex.getMuscles().get(muscInd).getGeometryPath().get_Appearance().set_color(
                osim.Vec3(31 / 255, 68 / 255, 156 / 255))
        else:
            # Shift to slightly better contrasting red
            scaledModel_complex.getMuscles().get(muscInd).getGeometryPath().get_Appearance().set_color(
                osim.Vec3(240 / 255, 80 / 255, 57 / 255))
    # Simple model
    for muscInd in range(scaledModel_simple.getMuscles().getSize()):
        if scaledModel_simple.getMuscles().get(muscInd).getName() in \
                ['rect_fem_r', 'vasti_r']:
            # Colour muscles blue
            scaledModel_simple.getMuscles().get(muscInd).getGeometryPath().get_Appearance().set_color(
                osim.Vec3(31 / 255, 68 / 255, 156 / 255))
        else:
            # Shift to slightly better contrasting red
            scaledModel_simple.getMuscles().get(muscInd).getGeometryPath().get_Appearance().set_color(
                osim.Vec3(240 / 255, 80 / 255, 57 / 255))

    # Finalise and print to file
    # -------------------------------------------------------------------------

    # Finalise model connections
    scaledModel_complex.finalizeConnections()
    scaledModel_simple.finalizeConnections()

    # Print to file (overwrites original adjusted model)
    scaledModel_complex.printToXML(
        os.path.join(participant_id, 'scaling', f'{participant_id}_complex.osim'))
    scaledModel_simple.printToXML(
        os.path.join(participant_id, 'scaling', f'{participant_id}_simple.osim'))


# Run inverse kinematics using complex model
# -------------------------------------------------------------------------
def run_ik(participant_id, trial):

    """
    :param participant_id: participant ID to run IK for
    :param trial: name of the trial to run IK for
    :return:

    """

    # =========================================================================
    # Organise files for inverse kinematics
    # =========================================================================

    # Identify trial label for present trial
    trial_params = selected_trials.loc[
        (selected_trials['participant'] == participant) &
        (selected_trials['trial_name'] == trial)][['direction','stair_height','trans_leg']].values[0]
    trial_label = '_'.join(str(tt) for tt in trial_params)

    # Create IK directory for files
    os.makedirs(os.path.join(participant_id, 'ik'), exist_ok=True)
    os.makedirs(os.path.join(participant_id, 'ik', trial_label), exist_ok=True)

    # # Identify the file name associated with this trial
    # # Also identify the trial timings from same info
    # # Order of trial label is direction, stair height, transition leg
    # trial_split = trial.split('_')
    # trial_data = selected_trials.loc[
    #     (selected_trials['participant'] == participant_id) &
    #     (selected_trials['direction'] == trial_split[0]) &
    #     (selected_trials['stair_height'] == int(trial_split[1])) &
    #     (selected_trials['trans_leg'] == trial_split[2]),]
    # trial_name = trial_data['trial_name'].values[0]
    # # trial_time = trial_data[['start_time','end_time']].values[0]

    # Load in GRF trial data
    grf_trial = osim.TimeSeriesTable(os.path.join(participant_id, f'{trial}_grf.mot'))

    # =========================================================================
    # Run inverse kinematics on trial
    # =========================================================================

    # Set model
    ikTool[participant_id][trial].set_model_file(
        os.path.join(participant_id, 'scaling', f'{participant_id}_complex.osim'))

    # Set task set
    for taskInd in range(ikTaskSet.getSize()):
        ikTool[participant_id][trial].getIKTaskSet().adoptAndAppend(ikTaskSet.get(taskInd))

    # Set to report marker locations
    ikTool[participant_id][trial].set_report_marker_locations(True)

    # Set the marker file
    ikTool[participant_id][trial].setMarkerDataFileName(os.path.join(participant_id, f'{trial}.trc'))

    # Use force plate data and trial direction to identify appropriate segment
    force_threshold = 20.0
    frame_threshold = 200
    if trial_params[0] == 'ascent':
        # Take from first contact on FP5 up until first instance of leaving FP1
        # Find first contact index for FP5 and associated time
        above = grf_trial.getDependentColumn('FP5_vy').to_numpy() >= force_threshold
        padded = np.pad(above, (1, 1), 'constant', constant_values=(False,))
        diff = np.diff(padded.astype(int))
        start_idx = np.where(diff == 1)[0]
        end_idx = np.where(diff == -1)[0]
        run_lengths = end_idx - start_idx
        valid_starts = start_idx[run_lengths >= frame_threshold]
        valid_ends = end_idx[run_lengths >= frame_threshold]
        start_time = grf_trial.getIndependentColumn()[valid_starts[0]]
        # Find first contact index leaving plate for FP1 and associated time
        above = grf_trial.getDependentColumn('FP1_vy').to_numpy() >= force_threshold
        padded = np.pad(above, (1, 1), 'constant', constant_values=(False,))
        diff = np.diff(padded.astype(int))
        start_idx = np.where(diff == 1)[0]
        end_idx = np.where(diff == -1)[0]
        run_lengths = end_idx - start_idx
        valid_starts = start_idx[run_lengths >= frame_threshold]
        valid_ends = end_idx[run_lengths >= frame_threshold]
        end_time = grf_trial.getIndependentColumn()[valid_ends[0]]
    elif trial_params[0] == 'descent':
        # Take from last contact on FP1 up until last instance of leaving FP5
        # Find last contact index for FP1 and associated time
        above = grf_trial.getDependentColumn('FP1_vy').to_numpy() >= force_threshold
        padded = np.pad(above, (1, 1), 'constant', constant_values=(False,))
        diff = np.diff(padded.astype(int))
        start_idx = np.where(diff == 1)[0]
        end_idx = np.where(diff == -1)[0]
        run_lengths = end_idx - start_idx
        valid_starts = start_idx[run_lengths >= frame_threshold]
        valid_ends = end_idx[run_lengths >= frame_threshold]
        start_time = grf_trial.getIndependentColumn()[valid_starts[-1]]
        # Find last contact index leaving plate for FP5 and associated time
        above = grf_trial.getDependentColumn('FP5_vy').to_numpy() >= force_threshold
        padded = np.pad(above, (1, 1), 'constant', constant_values=(False,))
        diff = np.diff(padded.astype(int))
        start_idx = np.where(diff == 1)[0]
        end_idx = np.where(diff == -1)[0]
        run_lengths = end_idx - start_idx
        valid_starts = start_idx[run_lengths >= frame_threshold]
        valid_ends = end_idx[run_lengths >= frame_threshold]
        end_time = grf_trial.getIndependentColumn()[valid_ends[-1]]

    # Set times in IK tool
    ikTool[participant_id][trial].setStartTime(start_time)
    ikTool[participant_id][trial].setEndTime(end_time)

    # Set output filename (relative to setup file location)
    ikTool[participant_id][trial].setOutputMotionFileName(
        os.path.join(participant_id, 'ik', trial_label, f'{participant_id}_{trial_label}_ik_complex.mot'))

    # Save IK tool to file
    ikTool[participant_id][trial].printToXML(f'{participant_id}_{trial_label}_ikSetup.xml')

    # Bring the tool back in and run it (this seems to avoid Python kernel crashing)
    ikRun = osim.InverseKinematicsTool(f'{participant_id}_{trial_label}_ikSetup.xml')
    ikRun.run()

    # Rename supplementary marker outputs
    shutil.move('_ik_marker_errors.sto',
                os.path.join(participant_id, 'ik', trial_label, f'{participant_id}_{trial_label}_ikMarkerErrors_complex.sto'))
    shutil.move('_ik_model_marker_locations.sto',
                os.path.join(participant_id, 'ik', trial_label, f'{participant_id}_{trial_label}_ikModelMarkerLocations_complex.sto'))
    shutil.move(f'{participant_id}_{trial_label}_ikSetup.xml',
                os.path.join(participant_id, 'ik', trial_label, f'{participant_id}_{trial_label}_ikSetup_complex.xml'))

    # Filter IK data
    # Uses 6Hz low pass filter as in original paper
    # Use table processor to assist with filtering
    table_proc = osim.TableProcessor(
        os.path.join(participant_id, 'ik', trial_label, f'{participant_id}_{trial_label}_ik_complex.mot'))
    ik_model = osim.Model(os.path.join(participant_id, 'scaling', f'{participant_id}_complex.osim'))
    ik_model.initSystem()
    # Add table operators
    table_proc.append(osim.TabOpLowPassFilter(6))
    table_proc.append(osim.TabOpConvertDegreesToRadians())
    table_proc.append(osim.TabOpUseAbsoluteStateNames())
    # Process table
    ik_table = table_proc.process(ik_model)
    # Trim back to original times
    orig_start = osim.TimeSeriesTable(
        os.path.join(participant_id, 'ik', trial_label, f'{participant_id}_{trial_label}_ik_complex.mot')
    ).getIndependentColumn()[0]
    orig_end = osim.TimeSeriesTable(
        os.path.join(participant_id, 'ik', trial_label, f'{participant_id}_{trial_label}_ik_complex.mot')
    ).getIndependentColumn()[-1]
    ik_table.trim(orig_start, orig_end)
    # Write clean IK data to file
    osim.STOFileAdapter().write(ik_table,
                                os.path.join(participant_id, 'ik', trial_label,
                                             f'{participant_id}_{trial_label}_ik_complex_filt.mot'))

    # Create a version of the IK that works with the simple model
    # This mainly requires inverting the knee angle
    # -------------------------------------------------------------------------

    # Read in the IK file
    ik_data = osim.TimeSeriesTable(
        os.path.join(participant_id, 'ik', trial_label, f'{participant_id}_{trial_label}_ik_complex_filt.mot'))

    # Set columns to remove
    remove_cols = [
        #'subtalar_angle_r', 'subtalar_angle_l',
        '/jointset/patellofemoral_r/knee_angle_r_beta/value',
        '/jointset/patellofemoral_l/knee_angle_l_beta/value'
    ]

    # Remove the columns
    for col in remove_cols:
        ik_data.removeColumn(col)

    # Invert the knee angle data
    # Find the column index of the knee angles
    knee_ind = [
        ii for ii in range(len(ik_data.getColumnLabels())) if '/knee_angle_' in ik_data.getColumnLabels()[ii]
    ]
    # Loop through rows and invert the knee data
    for ii in range(ik_data.getNumRows()):
        # Get row and time in numpy format
        row_data = ik_data.getRowAtIndex(ii).to_numpy()
        row_time = ik_data.getIndependentColumn()[ii]
        # Invert knee angle in row data
        for kk in knee_ind:
            row_data[kk] = row_data[kk] * -1
        # Create new row vector to set in IK data
        new_row = osim.RowVector().createFromMat(row_data)
        # Set new row at index in data
        ik_data.setRowAtIndex(ii, new_row)

    # Rename to state label for the knee joint
    ik_cols = ik_data.getColumnLabels()
    new_ik_cols = [col.replace('/walker_knee_','/knee_') if '/walker_knee_' in col else col for col in ik_cols]
    ik_data.setColumnLabels(new_ik_cols)

    # Write the new IK data to file
    osim.STOFileAdapter().write(ik_data,
                                os.path.join(participant_id, 'ik', trial_label,
                                             f'{participant_id}_{trial_label}_ik_simple_filt.mot'))


# =========================================================================
# Process experimental data
# =========================================================================

if __name__ == '__main__':

    # Run scaling
    # -------------------------------------------------------------------------
    if runScaling:
        # Loop through participants
        for participant in participant_list:
            run_scaling(participant)

    # Run inverse kinematics
    # -------------------------------------------------------------------------
    if runIK:
        # Loop through participants
        for participant in participant_list:
            # Identify trial names for participant
            trial_names = [
                os.path.splitext(os.path.basename(ff))[0] for ff in glob.glob(
                    os.path.join(participant, 'stair_*.trc'))]
            # Loop through trials
            for trial_name in trial_names:
                run_ik(participant, trial_name)

    # Finalise and exit kernel
    # -------------------------------------------------------------------------

    # Doing this seems to avoid an error code when completing the script run
    os._exit(00)

# %% ---------- end of process_data.py ---------- %% #

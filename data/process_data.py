# -*- coding: utf-8 -*-
"""

@author:

    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au

    This script runs some initial processing steps to prepare the Fukuchi2017
    dataset for the subsequent simulations.

    TODO:
        > Muscle paths don't fit that well with polynomials...probably not great to use

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

# =========================================================================
# Modify to run different parts of code
# =========================================================================

# Set which processing steps to run
runScaling = True
runIK = True
runMusclePathTool = False  # TODO: might not use...

# =========================================================================
# Set-up
# =========================================================================

# General settings
# -------------------------------------------------------------------------

# Get participant list from folder
participant_list = [ii for ii in os.listdir(os.getcwd()) if os.path.isdir(os.path.join(os.getcwd(), ii))]

# Read in participant info
participantInfo = pd.read_csv('participantInfo.csv')

# Set the list of speed conditions to process
# Modify this if you want to isolate different running speeds
# The present work focused on the slowest and fastest running speeds
speedList = [
    'T25',   # 2.5 m/s
    'T35',   # 3.5 m/s
    'T45',   # 4.5 m/s
    ]

# Add the utility geometry path for model visualisation
osim.ModelVisualizer.addDirToGeometrySearchPaths(os.path.join(os.getcwd(), '..', 'model', 'Geometry'))

# Create dictionaries for tools to avoid over-writing
scaleTool_complex = {participantId: osim.ScaleTool() for participantId in participant_list}
scaleTool_simple = {participantId: osim.ScaleTool() for participantId in participant_list}
ikTool = {participantId: {speed: osim.InverseKinematicsTool() for speed in speedList} for participantId in participant_list}

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
    'pelvis_d': {'markerPairs': [['R.ASIS', 'R.PSIS'], ['L.ASIS', 'L.PSIS'], ], 'bodyScale': ['pelvis'], 'axes': 'X'},
    'pelvis_h': {'markerPairs': [['R.ASIS', 'R.GTR'], ['L.ASIS', 'L.GTR'], ['R.PSIS', 'R.GTR'], ['L.PSIS', 'L.GTR'],
                                 ['R.Iliac.Crest', 'R.GTR'], ['L.Iliac.Crest', 'L.GTR'], ], 'bodyScale': ['pelvis'], 'axes': 'Y'},
    'pelvis_w': {'markerPairs': [['R.ASIS', 'L.ASIS'], ['R.PSIS', 'L.PSIS'], ], 'bodyScale': ['pelvis'], 'axes': 'Z'},
    # Right thigh
    'r_thigh_length': {'markerPairs': [['R.GTR', 'R.Knee'], ], 'bodyScale': ['femur_r'], 'axes': 'Y'},
    'r_thigh_breadth': {'markerPairs': [['R.Knee', 'R.Knee.Medial'], ], 'bodyScale': ['femur_r'], 'axes': 'XZ'},
    # Right patella
    'r_patella': {'markerPairs': [['R.GTR', 'R.Knee'], ], 'bodyScale': ['patella_r'], 'axes': 'XYZ'},
    # Right shank
    'r_tibia_length': {'markerPairs': [['R.HF', 'R.Ankle'], ], 'bodyScale': ['tibia_r', 'talus_r'], 'axes': 'Y'},
    'r_tibia_breadth': {'markerPairs': [['R.Ankle', 'R.Ankle.Medial'], ], 'bodyScale': ['tibia_r', 'talus_r'], 'axes': 'XZ'},
    # Right foot
    'r_heel': {'markerPairs': [['R.Ankle', 'R.Ankle.Medial'], ], 'bodyScale': ['calcn_r', 'toes_r'], 'axes': 'Z'},
    'r_foot_length': {'markerPairs': [['R.Heel.Top', 'R.MT2'], ], 'bodyScale': ['calcn_r', 'toes_r'], 'axes': 'X'},
    'r_foot_height': {'markerPairs': [['R.Heel.Top', 'R.Heel.Bottom'], ], 'bodyScale': ['calcn_r', 'toes_r'], 'axes': 'Y'},
    # Left thigh
    'l_thigh_length': {'markerPairs': [['L.GTR', 'L.Knee'], ], 'bodyScale': ['femur_l'], 'axes': 'Y'},
    'l_thigh_breadth': {'markerPairs': [['L.Knee', 'L.Knee.Medial'], ], 'bodyScale': ['femur_l'], 'axes': 'XZ'},
    # Left patella
    'l_patella': {'markerPairs': [['L.GTR', 'L.Knee'], ], 'bodyScale': ['patella_l'], 'axes': 'XYZ'},
    # Left shank
    'l_tibia_length': {'markerPairs': [['L.HF', 'L.Ankle'], ], 'bodyScale': ['tibia_l', 'talus_l'], 'axes': 'Y'},
    'l_tibia_breadth': {'markerPairs': [['L.Ankle', 'L.Ankle.Medial'], ], 'bodyScale': ['tibia_l', 'talus_l'], 'axes': 'XZ'},
    # Left foot
    'l_heel': {'markerPairs': [['L.Ankle', 'L.Ankle.Medial'], ], 'bodyScale': ['calcn_l', 'toes_l'], 'axes': 'Z'},
    'l_foot_length': {'markerPairs': [['L.Heel.Top', 'L.MT2'], ], 'bodyScale': ['calcn_l', 'toes_l'], 'axes': 'X'},
    'l_foot_height': {'markerPairs': [['L.Heel.Top', 'L.Heel.Bottom'], ], 'bodyScale': ['calcn_l', 'toes_l'], 'axes': 'Y'},
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
    'R.ASIS': {'weight': 10.0}, 'L.ASIS': {'weight': 10.0}, 'R.PSIS': {'weight': 10.0}, 'L.PSIS': {'weight': 10.0},
    'R.Iliac.Crest': {'weight': 0.0}, 'L.Iliac.Crest': {'weight': 0.0},
    # Right thigh
    'R.GTR': {'weight': 5.0},
    'R.Thigh.Top.Lateral': {'weight': 0.0}, 'R.Thigh.Bottom.Lateral': {'weight': 0.0},
    'R.Thigh.Top.Medial': {'weight': 0.0}, 'R.Thigh.Bottom.Medial': {'weight': 0.0},
    'R.Knee': {'weight': 5.0}, 'R.Knee.Medial': {'weight': 5.0},
    # Right shank
    'R.HF': {'weight': 2.5}, 'R.TT': {'weight': 2.5},
    'R.Shank.Top.Lateral': {'weight': 0.0}, 'R.Shank.Bottom.Lateral': {'weight': 0.0},
    'R.Shank.Top.Medial': {'weight': 0.0}, 'R.Shank.Bottom.Medial': {'weight': 0.0},
    'R.Ankle': {'weight': 10.0}, 'R.Ankle.Medial': {'weight': 10.0},
    # Right foot
    'R.Heel.Top': {'weight': 5.0}, 'R.Heel.Bottom': {'weight': 5.0}, 'R.Heel.Lateral': {'weight': 0.0},
    'R.MT1': {'weight': 2.5}, 'R.MT2': {'weight': 2.5}, 'R.MT5': {'weight': 2.5},
    # Left thigh
    'L.GTR': {'weight': 5.0},
    'L.Thigh.Top.Lateral': {'weight': 0.0}, 'L.Thigh.Bottom.Lateral': {'weight': 0.0},
    'L.Thigh.Top.Medial': {'weight': 0.0}, 'L.Thigh.Bottom.Medial': {'weight': 0.0},
    'L.Knee': {'weight': 5.0}, 'L.Knee.Medial': {'weight': 5.0},
    # Left shank
    'L.HF': {'weight': 2.5}, 'L.TT': {'weight': 2.5},
    'L.Shank.Top.Lateral': {'weight': 0.0}, 'L.Shank.Bottom.Lateral': {'weight': 0.0},
    'L.Shank.Top.Medial': {'weight': 0.0}, 'L.Shank.Bottom.Medial': {'weight': 0.0},
    'L.Ankle': {'weight': 10.0}, 'L.Ankle.Medial': {'weight': 10.0},
    # Left foot
    'L.Heel.Top': {'weight': 5.0}, 'L.Heel.Bottom': {'weight': 5.0}, 'L.Heel.Lateral': {'weight': 0.0},
    'L.MT1': {'weight': 2.5}, 'L.MT2': {'weight': 2.5}, 'L.MT5': {'weight': 2.5},
}

# Set the parameters for the scale joint set
jointParams = {'pelvis_tilt': 0.001, 'pelvis_list': 0.001, 'pelvis_rotation': 0.001,
               'hip_flexion_r': 0.001, 'hip_adduction_r': 0.001, 'hip_rotation_r': 0.001,
               'knee_angle_r': 0.001, 'ankle_angle_r': 0.001,
               'hip_flexion_l': 0.001, 'hip_adduction_l': 0.001, 'hip_rotation_l': 0.001,
               'knee_angle_l': 0.001, 'ankle_angle_l': 0.001,
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
    scaleTaskSet_simple.cloneAndAppend(jointTask)

# Create the IK task set for tracking

# Set the parameters for the IK task sets
ikTaskSetParams = {
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

    # Set participant mass
    mass_kg = participantInfo.loc[participantInfo['FileName'] == participant_id + 'static.c3d',]['Mass'].values[0]
    scaleTool_complex[participant_id].setSubjectMass(mass_kg)
    scaleTool_simple[participant_id].setSubjectMass(mass_kg)

    # Set generic model file
    scaleTool_complex[participant_id].getGenericModelMaker().setModelFileName(
        os.path.join('..', 'model', 'Uhlrich2022_LowerLimb_Fukuchi2017.osim'))
    scaleTool_simple[participant_id].getGenericModelMaker().setModelFileName(
        os.path.join('..', 'model', 'Denton2023_LowerLimb_Fukuchi2017.osim'))

    # Set measurement set in model scaler
    scaleTool_complex[participant_id].getModelScaler().setMeasurementSet(scaleMeasurementSet_complex)
    scaleTool_simple[participant_id].getModelScaler().setMeasurementSet(scaleMeasurementSet_simple)

    # Set scale tasks in tool
    for ii in range(scaleTaskSet_complex.getSize()):
        scaleTool_complex[participant_id].getMarkerPlacer().getIKTaskSet().cloneAndAppend(scaleTaskSet_complex.get(ii))
    for ii in range(scaleTaskSet_simple.getSize()):
        scaleTool_simple[participant_id].getMarkerPlacer().getIKTaskSet().cloneAndAppend(scaleTaskSet_simple.get(ii))

    # Set marker file
    scaleTool_complex[participant_id].getMarkerPlacer().setMarkerFileName(
        os.path.join(participant_id, f'{participant_id}static.trc'))
    scaleTool_complex[participant_id].getModelScaler().setMarkerFileName(
        os.path.join(participant_id, f'{participant_id}static.trc'))
    scaleTool_simple[participant_id].getMarkerPlacer().setMarkerFileName(
        os.path.join(participant_id, f'{participant_id}static.trc'))
    scaleTool_simple[participant_id].getModelScaler().setMarkerFileName(
        os.path.join(participant_id, f'{participant_id}static.trc'))

    # Set options
    scaleTool_complex[participant_id].getModelScaler().setPreserveMassDist(True)
    scaleTool_simple[participant_id].getModelScaler().setPreserveMassDist(True)
    scaleOrder = osim.ArrayStr()
    scaleOrder.set(0, 'measurements')
    scaleTool_complex[participant_id].getModelScaler().setScalingOrder(scaleOrder)
    scaleTool_simple[participant_id].getModelScaler().setScalingOrder(scaleOrder)

    # Set time ranges
    initial_time = osim.TimeSeriesTableVec3(
        os.path.join(participant_id, f'{participant_id}static.trc')).getIndependentColumn()[0]
    final_time = osim.TimeSeriesTableVec3(
        os.path.join(participant_id, f'{participant_id}static.trc')).getIndependentColumn()[-1]
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
    genModel_complex = osim.Model(os.path.join('..', 'model', 'Uhlrich2022_LowerLimb_Fukuchi2017.osim'))
    genModel_simple = osim.Model(os.path.join('..', 'model', 'Denton2023_LowerLimb_Fukuchi2017.osim'))
    genModelMass_complex = np.sum(
        [genModel_complex.getBodySet().get(bodyInd).getMass() for bodyInd in range(genModel_complex.getBodySet().getSize())])
    genModelMass_simple = np.sum(
        [genModel_simple.getBodySet().get(bodyInd).getMass() for bodyInd in range(genModel_simple.getBodySet().getSize())])
    genModelHeight_complex = 1.70
    genModelHeight_simple = 1.70

    # Get scaled model height (use mass from earlier)
    height_m = participantInfo.loc[participantInfo['FileName'] == participant_id + 'static.c3d',]['Height'].values[0] / 100

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
    [weld_vector.append(joint) for joint in ['mtp_r', 'mtp_l', 'subtalar_r', 'subtalar_l']]
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
def run_ik(participant_id, speed):

    """
    :param participant_id: participant ID to run IK for
    :param speed: speed label for trial to run IK for
    :return:

    """

    # =========================================================================
    # Organise files for inverse kinematics
    # =========================================================================

    # Create IK directory for files
    os.makedirs(os.path.join(participant_id, 'ik'), exist_ok=True)
    os.makedirs(os.path.join(participant_id, 'ik', speed), exist_ok=True)

    # =========================================================================
    # Run inverse kinematics on trial
    # =========================================================================

    # Set model
    ikTool[participant_id][speed].set_model_file(
        os.path.join(participant_id, 'scaling', f'{participant_id}_complex.osim'))

    # Set task set
    for taskInd in range(ikTaskSet.getSize()):
        ikTool[participant_id][speed].getIKTaskSet().adoptAndAppend(ikTaskSet.get(taskInd))

    # Set to report marker locations
    ikTool[participant_id][speed].set_report_marker_locations(True)

    # Set the marker file (note consistent use of trial20)
    ikTool[participant_id][speed].setMarkerDataFileName(
        os.path.join(participant_id, f'{participant_id}run{speed}.trc'))

    # Set times
    # Note that this is based on a period of getting 10 gait cycles (right foot strike to right foot strike)
    # The 10 gait cycles are taken from 10 gait cycles into the trial
    # Read in GRF data to identify stance phase timings
    trial_grf = osim.TimeSeriesTable(os.path.join(participant_id, f'{participant_id}run{speed}_grf.mot'))
    vgrf = trial_grf.getDependentColumn('ground_force_r_vy').to_numpy()
    grf_time = np.array(trial_grf.getIndependentColumn())
    # Identify right foot contacts based on rising edges above threshold of 20N
    force_above = vgrf > 20
    rising_edges = np.where((~force_above[:-1]) & (force_above[1:]))[0] + 1
    # Take the mid-point of the indices and take 5 strides either side
    # Get the associated times to run IK over
    middle_ind = np.where(rising_edges == rising_edges[len(rising_edges) // 2])[0][0]
    start_val = rising_edges[middle_ind - 5]
    end_val = rising_edges[middle_ind + 5]
    start_time = grf_time[start_val]
    end_time = grf_time[end_val]

    # Set times in IK tool
    ikTool[participant_id][speed].setStartTime(start_time)
    ikTool[participant_id][speed].setEndTime(end_time)

    # Set output filename (relative to setup file location)
    ikTool[participant_id][speed].setOutputMotionFileName(
        os.path.join(participant_id, 'ik', speed, f'{participant_id}_{speed}_ik_complex.mot'))

    # Save IK tool to file
    ikTool[participant_id][speed].printToXML(f'{participant_id}_{speed}_ikSetup.xml')

    # Bring the tool back in and run it (this seems to avoid Python kernel crashing)
    ikRun = osim.InverseKinematicsTool(f'{participant_id}_{speed}_ikSetup.xml')
    ikRun.run()

    # Rename supplementary marker outputs
    shutil.move('_ik_marker_errors.sto',
                os.path.join(participant_id, 'ik', speed, f'{participant_id}_{speed}_ikMarkerErrors_3d.sto'))
    shutil.move('_ik_model_marker_locations.sto',
                os.path.join(participant_id, 'ik', speed, f'{participant_id}_{speed}_ikModelMarkerLocations_3d.sto'))
    shutil.move(f'{participant_id}_{speed}_ikSetup.xml',
                os.path.join(participant_id, 'ik', speed, f'{participant_id}_{speed}_ikSetup.xml'))

    # Create a version of the IK that works with the simple model
    # This mainly requires inverting the knee angle
    # -------------------------------------------------------------------------

    # Read in the IK file
    ik_data = osim.TimeSeriesTable(
        os.path.join(participant_id, 'ik', speed, f'{participant_id}_{speed}_ik_complex.mot'))

    # Set columns to remove
    remove_cols = [
        #'subtalar_angle_r', 'subtalar_angle_l',
        'knee_angle_r_beta', 'knee_angle_l_beta'
                   ]

    # Remove the columns
    for col in remove_cols:
        ik_data.removeColumn(col)

    # Invert the knee angle data
    # Find the column index of the knee angles
    knee_ind = [
        ii for ii in range(len(ik_data.getColumnLabels())) if ik_data.getColumnLabels()[ii].startswith('knee_angle_')
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

    # Write the new IK data to file
    osim.STOFileAdapter().write(ik_data,
                                os.path.join(participant_id, 'ik', speed, f'{participant_id}_{speed}_ik_simple.mot'))


# Run the muscle path fitting tool
# -------------------------------------------------------------------------
def run_path_fitting(participant_id):

    # =========================================================================
    # Fit polynomials to muscle paths
    # =========================================================================

    # Create the PolynomialPathFitter for the complex and simple models
    # -------------------------------------------------------------------------
    fitter_complex = osim.PolynomialPathFitter()
    fitter_simple = osim.PolynomialPathFitter()

    # Set the parameters in the path fitters
    # -------------------------------------------------------------------------

    # Set the models
    model_complex = osim.Model(os.path.join(participant_id, 'scaling', f'{participant_id}_complex.osim'))
    model_simple = osim.Model(os.path.join(participant_id, 'scaling', f'{participant_id}_simple.osim'))
    model_complex.initSystem()
    model_simple.initSystem()
    fitter_complex.setModel(osim.ModelProcessor(model_complex))
    fitter_simple.setModel(osim.ModelProcessor(model_simple))

    # Set the coordinate values tables
    # Use the fastest speed processed to get the largest range of joint configurations
    values_complex = osim.TimeSeriesTable(
        os.path.join(participant_id, 'ik', speedList[-1], f'{participant_id}_{speedList[-1]}_ik_complex.mot'))
    values_simple = osim.TimeSeriesTable(
        os.path.join(participant_id, 'ik', speedList[-1], f'{participant_id}_{speedList[-1]}_ik_simple.mot'))
    time_complex = values_complex.getIndependentColumn()
    time_simple = values_complex.getIndependentColumn()
    # These IK data have more rows than needed, so some rows are removed to speed up the process
    for ii in range(len(time_complex)):
        if ii % 5 != 0:
            values_complex.removeRow(time_complex[ii])
    for ii in range(len(time_simple)):
        if ii % 5 != 0:
            values_simple.removeRow(time_simple[ii])

    # Set coordinate values in fitter
    fitter_complex.setCoordinateValues(osim.TableProcessor(values_complex))
    fitter_simple.setCoordinateValues(osim.TableProcessor(values_simple))

    # Set the directory to where fitting results will be saved
    fitter_complex.setOutputDirectory(os.path.join(participant_id, 'scaling', 'complex_fitter'))
    fitter_simple.setOutputDirectory(os.path.join(participant_id, 'scaling', 'simple_fitter'))

    # Set the maximum polynomial order
    fitter_complex.setMaximumPolynomialOrder(6)
    fitter_simple.setMaximumPolynomialOrder(6)

    # Set moment arm threshold
    # See: https://simtk.org/plugins/phpBB/viewtopicPhpbb.php?f=1815&t=17651&p=0&start=10&view=&sid=a562da3edb3423471892487d9e73dc15
    fitter_complex.setMomentArmThreshold(25e-4)
    fitter_simple.setMomentArmThreshold(25e-4)

    # Run the fitter tools
    # -------------------------------------------------------------------------
    fitter_simple.run()
    print(f'{"*"*10} POLYNOMIAL FITTING COMPLETED FOR SIMPLE MODEL {"*"*10}')
    fitter_complex.run()
    print(f'{"*" * 10} POLYNOMIAL FITTING COMPLETED FOR COMPLEX MODEL {"*" * 10}')

    # =========================================================================
    # Visualise fitting
    # =========================================================================

    # Set colouring for plot
    original_col = 'blue'
    fitted_col = 'orange'

    # Set row and column numbers
    nrows = 5
    ncols = 5

    # Loop through the two models
    for model_type in ['complex', 'simple']:

        # Loop through fitting variables
        for fit_var in ['path_lengths', 'moment_arms']:

            # Read in path lengths
            original = osim.TimeSeriesTable(os.path.join(os.path.join(participant_id, 'scaling', f'{model_type}_fitter',
                                                                      f'{participant_id}_{model_type}_{fit_var}.sto')))
            fitted = osim.TimeSeriesTable(os.path.join(os.path.join(participant_id, 'scaling', f'{model_type}_fitter',
                                                                    f'{participant_id}_{model_type}_{fit_var}_fitted.sto')))
            sampled = osim.TimeSeriesTable(os.path.join(os.path.join(participant_id, 'scaling', f'{model_type}_fitter',
                                                                     f'{participant_id}_{model_type}_{fit_var}_sampled.sto')))
            sampled_fitted = osim.TimeSeriesTable(os.path.join(os.path.join(participant_id, 'scaling', f'{model_type}_fitter',
                                                                            f'{participant_id}_{model_type}_{fit_var}_sampled_fitted.sto')))

            # Plot the results

            # Set the labels and colouring
            labels = original.getColumnLabels()
            if fit_var == 'path_lengths':
                ylabel = 'Length (cm)'
            elif fit_var == 'moment_arms':
                ylabel = 'Moment Arm (cm)'

            # Determine required number of figures
            nplots = nrows * ncols
            nfig = int(np.ceil(len(labels) / nplots))

            # Create figures
            for ifig in range(nfig):
                # Create the figure and axes
                fig, ax = plt.subplots(nrows, ncols,
                                       figsize=(12, 10))
                # Loop through rows and columns
                for irow in range(nrows):
                    for icol in range(ncols):
                        # Set plot and label
                        iplot = irow * ncols + icol
                        ilabel = iplot + ifig * nplots
                        if ilabel < len(labels):
                            # Set plotting axis
                            plot_ax = ax[irow, icol]
                            # Plot sampled values
                            plot_ax.scatter(sampled.getIndependentColumn(),
                                            sampled.getDependentColumn(labels[ilabel]).to_numpy(),
                                            alpha=0.15, color=original_col, s=0.4)
                            # Plot sample fitted values
                            plot_ax.scatter(sampled_fitted.getIndependentColumn(),
                                            sampled_fitted.getDependentColumn(labels[ilabel]).to_numpy(),
                                            alpha=0.15, color=fitted_col, s=0.4)
                            # Plot original values
                            plot_ax.plot(original.getIndependentColumn(),
                                         original.getDependentColumn(labels[ilabel]).to_numpy(),
                                         lw=1.5, color=original_col)
                            # Plot fitted values
                            plot_ax.plot(fitted.getIndependentColumn(),
                                         fitted.getDependentColumn(labels[ilabel]).to_numpy(),
                                         lw=1.5, color=fitted_col)
                            # Set axis limits and labels
                            plot_ax.set_xlim(original.getIndependentColumn()[0],
                                             original.getIndependentColumn()[-1])
                            plot_ax.set_title(labels[ilabel], fontsize=6, fontweight='bold')
                            plot_ax.set_xlabel('Time (s)', fontsize=6, fontweight='bold')
                            plot_ax.set_ylabel(ylabel, fontsize=6, fontweight='bold')
                        else:
                            # Switch the unused axis off
                            ax[irow, icol].axis('off')
                # Modify layout
                plt.tight_layout()
                # Save figure
                fig.savefig(os.path.join(participant_id, 'scaling', f'{model_type}_fitter',
                                         f'{participant_id}_{model_type}_{fit_var}_fig{ifig+1}.png'),
                            format='png', dpi=300)
                # Close figure
                plt.close('all')


# =========================================================================
# Run analyses
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
            # Loop through speeds
            for run_speed in speedList:
                run_ik(participant, run_speed)

    # Run polynomial muscle fitting tool
    # -------------------------------------------------------------------------
    if runMusclePathTool:
        # Loop through participants
        for participant in participant_list:
            run_path_fitting(participant)

    # Exit terminal to avoid any funny business
    # -------------------------------------------------------------------------
    os._exit(00)

# %% ---------- end of process_data.py ---------- %% #
# -*- coding: utf-8 -*-
"""

@author:

    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au
    
    This script runs the muscle driven simulations of running task for calcuating
    patellofemoral joint reaction forces.

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
from matplotlib import rcParams
import pickle
import argparse
import glob
import random

# =========================================================================
# Flags for running analyses
# =========================================================================

# Set participant ID to run
participant = 'RBDS002'
# parser = argparse.ArgumentParser()
# parser.add_argument('-p', '--participant', action = 'store', type = str, help = 'Enter the participant ID')
# args = parser.parse_args()
# participant = args.participant

# Settings for running specific sections of code
runScaling = True
runTorqueSim = True
runMuscleSim = True
runSimulations = True

# =========================================================================
# Set-up
# =========================================================================

# Set matplotlib parameters
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

# Add the utility geometry path for model visualisation
osim.ModelVisualizer.addDirToGeometrySearchPaths(os.path.join(os.getcwd(), '..', 'model', 'Geometry'))

# Set the participant list based on those in dataset folder
participantList = [ii for ii in os.listdir(os.path.join('..', 'data')) if
                   os.path.isdir(os.path.join('..', 'data',ii))]

# Check if input participant is in list
if participant not in participantList:
    raise ValueError(f'No data found for participant ID {participant}. Check input for error...')

# Read in participant info
participantInfo = pd.read_csv(os.path.join('..', 'data', 'participantInfo.csv'))

# Set the list of speed conditions to process
# Modify this if you want to isolate different running speeds
speedList = [
    'T25',   # 2.5 m/s
    # 'T35',   # 3.5 m/s
    'T45',   # 4.5 m/s
    ]

# Set weights for optimisations
globalMarkerTrackingWeight = 1e1
globalCoordinateTrackingWeight = 1e0
globalTorqueControlEffortGoal = 1e-2
globalMuscleControlEffortGoal = 1e-3

# Set mesh interval
# TODO: appropriate? setting interval time instead...?
meshIntervalTorque = 100
meshIntervalMuscle = 50
meshIntervalStep = 0.002 * 2  # times by 2 to get this as the interval given transcription scheme

# Set number of gait cycles to process from each speed condition
nGaitCycles = 3  # TODO: 1 is probably more feasbile time-wise...? But potentially lacks accuracy...?

# Set kinematics filter frequency
kinematicFiltFreq = 12

# =========================================================================
# Scale model
# =========================================================================

# Check for running scaling
if runScaling:

    # Create measurement set for scaling
    # -------------------------------------------------------------------------

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
    scaleMeasurementSet = osim.MeasurementSet()

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
        # Append the measurement to the set
        scaleMeasurementSet.cloneAndAppend(measurement)

    # Create scale task set
    # -------------------------------------------------------------------------

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
                   'knee_angle_r': 0.001, 'ankle_angle_r': 0.001, 'subtalar_angle_r': 0.001,
                   'hip_flexion_l': 0.001, 'hip_adduction_l': 0.001, 'hip_rotation_l': 0.001,
                   'knee_angle_l': 0.001, 'ankle_angle_l': 0.001, 'subtalar_angle_l': 0.001,
                   }

    # Create the task set
    scaleTaskSet = osim.IKTaskSet()

    # Append the tasks from the marker parameters
    for taskName in markerParams.keys():
        # Create the task and add details
        task = osim.IKMarkerTask()
        task.setName(taskName)
        task.setWeight(markerParams[taskName]['weight'])
        if markerParams[taskName]['weight'] == 0.0:
            task.setApply(False)
        # Append to task set
        scaleTaskSet.cloneAndAppend(task)

    # Append the tasks from the joint parameters
    for jointName in jointParams:
        # Create the task and add details
        jointTask = osim.IKCoordinateTask()
        jointTask.setName(jointName)
        jointTask.setWeight(jointParams[jointName])
        # Append to task set
        scaleTaskSet.cloneAndAppend(jointTask)

    # Create folder location and set-up files for for scaling
    # -------------------------------------------------------------------------

    # Make participant directory and scaling folder
    os.makedirs(os.path.join('..', 'simulations', participant), exist_ok=True)
    os.makedirs(os.path.join('..', 'simulations', participant, 'scaling'), exist_ok=True)

    # Copy the static TRC file to scaling directory
    shutil.copyfile(os.path.join('..', 'data', participant, participant+'static.trc'),
                    os.path.join('..', 'simulations', participant, 'scaling', participant+'static.trc'))

    # Set-up and run scaling tool
    # -------------------------------------------------------------------------

    # Create scaling tool
    scaleTool = osim.ScaleTool()

    # Get and set participant mass in tool by locating static trial row
    massKg = participantInfo.loc[participantInfo['FileName'] == participant+'static.c3d',]['Mass'].values[0]
    scaleTool.setSubjectMass(massKg)

    # Set generic model file
    scaleTool.getGenericModelMaker().setModelFileName(os.path.join('..', 'model', 'Uhlrich2022_LowerLimb_Fukuchi2017.osim'))

    # Set measurement set in model scaler
    scaleTool.getModelScaler().setMeasurementSet(scaleMeasurementSet)

    # Set scale tasks in tool
    for ii in range(scaleTaskSet.getSize()):
        scaleTool.getMarkerPlacer().getIKTaskSet().cloneAndAppend(scaleTaskSet.get(ii))

    # Set marker file
    scaleTool.getMarkerPlacer().setMarkerFileName(os.path.join('..', 'simulations', participant, 'scaling', participant+'static.trc'))
    scaleTool.getModelScaler().setMarkerFileName(os.path.join('..', 'simulations', participant, 'scaling', participant+'static.trc'))

    # Set options
    scaleTool.getModelScaler().setPreserveMassDist(True)
    scaleOrder = osim.ArrayStr()
    scaleOrder.set(0, 'measurements')
    scaleTool.getModelScaler().setScalingOrder(scaleOrder)

    # Set time ranges
    initialTime = osim.TimeSeriesTableVec3(os.path.join('..', 'simulations', participant, 'scaling',
                                                        participant + 'static.trc')).getIndependentColumn()[0]
    finalTime = osim.TimeSeriesTableVec3(os.path.join('..', 'simulations', participant, 'scaling',
                                                      participant + 'static.trc')).getIndependentColumn()[-1]
    timeRange = osim.ArrayDouble()
    timeRange.set(0, initialTime)
    timeRange.set(1, finalTime)
    scaleTool.getMarkerPlacer().setTimeRange(timeRange)
    scaleTool.getModelScaler().setTimeRange(timeRange)

    # Set output files
    scaleTool.getModelScaler().setOutputModelFileName(
        os.path.join('..', 'simulations', participant, 'scaling', f'{participant}_scaledModel.osim'))
    scaleTool.getModelScaler().setOutputScaleFileName(
        os.path.join('..', 'simulations', participant, 'scaling', f'{participant}_scaleSet.xml'))

    # Set marker adjustment parameters
    scaleTool.getMarkerPlacer().setOutputMotionFileName(
        os.path.join('..', 'simulations', participant, 'scaling', f'{participant}_staticMotion.mot'))
    scaleTool.getMarkerPlacer().setOutputModelFileName(
        os.path.join('..', 'simulations', participant, 'scaling', f'{participant}_scaledModelAdjusted.osim'))

    # Save and run scale tool
    scaleTool.printToXML(os.path.join('..', 'simulations', participant, 'scaling', f'{participant}_scaleSetup.xml'))
    scaleTool.run()

    # Adjust model
    # -------------------------------------------------------------------------

    # Load the model
    scaledModel = osim.Model(os.path.join('..', 'simulations', participant, 'scaling', f'{participant}_scaledModelAdjusted.osim'))

    # Set model name
    scaledModel.setName(participant)

    # Scale model muscle forces according to height-mass relationship

    # Get generic model mass and set generic height
    genModel = osim.Model(os.path.join('..', 'model', 'Uhlrich2022_LowerLimb_Fukuchi2017.osim'))
    genModelMass = np.sum([genModel.getBodySet().get(bodyInd).getMass() for bodyInd in range(genModel.getBodySet().getSize())])
    genModelHeight = 1.70

    # Get scaled model height (use mass from earlier)
    heightM = participantInfo.loc[participantInfo['FileName'] == participant + 'static.c3d',]['Height'].values[0] / 100

    # Get muscle volume totals based on mass and heights with linear equation
    genericMuscVol = 47.05 * genModelMass * genModelHeight + 1289.6
    scaledMuscVol = 47.05 * massKg * heightM + 1289.6

    # Loop through all muscles and scale according to volume and muscle parameters
    # Use this opportunity to also update contraction velocity
    for muscInd in range(scaledModel.getMuscles().getSize()):
        # Get current muscle name
        muscName = scaledModel.getMuscles().get(muscInd).getName()
        # Get optimal fibre length for muscle from each model
        genericL0 = genModel.getMuscles().get(muscName).getOptimalFiberLength()
        scaledL0 = scaledModel.getMuscles().get(muscName).getOptimalFiberLength()
        # Set force scale factor
        forceScaleFactor = (scaledMuscVol / genericMuscVol) / (scaledL0 / genericL0)
        # Scale current muscle strength
        scaledModel.getMuscles().get(muscInd).setMaxIsometricForce(
            forceScaleFactor * scaledModel.getMuscles().get(muscInd).getMaxIsometricForce())
        # Update max contraction velocity
        scaledModel.getMuscles().get(muscInd).setMaxContractionVelocity(30.0)

    # Finalise model connections
    scaledModel.finalizeConnections()

    # Print to file (overwrites original adjusted model)
    scaledModel.printToXML(os.path.join('..', 'simulations', participant, 'scaling', f'{participant}_scaledModelAdjusted.osim'))

# =========================================================================
# Run torque driven marker tracking simulations
# =========================================================================

"""

This section runs the torque driven marker tracking simulations, with the goal
here to generate a dynamically consistent motion for the set of gait cycles which
can be tracked with muscle-driven simulations. These simulations also generate the
knee angle and moment data required in the simplified PFJRF model.

"""

# Check for running simulations
if runTorqueSim:

    # Identify trials for running simulations
    # -------------------------------------------------------------------------

    # Look up the running trial names that match the speed condition list
    trialList = []
    for speed in speedList:
        if os.path.exists(os.path.join('..', 'data', participant, f'{participant}run{speed}.c3d')):
            trialList.append(os.path.join('..', 'data', participant, f'{participant}run{speed}'))

    # Set general parameters for simulations
    # -------------------------------------------------------------------------

    # Set marker tracking weights
    markerWeightParams = {
        # Pelvis
        'R.ASIS': {'weight': 2.5}, 'L.ASIS': {'weight': 2.5}, 'R.PSIS': {'weight': 2.5}, 'L.PSIS': {'weight': 2.5},
        'R.Iliac.Crest': {'weight': 1.0}, 'L.Iliac.Crest': {'weight': 1.0},
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

    # Set actuator forces to drive simulations
    actForces = {'pelvis_tx': {'actuatorType': 'residual', 'optForce': 5},
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
                 'subtalar_angle_r': {'actuatorType': 'torque', 'optForce': 100.0},
                 'hip_flexion_l': {'actuatorType': 'torque', 'optForce': 300.0},
                 'hip_adduction_l': {'actuatorType': 'torque', 'optForce': 200.0},
                 'hip_rotation_l': {'actuatorType': 'torque', 'optForce': 100.0},
                 'knee_angle_l': {'actuatorType': 'torque', 'optForce': 300.0},
                 'ankle_angle_l': {'actuatorType': 'torque', 'optForce': 200.0},
                 'subtalar_angle_l': {'actuatorType': 'torque', 'optForce': 100.0},
                 }

    # Simulate gait cycles from selected trials
    # -------------------------------------------------------------------------

    # Loop through trials
    for trial in trialList:

        # Get generic trial name
        trialName = os.path.split(trial)[-1].replace(participant,'')

        # Set-up folders and gait timings for trial
        # -------------------------------------------------------------------------

        # Create the folder for storing trial data
        os.makedirs(os.path.join('..', 'simulations', participant, trialName), exist_ok=True)
        os.makedirs(os.path.join('..', 'simulations', participant, trialName, 'torque-driven'), exist_ok=True)

        # Create the sub-folders for the gait cycles to be simulated
        for ii in range(nGaitCycles):
            os.makedirs(os.path.join('..', 'simulations', participant, trialName, 'torque-driven', f'cycle_{ii+1}'), exist_ok=True)

        # Read in GRF data to identify stance phase timings
        trialGRF = osim.TimeSeriesTable(os.path.join('..', 'data', participant, f'{participant}{trialName}_grf.mot'))

        # Get vertical ground reaction force for right limb
        vGRF = trialGRF.getDependentColumn('ground_force_r_vy').to_numpy()

        # Identify contact indices based on force threshold
        vertForceThreshold = 50
        thresholdCrossings = np.diff(vGRF > vertForceThreshold, prepend=False)
        thresholdInd = np.argwhere(thresholdCrossings)[:, 0]

        # Sort into pairs
        # If the first index is zero it means that the trial started on the plate and this needs to be accounted for
        contactPairs = []
        if thresholdInd[0] == 0:
            for ii in range(0, len(thresholdInd[2::]), 2):
                contactPairs.append(thresholdInd[2::][ii:ii + 2])
        else:
            for ii in range(0, len(thresholdInd), 2):
                contactPairs.append(thresholdInd[ii:ii + 2])

        # Trim last contact pair if only single contact (i.e. contact stayed on plate at end of trial
        if len(contactPairs[-1]) == 1:
            contactPairs = contactPairs[:-1]

        # Re-pair first listed indices to create full gait cycle pairings
        gaitCyclePairs = []
        for ii in range(len(contactPairs) - 1):
            gaitCyclePairs.append((contactPairs[ii][0], contactPairs[ii + 1][0]))

        # Randomly sample the desired number of cycles from the contact pairs
        random.seed(int(''.join(ii for ii in participant if ii.isdigit())) + int(''.join(ii for ii in trialName if ii.isdigit())))
        selectCycles = random.sample(gaitCyclePairs, nGaitCycles)

        # Identify timings of selected cycles
        gaitTimings = [(trialGRF.getIndependentColumn()[selectCycles[ii][0]],
                        trialGRF.getIndependentColumn()[selectCycles[ii][1]]) for ii in range(nGaitCycles)]

        # Run simulations of each gait cycle
        # -------------------------------------------------------------------------

        # Loop through gait cycles
        for ii in range(nGaitCycles):

            # Set-up files for simulation
            # -------------------------------------------------------------------------

            # Navigate to simulation folder for ease of use
            homeDir = os.getcwd()
            os.chdir(os.path.join('..', 'simulations', participant, trialName, 'torque-driven', f'cycle_{ii+1}'))

            # Copy external loads file to simulation directory
            shutil.copyfile(os.path.join('..', '..', '..', '..', '..', 'data', participant, f'{participant}{trialName}_grf.mot'),
                            f'{participant}{trialName}_grf.mot')
            shutil.copyfile(os.path.join('..', '..', '..', '..', '..', 'data', participant, f'{participant}{trialName}_grf.xml'),
                            f'{participant}{trialName}_grf.xml')

            # Copy marker file to simulation directory
            shutil.copyfile(os.path.join('..', '..', '..', '..', '..', 'data', participant, f'{participant}{trialName}.trc'),
                            f'{participant}{trialName}.trc')

            # Run inverse kinematics to get tracking guess
            # -------------------------------------------------------------------------

            # Create IK tool
            ikTool = osim.InverseKinematicsTool()

            # Set to report marker locations in tool
            ikTool.set_report_marker_locations(True)

            # Set tasks in IK tool
            for markerName in markerWeightParams.keys():
                if markerWeightParams[markerName]['weight'] != 0:
                    task = osim.IKMarkerTask()
                    task.setName(markerName)
                    task.setWeight(markerWeightParams[markerName]['weight'])
                    ikTool.getIKTaskSet().cloneAndAppend(task)

            # Set the model to use in IK
            ikTool.set_model_file(os.path.join('..', '..', '..', 'scaling', f'{participant}_scaledModelAdjusted.osim'))

            # Set marker file
            ikTool.setMarkerDataFileName(f'{participant}{trialName}.trc')

            # Set times to start and end of marker file
            ikTool.setStartTime(gaitTimings[ii][0])
            ikTool.setEndTime(gaitTimings[ii][1])

            # Set output file
            ikTool.setOutputMotionFileName(f'{participant}{trialName}_ik.mot')

            # Save IK file to base directory to bring back in and run
            ikTool.printToXML(f'{participant}{trialName}_ikSetup.xml')

            # Bring the tool back in and run standard IK (this seems to avoid Python kernel crashing)
            ikRun = osim.InverseKinematicsTool(f'{participant}{trialName}_ikSetup.xml')
            ikRun.run()

            # Rename marker error and location files
            shutil.move('_ik_marker_errors.sto', f'{participant}{trialName}_ikMarkerErrors.sto')
            shutil.move('_ik_model_marker_locations.sto', f'{participant}{trialName}_ikModelMarkerLocations.sto')

            # Clean up kinematic data for tracking guess
            # -------------------------------------------------------------------------

            # Load in the kinematic data
            kinematicsStorage = osim.Storage(f'{participant}{trialName}_ik.mot')

            # Create a copy of the kinematics data to alter the column labels in
            statesStorage = osim.Storage(f'{participant}{trialName}_ik.mot')

            # Filter both storage objects
            # Note that this resamples time stamps so eliminates the need to do so
            kinematicsStorage.lowpassFIR(4, kinematicFiltFreq)
            statesStorage.lowpassFIR(4, kinematicFiltFreq)

            # Get the column headers for the storage file
            angleNames = kinematicsStorage.getColumnLabels()

            # Get the corresponding full paths from the model to rename the
            # angles in the kinematics file
            kinematicModel = osim.Model(os.path.join('..', '..', '..', 'scaling', f'{participant}_scaledModelAdjusted.osim'))
            for angNo in range(angleNames.getSize()):
                currAngle = angleNames.get(angNo)
                if currAngle != 'time':
                    # Try getting the full path to coordinate
                    # This may fail due to their being marker data included in these files
                    try:
                        # Loo for full coordinate path
                        fullPath = kinematicModel.updCoordinateSet().get(currAngle).getAbsolutePathString() + '/value'
                        # Set angle name appropriately using full path
                        angleNames.set(angNo, fullPath)
                    except:
                        # Print out that current column isn't a coordinate
                        print(f'{currAngle} not a coordinate...skipping name conversion...')
                        # Set to the same as originaly
                        angleNames.set(angNo, currAngle)

            # Set the states storage object to have the updated column labels
            statesStorage.setColumnLabels(angleNames)

            # Convert from IK default of degrees to radians
            kinematicModel.initSystem()
            kinematicModel.getSimbodyEngine().convertDegreesToRadians(statesStorage)

            # Write the states storage object to file
            statesStorage.printToXML(f'{participant}{trialName}_coordinates.sto')

            # Set up model for tracking simulation
            # -------------------------------------------------------------------------

            # Construct a model processor to use with the tool
            modelProc = osim.ModelProcessor(os.path.join('..', '..', '..', 'scaling', f'{participant}_scaledModelAdjusted.osim'))

            # Append external loads
            modelProc.append(osim.ModOpAddExternalLoads(f'{participant}{trialName}_grf.xml'))

            # Weld desired locked joints
            # Create vector string object
            weldVectorStr = osim.StdVectorString()
            [weldVectorStr.append(joint) for joint in ['mtp_r', 'mtp_l']]
            # Append to model processor
            modelProc.append(osim.ModOpReplaceJointsWithWelds(weldVectorStr))

            # Remove muscles from model
            modelProc.append(osim.ModOpRemoveMuscles())

            # Process model for further edits
            trackingModel = modelProc.process()

            # Add coordinate actuators to model
            for coordinate in actForces:
                # Create actuator
                actu = osim.CoordinateActuator()
                # Set name
                actu.setName(f'{coordinate}_{actForces[coordinate]["actuatorType"]}')
                # Set coordinate
                actu.setCoordinate(trackingModel.updCoordinateSet().get(coordinate))
                # Set optimal force
                actu.setOptimalForce(actForces[coordinate]['optForce'])
                # Set min and max control
                actu.setMinControl(np.inf * -1)
                actu.setMaxControl(np.inf * 1)
                # Append to model force set
                trackingModel.updForceSet().cloneAndAppend(actu)

            # Finalise model connections
            trackingModel.finalizeConnections()

            # Print model to file in tracking directory
            trackingModel.printToXML(f'{participant}{trialName}_{ii+1}_torque-driven_model.osim')

            # Set up tracking simulation
            # -------------------------------------------------------------------------

            # Create tracking tool
            track = osim.MocoTrack()
            track.setName(f'{participant}{trialName}_{ii+1}_torque-driven')

            # Set model
            trackModelProc = osim.ModelProcessor(f'{participant}{trialName}_{ii+1}_torque-driven_model.osim')
            track.setModel(trackModelProc)

            # Set the marker reference file and settings
            track.setMarkersReferenceFromTRC(f'{participant}{trialName}.trc')
            track.set_markers_global_tracking_weight(globalMarkerTrackingWeight)

            # # Set the coordinates reference file and settings
            # # Note that a zero or quite low weight can be set here if desired
            # tableProcessor = osim.TableProcessor(f'{participant}{trialName}_coordinates.sto')
            # track.setStatesReference(tableProcessor)
            # track.set_states_global_tracking_weight(1e-2)  # TODO: zero/low weight for tracking coordinates?
            # track.set_track_reference_position_derivatives(True)
            # track.set_apply_tracked_states_to_guess(True)

            # Set to ignore unused columns
            track.set_allow_unused_references(True)

            # Set individual marker weights
            markerWeights = osim.MocoWeightSet()
            for marker in markerWeightParams.keys():
                if markerWeightParams[marker]['weight'] != 0:
                    markerWeights.cloneAndAppend(osim.MocoWeight(marker, markerWeightParams[marker]['weight']))
            track.set_markers_weight_set(markerWeights)

            # Set the timings
            # Slightly different due to potential re-sampling of time-stamps
            # track.set_initial_time(osim.TimeSeriesTable(f'{participant}{trialName}_coordinates.sto').getIndependentColumn()[0])
            # track.set_final_time(osim.TimeSeriesTable(f'{participant}{trialName}_coordinates.sto').getIndependentColumn()[-1])
            track.set_initial_time(gaitTimings[ii][0])
            track.set_final_time(gaitTimings[ii][1])
            # track.set_mesh_interval(meshIntervalStep)

            # Initialise to a Moco study and problem to finalise
            # -------------------------------------------------------------------------

            # Get study and problem
            study = track.initialize()
            problem = study.updProblem()

            # Update control effort goal
            # -------------------------------------------------------------------------

            # Get a reference to the MocoControlCost goal and set parameters
            effort = osim.MocoControlGoal.safeDownCast(problem.updGoal('control_effort'))
            effort.setWeight(globalTorqueControlEffortGoal)
            effort.setExponent(2)

            # Update individual weights in control effort goal
            # Put higher weight on residual use
            # TODO: higher residual weight?
            effort.setWeightForControlPattern('/forceset/.*_residual', 5.0)
            # Put heavy weight on the reserve actuators
            effort.setWeightForControlPattern('/forceset/.*_torque', 1.0)

            # Option to update marker tracking goal
            # -------------------------------------------------------------------------

            # TODO: enforce in a constraint manner so markers remain within threshold?

            # # Get a reference to the marker tracking goal
            # tracking = osim.MocoMarkerTrackingGoal.safeDownCast(problem.updGoal('marker_tracking'))

            # Update states tracking goal
            # With a zero weighted goal I doubt this does anything
            # -------------------------------------------------------------------------

            # # Get a reference to the states tracking goal
            # tracking = osim.MocoStateTrackingGoal.safeDownCast(problem.updGoal('state_tracking'))
            # tracking.setScaleWeightsWithRange(True)

            # Define and configure the solver
            # -------------------------------------------------------------------------
            solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())

            # Modify initial guess to use IK kinematics

            # Get the guess
            guess = solver.getGuess()

            # Get and resample the guess to match IK
            ikData = osim.TimeSeriesTable(f'{participant}{trialName}_coordinates.sto')
            guess.resampleWithNumTimes(ikData.getNumRows())

            # Insert the desired values from IK
            for colName in guess.getStateNames():
                if colName.endswith('/value') and colName in ikData.getColumnLabels():
                    guess.setState(colName, ikData.getDependentColumn(colName).to_numpy())

            # Write to file for reference
            guess.write(f'{participant}{trialName}_initialGuess.sto')

            # Set solver options
            solver.set_optim_max_iterations(2000)
            solver.set_num_mesh_intervals(meshIntervalTorque)
            solver.set_optim_constraint_tolerance(1e-3)
            solver.set_optim_convergence_tolerance(1e-3)
            # solver.set_minimize_implicit_multibody_accelerations(True)  # smoothness criterion?
            # solver.set_implicit_multibody_accelerations_weight(1e-2)
            solver.setGuessFile(f'{participant}{trialName}_initialGuess.sto')
            solver.resetProblem(problem)

            # Solve the problem
            # -------------------------------------------------------------------------
            trackingSolution = study.solve()

            # # Option to visualise solution
            # study.visualize(trackingSolution)

            # Save files and finalize
            # -------------------------------------------------------------------------

            # Write solution to file
            if trackingSolution.isSealed():
                trackingSolution.unseal()
            trackingSolution.write(f'{participant}{trialName}_{ii+1}_torque-driven_marker-solution.sto')

            # Remove initial tracked states and markers file
            os.remove(f'{participant}{trialName}_{ii+1}_torque-driven_tracked_markers.sto')

            # # Extract joint reaction forces from solution
            # jrfTable = osim.analyzeMocoTrajectorySpatialVec(trackingModel, trackingSolution,
            #                                                 ['.*patellofemoral_r.*reaction_on_child']).flatten()
            # osim.STOFileAdapter().write(jrfTable, f'{participant}{trialName}_{ii+1}_trackingSolution_JRF.sto')
            #
            # # Extract muscle forces from solution
            # outputPaths = osim.StdVectorString()
            # outputPaths.append('.*tendon_force')
            # muscleForceTable = osim.analyzeMocoTrajectory(trackingModel, trackingSolution, outputPaths)
            # osim.STOFileAdapter().write(muscleForceTable, f'{participant}{trialName}_{ii+1}_trackingSolution_muscleForce.sto')

            # Return to home directory
            os.chdir(homeDir)

            # Print out to console as a bookmark in any log file
            # TODO: other logging notes...
            print(f'***** ----- FINISHED TORQUE TRACKING SIM FOR CONDITION {participant}{trialName} GAIT CYCLE {ii+1} ----- *****')

# =========================================================================
# Run muscle driven inverse simulation
# =========================================================================

"""

TODO: add notes...

"""

# Check for running simulations
if runMuscleSim:

    # Identify trials for running simulations
    # -------------------------------------------------------------------------

    # Look up the running trial names from torque-driven simulations that match the speed condition list
    simList = []
    for speed in speedList:
        for ii in range(nGaitCycles):
            if os.path.exists(os.path.join('..', 'simulations', participant, f'run{speed}', 'torque-driven', f'cycle_{ii+1}',
                                           f'{participant}run{speed}_{ii+1}_torque-driven_marker-solution.sto')):
                simList.append(os.path.join('..', 'simulations', participant, f'run{speed}', 'torque-driven', f'cycle_{ii+1}',
                                            f'{participant}run{speed}_{ii+1}_torque-driven_marker-solution.sto'))

    # Set general parameters for simulations
    # -------------------------------------------------------------------------

    # Set actuator forces to drive simulations
    actForces = {'pelvis_tx': {'actuatorType': 'residual', 'optForce': 5},
                 'pelvis_ty': {'actuatorType': 'residual', 'optForce': 5},
                 'pelvis_tz': {'actuatorType': 'residual', 'optForce': 5},
                 'pelvis_tilt': {'actuatorType': 'residual', 'optForce': 2.5},
                 'pelvis_list': {'actuatorType': 'residual', 'optForce': 2.5},
                 'pelvis_rotation': {'actuatorType': 'residual', 'optForce': 2.5},
                 'hip_flexion_r': {'actuatorType': 'reserve', 'optForce': 2.5},
                 'hip_adduction_r': {'actuatorType': 'reserve', 'optForce': 1.0},
                 'hip_rotation_r': {'actuatorType': 'reserve', 'optForce': 1.0},
                 'knee_angle_r': {'actuatorType': 'reserve', 'optForce': 2.5},
                 'ankle_angle_r': {'actuatorType': 'reserve', 'optForce': 2.5},
                 'subtalar_angle_r': {'actuatorType': 'reserve', 'optForce': 1.0},
                 'hip_flexion_l': {'actuatorType': 'reserve', 'optForce': 2.5},
                 'hip_adduction_l': {'actuatorType': 'reserve', 'optForce': 1.0},
                 'hip_rotation_l': {'actuatorType': 'reserve', 'optForce': 1.0},
                 'knee_angle_l': {'actuatorType': 'reserve', 'optForce': 2.5},
                 'ankle_angle_l': {'actuatorType': 'reserve', 'optForce': 2.5},
                 'subtalar_angle_l': {'actuatorType': 'reserve', 'optForce': 1.0},
                 }

    # Simulate gait cycles from identified trials
    # -------------------------------------------------------------------------

    # Loop through trials
    for sim in simList:

        # Get generic trial name and solution file
        trialName = os.path.split(sim)[0].split(os.sep)[3]
        trackingSolutionFile = os.path.split(sim)[-1]

        # Set-up folders and gait timings for trial
        # -------------------------------------------------------------------------

        # Create the folder for storing trial data
        os.makedirs(os.path.join('..', 'simulations', participant, trialName, 'muscle-driven'), exist_ok=True)

        # Create the sub-folder for the current gait cycle to be simulated based on identified tracking solutions
        cycleName = os.path.split(sim)[0].split(os.sep)[-1]
        os.makedirs(os.path.join('..', 'simulations', participant, trialName, 'muscle-driven', cycleName), exist_ok=True)

        # Run simulation for the current gait cycle
        # -------------------------------------------------------------------------

        # Set-up files for simulation
        # -------------------------------------------------------------------------

        # Navigate to simulation folder for ease of use
        homeDir = os.getcwd()
        os.chdir(os.path.join('..', 'simulations', participant, trialName, 'muscle-driven', cycleName))

        # Copy external loads file to simulation directory
        shutil.copyfile(os.path.join('..', '..', 'torque-driven', cycleName, f'{participant}{trialName}_grf.mot'),
                        f'{participant}{trialName}_grf.mot')
        shutil.copyfile(os.path.join('..', '..', 'torque-driven', cycleName, f'{participant}{trialName}_grf.xml'),
                        f'{participant}{trialName}_grf.xml')

        # Copy solution file to simulation directory
        shutil.copyfile(os.path.join('..', '..', 'torque-driven', cycleName, trackingSolutionFile),
                        trackingSolutionFile)

        # Set up model for inverse simulation
        # -------------------------------------------------------------------------

        # Construct a model processor to use with the tool
        modelProc = osim.ModelProcessor(os.path.join('..', '..', '..', 'scaling', f'{participant}_scaledModelAdjusted.osim'))

        # Append external loads
        modelProc.append(osim.ModOpAddExternalLoads(f'{participant}{trialName}_grf.xml'))

        # Weld desired locked joints
        # Create vector string object
        weldVectorStr = osim.StdVectorString()
        [weldVectorStr.append(joint) for joint in ['mtp_r', 'mtp_l']]
        # Append to model processor
        modelProc.append(osim.ModOpReplaceJointsWithWelds(weldVectorStr))

        # Convert muscles to DeGrooteFregley model
        modelProc.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())

        # Increase muscle isometric force by a scaling factor to deal with potentially higher muscle forces
        modelProc.append(osim.ModOpScaleMaxIsometricForce(1.5))

        # Set to ignore tendon compliance
        modelProc.append(osim.ModOpIgnoreTendonCompliance())

        # Ignore passive fibre forces
        modelProc.append(osim.ModOpIgnorePassiveFiberForcesDGF())

        # Scale active force curve width
        modelProc.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))

        # Process model for further edits
        inverseModel = modelProc.process()

        # Add coordinate actuators to model
        for coordinate in actForces:
            # Create actuator
            actu = osim.CoordinateActuator()
            # Set name
            actu.setName(f'{coordinate}_{actForces[coordinate]["actuatorType"]}')
            # Set coordinate
            actu.setCoordinate(inverseModel.updCoordinateSet().get(coordinate))
            # Set optimal force
            actu.setOptimalForce(actForces[coordinate]['optForce'])
            # Set min and max control
            actu.setMinControl(np.inf * -1)
            actu.setMaxControl(np.inf * 1)
            # Append to model force set
            inverseModel.updForceSet().cloneAndAppend(actu)

        # Finalise model connections
        inverseModel.finalizeConnections()

        # Print model to file in tracking directory
        inverseModel.printToXML(f'{participant}{trialName}_{cycleName[-1]}_inverseModel.osim')

        # Set-up inverse simulation
        # -------------------------------------------------------------------------

        # Create inverse tool
        inverse = osim.MocoInverse()
        inverse.setName(f'{participant}{trialName}_{cycleName[-1]}')

        # Set model
        inverse.setModel(osim.ModelProcessor(inverseModel))

        # Set coordinates in inverse simulation from tracking solution
        values = osim.MocoTrajectory(trackingSolutionFile).exportToValuesTable()
        values.addTableMetaDataString('inDegrees', 'no')
        osim.STOFileAdapter().write(values, f'{participant}{trialName}_{cycleName[-1]}_coordinates.sto')
        inverse.setKinematics(osim.TableProcessor(f'{participant}{trialName}_{cycleName[-1]}_coordinates.sto'))

        # Set times
        inverse.set_initial_time(values.getIndependentColumn()[0])
        inverse.set_final_time(values.getIndependentColumn()[-1])

        # Set kinematics to have extra columns (even though this shouldn't be an issue)
        inverse.set_kinematics_allow_extra_columns(True)

        # Convert to Moco study for added flexibility
        # -------------------------------------------------------------------------

        # Get the study and problem
        study = inverse.initialize()
        problem = study.updProblem()

        # Update control effort goal
        # -------------------------------------------------------------------------

        # Get a reference to the MocoControlCost goal and set parameters
        effort = osim.MocoControlGoal.safeDownCast(problem.updGoal('excitation_effort'))
        effort.setWeight(globalMuscleControlEffortGoal)
        effort.setExponent(2)

        # Update individual weights in control effort goal
        # Put higher weight on residual use
        effort.setWeightForControlPattern('/forceset/.*_residual', 2.5)
        # Put heavy weight on the reserve actuators
        effort.setWeightForControlPattern('/forceset/.*_reserve', 5.0)

        # Define and configure the solver
        # -------------------------------------------------------------------------
        solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())

        # Set solver options
        solver.set_optim_max_iterations(1000)
        solver.set_num_mesh_intervals(meshIntervalMuscle) # TODO: appropriate interval?
        solver.set_optim_constraint_tolerance(1e-2)   # TODO: appropriate?
        solver.set_optim_convergence_tolerance(1e-3)  # TODO: appropriate?
        solver.resetProblem(problem)

        # Solve the problem
        # -------------------------------------------------------------------------
        inverseSolution = study.solve()

        # # Option to visualise solution
        # study.visualize(inverseSolution)

        # Save files and finalize
        # -------------------------------------------------------------------------

        # Write solution to file
        if inverseSolution.isSealed():
            inverseSolution.unseal()
        inverseSolution.write(f'{participant}{trialName}_{cycleName[-1]}_inverseSolution.sto')

        # Get the inverse and tracking trajectory to combine the data for analyses
        inverseTrajectory = osim.MocoTrajectory(f'{participant}{trialName}_{cycleName[-1]}_inverseSolution.sto')
        trackingTrajectory = osim.MocoTrajectory(trackingSolutionFile)

        # Resample tracking to match inverse solution
        trackingTrajectory.resampleWithNumTimes(inverseTrajectory.getNumTimes())

        # Insert the values and speeds from tracking simulation into inverse solution

        # Get the necessary values, speeds and states
        trackingValues = trackingTrajectory.exportToValuesTable()
        trackingSpeeds = trackingTrajectory.exportToSpeedsTable()
        inverseStates = inverseSolution.exportToStatesTable()

        # Loop through values and speeds to append to states
        for colName in trackingValues.getColumnLabels():
            inverseStates.appendColumn(colName, trackingValues.getDependentColumn(colName))
        for colName in trackingSpeeds.getColumnLabels():
            inverseStates.appendColumn(colName, trackingSpeeds.getDependentColumn(colName))

        # Write updated data to file
        osim.STOFileAdapter().write(inverseStates, f'{participant}{trialName}_{cycleName[-1]}_inverseSolutionStates.sto')

        # Extract joint reaction forces from solution
        jrfTable = osim.analyzeSpatialVec(inverseModel, inverseStates, inverseSolution.exportToControlsTable(),
                                          ['.*patellofemoral_r.*reaction_on_child']).flatten()
        osim.STOFileAdapter().write(jrfTable, f'{participant}{trialName}_{cycleName[-1]}_inverseSolution_JRF.sto')

        # Extract muscle forces from solution
        outputPaths = osim.StdVectorString()
        outputPaths.append('.*tendon_force')
        muscleForceTable = osim.analyze(inverseModel, inverseStates, inverseSolution.exportToControlsTable(), outputPaths)
        osim.STOFileAdapter().write(muscleForceTable, f'{participant}{trialName}_{cycleName[-1]}_inverseSolution_muscleForce.sto')

        # Return to home directory
        os.chdir(homeDir)

        # Print out to console as a bookmark in any log file
        # TODO: other logging notes...
        print(f'***** ----- FINISHED INVERSE SIM FOR CONDITION {participant}{trialName} GAIT CYCLE {cycleName[-1]} ----- *****')



# ---- OLD CODE PROBABLY NOT NEEDED BELOW HERE... ---- #

# # =========================================================================
# # Run torque driven coordinate tracking simulations --- MAYBE NOT USE?
# # =========================================================================
#
# """
#
# This section runs the torque driven coordinate tracking simulations, with the goal
# here to generate a dynamically consistent motion for the set of gait cycles which
# can be tracked with muscle-driven simulations. Inverse kinematics is first run to
# set the coordinate tracking targets.
#
# """
#
# # Check for running simulations
# if runTorqueSim:
#
#     # Identify trials for running simulations
#     # -------------------------------------------------------------------------
#
#     # Look up the running trial names that match the speed condition list
#     trialList = []
#     for speed in speedList:
#         if os.path.exists(os.path.join('..', 'data', participant, f'{participant}run{speed}.c3d')):
#             trialList.append(os.path.join('..', 'data', participant, f'{participant}run{speed}'))
#
#     # Set general parameters for simulations
#     # -------------------------------------------------------------------------
#
#     # Set marker tracking weights
#     markerWeightParams = {
#         # Pelvis
#         'R.ASIS': {'weight': 2.5}, 'L.ASIS': {'weight': 2.5}, 'R.PSIS': {'weight': 2.5}, 'L.PSIS': {'weight': 2.5},
#         'R.Iliac.Crest': {'weight': 0.0}, 'L.Iliac.Crest': {'weight': 0.0},
#         # Right thigh
#         'R.GTR': {'weight': 0.0},
#         'R.Thigh.Top.Lateral': {'weight': 5.0}, 'R.Thigh.Bottom.Lateral': {'weight': 5.0},
#         'R.Thigh.Top.Medial': {'weight': 5.0}, 'R.Thigh.Bottom.Medial': {'weight': 5.0},
#         'R.Knee': {'weight': 0.0}, 'R.Knee.Medial': {'weight': 0.0},
#         # Right shank
#         'R.HF': {'weight': 0.0}, 'R.TT': {'weight': 0.0},
#         'R.Shank.Top.Lateral': {'weight': 5.0}, 'R.Shank.Bottom.Lateral': {'weight': 5.0},
#         'R.Shank.Top.Medial': {'weight': 5.0}, 'R.Shank.Bottom.Medial': {'weight': 5.0},
#         'R.Ankle': {'weight': 0.0}, 'R.Ankle.Medial': {'weight': 0.0},
#         # Right foot
#         'R.Heel.Top': {'weight': 5.0}, 'R.Heel.Bottom': {'weight': 5.0}, 'R.Heel.Lateral': {'weight': 5.0},
#         'R.MT1': {'weight': 2.5}, 'R.MT2': {'weight': 2.5}, 'R.MT5': {'weight': 2.5},
#         # Left thigh
#         'L.GTR': {'weight': 0.0},
#         'L.Thigh.Top.Lateral': {'weight': 5.0}, 'L.Thigh.Bottom.Lateral': {'weight': 5.0},
#         'L.Thigh.Top.Medial': {'weight': 5.0}, 'L.Thigh.Bottom.Medial': {'weight': 5.0},
#         'L.Knee': {'weight': 0.0}, 'L.Knee.Medial': {'weight': 0.0},
#         # Left shank
#         'L.HF': {'weight': 0.0}, 'L.TT': {'weight': 0.0},
#         'L.Shank.Top.Lateral': {'weight': 5.0}, 'L.Shank.Bottom.Lateral': {'weight': 5.0},
#         'L.Shank.Top.Medial': {'weight': 5.0}, 'L.Shank.Bottom.Medial': {'weight': 5.0},
#         'L.Ankle': {'weight': 0.0}, 'L.Ankle.Medial': {'weight': 0.0},
#         # Left foot
#         'L.Heel.Top': {'weight': 5.0}, 'L.Heel.Bottom': {'weight': 5.0}, 'L.Heel.Lateral': {'weight': 5.0},
#         'L.MT1': {'weight': 2.5}, 'L.MT2': {'weight': 2.5}, 'L.MT5': {'weight': 2.5},
#     }
#
#     # Set actuator forces to drive simulations
#     actForces = {'pelvis_tx': {'actuatorType': 'residual', 'optForce': 5},
#                  'pelvis_ty': {'actuatorType': 'residual', 'optForce': 5},
#                  'pelvis_tz': {'actuatorType': 'residual', 'optForce': 5},
#                  'pelvis_tilt': {'actuatorType': 'residual', 'optForce': 2.5},
#                  'pelvis_list': {'actuatorType': 'residual', 'optForce': 2.5},
#                  'pelvis_rotation': {'actuatorType': 'residual', 'optForce': 2.5},
#                  'hip_flexion_r': {'actuatorType': 'torque', 'optForce': 300.0},
#                  'hip_adduction_r': {'actuatorType': 'torque', 'optForce': 200.0},
#                  'hip_rotation_r': {'actuatorType': 'torque', 'optForce': 100.0},
#                  'knee_angle_r': {'actuatorType': 'torque', 'optForce': 300.0},
#                  'ankle_angle_r': {'actuatorType': 'torque', 'optForce': 200.0},
#                  'subtalar_angle_r': {'actuatorType': 'torque', 'optForce': 100.0},
#                  'hip_flexion_l': {'actuatorType': 'torque', 'optForce': 300.0},
#                  'hip_adduction_l': {'actuatorType': 'torque', 'optForce': 200.0},
#                  'hip_rotation_l': {'actuatorType': 'torque', 'optForce': 100.0},
#                  'knee_angle_l': {'actuatorType': 'torque', 'optForce': 300.0},
#                  'ankle_angle_l': {'actuatorType': 'torque', 'optForce': 200.0},
#                  'subtalar_angle_l': {'actuatorType': 'torque', 'optForce': 100.0},
#                  }
#
#     # Simulate gait cycles from selected trials
#     # -------------------------------------------------------------------------
#
#     # Loop through trials
#     for trial in trialList:
#
#         # Get generic trial name
#         trialName = os.path.split(trial)[-1].replace(participant,'')
#
#         # Set-up folders and gait timings for trial
#         # -------------------------------------------------------------------------
#
#         # Create the folder for storing trial data
#         os.makedirs(os.path.join('..', 'simulations', participant, trialName), exist_ok=True)
#         os.makedirs(os.path.join('..', 'simulations', participant, trialName, 'torque-driven'), exist_ok=True)
#
#         # Create the sub-folders for the gait cycles to be simulated
#         for ii in range(nGaitCycles):
#             os.makedirs(os.path.join('..', 'simulations', participant, trialName, 'torque-driven', f'cycle_{ii+1}'), exist_ok=True)
#
#         # Read in GRF data to identify stance phase timings
#         trialGRF = osim.TimeSeriesTable(os.path.join('..', 'data', participant, f'{participant}{trialName}_grf.mot'))
#
#         # Get vertical ground reaction force for right limb
#         vGRF = trialGRF.getDependentColumn('ground_force_r_vy').to_numpy()
#
#         # Identify contact indices based on force threshold
#         # Higher force threshold seems to help with noisy-ish COP at ground contact
#         vertForceThreshold = 100
#         thresholdCrossings = np.diff(vGRF > vertForceThreshold, prepend=False)
#         thresholdInd = np.argwhere(thresholdCrossings)[:, 0]
#
#         # Sort into pairs
#         # If the first index is zero it means that the trial started on the plate and this needs to be accounted for
#         contactPairs = []
#         if thresholdInd[0] == 0:
#             for ii in range(0, len(thresholdInd[2::]), 2):
#                 contactPairs.append(thresholdInd[2::][ii:ii + 2])
#         else:
#             for ii in range(0, len(thresholdInd), 2):
#                 contactPairs.append(thresholdInd[ii:ii + 2])
#
#         # Trim last contact pair if only single contact (i.e. contact stayed on plate at end of trial
#         if len(contactPairs[-1]) == 1:
#             contactPairs = contactPairs[:-1]
#
#         # Re-pair first listed indices to create full gait cycle pairings
#         gaitCyclePairs = []
#         for ii in range(len(contactPairs) - 1):
#             gaitCyclePairs.append((contactPairs[ii][0], contactPairs[ii + 1][0]))
#
#         # Randomly sample the desired number of cycles from the contact pairs
#         random.seed(int(''.join(ii for ii in participant if ii.isdigit())) + int(''.join(ii for ii in trialName if ii.isdigit())))
#         selectCycles = random.sample(gaitCyclePairs, nGaitCycles)
#
#         # Identify timings of selected cycles
#         gaitTimings = [(trialGRF.getIndependentColumn()[selectCycles[ii][0]],
#                         trialGRF.getIndependentColumn()[selectCycles[ii][1]]) for ii in range(nGaitCycles)]
#
#         # Run simulations of each gait cycle
#         # -------------------------------------------------------------------------
#
#         # Loop through gait cycles
#         for ii in range(nGaitCycles):
#
#             # Set-up files for simulation
#             # -------------------------------------------------------------------------
#
#             # Navigate to simulation folder for ease of use
#             homeDir = os.getcwd()
#             os.chdir(os.path.join('..', 'simulations', participant, trialName, 'torque-driven', f'cycle_{ii+1}'))
#
#             # Copy external loads file to simulation directory
#             shutil.copyfile(os.path.join('..', '..', '..', '..', '..', 'data', participant, f'{participant}{trialName}_grf.mot'),
#                             f'{participant}{trialName}_grf.mot')
#             shutil.copyfile(os.path.join('..', '..', '..', '..', '..', 'data', participant, f'{participant}{trialName}_grf.xml'),
#                             f'{participant}{trialName}_grf.xml')
#
#             # Copy marker file to simulation directory
#             shutil.copyfile(os.path.join('..', '..', '..', '..', '..', 'data', participant, f'{participant}{trialName}.trc'),
#                             f'{participant}{trialName}.trc')
#
#             # Run inverse kinematics to get tracking coordinates
#             # -------------------------------------------------------------------------
#
#             # Create IK tool
#             ikTool = osim.InverseKinematicsTool()
#
#             # Set to report marker locations in tool
#             ikTool.set_report_marker_locations(True)
#
#             # Set tasks in IK tool
#             for markerName in markerWeightParams.keys():
#                 if markerWeightParams[markerName]['weight'] != 0:
#                     task = osim.IKMarkerTask()
#                     task.setName(markerName)
#                     task.setWeight(markerWeightParams[markerName]['weight'])
#                     ikTool.getIKTaskSet().cloneAndAppend(task)
#
#             # Set the model to use in IK
#             ikTool.set_model_file(os.path.join('..', '..', '..', 'scaling', f'{participant}_scaledModelAdjusted.osim'))
#
#             # Set marker file
#             ikTool.setMarkerDataFileName(f'{participant}{trialName}.trc')
#
#             # Set times to start and end of marker file
#             ikTool.setStartTime(gaitTimings[ii][0])
#             ikTool.setEndTime(gaitTimings[ii][1])
#
#             # Set output file
#             ikTool.setOutputMotionFileName(f'{participant}{trialName}_ik.mot')
#
#             # Save IK file to base directory to bring back in and run
#             ikTool.printToXML(f'{participant}{trialName}_ikSetup.xml')
#
#             # Bring the tool back in and run standard IK (this seems to avoid Python kernel crashing)
#             ikRun = osim.InverseKinematicsTool(f'{participant}{trialName}_ikSetup.xml')
#             ikRun.run()
#
#             # Rename marker error and location files
#             shutil.move('_ik_marker_errors.sto', f'{participant}{trialName}_ikMarkerErrors.sto')
#             shutil.move('_ik_model_marker_locations.sto', f'{participant}{trialName}_ikModelMarkerLocations.sto')
#
#             # Clean up kinematic data for tracking
#             # -------------------------------------------------------------------------
#
#             # Load in the kinematic data
#             kinematicsStorage = osim.Storage(f'{participant}{trialName}_ik.mot')
#
#             # Create a copy of the kinematics data to alter the column labels in
#             statesStorage = osim.Storage(f'{participant}{trialName}_ik.mot')
#
#             # Filter both storage objects
#             # Note that this resamples time stamps so eliminates the need to do so
#             kinematicsStorage.lowpassFIR(4, kinematicFiltFreq)
#             statesStorage.lowpassFIR(4, kinematicFiltFreq)
#
#             # Get the column headers for the storage file
#             angleNames = kinematicsStorage.getColumnLabels()
#
#             # Get the corresponding full paths from the model to rename the
#             # angles in the kinematics file
#             kinematicModel = osim.Model(os.path.join('..', '..', '..', 'scaling', f'{participant}_scaledModelAdjusted.osim'))
#             for angNo in range(angleNames.getSize()):
#                 currAngle = angleNames.get(angNo)
#                 if currAngle != 'time':
#                     # Try getting the full path to coordinate
#                     # This may fail due to their being marker data included in these files
#                     try:
#                         # Loo for full coordinate path
#                         fullPath = kinematicModel.updCoordinateSet().get(currAngle).getAbsolutePathString() + '/value'
#                         # Set angle name appropriately using full path
#                         angleNames.set(angNo, fullPath)
#                     except:
#                         # Print out that current column isn't a coordinate
#                         print(f'{currAngle} not a coordinate...skipping name conversion...')
#                         # Set to the same as originaly
#                         angleNames.set(angNo, currAngle)
#
#             # Set the states storage object to have the updated column labels
#             statesStorage.setColumnLabels(angleNames)
#
#             # Convert from IK default of degrees to radians
#             kinematicModel.initSystem()
#             kinematicModel.getSimbodyEngine().convertDegreesToRadians(statesStorage)
#
#             # Write the states storage object to file
#             statesStorage.printToXML(f'{participant}{trialName}_coordinates.sto')
#
#             # Set up model for tracking simulation
#             # -------------------------------------------------------------------------
#
#             # Construct a model processor to use with the tool
#             modelProc = osim.ModelProcessor(os.path.join('..', '..', '..', 'scaling', f'{participant}_scaledModelAdjusted.osim'))
#
#             # Append external loads
#             modelProc.append(osim.ModOpAddExternalLoads(f'{participant}{trialName}_grf.xml'))
#
#             # Weld desired locked joints
#             # Create vector string object
#             weldVectorStr = osim.StdVectorString()
#             [weldVectorStr.append(joint) for joint in ['mtp_r', 'mtp_l']]
#             # Append to model processor
#             modelProc.append(osim.ModOpReplaceJointsWithWelds(weldVectorStr))
#
#             # Remove muscles from model
#             modelProc.append(osim.ModOpRemoveMuscles())
#
#             # Process model for further edits
#             trackingModel = modelProc.process()
#
#             # Add coordinate actuators to model
#             for coordinate in actForces:
#                 # Create actuator
#                 actu = osim.CoordinateActuator()
#                 # Set name
#                 actu.setName(f'{coordinate}_{actForces[coordinate]["actuatorType"]}')
#                 # Set coordinate
#                 actu.setCoordinate(trackingModel.updCoordinateSet().get(coordinate))
#                 # Set optimal force
#                 actu.setOptimalForce(actForces[coordinate]['optForce'])
#                 # Set min and max control
#                 actu.setMinControl(np.inf * -1)
#                 actu.setMaxControl(np.inf * 1)
#                 # Append to model force set
#                 trackingModel.updForceSet().cloneAndAppend(actu)
#
#             # Finalise model connections
#             trackingModel.finalizeConnections()
#
#             # Print model to file in tracking directory
#             trackingModel.printToXML(f'{participant}{trialName}_{ii+1}_torque-driven_model.osim')
#
#             # Set up tracking simulation
#             # -------------------------------------------------------------------------
#
#             # Create tracking tool
#             track = osim.MocoTrack()
#             track.setName(f'{participant}{trialName}_{ii+1}_torque-driven')
#
#             # Set model
#             trackModelProc = osim.ModelProcessor(f'{participant}{trialName}_{ii+1}_torque-driven_model.osim')
#             track.setModel(trackModelProc)
#
#             # Set the coordinates reference file
#             tableProcessor = osim.TableProcessor(f'{participant}{trialName}_coordinates.sto')
#             track.setStatesReference(tableProcessor)
#
#             # Set to ignore unused columns
#             track.set_allow_unused_references(True)
#
#             # Set global markers tracking weight
#             track.set_states_global_tracking_weight(globalCoordinateTrackingWeight)
#
#             # Track positive derivaties (i.e. speeds)
#             track.set_track_reference_position_derivatives(True)
#
#             # Set tracked states to guess
#             track.set_apply_tracked_states_to_guess(True)
#
#             # Set the timings
#             # Slightly different due to potential re-sampling of time-stamps
#             track.set_initial_time(osim.TimeSeriesTable(f'{participant}{trialName}_coordinates.sto').getIndependentColumn()[0])
#             track.set_final_time(osim.TimeSeriesTable(f'{participant}{trialName}_coordinates.sto').getIndependentColumn()[-1])
#             track.set_mesh_interval(meshIntervalStep)
#
#             # Initialise to a Moco study and problem to finalise
#             # -------------------------------------------------------------------------
#
#             # Get study and problem
#             study = track.initialize()
#             problem = study.updProblem()
#
#             # Update control effort goal
#             # -------------------------------------------------------------------------
#
#             # Get a reference to the MocoControlCost goal and set parameters
#             effort = osim.MocoControlGoal.safeDownCast(problem.updGoal('control_effort'))
#             effort.setWeight(globalTorqueControlEffortGoal)
#             effort.setExponent(2)
#
#             # Update individual weights in control effort goal
#             # Put higher weight on residual use
#             # TODO: higher residual weight?
#             effort.setWeightForControlPattern('/forceset/.*_residual', 5.0)
#             # Put heavy weight on the reserve actuators
#             effort.setWeightForControlPattern('/forceset/.*_torque', 1.0)
#
#             # Update states tracking goal
#             # -------------------------------------------------------------------------
#
#             # Get a reference to the states tracking goal
#             tracking = osim.MocoStateTrackingGoal.safeDownCast(problem.updGoal('state_tracking'))
#             tracking.setScaleWeightsWithRange(True)
#
#             # Define and configure the solver
#             # -------------------------------------------------------------------------
#             solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())
#
#             # Set solver options
#             solver.set_optim_max_iterations(1000)
#             # solver.set_num_mesh_intervals(meshIntervalTorque)
#             solver.set_optim_constraint_tolerance(1e-3)
#             solver.set_optim_convergence_tolerance(1e-3)
#             solver.resetProblem(problem)
#
#             # Solve the problem
#             # -------------------------------------------------------------------------
#             trackingSolution = study.solve()
#
#             # # Option to visualise solution
#             # study.visualize(trackingSolution)
#
#             # Save files and finalize
#             # -------------------------------------------------------------------------
#
#             # Write solution to file
#             if trackingSolution.isSealed():
#                 trackingSolution.unseal()
#             trackingSolution.write(f'{participant}{trialName}_{ii+1}_torque-driven_coordinate-solution.sto')
#
#             # Remove initial tracked states and markers file
#             os.remove(f'{participant}{trialName}_{ii+1}_torque-driven_tracked_states.sto')
#
#             # # Extract joint reaction forces from solution
#             # jrfTable = osim.analyzeMocoTrajectorySpatialVec(trackingModel, trackingSolution,
#             #                                                 ['.*patellofemoral_r.*reaction_on_child']).flatten()
#             # osim.STOFileAdapter().write(jrfTable, f'{participant}{trialName}_{ii+1}_trackingSolution_JRF.sto')
#             #
#             # # Extract muscle forces from solution
#             # outputPaths = osim.StdVectorString()
#             # outputPaths.append('.*tendon_force')
#             # muscleForceTable = osim.analyzeMocoTrajectory(trackingModel, trackingSolution, outputPaths)
#             # osim.STOFileAdapter().write(muscleForceTable, f'{participant}{trialName}_{ii+1}_trackingSolution_muscleForce.sto')
#
#             # Return to home directory
#             os.chdir(homeDir)
#
#             # Print out to console as a bookmark in any log file
#             # TODO: other logging notes...
#             print(f'***** ----- FINISHED TORQUE TRACKING SIM FOR PARTICIPANT {participant} {trialName} GAIT CYCLE {ii+1} ----- *****')
#
# # =========================================================================
# # Run tracking simulations - MODIFYING WITH EARLIER CODE...
# # =========================================================================
#
# # Check for running simulations
# if runSimulations:
#
#     # Identify trials for running simulations
#     # -------------------------------------------------------------------------
#
#     # Look up the running trial names that match the speed condition list
#     trialList = []
#     for speed in speedList:
#         if os.path.exists(os.path.join('..', 'data', participant, f'{participant}run{speed}.c3d')):
#             trialList.append(os.path.join('..', 'data', participant, f'{participant}run{speed}'))
#
#     # Set general parameters for simulations
#     # -------------------------------------------------------------------------
#
#     # Set marker tracking weights
#     markerWeightParams = {
#         # Pelvis
#         'R.ASIS': {'weight': 10.0}, 'L.ASIS': {'weight': 10.0}, 'R.PSIS': {'weight': 10.0}, 'L.PSIS': {'weight': 10.0},
#         'R.Iliac.Crest': {'weight': 2.5}, 'L.Iliac.Crest': {'weight': 2.5},
#         # Right thigh
#         'R.GTR': {'weight': 0.0},
#         'R.Thigh.Top.Lateral': {'weight': 5.0}, 'R.Thigh.Bottom.Lateral': {'weight': 5.0},
#         'R.Thigh.Top.Medial': {'weight': 5.0}, 'R.Thigh.Bottom.Medial': {'weight': 5.0},
#         'R.Knee': {'weight': 0.0}, 'R.Knee.Medial': {'weight': 0.0},
#         # Right shank
#         'R.HF': {'weight': 0.0}, 'R.TT': {'weight': 0.0},
#         'R.Shank.Top.Lateral': {'weight': 5.0}, 'R.Shank.Bottom.Lateral': {'weight': 5.0},
#         'R.Shank.Top.Medial': {'weight': 5.0}, 'R.Shank.Bottom.Medial': {'weight': 5.0},
#         'R.Ankle': {'weight': 0.0}, 'R.Ankle.Medial': {'weight': 0.0},
#         # Right foot
#         'R.Heel.Top': {'weight': 10.0}, 'R.Heel.Bottom': {'weight': 10.0}, 'R.Heel.Lateral': {'weight': 10.0},
#         'R.MT1': {'weight': 5.0}, 'R.MT2': {'weight': 0.0}, 'R.MT5': {'weight': 5.0},
#         # Left thigh
#         'L.GTR': {'weight': 0.0},
#         'L.Thigh.Top.Lateral': {'weight': 5.0}, 'L.Thigh.Bottom.Lateral': {'weight': 5.0},
#         'L.Thigh.Top.Medial': {'weight': 5.0}, 'L.Thigh.Bottom.Medial': {'weight': 5.0},
#         'L.Knee': {'weight': 0.0}, 'L.Knee.Medial': {'weight': 0.0},
#         # Left shank
#         'L.HF': {'weight': 0.0}, 'L.TT': {'weight': 0.0},
#         'L.Shank.Top.Lateral': {'weight': 5.0}, 'L.Shank.Bottom.Lateral': {'weight': 5.0},
#         'L.Shank.Top.Medial': {'weight': 5.0}, 'L.Shank.Bottom.Medial': {'weight': 5.0},
#         'L.Ankle': {'weight': 0.0}, 'L.Ankle.Medial': {'weight': 0.0},
#         # Left foot
#         'L.Heel.Top': {'weight': 10.0}, 'L.Heel.Bottom': {'weight': 10.0}, 'L.Heel.Lateral': {'weight': 10.0},
#         'L.MT1': {'weight': 5.0}, 'L.MT2': {'weight': 0.0}, 'L.MT5': {'weight': 0.0},
#     }
#
#     # Set actuator forces to include alongside muscles
#     actForces = {'pelvis_tx': {'actuatorType': 'residual', 'optForce': 5},
#                  'pelvis_ty': {'actuatorType': 'residual', 'optForce': 5},
#                  'pelvis_tz': {'actuatorType': 'residual', 'optForce': 5},
#                  'pelvis_tilt': {'actuatorType': 'residual', 'optForce': 2.5},
#                  'pelvis_list': {'actuatorType': 'residual', 'optForce': 2.5},
#                  'pelvis_rotation': {'actuatorType': 'residual', 'optForce': 2.5},
#                  'hip_flexion_r': {'actuatorType': 'reserve', 'optForce': 2.5},
#                  'hip_adduction_r': {'actuatorType': 'reserve', 'optForce': 1.0},
#                  'hip_rotation_r': {'actuatorType': 'reserve', 'optForce': 1.0},
#                  'knee_angle_r': {'actuatorType': 'reserve', 'optForce': 2.5},
#                  'ankle_angle_r': {'actuatorType': 'reserve', 'optForce': 2.5},
#                  'subtalar_angle_r': {'actuatorType': 'reserve', 'optForce': 1.0},
#                  'hip_flexion_l': {'actuatorType': 'reserve', 'optForce': 2.5},
#                  'hip_adduction_l': {'actuatorType': 'reserve', 'optForce': 1.0},
#                  'hip_rotation_l': {'actuatorType': 'reserve', 'optForce': 1.0},
#                  'knee_angle_l': {'actuatorType': 'reserve', 'optForce': 2.5},
#                  'ankle_angle_l': {'actuatorType': 'reserve', 'optForce': 2.5},
#                  'subtalar_angle_l': {'actuatorType': 'reserve', 'optForce': 1.0},
#                  }
#
#     # Simulate gait cycles from selected trials
#     # -------------------------------------------------------------------------
#
#     # Loop through trials
#     for trial in trialList:
#
#         # Get generic trial name
#         trialName = os.path.split(trial)[-1].replace(participant,'')
#
#         # Set-up folders and gait timings for trial
#         # -------------------------------------------------------------------------
#
#         # Create the folder for storing trial data
#         os.makedirs(os.path.join('..', 'simulations', participant, trialName), exist_ok=True)
#
#         # Create the sub-folders for the gait cycles to be simulated
#         for ii in range(nGaitCycles):
#             os.makedirs(os.path.join('..', 'simulations', participant, trialName, f'cycle_{ii+1}'), exist_ok=True)
#
#         # Read in GRF data to identify stance phase timings
#         trialGRF = osim.TimeSeriesTable(os.path.join('..', 'data', participant, f'{participant}{trialName}_grf.mot'))
#
#         # Get vertical ground reaction force for right limb
#         vGRF = trialGRF.getDependentColumn('ground_force_r_vy').to_numpy()
#
#         # Identify contact indices based on force threshold
#         vertForceThreshold = 50
#         thresholdCrossings = np.diff(vGRF > vertForceThreshold, prepend=False)
#         thresholdInd = np.argwhere(thresholdCrossings)[:, 0]
#
#         # Sort into pairs
#         # If the first index is zero it means that the trial started on the plate and this needs to be accounted for
#         contactPairs = []
#         if thresholdInd[0] == 0:
#             for ii in range(0, len(thresholdInd[2::]), 2):
#                 contactPairs.append(thresholdInd[2::][ii:ii + 2])
#         else:
#             for ii in range(0, len(thresholdInd), 2):
#                 contactPairs.append(thresholdInd[ii:ii + 2])
#
#         # Trim last contact pair if only single contact (i.e. contact stayed on plate at end of trial
#         if len(contactPairs[-1]) == 1:
#             contactPairs = contactPairs[:-1]
#
#         # Re-pair first listed indices to create full gait cycle pairings
#         gaitCyclePairs = []
#         for ii in range(len(contactPairs) - 1):
#             gaitCyclePairs.append((contactPairs[ii][0], contactPairs[ii + 1][0]))
#
#         # Randomly sample the desired number of cycles from the contact pairs
#         random.seed(int(''.join(ii for ii in participant if ii.isdigit())) + int(''.join(ii for ii in trialName if ii.isdigit())))
#         selectCycles = random.sample(gaitCyclePairs, nGaitCycles)
#
#         # Identify timings of selected cycles
#         gaitTimings = [(trialGRF.getIndependentColumn()[selectCycles[ii][0]],
#                         trialGRF.getIndependentColumn()[selectCycles[ii][1]]) for ii in range(nGaitCycles)]
#
#         # Run simulations of each gait cycle
#         # -------------------------------------------------------------------------
#
#         # Loop through gait cycles
#         for ii in range(nGaitCycles):
#
#             # Set-up files for simulation
#             # -------------------------------------------------------------------------
#
#             # Navigate to simulation folder for ease of use
#             homeDir = os.getcwd()
#             os.chdir(os.path.join('..', 'simulations', participant, trialName, f'cycle_{ii+1}'))
#
#             # Copy external loads file to simulation directory
#             shutil.copyfile(os.path.join('..', '..', '..', '..', 'data', participant, f'{participant}{trialName}_grf.mot'),
#                             f'{participant}{trialName}_grf.mot')
#             shutil.copyfile(os.path.join('..', '..', '..', '..', 'data', participant, f'{participant}{trialName}_grf.xml'),
#                             f'{participant}{trialName}_grf.xml')
#
#             # Copy marker file to simulation directory
#             shutil.copyfile(os.path.join('..', '..', '..', '..', 'data', participant, f'{participant}{trialName}.trc'),
#                             f'{participant}{trialName}.trc')
#
#             # Set up model for tracking simulation
#             # -------------------------------------------------------------------------
#
#             # Construct a model processor to use with the tool
#             modelProc = osim.ModelProcessor(os.path.join('..', '..', 'scaling', f'{participant}_scaledModelAdjusted.osim'))
#
#             # Append external loads
#             modelProc.append(osim.ModOpAddExternalLoads(f'{participant}{trialName}_grf.xml'))
#
#             # Weld desired locked joints
#             # Create vector string object
#             weldVectorStr = osim.StdVectorString()
#             [weldVectorStr.append(joint) for joint in ['mtp_r', 'mtp_l']]
#             # Append to model processor
#             modelProc.append(osim.ModOpReplaceJointsWithWelds(weldVectorStr))
#
#             # Convert muscles to DeGrooteFregley model
#             modelProc.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
#
#             # Increase muscle isometric force by a scaling factor to deal with potentially higher muscle forces
#             modelProc.append(osim.ModOpScaleMaxIsometricForce(1.5))
#
#             # Set to ignore tendon compliance
#             modelProc.append(osim.ModOpIgnoreTendonCompliance())
#
#             # Ignore passive fibre forces
#             modelProc.append(osim.ModOpIgnorePassiveFiberForcesDGF())
#
#             # Scale active force curve width
#             modelProc.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))
#
#             # Process model for further edits
#             trackingModel = modelProc.process()
#
#             # Add coordinate actuators to model
#             for coordinate in actForces:
#                 # Create actuator
#                 actu = osim.CoordinateActuator()
#                 # Set name
#                 actu.setName(f'{coordinate}_{actForces[coordinate]["actuatorType"]}')
#                 # Set coordinate
#                 actu.setCoordinate(trackingModel.updCoordinateSet().get(coordinate))
#                 # Set optimal force
#                 actu.setOptimalForce(actForces[coordinate]['optForce'])
#                 # Set min and max control
#                 actu.setMinControl(np.inf * -1)
#                 actu.setMaxControl(np.inf * 1)
#                 # Append to model force set
#                 trackingModel.updForceSet().cloneAndAppend(actu)
#
#             # Finalise model connections
#             trackingModel.finalizeConnections()
#
#             # Print model to file in tracking directory
#             trackingModel.printToXML(f'{participant}{trialName}_{ii+1}_trackingModel.osim')
#
#             # Set up tracking simulation
#             # -------------------------------------------------------------------------
#
#             # Create tracking tool
#             track = osim.MocoTrack()
#             track.setName(f'{participant}{trialName}_{ii+1}')
#
#             # Set model
#             trackModelProc = osim.ModelProcessor(f'{participant}{trialName}_{ii+1}_trackingModel.osim')
#             track.setModel(trackModelProc)
#
#             # Set the marker reference file
#             track.setMarkersReferenceFromTRC(f'{participant}{trialName}.trc')
#
#             # Set to ignore unused markers
#             track.set_allow_unused_references(True)
#
#             # Set global markers tracking weight
#             track.set_markers_global_tracking_weight(globalMarkerTrackingWeight)
#
#             # Set individual marker weights
#             markerWeights = osim.MocoWeightSet()
#             for marker in markerWeightParams.keys():
#                 if markerWeightParams[marker]['weight'] != 0:
#                     markerWeights.cloneAndAppend(osim.MocoWeight(marker, markerWeightParams[marker]['weight']))
#             track.set_markers_weight_set(markerWeights)
#
#             # Set the timings
#             track.set_initial_time(gaitTimings[ii][0])
#             track.set_final_time(gaitTimings[ii][1])
#             # track.set_mesh_interval(meshIntervalStep)
#
#             # Initialise to a Moco study and problem to finalise
#             # -------------------------------------------------------------------------
#
#             # Get study and problem
#             study = track.initialize()
#             problem = study.updProblem()
#
#             # Update control effort goal
#             # -------------------------------------------------------------------------
#
#             # Get a reference to the MocoControlCost goal and set parameters
#             effort = osim.MocoControlGoal.safeDownCast(problem.updGoal('control_effort'))
#             effort.setWeight(globalControlEffortGoal)
#             effort.setExponent(2)
#
#             # Update individual weights in control effort goal
#             # Put higher weight on residual use
#             effort.setWeightForControlPattern('/forceset/.*_residual', 2.5)
#             # Put heavy weight on the reserve actuators
#             effort.setWeightForControlPattern('/forceset/.*_reserve', 5.0)
#
#             # Option to update marker tracking goal
#             # -------------------------------------------------------------------------
#
#             # TODO: enforce in a constraint manner so markers remain within threshold?
#
#             # # Get a reference to the marker tracking goal
#             # tracking = osim.MocoMarkerTrackingGoal.safeDownCast(problem.updGoal('marker_tracking'))
#
#             # Define and configure the solver
#             # -------------------------------------------------------------------------
#             solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())
#
#             # Set solver options
#             # TODO: set mesh interval using variable
#             solver.set_optim_max_iterations(1000)
#             solver.set_num_mesh_intervals(meshInterval)
#             solver.set_optim_constraint_tolerance(1e-2)
#             solver.set_optim_convergence_tolerance(1e-3)
#             solver.resetProblem(problem)
#
#             # Solve the problem
#             # -------------------------------------------------------------------------
#             trackingSolution = study.solve()
#
#             # # Option to visualise solution
#             # study.visualize(trackingSolution)
#
#             # Save files and finalize
#             # -------------------------------------------------------------------------
#
#             # Write solution to file
#             if trackingSolution.isSealed():
#                 trackingSolution.unseal()
#             trackingSolution.write(f'{participant}{trialName}_{ii+1}_trackingSolution.sto')
#
#             # Remove initial tracked states and markers file
#             os.remove(f'{participant}{trialName}_{ii+1}_tracked_markers.sto')
#
#             # Extract joint reaction forces from solution
#             jrfTable = osim.analyzeMocoTrajectorySpatialVec(trackingModel, trackingSolution,
#                                                             ['.*patellofemoral_r.*reaction_on_child']).flatten()
#             osim.STOFileAdapter().write(jrfTable, f'{participant}{trialName}_{ii+1}_trackingSolution_JRF.sto')
#
#             # Extract muscle forces from solution
#             outputPaths = osim.StdVectorString()
#             outputPaths.append('.*tendon_force')
#             muscleForceTable = osim.analyzeMocoTrajectory(trackingModel, trackingSolution, outputPaths)
#             osim.STOFileAdapter().write(muscleForceTable, f'{participant}{trialName}_{ii+1}_trackingSolution_muscleForce.sto')
#
#             # Return to home directory
#             os.chdir(homeDir)
#
#             # Print out to console as a bookmark in any log file
#             # TODO: other logging notes...
#             print(f'***** ----- FINISHED TRACKING SIM FOR CONDITION {participant}{trialName} STANCE PHASE {ii+1} ----- *****')

# =========================================================================
# Finalise and exit
# =========================================================================

# Exit console to avoid exit code error
os._exit(00)

# %% ---------- end of runSimulations.py ---------- %% #

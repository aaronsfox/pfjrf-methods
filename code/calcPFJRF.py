# -*- coding: utf-8 -*-
"""

@author:

    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au

    This script calculates and collates the different PFJRF approaches from
    each participant's simulated running trials.

    NOTES:
        > Right-side data taken across all participants?

    TODO:
        > Why is OpenSim muscle force still different given the lack of difference in moment arm approach?
        > More plots and better detail, tidying etc.
        > Saving the plots
        > Saving the calculated data

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
from scipy.interpolate import interp1d
from glob import glob

# =========================================================================
# Set-up
# =========================================================================

# Set matplotlib parameters
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

# Add the utility geometry path for model visualisation
osim.ModelVisualizer.addDirToGeometrySearchPaths(os.path.join(os.getcwd(), '..', 'model', 'Geometry'))

# Get the participant list from the simulation directory
participantList = [os.path.split(ii.path)[-1] for ii in os.scandir(os.path.join('..','simulations')) if ii.is_dir()]

# Set actuator forces from torque driven simulations
trackingOptForces = {'pelvis_tx': 5, 'pelvis_ty': 5, 'pelvis_tz': 5, 'pelvis_tilt': 2.5, 'pelvis_list': 2.5, 'pelvis_rotation': 2.5,
                     'hip_flexion_r': 300, 'hip_adduction_r': 200, 'hip_rotation_r': 100,
                     'knee_angle_r': 300, 'ankle_angle_r': 200, 'subtalar_angle_r': 100,
                     'hip_flexion_l': 300, 'hip_adduction_l': 200, 'hip_rotation_l': 100,
                     'knee_angle_l': 300, 'ankle_angle_l': 200, 'subtalar_angle_l': 100,
                     }

# Read in participant info
participantInfo = pd.read_csv(os.path.join('..', 'data', 'participantInfo.csv'))

# =========================================================================
# Run through calculations for different PF JRF methods
# =========================================================================

# TODO: this is currently just broad test code that plots the results
# TODO: need a place to actually store the data...
# TODO: calculating instead?

# Loop through participants
for participant in participantList:

    # Set-up folders for participant results
    # -------------------------------------------------------------------------

    # Create participant results directory
    os.makedirs(os.path.join('..', 'results', participant), exist_ok=True)

    # Identify the runing trials for the participant
    participantFolders = [os.path.split(ii.path)[-1] for ii in os.scandir(os.path.join('..','simulations', participant)) if ii.is_dir()]
    runTrials = [ii for ii in participantFolders if ii.startswith('run')]

    # Create dictionary to store data for participant
    discreteTrialData = {'participant': [], 'runSpeed': [], 'gaitCycle': [], 'method': [], 'variable': [], 'value': []}
    continuousTrialData = {'participant': [], 'runSpeed': [], 'gaitCycle': [], 'method': [], 'variable': [], 'value': []}

    # Get participant mass
    massKg = participantInfo.loc[participantInfo['FileName'] == participant + 'static.c3d',]['Mass'].values[0]

    # Loop through running trials
    # -------------------------------------------------------------------------
    for trial in runTrials:

        # Identify the cycles simulated for the trial
        cycleList = [os.path.split(ii.path)[-1] for ii in os.scandir(
            os.path.join('..', 'simulations', participant, trial, 'muscle-driven')) if ii.is_dir()]

        # Loop through cycles
        # -------------------------------------------------------------------------
        for cycle in cycleList:

            # Extract OpenSim modelled JRF
            # -------------------------------------------------------------------------

            # Load in the JRF data
            opensim_JRF = osim.TimeSeriesTable(os.path.join('..', 'simulations', participant, trial, 'muscle-driven', cycle,
                                                            f'{participant}{trial}_{cycle[-1]}_inverseSolution_JRF.sto'))

            # Get right limb PFJRF
            opensim_PFJRF = np.array((opensim_JRF.getDependentColumn('/jointset/patellofemoral_r|reaction_on_child_4').to_numpy(),
                                      opensim_JRF.getDependentColumn('/jointset/patellofemoral_r|reaction_on_child_5').to_numpy(),
                                      opensim_JRF.getDependentColumn('/jointset/patellofemoral_r|reaction_on_child_6').to_numpy()))
            opensim_time = np.array(opensim_JRF.getIndependentColumn())

            # Calculate resultant 3D PFJRF
            opensim_PFJRF_resultant = [np.sqrt((opensim_PFJRF[0, ii] ** 2) + \
                                               (opensim_PFJRF[1, ii] ** 2) + \
                                               (opensim_PFJRF[2, ii] ** 2)) for ii in range(opensim_PFJRF.shape[1])]

            # Calculate discrete variables for 3D opensim data
            # Set the variables
            discreteVars = ['peakAbs', 'peakNorm', 'impulseAbs', 'impulseNorm']
            # Get the values
            values = [
                # Absolute and normalised peaks
                np.nanmax(opensim_PFJRF_resultant),
                np.nanmax(opensim_PFJRF_resultant) / massKg,
                # Absolute and normalised impulse
                np.trapz(opensim_PFJRF_resultant, dx=np.diff(opensim_time).mean()),
                np.trapz(opensim_PFJRF_resultant / massKg, dx=np.diff(opensim_time).mean()),
                ]
            # Add to dictionary
            for var in discreteVars:
                discreteTrialData['participant'].append(participant)
                discreteTrialData['runSpeed'].append(trial)
                discreteTrialData['gaitCycle'].append(cycle)
                discreteTrialData['method'].append('opensim_JRF3D')
                discreteTrialData['variable'].append(var)
                discreteTrialData['value'].append(values[discreteVars.index(var)])

            # Calculate discrete variables for 2D (sagittal plane) opensim data
            # Set the variables
            discreteVars = ['peakAbs', 'peakNorm', 'impulseAbs', 'impulseNorm']
            # Get the values
            values = [
                # Absolute and normalised peaks
                np.nanmax(opensim_PFJRF[0,:]),
                np.nanmax(opensim_PFJRF[0,:]) / massKg,
                # Absolute and normalised impulse
                np.trapz(opensim_PFJRF[0,:], dx=np.diff(opensim_time).mean()),
                np.trapz(opensim_PFJRF[0,:] / massKg, dx=np.diff(opensim_time).mean()),
                ]
            # Add to dictionary
            for var in discreteVars:
                discreteTrialData['participant'].append(participant)
                discreteTrialData['runSpeed'].append(trial)
                discreteTrialData['gaitCycle'].append(cycle)
                discreteTrialData['method'].append('opensim_JRF2D')
                discreteTrialData['variable'].append(var)
                discreteTrialData['value'].append(values[discreteVars.index(var)])

            # Extract time-normalised continuous values for opensim data
            # Set variables
            contVars = ['absPFJRF', 'normPFJRF']
            # Create interpolation functions
            f_3D = interp1d(opensim_time, opensim_PFJRF_resultant)
            f_2D = interp1d(opensim_time, opensim_PFJRF[0,:])
            # Calculate values
            vals_3D = [
                f_3D(np.linspace(opensim_time[0], opensim_time[-1], 101)),
                f_3D(np.linspace(opensim_time[0], opensim_time[-1], 101)) / massKg
            ]
            vals_2D = [
                f_2D(np.linspace(opensim_time[0], opensim_time[-1], 101)),
                f_2D(np.linspace(opensim_time[0], opensim_time[-1], 101)) / massKg
            ]
            # Add to dictionary
            for var in contVars:
                # 3D data
                continuousTrialData['participant'].append(participant)
                continuousTrialData['runSpeed'].append(trial)
                continuousTrialData['gaitCycle'].append(cycle)
                continuousTrialData['method'].append('opensim_JRF3D')
                continuousTrialData['variable'].append(var)
                continuousTrialData['value'].append(vals_3D[contVars.index(var)])
                # 2D data
                continuousTrialData['participant'].append(participant)
                continuousTrialData['runSpeed'].append(trial)
                continuousTrialData['gaitCycle'].append(cycle)
                continuousTrialData['method'].append('opensim_JRF2D')
                continuousTrialData['variable'].append(var)
                continuousTrialData['value'].append(vals_2D[contVars.index(var)])

            # Use standardised equation method to calculate PF JRF
            # -------------------------------------------------------------------------

            # Load in the torque driven tracking solution data to get joint torque
            trackingData = osim.TimeSeriesTable(os.path.join('..', 'simulations', participant, trial, 'torque-driven', cycle,
                                                             f'{participant}{trial}_{cycle[-1]}_torque-driven_marker-solution.sto'))
            tracking_time = np.array(trackingData.getIndependentColumn())

            # Get the knee angle kinematics
            kneeAngle = np.rad2deg(trackingData.getDependentColumn('/jointset/walker_knee_r/knee_angle_r/value').to_numpy())

            # Get the knee angle torque data
            # Convert this to be positive for knee flexion moment
            kneeAngleTorque = trackingData.getDependentColumn('/forceset/knee_angle_r_torque').to_numpy() * trackingOptForces['knee_angle_r'] * -1

            # Calculate the quadriceps effective lever arm based on knee angle
            equation_quadMA = ((0.00000008 * (kneeAngle ** 3)) - (0.000013 * (kneeAngle ** 2)) + (0.00028 * kneeAngle) + 0.046)

            # Calculate quadriceps force based on knee moment
            equation_quadsForceN = kneeAngleTorque / equation_quadMA

            # Calculate k coefficient based on knee flexion angle
            aa = ((-0.000038) * (kneeAngle ** 2)) + (0.0015 * kneeAngle) + 0.462
            bb = ((-0.0000007) * (kneeAngle ** 3)) + (0.000155 * (kneeAngle ** 2)) - (0.016 * kneeAngle) + 1
            k = aa / bb

            # Calculate PF JRF
            equation_PFJRF = k * equation_quadsForceN

            # Set any negative values to zero
            equation_PFJRF[equation_PFJRF < 0] = 0

            # Calculate discrete variables for equation method
            # Set the variables
            discreteVars = ['peakAbs', 'peakNorm', 'impulseAbs', 'impulseNorm']
            # Get the values
            values = [
                # Absolute and normalised peaks
                np.nanmax(equation_PFJRF),
                np.nanmax(equation_PFJRF) / massKg,
                # Absolute and normalised impulse
                np.trapz(equation_PFJRF, dx=np.diff(tracking_time).mean()),
                np.trapz(equation_PFJRF / massKg, dx=np.diff(tracking_time).mean()),
                ]
            # Add to dictionary
            for var in discreteVars:
                discreteTrialData['participant'].append(participant)
                discreteTrialData['runSpeed'].append(trial)
                discreteTrialData['gaitCycle'].append(cycle)
                discreteTrialData['method'].append('equation')
                discreteTrialData['variable'].append(var)
                discreteTrialData['value'].append(values[discreteVars.index(var)])

            # Extract time-normalised continuous values for equation data
            # Set variables
            contVars = ['absPFJRF', 'normPFJRF']
            # Create interpolation functions
            f = interp1d(tracking_time, equation_PFJRF)
            # Calculate values
            vals = [
                f(np.linspace(tracking_time[0], tracking_time[-1], 101)),
                f(np.linspace(tracking_time[0], tracking_time[-1], 101)) / massKg
            ]
            # Add to dictionary
            for var in contVars:
                # 3D data
                continuousTrialData['participant'].append(participant)
                continuousTrialData['runSpeed'].append(trial)
                continuousTrialData['gaitCycle'].append(cycle)
                continuousTrialData['method'].append('equation')
                continuousTrialData['variable'].append(var)
                continuousTrialData['value'].append(vals[contVars.index(var)])

            # Use equation with OpenSim moment arm to calculate PFJRF
            # -------------------------------------------------------------------------

            # Run a muscle analysis to get muscle moment arms
            analyzeTool = osim.AnalyzeTool()
            muscleAnalysis = osim.MuscleAnalysis()

            # Set model and coordinates to use in analysis
            analyzeTool.setModelFilename(os.path.join('..', 'simulations', participant, 'scaling', f'{participant}_scaledModelAdjusted.osim'))
            analyzeTool.setCoordinatesFileName(os.path.join('..', 'simulations', participant, trial, 'muscle-driven', cycle,
                                                            f'{participant}{trial}_{cycle[-1]}_coordinates.sto'))

            # Set times
            analyzeTool.setStartTime(osim.TimeSeriesTable(os.path.join('..', 'simulations', participant, trial, 'muscle-driven', cycle,
                                                                       f'{participant}{trial}_{cycle[-1]}_coordinates.sto'
                                                                       )).getIndependentColumn()[0])
            analyzeTool.setFinalTime(osim.TimeSeriesTable(os.path.join('..', 'simulations', participant, trial, 'muscle-driven', cycle,
                                                                       f'{participant}{trial}_{cycle[-1]}_coordinates.sto'
                                                                       )).getIndependentColumn()[-1])

            # Set parameters in muscle analysis
            # Muscles to analyse
            musclesToAnalyse = osim.ArrayStr()
            [musclesToAnalyse.append(musc) for musc in ['recfem_r', 'vaslat_r', 'vasint_r', 'vasmed_r']]
            muscleAnalysis.setMuscles(musclesToAnalyse)
            # Moment arm coordinates
            coordToAnalyse = osim.ArrayStr()
            [coordToAnalyse.append(musc) for musc in ['knee_angle_r']]
            muscleAnalysis.setCoordinates(coordToAnalyse)
            muscleAnalysis.printToXML('analysis.xml')

            # Append muscle analysis to tool
            analyzeTool.getAnalysisSet().cloneAndAppend(muscleAnalysis)

            # Save and read in tool to run (seems to avoid kernel crashing)
            analyzeTool.printToXML('analyzeTool.xml')
            analyzeRun = osim.AnalyzeTool('analyzeTool.xml')
            analyzeRun.run()

            # Set muscle names to extract
            muscNames = ['recfem_r', 'vaslat_r', 'vasint_r', 'vasmed_r']

            # Create list to store average moment arm
            opensim_quadMA = []

            # Read in analysis results
            maResults = osim.TimeSeriesTable('_MuscleAnalysis_MomentArm_knee_angle_r.sto')

            # Calculate the average quadriceps moment arm from each state
            # Note this needs to be inverted to match the equation moment arms
            for timeInd in range(maResults.getNumRows()):
                opensim_quadMA.append(np.mean([maResults.getDependentColumn(musc)[timeInd] * -1 for musc in muscNames]))

            # Calculate quadriceps force based on previously used knee moment and new quads moment arm
            opensimMA_quadsForceN = kneeAngleTorque / opensim_quadMA

            # Calculate PF JRF using existing coefficient from knee angle and new quads force
            opensimMA_PFJRF = k * opensimMA_quadsForceN

            # Set any negative values to zero
            opensimMA_PFJRF[opensimMA_PFJRF < 0] = 0

            # Calculate discrete variables for equation+opensimMA method
            # Set the variables
            discreteVars = ['peakAbs', 'peakNorm', 'impulseAbs', 'impulseNorm']
            # Get the values
            values = [
                # Absolute and normalised peaks
                np.nanmax(opensimMA_PFJRF),
                np.nanmax(opensimMA_PFJRF) / massKg,
                # Absolute and normalised impulse
                np.trapz(opensimMA_PFJRF, dx=np.diff(tracking_time).mean()),
                np.trapz(opensimMA_PFJRF / massKg, dx=np.diff(tracking_time).mean()),
            ]
            # Add to dictionary
            for var in discreteVars:
                discreteTrialData['participant'].append(participant)
                discreteTrialData['runSpeed'].append(trial)
                discreteTrialData['gaitCycle'].append(cycle)
                discreteTrialData['method'].append('equation+opensimMA')
                discreteTrialData['variable'].append(var)
                discreteTrialData['value'].append(values[discreteVars.index(var)])

            # Extract time-normalised continuous values for equation+opensimMA data
            # Set variables
            contVars = ['absPFJRF', 'normPFJRF']
            # Create interpolation functions
            f = interp1d(tracking_time, opensimMA_PFJRF)
            # Calculate values
            vals = [
                f(np.linspace(tracking_time[0], tracking_time[-1], 101)),
                f(np.linspace(tracking_time[0], tracking_time[-1], 101)) / massKg
            ]
            # Add to dictionary
            for var in contVars:
                # 3D data
                continuousTrialData['participant'].append(participant)
                continuousTrialData['runSpeed'].append(trial)
                continuousTrialData['gaitCycle'].append(cycle)
                continuousTrialData['method'].append('equation+opensimMA')
                continuousTrialData['variable'].append(var)
                continuousTrialData['value'].append(vals[contVars.index(var)])

            # Use equation with OpenSim muscle forces to calculate PFJRF
            # -------------------------------------------------------------------------

            # Read in muscle force data from inverse simulation
            opensim_MF = osim.TimeSeriesTable(os.path.join('..', 'simulations', participant, trial, 'muscle-driven', cycle,
                                                           f'{participant}{trial}_{cycle[-1]}_inverseSolution_muscleForce.sto'))
            opensim_MF_time = np.array(opensim_MF.getIndependentColumn())

            # Create list to store total quads muscle force
            opensim_quadsForceN = []

            # Calculate the total quadriceps muscle force from each time index
            # Note this needs to be inverted to match the equation moment arms
            for timeInd in range(opensim_MF.getNumRows()):
                opensim_quadsForceN.append(np.sum([opensim_MF.getDependentColumn(
                    f'/forceset/{musc}|tendon_force').to_numpy()[timeInd] for musc in muscNames]))

            # Interpolate quadriceps force up to same number of time-stamps as tracking data
            f = interp1d(opensim_MF_time, np.array(opensim_quadsForceN))
            opensim_quadsForceN = f(np.linspace(opensim_MF_time[0], opensim_MF_time[-1], len(tracking_time)))

            # Calculate PF JRF using existing coefficient from knee angle and OpenSim quads force
            opensimMF_PFJRF = k * opensim_quadsForceN

            # Set any negative values to zero
            opensimMF_PFJRF[opensimMF_PFJRF < 0] = 0

            # Calculate discrete variables for equation+opensimMF method
            # Set the variables
            discreteVars = ['peakAbs', 'peakNorm', 'impulseAbs', 'impulseNorm']
            # Get the values
            values = [
                # Absolute and normalised peaks
                np.nanmax(opensimMF_PFJRF),
                np.nanmax(opensimMF_PFJRF) / massKg,
                # Absolute and normalised impulse
                np.trapz(opensimMF_PFJRF, dx=np.diff(tracking_time).mean()),
                np.trapz(opensimMF_PFJRF / massKg, dx=np.diff(tracking_time).mean()),
            ]
            # Add to dictionary
            for var in discreteVars:
                discreteTrialData['participant'].append(participant)
                discreteTrialData['runSpeed'].append(trial)
                discreteTrialData['gaitCycle'].append(cycle)
                discreteTrialData['method'].append('equation+opensimMF')
                discreteTrialData['variable'].append(var)
                discreteTrialData['value'].append(values[discreteVars.index(var)])

            # Extract time-normalised continuous values for equation+opensimMA data
            # Set variables
            contVars = ['absPFJRF', 'normPFJRF']
            # Create interpolation functions
            f = interp1d(tracking_time, opensimMF_PFJRF)
            # Calculate values
            vals = [
                f(np.linspace(tracking_time[0], tracking_time[-1], 101)),
                f(np.linspace(tracking_time[0], tracking_time[-1], 101)) / massKg
            ]
            # Add to dictionary
            for var in contVars:
                # 3D data
                continuousTrialData['participant'].append(participant)
                continuousTrialData['runSpeed'].append(trial)
                continuousTrialData['gaitCycle'].append(cycle)
                continuousTrialData['method'].append('equation+opensimMF')
                continuousTrialData['variable'].append(var)
                continuousTrialData['value'].append(vals[contVars.index(var)])

            # Clean-up analysis files
            # -------------------------------------------------------------------------

            # Remove analyze tool file
            os.remove('analyzeTool.xml')

            # Get muscle analysis results file names and delete
            deleteFiles = glob('_MuscleAnalysis*')
            for file in deleteFiles:
                os.remove(file)

    # TODO: UP TO HERE
    # -------------------------------------------------------------------------

    # TODO: sabve .pkl output dictionaries

    # Convert continuous dictionary to dataframe
    continuousData = pd.DataFrame().from_dict(continuousTrialData)

    # Plot to visualise data
    # -------------------------------------------------------------------------

    # TODO: multiple better looking plots...

    # 2D OpenSim data
    plotData = np.vstack(continuousData.loc[(continuousData['method'] == 'opensim_JRF2D') &
                                            (continuousData['runSpeed'] == 'runT25') &
                                            (continuousData['variable'] == 'absPFJRF')]['value'].values)
    plt.plot(plotData.T, ls='-', lw=1.0, c='black', alpha=0.3, zorder = 2)
    plt.plot(np.mean(plotData, axis = 0), ls='-', lw=1.5, c='black', zorder = 3)

    # Equation approach
    plotData = np.vstack(continuousData.loc[(continuousData['method'] == 'equation') &
                                            (continuousData['runSpeed'] == 'runT25') &
                                            (continuousData['variable'] == 'absPFJRF')]['value'].values)
    plt.plot(plotData.T, ls='-', lw=1.0, c='red', alpha=0.3, zorder=2)
    plt.plot(np.mean(plotData, axis=0), ls='-', lw=1.5, c='red', zorder=3)

    # Equation + opensim MA approach
    plotData = np.vstack(continuousData.loc[(continuousData['method'] == 'equation+opensimMA') &
                                            (continuousData['runSpeed'] == 'runT25') &
                                            (continuousData['variable'] == 'absPFJRF')]['value'].values)
    plt.plot(plotData.T, ls='-', lw=1.0, c='blue', alpha=0.3, zorder=2)
    plt.plot(np.mean(plotData, axis=0), ls='-', lw=1.5, c='blue', zorder=3)

    # Equation + opensim MF approach
    plotData = np.vstack(continuousData.loc[(continuousData['method'] == 'equation+opensimMF') &
                                            (continuousData['runSpeed'] == 'runT25') &
                                            (continuousData['variable'] == 'absPFJRF')]['value'].values)
    plt.plot(plotData.T, ls='-', lw=1.0, c='magenta', alpha=0.3, zorder=2)
    plt.plot(np.mean(plotData, axis=0), ls='-', lw=1.5, c='magenta', zorder=3)

    plt.title('Black (OpenSim 2D JRA), Red (Equation), \nBlue (Equation w/ OpenSim MA), Purple (Equation w/ OpenSim MF)', fontsize = 11)




# %% ---------- end of calcPFJRF.py ---------- %% #
# -*- coding: utf-8 -*-
"""

@author:

    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au
    
    This script extracts and cleans up the necessary files from the Fukuchi2017 dataset.

"""

# =========================================================================
# Import packages
# =========================================================================

import os
import glob
import shutil
import opensim as osim
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# =========================================================================
# Extract necessary files
# =========================================================================

# Identify subject list based on filenames
# Get file list of static trials
allParticipants = [os.path.split(file)[-1].split('static.c3d')[0] for file in glob.glob(os.path.join('raw', '*static.c3d'))]

# Get the desired sum-sample of participants (inc. and up to RBDS028)
participantList = []
for participant in allParticipants:
    participantNo = int(''.join(ii for ii in participant if ii.isdigit()))
    if participantNo <= 28:
        participantList.append(participant)

# Drop RBDS06 and RBDS07 as they have inconsistent lab set-up for trials
participantList.remove('RBDS06')
participantList.remove('RBDS07')

# Loop through participants and allocate files to folder
for participant in participantList:

    # Get files for current participant
    participantFiles = [file for file in glob.glob(os.path.join('raw',participant+'[a-z]*.c3d'))]

    # Create directory for current subject
    os.makedirs(participant, exist_ok=True)

    # Loop through file list and copy to directory
    for file in participantFiles:
        shutil.copy(file, os.path.join(participant, os.path.split(file)[-1]))

# =========================================================================
# Convert files to OpenSim format
# =========================================================================

# Loop through participants
for participant in participantList:

    # Convert static trial
    # -------------------------------------------------------------------------

    # Set static file name
    staticFile = os.path.join(participant, participant+'static.c3d')

    # Construct opensim 3d object
    c3dFile = osim.C3DFileAdapter()
    c3dFile.setLocationForForceExpression(osim.C3DFileAdapter.ForceLocation_CenterOfPressure)

    # Read in the static trial
    staticC3D = c3dFile.read(staticFile)

    # Get markers table
    staticMarkers = c3dFile.getMarkersTable(staticC3D)

    # Write static markers to TRC file
    osim.TRCFileAdapter().write(staticMarkers,
                                os.path.join(participant, participant + 'static.trc'))

    # Convert dynamic trials
    # -------------------------------------------------------------------------

    # Get file list
    dynamicFiles = glob.glob(os.path.join(participant, '*run*.c3d'))

    # Loop through dynamic files
    for file in dynamicFiles:

        # Construct opensim 3d object
        c3dFile = osim.C3DFileAdapter()
        c3dFile.setLocationForForceExpression(osim.C3DFileAdapter.ForceLocation_CenterOfPressure)

        # Read in the dynamic trial
        dynaC3D = c3dFile.read(file)

        # Marker data
        # -------------------------------------------------------------------------

        # Get markers table
        dynaMarkers = c3dFile.getMarkersTable(dynaC3D).flatten()

        # Get number of column labels and data rows
        nCols = dynaMarkers.getNumColumns()
        nRows = dynaMarkers.getNumRows()

        # Pre-allocate numpy array based on data size
        markerData = np.zeros((nRows, nCols))

        # Get marker labels
        markerLabelsFlat = list(dynaMarkers.getColumnLabels())

        # Extract data to columns
        for marker in markerLabelsFlat:
            markerData[:, markerLabelsFlat.index(marker)] = dynaMarkers.getDependentColumn(marker).to_numpy()

        # Create filter to apply to data. 10Hz low-pass as in original paper
        # Get the sampling rate
        fs = 1 / np.diff(np.array(dynaMarkers.getIndependentColumn())).mean()
        # Set filter frequency
        filtFreq = 10
        # Define low-pass Butterworth filter
        nyq = 0.5 * fs
        normCutoff = filtFreq / nyq
        b, a = butter(4, normCutoff, btype='low', analog=False)

        # Apply lowpass filter to marker data columns
        for iCol in range(nCols):
            markerData[:, iCol] = filtfilt(b, a, markerData[:, iCol])

        # Replace table data with rows of filtered data
        for iRow in range(nRows):
            dynaMarkers.setRowAtIndex(iRow, osim.RowVector().createFromMat(markerData[iRow, :]))

        # Pack table back to Vec3 format
        markerTableVec3 = dynaMarkers.packVec3()

        # Write dynamic markers to TRC file
        osim.TRCFileAdapter().write(markerTableVec3,
                                    os.path.join(participant, os.path.split(file)[-1].split('.c3d')[0]+'.trc'))

        # Forces data
        # -------------------------------------------------------------------------

        # Get forces table
        forcesFlat = c3dFile.getForcesTable(dynaC3D).flatten()

        # Convert to numpy array
        # Pre-allocate numpy array based on data size
        dataArray = np.zeros((forcesFlat.getNumRows(), forcesFlat.getNumColumns()))

        # Extract data
        for forceInd in range(forcesFlat.getNumColumns()):
            dataArray[:, forceInd] = forcesFlat.getDependentColumn(forcesFlat.getColumnLabels()[forceInd]).to_numpy()

        # Replace nan's for COP and moment data with zeros
        np.nan_to_num(dataArray, copy=False, nan=0.0)

        # Convert force point data from mm to m
        for forceName in list(forcesFlat.getColumnLabels()):
            if forceName.startswith('p') or forceName.startswith('m'):
                # Get force index
                forceInd = list(forcesFlat.getColumnLabels()).index(forceName)
                # Convert to m units in data array
                dataArray[:, forceInd] = dataArray[:, forceInd] / 1000

        # Create filter to apply to data. 10Hz low-pass as in original paper
        # Get the sampling rate
        fs = 1 / np.diff(np.array(forcesFlat.getIndependentColumn())).mean()
        # Set filter frequency
        filtFreq = 10
        # Define low-pass Butterworth filter
        nyq = 0.5 * fs
        normCutoff = filtFreq / nyq
        b, a = butter(4, normCutoff, btype='low', analog=False)

        # # Apply lowpass filter to marker data columns
        # for iCol in range(dataArray.shape[1]):
        #     dataArray[:, iCol] = filtfilt(b, a, dataArray[:, iCol])

        # Get the vertical force data out for allocating foot contacts
        # Use a filtered version of this to make identifying crossing easier
        forceInd = list(forcesFlat.getColumnLabels()).index('f1_2')
        vForce = filtfilt(b,a,dataArray[:, forceInd])

        # Find where data is below 100N threshold being used for contact identification
        # Higher force threshold seemed to help with noisy-ish COP data at foot contact
        vertForceThreshold = 100
        zeroDataLogical = vForce < vertForceThreshold

        # # Zero all data where no contact is specified
        # for ii in range(dataArray.shape[1]):
        #     dataArray[zeroDataLogical,ii] = 0

        # Zero vertical force data below threshold
        vForce[zeroDataLogical] = 0

        # Identify contact indices based on force threshold
        thresholdCrossings = np.diff(vForce > vertForceThreshold, prepend=False)
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

        # # Test visual of contact pairs
        # plt.plot(vForce)
        # plt.scatter([contactPairs[ii][0] for ii in range(len(contactPairs))],
        #             np.zeros(len(contactPairs)), s = 5, marker = 'o', color = 'green')
        # plt.scatter([contactPairs[ii][1] for ii in range(len(contactPairs))],
        #             np.zeros(len(contactPairs)), s=5, marker='o', color='red')

        # Create a new data array to store right and left contact forces
        # Note that right limb is first set of columns and left limb is second set of columns
        forcesData = np.zeros((forcesFlat.getNumRows(), forcesFlat.getNumColumns()*2))

        # Loop through pairs to figure out foot contact and allocate forces
        for pair in contactPairs:

            # Get time at mid-point of force contact
            midContactTime = forcesFlat.getIndependentColumn()[int(pair[0]+(np.round((pair[1] - pair[0]) / 2)))]

            # Get marker row index at time point
            markerInd = dynaMarkers.getNearestRowIndexForTime(midContactTime)

            # Get vertical position of designated right and left markers at specified index
            rMarker = dynaMarkers.getDependentColumn('R.Heel.Bottom_2').to_numpy()[markerInd]
            lMarker = dynaMarkers.getDependentColumn('L.Heel.Bottom_2').to_numpy()[markerInd]

            # Check for lower marker
            if rMarker < lMarker:
                contactLimb = 'right'
            elif lMarker < rMarker:
                contactLimb = 'left'

            # Extract the data into the appropriate columns for force array
            # Filter here across the contact period instead of the whole dataset
            if contactLimb == 'right':
                for ii in range(dataArray.shape[1]):
                    forcesData[pair[0]:pair[1],ii] = filtfilt(b,a,dataArray[pair[0]:pair[1],ii])
            elif contactLimb == 'left':
                for ii in range(dataArray.shape[1]):
                    forcesData[pair[0]:pair[1],ii+9] = filtfilt(b,a,dataArray[pair[0]:pair[1],ii])

        # Build the new time series table
        forcesStorage = osim.Storage()

        # Get the time data
        time = forcesFlat.getIndependentColumn()

        # Create labels for new force data
        forceLabels = []
        for limb in ['r', 'l']:
            forceLabels.append(f'ground_force_{limb}_vx')
            forceLabels.append(f'ground_force_{limb}_vy')
            forceLabels.append(f'ground_force_{limb}_vz')
            forceLabels.append(f'ground_force_{limb}_px')
            forceLabels.append(f'ground_force_{limb}_py')
            forceLabels.append(f'ground_force_{limb}_pz')
            forceLabels.append(f'ground_force_{limb}_mx')
            forceLabels.append(f'ground_force_{limb}_my')
            forceLabels.append(f'ground_force_{limb}_mz')

        # Set labels in table
        newLabels = osim.ArrayStr()
        newLabels.append('time')
        for label in forceLabels:
            newLabels.append(label)
        forcesStorage.setColumnLabels(newLabels)

        # Add data
        for iRow in range(forcesData.shape[0]):
            row = osim.ArrayDouble()
            for iCol in range(forcesData.shape[1]):
                row.append(forcesData[iRow, iCol])
            # Add data to storage
            forcesStorage.append(time[iRow], row)

        # Set name for storage object
        forcesStorage.setName(os.path.split(file)[-1].split('.')[0] + '_grf')

        # Write to file
        forcesStorage.printResult(forcesStorage,
                                  os.path.split(file)[-1].split('.')[0] + '_grf',
                                  participant,
                                  0.001, '.mot')

        # Create external loads files
        # Note that external loads are applied to feet based on timing of force plate contact and event name

        # Create the external loads
        forceXML = osim.ExternalLoads()

        # Create right side external force
        rightGRF = osim.ExternalForce()
        rightGRF.setName('RightGRF')
        rightGRF.setAppliedToBodyName('calcn_r')
        rightGRF.setForceExpressedInBodyName('ground')
        rightGRF.setPointExpressedInBodyName('ground')
        rightGRF.setForceIdentifier('ground_force_r_v')
        rightGRF.setPointIdentifier('ground_force_r_p')
        rightGRF.setTorqueIdentifier('ground_force_r_m')
        forceXML.cloneAndAppend(rightGRF)

        # Create left side external force
        leftGRF = osim.ExternalForce()
        leftGRF.setName('LeftGRF')
        leftGRF.setAppliedToBodyName('calcn_l')
        leftGRF.setForceExpressedInBodyName('ground')
        leftGRF.setPointExpressedInBodyName('ground')
        leftGRF.setForceIdentifier('ground_force_l_v')
        leftGRF.setPointIdentifier('ground_force_l_p')
        leftGRF.setTorqueIdentifier('ground_force_l_m')
        forceXML.cloneAndAppend(leftGRF)

        # Set GRF datafile in external loads
        forceXML.setDataFileName(os.path.split(file)[-1].split('.')[0] + '_grf.mot')

        # Write to file
        forceXML.printToXML(os.path.join(participant, os.path.split(file)[-1].split('.')[0] + '_grf.xml'))

    # Print confirmation
    print(f'Finished data extraction for {participant}...')

# %% ---------- end of extract_data.py ---------- %% #


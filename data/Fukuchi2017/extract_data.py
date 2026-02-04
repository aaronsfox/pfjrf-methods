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
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import label, median_filter
import matplotlib.pyplot as plt

# =========================================================================
# Set-up
# =========================================================================

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

# =========================================================================
# Extract necessary files
# =========================================================================

# Identify subject list based on filenames
# Get file list of static trials
allParticipants = [os.path.split(file)[-1].split('static.c3d')[0] for file in glob.glob(os.path.join('raw', '*static.c3d'))]

# Get the desired sub-sample of participants (inc. and up to RBDS028)
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

# # Option to get participant list from existing directories
# participantList = [d for d in os.listdir(os.getcwd()) if os.path.isdir(os.path.join(os.getcwd(), d))]

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
        # Firstly correct exactly zero values as this is where data is missing
        for iCol in range(nCols):
            # Replace any zeros with nan's
            markerData[:, iCol][markerData[:, iCol] == 0] = np.nan
            # Check need to interpolate over these before filtering
            if any(np.isnan(markerData[:, iCol])):
                # Apply spline to fill gaps
                mask = ~np.isnan(markerData[:, iCol])
                # Create cubic smoothing spline (k=3)
                # s controls smoothing: 0 = exact fit, larger = smoother
                # Check for enough data points
                if mask.sum() > 3:
                    spline = UnivariateSpline(np.array(dynaMarkers.getIndependentColumn())[mask], markerData[:, iCol][mask],
                                              k=3, s=2.0)
                    # Apply spline to missing data
                    splined_missing = spline(np.array(dynaMarkers.getIndependentColumn())[~mask])
                    # Fill the missing marker data
                    markerData[:, iCol][~mask] = splined_missing
            # Filter marker data
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

        # Get the vertical force data out for allocating foot contacts
        # Use a filtered version of this to make identifying crossing easier
        forceInd = list(forcesFlat.getColumnLabels()).index('f1_2')
        vForce = dataArray[:, forceInd]
        # vForce = filtfilt(b,a,dataArray[:, forceInd])

        # Find where data is above vertical force threshold of 50N for at least 25 frames
        vert_threshold = 50
        consec_frames = 25
        # Find where data is over threshold
        above = vForce > vert_threshold
        # Identify where above criteria meets frame length criteria
        labeled_array, num_features = label(above)
        contact_mask = np.zeros_like(vForce, dtype=bool)
        for feature in range(1, num_features + 1):
            indices = np.where(labeled_array == feature)[0]
            if len(indices) >= consec_frames:
                contact_mask[indices] = True

        # # Plot to check contact identification
        # plt.plot(vForce)
        # plt.plot(contact_mask*2000)

        # Set all force data to zero outside of contact periods
        dataArray[~contact_mask,:] = 0

        # Identify contact pairs
        diff = np.diff(contact_mask.astype(int))
        rising_edge = np.where(diff == 1)[0] + 1
        falling_edge = np.where(diff == -1)[0] + 1
        # Handle mask starting or ending True
        if contact_mask[0]:
            rising_edge = np.insert(rising_edge, 0, 0)
        if contact_mask[-1]:
            falling_edge = np.append(falling_edge, len(contact_mask))
        # Combine to contact pairs
        contact_pairs = list(zip(rising_edge, falling_edge))

        # # Test visual of contact pairs
        # plt.plot(dataArray[:, forceInd])
        # plt.scatter([contact_pairs[ii][0] for ii in range(len(contact_pairs))],
        #             np.zeros(len(contact_pairs)), s = 5, marker = 'o', color = 'green')
        # plt.scatter([contact_pairs[ii][1] for ii in range(len(contact_pairs))],
        #             np.zeros(len(contact_pairs)), s=5, marker='o', color='red')

        # Create a new data array to store right and left contact forces
        # Note that right limb is first set of columns and left limb is second set of columns
        forcesData = np.zeros((forcesFlat.getNumRows(), forcesFlat.getNumColumns()*2))

        # Loop through pairs to figure out foot contact and allocate forces
        for pair in contact_pairs:

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
            # Apply a median filter to centre of pressure data before filtering to fix noisy initial periods
            if contactLimb == 'right':
                for ii in range(dataArray.shape[1]):
                    if forcesFlat.getColumnLabels()[ii].startswith('p'):
                        forcesData[pair[0]:pair[1], ii] = filtfilt(
                            b,a,median_filter(dataArray[pair[0]:pair[1],ii],size=10)
                        )
                    else:
                        forcesData[pair[0]:pair[1],ii] = filtfilt(b,a,dataArray[pair[0]:pair[1],ii])
            elif contactLimb == 'left':
                for ii in range(dataArray.shape[1]):
                    if forcesFlat.getColumnLabels()[ii].startswith('p'):
                        forcesData[pair[0]:pair[1],ii+9] = filtfilt(
                            b, a, median_filter(dataArray[pair[0]:pair[1], ii], size=10)
                        )
                    else:
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
        for label_f in forceLabels:
            newLabels.append(label_f)
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


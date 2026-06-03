# -*- coding: utf-8 -*-
"""

@author:

    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au

    This script calculates and collates the different PFJRF approaches from
    each participant's simulated trials from the Loone2025 dataset.

    TODO:
        > Applies equation method with optimisation data instead of analyze tools forces
            >> Is this what we want?


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
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import interp1d
from glob import glob
import pickle
from matplotlib import rcParams
import matplotlib
from scipy.constants import g

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
# Note this uses simulation folder as this is where data is being extracted from
participant_list = [ii for ii in os.listdir(
    os.path.join('..', '..', 'simulations', dataset)) if os.path.isdir(
    os.path.join(os.path.join('..', '..', 'simulations', dataset, ii)))]

# Set the list of conditions to process
# Currently only one anyway, but if more were to be added it could be done
condition_list = [
    'SRRun',   # standard shoe running
    ]

# Create directory for storing results
os.makedirs(os.path.join('..', '..', 'outputs', dataset), exist_ok=True)

# Plot settings
# -------------------------------------------------------------------------

# Set matplotlib parameters
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

# Set colouring for simulation methods
plot_col = {
    'inverse_dynamics': '#363635',  # charcoal
    'static_optimisation': '#58BCAF',  # turquoise
    'dynamic_optimisation': '#C74298',  # pink
}

# OpenSim settings
# -------------------------------------------------------------------------

# Add the utility geometry path for model visualisation
osim.ModelVisualizer.addDirToGeometrySearchPaths(os.path.join(os.getcwd(), '..', 'model', 'Geometry'))

# Set the final mesh interval that was used for dynamic optimisations for file loading
mesh_int = 25

# Set the muscles to group together when it comes to plotting forces
# Some muscles are also excluded here for brevity
muscle_groups = {'Add. Brev.': ['addbrev_r'],
                 'Add. Long.': ['addlong_r'],
                 'Add. Mag.': ['addmagDist_r', 'addmagIsch_r', 'addmagMid_r', 'addmagProx_r'],
                 'Biceps Fem. LH': ['bflh_r'],
                 'Biceps Fem. SH': ['bfsh_r'],
                 'Gas. Lat.': ['gaslat_r'],
                 'Gas. Med.': ['gasmed_r'],
                 'Glut. Max.': ['glmax1_r', 'glmax2_r', 'glmax3_r'],
                 'Glut. Med.': ['glmed1_r', 'glmed2_r', 'glmed3_r'],
                 'Glut. Min.': ['glmin1_r', 'glmin2_r', 'glmin3_r'],
                 'Gracilis': ['grac_r'],
                 'Iliacus': ['iliacus_r'],
                 'Rec. Fem.': ['recfem_r'],
                 'Sartorius': ['sart_r'],
                 'Semimem.': ['semimem_r'],
                 'Semiten.': ['semiten_r'],
                 'Soleus.': ['soleus_r'],
                 'TFL': ['tfl_r'],
                 'Tib. Ant.': ['tibant_r'],
                 'Tib. Post.': ['tibpost_r'],
                 'Vas. Int.': ['vasint_r'],
                 'Vas. Med.': ['vasmed_r'],
                 'Vas. Lat.': ['vaslat_r']
                 }

# Set complex model quadricep muscles
complex_quads = ['recfem_r', 'vasmed_r', 'vasint_r', 'vaslat_r']

# Analysis settings
# -------------------------------------------------------------------------

# Set coefficients for various equations
# k coefficient for knee angle
# Uses method from van Eijden (1986) identified as most popular approach in Nunes et al. (2018)
k_numerator_coeffs = [-3.84e-05, 1.47e-03, 4.62e-01]
k_denominator_coeffs = [-6.98e-07, 1.55e-04, -1.62e-02, 1.0]
# Quadriceps lever arm (note that this is slightly modified from Nunes to express in metres)
q_lever_coeffs = [0.00000008, -0.000013, 0.00028, 0.046]
# Contact area based on knee angle
# See Doyle et al. (2023) and Nunes et al. (2018)
ca_mm2_coeffs = [2.0e-05, 0.0033, 0.1099, 3.5273, 81.058]
ca_mm2_coeffs_kernozek = [-0.0001, -0.0082, 3.5071, 73.81]

# =========================================================================
# Define functions
# =========================================================================

# Calculate the PFJRF and PFJS for a participant at a given speed
# -------------------------------------------------------------------------
def calculate_pfj_loads(participant):

    """
    Function calculates the desired PFJ loads for a participant across all speeds
    using the different available methods.
    """

    # Create directories for outputs
    os.makedirs(os.path.join('..', '..', 'outputs', dataset, participant, 'figures'), exist_ok=True)
    os.makedirs(os.path.join('..', '..', 'outputs', dataset, participant, 'results'), exist_ok=True)

    # Create dictionary to store data in
    pfjrf = {method: {condition: {'time': np.zeros(101), 'pfjrf': np.zeros(101)} for condition in condition_list} for method in plot_col.keys()}
    pfjs = {method: {condition: {'time': np.zeros(101), 'pfjs': np.zeros(101)}  for condition in condition_list} for method in plot_col.keys()}
    qf = {method: {condition: {'time': np.zeros(101), 'qf': np.zeros(101)} for condition in condition_list} for method in plot_col.keys()}
    computation_time = {method: {condition: np.zeros(1) for condition in condition_list} for method in plot_col.keys()}

    # Get participant mass
    mass_kg = participantInfo[participantInfo['participant_id'] == participant]['mass'].values[0]

    # Loop through speeds for calculations
    for condition in condition_list:

        # =========================================================================
        # Calculate PFJ loads for inverse dynamics method
        # =========================================================================

        # Set output path for ease of use
        data_folder = os.path.join('..', '..', 'simulations', dataset, participant, condition, 'inverse_dynamics')

        # Calculate PFJ loads
        # -------------------------------------------------------------------------

        # Read in joint moment and coordinate data
        coordinates = osim.TimeSeriesTable(os.path.join(data_folder, f'{participant}_{condition}_coordinates.sto'))
        moments = osim.TimeSeriesTable(os.path.join(data_folder, f'{participant}_{condition}_inverse_dynamics_results.sto'))

        # Trim the buffer 50 milliseconds of each side of the data
        coordinates.trim(coordinates.getIndependentColumn()[0] + 0.05,coordinates.getIndependentColumn()[-1] - 0.05)
        moments.trim(moments.getIndependentColumn()[0] + 0.05, moments.getIndependentColumn()[-1] - 0.05)

        # Extract necessary angles and moments
        # Set variables to extract
        extract_variables = ['hip_flexion_r', 'knee_angle_r', 'ankle_angle_r']
        # Extract time data
        # Note that separate time arrays are required as time step can slightly differ with inverse dynamics results
        angle_t = np.array(coordinates.getIndependentColumn())
        moment_t = np.array(moments.getIndependentColumn())
        # Set coefficient to invert moments if needed
        invert_moments = {'hip_flexion_r': 1, 'knee_angle_r': -1, 'ankle_angle_r': 1}
        # Extract angles
        angle_data = {var: np.rad2deg(coordinates.getDependentColumn(var).to_numpy()) for var in extract_variables}
        moment_data = {var: moments.getDependentColumn(f'{var}_moment').to_numpy() * invert_moments[var] for var in extract_variables}
        # Create functions to interpolate data to 101-points for consistency
        angle_f = {var: interp1d(angle_t, angle_data[var], kind='linear') for var in extract_variables}
        moment_f = {var: interp1d(moment_t, moment_data[var], kind='linear') for var in extract_variables}
        # Interpolate data with new 101-time array
        angle_new_t = np.linspace(angle_t.min(), angle_t.max(), 101)
        moment_new_t = np.linspace(moment_t.min(), moment_t.max(), 101)
        angle_norm = {var: angle_f[var](angle_new_t) for var in extract_variables}
        moment_norm = {var: moment_f[var](moment_new_t) for var in extract_variables}
        t_norm = np.linspace(moment_t.min(), moment_t.max(), 101)  # set inverse dynamics as normal time

        # Calculate k-coefficient for knee angle
        # Uses the method of van Eijden (1986) identified as most popular approach in Nunes et al. (2018)
        k = np.polyval(k_numerator_coeffs, angle_norm['knee_angle_r']) / \
            np.polyval(k_denominator_coeffs, angle_norm['knee_angle_r'])

        # Calculate quadriceps effective lever arm
        # Uses equation proposed in Nunes et al. (2018) as most commonly used
        q_lever = np.polyval(q_lever_coeffs, angle_norm['knee_angle_r'])

        # TODO: digitise Nemeth & Olsen data to create polynomials
        # # Test moment arm polynomials for Nemeth & Olsen data created by Eoin
        # # Note this theoretically outputs as moment arms in mm
        # h_flex = np.linspace(0, 90, 101)
        # h_men = ((-0.29 * h_flex) ** 2) - (4.3024 * h_flex) + 63.882
        # h_women = ((-0.2508 * h_flex) ** 2) - (3.4955 * h_flex) + 57.904
        # g_men = (-0.0668 * h_flex ** 2) - (1.5768 * h_flex) + 82.968
        # g_women = (-0.0677 * h_flex ** 2) - (1.3469 * h_flex) + 76.73
        # plt.plot(h_men, ls = '-', c = 'blue')
        # plt.plot(h_women, ls='--', c='blue')
        # # plt.plot(g_men, ls='-', c='red')
        # # plt.plot(g_women, ls='--', c='red')

        # # Calculate hamstring and gluteus force and lever arms
        # # This is the hip extension moment divided by hamstring lever arm as outlined in Doyle et al. (2023)
        # # Check gender as there are different calculations for males and females for lever arm
        # gender = participantInfo.loc[participantInfo['FileName'] == participant + 'static.c3d',]['Gender'].values[0]
        # # Calculate hamstrings and gluteus lever arm
        # # Note that similar scaling of coefficients has been applied to keep units consistent and appropriate
        # # Hip angle is inverted here to get positive moment arms
        # if gender == 'M':
        #     h_lever = (-0.29 * angle_norm['hip_flexion_r']*-1 ** 2) - (4.3024 * angle_norm['hip_flexion_r']*-1) + 63.882
        #     g_lever = (-0.0668e-03 * angle_norm['hip_flexion_r']*-1 ** 2) - (1.5768e-03 * angle_norm['hip_flexion_r']*-1) + 82.968e-03


        # Calculate quadriceps force
        # Remove negative force at knee extension moment portions
        # TODO: add hamstring and gastroc force?
        qf_id = moment_norm['knee_angle_r'] / q_lever
        qf_id[qf_id < 0] = 0

        # Store in dictionary
        qf['inverse_dynamics'][condition]['qf'] = qf_id
        qf['inverse_dynamics'][condition]['time'] = t_norm

        # Calculate PFJRF
        pfjrf_id = k * qf_id

        # Store in dictionary
        pfjrf['inverse_dynamics'][condition]['pfjrf'] = pfjrf_id
        pfjrf['inverse_dynamics'][condition]['time'] = t_norm

        # Calculate contact area based on knee flexion angle
        ca_mm2 = np.polyval(ca_mm2_coeffs, angle_norm['knee_angle_r'])
        ca_m2 = ca_mm2 / 1000000

        # Calculate PFJ stress in MPa (initial equation outputs this in Pa)
        pfjs_id = pfjrf_id / ca_m2 / 1000000

        # Store in dictionary
        pfjs['inverse_dynamics'][condition]['pfjs'] = pfjs_id
        pfjs['inverse_dynamics'][condition]['time'] = t_norm

        # Extract computation time
        # -------------------------------------------------------------------------

        # Load in computation time data
        with open(os.path.join(
                data_folder, f'{participant}_{condition}_inverse-dynamics_complex_computation-time.pkl'), 'rb') as pkl_file:
            id_computation = pickle.load(pkl_file)

        # Store in dictionary
        computation_time['inverse_dynamics'][condition][0] = id_computation['time_s']

        # =========================================================================
        # Calculate PFJ loads for static optimisation method
        # =========================================================================

        # Set output path for ease of use
        data_folder = os.path.join('..', '..', 'simulations', dataset, participant, condition, 'static_optimisation')

        # Calculate PFJ loads
        # -------------------------------------------------------------------------

        # Read in states, muscle force and moment arm data
        states = osim.TimeSeriesTable(os.path.join(
            data_folder, f'{participant}_{condition}_states.sto'))
        muscle_forces = osim.TimeSeriesTable(os.path.join(
            data_folder, f'{participant}_{condition}_StaticOptimization_Complex_force.sto'))
        # moment_arms = osim.TimeSeriesTable(os.path.join(
        #     data_folder, f'{participant}run{speed}_MuscleAnalysis_Complex_MomentArm_knee_angle_r.sto'))

        # Trim the buffer 50 milliseconds of each side of the data
        states.trim(states.getIndependentColumn()[0] + 0.05, states.getIndependentColumn()[-1] - 0.05)
        muscle_forces.trim(muscle_forces.getIndependentColumn()[0] + 0.05, muscle_forces.getIndependentColumn()[-1] - 0.05)

        # Extract muscle forces from static optimisation for quadriceps alongside timestamps
        # Timestamps are consistent across all static optimisation files
        so_muscle_force = {musc: muscle_forces.getDependentColumn(musc).to_numpy() for musc in complex_quads}
        so_time = np.array(muscle_forces.getIndependentColumn())

        # # Extract muscle moment arms
        # # Note that these get inverted to make them positive
        # so_moment_arms = {musc: moment_arms.getDependentColumn(musc).to_numpy() * -1 for musc in complex_quads}

        # Extract necessary angles from states data
        # Set variables to extract
        extract_variables = ['/jointset/hip_r/hip_flexion_r/value',
                             '/jointset/walker_knee_r/knee_angle_r/value',
                             '/jointset/ankle_r/ankle_angle_r/value']
        # Extract angles
        angle_data = {var.split('/')[3]: np.rad2deg(states.getDependentColumn(var).to_numpy()) for var in extract_variables}
        extract_variables = [var.split('/')[3] for var in extract_variables]
        # Create functions to interpolate data to 101-points for consistency
        angle_f = {var: interp1d(so_time, angle_data[var], kind='linear') for var in extract_variables}
        muscle_force_f = {musc: interp1d(so_time, so_muscle_force[musc], kind='linear') for musc in complex_quads}
        # moment_arm_f = {musc: interp1d(so_time, so_moment_arms[musc], kind='linear') for musc in complex_quads}
        # Interpolate data with new 101-time array
        t_norm = np.linspace(so_time.min(), so_time.max(), 101)
        angle_norm = {var: angle_f[var](t_norm) for var in extract_variables}
        so_muscle_force_norm = {musc: muscle_force_f[musc](t_norm) for musc in complex_quads}
        # so_moment_arms_norm = {musc: moment_arm_f[musc](t_norm) for musc in complex_quads}

        # Calculate k-coefficient for knee angle
        # Uses the method of van Eijden (1986) identified as most popular approach in Nunes et al. (2018)
        # Re-doing this here probably isn't necessary as knee angle is the same --- but why not...
        k = np.polyval(k_numerator_coeffs, angle_norm['knee_angle_r']) / \
            np.polyval(k_denominator_coeffs, angle_norm['knee_angle_r'])

        # Calculate total quadriceps force by summing values
        qf_so = np.vstack([so_muscle_force_norm[musc] for musc in complex_quads]).sum(axis=0)

        # Store in dictionary
        qf['static_optimisation'][condition]['qf'] = qf_so
        qf['static_optimisation'][condition]['time'] = t_norm

        # Calculate PFJRF
        pfjrf_so = k * qf_so

        # Store in dictionary
        pfjrf['static_optimisation'][condition]['pfjrf'] = pfjrf_so
        pfjrf['static_optimisation'][condition]['time'] = t_norm

        # Calculate contact area based on knee flexion angle
        ca_mm2 = np.polyval(ca_mm2_coeffs, angle_norm['knee_angle_r'])
        ca_m2 = ca_mm2 / 1000000

        # Calculate PFJ stress in MPa (initial equation outputs this in Pa)
        pfjs_so = pfjrf_so / ca_m2 / 1000000

        # Store in dictionary
        pfjs['static_optimisation'][condition]['pfjs'] = pfjs_so
        pfjs['static_optimisation'][condition]['time'] = t_norm

        # Extract computation time
        # -------------------------------------------------------------------------

        # Load in computation time data
        with open(os.path.join(
                data_folder, f'{participant}_{condition}_static-optimisation_complex_computation-time.pkl'), 'rb') as pkl_file:
            so_computation = pickle.load(pkl_file)

        # Store in dictionary
        computation_time['static_optimisation'][condition][0] = so_computation['time_s']

        # =========================================================================
        # Calculate PFJ loads for dynamic optimisation method
        # =========================================================================

        # Set output path for ease of use
        data_folder = os.path.join('..', '..', 'simulations', dataset, participant, condition, 'dynamic_optimisation')

        # Calculate PFJ loads
        # -------------------------------------------------------------------------

        # Read in states, muscle force and moment arm data
        states = osim.MocoTrajectory(os.path.join(
            data_folder, f'{participant}_{condition}_dynamic-optimisation_complex_solution.sto')).exportToStatesTable()
        muscle_forces = osim.TimeSeriesTable(os.path.join(
            data_folder, f'{participant}_{condition}_dynamic-optimisation_complex_muscle-forces.sto'))
        # moment_arms = osim.TimeSeriesTable(os.path.join(
        #     data_folder, f'{participant}run{speed}_MuscleAnalysis_Complex_MomentArm_knee_angle_r.sto'))

        # Trim the buffer 50 milliseconds of each side of the data
        states.trim(states.getIndependentColumn()[0] + 0.05, states.getIndependentColumn()[-1] - 0.05)
        muscle_forces.trim(muscle_forces.getIndependentColumn()[0] + 0.05, muscle_forces.getIndependentColumn()[-1] - 0.05)

        # Extract muscle forces from static optimisation for quadriceps alongside timestamps
        # Timestamps are consistent across all dynamic optimisation files
        do_muscle_force = {
            musc: muscle_forces.getDependentColumn(f'/forceset/{musc}|tendon_force').to_numpy() for musc in complex_quads}
        do_time = np.array(states.getIndependentColumn())

        # # Extract muscle moment arms
        # # Note that these get inverted to make them positive
        # so_moment_arms = {musc: moment_arms.getDependentColumn(musc).to_numpy() * -1 for musc in complex_quads}

        # Extract necessary angles from states data
        # Set variables to extract
        extract_variables = ['/jointset/hip_r/hip_flexion_r/value',
                             '/jointset/walker_knee_r/knee_angle_r/value',
                             '/jointset/ankle_r/ankle_angle_r/value']
        # Extract angles
        angle_data = {var.split('/')[3]: np.rad2deg(states.getDependentColumn(var).to_numpy()) for var in
                      extract_variables}
        extract_variables = [var.split('/')[3] for var in extract_variables]
        # Create functions to interpolate data to 101-points for consistency
        angle_f = {var: interp1d(do_time, angle_data[var], kind='linear') for var in extract_variables}
        muscle_force_f = {musc: interp1d(do_time, do_muscle_force[musc], kind='linear') for musc in complex_quads}
        # moment_arm_f = {musc: interp1d(so_time, so_moment_arms[musc], kind='linear') for musc in complex_quads}
        # Interpolate data with new 101-time array
        t_norm = np.linspace(do_time.min(), do_time.max(), 101)
        angle_norm = {var: angle_f[var](t_norm) for var in extract_variables}
        do_muscle_force_norm = {musc: muscle_force_f[musc](t_norm) for musc in complex_quads}

        # Calculate k-coefficient for knee angle
        # Uses the method of van Eijden (1986) identified as most popular approach in Nunes et al. (2018)
        # Re-doing this here probably isn't necessary as knee angle is the same --- but why not...
        k = np.polyval(k_numerator_coeffs, angle_norm['knee_angle_r']) / \
            np.polyval(k_denominator_coeffs, angle_norm['knee_angle_r'])

        # Calculate total quadriceps force by summing values
        qf_do = np.vstack([do_muscle_force_norm[musc] for musc in complex_quads]).sum(axis=0)

        # Store in dictionary
        qf['dynamic_optimisation'][condition]['qf'] = qf_do
        qf['dynamic_optimisation'][condition]['time'] = t_norm

        # Calculate PFJRF
        pfjrf_do = k * qf_do

        # Store in dictionary
        pfjrf['dynamic_optimisation'][condition]['pfjrf'] = pfjrf_do
        pfjrf['dynamic_optimisation'][condition]['time'] = t_norm

        # Calculate contact area based on knee flexion angle
        ca_mm2 = np.polyval(ca_mm2_coeffs, angle_norm['knee_angle_r'])
        ca_m2 = ca_mm2 / 1000000

        # Calculate PFJ stress in MPa (initial equation outputs this in Pa)
        pfjs_do = pfjrf_do / ca_m2 / 1000000

        # Store in dictionary
        pfjs['dynamic_optimisation'][condition]['pfjs'] = pfjs_do
        pfjs['dynamic_optimisation'][condition]['time'] = t_norm

        # Extract computation time
        # -------------------------------------------------------------------------

        # Load in computation time data
        with open(os.path.join(
                data_folder, f'{participant}_{condition}_dynamic-optimisation_computation-time.pkl'), 'rb') as pkl_file:
            do_computation = pickle.load(pkl_file)

        # Store in dictionary
        computation_time['dynamic_optimisation'][condition][0] = do_computation['time_s']

    # =========================================================================
    # Create plots of data for participant
    # =========================================================================

    # Set normalisation factor for setting data to body weights
    body_weight_factor = mass_kg * g

    # Loop through data to plot
    for plot_var in ['qf', 'pfjrf', 'pfjs']:

        # Grab dictionary to plot data from
        if plot_var == 'qf':
            plot_data = qf.copy()
        elif plot_var == 'pfjrf':
            plot_data = pfjrf.copy()
        elif plot_var == 'pfjs':
            plot_data = pfjs.copy()

        # Create figure and axes
        fig, plot_ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5), sharex=True, sharey=True)
        plt.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.125)

        # Plot on the singular axis
        # Add title
        plot_ax.set_title(f'Standard Run', fontsize=14, fontweight='bold')
        # Loop through methods
        for method in plot_col.keys():
            # Normalise quads force and pfjrf to body weight
            if plot_var == 'qf' or plot_var == 'pfjrf':
                plot_ax.plot(np.arange(0,101), plot_data[method][condition][plot_var] / body_weight_factor,
                             label=' '.join(method.split('_')).title(),
                             lw=1.5, ls='-', c=plot_col[method], zorder=3)
            # Otherwise just plot raw values
            else:
                plot_ax.plot(np.arange(0, 101), plot_data[method][condition][plot_var],
                             label=' '.join(method.split('_')).title(),
                             lw=1.5, ls='-', c=plot_col[method], zorder=3)

        # Set x-axis parameters
        plot_ax.set_xlim((0,100))
        plot_ax.set_xticks([0,25,50,75,100])

        # Set x-axis labels
        # All axes
        plot_ax.set_xlabel('0-100% Gait Cycle', fontsize=10, fontweight='bold', labelpad=10)

        # Set y-axis parameters
        plot_ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        if plot_var == 'qf':
            plot_ax.set_ylabel('Quadriceps Force (BW)', fontsize=10, fontweight='bold', labelpad=10)
        elif plot_var == 'pfjrf':
            plot_ax.set_ylabel('PFJRF (BW)', fontsize=10, fontweight='bold', labelpad=10)
        elif plot_var == 'pfjs':
            plot_ax.set_ylabel('PFJS (MPa)', fontsize=10, fontweight='bold', labelpad=10)

        # Add legend
        plot_ax.legend()

        # Add figure title
        plt.suptitle(f'Participant: {participant}', fontsize=14, fontweight='bold')

        # Save figure to participant folder
        fig.savefig(
            os.path.join('..', '..', 'outputs', dataset, participant, 'figures', f'{participant}_{plot_var}.png'),
            format = 'png', dpi = 150)

        # Close figure
        plt.close('all')

    # =========================================================================
    # Save calculated data to file
    # =========================================================================

    # Save separate dictionaries to file
    with open(os.path.join('..', '..', 'outputs', dataset, participant, 'results',
                           f'{participant}_pfjrf-calc.pkl'),
              'wb') as pkl_file:
        pickle.dump(pfjrf, pkl_file)
    with open(os.path.join('..', '..', 'outputs', dataset, participant, 'results',
                           f'{participant}_pfjs-calc.pkl'),
              'wb') as pkl_file:
        pickle.dump(pfjs, pkl_file)
    with open(os.path.join('..', '..', 'outputs', dataset, participant, 'results',
                           f'{participant}_qf-calc.pkl'),
              'wb') as pkl_file:
        pickle.dump(qf, pkl_file)
    with open(os.path.join('..', '..', 'outputs', dataset, participant, 'results',
                           f'{participant}_computation-time.pkl'),
              'wb') as pkl_file:
        pickle.dump(computation_time, pkl_file)

    # =========================================================================
    # Print confirmation
    # =========================================================================
    print(f'{"*" * 5} CALCULATED PFJ LOADS FOR {participant} {"*" * 5}')

# =========================================================================
# Run analysis
# =========================================================================

if __name__ == '__main__':

    # Calculate PFJ Loads
    # -------------------------------------------------------------------------
    for participant in participant_list:
        calculate_pfj_loads(participant)

    # Finalise and exit kernel
    # -------------------------------------------------------------------------

    # Doing this seems to avoid an error code when completing the script run
    os._exit(00)

# %% ---------- end of calc_PFJRF.py ---------- %% #

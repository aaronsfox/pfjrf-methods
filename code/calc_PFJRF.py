# -*- coding: utf-8 -*-
"""

@author:

    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au

    This script calculates and collates the different PFJRF approaches from
    each participant's simulated running trials.

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
import pickle

# =========================================================================
# Set-up
# =========================================================================

# General settings
# -------------------------------------------------------------------------

# Get list of participants with results in simulation directory
participant_list = [os.path.split(ii.path)[-1] for ii in os.scandir(os.path.join('..','simulations')) if ii.is_dir()]

# Read in participant info
participant_info = pd.read_csv(os.path.join('..', 'data', 'participantInfo.csv'))

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

# Set colouring for static optimisation vs. dynamic optimisation
optimisation_colours = {'static_optimisation': '#4885ed',  # blue
                        'dynamic_optimisation': '#ffa600',  # gold
                        }

# Set colouring for muscle moment arm comparison
# TODO: individual muscle moment arm equations seem off?
ma_colours = {'recfem_r': '#c871dd',  # purple
              'vasint_r': '#ff5ea6',  # pink
              'vasmed_r': '#ff755e',  # light orange
              'vaslat_r': '#ffa600',  # gold
              }

# Set line style for moment arm methods
ma_linestyle = {'kinematic': '-',
                'equation': '--'
                }

# Set plotting dictionary for PFJRF plot
pfjrf_plot_params = {
    'colour': {  # quadriceps force method
        'torque': '#000000',  # black
        'static': '#4885ed',  # blue
        'dynamic': '#ffa600',  # gold
        },
    'linestyle': {  # moment arm estimate method
        'equation': '--',
        'kinematic': ':',
        'none': '-'
    }
}

# OpenSim settings
# -------------------------------------------------------------------------

# Add the utility geometry path for model visualisation
osim.ModelVisualizer.addDirToGeometrySearchPaths(os.path.join(os.getcwd(), '..', 'model', 'Geometry'))

# Set the final mesh interval that was used for dynamic optimisations for file loading
mesh_int = 50

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
complex_quads = ['recfem_r', 'vasmed_r', 'vasint_r', 'vasmed_r']

# =========================================================================
# Define functions
# =========================================================================

# Calculate PFJRF using different approaches for participant
# -------------------------------------------------------------------------
def calculate_jrf(participant, speed, model_type):

    """
    TODO:
        > Is model type needed or can this function calculate all in one place?
    """

    # Create directories for outputs
    # -------------------------------------------------------------------------

    # Create directories
    os.makedirs(os.path.join('..', 'outputs', participant), exist_ok=True)
    os.makedirs(os.path.join('..', 'outputs', participant, 'figures'), exist_ok=True)
    os.makedirs(os.path.join('..', 'outputs', participant, 'results'), exist_ok=True)

    # Read in GRF and identify stance phase timing in trial
    # -------------------------------------------------------------------------

    # Read in GRF data to identify stance phase timings
    trial_grf = osim.TimeSeriesTable(os.path.join('..', 'data', participant,
                                                  f'{participant}run{speed}_grf.mot'))
    vgrf = trial_grf.getDependentColumn('ground_force_r_vy').to_numpy()
    grf_time = np.array(trial_grf.getIndependentColumn())

    # Identify right foot contacts based on rising edges above threshold of 20N
    force_above = vgrf > 20
    rising_edges = np.where((~force_above[:-1]) & (force_above[1:]))[0] + 1

    # Get timestamps of the start of stance phases
    stance_phase_times = grf_time[rising_edges]

    # Extract muscle moment arms using different approaches
    # -------------------------------------------------------------------------

    """
    This section uses the knee flexion angle data from the marker tracking simulation
    in calculating via the equation method given that this remains consistent across
    all other simulations. Similarly, the moment arms calculated from the muscle model
    in static optimisation are taken as the kinematic method as again these will be
    consistent across all simulations.    
    """

    # Equation based approach from knee flexion angle

    # Read in the data from the marker tracking solution
    marker_traj = osim.MocoTrajectory(
        os.path.join('..','simulations',participant,speed,'marker_tracking',
                     f'{participant}run{speed}_marker_tracking_solution.sto'))
    # Get the states as a table
    states = marker_traj.exportToStatesTable()
    # Get the torque data as a table
    torques = osim.TimeSeriesTable(
        os.path.join('..', 'simulations', participant, speed, 'marker_tracking',
                     f'{participant}run{speed}_marker_tracking_torques.sto'))
    # Extract the knee flexion angle and associated time data
    if model_type == 'complex':
        # Knee angle
        knee_angle = states.getDependentColumn('/jointset/walker_knee_r/knee_angle_r/value').to_numpy()
        knee_angle_t = np.array(states.getIndependentColumn())
        # Knee torque
        # Inverted knee flexion torque becomes positive
        knee_torque = torques.getDependentColumn('knee_angle_r_torque').to_numpy() * -1
        knee_torque_t = np.array(torques.getIndependentColumn())
    elif model_type == 'simple':
        print('TODO: extract knee angle from simple model...')

    # Identify the time and index where the stance phase starts in the marker tracking data
    states_stance_t = [
        val for val in stance_phase_times if \
        states.getIndependentColumn()[0] < val < states.getIndependentColumn()[-1]][0]
    states_stance_ind = states.getNearestRowIndexForTime(states_stance_t)

    # Identify the time and index where the stance phase starts in torque data (might be slightly different)
    torque_stance_t = [
        val for val in stance_phase_times if \
        torques.getIndependentColumn()[0] < val < torques.getIndependentColumn()[-1]][0]
    torque_stance_ind = torques.getNearestRowIndexForTime(torque_stance_t)

    # # Calculate muscle moment arms using equation method over stance phase
    # # See Kernozek et al. (2015) for equations
    # # Coefficients changed to negative? This seems to be the only way to make these make sense with the moment arm
    # # becoming smaller as the knee flexes?
    # # TODO: are they all negative coefficients though? It's probably a combination of positive and negative for the polynomial fit?
    # ma_eq = {}
    # ma_eq['recfem_r'] = 0.0519235 - (0.0064865 * knee_angle[states_stance_ind::])
    # ma_eq['vasmed_r'] = (0.0021733 * knee_angle[states_stance_ind::] ** 3) + \
    #                     (0.0089959 * knee_angle[states_stance_ind::] ** 2) + \
    #                     (0.0059805 * knee_angle[states_stance_ind::]) + 0.0434523
    # ma_eq['vasint_r'] = (0.0022705 * knee_angle[states_stance_ind::] ** 3) + \
    #                     (0.0097213 * knee_angle[states_stance_ind::] ** 2) + \
    #                     (0.0066606 * knee_angle[states_stance_ind::]) + 0.044273
    # ma_eq['vaslat_r']= (0.0033264 * knee_angle[states_stance_ind::] ** 3) + \
    #                    (0.0145048 * knee_angle[states_stance_ind::] ** 2) + \
    #                    (0.0138364 * knee_angle[states_stance_ind::]) + 0.0401728
    # ma_eq_avg = np.vstack([ma_eq[mm] for mm in ['recfem_r', 'vasmed_r', 'vasint_r', 'vaslat_r']]).mean(axis=0)
    # ma_eq_t = knee_angle_t[states_stance_ind::]

    # Calculate effective quadriceps lever arm based on knee angle
    # This is a simpler approach to the quadriceps moment arm calculation
    # It seems more accurate though for now give issues with above individual muscle calculations
    ma_eq_avg = (0.00000008 * np.rad2deg(knee_angle[states_stance_ind::]) ** 3) - \
                (0.000013 * np.rad2deg(knee_angle[states_stance_ind::]) ** 2) + \
                (0.00028 * np.rad2deg(knee_angle[states_stance_ind::])) + 0.046

    # Kinematic approach from muscle analysis

    # Read in muscle moment arms from static optimisation
    kinematic_moment_arms = osim.TimeSeriesTable(
        os.path.join('..', 'simulations', participant, speed, 'static_optimisation', model_type,
                     f'{participant}run{speed}_MuscleAnalysis_{model_type.title()}_MomentArm_knee_angle_r.sto'))

    # Identify the time and index where the stance phase starts in the moment arm data
    kinematic_stance_t = [
        val for val in stance_phase_times if \
        kinematic_moment_arms.getIndependentColumn()[0] < val < kinematic_moment_arms.getIndependentColumn()[-1]][0]
    kinematic_stance_ind = kinematic_moment_arms.getNearestRowIndexForTime(kinematic_stance_t)

    # Extract muscle moment arms
    if model_type == 'complex':
        # Moment arms are negative due to knee angle notation so are inverted here for positive values
        ma_kin = {}
        ma_kin['recfem_r'] = kinematic_moment_arms.getDependentColumn('recfem_r').to_numpy()[kinematic_stance_ind::] * -1
        ma_kin['vasmed_r'] = kinematic_moment_arms.getDependentColumn('vasmed_r').to_numpy()[kinematic_stance_ind::] * -1
        ma_kin['vasint_r'] = kinematic_moment_arms.getDependentColumn('vasint_r').to_numpy()[kinematic_stance_ind::] * -1
        ma_kin['vaslat_r'] = kinematic_moment_arms.getDependentColumn('vaslat_r').to_numpy()[kinematic_stance_ind::] * -1
        ma_kin_avg = np.vstack([ma_kin[mm] for mm in ['recfem_r', 'vasmed_r', 'vasint_r', 'vaslat_r']]).mean(axis=0)
        ma_kin_t = np.array(kinematic_moment_arms.getIndependentColumn()[kinematic_stance_ind::])
    elif model_type == 'simple':
        print('TODO: muscles from simple model...')

    # Create a plot to compare equation vs. muscle model kinematic moment arms

    # Convert time data from index to end of stance to a 0-100% notation
    eq_t = np.linspace(0, 100, len(ma_eq_avg))
    kin_t = np.linspace(0, 100, len(ma_kin_t))

    # # Create the figure for individual muscles
    # # TODO: sort out individual muscle calculations
    # fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 7), sharex=True, sharey=False)
    #
    # # Loop through the muscles and plot
    # for musc in ma_colours.keys():
    #     # Plot moment arms
    #     # Multiply by 100 to convert to centimetres
    #     plot_ax = ax.flatten()[list(ma_colours.keys()).index(musc)]
    #     plot_ax.plot(kin_t, ma_kin[musc] * 100, color=ma_colours[musc], lw=1.5, ls=ma_linestyle['kinematic'])
    #     plot_ax.plot(eq_t, ma_eq[musc] * 100, color=ma_colours[musc], lw=1.5, ls=ma_linestyle['equation'])
    #     # Set axis parameters
    #     plot_ax.set_xlim([0, 100])
    #     plot_ax.set_title(f'{musc.split("_")[0][:3].capitalize()}. {musc.split("_")[0][3:].capitalize()}.',
    #                       fontsize=12, fontweight='bold')
    #     plot_ax.set_ylabel('Moment Arm (cm)', fontsize=10, fontweight='bold')
    #     if list(ma_colours.keys()).index(musc) > 1:
    #         plot_ax.set_xlabel('0-100% Stance Phase', fontsize=10, fontweight='bold')
    #
    # # Set figure title
    # fig.suptitle(f'Muscle Forces for {participant} at {speed} Speed: Static (Blue) vs. Dynamic (Gold) Optimisation',
    #              fontsize=14, fontweight='bold')
    #
    # # Set layout
    # plt.tight_layout()

    # Create a plot to compare average equation vs. muscle model kinematic moment arms

    # Create the figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 7))

    # Plot moment arms
    # Multiply by 100 to convert to centimetres
    ax.plot(kin_t, ma_kin_avg * 100, color='black', lw=1.5, ls=ma_linestyle['kinematic'])
    ax.plot(eq_t, ma_eq_avg * 100, color='black', lw=1.5, ls=ma_linestyle['equation'])
    # Set axis parameters
    ax.set_xlim([0, 100])
    ax.set_title('Avg. Quadriceps Moment Arm',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Moment Arm (cm)', fontsize=10, fontweight='bold')
    ax.set_xlabel('0-100% Stance Phase', fontsize=10, fontweight='bold')

    # Set figure title
    fig.suptitle(f'{participant} at {speed} Speed: Kinematic (Solid) vs. Equation (Dashed) Methods',
                 fontsize=14, fontweight='bold')

    # Set layout
    plt.tight_layout()

    # Save figure
    fig.savefig(os.path.join('..', 'outputs', participant, 'figures',
                             f'{participant}_{speed}_quad-moment-arm_kinematic-v-equation.png'),
                format='png', dpi=300)

    # Close figure
    plt.close(fig)

    # Review muscle forces from different approaches
    # -------------------------------------------------------------------------

    """
    This section creates a visual to review the muscle forces from the static 
    versus dynamic optimisations. Data loaded in here will be used later to 
    calculate quadriceps force from these approaches.    
    """

    # Load muscle forces from static and dynamic optimisations
    static_opt_forces = osim.TimeSeriesTable(
        os.path.join('..', 'simulations', participant, speed, 'static_optimisation', model_type,
                     f'{participant}run{speed}_StaticOptimization_{model_type.title()}_force.sto'))
    dynamic_opt_forces = osim.TimeSeriesTable(
        os.path.join('..', 'simulations', participant, speed, 'dynamic_optimisation', model_type,
                     f'{participant}run{speed}_dynamic_optimisation_{model_type}_muscle_forces.sto'))

    # Identify the time and index where the stance phase starts in the optimisation data
    static_opt_stance_t = [
        val for val in stance_phase_times if \
        static_opt_forces.getIndependentColumn()[0] < val < static_opt_forces.getIndependentColumn()[-1]][0]
    dynamic_opt_stance_t = [
        val for val in stance_phase_times if \
        dynamic_opt_forces.getIndependentColumn()[0] < val < dynamic_opt_forces.getIndependentColumn()[-1]][0]
    static_opt_stance_ind = static_opt_forces.getNearestRowIndexForTime(static_opt_stance_t)
    dynamic_opt_stance_ind = dynamic_opt_forces.getNearestRowIndexForTime(dynamic_opt_stance_t)

    # Convert time data from index to end of stance to a 0-100% notation
    so_t = np.linspace(0, 100, len(static_opt_forces.getIndependentColumn()[static_opt_stance_ind::]))
    do_t = np.linspace(0, 100, len(dynamic_opt_forces.getIndependentColumn()[dynamic_opt_stance_ind::]))

    # Plot muscle force data over stance

    # Create the figure
    fig, ax = plt.subplots(nrows=6, ncols=4, figsize=(10, 10), sharex=True, sharey=False)

    # Loop through the muscle force groups
    for m_group in muscle_groups:
        # Sum the force data from the static and dynamic optimisations for the select muscles
        so_force = np.sum([
            static_opt_forces.getDependentColumn(mm).to_numpy()[static_opt_stance_ind::] for mm in muscle_groups[m_group]
        ], axis=0)
        do_force = np.sum([
            dynamic_opt_forces.getDependentColumn(
                f'/forceset/{mm}|tendon_force').to_numpy()[static_opt_stance_ind::] for mm in muscle_groups[m_group]
        ], axis=0)
        # Plot on appropriate axis
        plot_ax = ax.flatten()[list(muscle_groups.keys()).index(m_group)]
        plot_ax.plot(so_t, so_force, color=optimisation_colours['static_optimisation'], lw=1.5, ls='-')
        plot_ax.plot(do_t, do_force, color=optimisation_colours['dynamic_optimisation'], lw=1.5, ls='-')
        # Set axis parameters
        plot_ax.set_xlim([0,100])
        plot_ax.set_title(m_group, fontsize=12, fontweight='bold')
        plot_ax.set_ylabel('Force (N)', fontsize=10, fontweight='bold')
        if list(muscle_groups.keys()).index(m_group) > 19:
            plot_ax.set_xlabel('0-100% Stance Phase', fontsize=10, fontweight='bold')

    # Set figure title
    fig.suptitle(f'Muscle Forces for {participant} at {speed} Speed: Static (Blue) vs. Dynamic (Gold) Optimisation',
                 fontsize=14, fontweight='bold')

    # Set layout
    plt.tight_layout()

    # Remove any unused axes
    for plot_ax in ax.flatten():
        if not len(plot_ax.lines) > 0:
            plot_ax.axis('off')

    # Save figure
    fig.savefig(os.path.join('..', 'outputs', participant, 'figures',
                             f'{participant}_{speed}_muscle-forces_static-v-dynamic-opt.png'),
                format='png', dpi=300)

    # Close figure
    plt.close(fig)

    # =========================================================================
    # Calculate PFJRF with available data and different approaches
    # =========================================================================

    """
    Note there is likely a more efficient way to do this but for now this code pulls out the individual pieces
    and calculates the PFJRF.
    """

    # Setup dictionary to store calculations
    pfjrf_dict = {}

    # Add time stamps to dictionary
    pfjrf_dict['time'] = np.linspace(knee_angle_t[states_stance_ind::][0],
                                     knee_angle_t[states_stance_ind::][-1],
                                     101)

    # Calculate k-coefficient to use across methods
    # TODO: in degrees or radians for this calculation? I think it's degrees...
    interp_func = interp1d(knee_angle_t[states_stance_ind::], knee_angle[states_stance_ind::], kind='linear')
    angle_calc = np.rad2deg(interp_func(np.linspace(knee_angle_t[states_stance_ind::][0],
                                                    knee_angle_t[states_stance_ind::][-1],
                                                    101)))
    k = (4.62e-01 + (1.47e-03 * angle_calc) - (3.84e-05 * angle_calc**2)) / \
        (1 - (1.62e-02 * angle_calc) + (1.55e-04 * angle_calc**2) - (6.98e-07 * angle_calc**3))

    # Moment Arm = Equation (equation)
    # Quad Force = Torque / Moment Arm (torque)
    # MSK Model = None (none)
    # -------------------------------------------------------------------------

    # Extract the relevant data from stance and interpolate to 101-data points
    # Moment arm
    interp_func = interp1d(eq_t, ma_eq_avg, kind='linear')
    ma_calc = interp_func(np.linspace(0,100,101))
    # Quad force
    interp_func = interp1d(knee_torque_t[torque_stance_ind::], knee_torque[torque_stance_ind::], kind='linear')
    torque_calc = interp_func(np.linspace(knee_torque_t[torque_stance_ind::][0], knee_torque_t[torque_stance_ind::][-1],
                                          101))
    qf_calc = torque_calc / ma_calc
    # PFJRF
    # This includes a step to constrain any negative PFJRF value to zero as can occur with equation approaches
    pfjrf_calc = k * qf_calc
    pfjrf_calc[pfjrf_calc < 0] = 0
    # Add to dictionary
    pfjrf_dict['equation_torque_none'] = pfjrf_calc

    # Moment Arm = Kinematic Muscle Analysis (kinematic)
    # Quad Force = Torque / Moment Arm (torque)
    # MSK Model = Complex (complex)
    # -------------------------------------------------------------------------

    # Extract the relevant data from stance and interpolate to 101-data points
    # Moment arm
    interp_func = interp1d(kin_t, ma_kin_avg, kind='linear')
    ma_calc = interp_func(np.linspace(0, 100, 101))
    # Quad force
    interp_func = interp1d(knee_torque_t[torque_stance_ind::], knee_torque[torque_stance_ind::], kind='linear')
    torque_calc = interp_func(np.linspace(knee_torque_t[torque_stance_ind::][0], knee_torque_t[torque_stance_ind::][-1],
                                          101))
    qf_calc = torque_calc / ma_calc
    # PFJRF
    # This includes a step to constrain any negative PFJRF value to zero as can occur with equation approaches
    pfjrf_calc = k * qf_calc
    pfjrf_calc[pfjrf_calc < 0] = 0
    # Add to dictionary
    pfjrf_dict['kinematic_torque_complex'] = pfjrf_calc

    # Moment Arm = None (none)
    # Quad Force = Static Optimisation (static)
    # MSK Model = Complex (complex)
    # -------------------------------------------------------------------------

    # Quad force
    if model_type == 'complex':
        # Sum the force data from the static optimisation for the quadriceps muscles
        qf = np.sum([
            static_opt_forces.getDependentColumn(mm).to_numpy()[static_opt_stance_ind::] for mm in
            complex_quads], axis=0)
    interp_func = interp1d(np.array(static_opt_forces.getIndependentColumn())[static_opt_stance_ind::], qf,
                           kind='linear')
    qf_calc = interp_func(np.linspace(np.array(static_opt_forces.getIndependentColumn())[static_opt_stance_ind::][0],
                                      np.array(static_opt_forces.getIndependentColumn())[static_opt_stance_ind::][-1],
                                      101))
    # PFJRF
    # This includes a step to constrain any negative PFJRF value to zero as can occur with equation approaches
    pfjrf_calc = k * qf_calc
    pfjrf_calc[pfjrf_calc < 0] = 0
    # Add to dictionary
    pfjrf_dict['none_static_complex'] = pfjrf_calc

    # Moment Arm = None (none)
    # Quad Force = Dynamic Optimisation (dynamic)
    # MSK Model = Complex (complex)
    # -------------------------------------------------------------------------

    # Quad force
    if model_type == 'complex':
        # Sum the force data from the dynamic optimisation for the quadriceps muscles
        qf = np.sum([
            dynamic_opt_forces.getDependentColumn(f'/forceset/{mm}|tendon_force').to_numpy()[dynamic_opt_stance_ind::] for mm in
            complex_quads], axis=0)
    interp_func = interp1d(np.array(dynamic_opt_forces.getIndependentColumn())[dynamic_opt_stance_ind::], qf,
                           kind='linear')
    qf_calc = interp_func(np.linspace(np.array(dynamic_opt_forces.getIndependentColumn())[dynamic_opt_stance_ind::][0],
                                      np.array(dynamic_opt_forces.getIndependentColumn())[dynamic_opt_stance_ind::][-1],
                                      101))
    # PFJRF
    # This includes a step to constrain any negative PFJRF value to zero as can occur with equation approaches
    pfjrf_calc = k * qf_calc
    pfjrf_calc[pfjrf_calc < 0] = 0
    # Add to dictionary
    pfjrf_dict['none_dynamic_complex'] = pfjrf_calc

    # Save output dictionary
    # -------------------------------------------------------------------------
    with open(os.path.join('..', 'outputs', participant, 'results',
                           f'{participant}_{speed}_pfjrf-calc.pkl'), 'wb') as pkl_file:
        pickle.dump(pfjrf_dict, pkl_file)

    # Create figure comparing PFJRF calculations
    # -------------------------------------------------------------------------

    # Create the figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 7))

    # Loop through stored calculations
    pfjrf_methods = list(pfjrf_dict.keys())
    pfjrf_methods.remove('time')
    for method in pfjrf_methods:
        ax.plot(np.arange(0,101), pfjrf_dict[method],
                color=pfjrf_plot_params['colour'][method.split('_')[1]],
                lw=1.5,
                ls=pfjrf_plot_params['linestyle'][method.split('_')[0]])
    # Set axis parameters
    ax.set_xlim([0, 100])
    ax.set_title('Comparison of PFJRF Calculation Methods',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('PFJRF (N)', fontsize=12, fontweight='bold')
    ax.set_xlabel('0-100% Stance Phase', fontsize=12, fontweight='bold')

    # Set layout
    plt.tight_layout()

    # Save figure
    fig.savefig(os.path.join('..', 'outputs', participant, 'figures',
                             f'{participant}_{speed}_pfjrf-calc.png'),
                format='png', dpi=300)

    # Close figure
    plt.close(fig)

    # Print confirmation
    # -------------------------------------------------------------------------
    print(f'{"*" * 5} CALCULATED PFJRF FOR {participant} AT {speed} SPEED WITH {model_type.upper()} MODEL {"*" * 5}')

# =========================================================================
# Run calculations
# =========================================================================

if __name__ == '__main__':

    # TODO: currently only running for one speed and model

    # Calculate PFJRF
    # -------------------------------------------------------------------------
    for participant in participant_list:
        calculate_jrf(participant, 'T35', 'complex')

    # Exit terminal to avoid any funny business
    # -------------------------------------------------------------------------
    os._exit(00)

# %% ---------- end of calc_PFJRF.py ---------- %% #
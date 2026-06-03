import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.formula.api as smf  # Library for Statistical Modeling

# --- Configuration ---
script_dir = Path(__file__).resolve().parent
# Define paths relative to the script location
base_dir = script_dir.parent.parent / 'outputs' / 'Fukuchi2017'
info_path = script_dir.parent.parent / 'data' / 'Fukuchi2017' / 'participantInfo.csv'
output_excel = base_dir / 'All_Participants_PFJ_Results.xlsx'

# Set Pandas display options for better console visibility
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def load_pickle(file_path):
    """Helper function to open and load binary pickle files."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


# --- 1. Load Participant Metadata ---
try:
    # Load physical characteristics (Mass) required for body weight normalization
    info_df = pd.read_csv(info_path)
    print(f"--- Loaded info for {len(info_df)} subjects ---")
except Exception as e:
    print(f"Error loading {info_path}: {e}")
    exit()

all_extracted_data = []

# --- 2. Dynamic Folder Scanning ---
# Automatically identify all folders starting with 'RBDS' (e.g., RBDS001, RBDS03, RBDS10)
participant_folders = sorted([f for f in os.listdir(base_dir) if f.startswith('RBDS') and os.path.isdir(base_dir / f)])
print(f"Found {len(participant_folders)} participant folders.")

for p_id in participant_folders:
    results_path = base_dir / p_id / 'results'
    if not results_path.exists():
        continue

    try:
        # Extract subject number from string to match the CSV 'Subject' column
        sub_num = int(''.join(filter(str.isdigit, p_id)))
        # Retrieve mass and calculate Body Weight (BW) in Newtons
        mass = info_df.loc[info_df['Subject'] == sub_num, 'Mass'].iloc[0]
        bw_newtons = mass * 9.81

        # Load calculated data dictionaries from the results folder
        pfjrf_dict = load_pickle(results_path / f'{p_id}_pfjrf-calc.pkl')
        pfjs_dict = load_pickle(results_path / f'{p_id}_pfjs-calc.pkl')
        qf_dict = load_pickle(results_path / f'{p_id}_qf-calc.pkl')

        TIME_KEY, PFJRF_KEY, PFJS_KEY, QF_KEY = 'time', 'pfjrf', 'pfjs', 'qf'

        # Nested loops to process every Method (e.g., SO, ID) and Speed (e.g., T25, T35)
        for method in pfjrf_dict.keys():
            for speed in pfjrf_dict[method].keys():
                try:
                    time = np.array(pfjrf_dict[method][speed][TIME_KEY])
                    pfjrf = np.array(pfjrf_dict[method][speed][PFJRF_KEY])
                    pfjs = np.array(pfjs_dict[method][speed][PFJS_KEY])
                    qf = np.array(qf_dict[method][speed][QF_KEY])

                    # Calculate Peak values and Cumulative loads (Integrals using Trapezoidal rule)
                    # Variable names use underscores to remain compatible with LMM formulas
                    all_extracted_data.append({
                        'participant': p_id,
                        'method': method,
                        'speed': speed,
                        'peak_PFJS_MPa': np.max(pfjs),
                        'cum_PFJS_MPas': np.trapz(pfjs, x=time),
                        'peak_PFJRF_BW': np.max(pfjrf) / bw_newtons,
                        'cum_PFJRF_BWs': np.trapz(pfjrf, x=time) / bw_newtons,
                        'peak_QuadF_BW': np.max(qf) / bw_newtons,
                        'cum_QuadF_BWs': np.trapz(qf, x=time) / bw_newtons
                    })
                except KeyError:
                    pass
    except Exception as e:
        print(f"  Error processing {p_id}: {e}")

# --- 3. Save to Excel ---
# Compile results into a master DataFrame and export to Excel
df_final = pd.DataFrame(all_extracted_data)
if not df_final.empty:
    df_final.to_excel(output_excel, index=False)
    print(f"SUCCESS: Master results saved to {output_excel}")
else:
    print("No data collected.");
    exit()

# --- 4. Linear Mixed Models (LMM) Analysis ---
print("\n" + "=" * 80)
print("PART 4: LINEAR MIXED MODELS (LMM) STATISTICAL ANALYSIS")
print("=" * 80)


def report_effect_size(model_result, df, metric):
    """Calculates and prints R-squared and Cohen's f2 effect size."""
    try:
        # Variance of the residuals
        var_resid = model_result.scale
        # Variance of the random effect (Participant)
        var_random = float(model_result.cov_re.iloc[0, 0])
        # Variance of the fixed effects (Method, Speed, Interaction)
        fixed_pred = model_result.predict(df)
        var_fixed = fixed_pred.var()

        # Marginal R2 (Fixed effects only)
        r2_m = var_fixed / (var_fixed + var_random + var_resid)
        # Conditional R2 (Fixed + Random effects)
        r2_c = (var_fixed + var_random) / (var_fixed + var_random + var_resid)

        # Cohen's f2 for the fixed effects
        f2 = r2_m / (1 - r2_c)

        size = "Large" if f2 > 0.35 else "Medium" if f2 > 0.15 else "Small" if f2 > 0.02 else "Negligible"

        print(f"--- Effect Size for {metric} ---")
        print(f"Marginal R2 (Fixed Effects): {r2_m:.3f}")
        print(f"Conditional R2 (Total Model): {r2_c:.3f}")
        print(f"Cohen's f2: {f2:.3f} ({size} effect)")
    except Exception as e:
        print(f"Effect size calculation failed: {e}")


# List of biomechanical variables to be tested
metrics_to_test = [
    'peak_PFJRF_BW', 'cum_PFJRF_BWs',
    'peak_PFJS_MPa', 'cum_PFJS_MPas',
    'peak_QuadF_BW', 'cum_QuadF_BWs'
]

for metric in metrics_to_test:
    print(f"\n>>> Statistical Model for: {metric}")
    try:
        # Fit the model
        model = smf.mixedlm(f"{metric} ~ C(method) * C(speed)", df_final, groups=df_final["participant"])
        result = model.fit()

        # 1. Print standard summary
        print(result.summary())

        # 2. Print Effect Size (New part)
        report_effect_size(result, df_final, metric)

        # 3. Quick trend interpretation
        p_values = result.pvalues
        interaction_p = p_values[p_values.index.str.contains(':')].min()
        print(f"\nInterpretation: {'Significant Interaction' if interaction_p < 0.05 else 'Consistent Trends'}")
        print("-" * 50)

    except Exception as e:
        print(f"LMM Failed for {metric}: {e}")

print("\nAll statistical analyses completed.")

# --- 5. Optimized Time-Series Visualization (SD Shading & Legend Positioning) ---
print("\n" + "=" * 80)
print("PART 5: FINAL VISUALIZATION (MEAN ± SD)")
print("=" * 80)

import matplotlib.pyplot as plt

# 1. Setup constants and storage
common_time = np.linspace(0, 100, 101)
speeds = ['T25', 'T35', 'T45']
speed_labels = ['Speed: 2.5 m·s⁻¹', 'Speed: 3.5 m·s⁻¹', 'Speed: 4.5 m·s⁻¹']
methods = ['inverse_dynamics', 'static_optimisation', 'dynamic_optimisation']
method_labels = ['Inverse Dynamics', 'Static Optimisation', 'Dynamic Optimisation']

# Use a high-contrast color palette for better distinction
method_colors = {
    'inverse_dynamics': '#000000',  # Black
    'static_optimisation': '#0072B2',  # Deep Blue
    'dynamic_optimisation': '#D55E00'  # Orange-Red
}

# 2. Re-process data (Ensuring full_curve_data is ready for plotting)
full_curve_data = {m: {s: {'rf': [], 'js': [], 'qf': []} for s in speeds} for m in methods}

for p_id in participant_folders:
    try:
        results_path = base_dir / p_id / 'results'
        sub_num = int(''.join(filter(str.isdigit, p_id)))
        mass = info_df.loc[info_df['Subject'] == sub_num, 'Mass'].iloc[0]
        bw = mass * 9.81

        rf_dict = load_pickle(results_path / f'{p_id}_pfjrf-calc.pkl')
        js_dict = load_pickle(results_path / f'{p_id}_pfjs-calc.pkl')
        qf_dict = load_pickle(results_path / f'{p_id}_qf-calc.pkl')

        for m in methods:
            for s in speeds:
                if m in rf_dict and s in rf_dict[m]:
                    t = np.array(rf_dict[m][s]['time'])
                    # Normalize time to 0-100% of the gait cycle
                    norm_t = (t - t[0]) / (t[-1] - t[0]) * 100

                    # Store interpolated values for each metric
                    full_curve_data[m][s]['rf'].append(
                        np.interp(common_time, norm_t, np.array(rf_dict[m][s]['pfjrf']) / bw))
                    full_curve_data[m][s]['js'].append(np.interp(common_time, norm_t, np.array(js_dict[m][s]['pfjs'])))
                    full_curve_data[m][s]['qf'].append(
                        np.interp(common_time, norm_t, np.array(qf_dict[m][s]['qf']) / bw))
    except Exception as e:
        # Silently skip missing or corrupted files during batch processing
        continue


# 3. Define the Plotting Function
def generate_final_plot(metric_key, y_label, file_name):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    # Calculate global max to unify Y-axis and create space for the legend in the top-left
    all_values = []
    for m in methods:
        for s in speeds:
            if full_curve_data[m][s][metric_key]:
                all_values.append(np.max(np.mean(full_curve_data[m][s][metric_key], axis=0)))
    max_y = max(all_values) if all_values else 10

    for i, speed in enumerate(speeds):
        ax = axes[i]
        ax.set_title(speed_labels[i], fontsize=14, fontweight='bold', pad=15)

        for method in methods:
            data = np.array(full_curve_data[method][speed][metric_key])
            if data.size == 0: continue

            # Compute statistical Mean and Standard Deviation (SD)
            mean_curve = np.mean(data, axis=0)
            std_curve = np.std(data, axis=0)
            color = method_colors[method]

            # Plot the Mean curve
            ax.plot(common_time, mean_curve, label=method_labels[methods.index(method)], color=color, lw=2.5)
            # Plot the SD shaded area (Alpha increased to 0.2 for better visibility)
            ax.fill_between(common_time, mean_curve - std_curve, mean_curve + std_curve,
                            color=color, alpha=0.2, edgecolor=color, linewidth=0.5)

        # Layout and Axis optimization
        ax.set_xlabel('0-100% Gait Cycle', fontsize=12)
        if i == 0: ax.set_ylabel(y_label, fontsize=13, fontweight='bold')

        ax.set_xlim(0, 100)
        # Critical adjustment: Set Y-limit higher (1.35x) to prevent legend overlap with peaks
        ax.set_ylim(0, max_y * 1.35)

        # Stylistic cleanup for academic publication
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle=':', alpha=0.6)

        # Position legend in the top-left corner without a frame
        ax.legend(loc='upper left', frameon=False, fontsize=10)

    plt.tight_layout()
    # Save as high-resolution PNG for publication
    save_path = base_dir / f"{file_name}.png"
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")
    plt.show()


# 4. Execute Plotting for each biomechanical metric
generate_final_plot('rf', 'PFJRF (BW)', 'Final_PFJRF_Comparison')
generate_final_plot('js', 'PFJS (MPa)', 'Final_PFJS_Comparison')
generate_final_plot('qf', 'Quadriceps Force (BW)', 'Final_QF_Comparison')

print("\nAll Part 5 analysis and visualization completed successfully.")
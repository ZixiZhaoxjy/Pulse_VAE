import numpy as np
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

def mean_absolute_percentage_error(y_true, y_pred):
    return np.nanmean(np.abs((y_true - y_pred) / y_true)) * 100, np.std(np.abs((y_true - y_pred) / y_true)) * 100

def split_SOC_data(condition, Fts, train_SOC_values):

    # Assuming SOC is the first column
    train_mask = np.isin(condition[:, 0], train_SOC_values)
    # print("train_mask",train_mask)
    test_mask = ~train_mask
    # print("test_mask",test_mask)
    train_Fts = Fts[train_mask]
    train_condition = condition[train_mask]
    test_Fts = Fts[train_mask + test_mask]
    test_condition = condition[train_mask + test_mask]
    test_condition2 = condition[test_mask]

    return train_Fts, train_condition, test_Fts, test_condition, test_condition2, test_mask


def plot_soh_scatter(y_true, y_pred, title):
    plt.figure(figsize=(12, 10))
    plt.scatter(y_true, y_pred, alpha=0.5, s=1000, color='purple')
    plt.xlabel('True SOH')
    plt.ylabel('Predicted SOH')
    plt.title(title)

    # Line for perfect predictions
    ymax = max(max(y_true), max(y_pred))
    ymin = min(min(y_true), min(y_pred))
    delta = 0.01

    # dashed line
    plt.plot([ymin-delta, ymax+delta], [ymin-delta, ymax+delta], '--', lw=8, color='gray')

    #plt.grid(True)
    # Set aspect of the axis to be equal
    plt.gca().set_aspect('equal', adjustable='box')

    # Format the tick labels to two decimal places
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    # save the figure with tilte in file name
    plt.savefig('SOH_prediction'+title+'.jpeg', dpi=600, bbox_inches='tight')
    #plt.show()


def visualize_experiment_results(train_SOC_cases, all_SOC_values, mape_results_phase1, mape_results_phase2):
    """
    Visualizes the MAPE performance across different SOC levels for both phases on the same plot.

    :param train_SOC_cases: List of lists, where each inner list contains the SOC levels used for training in each case.
    :param all_SOC_values: List of all SOC values in the dataset.
    :param mape_results_phase1: List of MAPE results for phase 1 for each case.
    :param mape_results_phase2: List of MAPE results for phase 2 for each case.
    """
    # Number of cases
    num_cases = len(train_SOC_cases)

    # Setup plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Common X-axis labels
    x_labels = [str(soc) for soc in all_SOC_values]

    # Using a qualitative colormap for distinct case representation
    cmap = plt.cm.get_cmap('Set2', min(num_cases, 9))  # Set1 has 9 distinct colors

    for case_index in range(num_cases):
        # Determine test SOC indices for the current case
        test_SOC_indices = [soc for soc in all_SOC_values if soc not in train_SOC_cases[case_index]]

        for phase_index, mape_results in enumerate([mape_results_phase1, mape_results_phase2]):
            # Initialize arrays to store MAPE for each SOC level
            mape_values = np.full(len(all_SOC_values), np.nan)

            # Update MAPE values for SOC levels used in testing
            for i, soc in enumerate(all_SOC_values):
                if soc in test_SOC_indices:
                    mape_values[i] = mape_results[case_index][test_SOC_indices.index(soc)]

            # Plot MAPE for each SOC level with different markers
            # Star for Phase 2, Circle for Phase 1
            marker = '*' if phase_index == 1 else 'o'
            ax.scatter(x_labels, mape_values, marker=marker, s=1000, alpha=1, color=cmap(case_index % 9))

        # Highlight training SOC levels
        #for train_soc in train_SOC_cases[case_index]:
        #   ax.axvline(x=str(train_soc), color='grey', linestyle='--', alpha=0.3)

    # Adding labels and title
    ax.set_xlabel('Testing SOC(%)')
    ax.set_ylabel('MAPE(%)')

    # Create a colorbar
    norm = mcolors.Normalize(vmin=0, vmax=num_cases - 1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, ticks=np.arange(0, min(num_cases, 9)), boundaries=np.arange(-0.5, min(num_cases, 9) + 0.5, 1))
    cbar.set_label('Case Index')
    cbar.ax.set_yticklabels(range(min(num_cases, 9)))

    ax.grid(True)
    plt.tight_layout()
    #plt.savefig('SOH_prediction_case_results'+'.jpeg', dpi=600, bbox_inches='tight')
    #plt.show()

# --- Load Data ---
def load_data(hyperparams):
    if (hyperparams['battery'] == "NMC2.1"):
        data = pd.read_excel('battery_data/NMC_2.1Ah_W_3000.xlsx', sheet_name="Sheet1")
    if (hyperparams['battery'] == "LFP"):
        data = pd.read_excel('battery_data/LFP_35Ah_W_3000.xlsx', sheet_name="SOC ALL")
    if (hyperparams['battery'] == "LMO"):
        data = pd.read_excel('battery_data/LMO_10Ah_W_3000.xlsx', sheet_name="SOC ALL")
    if (hyperparams['battery'] == "NMC21"):
        data = pd.read_excel('battery_data/NMC_21Ah_W_3000.xlsx', sheet_name="SOC ALL")


    data['SOC'] /= 100  # Normalize SOC by dividing by 100
    # Filter data to include only rows where SOC is less than or equal to 50% (0.5 after normalization)
    filtered_data = data[data['SOC'] <= 0.5]
    print("Data shape after filtering:", filtered_data.shape)

    features = filtered_data.loc[:, 'U1':'U21'].values
    soc = filtered_data['SOC'].values
    soh = filtered_data['SOH'].values

    print("Features shape:", features.shape)
    print("SOC shape:", soc.shape)
    print("SOH shape:", soh.shape)

    # Combining SOC and SOH as conditional input
    condition = np.column_stack((soc, soh))
    return features, condition, filtered_data
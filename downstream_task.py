
from sklearn.ensemble import RandomForestRegressor
import matplotlib.colors as mcolors
from Attention_VAE import augment_data, attention_vae
from utils import *

def run_SOH_experiment(masked_augmented_Fts_dir, masked_augmented_SOH_dir, case_index , hyperparams,train_SOC_values, test_SOC_index, data, augmented_data, masked_augmented_Fts, test_condition):
    # train_SOC_values is a list of SOC levels used for training
    # test_SOC_index is the SOC level used for testing
    # Filtering training data for the specified list of SOC levels
    train_data = data[data['SOC'].isin(train_SOC_values)]
    test_data = data[data['SOC'] == test_SOC_index]

    X_train = train_data.loc[:, 'U1':'U21'].values
    y_train = train_data['SOH'].values
    X_test = test_data.loc[:, 'U1':'U21'].values
    y_test = test_data['SOH'].values

    # Generate augmented data for the test_SOC_index
    data = augmented_data[test_condition[:, 0] == test_SOC_index]
    print("test_SOC_index:",test_SOC_index)

    if (case_index > hyperparams['watershed']):
        X_augmented = np.array(masked_augmented_Fts_dir[test_SOC_index])  # Assuming features are the first 21 columns
        SOH_augmented = np.array(masked_augmented_SOH_dir[test_SOC_index])
    else:
        X_augmented = data[:, 0:21]  # Assuming features are the first 21 columns
        SOH_augmented = data[:, -1]


    # Phase 1: Train Model on Available Data for Training SOC Levels
    model_phase1 = RandomForestRegressor(n_estimators=20,max_depth=64,bootstrap=False).fit(X_train, y_train)
    y_pred_phase1 = model_phase1.predict(X_test)
    mape_phase1, std_phase1 = mean_absolute_percentage_error(y_test, y_pred_phase1)
    #plot_soh_scatter(y_test, y_pred_phase1, title=f"Phase 1: Testing SOC={test_SOC_index}")

    # Phase 2: Train Model on Generated Data for Selected Testing SOC
    model_phase2 = RandomForestRegressor(n_estimators=20,max_depth=64,bootstrap=False).fit(X_augmented, SOH_augmented)
    y_pred_phase2 = model_phase2.predict(X_test)
    mape_phase2, std_phase2 = mean_absolute_percentage_error(y_test, y_pred_phase2)
    #plot_soh_scatter(y_test, y_pred_phase2, title=f"Phase 2: Testing SOC={test_SOC_index}")

    return mape_phase1, std_phase1, mape_phase2, std_phase2

def run_SOH_experiments(masked_augmented_Fts_dir, masked_augmented_SOH_dir, case_index , hyperparams,train_SOC_values, all_SOC_values,data, augmented_data, masked_augmented_Fts, test_condition):
    mape_results_phase1 = []
    mape_results_phase2 = []
    std_results_phase1 = []
    std_results_phase2 = []
    test_SOC_indices = [soc for soc in all_SOC_values if soc not in train_SOC_values]

    for test_SOC_index in test_SOC_indices:

        mape_phase1, std_phase1, mape_phase2, std_phase2 = run_SOH_experiment(
            masked_augmented_Fts_dir, masked_augmented_SOH_dir,
            case_index, hyperparams,
            train_SOC_values, test_SOC_index, data, augmented_data, masked_augmented_Fts, test_condition
        )
        mape_results_phase1.append(mape_phase1)
        mape_results_phase2.append(mape_phase2)
        std_results_phase1.append(std_phase1)
        std_results_phase2.append(std_phase2)

    # Plotting the bar plot with error bars
    '''
    x = np.arange(len(test_SOC_indices))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 10))
    rects1 = ax.bar(x - width/2, mape_results_phase1, width, yerr=std_results_phase1, label='No Generation')
    rects2 = ax.bar(x + width/2, mape_results_phase2, width, yerr=std_results_phase2, label='With Generation')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('MAPE(%)')
    ax.set_xlabel('SOC(%)')
    ax.set_xticks(x)
    ax.set_xticklabels(test_SOC_indices)
    ax.legend()

    # change the digits on bar label to 2
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom')
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom')
    #plt.show()
    '''

    return mape_results_phase1, mape_results_phase2, std_results_phase1, std_results_phase2

def preprocess(case_index, hyperparams, test_condition, test_Fts, train_SOC_values, train_SOC_values_cases, augmented_Fts):
    soc_mape_avg = []
    soc_mape_std = []
    x_ticks = [soc * 100 for soc in hyperparams['all_SOC_values']]  # All SOC values

    repeated_test_Fts = np.repeat(test_Fts, hyperparams['sampling_multiplier'], axis=0)
    repeated_test_condition = np.repeat(test_condition[:, 0:2], hyperparams['sampling_multiplier'], axis=0)

    # Loop over testing SOC values and features
    # Exclude the training SOC values from the all SOC values
    test_SOCs = [soc for soc in hyperparams['all_SOC_values'] if soc not in train_SOC_values]
    test_result = []
    rate = 0
    masked_augmented_Fts_dir = dict.fromkeys(test_SOCs)
    masked_augmented_SOH_dir = dict.fromkeys(test_SOCs)

    for test_SOC in test_SOCs:

        rate = rate + 1
        print(f"Testing SOC: {test_SOC}")
        print(test_SOC)
        feature_mape = []

        # Create a boolean mask for the specific SOC value
        mask1_soc = train_SOC_values[1] if (len(train_SOC_values_cases)) != 1 else (train_SOC_values[0])

        soc_mask1 = repeated_test_condition[:, 0] == mask1_soc

        soc_mask2 = repeated_test_condition[:, 0] == test_SOC
        masked_repeated_test_Fts = repeated_test_Fts[soc_mask2]  # Apply the mask to repeated_test_Fts
        # print(len(soc_mask1),mask1_soc)

        masked_augmented_Fts = augmented_Fts[soc_mask1]  # Apply the mask to augmented_Fts\
        masked_augmented_SOH = repeated_test_condition[soc_mask2, 1]  # Apply the mask to augmented SOH
        ###############################################################################################################################


        test_result.append(masked_augmented_Fts)
        n = len(masked_augmented_Fts)
        m = len(masked_augmented_Fts[0])
        # print(n,m)
        k = 0
        if (hyperparams['battery'] == "NMC"):
            physic_weight = 0.043
        if (hyperparams['battery'] == "LFP"):
            physic_weight = 0.0295
        if (hyperparams['battery'] == "LMO"):
            physic_weight = 0.095

        while (k < len(masked_augmented_Fts)):
            l = 0
            while (l < len(masked_augmented_Fts[0])):

                if (case_index > hyperparams['watershed']):
                    masked_augmented_Fts[k][l] = masked_augmented_Fts[k][l] + ((test_SOC - mask1_soc) * physic_weight * 20)
                else:
                    break
                # 0.042为相邻两个soc，对应的U之差的 均值，20为比例系数

                l = l + 1
            k = k + 1
        feature_id = 0
        masked_augmented_Fts_dir[test_SOC] = masked_augmented_Fts
        masked_augmented_SOH_dir[test_SOC] = masked_augmented_SOH


    return masked_augmented_Fts, masked_augmented_SOH, masked_augmented_Fts_dir, masked_augmented_SOH_dir

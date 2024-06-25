# Rapid and Sustainable Pretreatment for Retired Battery Recycling Amid Random Conditions Using Generative Machine Learning
Rapid and accurate pretreatment for state of health (SOH) estimation in retired batteries is crucial for recycling sustainability. Data-driven approaches, while innovative in SOH estimation, require exhaustive data collection and are sensitive to retirement conditions. Here we show that the generative machine learning strategy can alleviate such a challenge, validated through a unique dataset of 2720 retired lithium-ion battery samples, covering 3 cathode material types, 3 physical formats, 4 capacity designs, and 4 historical usages. With generated data, a simple regressor realizes an accurate pretreatment, with mean absolute percentage errors below 6%, even under unseen retirement conditions.

# 1. Setup
## Enviroments
* Python (Jupyter notebook) 
## Python requirements
* python=3.10
* numpy=1.26.4
* tensorflow=2.15.0
* keras=2.15.0
* matplotlib=3.9.0
* scipy=1.13.1
* scikit-learn=1.5.0
* pandas=2.2.2

# 2. Datasets
We physically tested 270 retired lithium-ion batteries, covering 3 cathode types, 4 historical usages, 3 physical formats, and 4 capacity designs. See more details on [Pulse-Voltage-Response-Generation](https://github.com/terencetaothucb/Pulse-Voltage-Response-Generation).
## Battery Types
|Cathode Material|Nominal Capacity (Ah)|Physical Format|Historical Usage|Quantity|
|:--|:--|:--|:--|:--|
|NMC|2.1|Cylinder|Lab Accelerated Aging|67 (from 12 physical batteries)|
|LMO|10|Pouch|HEV1|95|
|NMC|21|Pouch|BEV1|52|
|LFP|35|Square Aluminum Shell|HEV2|56|

# 3. Run
## Experiment setting
Python file "configuration" contains all the hyperparameters.Change these parameters to choose battery type, model size and testing conditions.
```python
hyperparams = {
    'battery': 'NMC',
    'file_path': 'battery_data/CNM_2.1Ah_W_3000.xlsx',
    'sampling_multiplier': 1,
    'feature_dim': 21,  # Dimension of the main input features
    'condition_dim': 2,  # Dimension of the conditional input (SOC + SOH)
    'embedding_dim': 64,
    'intermediate_dim': 64,
    'latent_dim': 2,
    'batch_size': 32,
    'epochs': 50,
    'num_heads': 1,
    'train_SOC_values': [0.05, 0.15, 0.25, 0.35, 0.45, 0.50],  # SOC values to use for training
    'all_SOC_values': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],  # All SOC values in the dataset
    'mode': 3,  # when case > 3, interpolation ends; set mode to 99 for only interpolation, to -1 for only extrapolation
}
```
# 4. Experiment Details


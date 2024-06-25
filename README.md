# Rapid and Sustainable Pretreatment for Retired Battery Recycling Amid Random Conditions Using Generative Machine Learning
Rapid and accurate pretreatment for state of health (SOH) estimation in retired batteries is crucial for recycling sustainability. Data-driven approaches, while innovative in SOH estimation, require exhaustive data collection and are sensitive to retirement conditions. Here we show that the generative machine learning strategy can alleviate such a challenge, validated through a unique dataset of 2720 retired lithium-ion battery samples, covering 3 cathode material types, 3 physical formats, 4 capacity designs, and 4 historical usages. With generated data, a simple regressor realizes an accurate pretreatment, with mean absolute percentage errors below 6%, even under unseen retirement conditions.

# Setup
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

# Datasets
We physically tested 270 retired lithium-ion batteries, covering 3 cathode types, 4 historical usages, 3 physical formats, and 4 capacity designs. See more details on [Pulse-Voltage-Response-Generation](https://github.com/terencetaothucb/Pulse-Voltage-Response-Generation).
## Battery Types
|Cathode Material|Nominal Capacity (Ah)|Physical Format|Historical Usage|Quantity|
|:--|:--|:--|:--|:--|
|NMC|2.1|Cylinder|Lab Accelerated Aging|67 (from 12 physical batteries)|
|LMO|10|Pouch|HEV1|95|
|NMC|21|Pouch|BEV1|52|
|LFP|35|Square Aluminum Shell|HEV2|56|

# Run

# Experiment Details


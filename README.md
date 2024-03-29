
# HBV Hydrological Model Simulation

This project contains a Python implementation of the HBV (Hydrologiska Byråns Vattenbalansavdelning) hydrological model. The model is used for simulating snow accumulation and melt, evapotranspiration, soil moisture, and discharge.

## Description

The HBV model takes meteorological data as input and simulates several hydrological processes. The core of this implementation lies in the `HBVModel` class, which processes temperature and precipitation data to simulate snow accumulation and melt, soil moisture dynamics, evapotranspiration, and streamflow generation.

## Installation

To run this script, you need Python installed on your system. The script has been tested with Python 3.8. Additionally, you need to install the required packages. You can install them using the following command:

```bash
pip install -r requirements.txt
```

## Usage

To run the model, execute the script `Poseidon_paralell.py`. The script performs a Monte Carlo simulation to find optimal parameters for the model. The number of simulations and the R2 score threshold for saving results are configurable by the user.

```bash
python Poseidon_paralell.py
```

Follow the prompts to enter the number of simulations (`num_sim`) and the R2 score threshold (`R2_threshold`).

## Poseidon_optim.py - HBV Model Optimization

### Description

The script `Poseidon_optim.py` extends the functionality of the HBV hydrological model by introducing an optimization layer. It includes the `HydroModelOptimizer` class, which leverages different optimization algorithms (BFGS, GAP) to fine-tune the model parameters. Users can choose between various objective functions (R2, MSE) and apply regularization techniques (L1, L2) to control overfitting.

### Usage

To run the optimization script:

1. Choose the optimization algorithm (BFGS or GAP) and the objective function (R2 or MSE).
2. Optionally, specify a regularization term (L1 or L2) and its corresponding lambda value.
3. Execute `Poseidon_optim.py` and follow the on-screen prompts to set the optimization parameters.

The script will output the optimized parameters and generate relevant plots showing the model's performance metrics across iterations.

### Dependencies

Ensure all dependencies listed in `requirements.txt` are installed. Update the file if `Poseidon_optim.py` requires additional packages.

### Output

The script outputs the optimized model parameters and visualizations of the optimization process, including soil moisture dynamics, snow accumulation, and evapotranspiration rates.

## Author

- Jouval Somer - email: jmsomer@uio.no

## References

This Python implementation of the HBV hydrological model is inspired by and based on concepts presented in the following academic papers:

1. AghaKouchak, A., & Habib, E. (2010). Application of a Conceptual Hydrologic Model in Teaching Hydrologic Processes. International Journal of Engineering Education, 26(4), 963-973.

2. AghaKouchak, A., et al. (2013). An Educational Model for Ensemble Streamflow Simulation and Uncertainty Analysis. Hydrology and Earth System Sciences, 17, 445-452.

The first version of the Excel-based HBV model, which this project references, was developed by Prof. Andras Bardossy. The Excel model and its demonstration are available via the following resources:

- YouTube video demonstration: [Excel HBV Model Demonstration](https://www.youtube.com/watch?v=SYqKCu8lAVM)
- Direct download link for the Excel model: [Model Spreadsheet](https://amir.eng.uci.edu/downloads/Model_Spreadsheet_Final.xlsx)

Special thanks to the authors of these papers, the creator of the YouTube video, and Prof. Andras Bardossy for their valuable contributions to understanding and simulating hydrological processes.

This Excel model was then further developed by incorporating some more fetures from 3.

3. Seibert, J. (2005). HBV light Version 2: User’s Manual. Stockholm University, Department of Physical Geography and Quaternary Geology.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from skopt import gp_minimize
from Poseidon_paralell import HBVModel


class HydroModelOptimizer:
    def __init__(self, model_class, mesure, parameter_bounds, regularization=None, lambda_=0.01, input_data=None):
        self.model_class = model_class
        self.parameter_bounds = parameter_bounds
        self.regularization = regularization
        self.lambda_ = lambda_
        self.input_data = input_data
        self.monthly_tave = [-4.66101, -4.18881, -0.55231, 4.113548, 10.97489, 14.11158, 16.83401, 15.11313, 10.44769, 6.41085, 1.062845, -3.07723]
        self.daily_pem = [0.05, 0.14, 0.46, 1.5, 3.01, 4.15, 3.66, 2.72, 1.42, 0.43, 0.03, 0.0]
        self.objective_log = []  # List to store objective values
        self.iteration_count = 0  # Initialize an iteration counter
        self.mesure = mesure


    def objective_function(self, params):        
        parameters_dict = {key: value for key, value in zip(self.parameter_bounds.keys(), params)}
        model = self.model_class(parameters_dict, self.monthly_tave, self.daily_pem, self.input_data)
        model.run_model()
        R2, MSE = model.evaluate()
        
        if self.mesure == 'R2':
            objective = 1 - R2
            self.objective_log.append(1 - objective)
        elif self.mesure == 'MSE':
            objective = MSE
            self.objective_log.append(objective)

        if self.regularization is not None:
            regularization_term = self.lambda_ * np.sum(np.array(params)**2) if self.regularization == 'L2' else self.lambda_ * np.sum(np.abs(params))
            objective_w_reg = objective + regularization_term
        
        objective_w_reg = objective

        # Increment the iteration counter
        self.iteration_count += 1

        if self.iteration_count % 100 == 0:
            if self.mesure == 'R2':
                print(f"Iteration {self.iteration_count}: {self.mesure} = {1 - objective:4f}")
            elif self.mesure == 'MSE':
                print(f"Iteration {self.iteration_count}: {self.mesure} = {objective:4f}")

        return objective_w_reg


    def print_last_objective_value(self):
        print(f'\nThe last value of the objective function is: {self.objective_log[-1]:4f}')


    def plot_objective_log(self):
        plt.figure(figsize=(14, 7))
        plt.plot(self.objective_log, linewidth=1.5)
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Objective Function Value', fontsize=14)
        plt.title(f'Objective Function Value over Iterations', fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.grid(True)
        plt.savefig(f'Objective_Function_Value.png')
        plt.show()


    def optimize_with_BFGS(self):
        initial_guess = [np.mean(bounds) for bounds in self.parameter_bounds.values()]
        # result = minimize(self.objective_function, initial_guess, method='L-BFGS-B', bounds=list(self.parameter_bounds.values()))
        result = minimize(self.objective_function, initial_guess, method='L-BFGS-B',
                  bounds=list(self.parameter_bounds.values()), tol=1e-8,
                  options={'ftol': 1e-11, 'gtol': 1e-7})
        final_params = result.x
        return final_params


    def optimize_with_GAP(self):
        result = differential_evolution(self.objective_function, bounds=list(self.parameter_bounds.values()), maxiter=40)
        final_params = result.x
        return final_params
    

    def optimize_with_Bayesian(self):
        result = gp_minimize(self.objective_function, dimensions=list(self.parameter_bounds.values()), n_calls=50, random_state=0)
        final_params = result.x
        return final_params


def main(optim, mesure, reg, lamb):
    # Load data
    df = pd.read_csv('Data/ptq.txt', delim_whitespace=True, parse_dates=['date'])
    df.rename(columns={'date': 'Date', 'Prec.': 'Preci. (mm)', 'Temp': 'Temp. (C)', 'Qobs': 'Q Observations'}, inplace=True)
    df['Month ID'] = df['Date'].dt.month

    # Monthly Average Temperatures and Daily Potential Evapotranspiration
    monthly_tave = [-4.66101,-4.18881, -0.55231, 4.113548, 10.97489, 14.11158, 16.83401, 15.11313, 10.44769,6.41085, 1.062845, -3.07723] 
    daily_pem = [0.05, 0.14, 0.46, 1.5, 3.01, 4.15, 3.66, 2.72, 1.42, 0.43, 0.03, 0.0] 

    parameters_bounds_dict  = {'TT': (0.6, 1.2), 'CFMAX': (2, 5), 'SFCF': (0.6, 0.6), 'CFR': (0.05, 0.05), 'CWH': (0.06, 0.06), 'FC': (200, 400), 'LP': (0.5, 0.9), 'BETA': (2, 5), 'Kperc': (0.1, 0.3), 'UZL': (20, 80), 'K0': (0.02, 0.5), 'K1': (0.02, 0.2), 'K2': (0.02, 0.1), 'MAXBAS': (1, 5), 'Cet': (0.1, 0.1)}

    optimizer = HydroModelOptimizer(HBVModel, mesure, parameters_bounds_dict, regularization=reg, lambda_=lamb, input_data=df)

    if optim == 'BFGS':
        final_params = optimizer.optimize_with_BFGS()
    elif optim == 'GAP':
        final_params = optimizer.optimize_with_GAP()

    optimizer.print_last_objective_value()

    formatted_params = [f"{param:.6f}" for param in final_params]
    optimized_params = {}
    print('\n\nThe optimized parameters are:')
    for key, param_value in zip(parameters_bounds_dict.keys(), formatted_params):
        print(key,': ', param_value)
        optimized_params[key] = param_value

    hbv_model = HBVModel(optimized_params, monthly_tave, daily_pem, df)

    # Plotting
    hbv_model.plot_variable(variable_name='Soil Moisture', title='Soil Moisture', ylabel='Soil Moisture [mm]')
    hbv_model.plot_variable(variable_name='Snow (mm)', title='Snow Pack', ylabel='Snow Pack [mm]')
    hbv_model.plot_variable(variable_name='Ea (mm/day)', title='Evapotranspiration', ylabel='Evapotranspiration [mm/day]')
    hbv_model.plot_daily_average_discharge('Q Simulations')

    # Plot the objective log
    optimizer.plot_objective_log()


if __name__ == "__main__":
    optim = ['BFGS', 'GAP']
    mesure = ['R2', 'MSE']
    reg = None
    lamb = 0.0001

    print('Select the optimization algorithm by entering either 0 or 1:\n 0 : BFGS (quasi-Newton method),\n 1 : GAP (differential evolution method)')
    algorithm = int(input('Algorithm = '))

    print('\nSelect the objective function by entering either 0 or 1:\n 0 : R2,\n 1 : MSE')
    objective = int(input('Objective function = '))

    print('\nWould you like a regularization term in the objective function?')
    reg_y_n = input('Proceed (y/n)? ')
    if reg_y_n == 'y':
        reg = input('\nEnter L1 or L2 for regularization type: ')
        print('\nAlso, please enter the value for lambda.')
        lamb = float(input('Lambda = '))

    print('\nStarting optimization:')

    main(optim[algorithm], mesure[objective], reg, lamb)
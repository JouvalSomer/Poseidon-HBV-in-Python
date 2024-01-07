import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


class HBVModel:
    """
    Python adaptation of the HBV hydrological model for simulating snow accumulation and melt, 
    evapotranspiration, soil moisture, and discharge.

    Attributes:
        parameters (dict): Model parameters.
        monthly_tave (list): Monthly average temperatures.
        daily_pem (list): Daily potential evapotranspiration.
        df (DataFrame): Data frame containing weather and flow data.
    """
        
    def __init__(self, parameters, monthly_tave, daily_pem, df):
        self.parameters = parameters
        self.monthly_tave = monthly_tave
        self.daily_pem = daily_pem
        self.df = df

        self.Ndays = df.shape[0]

        self.snow = np.zeros(self.Ndays)
        self.melted_snow = np.zeros(self.Ndays)
        self.refreezing = np.zeros(self.Ndays)
        self.liquid_water = np.zeros(self.Ndays)
        self.soil_moisture = np.zeros(self.Ndays)
        self.recharge = np.zeros(self.Ndays)
        self.pea = np.zeros(self.Ndays)
        self.ea = np.zeros(self.Ndays)
        self.s1 = np.zeros(self.Ndays)
        self.s2 = np.zeros(self.Ndays)
        self.q_simulations = np.zeros(self.Ndays)
        self.q_gw = np.zeros(self.Ndays)
        
        # Setting initial conditions 
        self.melt = 0
        self.snow[0] = 25.0
        self.soil_moisture[0] = 100.0
        self.s1[0] = 2.0
        self.s2[0] = 200.0

    def run_model(self):
        """
        Executes the HBV hydrological model simulation.
        
        The method processes temperature and precipitation data to simulate various hydrological processes 
        such as snow accumulation and melt, evapotranspiration, soil moisture dynamics, and streamflow generation.
        """      
        temperatures = self.df['Temp. (C)'].values
        precipitations = self.df['Preci. (mm)'].values
        month_ids = self.df['Month ID'].values - 1  # Adjust for zero-indexed arrays

        TT = self.parameters['TT']
        CFMAX = self.parameters['CFMAX']
        CWH = self.parameters['CWH']
        SFCF = self.parameters['SFCF']
        CFR = self.parameters['CFR']
        Cet = self.parameters['Cet']
        FC = self.parameters['FC']
        LP = self.parameters['LP']
        BETA = self.parameters['BETA']
        K0 = self.parameters['K0']
        K1 = self.parameters['K1']
        K2 = self.parameters['K2']
        UZL = self.parameters['UZL']
        Kperc = self.parameters['Kperc']
        MAXBAS = self.parameters['MAXBAS']

        # Calculate the ci's for the routing routine
        ci = np.zeros(int(MAXBAS))
        for i in range(1, int(MAXBAS) + 1):
            u = np.linspace(i-1, i, num=1000)
            integrand = (2 / MAXBAS) - np.abs(u - (MAXBAS / 2)) * (4 / MAXBAS**2)
            ci[i-1] = np.trapz(integrand, u)
        ci /= np.sum(ci)

        self.monthly_tave = np.array(self.monthly_tave)
        self.daily_pem = np.array(self.daily_pem)

        base_temp_factors = 1 + Cet * (temperatures - self.monthly_tave[month_ids])
        base_pem_factors = self.daily_pem[month_ids]

        for t in range(1, self.Ndays):
            temp = temperatures[t]
            preci = precipitations[t]
            S_M_priv = self.soil_moisture[t-1]


            # Snow, snow melt, refreeze and liquid water = precipitation + snow melt
            if temp > TT:
                self.melt = min(self.snow[t-1], CFMAX * (temp - TT))

                self.snow[t] = self.snow[t-1] - self.melt

                cold_water_holding = self.snow[t-1] * CWH
                if self.melt < cold_water_holding:
                    Liquid_water = preci
                else:
                    Liquid_water = preci + self.melt

            else: 
                self.snow[t] = self.snow[t-1] + SFCF * preci
                Liquid_water = 0

                if self.melt > 0:
                    potential_refreeze = CFR * CFMAX * (TT - temp)
                    self.snow[t] += min(self.melt, potential_refreeze)

            # Recharge
            recharge = Liquid_water * (S_M_priv / FC) ** BETA

            # E_pot
            E_pot = max(min(base_temp_factors[t] * base_pem_factors[t], 2 * base_pem_factors[t]), 0)

            # E_act
            E_act = E_pot * min((S_M_priv / (FC * LP)), 1)
            self.ea[t] = E_act

            # Soil Moisture
            self.soil_moisture[t] = S_M_priv + Liquid_water - recharge - E_act

            # S1
            S1_priv = self.s1[t-1]
            S1 = S1_priv + recharge - max(0, S1_priv - UZL) * K0 - (S1_priv * K1) - (S1_priv * Kperc)
            self.s1[t] = S1

            # S2
            S2_priv = self.s2[t-1]
            S2 = S2_priv + S1_priv * Kperc - S2_priv * K2
            self.s2[t] = S2

            # Q_tot
            Q_tot = max(0, S1 - UZL) * K0 + S1 * K1 + S2 * K2
            self.q_gw[t] = Q_tot

            Q_sim_t = 0
            for i in range(1, int(MAXBAS) + 1):
                Q_sim_t += ci[i-1] * self.q_gw[t - i + 1]
            self.q_simulations[t] = Q_sim_t


        self.df['Q Simulations'] =  self.q_simulations
        self.df['Soil Moisture'] =  self.soil_moisture
        self.df['Snow (mm)'] =  self.snow
        self.df['Ea (mm/day)'] =  self.ea
    

    def evaluate(self, start_date='1982-01-01'):
        filtered_df = self.df[self.df['Date'] >= pd.to_datetime(start_date)]
        
        R2 = r2_score(filtered_df['Q Observations'], filtered_df['Q Simulations'])
        MSE = mean_squared_error(filtered_df['Q Observations'], filtered_df['Q Simulations'])
    
        return R2, MSE
    

    def plot_daily_average_discharge(self, sim, start_date='1982-01-01'):
        df_copy = self.df[self.df['Date'] >= pd.to_datetime(start_date)].copy()
        df_copy['Normalized Date'] = df_copy['Date'].apply(lambda x: x.replace(year=2000))
        daily_avg = df_copy.groupby('Normalized Date').mean()

        plt.figure(figsize=(14, 7))
        if sim in daily_avg:
            plt.plot(daily_avg.index, daily_avg[sim], linewidth=1.5, label='Qsim')
        if 'Q Observations' in daily_avg:
            plt.plot(daily_avg.index, daily_avg['Q Observations'], linewidth=1.5, label='Qobs')

        plt.xlabel('Day of the Year', fontsize=14)
        plt.ylabel('Average Discharge [mm/day]', fontsize=14)
        plt.title('Daily Average Discharge Over 10 Years', fontsize=20)
        plt.legend()

        # Set x-axis to display months
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.grid(True)
        plt.savefig('average_discharge.png')
        plt.show()


    def plot_variable(self, variable_name, title, ylabel, start_date = '1982-01-01'):
        self.df = self.df[self.df['Date'] >= pd.to_datetime(start_date)]

        plt.figure(figsize=(14, 7))
        plt.plot(self.df['Date'], self.df[variable_name], linewidth=1.5, label=title)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.title(f'{title} Over Time', fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.savefig(f'{title}.png')
        plt.show()


def run_simulation(params_tuple):
    i, parameters_bounds_dict, monthly_tave, daily_pem, df, R2_threshold = params_tuple
    param_values = [np.random.uniform(low, high) for low, high in parameters_bounds_dict.values()]
    parameters = dict(zip(parameters_bounds_dict.keys(), param_values))
    hbv_model = HBVModel(parameters, monthly_tave, daily_pem, df)
    hbv_model.run_model()
    R2, MSE = hbv_model.evaluate()
    return i, R2 if R2 >= R2_threshold else None, param_values


def main(num_iter, R2_threshold):
    # Load data
    df = pd.read_csv('Data/ptq.txt', delim_whitespace=True, parse_dates=['date'])
    df.rename(columns={'date': 'Date', 'Prec.': 'Preci. (mm)', 'Temp': 'Temp. (C)', 'Qobs': 'Q Observations'}, inplace=True)
    df['Month ID'] = df['Date'].dt.month

    # Monthly Average Temperatures and Daily Potential Evapotranspiration
    monthly_tave = [-4.66101,-4.18881, -0.55231, 4.113548, 10.97489, 14.11158, 16.83401, 15.11313, 10.44769,6.41085, 1.062845, -3.07723] 
    daily_pem = [0.05, 0.14, 0.46, 1.5, 3.01, 4.15, 3.66, 2.72, 1.42, 0.43, 0.03, 0.0] 

    # Parameters and Bounds Dictionary
    parameters_bounds_dict  = {'TT': (0.6, 1.2), 'CFMAX': (2, 5), 'SFCF': (0.6, 0.6), 'CFR': (0.05, 0.05),
                                'CWH': (0.06, 0.06), 'FC': (200, 400), 'LP': (0.5, 0.9), 'BETA': (2, 5),
                                'Kperc': (0.1, 0.3), 'UZL': (20, 80), 'K0': (0.02, 0.5), 'K1': (0.02, 0.2),
                                'K2': (0.02, 0.1), 'MAXBAS': (1, 5), 'Cet': (0.1, 0.1)}

    # R2 Scores and Parameters Arrays Initialization
    R2_arr = np.full(num_iter, -5.0)
    params_arr = np.zeros((len(parameters_bounds_dict), num_iter))

    # Simulation Preparation
    print('\nPreparing the parameters for the simulation...')
    params_tuples = [(i, parameters_bounds_dict, monthly_tave, daily_pem, df.copy(), R2_threshold) for i in range(num_iter)]

    # Simulation Execution
    # Use ProcessPoolExecutor to run simulations in parallel
    print('Starting simulation:')
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_simulation, params_tuple) for params_tuple in params_tuples]
        for future in tqdm(as_completed(futures), total=num_iter):
            i, R2, param_values = future.result()
            if R2 is not None:
                R2_arr[i] = R2
                params_arr[:, i] = param_values 

        # Post-Simulation Analysis
    max_R2_index = np.argmax(R2_arr)
    max_R2_value = R2_arr[max_R2_index]
    optimal_parameters = params_arr[:, max_R2_index]
    optimal_parameters_dict = dict(zip(parameters_bounds_dict.keys(), optimal_parameters))

    # Model Initialization with Optimal Parameters
    hbv_model = HBVModel(optimal_parameters_dict, monthly_tave, daily_pem, df)
    hbv_model.run_model()

    # Output Results
    print("Highest R2 Value:", max_R2_value)
    print("Optimal Parameters:", optimal_parameters_dict)
    
    # Plotting
    hbv_model.plot_variable(variable_name='Soil Moisture', title='Soil Moisture', ylabel='Soil Moisture [mm]')
    hbv_model.plot_variable(variable_name='Snow (mm)', title='Snow Pack', ylabel='Snow Pack [mm]')
    hbv_model.plot_variable(variable_name='Ea (mm/day)', title='Evapotranspiration', ylabel='Evapotranspiration [mm/day]')
    hbv_model.plot_daily_average_discharge('Q Simulations')

    # Saving Parameters
    with open('optimal_parameters.txt', 'w') as file:
        file.write(str(optimal_parameters_dict))

    # Scatter Plot
    fig, axes = plt.subplots(5, 3, figsize=(18, 24))
    axes = axes.flatten()
    for i, param_key in enumerate(parameters_bounds_dict.keys()):
        ax = axes[i]
        ax.scatter(params_arr[i, :], R2_arr, alpha=0.5, s=20)
        ax.set_title(f'{param_key} vs R2', fontsize=14)
        ax.set_xlabel(param_key, fontsize=12)
        ax.set_ylabel('R2', fontsize=12)
        ax.grid(True)
        param_bounds = parameters_bounds_dict[param_key]
        ax.set_xlim([param_bounds[0] - param_bounds[0]/10, param_bounds[1] + param_bounds[1]/10])
        ax.set_ylim([max_R2_value - max_R2_value/10, max_R2_value + max_R2_value/10])
    plt.tight_layout(pad=2.0)
    plt.savefig('param_scatter.png')
    plt.show()



if __name__ == "__main__":
    print('How many Monte Carlo simulation do you want to run?')
    num_sim = int(input('num_sim = '))

    print('\nAt what R2 score threshold would you like to save the results? (float between 0 and 1)')
    R2_threshold = float(input('R2_threshold = '))

    main(num_sim, R2_threshold)
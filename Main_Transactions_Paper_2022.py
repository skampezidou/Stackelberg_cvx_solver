import visualization
import pandas as pd
import custom_fun
from sklearn import linear_model
import numpy as np
import time
from datetime import datetime
from pytz import timezone

tz = timezone('EST')
print(datetime.now(tz))

#visualization.plot_bars_co2()

#READ BARPLOT vs NUMBER OF SAMPLES
# df_bars_samples = pd.read_excel('Data_May_2022/'+'ga_gi_samples_all.xlsx', index_col=0)
# print(df_bars_samples)
# visualization.plot_bars_samples(df_bars_samples)
# exit(8)

#READ BARPLOT vs NUMBER OR PROSUMERS
# df_bars_pros = pd.read_excel('Data_May_2022/'+'ga_gi_samples100_N_10_30_50_71_AVG.xlsx', index_col=None)
# visualization.plot_bars_prosumers(df_bars_pros)
# exit(8)

# General parameterization
T = 24
a = 2
Q_perc = 0.1 #0 or 0.01 or 0.1
ui_divider = 6 #!!! use 6 only when a=2 and Qi=0.1, otherwise use 1

min_u = 0.001#0.01#0.00001#0.0003
max_u = 0.002#0.02#0.00009#0.0004
number_samples = 100#80 #samples per day for training
start = time.time()

# Read data
df = pd.read_excel('Data_May_2022/'+'May_2022_Data_fixed5.xlsx')
nom_loads_names = df.columns.tolist()
nom_loads_names.remove('Start Date Time')
nom_loads_names.remove('End Date Time')
nom_loads_names.remove('Price ($/KW)')
nom_loads_names.remove('Total Demand (KW)')

# Generate ui
#all_ui = np.random.uniform(min_u, max_u, len(nom_loads_names))
##all_ui = np.reshape(all_ui, (1, all_ui.shape[0]))
##df = pd.DataFrame(all_ui, columns = nom_loads_names)
##df.to_csv('Data_May_2022/ui.csv', index=False)

df_ui = pd.read_csv('Data_May_2022/'+'ui_fixed5.csv')  #ui
df_ui_columns = df_ui.columns.tolist()
all_ui = df_ui.to_numpy()
all_ui = np.reshape(all_ui, (all_ui.shape[1], ))


divide_by = np.ones((all_ui.shape[0],))*ui_divider
all_ui = np.divide(all_ui, divide_by)

#visualization.plot_all_alpha_demands(df, nom_loads_names, T)


coeff_all_days = []
intercept_all_days = []
all_pmax = []
p_random_all = []
temp_gi = []
temp_ga = []
Qi_all = []
p_random_all_days = []
d_star_from_p_random_all_days = []
print('----Sampling----')
for i in range(0, len(df), T): #for each day
    lambda_prices = df['Price ($/KW)'][i:i+T].values.tolist()
    pmax = max(np.abs(lambda_prices)) #max(lambda_prices)
    all_pmax.append(pmax)
    nom_loads = df[nom_loads_names][i:i+T].values

    training_p_random_for_i_day = []
    training_d_star_for_i_day = []
    p_random_all = []
    for s in range(0, number_samples):
        p_random = np.random.uniform(0, pmax, T)
        x_j_star_for_i_day = []
        for j in range(0, len(nom_loads_names)): #for each prosumer
            Qi = sum(nom_loads[:,j])*Q_perc
            Qi_all.append(Qi)
            #print('prosumer :', j, ' with column :', nom_loads_names[j])
            [obj_j, x_j_star] = custom_fun.pros_opt_RLS(nom_loads[:,j], p_random, a, all_ui[j], Qi, T)
            temp_gi.append(obj_j)
            x_j_star_for_i_day.append(x_j_star)

        d_star_for_i_day = []
        x_j_star_for_i_day_array = np.array(x_j_star_for_i_day)
        x_j_star_for_i_day_array = x_j_star_for_i_day_array.transpose()
        for k in range(0, T):
            d_star_for_i_day.append(sum(x_j_star_for_i_day_array[k,:]))

        p_random_all.append(p_random.tolist())
        training_d_star_for_i_day.append(d_star_for_i_day)

    p_random_all_arr = np.array(p_random_all)
    training_d_star_for_i_day_arr = np.array(training_d_star_for_i_day)

    #----
    p_random_all_days.append(p_random_all)
    d_star_from_p_random_all_days.append(training_d_star_for_i_day)
    #----

    reg = linear_model.LinearRegression(fit_intercept=True).fit(p_random_all_arr, training_d_star_for_i_day_arr)
    coeff_all_days.append(reg.coef_)
    intercept_all_days.append(reg.intercept_)


# Solve game for each day
all_p_star = []
all_ga_tilde = []
all_ga_tilde_fun = []
all_ga_pes_dstar = []
all_xi_star = []
all_gi = []
day = 0
#print('the ui are:')
#print(all_ui)
print('------------RESULTS-------------')
print(datetime.now(tz))
for i in range(0, len(df), T): #for each day
    print('Day: ', day)
    nom_loads = df[nom_loads_names][i:i + T].values
    lambda_prices = df['Price ($/KW)'][i:i + T].values.tolist()
    pmax = max(np.abs(lambda_prices)) #max(lambda_prices)
    #print(lambda_prices)
    #print('Pmax: ', pmax)

    res = custom_fun.solve_agg_from_least_squares(lambda_prices, pmax, coeff_all_days[day], intercept_all_days[day], nom_loads, T)
    all_ga_tilde.append(np.inner(lambda_prices - res.x, np.sum(nom_loads, axis=1) - np.inner(res.x, coeff_all_days[day]) - intercept_all_days[day]))
    all_ga_tilde_fun.append(res.fun)
    all_p_star.append(res.x)

    all_xi_star_day_k = []
    all_gi_day_k = []
    for j in range(0, len(nom_loads_names)):  # for each prosumer
        #print('prosumer: ', j)
        Qi = sum(nom_loads[:, j]) * Q_perc
        [obj_i, xi_star] = custom_fun.pros_opt_RLS(nom_loads[:, j], res.x, a, all_ui[j], Qi, T)
        all_gi_day_k.append(obj_i)
        all_xi_star_day_k.append(xi_star)

    all_gi.append(all_gi_day_k)
    all_xi_star.append(all_xi_star_day_k)

    # For plotting
    d_star_day_k = np.sum(all_xi_star_day_k, axis=0)
    all_ga_pes_dstar.append(np.inner(lambda_prices - res.x, np.sum(nom_loads, axis=1) - d_star_day_k))

    # move to next day
    day = day + 1

end = time.time()
print(end - start, " seconds")
print(datetime.now(tz))

all_d_star = []
for i in range(0, len(all_xi_star)):
    d_star = np.sum(all_xi_star[i], axis=0)
    all_d_star.append(d_star)
    d_tilde = np.inner(all_p_star[i], coeff_all_days[i]) - intercept_all_days[i]
    temp = [x-y for x,y in zip(d_star, d_tilde)]
    norm = np.linalg.norm(temp)
    mse_i = norm*norm
    #print('Day ', i, ' mse: ', mse_i)


#df_d = pd.DataFrame(data=all_d_star)
#df_d.to_csv('all_d_'+str(Q_perc)+'.csv')

print('## Plots ##')
visualization.plot_demand_duck_curve_simple('Data_May_2022/CAISO-netdemandMW-May2022_hourly_avg.xlsx', df, nom_loads_names, all_xi_star, T, df['Price ($/KW)'].values.tolist(), all_p_star, coeff_all_days, intercept_all_days, d_star_from_p_random_all_days, number_samples)
#visualization.plot_demand_duck_curve('Data_May_2022/CAISO-netdemandMW-May2022_hourly_avg.xlsx', df, nom_loads_names, all_xi_star, T, df['Price ($/KW)'].values.tolist(), all_p_star, coeff_all_days, intercept_all_days, d_star_from_p_random_all_days, number_samples)
exit(9)
print(all_ga_tilde)
print(all_ga_tilde_fun)
print(all_gi)
visualization.plot_avg_obj_over_month(all_ga_tilde, all_gi, all_ui, all_ga_pes_dstar, number_samples, ui_divider, Q_perc, df_ui_columns, a)
visualization.plot_all_d0_dstar(df, nom_loads_names, all_xi_star, T, d_star_from_p_random_all_days, number_samples)
visualization.plot_prices(df['Price ($/KW)'].values.tolist(), all_p_star, number_samples)
visualization.plot_demands_prices(df, nom_loads_names, all_xi_star, T, df['Price ($/KW)'].values.tolist(), all_p_star, coeff_all_days, intercept_all_days, number_samples)
visualization.plot_demands_prices_simple(df, nom_loads_names, all_xi_star, T, df['Price ($/KW)'].values.tolist(), all_p_star, coeff_all_days, intercept_all_days, d_star_from_p_random_all_days, number_samples)
#visualization.plot_x0_data(df, nom_loads_names)

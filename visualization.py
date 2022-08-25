import plotly
import plotly.graph_objs as go
import numpy as np
import plotly.io as pio
from plotly.subplots import make_subplots
import random
import pandas as pd

pio.templates.default = "plotly_white"

def plot_obj_RLS(ga_true_all, ga_est, ga_tilde_all, pros_all, rounds, nom_loads):
    g_j_sum = []
    for j in range(0, rounds):
        g_i_j_all = []
        for i in range(0, len(nom_loads)):
            temp = [x for x in pros_all[j][i]]
            gi_j = temp[0]
            g_i_j_all.append(gi_j)
        g_j_sum.append(np.sum(g_i_j_all))

    samples = [x for x in range(0, len(ga_true_all))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=samples, y=ga_true_all, line=dict(color='black', width=5), name="g<sub>a</sub>"))

    #temp
    fig.add_trace(go.Scatter(x=samples, y=ga_tilde_all, line=dict(color='red', width=5), name="g&#771;<sub>a</sub>"))
    #

    #fig.add_trace(go.Scatter(x=samples, y=ga_est, line=dict(color='red', width=5),name="g&#771;<sub>a</sub>"))
    #fig.add_trace(go.Scatter(x=samples, y=ga_tilde_all, line=dict(color='grey', width=5), name="ga tilde"))
    fig.add_trace(go.Scatter(x=samples, y=g_j_sum, line=dict(color='green', width=5), name="Σ<sub>i</sub>g<sub>p<sub>i</sub></sub>"))

    fig.update_layout(xaxis_title="j",
                      yaxis=dict(tickmode='linear', tick0=0, dtick=35, zeroline=False, showgrid=False, showline=True,
                                 linewidth=2, linecolor='black', mirror=True, ticks='inside'),
                      xaxis=dict(tickmode='linear', tick0=0, dtick=10, showgrid=False, showline=True, linewidth=2,
                                 linecolor='black', mirror=True, ticks='inside'), yaxis_title="Payoff ($)",
                      font=dict(family="Times New Roman, monospace", size=60, color="black"),
                      legend_orientation='h')

    plotly.offline.plot(fig, filename='Output_EE_RLS/' + 'Aggreg_obj_RLS' + '.html')
    return

def plot_error_RLS(d_all, x_tot_est_all):
    error = []
    for i in range(0, len(d_all)):
        error.append([x-y for x,y in zip(d_all[i], x_tot_est_all[i])])


    #error_l2_norm = [np.linalg.norm(x - y) for x, y in zip(d_all, x_tot_est_all)]
    error_l2_norm = [np.linalg.norm(x) for x in error]
    samples = [x for x in range(0, len(error_l2_norm))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=samples, y=error_l2_norm, line=dict(color='black', width=5), name="d_error_norm"))
    fig.update_layout(xaxis_title="j",
                      yaxis=dict(tickmode='linear', tick0=0, dtick=500, zeroline=False, showgrid=False, showline=True,
                                 linewidth=2, linecolor='black', mirror=True, ticks='inside'),
                      xaxis=dict(tickmode='linear', tick0=0, dtick=10, showgrid=False, showline=True, linewidth=2,
                                 linecolor='black', mirror=True, ticks='inside'), yaxis_title="Error ||d<sup>j</sup> - d&#771;<sup>j</sup>||",
                      font=dict(family="Times New Roman, monospace", size=60, color="black"), legend_orientation='h')
    plotly.offline.plot(fig, filename='Output_EE_RLS/' + 'Error_RLS' + '.html')

    return

def plot_actions_RLS(d_all, x_tot_est_all, a, nom_total):
    d_all_flat = [item for sublist in d_all for item in sublist]
    x_tot_est_all_flat = [item for sublist in x_tot_est_all for item in sublist]

    samples = [x for x in range(0, len(d_all_flat))]
    nom_total_upper = [a*x for x in nom_total]*len(samples)
    nom_total_all = nom_total*len(samples)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=samples, y=[0] * len(samples), line=dict(color='grey', width=5, dash='dash'),name="0"))
    fig.add_trace(go.Scatter(x=samples, y=nom_total_all, line=dict(color='rgb(158, 185, 243)', width=5, dash='dash'), name='d <sup>&#x000F8;</sup>'))  #"d<sub>0</sub>"))
    fig.add_trace(go.Scatter(x=samples, y=nom_total_upper, line=dict(color='grey', width=5, dash='dash'), name="αd<sup>&#x000F8;</sup>"))
    fig.add_trace(go.Scatter(x=samples, y=d_all_flat, line=dict(color='black', width=5, dash='dash'), name="d<sub>t</sub><sup>j</sup>"))
    fig.add_trace(go.Scatter(x=samples, y=x_tot_est_all_flat, line=dict(color='red', width=5, dash='dash'),name="d&#771;<sub>t</sub><sup>j</sup>"))


    fig.update_layout(xaxis_title="j &#x000B7; T",
                      yaxis=dict(tickmode='linear', tick0=0, dtick=5000, zeroline=False, showgrid=False, showline=True,
                                 linewidth=2, linecolor='black', mirror=True, ticks='inside'),
                      xaxis=dict(tickmode='linear', tick0=0, dtick=24, showgrid=False, showline=True, linewidth=2,
                                 linecolor='black', mirror=True, ticks='inside'), yaxis_title="Demand (W)",
                      font=dict(family="Times New Roman, monospace", size=60, color="black"), legend_orientation='h')
    plotly.offline.plot(fig, filename='Output_EE_RLS/' + 'Actions_RLS' + '.html')
    return

def plot_agg_actions_RLS(p_all, p_max, prices, rounds):
    temp = [item for sublist in p_all for item in sublist]
    samples = [x for x in range(0, len(temp))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=samples, y=[0] * len(samples), line=dict(color='grey', width=5, dash='dash'), name="0"))
    fig.add_trace(go.Scatter(x=samples, y=[p_max] * len(samples), line=dict(color='grey', width=5, dash='dash'), name="p<sup>max</sup>"))
    fig.add_trace(go.Scatter(x=samples, y=temp, line=dict(color='black', width=5), name="p<sub>t</sub><sup>j</sup>"))
    #fig.add_trace(go.Scatter(x=samples, y=prices * rounds, line=dict(color='red', width=5, dash='dash'), name="λ<sub>t</sub>"))

    fig.update_layout(xaxis_title="j &#x000B7; T",
                      yaxis=dict(tickmode='linear', tick0=0, dtick=0.005, zeroline=False, showgrid=False, showline=True,
                                 linewidth=2, linecolor='black', mirror=True, ticks='inside'),
                      xaxis=dict(tickmode='linear', tick0=0, dtick=24, showgrid=False, showline=True, linewidth=2,
                                 linecolor='black', mirror=True, ticks='inside'), yaxis_title="Price ($/KW)",
                      font=dict(family="Times New Roman, monospace", size=60, color="black"), legend_orientation='h')
    plotly.offline.plot(fig, filename='Output_EE_RLS/'+'Agg_actions' + '.html')
    return

def plot_avg_obj_over_month(all_ga_tilde, all_gi, all_ui, all_ga_pes_dstar, number_samples, ui_divider, Q_perc, df_ui_columns, a):
    agg_plus_ui = [0] + all_ui.tolist()
    avg_ga = np.mean(all_ga_tilde)
    avg_ga_pes_dstar = np.mean(all_ga_pes_dstar)

    all_gi = np.array(all_gi)
    all_gi_monthly_sum = np.sum(all_gi, axis=0)
    all_gi_monthly_sum = all_gi_monthly_sum.tolist()
    all_gi_monthly_avg = [x/31 for x in all_gi_monthly_sum]

    avg_all_monthly = [avg_ga] + all_gi_monthly_avg

    bar_width_weight = 0.00001

    # text=df_ui_columns, texttemplate="%{text}", textfont={"size":20}

    all_ui_str = [str(x) for x in all_ui]
    all_gi_monthly_avg_str = [' - '+str(x) for x in all_gi_monthly_avg]
    text_bars = [x+y+z for x,y,z in zip(df_ui_columns, all_ui_str, all_gi_monthly_avg_str)]

    #fig = go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])#Θ&#770;
    if avg_ga <= avg_ga_pes_dstar:
         fig.add_trace(go.Bar(x=[0], y=[avg_ga_pes_dstar], marker_color='Blue', width=[bar_width_weight] * len(avg_all_monthly), opacity=1, name='g<sub>a</sub>(p<sup>es</sup>,d<sup>*</sup>(p<sup>es</sup>))'), secondary_y=False)
         fig.add_trace(go.Bar(x=[0], y=[avg_ga], marker_color='Green', width=[bar_width_weight] * len(avg_all_monthly), opacity=1, name='g&#771;<sub>a</sub>(p<sup>es</sup>;Θ&#770;<sup>J</sup>)'), secondary_y=False)
    else:
        fig.add_trace(go.Bar(x=[0], y=[avg_ga], marker_color='Green', width=[bar_width_weight]*len(avg_all_monthly), opacity=1, name='g&#771;<sub>a</sub>(p<sup>es</sup>;Θ&#770;<sup>J</sup>)'), secondary_y=False)
        fig.add_trace(go.Bar(x=[0], y=[avg_ga_pes_dstar], marker_color='Blue', width=[bar_width_weight] * len(avg_all_monthly), opacity=1, name='g<sub>a</sub>(p<sup>es</sup>,d<sup>*</sup>(p<sup>es</sup>))'), secondary_y=False)
    fig.add_trace(go.Bar(x=all_ui.tolist(), y=all_gi_monthly_avg, hoverinfo='text', hovertext=text_bars, marker_color='Grey', width=[bar_width_weight] * len(avg_all_monthly), name='g<sub>i</sub>(x<sub>i</sub><sup>es</sup>,p<sup>es</sup>)'), secondary_y=True)

    fig.update_yaxes(title_text="Utility g&#771;<sub>a</sub>, g<sub>a</sub> ($)", secondary_y=False)
    fig.update_yaxes(title_text="Utility g<sub>i</sub> ($)", secondary_y=True)

    fig.update_layout(title='', xaxis_tickangle=0, xaxis_title='u<sub>i</sub>', xaxis=dict(tickformat=".2", dtick=0.0001),
                      font=dict(family="Times New Roman, monospace",size=60, color="black"), legend=dict(orientation='h', yanchor='bottom', y=1.02), barmode='overlay')
    plotly.offline.plot(fig, filename='Data_May_2022/' + 'Avg_Utility' + "_" + str(number_samples) + 'samples_' + str(ui_divider) + 'ui_divider_' + str(Q_perc) + 'Qi_' + str(a) + 'a' + '.html', include_mathjax='cdn') #.2e #-45 #'Daily average utility across May 1<sup>st</sup> - 31<sup>st</sup>, 2022'

    temp_ga_gi = [avg_ga] + [avg_ga_pes_dstar] + all_gi_monthly_avg
    df_ga_gi = pd.DataFrame(temp_ga_gi)
    df_ga_gi.to_csv('Data_May_2022/ga_gi_samples' + str(number_samples) + '.csv')
    return

def plot_all_d0_dstar(df, nom_loads_names, all_xi_star, T, d_star_from_p_random_all_days, number_samples):
    all_d0 = []
    for i in range(0, len(df), T):  # for each day
        nom_loads = df[nom_loads_names][i:i + T].values
        d0 = np.sum(nom_loads, axis=1)
        all_d0.append(d0)

    all_dstar = []
    for i in range(0, len(all_xi_star)):
        x_of_this_day = np.array(all_xi_star[i])
        all_dstar.append(np.sum(x_of_this_day, axis=0))

    all_d0 = [item for sublist in all_d0 for item in sublist]
    all_dstar = [item for sublist in all_dstar for item in sublist]

    samples = [x for x in range(0, len(all_d0))]

    d_star_from_p_random_all_days = [item for sublist in d_star_from_p_random_all_days for item in sublist]
    d_star_from_p_random_all_days = [item for sublist in d_star_from_p_random_all_days for item in sublist]
    samples_extended = []
    for j in range(0, len(samples), T):
        samples_extended.append(samples[j:j+T]*number_samples)
    samples_extended = [item for sublist in samples_extended for item in sublist]

    fig = go.Figure()
    #fig.add_trace(go.Scatter(x=samples, y=all_d0, line=dict(color='blue', width=5, dash='dash'), name="d<sup>0</sup>"))
    fig.add_trace(go.Scatter(x=samples_extended, y=d_star_from_p_random_all_days, name='d<sup>*</sup>(p<sub>r</sub>)', marker=dict(color="crimson", size=12), mode="markers"))
    fig.add_trace(go.Scatter(x=samples, y=all_dstar, line=dict(color='green', width=5), name="Θ&#770;<sup>J<sup>T</sup></sup>φ(p<sup>es</sup>)"))
    fig.update_layout(xaxis_title="Hour",
                      yaxis=dict(linewidth=2, linecolor='black', mirror=True, ticks='inside'),
                      xaxis=dict(linewidth=2, linecolor='black', mirror=True, ticks='inside'), yaxis_title="Demand (KWh)",
                      font=dict(family="Times New Roman, monospace", size=60, color="black"), legend=dict(orientation='h', yanchor='bottom', y=1.02))
    plotly.offline.plot(fig, filename='Data_May_2022/' + 'd0_dstar'+ "_" + str(number_samples) + 'samples' + '.html')
    return

def plot_prices(all_lambda_prices, all_p_star, number_samples):
    all_p_star = [item for sublist in all_p_star for item in sublist]

    samples = [x for x in range(0, len(all_p_star))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=samples, y=all_lambda_prices, line=dict(color='blue', width=5, dash='dash'), name="λ<sub>t</sub>"))
    fig.add_trace(go.Scatter(x=samples, y=all_p_star, line=dict(color='green', width=5), name="p<sub>t</sub><sup>*</sup>"))
    fig.update_layout(xaxis_title="Hour",
                      yaxis=dict(linewidth=2, linecolor='black', mirror=True, ticks='inside'),
                      xaxis=dict(linewidth=2, linecolor='black', mirror=True, ticks='inside'),
                      yaxis_title="Price ($/KWh)",
                      font=dict(family="Times New Roman, monospace", size=20, color="black"), legend_orientation='h')
    plotly.offline.plot(fig, filename='Data_May_2022/' + 'prices'+ "_" + str(number_samples) + 'samples' + '.html', include_mathjax='cdn')
    return

def plot_demands_prices(df, nom_loads_names, all_xi_star, T, all_lambda_prices, all_p_star, coeff_all_days, intercept_all_days, number_samples):
    all_d_tilde = []
    for i in range(0, len(all_p_star)):
        temp_d = np.inner(all_p_star[i], coeff_all_days[i]) + intercept_all_days[i]
        temp_d = temp_d.tolist()
        all_d_tilde.append(temp_d)
    #print(all_d_tilde)
    all_d_tilde = [item for sublist in all_d_tilde for item in sublist]
    #print(all_d_tilde)

    all_d0 = []
    for i in range(0, len(df), T):  # for each day
        nom_loads = df[nom_loads_names][i:i + T].values
        d0 = np.sum(nom_loads, axis=1)
        all_d0.append(d0)

    all_dstar = []
    for i in range(0, len(all_xi_star)):
        x_of_this_day = np.array(all_xi_star[i])
        all_dstar.append(np.sum(x_of_this_day, axis=0))

    all_d0 = [item for sublist in all_d0 for item in sublist]
    all_dstar = [item for sublist in all_dstar for item in sublist]

    all_p_star = [item for sublist in all_p_star for item in sublist]


    samples = [x for x in range(0, len(all_p_star))]
    #fig = go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=samples, y=all_d0, line=dict(color='Blue', width=5, dash='dash'), name="d<sup>0</sup><sub>t</sub> (KW)"), secondary_y=False)
    fig.add_trace(go.Scatter(x=samples, y=all_dstar, line=dict(color='Green', width=5), name="d<sup>*</sup> (KW)"), secondary_y=False)
    fig.add_trace(go.Scatter(x=samples, y=all_d_tilde, line=dict(color='Green', width=5, dash='dash'), name="d<sub>tilde</sub> (KW)"), secondary_y=False)
    fig.add_trace(go.Scatter(x=samples, y=all_lambda_prices, line=dict(color='Red', width=5), name="λ<sub>t</sub> ($/KWh)"), secondary_y=True)
    fig.add_trace(go.Scatter(x=samples, y=all_p_star, line=dict(color='Black', width=5), name="p<sub>t</sub><sup>*</sup> ($/KWh)"), secondary_y=True)

    fig.add_trace(go.Scatter(x=samples, y=[x-y for x,y in zip(all_d0, all_dstar)], line=dict(color='Purple', width=5), name="d<sup>0</sup> - d<sup>*</sup> (KW)"), secondary_y=False)
    fig.add_trace(go.Scatter(x=samples, y=[x - y for x, y in zip(all_d0, all_d_tilde)], line=dict(color='Purple', width=5, dash='dash'), name="d<sup>0</sup> - d<sub>tilde</sub> (KW)"), secondary_y=False)
    fig.add_trace(go.Scatter(x=samples, y=[x-y for x, y in zip(all_lambda_prices, all_p_star)], line=dict(color='Orange', width=5), name="λ<sub>t</sub> - p<sub>t</sub><sup>*</sup> ($/KWh)"), secondary_y=True)

    d0_dstar = [x-y for x,y in zip(all_d0, all_dstar)]
    l_p = [x-y for x, y in zip(all_lambda_prices, all_p_star)]
    d0_dtilde = [x-y for x,y in zip(all_d0, all_d_tilde)]
    fig.add_trace(go.Scatter(x=samples, y=[x*y for x, y in zip(l_p, d0_dstar)],line=dict(color='Grey', width=5),name="ga"), secondary_y=False)
    fig.add_trace(go.Scatter(x=samples, y=[x * y for x, y in zip(l_p, d0_dtilde)], line=dict(color='Grey', width=5, dash='dash'), name="ga<sub>tilde</sub>"), secondary_y=False)


    fig.update_layout(xaxis_title="Hour",
                      yaxis=dict(linewidth=2, linecolor='black', mirror=True, ticks='inside'),
                      xaxis=dict(linewidth=2, linecolor='black', mirror=True, ticks='inside'),
                      font=dict(family="Times New Roman, monospace", size=20, color="black"), legend_orientation='h')
    fig.update_yaxes(title_text="Demand (KW)", secondary_y=False)
    fig.update_yaxes(title_text="Price ($/KWh)", secondary_y=True, range=[-5,0.2])
    plotly.offline.plot(fig, filename='Data_May_2022/' + 'demands_prices'+ "_" + str(number_samples) + 'samples' + '.html')
    return

def plot_demands_prices_simple(df, nom_loads_names, all_xi_star, T, all_lambda_prices, all_p_star, coeff_all_days, intercept_all_days, d_star_from_p_random_all_days, number_samples):
    all_d_tilde = []
    for i in range(0, len(all_p_star)):
        temp_d = np.inner(all_p_star[i], coeff_all_days[i]) + intercept_all_days[i]
        temp_d = temp_d.tolist()
        all_d_tilde.append(temp_d)
    all_d_tilde = [item for sublist in all_d_tilde for item in sublist]

    all_d0 = []
    for i in range(0, len(df), T):  # for each day
        nom_loads = df[nom_loads_names][i:i + T].values
        d0 = np.sum(nom_loads, axis=1)
        all_d0.append(d0)

    all_dstar = []
    for i in range(0, len(all_xi_star)):
        x_of_this_day = np.array(all_xi_star[i])
        all_dstar.append(np.sum(x_of_this_day, axis=0))

    all_d0 = [item for sublist in all_d0 for item in sublist]
    all_dstar = [item for sublist in all_dstar for item in sublist]
    all_p_star = [item for sublist in all_p_star for item in sublist]

    samples = [x for x in range(0, len(all_p_star))]

    d_star_from_p_random_all_days = [item for sublist in d_star_from_p_random_all_days for item in sublist]
    d_star_from_p_random_all_days = [item for sublist in d_star_from_p_random_all_days for item in sublist]
    samples_extended = []
    for j in range(0, len(samples), T):
        samples_extended.append(samples[j:j+T]*number_samples)

    samples_extended = [item for sublist in samples_extended for item in sublist]


    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=samples, y=all_d0, line=dict(color='Blue', width=5, dash='dash'), name="d<sup>0</sup>"), secondary_y=False)
    #fig.add_trace(go.Scatter(x=samples, y=all_dstar, line=dict(color='Green', width=5), name="d<sup>*</sup> (KW)"), secondary_y=False)
    fig.add_trace(go.Scatter(x=samples, y=all_d_tilde, line=dict(color='Green', width=5), name="Θ&#770;<sup>J<sup>T</sup></sup>φ(p<sup>es</sup>)"), secondary_y=False)#d&#771;
    #fig.add_trace(go.Scatter(x=samples_extended, y=d_star_from_p_random_all_days, name='d data', marker=dict(color="crimson", size=12), mode="markers"), secondary_y=False)
    fig.add_trace(go.Scatter(x=samples, y=all_lambda_prices, line=dict(color='Red', width=5), name="λ"), secondary_y=True)
    fig.add_trace(go.Scatter(x=samples, y=all_p_star, line=dict(color='Black', width=5), name="p<sup>*</sup>"), secondary_y=True)


    fig.update_layout(xaxis_title="Hour (May 1<sup>st</sup> - May 31<sup>st</sup>, 2022)",
                      yaxis=dict(linewidth=2, linecolor='black', mirror=True, ticks='inside', tickmode='linear', tick0=8000, dtick=2000),
                      xaxis=dict(linewidth=2, linecolor='black', mirror=True, ticks='inside'),
                      font=dict(family="Times New Roman, monospace", size=60, color="black"), legend=dict(orientation='h', yanchor='bottom', y=1.02))
    fig.update_yaxes(title_text="Demand (KWh)", secondary_y=False)
    fig.update_yaxes(title_text="Price ($/KWh)", secondary_y=True, range=[-3,0.5])
    plotly.offline.plot(fig, filename='Data_May_2022/' + 'demands_prices_simple'+ "_" + str(number_samples) + 'samples' + '.html') #g&#771;<sub>a</sub> #d&#771;<sup>j</sup>

    return

def plot_x0_data(df, nom_loads_names):
    nom_loads = df[nom_loads_names].values

    samples = [x for x in range(0, nom_loads.shape[0])]
    fig = go.Figure()
    for i in range(0, nom_loads.shape[1]):
        number1 = random.randint(0,255)
        number2 = random.randint(0,255)
        number3 = random.randint(0,255)
        fig.add_trace(go.Scatter(x=samples, y=nom_loads[:, i], line=dict(color='rgb('+ str(number1) +',' + str(number2) + ',' + str(number3) + ')', width=5), showlegend=False))

    fig.update_layout(xaxis_title="Hour (May 1<sup>st</sup> - May 31<sup>st</sup>, 2022)",
                      yaxis=dict(linewidth=2, linecolor='black', mirror=True, ticks='inside'),
                      xaxis=dict(linewidth=2, linecolor='black', mirror=True, ticks='inside'),
                      font=dict(family="Times New Roman, monospace", size=60, color="black"))
    fig.update_yaxes(title_text="Demand x<sub>i</sub><sup>0</sup> (KWh)")
    plotly.offline.plot(fig, filename='Data_May_2022/' + 'x0_demands' + '.html')

    d0 = np.sum(nom_loads, axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=samples, y=d0, line=dict(color='Blue', width=5), showlegend=False))
    fig.update_layout(xaxis_title="Hour (May 1<sup>st</sup> - May 31<sup>st</sup>, 2022)",
                      yaxis=dict(linewidth=2, linecolor='black', mirror=True, ticks='inside'),
                      xaxis=dict(linewidth=2, linecolor='black', mirror=True, ticks='inside'),
                      font=dict(family="Times New Roman, monospace", size=60, color="black"))
    fig.update_yaxes(title_text="Total Demand d<sup>0</sup> (KWh)")
    plotly.offline.plot(fig, filename='Data_May_2022/' + 'd0_demand' + '.html')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=samples, y=df['Price ($/KW)'].values.tolist(), line=dict(color='Red', width=5), showlegend=False))
    fig.update_layout(xaxis_title="Hour (May 1<sup>st</sup> - May 31<sup>st</sup>, 2022)",
                      yaxis=dict(linewidth=2, linecolor='black', mirror=True, ticks='inside'),
                      xaxis=dict(linewidth=2, linecolor='black', mirror=True, ticks='inside'),
                      font=dict(family="Times New Roman, monospace", size=60, color="black"))
    fig.update_yaxes(title_text="LMP λ<sub>t</sub> ($/KWh)")
    plotly.offline.plot(fig, filename='Data_May_2022/' + 'lambda' + '.html')
    return

def plot_bars_samples(df):
    fig = make_subplots(rows=1, cols=2, shared_xaxes=True, horizontal_spacing=0.2, specs=[[{"secondary_y": True}, {"secondary_y": True}]])
    samples = df.columns.tolist()
    temp_df = df.iloc[2:,:]
    temp_df_arr = np.asarray(temp_df)
    temp_df_arr_avg = temp_df_arr.mean(axis=0)

    fig.add_trace(go.Bar(x=['10', '20'], y=df.iloc[0,0:2].tolist(), marker_color='Blue', opacity=1, text=[round(x,2) for x in df.iloc[0,0:2].tolist()], name='g<sub>a</sub>(p<sup>es</sup>,d<sup>*</sup>(p<sup>es</sup>))'), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Bar(x=['10', '20'], y=df.iloc[1,0:2].tolist(), marker_color='Green', opacity=1, text=[round(x,2) for x in df.iloc[1,0:2].tolist()], name='g&#771;<sub>a</sub>(p<sup>es</sup>;Θ&#770;<sup>J</sup>)'), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Bar(x=['10', '20'], y=temp_df_arr_avg[0:2], marker_color='Grey', opacity=0.7, text=[round(x,2) for x in temp_df_arr_avg[0:2]], name='Average g<sub>i</sub>(x<sub>i</sub><sup>es</sup>,p<sup>es</sup>)'), row=1, col=1, secondary_y=True)

    fig.add_trace(go.Bar(x=['50', '80', '100'], y=df.iloc[0,2:].tolist(), marker_color='Blue', opacity=1, text=[round(x,2) for x in df.iloc[0,2:].tolist()], name='g<sub>a</sub>(p<sup>es</sup>,d<sup>*</sup>(p<sup>es</sup>))'), row=1, col=2, secondary_y=False)
    fig.add_trace(go.Bar(x=['50', '80', '100'], y=df.iloc[1,2:].tolist(), marker_color='Green', opacity=1, text=[round(x,2) for x in df.iloc[1,2:].tolist()], name='g&#771;<sub>a</sub>(p<sup>es</sup>;Θ&#770;<sup>J</sup>)'), row=1, col=2, secondary_y=False)
    fig.add_trace(go.Bar(x=['50', '80', '100'], y=temp_df_arr_avg[2:], marker_color='Grey', opacity=1, text=[round(x,2) for x in temp_df_arr_avg[2:]], name='Average g<sub>i</sub>(x<sub>i</sub><sup>es</sup>,p<sup>es</sup>)'), row=1, col=2, secondary_y=True)

    fig.update_yaxes(title_text="Utility g&#771;<sub>a</sub>, g<sub>a</sub> ($)", secondary_y=False)
    fig.update_yaxes(title_text="Average Utility g<sub>i</sub> ($)", secondary_y=True, range=[0,21])
    fig.update_xaxes(title_text="Number of samples j")

    fig.update_layout(title='', xaxis_tickangle=0,  xaxis=dict(tickvals = samples), #xaxis_title='Number of samples',
                      font=dict(family="Times New Roman, monospace", size=53, color="black"),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(size=53)), barmode='group') #36

    plotly.offline.plot(fig, filename='Data_May_2022/' + 'Bars_samples' + '.html', include_mathjax='cdn')
    return

def plot_bars_prosumers(df):
    prosumers = df.columns.tolist()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(x=prosumers, y=df.iloc[0,:].tolist(), marker_color='Blue', opacity=1, text=[round(x,2) for x in df.iloc[0,:].tolist()], name='g<sub>a</sub>(p<sup>es</sup>,d<sup>*</sup>(p<sup>es</sup>))'), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Bar(x=prosumers, y=df.iloc[1,:].tolist(), marker_color='Green', opacity=1, text=[round(x,2) for x in df.iloc[1,:].tolist()], name='g&#771;<sub>a</sub>(p<sup>es</sup>;Θ&#770;<sup>J</sup>)'), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Bar(x=prosumers, y=df.iloc[2,:].tolist(), marker_color='Grey', opacity=0.8, text=[round(x,2) for x in df.iloc[2,:].tolist()], name='Average g<sub>i</sub>(x<sub>i</sub><sup>es</sup>,p<sup>es</sup>)'), row=1, col=1, secondary_y=True)

    fig.update_yaxes(title_text="Utility g&#771;<sub>a</sub>, g<sub>a</sub> ($)", secondary_y=False)
    fig.update_yaxes(title_text="Average Utility g<sub>i</sub> ($)", secondary_y=True, range=[0,3])
    fig.update_xaxes(title_text="Number of prosumers N")

    fig.update_layout(title='', xaxis_tickangle=0,xaxis_title='Number of prosumers N',  xaxis=dict(tickvals = prosumers),
                      font=dict(family="Times New Roman, monospace", size=53, color="black"),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(size=53)), barmode='group') #36
    plotly.offline.plot(fig, filename='Data_May_2022/' + 'Bars_prosumers' + '.html', include_mathjax='cdn')
    return

def plot_demand_duck_curve(filename_duck, df, nom_loads_names, all_xi_star, T, all_lambda_prices, all_p_star, coeff_all_days, intercept_all_days, d_star_from_p_random_all_days, number_samples):
    df_duck = pd.read_excel(filename_duck)
    duck_demand = df_duck['May_2022'].tolist() #in MWh
    duck_demand_kwh = [1000*x for x in duck_demand]

    all_d_tilde = []
    for i in range(0, len(all_p_star)):
        temp_d = np.inner(all_p_star[i], coeff_all_days[i]) + intercept_all_days[i]
        temp_d = temp_d.tolist()
        all_d_tilde.append(temp_d)
    all_d_tilde = [item for sublist in all_d_tilde for item in sublist]

    all_d0 = []
    for i in range(0, len(df), T):  # for each day
        nom_loads = df[nom_loads_names][i:i + T].values
        d0 = np.sum(nom_loads, axis=1)
        all_d0.append(d0)

    all_d0 = [item for sublist in all_d0 for item in sublist]
    all_p_star = [item for sublist in all_p_star for item in sublist]
    samples = [x for x in range(0, len(all_p_star))]

    agg_N = 1000
    effect = [x+agg_N*y-agg_N*z for x,y,z in zip(duck_demand_kwh, all_d_tilde, all_d0)]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=samples, y=all_d0, line=dict(color='Blue', width=5, dash='dash'), name="d<sup>0</sup>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=samples, y=all_d_tilde, line=dict(color='Green', width=5), name="Θ&#770;<sup>J<sup>T</sup></sup>φ(p<sup>es</sup>)"), secondary_y=False)#d&#771;
    fig.add_trace(go.Scatter(x=samples, y=duck_demand_kwh, line=dict(color='Red', width=5), name="d<sub>net</sub>"), secondary_y=True)
    fig.add_trace(go.Scatter(x=samples, y=effect, line=dict(color='Purple', width=5), name="d<sub>net</sub>-N<sub>agg</sub> (d<sup>0</sup>-Θ&#770;<sup>J<sup>T</sup></sup>φ(p<sup>es</sup>))"), secondary_y=True)

    fig.update_layout(xaxis_title="Hour (May 1<sup>st</sup> - May 31<sup>st</sup>, 2022)",
                      yaxis=dict(linewidth=2, linecolor='black', mirror=True, ticks='inside', tickmode='linear', tick0=8000, dtick=2000),
                      xaxis=dict(linewidth=2, linecolor='black', mirror=True, ticks='inside'),
                      font=dict(family="Times New Roman, monospace", size=60, color="black"), legend=dict(orientation='h', yanchor='bottom', y=1.02))
    fig.update_yaxes(title_text="Demand (KWh)", secondary_y=False)
    fig.update_yaxes(title_text="Net Demand CAISO (KWh)", secondary_y=True)
    plotly.offline.plot(fig, filename='Data_May_2022/' + 'demands_net_dem_duck' + '.html')
    return

def plot_demand_duck_curve_simple(filename_duck, df, nom_loads_names, all_xi_star, T, all_lambda_prices, all_p_star, coeff_all_days, intercept_all_days, d_star_from_p_random_all_days, number_samples):
    df_duck = pd.read_excel(filename_duck)
    duck_demand = df_duck['May_2022'].tolist() #in MWh
    duck_demand_kwh = [1000*x for x in duck_demand]

    all_d_tilde = []
    for i in range(0, len(all_p_star)):
        temp_d = np.inner(all_p_star[i], coeff_all_days[i]) + intercept_all_days[i]
        temp_d = temp_d.tolist()
        all_d_tilde.append(temp_d)
    all_d_tilde = [item for sublist in all_d_tilde for item in sublist]

    all_d0 = []
    for i in range(0, len(df), T):  # for each day
        nom_loads = df[nom_loads_names][i:i + T].values
        d0 = np.sum(nom_loads, axis=1)
        all_d0.append(d0)

    all_d0 = [item for sublist in all_d0 for item in sublist]
    all_p_star = [item for sublist in all_p_star for item in sublist]
    samples = [x for x in range(0, len(all_p_star))]

    agg_N = 1000
    effect = [x+agg_N*y-agg_N*z for x,y,z in zip(duck_demand_kwh, all_d_tilde, all_d0)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=samples, y=duck_demand_kwh, line=dict(color='Red', width=5), name="d<sub>net</sub>"))
    fig.add_trace(go.Scatter(x=samples, y=effect, line=dict(color='Purple', width=5), name="d<sub>net</sub>-N<sub>agg</sub> (d<sup>0</sup>-Θ&#770;<sup>J<sup>T</sup></sup>φ(p<sup>es</sup>))"))

    fig.update_layout(xaxis_title="Hour (May 1<sup>st</sup> - May 31<sup>st</sup>, 2022)",
                      yaxis=dict(linewidth=2, linecolor='black', mirror=True, ticks='inside', tickmode='linear', tick0=-5000000, dtick=10000000),
                      xaxis=dict(linewidth=2, linecolor='black', mirror=True, ticks='inside'),
                      font=dict(family="Times New Roman, monospace", size=60, color="black"), legend=dict(orientation='h', yanchor='bottom', y=1.02))
    fig.update_yaxes(title_text="Net Demand (KWh)")
    plotly.offline.plot(fig, filename='Data_May_2022/' + 'demands_net_dem_duck_simple' + '.html')
    return

def plot_bars_co2():
    Qi = ['Q<sub>i</sub>=0', 'Q<sub>i</sub>=0.01W<sub>i</sub>', 'Q<sub>i</sub>=0.1W<sub>i</sub>']
    kwh_saved = [0, 8807592.147*0.01, 8807592.147*0.1]
    co2_saved = [0, 8807592.147*0.01*0.85, 8807592.147*0.1*0.85]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=co2_saved, y=Qi, orientation='h', marker_color='darkmagenta', opacity=1, text=[int(x) for x in co2_saved], name='CO<sub>2</sub> lbs saved'), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Bar(x=kwh_saved, y=Qi, orientation='h', marker_color='darkgoldenrod', opacity=1, text=[int(x) for x in kwh_saved], name='KWh saved'), row=1, col=1, secondary_y=False)

    #fig.update_yaxes(title_text="KWh", secondary_y=False)
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="", secondary_y=False) #Q<sub>i</sub>

    fig.update_layout(title='', xaxis_tickangle=0, xaxis_title='', xaxis=dict(tick0=0, dtick=150000),#xaxis=dict(tickvals = Qi),
                      font=dict(family="Times New Roman, monospace", size=60, color="black"), #46
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, font=dict(size=53)), barmode='group') #36
    plotly.offline.plot(fig, filename='Data_May_2022/' + 'Bars_CO2' + '.html', include_mathjax='cdn')
    return

def plot_all_alpha_demands(df, nom_loads_names, T):
    df_0 = pd.read_csv('all_d_0.csv', index_col=0)
    df_001 = pd.read_csv('all_d_001.csv', index_col=0)
    df_01 = pd.read_csv('all_d_01.csv', index_col=0)

    all_d_tilde_Q0 = df_0.values.tolist()
    all_d_tilde_Q001 = df_001.values.tolist()
    all_d_tilde_Q01 = df_01.values.tolist()

    all_d_tilde_Q0 = [item for sublist in all_d_tilde_Q0 for item in sublist]
    all_d_tilde_Q001 = [item for sublist in all_d_tilde_Q001 for item in sublist]
    all_d_tilde_Q01 = [item for sublist in all_d_tilde_Q01 for item in sublist]


    all_d0 = []
    for i in range(0, len(df), T):
        nom_loads = df[nom_loads_names][i:i + T].values
        d0 = np.sum(nom_loads, axis=1)
        all_d0.append(d0)

    all_d0 = [item for sublist in all_d0 for item in sublist]
    samples = [x for x in range(0, len(all_d0))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=samples, y=all_d0, line=dict(color='Blue', width=5, dash='dash'), name="d<sup>0</sup>"))
    fig.add_trace(go.Scatter(x=samples, y=all_d_tilde_Q0, line=dict(color='Green', width=5), name="Θ&#770;<sup>J<sup>T</sup></sup>φ(p<sup>es</sup>) for Q<sub>i</sub>=0"))
    fig.add_trace(go.Scatter(x=samples, y=all_d_tilde_Q001, line=dict(color='red', width=5), name="Θ&#770;<sup>J<sup>T</sup></sup>φ(p<sup>es</sup>) for Q<sub>i</sub>=0.01W<sub>i</sub>"))
    fig.add_trace(go.Scatter(x=samples, y=all_d_tilde_Q01, line=dict(color='purple', width=5), name="Θ&#770;<sup>J<sup>T</sup></sup>φ(p<sup>es</sup>) for Q<sub>i</sub>=0.1W<sub>i</sub>"))

    fig.update_layout(xaxis_title="Hour (May 1<sup>st</sup> - May 31<sup>st</sup>, 2022)",
                      yaxis=dict(linewidth=2, linecolor='black', mirror=True, ticks='inside', tickmode='linear', tick0=8000, dtick=2000),
                      xaxis=dict(linewidth=2, linecolor='black', mirror=True, ticks='inside'),
                      font=dict(family="Times New Roman, monospace", size=54, color="black"), legend=dict(orientation='h', yanchor='bottom', y=1.02))
    fig.update_yaxes(title_text="Demand (KWh)")
    plotly.offline.plot(fig, filename='Data_May_2022/' + 'all_dem_alpha' + '.html')
    return
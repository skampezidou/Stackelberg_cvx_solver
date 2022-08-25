import numpy as np
import cvxpy as cp
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint
from scipy.optimize import minimize
import pandas, os
from cvxpy.atoms.affine.wraps import psd_wrap
from symfit import exp

def sample(p_max, T):
    p0 = np.random.uniform(0, p_max, T + 1)
    p0_list = p0.tolist()
    p_avg = np.mean(p0_list[:T])
    p0 = np.asarray(p0_list)
    p0[T] = p_avg
    return p0

def error(x_true, x_est):
    z = [np.abs(x-y) for x,y in zip(x_true,x_est[0])]
    e = np.linalg.norm(z)
    return e

def read_large_data(file):
    df = pandas.read_csv(file)
    nom_total = df['Total'].tolist()
    df = df.drop(columns=['Date','Timestamp','Total'])
    df_array = df.values
    nom_loads = np.transpose(df_array).tolist()
    return nom_loads, nom_total

def subdata_selection(nom_loads, nom_total, prices, T):
    nom_total = nom_total[0:T]
    prices = prices[0:T]
    loads = []
    for i in range(0, len(nom_loads)):
        loads.append(nom_loads[i][0:T])

    return loads, nom_total, prices

def agg_Dd(nom_loads, x):
    nom_loads_sum_i = [x + y for x, y in zip(nom_loads[0], nom_loads[1])]
    x_sum_i = [x + y for x, y in zip(x[0], x[1])]
    Dd = [x - y for x, y in zip(nom_loads_sum_i, x_sum_i)]
    return Dd

def pros_opt_RLS(nom_loads_i, agg_solver_res_k_1, a, u_pros_i, Q_pros_i, T):
    A1 = np.identity(T)
    A2 = -A1
    A3 = np.concatenate((A1, A2), axis=0)

    x = cp.Variable(T)
    x0 = nom_loads_i
    x0_array = np.asarray(x0)

    p_played_array = np.asarray(agg_solver_res_k_1)
    b = np.ones((T,))
    b_T = b.T
    b1 = [a*x for x in x0]
    b2 = b*0
    b3 = np.concatenate((b1, b2), axis=0)

    W = sum(x0) - Q_pros_i

    prob = cp.Problem(cp.Maximize(-u_pros_i * cp.quad_form(x, psd_wrap(psd_wrap(np.identity(T)))) -u_pros_i * x0_array.T @ x0_array + 2*u_pros_i*x0_array.T @ x - p_played_array.T @ x + p_played_array.T @ x0_array), [A3 @ x <= b3,  b_T @ x == W])
    #psd_wrap(psd_wrap(np.identity(T))
    prob.solve(verbose=False) #True
    return [prob.value, x.value.tolist()]

def agg_true_obj_RLS(p_star, d_temp, nom_total, prices): #all lists
    Dd = [x-y for x,y in zip(nom_total, d_temp)]
    Dd_array = np.asarray(Dd)
    l_p = [x - y for x, y in zip(prices, p_star)]
    l_p_array = np.asarray(l_p)
    return np.inner(l_p_array, Dd_array)

def agg_temp_obj_RLS(p_star, d_tilde, nom_total, prices):
    d_tilde = d_tilde.tolist()
    d_tilde = [item for sublist in d_tilde for item in sublist]

    Dd = [x-y for x,y in zip(nom_total, d_tilde)]
    Dd_array = np.asarray(Dd)
    l_p = [x - y for x, y in zip(prices, p_star)]
    l_p_array = np.asarray(l_p)
    return np.inner(l_p_array, Dd_array)

def agg_est_opt_RLS(p0, Th_j_1, prices, nom_total, A_basis, c_basis, p_max, T):
    prices = np.asarray(prices)
    nom_total = np.asarray(nom_total)
    nom_total = np.reshape(nom_total, (nom_total.shape[0], 1))
    c_basis = np.reshape(c_basis, (c_basis.shape[0],1))

    bounds = Bounds(np.zeros(T), np.ones(T)*p_max)

    res = minimize(agg_est_opt_RLS_obj, p0, bounds=bounds, args=(Th_j_1, prices, nom_total, A_basis, c_basis, T), method='SLSQP', options={'disp': True})
    return res

def agg_est_opt_RLS_obj(p_star, Th_j_1, prices, nom_total, A_basis, c_basis, T):
    d_tilde = RLS_predict(Th_j_1, A_basis, c_basis, p_star)

    prices = prices.tolist()
    p_star = p_star.tolist()
    nom_total = nom_total.tolist()
    nom_total = [item for sublist in nom_total for item in sublist]
    d_tilde = d_tilde.tolist()
    d_tilde = [item for sublist in d_tilde for item in sublist]

    res = []
    for i in range(0, T):
        #print(-(prices[i] - p_star[i]) * (nom_total[i] - d_tilde[i]))
        res.append(-(prices[i] - p_star[i]) * (nom_total[i] - d_tilde[i]))
    obj = np.sum(res)
    return obj

def agg_test(p_star, Th_j_1, prices, nom_total, A_basis, c_basis, T):
    print(p_star) #np.subtract(prices, p_star)
    res = -np.matmul(np.transpose(np.reshape(np.subtract(prices, p_star), (T,1))), np.subtract(nom_total, np.matmul(np.transpose(Th_j_1), np.asarray([np.tanh(x) for x in np.concatenate((np.add(np.matmul(A_basis, p_star), c_basis), np.ones(1)), axis=0).tolist()]))))
    exit(9)
    return res

def RLS_phi_j(p_star, A_basis, c_basis):  ###check if operations are correct with inner products
    p_star_arr = np.asarray(p_star)
    c_basis = np.reshape(c_basis, (c_basis.shape[0], 1))
    p_star_arr = np.reshape(p_star_arr, (p_star_arr.shape[0],1))
    temp = np.add(np.dot(A_basis,p_star_arr),c_basis)

    #temp_list = temp.tolist()
    #print(len(temp_list[0]))
    #temp_list_2 = [item for sublist in temp_list for item in sublist]
    #print(len(temp_list_2))
    #temp_list_2 = temp_list[0]
    #print(temp_list_2)


    phi_j = np.tanh(temp)#[np.tanh(x) for x in temp_list] #tanh
    #phi_j.append(1.0)
    #b = np.asarray(phi_j)
    #b = np.reshape(b, (b.shape[0], 1))

    one = np.ones(1)
    one = np.reshape(one, (1,1))
    b = np.concatenate((phi_j, one), axis=0)
    return b

def RLS_predict(Th_j_1, A_basis, c_basis, p_star): #Change tanh accordingly
    phi_j = RLS_phi_j(p_star, A_basis, c_basis)
    d_tilde = np.dot(np.transpose(Th_j_1), phi_j)
    return d_tilde

def update_RLS(p_star, Th_j_1, A_basis, c_basis, d_temp, m, P_j_1):
    phi_j = RLS_phi_j(p_star, A_basis, c_basis)
    phi_j = np.reshape(phi_j, (phi_j.shape[0], 1))

    epsilon_j = (np.subtract(np.transpose(d_temp),np.matmul(np.transpose(phi_j),Th_j_1))) /(m**2)


    nominator = np.matmul(np.matmul(P_j_1, phi_j), np.matmul(np.transpose(phi_j), P_j_1))
    denominator = m ** 2 + np.matmul(np.matmul(np.transpose(phi_j), P_j_1), phi_j)

    P_j = np.subtract(P_j_1, nominator / denominator)

    t = np.dot(P_j, phi_j)
    t = np.reshape(t, (t.shape[0], 1))
    test = np.kron(t, epsilon_j)

    Th_j = np.add(Th_j_1, test)
    return Th_j, P_j, epsilon_j, nominator, denominator

def output_data(filename, nom_loads, Q_pros, pros_all): #checks the violation of the Sum_x_i =/< Wi constraint
    pros_all_last_round = pros_all[-1]
    Wi_all = []
    Sum_xi_all = []
    for i in range(0, len(nom_loads)):
        Wi = sum(nom_loads[i]) - Q_pros[i]
        Sum_xi = sum(pros_all_last_round[i][1])

        Wi_all.append(Wi)
        Sum_xi_all.append(Sum_xi)

    diff = [x-y for x,y in zip(Sum_xi_all, Wi_all)]
    df = pandas.DataFrame()
    df['Wi'] = Wi_all
    df['Sum_xi'] = Sum_xi_all
    df['Diff'] = diff
    df.to_csv(filename)
    return

def normalize_training_data(training_p_random_for_i_day, training_d_star_for_i_day):
    var_p_rand = np.var(training_p_random_for_i_day)
    std_p_rand = np.sqrt(var_p_rand)
    var_d_star = np.var(training_d_star_for_i_day)
    std_d_star = np.sqrt(var_d_star)

    training_p_random_for_i_day_array = np.array(training_p_random_for_i_day)
    training_d_star_for_i_day_array = np.array(training_d_star_for_i_day)

    mean_p_rand = np.mean(training_p_random_for_i_day_array, axis=0)
    mean_d_star = np.mean(training_d_star_for_i_day_array, axis=0)

    normalized_training_p_random_for_i_day = [(x-mean_p_rand)/std_p_rand for x in training_p_random_for_i_day]
    normalized_training_d_star_for_i_day = [(x-mean_d_star)/std_d_star for x in training_d_star_for_i_day]

    return normalized_training_p_random_for_i_day, normalized_training_d_star_for_i_day

# ------
def solve_agg_from_least_squares(lambda_prices, pmax, coeff, intercept, nom_loads, T):
    d0 = np.sum(nom_loads, axis=1)

    lambda_prices = np.asarray(lambda_prices)
    d0 = np.asarray(d0)
    d0 = np.reshape(d0, (d0.shape[0], 1))

    bounds = Bounds(np.zeros(T), np.ones(T)*pmax)
    p0 = np.random.uniform(0, pmax, T)

    res = minimize(ga_with_theta, p0, bounds=bounds, args=(lambda_prices, d0, coeff, intercept, T), method='SLSQP', options={'disp': False}) #True
    return res

def solve_agg_from_least_squares_2(lambda_prices, pmax, coeff, intercept, nom_loads, T):
    d0 = np.sum(nom_loads, axis=1)

    lambda_prices = np.asarray(lambda_prices)
    d0 = np.asarray(d0)
    d0 = np.reshape(d0, (d0.shape[0], 1))

    bounds = Bounds(-np.ones(T)*pmax, np.ones(T)*pmax)
    p0 = np.random.uniform(0, pmax, T)

    res = minimize(ga_with_theta, p0, bounds=bounds, args=(lambda_prices, d0, coeff, intercept, T), method='SLSQP', options={'disp': True}) #True
    return res

def ga_with_theta(p, lambda_prices, d0, coeff, intercept, T):
    lambda_prices = lambda_prices.tolist()
    d0 = d0.tolist()
    d0 = [item for sublist in d0 for item in sublist]
    obj = -np.inner(lambda_prices-p, d0 - np.inner(p, coeff) - intercept)
    return obj

def agg_function_appr(lambda_prices, nom_loads, coeff, intercept, T, pmax):
    p = cp.Variable(T)

    d0 = np.sum(nom_loads, axis=1)
    lambda_prices = np.asarray(lambda_prices)

    A1 = np.identity(T)
    A2 = -A1
    A3 = np.concatenate((A1, A2), axis=0)

    b = np.ones((T,))
    b1 = [x*pmax for x in b]
    b2 = b*0
    b3 = np.concatenate((b1, b2), axis=0)

    # Lets see if it works that way
    #coeff = -np.identity(T)

    prob = cp.Problem(cp.Maximize(cp.quad_form(p, coeff) - p.T @ (d0 - intercept + np.inner(np.transpose(coeff),lambda_prices)) + lambda_prices.T @ (d0 - intercept)), [A3 @ p <= b3])
    # Try without psd_wrap(coeff)
    # check the inner product with transpose
    # change solver

    #print(cp.installed_solvers())

    prob.solve(solver='CPLEX', verbose=False) # CVXOPT #XPRESS, CPLEX, ECOS, CVXOPT, SCS
    return [prob.value, p.value.tolist()]
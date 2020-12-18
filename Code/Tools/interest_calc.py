# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:33:41 2020

@author: Alden Porter
"""

# =============================================================================
# Import Libraries
# =============================================================================

import numpy as np
from numba import jit
import matplotlib.pyplot as plt

# =============================================================================
# Define Functions
# =============================================================================

def ar_1(mu, a, sigma, T, x_0):
    """
    This function computes a simulated ar1 process assuming  x_t = mu + a*x_{t-1} + e_t
    """
    x_path = np.zeros(T)
    x_path[0] = x_0
    shocks = np.random.normal(0,sigma,T) # The first term isn't used and will be ignored for sake of code readability
    # iteratively construct the AR1 according to x_t = mu + a*x_{t-1} + e_t
    for t in range(1,T):
        x_path[t] = mu + a*x_path[t-1] + shocks[t]
    
    return x_path # Return the path of the AR1

def censored_ar_1(mu, a, sigma, T, x_0):
    """
    This function computes a simulated ar1 process assuming  x_t = max(mu + a*x_{t-1} + e_t,0)
    """
    x_path = np.zeros(T)
    x_path[0] = x_0
    shocks = np.random.normal(0,sigma,T) # The first term isn't used and will be ignored for sake of code readability
    # iteratively construct the AR1 according to x_t = mu + a*x_{t-1} + e_t
    for t in range(1,T):
        x_path[t] = max(mu + a*x_path[t-1] + shocks[t], 0)
    
    return x_path # Return the path of the AR1



def compound_interest_rates(interest_path):
    '''
    This function takes in a path of portfolio returns (a Tx1 numpy array) and it returns a T+1xT+1
    numpy array. The returned array can be seen as a lower triangular matrix with ones on the diagonal and
    whose lower diagonal entries correspond to the product of the returns up to that index.
    '''
    T = len(interest_path) # The number of periods - 1 (because we exclued the intial period)
    CI = np.zeros([T+1,T+1]) # Initialize the matrix of compund interest paths in each period.

    # Loop over rows and columns and sub in the corresponding compound interest rates for the matrix multiplication
    for i in range(T+1):
        for j in range(T+1):
            if j < i:
                CI[i, j] = np.prod(interest_path[j:i])
            elif j == i:
                CI[i, j] = 1
            elif j > i:
                continue

    return CI

def asset_path(income_path, consumption_path, initial_savings, interest_path):
    """
    This fucntion computes the total amount you would have saved given a time series of interest rates
    given by interest_path and a time series of savings amounts given by savings path with the first index
    corresponding to the first time period. It computes the value of the asset at time T-1, the final index.
    Inputs:
        All inputs need to be Tx1 Numpy Arrays
    """
   
    T = len(income_path) # How many time periods?
    S = np.subtract(income_path, consumption_path) # Compute per period savings as per period income minus consumption
    S = np.insert(arr = S, obj = 0, values = initial_savings) # Technical trick, from a mathemtical perspective we can consider initial assets to simply be the savings in period 0.
    CI = compound_interest_rates(interest_path) # Convert the per period interest path to a compounded interest matrix per period
    
    A = np.dot(CI,S) #Final asset time series is just this dot product

    return A

#@jit(nopython = True) # This skips the python interpreter in favor of a more low level interpreter, helps speed up large scale simulations
def asset_monte_carlo(N, T, percentiles, initial_savings, inc_fn, cons_fn, int_fn):
    '''
    This function runs a monte-carlo simulation on the intrest rate, income and consumption stochastic processes    to obtain quantiles on asset values (Q_t) for each time period and the expected return
    '''

    sim_A = np.empty([N,T+1]) #Simulated assets form an NXT+1 matrix which  will be collapsed into T+1x1 vectors corresponding to the percentiles 
    
    #Randomly simulate asset paths according to the inc_fun, cons_fn and int_fn functions,
    #   then compile the simulated paths into a matrix
    for n in range(N):
        
        income_path = inc_fn()
        consumption_path = cons_fn()
        interest_path = int_fn()
        
        A_n = asset_path(income_path, consumption_path, initial_savings, interest_path)
        
        sim_A[n, :] = np.transpose(A_n) #Replace the row in teh simulations matrix with the asset path

    E_A = np.empty([T+1,1]) # The expected return
    Q_A = np.empty([T+1, len(percentiles)]) # The desired percentiles

    #This loop actually estimates the statistics based on the above simulation
    for t in range(T+1):
        E_A[t] = np.mean(sim_A[:, t]) #Take the mean portfolio value in time t
        
        # This little loops gets the percentiles at each time period
        for k, q in enumerate(percentiles):
            Q_A[t, k] = np.percentile(sim_A[:, t] , q)

    return Q_A, E_A

def main():
    '''
    This function runs the program
    '''
    # Set up simple set up
    simple_setup_bool = input('Do you want easy setup? [y/n] \n')
    
    # Run simple setup
    if simple_setup_bool.lower().strip() != 'n': 
        
        T = int(input('How many periods? i.e. T = '))
        C_bar = float(input('How much is your consumption budget in each period? i.e. C_bar = '))
        Y_bar = float(input('How much are you earning each period in outside income? i.e. Y_bar = '))
        r_bar = float(input("What's your average return? i.e. r_bar = "))
        a_0 = float(input('What are you intital assets? i.e. a_0 = '))
        sigma_bar = input("What's the variance on your per period return? (press enter if you don't know, or enter 0) i.e. sigma_bar = ")
        if sigma_bar.strip() == '':
            sigma_bar = r_bar/((2*3.14159))
        else:
            sigma_bar = float(sigma_bar)
    
    N = 2000
    percentiles = [1, 10, 25, 75, 90, 99]

    ci_color = 'mediumturquoise'
    
    # ========================= Run the simulation
    inc_sim = lambda : ar_1(0, 1, 0 , T, Y_bar)
    con_sim = lambda : ar_1(0, 1, 0, T, C_bar)
    int_sim = lambda : ar_1(0, 1, sigma_bar, T, r_bar)

    Q, E = asset_monte_carlo(N, T, percentiles, a_0, inc_sim, con_sim, int_sim)

    # ========================= Plot the pretty figure

    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.plot(E, color = 'black', linewidth = 1, linestyle = 'dotted')

    # Shade confidence bands
    ax1.fill_between(list(range(T+1)), Q[:, 0], Q[:, 5], facecolor = ci_color, alpha = .1) # 1-99, thinnest
    ax1.fill_between(list(range(T+1)), Q[:, 1], Q[:, 4], facecolor = ci_color, alpha = .3) # 10-90
    ax1.fill_between(list(range(T+1)), Q[:, 2], Q[:, 3], facecolor = ci_color, alpha = .5) # 25-75, thickest

    ax1.set_ylabel('Total Savings')
    ax1.set_xlabel('Time')
    ax1.set_title('Possible Savings Paths, Expected Savings and Confidence Intervals')
    plt.show()       

# =============================================================================
# Run Code
# =============================================================================

main()

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp
import math
import random
import scipy.integrate as integrate
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.signal import csd
from sympy import *
from sympy import symbols
from sympy import integrate

import numpy as np
from scipy.optimize import curve_fit


# In[ ]:


# Quadratic function with more precise iteration process

def quadratic_function_iteration(x, y, x_err, y_err, tol=1e-8, max_iter=300):
    def quadratic_func_iteration(x, a, b, c):
        return a * x**2 + b * x + c

    # Initial parameter estimates
    params = np.array([1, 1, 1])
    prev_params = np.inf * np.ones_like(params)  # Placeholder for previous parameter values

    for i in range(max_iter):
        # Predicted values with current estimates
        y_pred = quadratic_func_iteration(x, *params)
        
        # Use Poisson uncertainties as Ibata did in his paper
        sigma_poisson = np.sqrt(np.clip(y_pred, 0, None))
        # Total sigma (includes both Poisson uncertainty and propagated x_err)
        sigma_total = np.sqrt(sigma_poisson**2 + ((2 * params[0] * x + params[1])**2) * x_err**2)
        
        # Fit the quadratic function with updated sigma values
        params, covariance = curve_fit(quadratic_func, x, y, sigma=sigma_total, absolute_sigma=True)
        
        # Stop if parameter changes are below tolerance
        if np.all(np.abs(params - prev_params) < tol):
            print(f"Converged after {i+1} iterations.")
            break

        # Update previous parameters for the next iteration
        prev_params = params

    # Final values
    quadratic_fit = quadratic_func_iteration(x, *params)

    return params, quadratic_fit


# In[ ]:


# Poisson uncertainties
# Scipy curve fit uses the Levenberg-Marquardt-Verfahren

def quadratic_function(x, y, x_err, y_err):
    def quadratic_func(x, a, b, c):
        return a * x**2 + b * x + c

    # Initial parameter estimates
    params = np.array([1, 1, 1])

    # Iterative fitting process
    for _ in range(10):
        # Predicted values by estimates
        y_pred = quadratic_func(x, *params)
        
        # We use Poisson-uncertainties, as Ibata did in his paper
        # sigma_poisson = np.sqrt(y_pred)  # Poisson-Fehler ist sqrt(y_pred)
        sigma_poisson = np.sqrt(np.clip(y_pred, 0, None))
        sigma_total = np.sqrt(sigma_poisson**2 + ((2 * params[0] * x + params[1])**2) * x_err**2)
        params, covariance = curve_fit(quadratic_func, x, y, sigma=sigma_total, absolute_sigma=True)

    # Values after adjusting
    quadratic_fit = quadratic_func(x, *params)

    return params, quadratic_fit


# In[ ]:

def quintic_function(x, y, x_err, y_err):
    def quintic_func(x, a, b, c, d, e, f):
        return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f

    # Initial parameter estimates
    params = np.array([1, 1, 1, 1, 1, 1])

    # Iterative fitting process
    for _ in range(10):
        # Predicted values by estimates
        y_pred = quintic_func(x, *params)
        
        # We use Poisson-uncertainties, as Ibata did in his paper
        sigma_poisson = np.sqrt(np.clip(y_pred, 0, None))
        sigma_total = np.sqrt(sigma_poisson**2 + ((5 * params[0] * x**4 + 4 * params[1] * x**3 +
                                                   3 * params[2] * x**2 + 2 * params[3] * x +
                                                   params[4])**2) * x_err**2)
        params, covariance = curve_fit(quintic_func, x, y, sigma=sigma_total, absolute_sigma=True)

    # Values after adjusting
    quintic_fit = quintic_func(x, *params)

    return params, quintic_fit



# Gauss uncertainties

def quadratic_function_gauss(x, y, x_err, y_err):
    def quadratic_func_gauss(x, a, b, c):
        return a * x**2 + b * x + c

    # Initial parameter estimates
    params = np.array([1, 1, 1])  

    # Iterative fitting process with Gauss uncertainties
    for _ in range(10):  
        # y_pred = quadratic_func(x, *params)
        sigma_quadratic = np.sqrt((y_err**2) + ((2 * params[0] * x + params[1])**2) * x_err**2)
        params, covariance = curve_fit(quadratic_func_gauss, x, y, sigma=sigma_quadratic, absolute_sigma=True)

    # Predicted values
    quadratic_fit_gauss = quadratic_func_gauss(x, *params)
    
    return params, quadratic_fit_gauss


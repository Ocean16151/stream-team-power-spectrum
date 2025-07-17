#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp
import math
import random
import scipy.integrate as integrate
from scipy.integrate import quad
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from scipy.signal import csd
from sympy import *
from sympy import symbols
from sympy import integrate

import numpy as np
from scipy.optimize import curve_fit

def calculate_D_galactocentric(tr, rsun=[-8,0,0]):
    
    # Extract the galactocentric coordinates
    x = tr[:, -1, 1]
    y = tr[:, -1, 2]
    z = tr[:, -1, 3]

    # Calculate relative coordinates using numpy broadcasting
    xrel = x - rsun[0]
    yrel = y - rsun[1]
    zrel = z - rsun[2]
    
    D = np.sqrt(xrel**2 + yrel**2 + zrel**2)

    # Return D
    return D

def integrand_galactocentric(phi_f_value, a5, a4, a3, a2, a1, a0):
    phi_f = symbols('phi_f')
    
    # Polynomial with coefficients passed as arguments
    D = a5*(phi_f**5) + a4*(phi_f**4) + a3*(phi_f**3) + a2*(phi_f**2) + a1*(phi_f) + a0

    # Derivative of the polynomial
    D_derivative = diff(D, phi_f)

    # Calculate the integrand
    integrand_squared = D**2 + D_derivative**2
    integrand_non_numerical = sqrt(integrand_squared)
    
    # Create the numerical function of the integrand
    integrand_function = lambdify(phi_f, integrand_non_numerical, 'numpy')
    integrand_result = integrand_function(phi_f_value)

    # Return the result
    return integrand_result


def integrand_differential_equation_galactocentric(phi_f, y, a5, a4, a3, a2, a1, a0):
    return integrand_galactocentric(phi_f, a5, a4, a3, a2, a1, a0)


def perform_integration_galactocentric(random_phi, a5, a4, a3, a2, a1, a0):
    # Array for results
    integral_results = np.zeros(len(random_phi))

    # Integration for each phi_f
    for i, phi_f_end in enumerate(random_phi):
        result = solve_ivp(
            integrand_differential_equation_galactocentric,  # Differential equation
            [0, phi_f_end],                   # Integration range
            [0],                               # Initial value for y
            t_eval=[phi_f_end],               # Evaluation at phi_f_end
            args=(a5, a4, a3, a2, a1, a0)     # Passing the coefficients
        )
        integral_results[i] = result.y[0][-1]

    return integral_results


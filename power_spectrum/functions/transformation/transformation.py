#!/usr/bin/env python
# coding: utf-8


import numpy as np
from scipy.integrate import solve_ivp
from sympy import symbols, diff, sqrt, lambdify


# This script contains functions to convert from the Koposov coordinate phi_1 to the physical length along the stream. 
# The coefficients (see function integrand) are specifically tailored to the actual stellar positions of the GD-1 stream, 
# as studied in Ibata et al. (2019)



def integrand(phi_f_value):

    """
    This function computes the integrand sqrt(D(phi)^2 + D'(phi)^2), where D (= heliocentric distance) is a 5th-degree polynomial.
    The coefficients are describing the shape of the heliocentric distance of the stars as function of their angular position along the stream for the stream data of Ibata et al.
    
    Parameters:
        phi_f_value (float or array-like): The evaluation point(s) for the integrand.
    
    Returns:
        float or array: The value of the integrand at the given point(s).
    """
    
    phi_f = symbols('phi_f')
    
    # Polynom
    # Coefficients
    a5, a4, a3, a2, a1, a0 = -4.302, -11.54, -7.161, 5.985, 8.595, 10.36

    # Function
    D = a5*(phi_f**5) + a4*(phi_f**4) + a3*(phi_f**3) + a2*(phi_f**2) + a1*(phi_f) + a0

    # Calculate derivative
    D_derivative = diff(D, phi_f)

    # Calculate integrand
    integrand_squared = D**2 + D_derivative**2
    integrand_non_numerical = sqrt(integrand_squared)
    
    integrand_function = lambdify(phi_f, integrand_non_numerical, 'numpy')
    integrand_result = integrand_function(phi_f_value)

    # Result
    return integrand_result



def integrand_differential_equation(phi_f, y):
    
    """
    This is a so-called "wrapper function" for use in ode solvers. It returns the value of the integrand for a given phi_f.
    
    Parameters:
        phi_f (float): Current integration variable.
        y (float): Dummy argument required by solve_ivp (not used).
    
    Returns:
        float: Value of the integrand at phi_f.
    """
    
    return integrand(phi_f)



def perform_integration(random_phi):

    """
    This function performs the numerical integration of the integrand from 0 to each value in random_phi,
    using solve_ivp.
    
    Parameters:
        random_phi (array-like): Array of phi_f upper limits for integration.
    
    Returns:
        np.ndarray: Array of integral results corresponding to each phi_f upper limit.
    """

    # Array for results
    integral_results = np.zeros(len(random_phi))

    # Integrate 
    for i, phi_f_end in enumerate(random_phi):
        result = solve_ivp(integrand_differential_equation, [0, phi_f_end], [0], t_eval=[phi_f_end])
        integral_results[i] = result.y[0][-1]

    return integral_results






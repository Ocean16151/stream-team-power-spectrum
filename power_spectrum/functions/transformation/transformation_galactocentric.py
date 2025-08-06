#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sympy import symbols, diff, sqrt, lambdify
from scipy.integrate import solve_ivp


# This script contains functions to convert from the Koposov coordinate phi_1 to the physical length s along the stream. 
# The positions of the stars of the object (here: the GD-1 stream) can be extracted from an array (here: tr, from the stream simulation code).


def calculate_D_galactocentric(tr, rsun=[-8,0,0]):
    """
    This function computes the heliocentric distance D between the Sun and the considered objects (in this case: the stars of the GD-1 stream).
    It is specifically tailored to the array tr from the stream simulation code (Part 1), as described in the Master thesis.

    Parameters:
        tr (ndarray): Trajectory of the stars of GD-1 (output of the stream simulation code)
        rsun (list): Cartesian coordinates of the Sun in kpc, set to: [-8, 0, 0]

    Returns:
        ndarray: Array of distances D for each particle at the final timestep.
    """

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
    """
    This function computes the integrand sqrt(D² + (dD/dφ)²) for a given φ, where D(φ) is a 5th-degree polynomial.

    Parameters:
        phi_f_value (float or ndarray): Value(s) of φ for which the integrand is evaluated.
        a5, a4, a3, a2, a1, a0 (float): Polynomial coefficients of D(φ).

    Returns:
        float or ndarray: Value(s) of the integrand.
    """

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
    """
    This is again a so-called "wrapper function" to pass the integrand into solve_ivp, representing dy/dφ = sqrt(D² + (dD/dφ)²).

    Parameters:
        phi_f (float): Current value of φ.
        y (float): Current value of the integrated function (unused internally).
        a5, a4, a3, a2, a1, a0 (float): Polynomial coefficients of D(φ).

    Returns:
        float: Value of the differential equation at φ.
    """

    return integrand_galactocentric(phi_f, a5, a4, a3, a2, a1, a0)


def perform_integration_galactocentric(random_phi, a5, a4, a3, a2, a1, a0):
    """
    This function performs the numerical integration of dy/dφ = sqrt(D² + (dD/dφ)²) from 0 to φ for each value in random_phi.

    Parameters:
        random_phi (ndarray): Array of φ values to integrate up to.
        a5, a4, a3, a2, a1, a0 (float): Polynomial coefficients of D(φ).

    Returns:
        ndarray: Array of integrated values corresponding to each φ in random_phi.
    """

    # Array for results
    integral_results = np.zeros(len(random_phi))

    # Integration for each phi_f
    for i, phi_f_end in enumerate(random_phi):
        result = solve_ivp(
            integrand_differential_equation_galactocentric,  # Differential equation
            [0, phi_f_end],                   # Integration range
            [0],                              # Initial value for y
            t_eval=[phi_f_end],               # Evaluation at phi_f_end
            args=(a5, a4, a3, a2, a1, a0)     # Passing the coefficients
        )
        integral_results[i] = result.y[0][-1]

    return integral_results


#!/usr/bin/env python
# coding: utf-8


import numpy as np
from scipy.optimize import curve_fit


# This script contains polynomial fitting functions of various degrees, where the uncertainties can be chosen to be Gaussian or Poisson.


# Quadratic function with more precise iteration process
def quadratic_function_iteration(x, y, x_err, y_err, tol=1e-8, max_iter=300):
    """
    This function performs an iterative fit of a quadratic function y = ax^2 + bx + c using Poisson-based uncertainties.
    It stops when the parameter changes are below the tolerance.

    Parameters:
        x (array): Independent variable.
        y (array): Dependent variable.
        x_err (array): Errors in x values.
        y_err (array): (Unused)
        tol (float): Convergence threshold.
        max_iter (int): Maximum number of iterations.

    Returns:
        params (array): Best-fit parameters [a, b, c].
        quadratic_fit (array): Fitted y values.
    """

    def quadratic_func_iteration(x, a, b, c):
        return a * x**2 + b * x + c

    params = np.array([1, 1, 1])
    prev_params = np.inf * np.ones_like(params)

    for i in range(max_iter):
        y_pred = quadratic_func_iteration(x, *params)
        sigma_poisson = np.sqrt(np.clip(y_pred, 0, None))
        sigma_total = np.sqrt(sigma_poisson**2 + ((2 * params[0] * x + params[1])**2) * x_err**2)

        params, covariance = curve_fit(quadratic_func_iteration, x, y, sigma=sigma_total, absolute_sigma=True)

        if np.all(np.abs(params - prev_params) < tol):
            print(f"Converged after {i+1} iterations.")
            break

        prev_params = params

    quadratic_fit = quadratic_func_iteration(x, *params)
    return params, quadratic_fit


def quadratic_function(x, y, x_err, y_err):
    """
    This function performs an iterative fit of a quadratic function y = ax^2 + bx + c using Poisson-based uncertainties.
    It runs a fixed number of iterations.

    Parameters:
        x (array): Independent variable.
        y (array): Dependent variable.
        x_err (array): Errors in x values.
        y_err (array): (Unused)

    Returns:
        params (array): Best-fit parameters [a, b, c].
        quadratic_fit (array): Fitted y values.
    """

    def quadratic_func(x, a, b, c):
        return a * x**2 + b * x + c

    params = np.array([1, 1, 1])

    for _ in range(10):
        y_pred = quadratic_func(x, *params)
        sigma_poisson = np.sqrt(np.clip(y_pred, 0, None))
        sigma_total = np.sqrt(sigma_poisson**2 + ((2 * params[0] * x + params[1])**2) * x_err**2)

        params, covariance = curve_fit(quadratic_func, x, y, sigma=sigma_total, absolute_sigma=True)

    quadratic_fit = quadratic_func(x, *params)
    return params, quadratic_fit


def quintic_function(x, y, x_err, y_err):
    """
    This function fits a 5th-degree polynomial y = ax^5 + bx^4 + cx^3 + dx^2 + ex + f, using Poisson-based uncertainties.

    Parameters:
        x (array): Independent variable.
        y (array): Dependent variable.
        x_err (array): Errors in x values.
        y_err (array): (Unused)

    Returns:
        params (array): Best-fit parameters [a, b, c, d, e, f].
        quintic_fit (array): Fitted y values.
    """

    def quintic_func(x, a, b, c, d, e, f):
        return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f

    params = np.array([1, 1, 1, 1, 1, 1])

    for _ in range(10):
        y_pred = quintic_func(x, *params)
        derivative = (5 * params[0] * x**4 + 4 * params[1] * x**3 +
                      3 * params[2] * x**2 + 2 * params[3] * x + params[4])
        sigma_poisson = np.sqrt(np.clip(y_pred, 0, None))
        sigma_total = np.sqrt(sigma_poisson**2 + (derivative**2) * x_err**2)

        params, covariance = curve_fit(quintic_func, x, y, sigma=sigma_total, absolute_sigma=True)

    quintic_fit = quintic_func(x, *params)
    return params, quintic_fit


def quadratic_function_gauss(x, y, x_err, y_err):
    """
    This function performs an iterative fit of a quadratic function y = ax^2 + bx + c using Gaussian uncertainties in x and y.

    Parameters:
        x (array): Independent variable.
        y (array): Dependent variable.
        x_err (array): Errors in x values.
        y_err (array): Errors in y values.

    Returns:
        params (array): Best-fit parameters [a, b, c].
        quadratic_fit_gauss (array): Fitted y values.
    """

    def quadratic_func_gauss(x, a, b, c):
        return a * x**2 + b * x + c

    params = np.array([1, 1, 1])

    for _ in range(10):
        sigma_total = np.sqrt(y_err**2 + ((2 * params[0] * x + params[1])**2) * x_err**2)
        params, covariance = curve_fit(quadratic_func_gauss, x, y, sigma=sigma_total, absolute_sigma=True)

    quadratic_fit_gauss = quadratic_func_gauss(x, *params)
    return params, quadratic_fit_gauss


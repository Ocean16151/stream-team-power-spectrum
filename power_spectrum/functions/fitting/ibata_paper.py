#!/usr/bin/env python
# coding: utf-8


import numpy as np


# This function(s) is (are) specifically tailored to the binning of the counts along the Koposov coordinate for the data from the paper by Ibata et al. 2019.


def ibata_fitting(phi1):

    """
    This function computes the histogram fit function used in Ibata et al. for the number of stars 
    as a function of the angle φ₁ (in degrees).

    The formula is a quadratic function based on the paper:
        C(φ₁) = -37.51 * φ₁² - 46.51 * φ₁ + 14.37

    Parameters:
        phi1 (float or array-like): Angle(s) in degrees.

    Returns:
        float or ndarray: The computed value(s) of the fitting function.
    """
    
    phi1 = phi1*np.pi/180
    C = -37.51*phi1**2 - 46.51*phi1+14.37
    return C


def ibata_fitting_modified(phi1):

    """
    This function computes a modified version of the Ibata et al. fit function by dividing the original 
    result by 1.25.

    The formula is:
        C(φ₁) = ( -37.51 * φ₁² - 46.51 * φ₁ + 14.37 ) / 1.25

    Parameters:
        phi1 (float or array-like): Angle(s) in degrees.

    Returns:
        float or ndarray: The modified fit values.
    """
    
    phi1 = phi1*np.pi/180
    C = -37.51*phi1**2 - 46.51*phi1+14.37
    C = C/1.25
    return C


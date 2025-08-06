#!/usr/bin/env python
# coding: utf-8


import numpy as np
import random


def binning(x, number_of_bins, minimum, maximum):

    """
    This function bins the data in the array `x` into a specified number of equally spaced bins 
    between `minimum` and `maximum`. Returns both the bin counts and the bin centers.

    Parameters:
        x (array-like): The data to be binned.
        number_of_bins (int): The number of bins to divide the data into.
        minimum (float): The lower bound of the binning range.
        maximum (float): The upper bound of the binning range.

    Returns:
        bin_counts (ndarray): Number of data points in each bin.
        bin_centers (ndarray): Center value of each bin.
        
    """
    bin_counts, bin_edges = np.histogram(x, bins=number_of_bins, range=(minimum, maximum))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_counts, bin_centers


def calculate_bin_size_array(x, number_of_bins):

    """
    This function calculates the bin size for a dataset based on its minimum and maximum values 
    and the specified number of bins.

    Parameters:
        x (array-like): The input data array.
        number_of_bins (int): The number of bins to divide the data range into.

    Returns:
        float: The size (width) of each bin.
    """
    
    # Reads in the limits of an array, a number of bins can be chosen and the bin size will be calculated
    bin_size = (max(x) - min(x)) / (number_of_bins)
       
    return bin_size


def calculate_bin_size_bounds(minimum, maximum, number_of_bins):

    """
    This function calculates the bin size given explicit minimum and maximum bounds and the number of bins.

    Parameters:
        minimum (float): The lower bound of the binning range.
        maximum (float): The upper bound of the binning range.
        number_of_bins (int): The number of bins to divide the range into.

    Returns:
        float: The size (width) of each bin.
    """
    
    # Calculates the bin size, if a mininum (not mininum bin), maximum and the number of bins are given
    bin_size = (maximum - minimum) / (number_of_bins)
    
    return bin_size


def calculate_bin_size_bins(minimum, maximum, number_of_bins):

    """
    This function calculates the bin size given minimum and maximum bin midpoints and the number of bins.

    Parameters:
        minimum (float): The midpoint of the first bin.
        maximum (float): The midpoint of the last bin.
        number_of_bins (int): The total number of bins.

    Returns:
        float: The bin size (distance between bin midpoints).
    """
    
    
    # Calculates the bin size, if a mininum and maximum bin (midpoints of the bins) and the number of bins are given
    bin_size = (maximum - minimum) / (number_of_bins - 1)
    
    return bin_size


def create_binning(bin_min, bin_size, number_of_bins):

    """
    This function creates a list of bin midpoints starting from bin_min with a given bin size.

    Parameters:
        bin_min (float): The starting point (midpoint of the first bin).
        bin_size (float): The distance between consecutive bin midpoints.
        number_of_bins (int): The total number of bins.

    Returns:
        list of float: List containing the bin midpoints.
    """
    
    # Creates a binning
    binning = []
    binning.append(bin_min)
    num_bins = number_of_bins - 1

    binning_i = bin_min + bin_size
    for i in range(num_bins):
        binning_i += bin_size
        binning.append(binning_i)
    
    return binning


def binning_bounds(bins, bin_size):
    
    """
    Calculates the lower and upper bounds of bins given their midpoints and bin size.

    Parameters:
    bins (list or array-like): List of bin midpoints.
    bin_size (float): Width of each bin.

    Returns:
    tuple: Two lists containing the lower bounds and upper bounds of the bins.
    """
    
    # Calculate upper and lower bounds for the bins
    binning_min = [bin_value - bin_size / 2 for bin_value in bins]
    binning_max = [bin_value + bin_size / 2 for bin_value in bins]
    
    return binning_min, binning_max


def bin_data(phi_sim, bin_min, bin_max, number_of_bins):

    """
    Counts how many values from the array phi_sim fall within each bin defined by 
    the lower bounds (bin_min) and upper bounds (bin_max).

    Parameters:
    -----------
    phi_sim : array-like
        Array of data points to be binned.
    bin_min : list or array-like
        List of lower bounds of the bins.
    bin_max : list or array-like
        List of upper bounds of the bins.
    number_of_bins : int
        Number of bins to consider.

    Returns:
    --------
    phi_counts : list of int
        List containing counts of data points in each bin.
    """
    
    phi_counts = []  
    size_phi_sim = len(phi_sim)
    
    for i in range(number_of_bins):
        counts_i = 0  
        
        # Check if a value lies within a bin
        for j in range(size_phi_sim):
            if bin_min[i] < phi_sim[j] < bin_max[i]:
                counts_i += 1
        
        # Append list
        phi_counts.append(counts_i)
    
    return phi_counts


def random_values(counts_phi, binning, bin_size):

    """
    Generates a list of random angular values (in radians) distributed within bins.

    For each bin, a number of random values proportional to the count in that bin
    are generated uniformly within the bin boundaries defined by binning and bin_size.
    The values are converted from degrees to radians before returning.

    Parameters:
    counts_phi (list or array): Number of counts per bin.
    binning (list or array): Bin center positions in degrees.
    bin_size (float): Width of each bin in degrees.

    Returns:
    list: Random values in radians distributed according to counts_phi.
    """

    random_phi = []
    
    for j in range(len(counts_phi)):
        counts_bin = int(round(counts_phi[j], 0))  
        for i in range(counts_bin):
            random_number = random.uniform(binning[j] - bin_size / 2, binning[j] + bin_size / 2)
            random_number_rad = random_number * (np.pi / 180)  
            random_phi.append(random_number_rad)
    
    return random_phi 


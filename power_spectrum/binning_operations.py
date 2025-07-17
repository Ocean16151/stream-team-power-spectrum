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

def binning(x, number_of_bins, minimum, maximum):
    bin_counts, bin_edges = np.histogram(x, bins=number_of_bins, range=(minimum, maximum))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_counts, bin_centers


def calculate_bin_size_array(x, bin_number):
    
    # Reads in the limits of an array, a number of bins can be chosen and the bin size will be calculated
    bin_size = (max(x) - min(x)) / (bin_number)
       
    return bin_size


# In[ ]:


def calculate_bin_size_bounds(minimum, maximum, bin_number):
    
    # Calculates the bin size, if a mininum, maximum and the number of bins are given
    bin_size = (maximum - minimum) / (bin_number)
    
    return bin_size


# In[ ]:


def create_binning(bin_min, bin_size, bin_number):
    
    # Creates a binning
    binning = []
    binning.append(bin_min)
    num_bins = bin_number - 1

    binning_i = bin_min + bin_size
    for i in range(num_bins):
        binning_i += bin_size
        binning.append(binning_i)
    
    return binning


# In[ ]:


def binning_bounds(bins, bin_size):
    
    # Calculate upper and lower bounds for the bins
    binning_min = [bin_value - bin_size / 2 for bin_value in bins]
    binning_max = [bin_value + bin_size / 2 for bin_value in bins]
    
    return binning_min, binning_max


# In[ ]:


def bin_data(phi_sim, bin_min, bin_max, bin_number):
    
    phi_counts = []  
    size_phi_sim = len(phi_sim)
    
    for i in range(bin_number):
        counts_i = 0  
        
        # Check if a value lies within a bin
        for j in range(size_phi_sim):
            if bin_min[i] < phi_sim[j] < bin_max[i]:
                counts_i += 1
        
        # Append list
        phi_counts.append(counts_i)
    
    return phi_counts


# In[ ]:


def random_values(counts_phi, binning, bin_size):
    random_phi = []
    
    for j in range(len(counts_phi)):
        counts_bin = int(round(counts_phi[j], 0))  
        for i in range(counts_bin):
            random_number = random.uniform(binning[j] - bin_size / 2, binning[j] + bin_size / 2)
            random_number_rad = random_number * (np.pi / 180)  
            random_phi.append(random_number_rad)
    
    return random_phi 


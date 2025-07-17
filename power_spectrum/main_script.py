#!/usr/bin/env python
# coding: utf-8

# # Power spectrum

# ## As a function of $\phi_1$

# ### Import data

# In[1]:


import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import scipy.integrate as integrate
from scipy.integrate import quad
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from scipy.signal import csd
import pandas as pd
from sympy import *
from sympy import symbols
from sympy import integrate
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogFormatter
from IPython.display import display, Image
import io
import os
# from mpi4py import MPI

import spectrum
import stream


# In[2]:


# Retrieve num_data from environment variable in bash-script
num_data = int(os.getenv("num_data", "0"))  # Default = 0


# In[3]:


# num_data = 1


# In[4]:


# Chose subhalo profile
subhalo_profile = 'DSIDM553_sph'


# In[5]:


# Import data
simulation_data = pd.read_csv(f'../results/{subhalo_profile}/data_output_power_section{num_data}.csv')
phi_sim = simulation_data['x'].values


# In[6]:


# Import data from the Ibata paper for the binning
data = pd.read_csv('../data/ibata/Ibata_Hist_Gaia.csv')
# count = data['y'].values
phi = data['x'].values


# ### Bin the data

# In[7]:


# As default we use the binning of Ibata for both phi and s
# Set number of bins
bin_number = 40  

# Calculate bin size
# x = np.array(phi_sim)
# bin_size = spectrum.calculate_bin_size_array(x, bin_number)

# print(bin_size)


# In[8]:


# Create binning
bin_min = min(phi)
bin_size = 2.5
binning = spectrum.create_binning(bin_min, bin_size, bin_number)
# print(binning)


# In[9]:


# Calculate bounds of the bins
binning_min, binning_max = spectrum.binning_bounds(binning, bin_size)
# print(binning_min), print(binning_max)


# In[10]:


# Bin the data
counts_phi = spectrum.bin_data(phi_sim, binning_min, binning_max, bin_number)
# print(counts_phi)


# ### Calculate a fitting function

# In[11]:


# Set the error
x = np.array(binning)
y = np.array(counts_phi)
x_err = np.full_like(x, 1.0)  # Uncertainty for x
y_err = np.sqrt(y) # Uncertainty for y
# x_err = 1.0
# y_err = 1.0


# In[12]:


# Import fitting function

params, quadratic_fit = spectrum.quadratic_function(x, y, x_err, y_err)


# In[13]:


# Plot results
# plt.scatter(x, y, label='Data')
# plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='o', label='Data with error bars')
# plt.plot(x, quadratic_fit, color='red', label='Quadratic Fit')
# plt.xlabel(r'$\phi_1$ [deg]')
# plt.ylabel('Counts')
# plt.legend()
# plt.show()

# Print the parameters
# print("Parameters a, b, c:", params)


# ### Plot histogram

# In[14]:


# Plot
plt.figure(figsize=(13,6))
plt.bar(binning, counts_phi,width=2, color='r',alpha=0.5, label = 'Data from simulate examples')
plt.plot(binning, quadratic_fit,label='Fitting Function')
plt.title(f'Density contrast $\phi_1$ (Subhalo profile: {subhalo_profile}, Data section: {num_data})')
# plt.plot(phi,fit_not_div,label='Fitting Function Ibata')
plt.xlim([20,-100])
plt.ylabel('Counts',size=15)
plt.xlabel(r'$\phi_1$ [deg]',size=15)
plt.legend()

# Save and show
plt.savefig(f'../results/{subhalo_profile}/density_contrast_phi1_section{num_data}_gc.pdf')
plt.show()


# ### Calculate power spectrum

# In[15]:


# Calculate density contrast
dens_fit = counts_phi/quadratic_fit


# For the calculation of the power spectrum we use the CSD-algorithm by SciPy (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.csd.html) and set:
# 
# - *fs* = sympling frequency (1 divided by bin-size)
# - *nperseg* = length of each segment (len(phi) = 40)

# In[16]:


f_fit, pxx_fit = csd(dens_fit,dens_fit,fs=1/bin_size,scaling='spectrum',nperseg=bin_number)


# In[17]:


# Calculate power
power = 1/f_fit
# print(power)


# ### Plot and save power spectrum

# In[18]:


# Save values for band plot
x = power  
y = 1 * np.sqrt(power[1] * np.array(pxx_fit))  

# Create data frame and save
data = pd.DataFrame({'x': x, 'y': y})
output_path = f'../results/{subhalo_profile}/power_spectrum_phi1_data_section{num_data}_gc.csv'
data.to_csv(output_path, index=False)


# In[19]:


# print(x)
# print(y)


# In[20]:


# Plot power spectrum 
### Plot 1/frequency and take the square root of the power * (phi[-1] - phi[0] = 97.5)
plt.figure(figsize=(12,10))
plt.loglog(power, 1*np.sqrt((power[1])*np.array(pxx_fit)),label='Power Spectrum')

plt.ylabel(r'$\sqrt{P_{\rho\rho}}$',size=15)
plt.xlabel(r'$1/k_{\phi_1}$ [deg]',size=15)
plt.gca().xaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=True))
plt.legend()
plt.title(f'$Subhalo$ $profile:$ {subhalo_profile}, $Data$ $section:$ {num_data}')

# Save and show
plt.savefig(f'../results/{subhalo_profile}/power_spectrum_phi1_section{num_data}_gc.pdf')
plt.show()


# ## As a function of s

# ### Use galactocentric coordinates
# 
# In this version of the code, we calculate D and later s directly from the galactocentric coordinates of every single star.

# In[21]:


# Generate random values
# random_phi = spectrum.random_values(counts_phi, binning, bin_size)
# print(random_phi)


# ### Load data

# In[22]:


data = np.load(f'../results/{subhalo_profile}/stream_trajectories_section{num_data}.npz')
tr = data['tr']


# In[25]:


num_stars = tr.shape[0]
print(num_stars)


# ### Calculate photometric distance D

# In[33]:


D = spectrum.calculate_D_galactocentric(tr, rsun=[-8,0,0])


# In[35]:


# print(len(D))
# print(D)


# ### Calculate D and dD/dphi1

# ### Perform calculation from $\phi_1$ to s with galactocentric coordinates

# In[36]:


pos = stream.coordinates.transform(tr)


# In[37]:


# Determine parameter with fitfunction
x = np.array(pos['phi1'])
y = np.array(D)
x_err = x_err = np.full_like(x, 1.0)
y_err = np.sqrt(y)


# In[38]:


# Sort values for fit
sorted_indices = np.argsort(x)
x_sorted = x[sorted_indices]
y_sorted = y[sorted_indices]
x_err_sorted = x_err[sorted_indices]
y_err_sorted = y_err[sorted_indices]


# In[39]:


params, quintic_fit = spectrum.quintic_function(x_sorted, y_sorted, x_err_sorted, y_err_sorted)


# In[40]:


# Determine coefficients for D as a function of phi1
a5, a4, a3, a2, a1, a0 = params
# print(a5, a4, a3, a2, a1, a0)


# In[41]:


# Plot 
plt.figure(figsize=(8, 6))  # Größe des Plots festlegen
plt.scatter(pos['phi1'], D, color='blue', s=10, marker='.', label='Data')  
plt.plot(x_sorted, quintic_fit, label='Fitfunction', color='red')
# s=10 macht die Punkte klein, marker='.' für kleine Marker

# Title
plt.xlabel(r'$\phi_1$ [deg]', fontsize=12)  # Phi1 mit griechischem Symbol
plt.ylabel(r'Photometric Distance $D$ [kpc]', fontsize=12)

# Lim
plt.xlim(20, -120)  
plt.ylim(6, 12)     

# Grid
plt.grid(True, linestyle='--', alpha=0.5)  
plt.legend()

# Plot anzeigen
plt.savefig(f'../results/{subhalo_profile}/photometric_distance{num_data}_gc.pdf')
plt.show()


# In[43]:


phi_rad = pos['phi1']*(np.pi/180)


# In[44]:


s = spectrum.perform_integration_galactocentric(phi_rad, a5, a4, a3, a2, a1, a0)


# In[45]:


# print(s)
# print(max(s))
# print(min(s))


# ### Binning for s

# In[31]:


# Calculate bin size of s
# bin_size_s = spectrum.calculate_bin_size_array(s, bin_number)

# Print bin size
# Print(bin_size_s)


# In[46]:


# Use Ibata's binning

binning_lower_limit = -10.5
binning_upper_limit = 1.5

bin_size_s = spectrum.calculate_bin_size_bounds(binning_lower_limit, binning_upper_limit, bin_number)


# In[47]:


# Create bins for s (adjust minimum_s, if bounds manually chosen)

minimum_s = binning_lower_limit
binning_s = spectrum.create_binning(minimum_s, bin_size_s, bin_number)

# print(binning_s)
# print(len(binning_s))


# In[48]:


# Calculate bounds of the bins
binning_s_min, binning_s_max = spectrum.binning_bounds(binning_s, bin_size_s)
# print(binning_s_min), print(binning_s_max)


# In[49]:


# Assign the values to binning_s

counts_s = spectrum.bin_data(s, binning_s_min, binning_s_max, bin_number)
# print(counts_s)


# ### Calculate a fitting function

# In[44]:


# Set the error
x = np.array(binning_s)
y = np.array(counts_s)
x_err = np.full_like(x, 1.0)  # Uncertainty for x
y_err = np.sqrt(y) # Uncertainty for y
# x_err = 1.0
# y_err = 1.0


# In[45]:


# Import fitting function

params, quadratic_fit = spectrum.quadratic_function(x, y, x_err, y_err)


# In[46]:


# Plot
# plt.scatter(x, y, label='Daten')
# plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='o', label='Daten mit Fehlerbalken')
# plt.plot(x, quadratic_fit, color='red', label='Polynomialer Fit')
# plt.xlabel('s [kpc]')
# plt.ylabel('Counts')
# plt.legend()
# plt.show()


# ### Plot histogram

# In[47]:


# Plot
plt.figure(figsize=(13,6))
plt.plot(binning_s, quadratic_fit, label='Fitting Function', zorder=2)
plt.bar(binning_s, counts_s, width=0.28,color='r',alpha=0.5,label = 'Gaia + PS1',zorder=1)
plt.title(f'Density contrast s (Subhalo profile: {subhalo_profile}, Data section: {num_data})')
# plt.plot(bins_s_ibata, quadratic_fit, label='Fitting Function', zorder=2)
# plt.bar(bins_s_ibata, counts_s_ibata, width=0.27,color='r',alpha=0.5,label = 'Gaia + PS1',zorder=1)
# plt.xlim([20,-100])
plt.ylabel('Counts',size=15)
plt.xlabel('s [kpc]',size=15)
plt.legend()

# Save and show
plt.savefig(f'../results/{subhalo_profile}/density_contrast_s_section{num_data}_gc.pdf')
plt.show()


# ### Calculate power spectrum
# 
# We calculate the power spectrum with the CSD algorithm by SciPy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.csd.html) and define:
# 
# - *fs* = sampling frequency (set to 1/bin-size ~ 2.5)
# - *nperseg* = length of each segment (set to len(phi) = 40)

# In[48]:


# Calculate density contrast

dens_fit_s = counts_s/quadratic_fit


# In[49]:


# Calculate the power spectrum 

f_fit, pxx_fit = csd(dens_fit_s,dens_fit_s,fs=1/bin_size_s,scaling='spectrum',nperseg=bin_number)

# f_fit_err, pxx_fit_err = csd(dens_fit_s_err,dens_fit_s_err, fs=1/0.4,scaling='spectrum',nperseg=40)
# pxx_fit_adj = (180/(np.pi))*pxx_fit
# I'm plotting 1/frequency and I'm taking the square root of the power * s[-1] - s[0] 
# which in this case was 16


# In[50]:


# Calculate power
power = 1/f_fit
# print(power)


# In[51]:


# Save values for band plot
x = power  
y = 1 * np.sqrt(power[1] * np.array(pxx_fit))  

# Create data frame and save
data = pd.DataFrame({'x': x, 'y': y})
output_path = f'../results/{subhalo_profile}/power_spectrum_s_data_section{num_data}_gc.csv'
data.to_csv(output_path, index=False)


# In[52]:


# print(x)
# print(y)


# In[54]:


# Plot
plt.figure(figsize=(12,10))
plt.loglog(power, 1*np.sqrt(power[1]*np.array(pxx_fit)), color='green', label='Power Spectrum') # adjusted
plt.ylabel(r'$\sqrt{P_{\rho\rho}}$',size=15)
plt.xlabel(r'$1/k_{s}$ [kpc]',size=15)
plt.xlim([0,10])
plt.legend()
plt.title(f'$Subhalo$ $profile:$ {subhalo_profile}, $Data$ $section:$ {num_data}')

ax = plt.gca()
# ax.axvline(x=2.6, color='r', linestyle='--', linewidth=2)  # Linie bei x=2.6 einfügen
ax.set_xticks([1, 2, 2.6, 3, 4, 5])
ax.xaxis.set_major_formatter(ScalarFormatter())

# Save and show
plt.savefig(f'../results/{subhalo_profile}/power_spectrum_s_section{num_data}_gc.pdf')
plt.show()


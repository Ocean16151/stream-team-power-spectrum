#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import pandas as pd
import os
# from mpi4py import MPI

import stream


# In[2]:


# Retrieve num_data from environment variable in bash-script
num_data = int(os.getenv("num_data", "0"))  # Default = 0


# In[ ]:


# Updated subhalo list:
# 
# tNFW: `subhalo_profile_type='tNFW'`
# Cored profiles (SIDM with 5 cm²/g): `subhalo_profile_type='SIDM5_sph'`
# Super-cuspy profiles for dissipative SIDM: `subhalo_profile ='DSIDM553_sph'`


# In[3]:


# Chose subhalo profile
subhalo_profile = 'DSIDM553_sph'


# In[4]:


# Import data
data = pd.read_csv('../data/subhalo/Multiple_subhalo_impacts_samples.csv')


# In[5]:


# Chose section and subhalos with t < 2 Gyr
filtered_data = data[(data['num'] == num_data) & (data['t_since_impact'] <= 2)]


# In[6]:


# Assign parameter
M_subhalo = data['M_subhalo'].values
impact_position = data['fimp'].values
b_impact_param = data['b'].values
t_since_impact = data['t_since_impact'].values
delta_logc = data['delta_logc'].values
phi_impact = data['phi'].values
theta_impact = data['theta'].values
v_subhalo = data['v_subhalo'].values


# In[7]:


# Create subhalo list
subhalo_list = []

for _, row in filtered_data.iterrows():
    subhalo = {
        'M_subhalo': row['M_subhalo'],
        'impact_position': row['fimp'],
        'b_impact_param': row['b'],
        't_since_impact': row['t_since_impact'],
        'delta_logc': row['delta_logc'],
        'phi_impact': row['phi'],
        'theta_impact': row['theta'],
        'v_subhalo': row['v_subhalo']
    }
    subhalo_list.append(subhalo)


# In[8]:


for subhalo in subhalo_list:
    subhalo['subhalo_profile_type'] = subhalo_profile


# In[9]:


# Simulate
tr = stream.simulate(subhalo_list=subhalo_list,verbose=False)


# In[ ]:


# Multiprocessing
# 
# By default, `stream.simulate()` makes a linear uniformly-distributed stream. A more realistic stream can be made by setting the keyword `star_distribution='random'`. One can also increase the number of stars by setting the `num_stars` keyword (the default is 100). 
# The simulation time increases with the number of stars. This can be circumvented by running multiple simulations in parallel and combining the results into one stream. Remember to set `random_seed` to be different for each parallel process, otherwise you will get copies of the same random stream.
# Here we will get a respectable stream of 2000 stars by running 8 simulations in parallel.


# In[17]:


num_processes = 20

inputs = [{'star_distribution':'random', 'random_seed':n, 'subhalo_list':subhalo_list, 'num_stars':100} for n in range(num_processes)]

# inputs = [{'star_distribution':'random',
#            'random_seed':n,
#            'subhalo_profile_type':'SIDM5',
#            'M_subhalo':1e7,
#            'num_stars':250} for n in range(8)]

# Divide one stream simulation into parallel simulations
with Pool(num_processes) as p:
    output = p.map(stream.simulate, inputs)

# Put all trajectories together
tr = np.concatenate(output)
pos = stream.coordinates.transform(tr)


# In[18]:


# Use bootstrap sampling 

phi1_positions = pos['phi1']
phi2_positions = pos['phi2']

# Chose stars in range -90° to 10°

chosen_indices = (phi1_positions >= -90) & (phi1_positions <= 10) 
phi1_filtered = phi1_positions[chosen_indices]
phi2_filtered = phi2_positions[chosen_indices]

# Do bootstrap sampling with 2000 stars

bootstrap_indices = np.random.choice(len(phi1_filtered), 60, replace=True)
phi1_bootstrap = phi1_filtered[bootstrap_indices]
phi2_bootstrap = phi2_filtered[bootstrap_indices]  


# In[27]:


# Show and save stream plot

plt.figure(figsize=(8,2))
# plt.plot(pos['phi1'],pos['phi2'],'.', markersize=2)
plt.plot(phi1_bootstrap, phi2_bootstrap, '.', markersize=2, color='red')
plt.xlabel(r'$\phi_1$')
plt.ylabel(r'$\phi_2$')
plt.title(f'$Subhalo$ $profile:$ {subhalo_profile}, $Data$ $section:$ {num_data}')
plt.savefig(f'../results/{subhalo_profile}/stream_simulation_bootstrap_section{num_data}.pdf')
plt.show()


# In[24]:


# Indices of the filtered stars
original_indices = np.where(chosen_indices)[0]  
bootstrap_original_indices = original_indices[bootstrap_indices]

# Find positions in tr
tr_bootstrap = tr[bootstrap_original_indices]  


# In[25]:


# Save trajectories data
# np.savez(f'../results/{subhalo_profile}/stream_trajectories_section{num_data}.npz', tr=tr)


# In[26]:


# Save trajectories data

np.savez(f'../results/{subhalo_profile}/stream_trajectories_bootstrap_section{num_data}.npz', tr=tr_bootstrap)


# In[ ]:


# Data output for the power spectrum

# data_power_spectrum = pd.DataFrame(pos['phi1'], columns=['x'])
# simulation_data = f'../results/{subhalo_profile}/data_output_power_section{num_data}.csv'
# data_power_spectrum.to_csv(simulation_data, index=False) 


# In[ ]:


# Data output with bootstrapped data

data_ps_bootstrap = pd.DataFrame(phi1_bootstrap, columns=['x'])
simulation_data_bootstrap = f'../results/{subhalo_profile}/data_output_power_bootstrap_section{num_data}.csv'
data_ps_bootstrap.to_csv(simulation_data_bootstrap, index=False)


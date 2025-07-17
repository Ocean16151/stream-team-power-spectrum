#!/usr/bin/env python
# coding: utf-8

# # Stream simulation
# 
# We provide some examples for using the `stream` package for running simulations of GD-1. Simulations are made using the code `stream.simulate()`, which returns a list of phase space trajectories for $N$ stars. By default, `stream.simulate()` runs a simulation with default parameters, including a single perturbing subhalo described by the Bonaca-Hernquist profile.
# 
# Another useful module is the `coordinates` module, which transforms from galactocentric coordinates to celestial coordinates (especially, the sky positions $\phi_1$-$\phi_2$).

# In[2]:


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


# In[3]:


# num_data = 0


# ## Subhalo profiles
# 
# Alternative profiles can be specified using the keyword `subhalo_profile_type`. 
# Currently the following profiles are allowed:
# - tNFW (`subhalo_profile_type='tNFW'`): Also needs keyword `M_subhalo` to be set.
# - Hernquist (`subhalo_profile_type='Hern')`: Also needs keyword `M_subhalo` to be set.
# - SIDM from Jeans model (`subhalo_profile_type='Jeans')`: Also needs keywords `M_subhalo` and `sigma_m` to be set.
# - SIDM5 from numerical grid of profile data (`subhalo_profile_type='SIDM5')`: Also needs keywords `M_subhalo`.
# - Custom numerical profile (`subhalo_profile_type='numerical'`): Also needs keywords `M` and `r` to be set, as equal-length arrays.

# Updated subhalo list:
# 
# - tNFW: `subhalo_profile_type='tNFW'`
# - Cored profiles (SIDM with 5 cm²/g): `subhalo_profile_type='SIDM5_sph'`
# - Super-cuspy profiles for dissipative SIDM: `subhalo_profile ='DSIDM553_sph'`

# In[4]:


# Chose subhalo profile
subhalo_profile = 'DSIDM553_sph'


# ## Import data

# In[5]:


data = pd.read_csv('../data/subhalo/Multiple_subhalo_impacts_samples.csv')


# In[6]:


# Chose section and subhalos with t < 2 Gyr
filtered_data = data[(data['num'] == num_data) & (data['t_since_impact'] <= 2)]

# Assign parameter
M_subhalo = data['M_subhalo'].values
impact_position = data['fimp'].values
b_impact_param = data['b'].values
t_since_impact = data['t_since_impact'].values
delta_logc = data['delta_logc'].values
phi_impact = data['phi'].values
theta_impact = data['theta'].values
v_subhalo = data['v_subhalo'].values

# Print data
# print(filtered_data)


# ## Create list

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


# ## Simulate

# In[9]:


tr = stream.simulate(subhalo_list=subhalo_list,verbose=False)
# tr = stream.simulate(subhalo_list=subhalo_list,verbose=False)


# ## Multiprocessing
# 
# By default, `stream.simulate()` makes a linear uniformly-distributed stream. A more realistic stream can be made by setting the keyword `star_distribution='random'`. One can also increase the number of stars by setting the `num_stars` keyword (the default is 100). 
# 
# The simulation time increases with the number of stars. This can be circumvented by running multiple simulations in parallel and combining the results into one stream. Remember to set `random_seed` to be different for each parallel process, otherwise you will get copies of the same random stream.
# 
# Here we will get a respectable stream of 2000 stars by running 8 simulations in parallel.

# In[10]:


# inputs = [{'star_distribution':'random','random_seed':n, 'subhalo_list':subhalo_list, 'num_stars':250} for n in range(8)]

# inputs = [{'star_distribution':'random',
#            'random_seed':n,
#            'subhalo_profile_type':'SIDM5',
#            'M_subhalo':1e7,
#            'num_stars':250} for n in range(8)]

# num_processes = 20

# Divide one stream simulation into parallel simulations
# with Pool(num_processes) as p:
#     output = p.map(stream.simulate,inputs)
    
# Put all trajectories together
# tr = np.concatenate(output)
# pos = stream.coordinates.transform(tr)


# In[11]:


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


# On the cluster, we use MPI to distribute the parallel processes on different nodes of the cluster such that one process runs on one node. 

# In[26]:


# Initialize MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()  # Current rank
# size = comm.Get_size()  # Number of processes

# Number of simulations
# num_simulations = 8

# Generate inputs
# inputs = [{'star_distribution': 'random', 'random_seed': rank, 'subhalo_list':subhalo_list, 'num_stars': 250} for rank in range(num_simulations)]

# Divide the stream simulation into parallel simulations
# output = stream.simulate(inputs[rank])

# Put results together
# all_outputs = comm.gather(output, root=0)  

# if rank == 0:
    # Put trajectories together
#     tr = np.concatenate(all_outputs)
#     pos = stream.coordinates.transform(tr)


# ## Plot and save

# In[12]:


# Show and save stream plot

plt.figure(figsize=(8,2))

plt.plot(pos['phi1'],pos['phi2'],'.',markersize=2)

plt.xlabel(r'$\phi_1$')
plt.ylabel(r'$\phi_2$')
plt.title(f'$Subhalo$ $profile:$ {subhalo_profile}, $Data$ $section:$ {num_data}')
plt.savefig(f'../results/{subhalo_profile}/stream_simulation_section{num_data}.pdf')
plt.show()


# In[13]:


# Save trajectories data

np.savez(f'../results/{subhalo_profile}/stream_trajectories_section{num_data}.npz', tr=tr)


# In[14]:


# Show stream trajectories data
# Load
# data = np.load(f'../results/tNFW/stream_trajectories_section{num_data}.npz')

# Show
# for key in data:
#     print(f"Key '{key}':\n", data[key], "\n")


# In[15]:


# Data output for the power spectrum

data_power_spectrum = pd.DataFrame(pos['phi1'], columns=['x'])
simulation_data = f'../results/{subhalo_profile}/data_output_power_section{num_data}.csv'

data_power_spectrum.to_csv(simulation_data, index=False) 


# In[ ]:





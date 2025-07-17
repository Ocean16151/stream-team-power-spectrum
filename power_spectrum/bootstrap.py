#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


def bootstrap_sampling(phi1_positions, phi2_positions, num_stars=2000):
    """Führt ein einmaliges Bootstrap-Sampling durch und gibt die gesampelten phi1- und phi2-Werte zurück."""
    
    # Wähle Sterne im Bereich -90° bis 10°
    chosen_indices = (phi1_positions >= -90) & (phi1_positions <= 10)
    phi1_filtered = phi1_positions[chosen_indices]
    phi2_filtered = phi2_positions[chosen_indices]

    # Bootstrap-Sampling mit der gewünschten Anzahl an Sternen
    bootstrap_indices = np.random.choice(len(phi1_filtered), num_stars, replace=True)
    phi1_bootstrap = phi1_filtered[bootstrap_indices]
    phi2_bootstrap = phi2_filtered[bootstrap_indices]

    return phi1_bootstrap, phi2_bootstrap, chosen_indices, bootstrap_indices



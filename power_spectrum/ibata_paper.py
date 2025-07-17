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

# This is the fitting function that's used in the paper
def ibata_fitting(phi1):
    phi1 = phi1*np.pi/180
    C = -37.51*phi1**2 - 46.51*phi1+14.37
    return C

# This is the fitting function that's used in the paper divided by 1.25
def ibata_fitting_modified(phi1):
    phi1 = phi1*np.pi/180
    C = -37.51*phi1**2 - 46.51*phi1+14.37
    C = C/1.25
    return C


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
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from scipy.signal import csd
from sympy import *
from sympy import symbols
from sympy import integrate

import numpy as np
from scipy.optimize import curve_fit


# In[ ]:


def integrand(phi_f_value):
    
    phi_f = symbols('phi_f')
    
    # Polynom
    # Coefficients
    a5, a4, a3, a2, a1, a0 = -4.302, -11.54, -7.161, 5.985, 8.595, 10.36

    # Function
    D = a5*(phi_f**5) + a4*(phi_f**4) + a3*(phi_f**3) + a2*(phi_f**2) + a1*(phi_f) + a0

    # Calculate derivative
    D_derivative = diff(D, phi_f)

    # Calculate integrand
    integrand_squared = D**2 + D_derivative**2
    integrand_non_numerical = sqrt(integrand_squared)
    
    integrand_function = lambdify(phi_f, integrand_non_numerical, 'numpy')
    integrand_result = integrand_function(phi_f_value)

    # Result
    return integrand_result


# In[ ]:


def integrand_differential_equation(phi_f, y):
    return integrand(phi_f)


# In[ ]:


def perform_integration(random_phi):
    # Array for results
    integral_results = np.zeros(len(random_phi))

    # Integrate 
    for i, phi_f_end in enumerate(random_phi):
        result = solve_ivp(integrand_differential_equation, [0, phi_f_end], [0], t_eval=[phi_f_end])
        integral_results[i] = result.y[0][-1]

    return integral_results


# In[ ]:





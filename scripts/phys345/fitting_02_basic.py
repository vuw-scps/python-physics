#!/usr/bin/env python
# coding: utf-8

# ## Nonlinear fitting example
# 
# We'll use an artificial dataset to fit a model with 5 parameters, 
# 
# $$
# y(x) = b_1 \exp\left(-b_2 x\right) + \frac{b_3}{b_5\sqrt{2\pi}} \exp\left(-\tfrac12\frac{(x-b_4)^2}{b_5^2}\right)
# $$
# 
# It's the sum of a decaying exponential background, and a Gaussian peak.
# 
# We first load some standard packages, including `scipy.optimize` and `lmfit` for the nonlinear optimisation.

# In[2]:


import os, sys
import numpy as np
from numpy import exp, pi, sqrt

from numpy import linspace
from scipy.optimize import leastsq

import pandas as pd
from matplotlib import pyplot

import scipy.optimize as optimization
# conda install -c conda-forge lmfit
from lmfit import minimize, Parameters 


# ### Loading and displaying the dataset
# 
# 

# In[3]:


data = pd.read_csv("exppeak.txt", header=0, delim_whitespace=True)
# note: three ways to refer to a particular column
data.iloc[:,1]
data.loc[:,'x']
data.x
pyplot.errorbar(data.x, data.y, yerr=data.sigma, fmt='.')


# ### Defining the model and testing it
# 
# We now create a function that computes the predicted values for a given set of parameters $b_1\dots,b_5$:

# In[4]:


def model_peak(x, b1, b2, b3, b4, b5):
    return b1*exp(-b2*x) + b3/(b5*sqrt(2*np.pi))*exp(-0.5*(x-b4)**2 / b5**2)


# We test it by superposing the prediction and the data (in this artificial example, we know the "true" parameters that were used to generate the dataset).

# In[5]:


xx = np.arange(0,10,0.1)
truth = model_peak(xx, 1.12, 1.52, 3, 3.26,  1.48)
pyplot.errorbar(data.loc[:,'x'], data.loc[:,'y'], yerr=data.loc[:,'sigma'], fmt='.', label='data')
pyplot.plot(xx, truth, 'k', label='Truth')
pyplot.legend(loc='best')


# ## Simple nonlinear fitting with `scipy.optimize.curve_fit`
# 
# Perhaps the simplest interface for nonlinear fitting is the `curve_fit` function, which takes as arguments the model defined above, the data, and a set of initial guess parameters. The output of the function is the optimised parameters, and the covariance matrix for the fitted parameters:

# In[6]:


import scipy.optimize as optimization
p0 = [1, 1, 1, 4,  1]
popt, pcov = optimization.curve_fit(model_peak, data.x, data.y, p0, data.sigma)
popt


# In[7]:


pcov


# We can visualise the output to confirm that the fit is reasonable.

# In[10]:


guess = model_peak(xx, *p0)
fit = model_peak(xx, *popt)
pyplot.errorbar(data.loc[:,'x'], data.loc[:,'y'], yerr=data.loc[:,'sigma'], fmt='.', label='data')
pyplot.plot(xx, truth, 'k-', label='Truth')
pyplot.plot(xx, guess, 'g', label='Guess')
pyplot.plot(xx, fit, 'r', label='curve_fit')
pyplot.legend(loc='best')
pyplot.xlabel("x")
pyplot.ylabel("y")


# ### More optimisation methods via `lmfit.minimize`
# 
# `curve_fit` is simple to use, but not very flexible. We'll try another alternative here, which requires a little more effort in the setup, but provides more options such as different algorithms for the search.

# In[11]:


from lmfit import minimize, Parameters, Model, report_fit
from scipy.optimize import leastsq


# Here we need to provide `minimize` with the normalised residuals (note: they are not squared, as minimize will square them internally for least-square based methods):

# In[19]:


def residuals(pars, x, y, eps):
    A1 = pars['A1'].value
    Decay = pars['Decay'].value
    A2 = pars['A2'].value
    Position = pars['Position'].value
    Width = pars['Width'].value

    model = A1*exp(-Decay*x) + A2/(Width*sqrt(2*np.pi))*exp(-0.5*(x-Position)**2 / Width**2)
    
    return (y-model) / eps


# The rest of the setup consists in creating a `Parameters` object holding the information for all free parameters (at a minimum, their name and initial guess for the value, but one can also set constraints). We can then call the minimize function:

# In[20]:


params = Parameters()
params.add('A1', value=1.0)
params.add('Decay', value=1.5)
params.add('A2', value=0.5)
params.add('Position', value=3.0)
params.add('Width', value=1.5)

out = minimize(residuals, params, args=(data.x, data.y, data.sigma), method='leastsq')

print("# Fit using leastsq:")
report_fit(out)


# We can compare the fitted parameters to those obtained via curve_fit:

# In[24]:



with np.printoptions(precision=3, suppress=True):
    print("# Fit using curve_fit:")
    print(popt)


# and their uncertainty, which are estimated by the diagonal components of the covariance matrix:

# In[28]:



with np.printoptions(precision=3, suppress=True):
    print("# Uncertainties using curve_fit:")
    print(np.sqrt(np.diag(pcov)))
    


# We could also obtain such undertainty estimates "manually", by evaluating the Hessian matrix at the optimum position in the parameter space:

# In[35]:


# conda install -c conda-forge numdifftools 
import numdifftools as nd

# vals = out.params.valuesdict()
opt = np.array(out.params)

# residuals(pars, x, y, eps)
def objective(p):
    model = model_peak(data.x, *p)
    chi2 = np.sum(((data.y-model) / data.sigma)**2)
    return chi2

H = nd.Hessian(objective)(popt)

with np.printoptions(precision=3, suppress=True):
    print(H)


# The covariance matrix is obtained as the inverse of half the Hessian matrix. The uncertainty of each parameter are the square root of the diagonal elements of this covariance matrix.

# In[36]:


cov = np.linalg.inv(0.5*H)
with np.printoptions(precision=3, suppress=True):
    print(cov)
    print("# Uncertainties from the covariance matrix:")
    print(np.sqrt(np.diag(cov)))


# Changing the method produces similar results:

# In[37]:


out2 = minimize(residuals, params, args=(data.x, data.y, data.sigma), method='lbfgsb')
print("# Fit using lbfgsb:")
report_fit(out2)


# _Download this page [as a Jupyter notebook](https://github.com/vuw-scps/python-physics/raw/master/notebooks/phys345/fitting_02_basic.ipynb) or as a [standalone Python script](https://github.com/vuw-scps/python-physics/raw/master/scripts/phys345/fitting_02_basic.py)._

# 

#!/usr/bin/env python
# coding: utf-8

# ## Simple plots
# 
# 

# In[1]:


from matplotlib import pyplot as plt
import numpy as np


# ## Calculation
# 
# Code to calculate trajectory of an object undergoing constant acceleration

# In[2]:


# Set time interval
ti = 0
tf = 10.0                 # Use 15.0, not 15 to tell Python what kind of variable to use for the time
NumPts = 10

# Set launch speed and angle, and initial x and y position
vi = 90
theta = 30*np.pi/180
xi = 0                    
yi = 0

# Calculate components of initial velocity
vxi=vi*np.cos(theta)
vyi=vi*np.sin(theta)

# Define x and y components of acceleration
acc_x = 0
acc_y = -10

# Create a list of values of the time
times = np.linspace(ti,tf,NumPts)	

# Calculate position at each time
xx = xi + 12*times + 0.5*acc_x*times**2			
yy = yi + 20*times + 0.5*acc_y*times**2


# ## Plot the results
# 

# In[3]:



fig1 = plt.figure()
plt.plot(xx,yy,'r*')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
#fig1.savefig('Projectile.jpg')


# _Download this page [as a Jupyter notebook](https://github.com/vuw-scps/python-physics/raw/master/notebooks/phys114/SimplePlots.ipynb) or as a [standalone Python script](https://github.com/vuw-scps/python-physics/raw/master/scripts/phys114/SimplePlots.py)._

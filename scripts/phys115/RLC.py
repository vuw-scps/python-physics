#!/usr/bin/env python
# coding: utf-8

# ## RLC circuit
# 
# 

# In[5]:


from matplotlib import pyplot as plt
import numpy as np


# ## Constants
# 
# 

# In[6]:


I=1
R=2
L=3
C=4
f=5

w=2*np.pi*f


# ### IMPEDANCE

# In[7]:


XL=w*L
XC=1/(w*C)
X=XL-XC

# Magnitude and angle of impedance
Z=np.sqrt((R**2+X**2))
Angle=np.arctan(X/R)


# ### VOLTAGE

# In[8]:


VR=I*R
VL=I*XL
VC=I*XC

# Magnitude and angle of total voltage
V=np.sqrt((VR**2+(VL-VC)**2))
Angle=np.arctan((VL-VC)/VR)

print("")
print("voltage, current, and impedance in the circuit")
print("Current (I)"," ","="," ",I,"Amperes")
print("Impedance (Z) in 'i,j' notation"," ","="," ",R,"+",X,"j"," ","Ohms")
print("Voltage (V) in 'i,j' notation"," ","="," ",V,"+",VL-VC,"j"," ","Volts")

print("")
print("voltage, current, and resistance in the resistor")
print("Current (I)"," ","="," ",I,"Amperes")
print("Resistance (R)"," ","="," ",R,"Ohms")
print("Voltage (VR)","="," ",VR,"Volts")


# _Download this page [as a Jupyter notebook](https://github.com/vuw-scps/python-physics/raw/master/notebooks/phys115/RLC.ipynb) or as a [standalone Python script](https://github.com/vuw-scps/python-physics/raw/master/scripts/phys115/RLC.py)._

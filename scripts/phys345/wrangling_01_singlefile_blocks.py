#!/usr/bin/env python
# coding: utf-8

# ## Reading a data file with multiple data blocks
# 
# 

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt


# In[2]:


d = pd.read_csv("SIO_2.TXT", delim_whitespace=True,  skiprows=4)
d


# In[100]:


with open("SIO_2.TXT") as f:
    head = [next(f) for x in range(4)]
print(head)


# In[101]:


V_G = np.arange(-40,9.95+0.15,step=0.15) # note arange does not include the endpoint, so we go one step further
V_G.shape
V_DS = np.arange(-20,0+5,step=5) # note arange does not include the endpoint, so we go one step further
V_DS.shape


# Let's check that the information is consistent with the data

# In[102]:


d.shape[0] / V_G.shape[0] == V_DS.shape[0] 


# In[119]:


# d["V_DS"] = np.tile(V_DS, V_G.shape[0])
d["V_DS"] = np.repeat(V_DS, V_G.shape[0])
d


# We may sometimes want to convert from long to wide format, so that all the measurements are in separate columns:

# In[104]:


d.pivot(index='V', columns='V_DS', values='A')


# We can now, for example, plot the I/V data by groups:

# In[111]:


#conda install -c conda-forge plotnine 
from plotnine import *
# note the use of parentheses, because the syntax below (+) is non-standard in Python
(ggplot(d) +
  geom_point(aes(x = 'V',
                  y = 'A',
                  color = 'factor(V_DS)')) +
  labs(colour = 'V_DS', x = "Voltage", y = "Current"))


# Another type of thing we might do with this 'long' format is to transform or summarise the data by V_DS value, with the [split-apply-combine strategy](https://pandas.pydata.org/docs/user_guide/groupby.html):

# In[106]:


def IV_ratio(x):
    return  x['V'] / x['A']

d["ratio"] = d.groupby("V_DS").apply(IV_ratio).values
d


# In[117]:


# d.groupby("V_DS").apply(lambda x: x.max() - x.min())


# In[109]:


(ggplot(d) +
  geom_point(aes(x = 'V',
                  y = 'ratio',
                  color = 'factor(V_DS)')) +
  scale_y_log10() +
  labs(colour = 'V_DS', x = "Voltage", y = "Impedance"))


# _Download this page [as a Jupyter notebook](https://github.com/vuw-scps/python-physics/raw/master/notebooks/phys345/wrangling_01_singlefile_blocks.ipynb) or as a [standalone Python script](https://github.com/vuw-scps/python-physics/raw/master/scripts/phys345/wrangling_01_singlefile_blocks.py)._

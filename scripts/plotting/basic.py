#!/usr/bin/env python
# coding: utf-8

# ## Basic plots
# 
# 

# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# ## Line plot
# 
# Let's plot a few sine functions with different line types; we assign a label to each trace, which can be displayed in a legend.

# In[3]:


x = np.linspace(-np.pi,np.pi, 200)

lts = ['-',':','--','-.','-']

for i in range(5):
    plt.plot(x, np.sin(i*x)+i,lts[i],label="{0}*x".format(i))
    
plt.ylabel("y")
plt.xlabel("x")
plt.legend(ncol=1,bbox_to_anchor=(1.0,1), title='variable')
plt.show()


# ## Scatter plot

# In[4]:


x = np.linspace(-np.pi,np.pi, 200)

lts = ['-',':','--','-.','-']

for i in range(5):
    plt.plot(x, np.sin(i*x)+i,lts[i],label="{0}*x".format(i))
    
plt.ylabel("y")
plt.xlabel("x")
plt.legend(ncol=1,bbox_to_anchor=(1.0,1), title='variable')
plt.show()


# ## Subplots
# 
# To display multiple plots side-by-side, set up a subplot. Note the slightly different syntax for labels etc. 

# In[5]:


fig,ax=plt.subplots(2,1,sharex='col')

for i in range(5):
    ax[0].plot(x,np.sin(i*x)+i,lts[i],label="{0}*x".format(i))
    ax[1].plot(x,np.cos(i*x)+i,lts[i]                        )
    
ax[0].set_ylabel("sin")
ax[1].set_ylabel("cos")
ax[1].set_xlabel("x")
ax[0].legend(ncol=5,bbox_to_anchor=(0.1,1.02))
plt.show()


# ## Grammar of graphics with Pandas
# 
# An alternative is to create plots following _a grammar of graphics_, such as implemented in [plotly](https://plotly.com/python). Data should typically be prepared in long format [`Panda ` `DataFrame`s](https://pandas.pydata.org/).
# 

# In[6]:


import pandas as pd

x = np.linspace(-np.pi,np.pi, 200)
mat = [np.sin(i*x.T) + i for i in range(5)]

# set column names to refer to them
columns = ['a','b','c','d','e']

d = pd.DataFrame(dict(zip(columns, mat)))
d['x'] = x
d.head()


# Converting from this wide format to long format can be achieved with `melt`,

# In[7]:


m = pd.melt(d, id_vars='x')
m.head()          


# In[12]:


import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
# pio.renderers.default = "browser"

fig = px.line(m, x='x', y='value', color='variable')
fig.show()


# In[11]:



#fig = go.Figure()

# Add scatter trace with medium sized markers
#fig.add_trace(
#    go.Scatter(mode='lines',
#        x=m['x'],
#        y=m['value']))


# _Download this page [as a Jupyter notebook](https://github.com/vuw-scps/python-physics/raw/master/notebooks/plotting/line.ipynb) or as a [standalone Python script](https://github.com/vuw-scps/python-physics/raw/master/scripts/plotting/line.py)._

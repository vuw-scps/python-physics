#!/usr/bin/env python
# coding: utf-8

# ## Reading many similar files from a folder and combining the results
# 
# 

# In[2]:


import numpy as np
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt


# First we try to read one file to make sure we're using the correct parameters:

# In[3]:


test = pd.read_csv('filenames/Labram_laser514nm_1mW_30sx4_800nmgrating_T-200K_sample-B.txt', delim_whitespace=True, names=['wavenumber','intensity'])
test


# In[ ]:





# In[ ]:





# This seems reasonable, but now we'd like to extract parameters from the filename.

# In[7]:


import re
def read_file(f) :
    d = pd.read_csv(f, delim_whitespace=True, names=['wavenumber','intensity'])
    info = f.split('_')
    d['sample'] = re.sub('sample-|.txt','', info[6])
    d['temperature'] = float(re.sub('T-|K','', info[5]))
    
    return(d)

read_file('filenames/Labram_laser514nm_1mW_30sx4_800nmgrating_T-200K_sample-B.txt')


# We can now grab all the filenames, and read them in:

# In[ ]:


import glob
import os

path = r'filenames' 
txt_files = glob.glob(os.path.join(path , "*.txt"))
n = len(txt_files)
txt_files


# In[8]:


d = pd.concat((read_file(f) for f in txt_files))
d


# We now have access to the whole set of variables corresponding to each data point:

# In[12]:


from plotnine import *
(ggplot(d) +
  geom_line(aes(x = 'wavenumber',
                  y = 'intensity')) +
  facet_grid('temperature ~ sample', scales='free'))


# If needed, we can also reshape the dataset to 'wide' format,  

# In[14]:


w = d.pivot(index='wavenumber', columns=['temperature','sample'], values='intensity')
w


# Incidentally, the reverse operation can be done with `melt`, and would look like:

# In[34]:


var_list=list(w.columns)
l = pd.melt(w, value_vars=var_list,value_name='I', ignore_index=False)
l = l.reset_index(level=['wavenumber']) # trick to convert the confusing (to me) multiindex into a standard column
l


# _Download this page [as a Jupyter notebook](https://github.com/vuw-scps/python-physics/raw/master/notebooks/phys345/wrangling_02_folder_filenames.ipynb) or as a [standalone Python script](https://github.com/vuw-scps/python-physics/raw/master/scripts/phys345/wrangling_02_folder_filenames.py)._

# 

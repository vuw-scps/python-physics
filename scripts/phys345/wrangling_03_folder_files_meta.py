#!/usr/bin/env python
# coding: utf-8

# ## Reading many similar files from a folder and combining the results with info from a different source
# 
# 

# In[39]:


import numpy as np
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt


# First we try to read one file to make sure we're using the correct parameters:

# In[40]:


test = pd.read_csv('metadata/001.txt', delim_whitespace=True, names=['wavenumber','intensity'])
test


# In[41]:


import glob
import os

path = r'metadata' 
txt_files = glob.glob(os.path.join(path , "[0-9][0-9][0-9].txt"))
n = len(txt_files)
txt_files.sort()
txt_files


# In[42]:


d = pd.concat((pd.read_csv(f, delim_whitespace=True, names=['wavenumber','intensity']) for f in txt_files), ignore_index=False, keys=range(n), names=['ID','row'])
d = d.reset_index(level=['ID']) # trick to convert the confusing (to me) multiindex into a standard column
d


# Let's plot these to check that it makes sense:

# In[43]:


#conda install -c conda-forge plotnine 
from plotnine import *
# note the use of parentheses, because the syntax below (+) is non-standard in Python
(ggplot(d) +
  geom_line(aes(x = 'wavenumber',
                  y = 'intensity+ID/10',
                  color = 'factor(ID)')))


# This seems reasonable, but now we'd like to retrieve the parameters for each file, which are stored in file `parameters.txt`.

# In[44]:


meta = pd.read_csv('metadata/parameters.txt', delim_whitespace=True)

meta['ID'] = range(meta.shape[0])
meta


# Now we need to join those two data sets:

# In[45]:


full = pd.merge(d, meta, on='ID')


# We now have access to the whole set of variables corresponding to each data point:

# In[46]:


(ggplot(full) +
  geom_line(aes(x = 'wavenumber',
                  y = 'intensity',
                  color = 'factor(ID)')) +
  facet_grid('temperature ~ sample', scales='free'))


# _Download this page [as a Jupyter notebook](https://github.com/vuw-scps/python-physics/raw/master/notebooks/phys345/wrangling_03_folder_files_meta.ipynb) or as a [standalone Python script](https://github.com/vuw-scps/python-physics/raw/master/scripts/phys345/wrangling_03_folder_files_meta.py)._

# In[ ]:





# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# ## Regular expressions
# 
# https://byuidatascience.github.io/python4ds/strings.html
# 
# https://jakevdp.github.io/WhirlwindTourOfPython/14-strings-and-regular-expressions.html

# In[12]:


import numpy as np
import pandas as pd
import re


# First we try to read one file to make sure we're using the correct parameters:

# In[2]:


s = 'filenames/Labram_laser514nm_1mW_30sx4_800nmgrating_T-200K_sample-B.txt'


# In[15]:


pd.Series(["x", "y", "z"]).str.cat()


# In[16]:


pd.Series(["x", "y"]).str.cat(sep = '_')


# This seems reasonable, but now we'd like to extract parameters from the filename.

# In[17]:


x = pd.Series(["Apple", "Banana", "Pear"])
x.str[0:3]


# In[18]:


x.str[-3:]


# ### matching patterns with regular expressions

# In[20]:


import glob
import os

x = pd.Series(["apple", "banana", "pear"])
x.str.replace("an", "zzz")


# In[22]:


x.str.replace("^a", "zzz", regex=True)


# Repetitions
# 
# - ?: 0 or 1
# - +: 1 or more
# - *: 0 or more

# In[56]:


s = 'filenames/Labram_laser514nm_1mW_30sx4_800nmgrating_T-200K_sample-B.txt'

# temp = re.compile('[0-9]{3}')
regex = re.compile('[0-9]{3}K')
regex = re.compile('[0-9]+K')
regex = re.compile('([0-9]+)K')

regex.findall(s)



# _Download this page [as a Jupyter notebook](https://github.com/vuw-scps/python-physics/raw/master/notebooks/phys345/wrangling_00_regex.ipynb) or as a [standalone Python script](https://github.com/vuw-scps/python-physics/raw/master/scripts/phys345/wrangling_00_regex.py)._

# 

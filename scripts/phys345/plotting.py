#!/usr/bin/env python
# coding: utf-8

# # Tutorial example: plotting with Python
# 
# We'll load the `palmerpenguins` dataset, and produce a simple scatter plot using different plotting packages for comparison. Each of these plots can be adjusted in many ways, and you should explore the documentation of those packages to get an idea of the possibilities (and the specific syntax to use).

# In[1]:


# pip install palmerpenguins 
import palmerpenguins
import pandas as pd


# Let's check that things work by looking at the first few rows of the dataset

# In[2]:


from palmerpenguins import load_penguins
penguins = load_penguins()
penguins.head()


# ## Plotting with matplotlib
# 
# Let's make a plot with the standard pyplot package:

# In[3]:


import matplotlib.pyplot as plt

colors = {'Adelie':'blue', 'Gentoo':'orange', 'Chinstrap':'green'}
plt.scatter(penguins.flipper_length_mm,
penguins.body_mass_g, 
c= penguins.species.apply(lambda x: colors[x]))
plt.xlabel('Flipper Length')
plt.ylabel('Body Mass')


# ## Using plotnine

# In[4]:


#conda install -c conda-forge plotnine 
from plotnine import *
# note the use of parentheses, because the syntax below (+) is non-standard in Python
(ggplot(penguins) +
  geom_point(aes(x = 'flipper_length_mm',
                  y = 'body_mass_g',
                  color = 'species',
                  shape = 'species')) +
  xlab("Flipper Length") +
  ylab("Body Mass"))


# ## Interactive plotting with plotly

# In[5]:


import plotly.express as px

fig = px.scatter(penguins,
                 x="flipper_length_mm",
                 y="body_mass_g",
                 color= "species",
                 symbol= "species",
                 labels=dict(flipper_length_mm="Flipper Length",
                             body_mass_g="Body Mass"))
fig.show()


# ## Plotting with seaborn

# In[6]:


import seaborn as sns 

# Apply the default theme
sns.set_theme()
# sns.set_style('whitegrid')
p = sns.relplot(x = 'flipper_length_mm',
            y ='body_mass_g',
            hue = 'species',
            style = 'species',
            data = penguins)
p.set_xlabels('Flipper Length')
p.set_ylabels('Body Mass') 


# ## Plotting with altair

# In[7]:


import altair as alt


# In[8]:


chart = alt.Chart(penguins).mark_point().encode(
    x = 'flipper_length_mm:Q',
    y ='body_mass_g:Q',
    color='species:N',
).properties(width=600)
chart


# The x and y axes include 0 by default, which we can adjust by providing a scale. While we're at it, let's split the plot into different panels for illustration:

# In[10]:


chart = alt.Chart(penguins).mark_point().encode(
    alt.X('flipper_length_mm:Q',
        scale=alt.Scale(zero=False)
    ),
    alt.Y('body_mass_g:Q',
        scale=alt.Scale(zero=False)
    ),
    color='species:N',
    column='island',
).properties(width=220)
chart


# ## Suggested exercises
# 
# 
# ### Data input/output
# 
# Instead of using a built-in dataset like in the examples above:
# 
# 1. Take a look at the [10 mins introduction to `pandas`](https://pandas.pydata.org/docs/user_guide/10min.html#min); the dataframe format is very useful to mix numeric variables (as in a standard numpy matrix) together with other types (dates, categories, strings, etc.)
# 1. Create a synthetic dataset with 5 (or more) variables, at least one of which should be a categorical type (as in the penguin species, or islands above)
# 1. Export these data into a CSV file on your computer, and re-import them as if this was a dataset provided to you in this format
# 
# ### Faceted plots
# 
# Try to create small multiples (facets, also called trellis) for the plots above, using one or two categories to facet by rows and/or columns.
# 
# ### Different aesthetics
# 
# Try mapping different aesthetics, such as:
# 
# - line type (solid, dashed, etc.)
# - point shape
# - point size
# - line width, opacity, ... see what is available
# 
# ### Different plot types
# 
# Try producing the following kinds of plots:
# 
# - A boxplot
# - A violing plot
# - A heatmap
# 
# ### Fine-tuning
# 
# Read the documentation to figure out how to polish your plot, notably:
# 
# - change the colour scheme/palette
# - change the theme (e.g. dark background, large font size for all the text)
# - suppress the legend
# - render LaTeX-like strings in the axis labels, e.g `$\alpha = \int_0^\infty \beta(x)dx$`
# - optional: add interactive tooltips (in plotly)
# 
# ### Saving plots
# 
# Check the options to save your graphic as 
# 
# - png, with size 6x4 inches and resolution of 300dpi
# - pdf, with transparent background and size 6x4 inches
# - square svg of size 4x4 inches
# 

# In[ ]:





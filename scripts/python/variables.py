#!/usr/bin/env python
# coding: utf-8

# # Variables
# 
# Just like the familiar variables $x$ and $y$ in mathematics, we use variables in programming to easily manipulate values. In this section, we introduce the assignment operator `=`, namespaces and naming conventions for variables.

# ## Assign Values to Variables
# 
# We assign a value to a variable using the assignment operator `=`. For example, assign the integer 2 to the variable `x`:

# In[1]:


x = 2


# The assignment operator does not produce any output and so the cell above does not produce any output in a Jupyter notebook. Use the built-in function `print` to display the value assigned to a variable:

# In[2]:


print(x)


# Compute new values using variables and operators:

# In[3]:


1 + x + x**2 + x**3


# Use the built-in function `type` to verify the datatype of the value assigned to a variable:

# In[4]:


pi = 3.14159
type(pi)


# ## Naming Conventions
# 
# We can use any set of letters, numbers and underscores to make variable names however a variable name *cannot* begin with a number. There are many different kinds of naming conventions and we refer to the [Style Guide for Python Code (PEP8)](https://www.python.org/dev/peps/pep-0008/#naming-conventions) for a summary.
# 
# In this book we use `lower_case_with_underscores` variable names and single lowercase letter variable names such as `x`. It is good practice to use descriptive variable names to make your code more readable for other people.
# 
# For example, the distance from Vancouver to Halifax along the Trans-Canada Highway is approximately 5799 kilometres. We write the following code to convert this value to miles:

# In[5]:


distance_km = 5799
miles_per_km = 0.6214
distance_miles = distance_km * miles_per_km
print(distance_miles)


# ## Names to Avoid
# 
# It is good practice to use variable names which describe the value assigned to it. However there are words that we should not use as variable names because these words already have special meaning in Python.

# ### Reserved Words
# 
# Summarized below are the [reserved words in Python 3](https://docs.python.org/3.3/reference/lexical_analysis.html#keywords). Python will raise an error if you try to assign a value to any of these keywords and so you must avoid these as variable names.
# 
# | | | | | |
# | :---: | :---: | :---: | :---: | :---: |
# | `False` | `class` | `finally` | `is` | `return` |
# | `None` |  `continue` | `for` | `lambda` | `try` |
# | `True` | `def` | `from` |  `nonlocal` | `while` |
# | `and` | `del` | `global` | `not` | `with` |
# | `as` | `elif` | `if` | `or` | `yield` |
# | `assert` | `else` | `import` | `pass` | `break` |
# | `except` | `in` | `raise` | | |

# ### Built-in Function Names
# 
# There are several functions which are included in the standard Python library. Do *not* use the names of these functions as variable names otherwise the reference to the built-in function will be lost. For example, do not use `sum`, `min`, `max`, `list` or `sorted` as a variable name. See the full list of [builtin functions](https://docs.python.org/3/library/functions.html).

# ## Jupyter Magic: whos
# 
# The Jupyer magic command [whos](http://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-whos) lists all variables in the current Jupyter notebook and their types:

# In[6]:


x = 2
pi = 3.14159
distance_km = 5799
miles_per_km = 0.6214
distance_miles = distance_km * miles_per_km


# In[7]:


whos


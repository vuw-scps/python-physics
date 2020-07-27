#!/usr/bin/env python
# coding: utf-8

# # Sequences
# 
# The main [sequence types](https://docs.python.org/3/library/stdtypes.html#sequence-types-list-tuple-range) in Python are lists, tuples and range objects. The main differences between these sequence objects are:
# 
# * Lists are [mutable](https://docs.python.org/3/glossary.html#term-mutable) and their elements are usually *homogeneous* (things of the same type making a list of similar objects)
# * Tuples are [immutable](https://docs.python.org/3/glossary.html#term-immutable) and their elements are usually *heterogeneous* (things of different types making a tuple describing a single structure)
# * Range objects are *efficient* sequences of integers (commonly used in `for` loops), use a small amount of memory and yield items only when needed

# ## Lists
# 
# Create a list using square brackets `[ ... ]` with items separated by commas. For example, create a list of square integers, assign it to a variable and use the built-in function `print()` to display the list:

# In[1]:


squares = [1,4,9,16,25]
print(squares)


# Lists may contain data of any type including other lists:

# In[2]:


points = [[0,0],[0,1],[1,1],[0,1]]
print(points)


# ### Index
# 
# Access the elements of a list by their index:

# In[3]:


primes = [2,3,5,7,11,13,17,19,23,29]
print(primes[0])


# Notice that lists are indexed starting at 0:

# In[4]:


print(primes[1])
print(primes[2])
print(primes[6])


# Use negative indices to access elements starting from the end of the list:

# In[5]:


print(primes[-1])
print(primes[-2])


# Since lists are mutable, we may assign new values to entries in a list:

# In[6]:


primes[0] = -1
print(primes)


# Use multiple indices to access entries in a list of lists:

# In[7]:


pairs = [[0,1],[2,3],[4,5],[6,7]]
print(pairs[2][1])


# ### Slice
# 
# Create a new list from a sublist (called a slice):

# In[8]:


fibonacci = [1,1,2,3,5,8,13,21,34,55,89,144]
print(fibonacci[4:7])
print(fibonacci[6:])
print(fibonacci[:-2])


# Notice in the example `fibonacci[4:7]` the slice begins at index 4 and goes up to *but not including* index 7. This makes sense since the length of the slice is then 7 - 4 = 3.
# 
# A slice can skip over entries in a list. For example, create a slice from every third entry from index 0 to 11:

# In[9]:


print(fibonacci[0:11:3])


# ### Concatenate
# 
# The addition operator `+` concatenates lists:

# In[10]:


one = [1]
two = [2,2]
three = [3,3,3]
numbers = one + two + three
print(numbers)


# ### Append
# 
# Add a value to the end of a list using the `append()` list method:

# In[11]:


squares = [1,4,9,16,25]
squares.append(36)
print(squares)


# What is an object method? First, an object in Python (such as a list) contains data as well as functions (called methods) to manipulate that data. Everything in Python is an object! The list `squares` in the cell above contains the integer entries (the data) but it also has methods like `append()` to manipulate the data. We'll see more about objects and methods later on. For now, see the documentation for a complete list of [list methods](https://docs.python.org/3/tutorial/datastructures.html#more-on-lists).

# ## Tuples
# 
# Create a tuple with parentheses `( ... )`:

# In[12]:


today = (2019,7,11)
print(today)


# Indexing, slicing and concatenating work for tuples in the exact same way as for lists:

# In[13]:


print(today[0])
print(today[-1])
print(today[1:3])


# ## Range Objects
# 
# Create a range object with the built-in function `range()`. The parameters `a`, `b` and `step` in `range(a,b,step)` are integers and the function creates an object which represents the sequence of integers from `a` to `b` (exclusively) incremented by `step`. (The parameter `step` may be omitted and is equal to 1 by default.)

# In[14]:


digits_range = range(0,10)
print(digits_range)


# Notice that a range object does not display the values of its entries when printed. This is because a range object is an efficient sequence which yields values only when needed.
# 
# Use the built-in function `list()` to convert a range object to a list:

# In[15]:


digits_range = range(0,10)
digits_list = list(digits_range)
print(digits_list)


# Create a range of even integers and convert it to a list:

# In[16]:


even_list = list(range(0,10,2))
print(even_list)


# ## Unpacking a Sequence
# 
# One of the features of a Python sequence is *unpacking* where we assign all the entries of a sequence to variables in a single operation. For example, create a tuple representing a date and unpack the data as `year`, `month` and and `day`:

# In[17]:


today = (2019,7,11)
year, month, day = today
print(year)
print(month)
print(day)


# ## List Comprehensions
# 
# The built-in function `range()` is an efficient tool for creating sequences of integers but what about an arbitrary sequence? It is very inefficient to create a sequence by manually typing the numbers. For example, simply typing out the numbers from 1 to 20 takes a long time!

# In[18]:


numbers = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
print(numbers)


# Python has a beautiful syntax for creating lists called [list comprehensions](https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions). The syntax is:

# ```python
# [expression for item in iterable]
# ```

# where:
# 
# * `iterable` is a range, list, tuple, or any kind of sequence object
# * `item` is a variable name which takes each value in the iterable
# * `expression` is a Python expression which is calculated for each value of `item`
# 
# Use a list comprehension to create the list from 1 to 20:

# In[19]:


numbers = [n for n in range(1,21)]
print(numbers)


# Create the list of square integers from $1$ to $100$:  

# In[20]:


squares = [n**2 for n in range(1,11)]
print(squares)


# Create the periodic sequence $0,1,2,0,1,2,0,1,2,\dots$ of length 21 (using the remainder operator `%`):

# In[21]:


zero_one_two = [n%3 for n in range(0,21)]
print(zero_one_two)


# ## Built-in Functions for Sequences
# 
# Python has several [built-in functions](https://docs.python.org/3/library/functions.html) for computing with sequences. For example, compute the length of a list:

# In[22]:


len([1,2,3])


# Compute the sum, maximum and minimum of a list of numbers:

# In[23]:


random = [3,-5,7,8,-1]
print(sum(random))
print(max(random))
print(min(random))


# Sort the list:

# In[24]:


sorted(random)


# Sum the numbers from 1 to 100:

# In[25]:


one_to_hundred = range(1,101)
print(sum(one_to_hundred))


# ## Examples
# 
# ### Triangular Numbers
# 
# The formula for the sum of integers from 1 to $N$ (also known as [triangular numbers](https://en.wikipedia.org/wiki/Triangular_number)) is given by:
# 
# $$
# \sum_{k=1}^N k = \frac{N(N+1)}{2}
# $$
# 
# Let's verify the formula for $N=1000$:

# In[26]:


N = 1000
left_side = sum([k for k in range(1,N+1)])
right_side = N*(N+1)/2
print(left_side)
print(right_side)


# Notice the results agree (although the right side is a float since we used division).

# ### Sum of Squares
# 
# The sum of squares (a special case of a [geometric series](https://en.wikipedia.org/wiki/Geometric_series)) is given by the formula:
# 
# $$
# \sum_{k=1}^N k^2 = \frac{N(N+1)(2N+1)}{6}
# $$
# 
# Let's verify the formula for $N=2000$:

# In[27]:


N = 2000
left_side = sum([k**2 for k in range(1,N+1)])
right_side = N*(N+1)*(2*N+1)/6
print(left_side)
print(right_side)


# ### Riemann Zeta Function
# 
# The [Riemann zeta function](https://en.wikipedia.org/wiki/Riemann_zeta_function) is the infinite series
# 
# $$
# \zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s}
# $$
# 
# Its [values](https://en.wikipedia.org/wiki/Riemann_zeta_function#Specific_values) are very mysterious! Let's verify the special value formula
# 
# $$
# \zeta(4) = \sum_{n=1}^{\infty} \frac{1}{n^4} = \frac{\pi^4}{90}
# $$
# 
# Compute the 1000th partial sum of the series:

# In[28]:


terms = [1/n**4 for n in range(1,1001)]
sum(terms)


# Compare to an approximation of $\frac{\pi^4}{90}$:

# In[29]:


3.14159**4/90


# ## Exercises
# 
# 1. The [Maclaurin series](https://en.wikipedia.org/wiki/Taylor_series#Trigonometric_functions) of $\arctan(x)$ is
# 
#     $$
#     \arctan(x) = \sum_{n = 0}^{\infty} \frac{(-1)^nx^{2n + 1}}{2n+1}
#     $$
# 
#     Substituting $x = 1$ gives a series representation of $\pi/4$. Compute the 5000th partial sum of the series to approximate $\pi/4$.
# 
# 2. Compute the 2000th partial sum of the [alternating harmonic series](https://en.wikipedia.org/wiki/Harmonic_series_(mathematics)#Alternating_harmonic_series):
# 
#     $$\sum_{n=1}^{\infty}\frac{(-1)^n}{n}$$
# 
# 5. Write a list comprehension to create the list of lists:
# 
#     $$
#     [[0, 0], [1, 1], [2, 4], [3, 9], [4, 16], [5, 25], [6, 36], [7, 49]]
#     $$

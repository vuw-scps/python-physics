#!/usr/bin/env python
# coding: utf-8

# # Loops
# 
# ## for Loops
# 
# A [for loop](https://docs.python.org/3/reference/compound_stmts.html#for) allows us to execute a block of code multiple times with some parameters updated each time through the loop. A `for` loop begins with the `for` statement:

# In[1]:


iterable = [1,2,3]
for item in iterable:
    # code block indented 4 spaces
    print(item)


# The main points to observe are:
# 
# * `for` and `in` keywords
# * `iterable` is a sequence object such as a list, tuple or range
# * `item` is a variable which takes each value in `iterable`
# * end `for` statement with a colon `:`
# * code block indented 4 spaces which executes once for each value in `iterable`
# 
# For example, let's print $n^2$ for $n$ from 0 to 5:

# In[2]:


for n in [0,1,2,3,4,5]:
    square = n**2
    print(n,'squared is',square)
print('The for loop is complete!')


# Copy and paste this code and any of the examples below into the [Python visualizer](http://www.pythontutor.com/visualize.html#mode=edit) to see each step in a `for` loop!

# ## while Loops
# 
# What if we want to execute a block of code multiple times but we don't know exactly how many times? We can't write a `for` loop because this requires us to set the length of the loop in advance. This is a situation when a [while loop](https://en.wikipedia.org/wiki/While_loop#Python) is useful.
# 
# The following example illustrates a [while loop](https://docs.python.org/3/tutorial/introduction.html#first-steps-towards-programming):

# In[3]:


n = 5
while n > 0:
    print(n)
    n = n - 1


# The main points to observe are:
# 
# * `while` keyword
# * a logical expression followed by a colon `:`
# * loop executes its code block if the logical expression evaluates to `True`
# * update the variable in the logical expression each time through the loop
# * **BEWARE!** If the logical expression *always* evaluates to `True`, then you get an [infinite loop](https://en.wikipedia.org/wiki/While_loop#Python)!
# 
# We prefer `for` loops over `while` loops because of the last point. A `for` loop will never result in an infinite loop. If a loop can be constructed with `for` or `while`, we'll always choose `for`.

# ## Constructing Sequences
# 
# There are several ways to construct a sequence of values and to save them as a Python list. We have already seen Python's list comprehension syntax. There is also the `append` list method described below.
# 
# ### Sequences by a Formula
# 
# If a sequence is given by a formula then we can use a list comprehension to construct it. For example, the sequence of squares from 1 to 100 can be constructed using a list comprehension:

# In[4]:


squares = [d**2 for d in range(1,11)]
print(squares)


# However, we can achieve the same result with a `for` loop and the `append` method for lists:

# In[5]:


# Intialize an empty list
squares = []
for d in range(1,11):
    # Append the next square to the list
    squares.append(d**2)
print(squares)


# In fact, the two examples above are equivalent. The purpose of list comprehensions is to simplify and compress the syntax into a one-line construction.

# ### Recursive Sequences
# 
# We can only use a list comprehension to construct a sequence when the sequence values are defined by a formula. But what if we want to construct a sequence where the next value depends on previous values? This is called a [recursive sequence](https://en.wikipedia.org/wiki/Recursion).
# 
# For example, consider the [Fibonacci sequence](https://en.wikipedia.org/wiki/Fibonacci_number):
# 
# $$
# x_1 = 1, x_2 = 1, x_3 = 2, x_4 = 3, x_5 = 5, ...
# $$
# 
# where
# 
# $$
# x_{n} = x_{n-1} + x_{n-2}
# $$
# 
# We can't use a list comprehension to build the list of Fibonacci numbers, and so we must use a `for` loop with the `append` method instead. For example, the first 15 Fibonacci numbers are:

# In[6]:


fibonacci_numbers = [1,1]
for n in range(2,15):
    fibonacci_n = fibonacci_numbers[n-1] + fibonacci_numbers[n-2]
    fibonacci_numbers.append(fibonacci_n)
    print(fibonacci_numbers)


# ## Computing Sums
# 
# Suppose we want to compute the sum of a sequence of numbers $x_0$, $x_1$, $x_2$, $x_3$, $\dots$, $x_n$. There are at least two approaches:
# 
# 1. Compute the entire sequence, store it as a list $[x_0,x_1,x_2,\dots,x_n]$ and then use the built-in function `sum`.
# 2. Initialize a variable with value 0 (and name it `result` for example), create and add each element in the sequence to `result` one at a time.
# 
# The advantage of the second approach is that we don't need to store all the values at once. For example, here are two ways to write a function which computes the sum of squares.
# 
# For the first approach, use a list comprehension:

# In[7]:


def sum_of_squares_1(N):
    "Compute the sum of squares 1**2 + 2**2 + ... + N**2."
    return sum([n**2 for n in range(1,N + 1)])


# In[8]:


sum_of_squares_1(4)


# For the second approach, use a `for` loop with the initialize-and-update construction:

# In[9]:


def sum_of_squares_2(N):
    "Compute the sum of squares 1**2 + 2**2 + ... + N**2."
    # Initialize the output value to 0
    result = 0
    for n in range(1,N + 1):
        # Update the result by adding the next term
        result = result + n**2
    return result


# In[10]:


sum_of_squares_2(4)


# Again, both methods yield the same result however the second uses less memory!

# ## Computing Products
# 
# There is no built-in function to compute products of sequences therefore we'll use an initialize-and-update construction similar to the example above for computing sums.
# 
# Write a function called `factorial` which takes a positive integer $N$ and return the factorial $N!$.

# In[11]:


def factorial(N):
    "Compute N! = N(N-1) ... (2)(1) for N >= 1."
    # Initialize the output variable to 1
    product = 1
    for n in range(2,N + 1):
        # Update the output variable
        product = product * n
    return product


# Let's test our function for input values for which we know the result:

# In[12]:


factorial(2)


# In[13]:


factorial(5)


# We can use our function to approximate $e$ using the Taylor series for $e^x$:
# 
# $$
# e^x = \sum_{k=0}^{\infty} \frac{x^k}{k!}
# $$
# 
# For example, let's compute the 100th partial sum of the series with $x=1$:

# In[14]:


sum([1/factorial(k) for k in range(0,101)])


# ## Searching for Solutions
# 
# We can use `for` loops to search for integer solutions of equations. For example, suppose we would like to find all representations of a positive integer $N$ as a [sum of two squares](https://en.wikipedia.org/wiki/Sum_of_two_squares_theorem). In other words, we want to find all integer solutions $(x,y)$ of the equation:
# 
# $$
# x^2 + y^2 = N
# $$
# 
# Write a function called `reps_sum_squares` which takes an integer $N$ and finds all representations of $N$ as a sum of squares $x^2 + y^2 = N$ for $0 \leq x \leq y$. The function returns the representations as a list of tuples. For example, if $N = 50$ then $1^2 + 7^2 = 50$ and $5^2 + 5^2 = 50$ and the function returns the list `[(1, 7),(5, 5)]`.
# 
# Let's outline our approach before we write any code:
# 
# 1. Given $x \leq y$, the largest possible value for $x$ is $\sqrt{\frac{N}{2}}$
# 2. For $x \leq \sqrt{\frac{N}{2}}$, the pair $(x,y)$ is a solution if $N - x^2$ is a square
# 3. Define a helper function called `is_square` to test if an integer is square

# In[15]:


def is_square(n):
    "Determine if the integer n is a square."
    if round(n**0.5)**2 == n:
        return True
    else:
        return False

def reps_sum_squares(N):
    '''Find all representations of N as a sum of squares x**2 + y**2 = N.

    Parameters
    ----------
    N : integer

    Returns
    -------
    reps : list of tuples of integers
        List of tuples (x,y) of positive integers such that x**2 + y**2 = N.

    Examples
    --------
    >>> reps_sum_squares(1105)
    [(4, 33), (9, 32), (12, 31), (23, 24)]
    '''
    reps = []
    if is_square(N/2):
        # If N/2 is a square, search up to x = (N/2)**0.5
        max_x = round((N/2)**0.5)
    else:
        # If N/2 is not a square, search up to x = floor((N/2)**0.5)
        max_x = int((N/2)**0.5)
    for x in range(0,max_x + 1):
        y_squared = N - x**2
        if is_square(y_squared):
            y = round(y_squared**0.5)
            # Append solution (x,y) to list of solutions
            reps.append((x,y))
    return reps


# In[16]:


reps_sum_squares(1105)


# What is the smallest integer which can be expressed as the sum of squares in 5 different ways?

# In[17]:


N = 1105
num_reps = 4
while num_reps < 5:
    N = N + 1
    reps = reps_sum_squares(N)
    num_reps = len(reps)
print(N,':',reps_sum_squares(N))


# ## Examples
# 
# ### Prime Numbers
# 
# A positive integer is [prime](https://en.wikipedia.org/wiki/Prime_number) if it is divisible only by 1 and itself. Write a function called `is_prime` which takes an input parameter `n` and returns `True` or `False` depending on whether `n` is prime or not.
# 
# Let's outline our approach before we write any code:
# 
# 1. An integer $d$ divides $n$ if there is no remainder of $n$ divided by $d$.
# 2. Use the modulus operator `%` to compute the remainder.
# 3. If $d$ divides $n$ then $n = d q$ for some integer $q$ and either $d \leq \sqrt{n}$ or $q \leq \sqrt{n}$ (and not both), therefore we need only test if $d$ divides $n$ for integers $d \leq \sqrt{n}$

# In[18]:


def is_prime(n):
    "Determine whether or not n is a prime number."
    if n <= 1:
        return False
    # Test if d divides n for d <= n**0.5
    for d in range(2,round(n**0.5) + 1):
        if n % d == 0:
            # n is divisible by d and so n is not prime
            return False
    # If we exit the for loop, then n is not divisible by any d
    # and therefore n is prime
    return True


# Let's test our function on the first 30 numbers:

# In[19]:


for n in range(0,31):
    if is_prime(n):
        print(n,'is prime!')


# Our function works! Let's find all the primes between 20,000 and 20,100.

# In[20]:


for n in range(20000,20100):
    if is_prime(n):
        print(n,'is prime!')


# ### Divisors
# 
# Let's write a function called `divisors` which takes a positive integer $N$ and returns the list of positive integers which divide $N$.

# In[21]:


def divisors(N):
    "Return the list of divisors of N."
    # Initialize the list of divisors (which always includes 1)
    divisor_list = [1]
    # Check division by d for d <= N/2
    for d in range(2,N // 2 + 1):
        if N % d == 0:
            divisor_list.append(d)
    # N divides itself and so we append N to the list of divisors
    divisor_list.append(N)
    return divisor_list


# Let's test our function:

# In[22]:


divisors(10)


# In[23]:


divisors(100)


# In[24]:


divisors(59)


# ### Collatz Conjecture
# 
# Let $a$ be a positive integer and consider the recursive sequence where $x_0 = a$ and
# 
# $$
# x_{n+1} = \left\\{ \begin{array}{cl} x_n/2 & \text{if } x_n \text{ is even} \\\\ 3x_n+1 & \text{if } x_n \text{ is odd}  \end{array} \\right.
# $$
# 
# The [Collatz conjecture](https://en.wikipedia.org/wiki/Collatz_conjecture) states that this sequence will *always* reach 1. For example, if $a = 10$ then $x_0 = 10$, $x_1 = 5$, $x_2 = 16$, $x_3 = 8$, $x_4 = 4$, $x_5 = 2$ and $x_6 = 1$.
# 
# Write a function called `collatz` which takes one input parameter `a` and returns the sequence of integers defined above and ending with the first occurrence $x_n=1$.

# In[25]:


def collatz(a):
    "Compute the Collatz sequence starting at a and ending at 1."
    # Initialize list with first value a
    sequence = [a]
    # Compute values until we reach 1
    while sequence[-1] > 1:
        # Check if the last element in the list is even
        if sequence[-1] % 2 == 0:
            # Compute and append the new value
            sequence.append(sequence[-1] // 2)
        else:
            # Compute and append the new value
            sequence.append(3*sequence[-1] + 1)
    return sequence


# Let's test our function:

# In[26]:


print(collatz(10))


# In[27]:


collatz(22)


# The Collatz conjecture is quite amazing. No matter where we start, the sequence always terminates at 1!

# In[28]:


a = 123456789
seq = collatz(a)
print("Collatz sequence for a =",a)
print("begins with",seq[:5])
print("ends with",seq[-5:])
print("and has",len(seq),"terms.")


# Which $a < 1000$ produces the longest sequence?

# In[29]:


max_length = 1
a_max = 1
for a in range(1,1001):
    seq_length = len(collatz(a))
    if seq_length > max_length:
        max_length = seq_length
        a_max = a
print('Longest sequence begins with a =',a_max,'and has length',max_length)


# ## Exercises
# 
# 1. [Fermat's theorem on the sum of two squares](https://en.wikipedia.org/wiki/Fermat%27s_theorem_on_sums_of_two_squares) states that every prime number $p$ of the form $4k+1$ can be expressed as the sum of two squares. For example, $5 = 2^2 + 1^2$ and $13 = 3^2 + 2^2$. Find the smallest prime greater than $2019$ of the form $4k+1$ and write it as a sum of squares. (Hint: Use the functions `is_prime` and `reps_sum_squares` from this section.)
# 
# 2. What is the smallest prime number which can be represented as a sum of squares in 2 different ways?
# 
# 3. What is the smallest integer which can be represented as a sum of squares in 3 different ways?
# 
# 4. Write a function called `primes_between` which takes two integer inputs $a$ and $b$ and returns the list of primes in the closed interval $[a,b]$.
# 
# 5. Write a function called `primes_d_mod_N` which takes four integer inputs $a$, $b$, $d$ and $N$ and returns the list of primes in the closed interval $[a,b]$ which are congruent to $d$ mod $N$ (this means that the prime has remainder $d$ after division by $N$). This kind of list is called [primes in an arithmetic progression](https://en.wikipedia.org/wiki/Dirichlet%27s_theorem_on_arithmetic_progressions).
# 
# 6. Write a function called `reciprocal_recursion` which takes three positive integers $x_0$, $x_1$ and $N$ and returns the sequence $[x_0,x_1,x_2,\dots,x_N]$ where
# 
#     $$
#     x_n = \frac{1}{x_{n-1}} + \frac{1}{x_{n-2}}
#     $$
# 
# 7. Write a function called `root_sequence` which takes input parameters $a$ and $N$, both positive integers, and returns the $N$th term $x_N$ in the sequence:
# 
#     $$
#     \begin{align}
#     x_0 &= a \\\
#     x_n &= 1 + \sqrt{x_{n-1}}
#     \end{align}
#     $$
# 
#     Does the sequence converge to different values for different starting values $a$?
# 
# 8. Write a function called `fib_less_than` which takes one input $N$ and returns the list of Fibonacci numbers less than $N$.
# 
# 9. Write a function called `fibonacci_primes` which takes an input parameter $N$ and returns the list of Fibonacci numbers less than $N$ which are also prime numbers.
# 
# 10. Let $w(N)$ be the number of ways $N$ can be expressed as a sum of two squares $x^2 + y^2 = N$ with $1 \leq x \leq y$. Then
# 
#     $$
#     \lim_{N \to \infty} \frac{1}{N} \sum_{n=1}^{N} w(n) = \frac{\pi}{8}
#     $$
# 
#     Compute the left side of the formula for $N=100$ and compare the result to $\pi / 8$.
# 
# 11. A list of positive integers $[a,b,c]$ (with $1 \leq a < b$) are a [Pythagorean triple](https://en.wikipedia.org/wiki/Pythagorean_triple) if $a^2 + b^2 = c^2$. Write a function called `py_triples` which takes an input parameter $N$ and returns the list of Pythagorean triples `[a,b,c]` with $c \leq N$.

# Modules and Packages

*Under construction*

<!--

A [module](https://docs.python.org/3/tutorial/modules.html) is simply a file containing Python code which defines variables, functions and classes, and a [package](https://docs.python.org/3/tutorial/modules.html#packages) is a collection of modules.

Use the keyword [import](https://docs.python.org/3/tutorial/modules.html#more-on-modules) to import a module or packages into your Python environment. We access variables, functions, classes, etc. from a module or package using the dot notation.

For example, let's import the [math module](https://docs.python.org/3/library/math.html) and do some calculations with the variable `math.pi` and the functions `math.sin` and `math.cos`:


```python
import math
```


```python
math.pi
```




    3.141592653589793




```python
math.cos(0)
```




    1.0




```python
math.sin(math.pi/2)
```




    1.0

## 1. Python modules and packages

So far we have been using the standard [Python library](https://docs.python.org/3/library/) consisting of [builtin functions](https://docs.python.org/3/library/functions.html) (like `sum`, `len`, `round`, etc) and [buitlin datatypes](https://docs.python.org/3/library/stdtypes.html) (like `int`, `float`, `list`, etc). But what if we want to do more like work with matrices and exponential functions and trigonometric functions? Python packages NumPy, SciPy, matplotlib, pandas and many more have been built for scientific computing and many other applications!

What is a package or a module? A [module](https://docs.python.org/3/tutorial/modules.html) is simply a collection of functions and other things saved to a `.py` file (just a text file containing Python code). A [package](https://docs.python.org/3/tutorial/modules.html#packages) is a whole collection of modules.

### Creating our own number theory module

Let's make our own module by assembling all our functions related to number theory! Create a new text file (in the same directory as this notebook) called `number_theory.py` (the extension `.py` let's our operating system know that it's a text file with Python code) and paste all the functions in the cell below:


```python
# number_theory.py

# A module containing functions related to number theory
# UBC Math 210 Introduction to Mathematical Computing
# January 27, 2017

def factorial(N):
    """Compute N!=N(N-1) ... (2)(1)."""
    # Initialize the outpout variable to 1
    product = 1
    for n in range(2,N + 1):
        # Update the output variable
        product = product * n
    return product

def n_choose_k(N,K):
    """Compute N choose K."""
    return factorial(N) // (factorial(N - K) * factorial(K))

def divisors(N):
    """Return the list of divisors of N."""
    # Initialize the list of divisors
    divisor_list = [1]
    # Check division by d for d <= N/2
    for d in range(2,N // 2 + 1):
        if N % d == 0:
            divisor_list.append(d)
    divisor_list.append(N)
    return divisor_list

def is_square(N):
    """Determine is N is square."""
    return N == round(N**(0.5))**2

def rep_as_squares(N):
    """Find all representations of N as a sum of squares a**2 + b**2 = N."""
    reps = []
    stop = int((N/2)**0.5) + 1 # a must be less than \sqrt{N/2}
    for a in range(1,stop):
        b_squared = N - a**2
        if is_square(b_squared):
            b = round((b_squared)**(0.5))
            reps.append([a,b])
    return reps

def collatz(a):
    """Compute the Collatz sequence starting at a."""
    # Initialze the sequence with the fist value a.
    x_list = [a]
    # Continue computing values in the sequence until we reach 1.
    while x_list[-1] != 1:
        # Check if the last element in the list is even
        if x_list[-1] % 2 == 0:
            # Compute and append the new values
            x_list.append(x_list[-1] // 2)
        else:
            # Compute and append the new values
            x_list.append(3*x_list[-1] + 1)
    return x_list

def is_prime(N):
    "Determine whether or not N is a prime number."
    if N <= 1:
        return False
    # N is prime if N is only divisible by 1 and itself
    # We should test whether N is divisible by d for all 1 < d < N
    for d in range(2,N):
        # Check if N is divisible by d
        if N % d == 0:
            return False
    # If we exit the for loop, then N is not divisible by any d
    # Therefore N is prime
    return True
```

### Importing a module

We import our module using the `import` keyword:


```python
import number_theory
```

Now all the functions in the file `number_theory.py` are available to us! We use the dot notation to access them by name!


```python
number_theory.is_prime(2017)
```




    True




```python
number_theory.rep_as_squares(2017)
```




    [[9, 44]]




```python
9**2 + 44**2
```




    2017




```python
number_theory.is_square(9)
```




    True




```python
number_theory.rep_as_squares(2000000)
```




    [[200, 1400], [584, 1288], [680, 1240], [1000, 1000]]



### Example: Sums of squares

Let's find the smallest integer that can be expressed as a sum of squares in 4 different ways.


```python
# After some trial and error, we find the smallest integer
# that can be written as a sum of squares in 4 ways
for N in range(1,1200):
    reps = number_theory.rep_as_squares(N)
    if len(reps) > 2:
        print(N,' - ',reps)
```

    325  -  [[1, 18], [6, 17], [10, 15]]
    425  -  [[5, 20], [8, 19], [13, 16]]
    650  -  [[5, 25], [11, 23], [17, 19]]
    725  -  [[7, 26], [10, 25], [14, 23]]
    845  -  [[2, 29], [13, 26], [19, 22]]
    850  -  [[3, 29], [11, 27], [15, 25]]
    925  -  [[5, 30], [14, 27], [21, 22]]
    1025  -  [[1, 32], [8, 31], [20, 25]]
    1105  -  [[4, 33], [9, 32], [12, 31], [23, 24]]



-->

<!--


## 2. Modules

A [module](https://docs.python.org/3/tutorial/modules.html) is a file (with extension `.py`) containing Python code. We import modules with the `import` keyword and access the functions and variables defined in the module by the .dot notation.

For example, let's create a module called `poly` which puts together all the polynomial functions from Assignment 1. Open a new `.txt` file, paste the code below into the empty file, and save as `poly.py`. Make sure to save the file in the same directory as this notebook!

```
# poly.py

def poly_diff(p):
    '''Compute the derivative of a polynomial p(x) = a0 + a1*x + a2*x**2 + ... + an*x**n.

    Input:
        p : list of length n+1 [a0,a1,a2,...,an] representing
            a polynomial p(x) = a0 + a1*x + a2*x**2 + ... + an*x**n.
    Output:
        List [a1,2*a2,...,n*an] of length n (or [0] if n=0) representing the derivative p'(x).
    Example:
        >>> poly_diff([1,2,3,4])
        [2, 6, 12]
    '''
    deg_p = len(p) - 1
    if deg_p == 0:
        return [0]
    else:
        return [p[k]*k for k in range(1,deg_p+1)]

def poly_eval(p,a):
    '''Evaluate p(a) for p(x) = a0 + a1*x + a2*x**2 + ... + an*x**n.

    Input:
        p : list of length n+1 [a0,a1,a2,...,an] representing
            a polynomial p(x) = a0 + a1*x + a2*x**2 + ... + an*x**n.
        a : number
    Output:
        Polynomial p(x) evaluated at a.
    Example:
        >>> poly_eval([1,2,3,4],-2)
        -23
    '''
    return sum([p[k]*a**k for k in range(0,len(p))])

def poly_max(p,a,b,N):
    '''Approximate the maximum value of p(x) = a0 + a1*x + a2*x**2 + ... + an*x**n in the interval [a,b].

    Input:
        p : list of length n+1 [a0,a1,a2,...,an] representing
            a polynomial p(x) = a0 + a1*x + a2*x**2 + ... + an*x**n.
        a,b : numbers defining the interval [a,b]
        N : positive integer defining the length of the partition x0 = a, x1, x2, ... , xN = b
    Output:
        Maximum value from the list of values p(x0), p(x1), p(x2), ... , p(xN).
    Example:
        >>> poly_max([1,0,-1],-1,1,10)
        1.0
    '''
    h = (b - a)/N # Length of each subinterval
    x_values = [a + h*k for k in range(0,N+1)] # Partition interval x0 = a, x1, x2, ... , xN = b
    y_values = [poly_eval(p,x) for x in x_values] # Compute y values y = p(x)
    return max(y_values)
```

Let's import our module! The [import](https://docs.python.org/3/reference/import.html) keyword loads our file into the current Python environment, creates a module object named `poly` and assigns all the functions in our file to the variable name `poly`.


```python
import poly
```

We can use the Jupyter magic `whos` to see that the module name is available in the current Python environment:


```python
whos
```

    Variable             Type      Data/Info
    ----------------------------------------
    course               str       MATH 210 Introduction to Mathematical Computing
    euler                str       Euler's Method
    letter               str       h
    lyrics               str       To the left, to the left\<...>ght it please don't touch
    password             str       syzygy
    poly                 module    <module 'poly' from '/hom<...>8/notes-week-05/poly.py'>
    sentence             str       The quick brown fox jumped over the lazy dog.
    today                str       It's a rainy day.
    uppercase_sentence   str       THE QUICK BROWN FOX JUMPED OVER THE LAZY DOG.
    word                 str       Math


Use the built-in function `dir` to list the functions available in the module:


```python
dir(poly)
```




    ['__builtins__',
     '__cached__',
     '__doc__',
     '__file__',
     '__loader__',
     '__name__',
     '__package__',
     '__spec__',
     'poly_diff',
     'poly_eval',
     'poly_max']



The items with double underscores `__` are methods attached to all module objects.

Finally, let's verify the type of `poly`:


```python
type(poly)
```




    module



Use our module to differentiate $p(x) = 1 - 2x + 3x^2 - 4x^3$:


```python
p = [1,-2,3,-4]
poly.poly_diff(p)
```




    [-2, 6, -12]



Use our module to compute $q(11)$ for $q(x) = 3 + x^2 - x^5$:


```python
poly.poly_eval([3,0,1,0,0,-1],11)
```




    -160927



Use our module to approximate the maximum of $f(x) = 5 + 2x + x^2 - 10x^3$ on the interval $[-10,10]$:


```python
poly.poly_max([5,2,1,-10],-10,10,1000)
```




    10085.0


-->

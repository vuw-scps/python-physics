# Functions

A [function](https://docs.python.org/3/tutorial/controlflow.html#defining-functions) takes input parameters, executes a series of computations with those inputs and then returns a final output value. Functions give us an efficient way to save and reuse a block of code over and over again with different input values. In this section, we summarize the built-in functions in the standard Python library and then we discuss how to define our own functions.

## Built-in Functions

The standard Python library has a collection of [built-in functions](https://docs.python.org/3/library/functions.html) ready for us to use. We have already seen a few of these functions in previous sections such as `type()`, `print()` and `sum()`. The following is a list of built-in functions that we'll use most often:

| Function | Description |
| ---: | :--- |
| `print(object)` | print `object` to output |
| `type(object)` | return the type of `object` |
| `abs(x)` | return the absolute value of `x` (or modulus if `x` is complex) |
| `int(x)` | return the integer constructed from float `x` by truncating decimal |
| `len(sequence)` | return the length of the `sequence` |
| `sum(sequence)` | return the sum of the entries of `sequence` |
| `max(sequence)`  | return the maximum value in `sequence` |
| `min(sequence)` | return the minimum value in `sequence` |
| `range(a,b,step)` | return the range object of integers from `a` to `b` (exclusive) by `step` |
| `list(sequence)` | return a list constructed from `sequence` |
| `sorted(sequence)` | return the sorted list from the items in `sequence` |
| `reversed(sequence)` | return the reversed iterator object from the items in `sequence` |
| `enumerate(sequence)` | return the enumerate object constructed from `sequence` |
| `zip(a,b)` | return an iterator that aggregates items from sequences `a` and `b` |

Use the function `print()` to display values:


```python
pi = 3.14159
print(pi)
```

    3.14159


Use the function `type()` to see the datatype of a value:


```python
type(pi)
```




    float



Use the function `abs()` to compute the absolute value of a real number:


```python
x = -2019
abs(x)
```




    2019



Or compute the magnitude of a complex number:


```python
z = 3 - 4j
abs(z)
```




    5.0



Use the function `int()` to truncate a float into an int:


```python
pi = 3.14159
int(pi)
```




    3



The function truncates floats always towards 0:


```python
c = -1.2345
int(c)
```




    -1



Use the function `len()` to compute the length of a sequence:


```python
primes = [2,3,5,7,11,13,17,19,23,29,31,37,41]
len(primes)
```




    13



Use the function `sum()` to compute the sum of a sequence:


```python
one_to_hundred = range(1,101)
sum(one_to_hundred)
```




    5050



Use the functions `max()` and `min()` to compute the maximum and minimum values in a sequence.


```python
random = [8,27,3,7,6,14,28,19]
print(max(random))
print(min(random))
```

    28
    3


Use the function `list()` to convert a sequence (such as a range or a tuple) into a list:


```python
list(range(0,10,2))
```




    [0, 2, 4, 6, 8]



Use the function `sorted()` to sort a sequence:


```python
sorted_random = sorted(random)
print(random)
print(sorted_random)
```

    [8, 27, 3, 7, 6, 14, 28, 19]
    [3, 6, 7, 8, 14, 19, 27, 28]


Use the function `reversed()` to reverse the order of a sequence:


```python
reversed_random = list(reversed(random))
print(random)
print(reversed_random)
```

    [8, 27, 3, 7, 6, 14, 28, 19]
    [19, 28, 14, 6, 7, 3, 27, 8]


Use the function `enumerate()` to enumerate a sequence:


```python
squares = [n**2 for n in range(0,6)]
print(squares)
enum_squares = list(enumerate(squares))
print(enum_squares)
```

    [0, 1, 4, 9, 16, 25]
    [(0, 0), (1, 1), (2, 4), (3, 9), (4, 16), (5, 25)]


Use the function `zip()` to combine sequences into a list of pairs:


```python
random_1 = [-2,4,0,5]
random_2 = [7,-1,9,3]
random_zip = list(zip(random_1,random_2))
print(random_zip)
```

    [(-2, 7), (4, -1), (0, 9), (5, 3)]


Notice in the last three examples `reversed()`, `enumerate()` and `zip()` we use the function `list()` to create a list from the output of each function. This is because these functions return *iterator* objects (similar to range objects) which only yield values when explicitly told to do so.

## Defining Functions

Let's begin with a simple example. Define a function which returns the average of a sequence of numbers:


```python
def average(x):
    "Compute the average of the values in the sequence x."
    sum_x = sum(x)
    length_x = len(x)
    return sum_x / length_x
```

The main points to observe are:

1. Start the function definition with the `def` keyword.
2. Follow `def` with the name of the function.
3. Follow the function name with the list of input parameters separated by commas and within parentheses.
4. End the `def` statement with a colon `:`.
5. Indent the body of the function by 4 spaces.
6. Use the `return` keyword to specify the output of the function (but it is not always necessary).
7. The second line is a [documentation string](https://docs.python.org/3/tutorial/controlflow.html#documentation-strings) (enclosed in quotation marks " ... ") which describes the function.

In Python, code blocks are defined using [indentation](https://www.python.org/dev/peps/pep-0008/#indentation). This means that lines of code indented the same amount are considered one block. In the example above, the four indented lines below the `def` statement form the body of the function.

Notice that there is no output when we execute the cell containing the function definition. This is because we've only defined the function and it's waiting for us to use it! We need to call the function with values for the input parameters and then the function will compute and return the output value.

Let's test our function:


```python
average([1,2,3,4])
```




    2.5



The function returns the expected value. Success!

## Documentation Strings

The first line after the `def` statement in a function definition should be a [documentation string](https://docs.python.org/3/tutorial/controlflow.html#documentation-strings) (or docstring). A docstring is text (enclosed in double quotes `" ... "` or triple quotes ` ''' ... ''' `) which describes your function. Use triple quotes for a multiline docstring. See the [Python documentation](https://www.python.org/dev/peps/pep-0257/) for all the conventions related to documentation strings.

A helpful feature of the Jupyter notebook is the question mark operator `?`. This will display the docstring of a function. Keep this in mind when writing your docstrings: other people will read your docstring to learn how to use your function.

For example,  use the question mark `?` to view the documentation for the built-in function `sum()`:


```python
sum?
```

I recommend (but it's up to you) a style similar to NumPy's style guide for docstrings:

```python
def function_name(param1,param2,param3):
    '''First line is a one-line general summary.

    A longer paragraph describing the function and
    relevant equations or algorithms used in the
    function.

    Parameters
    ----------
    param1 : datatype
        Describe the parameter.
    param2 : datatype
        Describe the parameter.
    param3 : datatype
        Describe the parameters and continue with more
        details if necessary on a new set of indented lines.

    Returns
    -------
    datatype
        A description of the output of the function and also
        describe special behaviour.

    Examples
    --------
    >>> function_name(1,2,3)
    1.2345
    '''

```

See these [examples](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) and these [examples](https://google.github.io/styleguide/pyguide.html?showone=Comments#Comments).

## Keyword Arguments

When we define functions, we list the input parameters. These are called positional parameters (or positional arguments) because the position in the `def` statement determines which parameter is which.


```python
def poly(x,y):
    "Compute x + y**2."
    return x + y**2
```


```python
poly(1,2)
```




    5




```python
poly(2,1)
```




    3



A [keyword argument](https://docs.python.org/3/tutorial/controlflow.html#keyword-arguments) allows us to insert default values for some parameters and we call them by name and the order doesn't matter.


```python
def greeting(first_name,last_name,salutation='Hello, '):
    return "{0}{1} {2}!".format(salutation, first_name, last_name)
```


```python
greeting('Patrick','Walls')
```




    'Hello, Patrick Walls!'




```python
greeting('Walls','Patrick')
```




    'Hello, Walls Patrick!'




```python
greeting('LeBron','James',salutation='I love you ')
```




    'I love you LeBron James!'



In this function, `first_name` and `last_name` are positional arguments and `saluation` is a keyword argument.

For example, the function `pandas.read_csv` in the `pandas` package has *many* keyword arguments:


```python
import pandas as pd
```


```python
pd.read_csv?
```

So *many* keyword arguments! The keyword arguments I use most often are `encoding`, `skiprows` and `usecols`.

## Comments

[Comments](https://docs.python.org/3/tutorial/introduction.html) in a Python program are plain text descriptions of Python code which explain code to the reader. Python will ignore lines which begin with the hash symbol `#` and so we use the hash symbol to write comments to explain the steps in a program. See the examples below.

## Examples

### Area of a Triangle

Let's define a function called `area_triangle` which takes an input parameter `vertices` which is a list of tuples representing the vertices of a triangle and returns the area of the triangle using [Heron's Formula](https://en.wikipedia.org/wiki/Heron%27s_formula):

$$
A = \sqrt{s(s-a)(s-b)(s-c)}
$$

where $a$, $b$ and $c$ are the side lengths and $s$ is the semiperimeter

$$
s = \frac{a+b+c}{2}
$$


```python
def area_triangle(vertices):
    '''Compute the area of the triangle with given vertices.

    Parameters
    ----------
    vertices : list of tuples of numbers
        The vertices of a triangle [(x1,y1),(x2,y2),(x3,y3)].

    Returns
    -------
    float
        Area of the triangle computed by Heron's formula.

    Examples
    --------
    >>> area_triangle([(0,0),(3,0),(3,4)])
    6.0
    >>> area_triangle([(-1,2),(-3,-1),(4,1)])
    8.499999999999996
    '''
    # Find the x distance between vertices 0 and 1
    a_x = abs(vertices[0][0] - vertices[1][0])
    # Find the y distance between vertices 0 and 1
    a_y = abs(vertices[0][1] - vertices[1][1])
    # Compute length of side a
    a = (a_x**2 + a_y**2)**0.5

    # Find the x distance between vertices 1 and 2
    b_x = abs(vertices[1][0] - vertices[2][0])
    # Find the y distance between vertices 1 and 2
    b_y = abs(vertices[1][1] - vertices[2][1])
    # Compute length of side b
    b = (b_x**2 + b_y**2)**0.5

    # Find the x distance between vertices 0 and 2
    c_x = abs(vertices[0][0] - vertices[2][0])
    # Find the y distance between vertices 0 and 2
    c_y = abs(vertices[0][1] - vertices[2][1])
    # Compute length of side c
    c = (c_x**2 + c_y**2)**0.5

    # Compute semiperimeter
    s = (a + b + c)/2
    # Compute area
    area = (s*(s - a)*(s - b)*(s - c))**0.5

    return area
```

Let's test our function. We know that the area of a right angle triangle with sides of length 1 and hypotenuse $\sqrt{2}$ has area $0.5$.


```python
area_triangle([(0,0),(0,1),(1,0)])
```




    0.49999999999999983



Let's test again on another triangle with base $b=3$ and height $h=4$ and therefore its area is $A=3(4)/2=6$.


```python
area_triangle([(0,0),(3,0),(1,4)])
```




    6.000000000000003



The function `area_triangle` returns the expected values. Success!

### Riemann Zeta Function

The [Riemann zeta function](https://en.wikipedia.org/wiki/Riemann_zeta_function) is the infinite sum

$$
\zeta(s) = \sum_{n = 1}^{\infty} \frac{1}{n^s}
$$

Write a function called `zeta` which takes 2 input parameters `s` and `N` and returns the partial sum:

$$
\sum_{n=1}^N \frac{1}{n^s}
$$


```python
def zeta(s,N):
    "Compute the Nth partial sum of the zeta function at s."
    terms = [1/n**s for n in range(1,N+1)]
    partial_sum = sum(terms)
    return partial_sum
```

Let's test our function on input values for which we know the result:


```python
zeta(1,1)
```




    1.0




```python
zeta(2,2)
```




    1.25



Now let's use our function to approximate [special values of the Riemann zeta function](https://en.wikipedia.org/wiki/Riemann_zeta_function#Specific_values):

$$
\zeta(2) = \frac{\pi^2}{6} \hspace{10mm} \text{and} \hspace{10mm} \zeta(4) = \frac{\pi^4}{90}
$$

Compute the partial sum for $s=2$ and $N=100000$:


```python
zeta(2,100000)
```




    1.6449240668982423



Compare to an approximation of the special value $\pi^2/6$:


```python
3.14159265**2/6
```




    1.6449340630890041



Compute the partial sum for $s=4$ and $N=100000$:


```python
zeta(4,100000)
```




    1.082323233710861



Compare to an approximation of the special value $\pi^4/90$:


```python
3.14159265**4/90
```




    1.0823232287641997



### Harmonic Mean

Write a function called `harmonic_mean` which takes an input parameter `s`, a list of numbers $x_1, \dots, x_n$ of length $n$, and returns the [harmonic mean](https://en.wikipedia.org/wiki/Harmonic_mean) of the sequence:

$$
\frac{n}{\frac{1}{x_1} + \frac{1}{x_2} + \cdots + \frac{1}{x_n}}
$$


```python
def harmonic_mean(s):
    "Compute the harmonic mean of the numbers in the sequence s."
    n = len(s)
    terms = [1/s[i] for i in range(0,n)]
    result = n/sum(terms)
    return result
```

Let's test our function:


```python
harmonic_mean([1,1,1,1])
```




    1.0




```python
harmonic_mean([1,2,3])
```




    1.6363636363636365



### Riemann Sums

Write a function called `mn_integral` which takes input parameters `m`, `n`, `a`, `b` and `N` and returns the (right) [Riemann sum](https://en.wikipedia.org/wiki/Riemann_sum):
$$
\int_a^b f(x) \, dx \approx \sum_{k=1}^N f(x_k) \Delta x \ \ , \ \ f(x) = \frac{x^m + 1}{x^n + 1}
$$

and $\Delta x = (b-a)/N$ and $x_k = a + k \Delta x$.


```python
def mn_integral(m,n,a,b,N):
    '''Compute the (right) Riemann sum for the function
    f(x) = (x^m + 1)/(x^n + 1) on interval [a,b] with a
    partition of N subintervals of equal size.

    Parameters
    ----------
    m , n : numbers
        Parameters in function f(x) = (x^m + 1)/(x^n + 1)
    a , b : numbers
        Limits of integration.
    N : integer
        Size of partition of interval [a,b].

    Returns
    -------
    float
        The (right) Riemann sum of f(x) from a to b
        using a partition of size N.

    Examples
    --------
    >>> mn_integral(0,1,0,1,2)
    1.1666666666666665
    >>> mn_integral(1,2,0,1,100000)
    1.1319717536649336
    '''
    # Compute the width of subintervals
    delta_x = (b - a)/N

    # Create N+1 evenly spaced x values from a to b
    x = [a + k*delta_x for k in range(0,N+1)]

    # Compute terms of the sum
    terms = [(x[k]**m + 1)/(x[k]**n + 1)*delta_x for k in range(1,N+1)]

    # Compute the sum
    riemann_sum = sum(terms)

    return riemann_sum
```

Let's test our function on input for which we know the result. Let $m=0$, $n=1$, $a=0$, $b=1$ and $N=2$. Then $x_0 = 0$, $x_1 = 1/2$, $x_2 = 1$ and $\Delta x = 1/2$, and we compute:

$$
\begin{aligned}
\sum_{k=1}^N f(x_k) \Delta x &= \sum_{k=1}^2 \frac{x_k^0 + 1}{x_k^1 + 1} \Delta x \\\
&= \frac{2}{(1/2) + 1} \cdot \frac{1}{2} + \frac{2}{1 + 1} \cdot \frac{1}{2} \\\
&= \frac{7}{6}
\end{aligned}
$$


```python
mn_integral(0,1,0,1,2)
```




    1.1666666666666665




```python
7/6
```




    1.1666666666666667



Let's test our function on another example. Let $m=1$, $n=2$, $a=0$, and $b=1$. We can solve this integral exactly:

$$
\begin{aligned}
\int_0^1 \frac{x + 1}{x^2 + 1} dx &= \int_0^1 \frac{x}{x^2 + 1} dx + \int_0^1 \frac{1}{x^2 + 1} dx \\\
&= \left. \left( \frac{1}{2} \ln(x^2 + 1) + \arctan x \right) \right|_0^1 \\\
&= \frac{1}{2} \ln(2) + \frac{\pi}{4}
\end{aligned}
$$

Approximate this integral with a Riemann sum for $N=100000$:


```python
mn_integral(1,2,0,1,100000)
```




    1.1319717536649336



Since $\pi \approx 3.14159265$ and $\ln(2) \approx 0.69314718$, we compare to the approximation:


```python
0.5*0.69314718 + 3.14159265/4
```




    1.1319717525000002



Our function computes the expected values!

## Exercises

1. Write a function called `power_mean` which takes input parameters `sequence` and `p` where `sequence` is a list of positive real numbers $x_1, \dots, x_n$ and `p` is a nonzero number. The function returns the [power mean with exponent p](https://en.wikipedia.org/wiki/Generalized_mean):

    $$
    \left( \frac{1}{n} \sum_{i=1}^n x_i^p \right)^{1/p}
    $$

    Plug in large positive values of $p$ and various lists of numbers to verify

    $$
    \lim_{p \to \infty} \left( \frac{1}{n} \sum_{i=1}^n x_i^p \right)^{1/p} = \max \{x_1, \dots, x_n \}
    $$

    Plug in large negative values of $p$ and various lists of numbers to verify

    $$
    \lim_{p \to -\infty} \left( \frac{1}{n} \sum_{i=1}^n x_i^p \right)^{1/p} = \min \{x_1, \dots, x_n \}
    $$

2. Write a function called `arctan_taylor` which takes input parameters `x` and `N` and return the Taylor polynomial of degree $N$ of the function $\arctan x$ evaluated at `x`:

    $$
    \sum_{k=0}^N (-1)^k \frac{x^{2k + 1}}{2k + 1}
    $$

3. Write a function called `zips` which takes input parameters `a` and `b`, where `a` and `b` are lists of the equal length, and returns the list of tuples which aggregates the sequence. (In other words, write your own version of the built-in function `zip()`... without using `zip()` of course.) For example `zips([-1,3,4,0],[5,7,1,-9])` returns the list `[(-1, 5), (3, 7), (4, 1), (0, -9)]`.

4. Write a function called `sqrt_integral` which takes input parameters `u`, `p` and `N` and returns the Riemann sum (using the *midpoints* $x_k^\*$ of a partition of size $N$):

    $$
    \int_0^u \frac{1}{\sqrt{1 + x^p}} dx \approx \sum_{k=1}^N \frac{1}{\sqrt{1 + (x_k^*)^p}} \Delta x
    $$

    where $\Delta x = u/N$ and $x_k^* = (x_k + x_{k-1})/2$ for endpoints $x_k = k \Delta x$.

# Simpson's Rule


```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

## Definition

[Simpson's rule](https://en.wikipedia.org/wiki/Simpson%27s_rule) uses a quadratic polynomial on each subinterval of a partition to approximate the function $f(x)$ and to compute the definite integral. This is an improvement over the trapezoid rule which approximates $f(x)$ by a straight line on each subinterval of a partition.

The formula for Simpson's rule is

$$
S_N(f) = \frac{\Delta x}{3} \sum_{i=1}^{N/2} \left( f(x_{2i-2}) + 4 f(x_{2i-1}) + f(x_{2i}) \right)
$$

where $N$ is an *even* number of subintervals of $[a,b]$, $\Delta x = (b - a)/N$ and $x_i = a + i \Delta x$.

## Error Formula

We have seen that the error in a Riemann sum is inversely proportional to the size of the partition $N$ and the trapezoid rule is inversely proportional to $N^2$. The error formula in the theorem below shows that Simpson's rule is even better as the error is inversely proportional to $N^4$.

---

**Theorem** Let $S_N(f)$ denote Simpson's rule

$$
S_N(f) = \frac{\Delta x}{3} \sum_{i=1}^{N/2} \left( f(x_{2i-2}) + 4 f(x_{2i-1}) + f(x_{2i}) \right)
$$

where $N$ is an *even* number of subintervals of $[a,b]$, $\Delta x = (b - a)/N$ and $x_i = a + i \Delta x$. The error bound is

$$
E_N^S(f) = \left| \ \int_a^b f(x) \, dx - S_N(f) \ \right| \leq \frac{(b-a)^5}{180N^4} K_4
$$

where $\left| \ f^{(4)}(x) \ \right| \leq K_4$ for all $x \in [a,b]$.

---

## Implementation

Let's write a function called `simps` which takes input parameters $f$, $a$, $b$ and $N$ and returns the approximation $S_N(f)$. Furthermore, let's assign a default value $N=50$.


```python
def simps(f,a,b,N=50):
    '''Approximate the integral of f(x) from a to b by Simpson's rule.

    Simpson's rule approximates the integral \int_a^b f(x) dx by the sum:
    (dx/3) \sum_{k=1}^{N/2} (f(x_{2i-2} + 4f(x_{2i-1}) + f(x_{2i}))
    where x_i = a + i*dx and dx = (b - a)/N.

    Parameters
    ----------
    f : function
        Vectorized function of a single variable
    a , b : numbers
        Interval of integration [a,b]
    N : (even) integer
        Number of subintervals of [a,b]

    Returns
    -------
    float
        Approximation of the integral of f(x) from a to b using
        Simpson's rule with N subintervals of equal length.

    Examples
    --------
    >>> simps(lambda x : 3*x**2,0,1,10)
    1.0
    '''
    if N % 2 == 1:
        raise ValueError("N must be an even integer.")
    dx = (b-a)/N
    x = np.linspace(a,b,N+1)
    y = f(x)
    S = dx/3 * np.sum(y[0:-1:2] + 4*y[1::2] + y[2::2])
    return S
```

Let's test our function on integrals for which we know the exact value. For example, we know

$$
\int_0^1 3x^2 dx = 1
$$


```python
simps(lambda x : 3*x**2,0,1,10)
```




    1.0



Test our function again with the integral

$$
\int_0^{\pi/2} \sin(x) dx = 1
$$


```python
simps(np.sin,0,np.pi/2,100)
```




    1.000000000338236



## scipy.integrate.simps

The [SciPy](https://scipy.org/) subpackage [scipy.integrate](https://docs.scipy.org/doc/scipy-0.18.1/reference/integrate.html) contains several functions for approximating definite integrals and numerically solving differential equations. Let's import the subpackage under the name `spi`.


```python
import scipy.integrate as spi
```

The function `scipy.integrate.simps` computes the approximation of a definite integral by Simpson's rule. Consulting the documentation, we see that all we need to do it supply arrays of $x$ and $y$ values for the integrand and `scipy.integrate.simps` returns the approximation of the integral using Simpson's rule.

## Examples

### Approximate ln(2)

Find a value $N$ which guarantees that Simpson's rule approximation $S_N(f)$ of the integral

$$
\int_1^2 \frac{1}{x} dx
$$

satisfies $E_N^S(f) \leq 0.0001$.

Compute

$$
f^{(4)}(x) = \frac{24}{x^5}
$$

therefore $\left| \, f^{(4)}(x) \, \right| \leq 24$ for all $x \in [1,2]$ and so

$$
\frac{1}{180N^4} 24 \leq 0.0001 \ \Rightarrow \ 
\frac{20000}{15N^4} \leq 1 \ \Rightarrow \ 
\left( \frac{20000}{15} \right)^{1/4} \leq N
$$

Compute


```python
(20000/15)**0.25
```




    6.042750794713537



Compute Simpson's rule with $N=8$ (the smallest even integer greater than 6.04)


```python
approximation = simps(lambda x : 1/x,1,2,8)
print(approximation)
```

    0.6931545306545306


We could also use the function `scipy.integrate.simps` to compute the exact same result


```python
N = 8; a = 1; b = 2;
x = np.linspace(a,b,N+1)
y = 1/x
approximation = spi.simps(y,x)
print(approximation)
```

    0.6931545306545306


Verify that $E_N^S(f) \leq 0.0001$


```python
np.abs(np.log(2) - approximation) <= 0.0001
```




    True



## Exercises

*Under construction*

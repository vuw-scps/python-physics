# Bisection Method

The simplest root finding algorithm is the [bisection method](https://en.wikipedia.org/wiki/Bisection_method). The algorithm applies to any continuous function $f(x)$ on an interval $[a,b]$ where the value of the function $f(x)$ changes sign from $a$ to $b$. The idea is simple: divide the interval in two, a solution must exist within one subinterval, select the subinterval where the sign of $f(x)$ changes and repeat.

## Algorithm

The bisection method procedure is:

1. Choose a starting interval $[a_0,b_0]$ such that $f(a_0)f(b_0) < 0$.
2. Compute $f(m_0)$ where $m_0 = (a_0+b_0)/2$ is the midpoint.
3. Determine the next subinterval $[a_1,b_1]$:
    1. If $f(a_0)f(m_0) < 0$, then let $[a_1,b_1]$ be the next interval with $a_1=a_0$ and $b_1=m_0$.
    2. If $f(b_0)f(m_0) < 0$, then let $[a_1,b_1]$ be the next interval with $a_1=m_0$ and $b_1=b_0$.
4. Repeat (2) and (3) until the interval $[a_N,b_N]$ reaches some predetermined length.
5. Return the midpoint value $m_N=(a_N+b_N)/2$.

A solution of the equation $f(x)=0$ in the interval $[a,b]$ is guaranteed by the <a href="https://en.wikipedia.org/wiki/Intermediate_value_theorem">Intermediate Value Theorem</a> provided $f(x)$ is continuous on $[a,b]$ and $f(a)f(b) < 0$. In other words, the function changes sign over the interval and therefore must equal 0 at some point in the interval $[a,b]$.

## Absolute Error

The bisection method does not (in general) produce an exact solution of an equation $f(x)=0$. However, we can give an estimate of the absolute error in the approxiation.

---

**Theorem**. Let $f(x)$ be a continuous function on $[a,b]$ such that $f(a)f(b) < 0$. After $N$ iterations of the biection method, let $x_N$ be the midpoint in the $N$th subinterval $[a_N,b_N]$

$$
x_N = \frac{a_N + b_N}{2}
$$

There exists an exact solution $x_{\mathrm{true}}$ of the equation $f(x)=0$ in the subinterval $[a_N,b_N]$ and the absolute error is

$$
\left| \ x_{\text{true}} - x_N \, \right| \leq \frac{b-a}{2^{N+1}}
$$

---

Note that we can rearrange the error bound to see the minimum number of iterations required to guarantee absolute error less than a prescribed $\epsilon$:

\begin{align}
\frac{b-a}{2^{N+1}} & < \epsilon \\\
\frac{b-a}{\epsilon} & < 2^{N+1} \\\
\ln \left( \frac{b-a}{\epsilon} \right) & < (N+1)\ln(2) \\\
\frac{\ln \left( \frac{b-a}{\epsilon} \right)}{\ln(2)} - 1 & < N
\end{align}

## Implementation

Write a function called `bisection` which takes 4 input parameters `f`, `a`, `b` and `N` and returns the approximation of a solution of $f(x)=0$ given by $N$ iterations of the bisection method. If $f(a_n)f(b_n) \geq 0$ at any point in the iteration (caused either by a bad initial interval or rounding error in computations), then print `"Bisection method fails."` and return `None`.


```python
def bisection(f,a,b,N):
    '''Approximate solution of f(x)=0 on interval [a,b] by bisection method.

    Parameters
    ----------
    f : function
        The function for which we are trying to approximate a solution f(x)=0.
    a,b : numbers
        The interval in which to search for a solution. The function returns
        None if f(a)*f(b) >= 0 since a solution is not guaranteed.
    N : (positive) integer
        The number of iterations to implement.

    Returns
    -------
    x_N : number
        The midpoint of the Nth interval computed by the bisection method. The
        initial interval [a_0,b_0] is given by [a,b]. If f(m_n) == 0 for some
        midpoint m_n = (a_n + b_n)/2, then the function returns this solution.
        If all signs of values f(a_n), f(b_n) and f(m_n) are the same at any
        iteration, the bisection method fails and return None.

    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> bisection(f,1,2,25)
    1.618033990263939
    >>> f = lambda x: (2*x - 1)*(x - 3)
    >>> bisection(f,0,1,10)
    0.5
    '''
    if f(a)*f(b) >= 0:
        print("Bisection method fails.")
        return None
    a_n = a
    b_n = b
    for n in range(1,N+1):
        m_n = (a_n + b_n)/2
        f_m_n = f(m_n)
        if f(a_n)*f_m_n < 0:
            a_n = a_n
            b_n = m_n
        elif f(b_n)*f_m_n < 0:
            a_n = m_n
            b_n = b_n
        elif f_m_n == 0:
            print("Found exact solution.")
            return m_n
        else:
            print("Bisection method fails.")
            return None
    return (a_n + b_n)/2
```

## Examples

### Golden Ratio

Let's use our function with input parameters $f(x)=x^2 - x - 1$ and $N=25$ iterations on $[1,2]$ to approximate the [golden ratio](https://en.wikipedia.org/wiki/Golden_ratio)

$$
\phi = \frac{1 + \sqrt{5}}{2}
$$

The golden ratio $\phi$ is a root of the quadratic polynomial $x^2 - x - 1 = 0$.


```python
f = lambda x: x**2 - x - 1
approx_phi = bisection(f,1,2,25)
print(approx_phi)
```

    1.618033990263939


The absolute error is guaranteed to be less than $(2 - 1)/(2^{26})$ which is:


```python
error_bound = 2**(-26)
print(error_bound)
```

    1.4901161193847656e-08


Let's verify the absolute error is then than this error bound:


```python
abs( (1 + 5**0.5)/2 - approx_phi) < error_bound
```




    True



## Exercises

*Under construction* 

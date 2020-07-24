# Newton's Method

[Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method) is a root finding method that uses linear approximation. In particular, we guess a solution $x_0$ of the equation $f(x)=0$, compute the linear approximation of $f(x)$ at $x_0$ and then find the $x$-intercept of the linear approximation.

## Formula

Let $f(x)$ be a differentiable function. If $x_0$ is near a solution of $f(x)=0$ then we can approximate $f(x)$ by the tangent line at $x_0$ and compute the $x$-intercept of the tangent line. The equation of the tangent line at $x_0$ is

$$
y = f'(x_0)(x - x_0) + f(x_0)
$$

The $x$-intercept is the solution $x_1$ of the equation

$$
0 = f'(x_0)(x_1 - x_0) + f(x_0)
$$

and we solve for $x_1$

$$
x_1 = x_0 - \frac{f(x_0)}{f'(x_0)}
$$

If we implement this procedure repeatedly, then we obtain a sequence given by the recursive formula

$$
x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}
$$

which (potentially) converges to a solution of the equation $f(x)=0$.

## Advantages/Disadvantages

When it converges, Newton's method usually converges very quickly and this is its main advantage. However, Newton's method is not guaranteed to converge and this is obviously a big disadvantage especially compared to the bisection and secant methods which are guaranteed to converge to a solution (provided they start with an interval containing a root).

Newton's method also requires computing values of the derivative of the function in question. This is potentially a disadvantage if the derivative is difficult to compute.

The stopping criteria for Newton's method differs from the bisection and secant methods. In those methods, we know how close we are to a solution because we are computing intervals which contain a solution. In Newton's method, we don't know how close we are to a solution. All we can compute is the value $f(x)$ and so we implement a stopping criteria based on $f(x)$.

Finally, there's no guarantee that the method converges to a solution and we should set a maximum number of iterations so that our implementation ends if we don't find a solution.

## Implementation

Let's write a function called `newton` which takes 5 input parameters `f`, `Df`, `x0`, `epsilon` and `max_iter` and returns an approximation of a solution of $f(x)=0$ by Newton's method. The function may terminate in 3 ways:

1. If `abs(f(xn)) < epsilon`, the algorithm has found an approximate solution and returns `xn`.
2. If `f'(xn) == 0`, the algorithm stops and returns `None`.
3. If the number of iterations exceed `max_iter`, the algorithm stops and returns `None`.


```python
def newton(f,Df,x0,epsilon,max_iter):
    '''Approximate solution of f(x)=0 by Newton's method.

    Parameters
    ----------
    f : function
        Function for which we are searching for a solution f(x)=0.
    Df : function
        Derivative of f(x).
    x0 : number
        Initial guess for a solution f(x)=0.
    epsilon : number
        Stopping criteria is abs(f(x)) < epsilon.
    max_iter : integer
        Maximum number of iterations of Newton's method.

    Returns
    -------
    xn : number
        Implement Newton's method: compute the linear approximation
        of f(x) at xn and find x intercept by the formula
            x = xn - f(xn)/Df(xn)
        Continue until abs(f(xn)) < epsilon and return xn.
        If Df(xn) == 0, return None. If the number of iterations
        exceeds max_iter, then return None.

    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> Df = lambda x: 2*x - 1
    >>> newton(f,Df,1,1e-8,10)
    Found solution after 5 iterations.
    1.618033988749989
    '''
    xn = x0
    for n in range(0,max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            print('Found solution after',n,'iterations.')
            return xn
        Dfxn = Df(xn)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    print('Exceeded maximum iterations. No solution found.')
    return None
```

## Examples

### Supergolden Ratio

Let's test our function `newton` on the polynomial $p(x) = x^3 - x^2 - 1$ to approximate the [super golden ratio](https://en.wikipedia.org/wiki/Supergolden_ratio).


```python
p = lambda x: x**3 - x**2 - 1
Dp = lambda x: 3*x**2 - 2*x
approx = newton(p,Dp,1,1e-10,10)
print(approx)
```

    Found solution after 6 iterations.
    1.4655712318767877


How many iterations of the bisection method starting with the interval $[1,2]$ can achieve the same accuracy?

### Divergent Example

Newton's method diverges in certain cases. For example, if the tangent line at the root is vertical as in $f(x)=x^{1/3}$. Note that bisection and secant methods would converge in this case.


```python
f = lambda x: x**(1/3)
Df = lambda x: (1/3)*x**(-2/3)
approx = newton(f,Df,0.1,1e-2,100)
```

    Exceeded maximum iterations. No solution found.


## Exercises

1. Let $p(x) = x^3 - x - 1$. The only real root of $p(x)$ is called the [plastic number](https://en.wikipedia.org/wiki/Plastic_number) and is given by

$$
\frac{\sqrt[3]{108 + 12\sqrt{69}} + \sqrt[3]{108 - 12\sqrt{69}}}{6}
$$

2. Choose $x_0 = 1$ and implement 2 iterations of Newton's method to approximate the plastic number.
3. Use the exact value above to compute the absolute error after 2 iterations of Newton's method.
4. Starting with the subinterval $[1,2]$, how many iterations of the bisection method is required to achieve the same accuracy?

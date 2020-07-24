# Problems

## Optimization

### Newton's Method

Write a function called `newton` which takes input parameters $f$, $x_0$, $h$ (with default value 0.001), `tolerance` (with default value 0.001) and `max_iter` (with default value 100). The function implements Newton's method to approximate a solution of $f(x) = 0$. In other words, compute the values of the recursive sequence starting at $x_0$ and defined by

$$
x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}
$$

Use the central difference formula with step size $h$ to approximate the derivative $f'(x_n)$. The desired result is that the method converges to an approximate root of $f(x)$ however there are several possibilities:

1. The sequence reaches the desired tolerance $|f(x_n)| \leq \mathtt{tolerance}$ and <code>newton</code> returns the value $x_n$.
2. The number of iterations exceeds the maximum number of iterations `max_iter`, the function prints the statement "Maximum iterations exceeded" and returns `None`.
3. A zero derivative is computed $f'(x_n) = 0$, the function prints the statement "Zero derivative" and returns `None`.

## Numerical Differentiation

### Central Difference Formula

Write a function called `derivatives` which takes input parameters $f$, $a$, $n$ and $h$ (with default value `h = 0.001`) and returns approximations of the derivatives $f'(a)$, $f''(a)$, $\dots$, $f^{(n)}(a)$ (as a NumPy array) using the formula

$$
f^{(n)}(a) \approx \frac{1}{2^n h^n} \sum_{k=0}^n (-1)^k {n \choose k} f \left( a + ( n - 2k ) h \right)
$$

Use either `scipy.misc.factorial` or `scipy.misc.comb` to compute $n$ choose $k$:

$$
{n \choose k} = \frac{n!}{k!(n-k)!}
$$

### Taylor Polynomials

Write a function called `taylor` which takes input parameters $f$, $a$, $n$ and $L$ and plots both $f(x)$ and the Taylor polynomial $T_n(x)$ of $f(x)$ at $x=a$ of degree $n$

$$
T_n(x) = \sum_{k=0}^n \frac{f^{(k)}(a)}{k!}(x - a)^k
$$

on the interval $[a-L,a+L]$ (in the same figure).

## Numerical Integration

### Trapezoid Rule

1. Find $f''(x)$ for $f(x) = \ln( \ln x)$.
2. Prove $|f''(x)| \leq \displaystyle \frac{2}{e^2}$ for $x \geq e$.
3. Write a function called `log_log` which takes input parameters `u` and `abs_tolerance` such that $u \geq e$ and `abs_tolerance` is a positive number (with default value 0.0001). The function uses the trapezoid rule to compute and return an approximation of the integral
$$
\int_e^u \ln( \ln x) dx
$$
The number $N$ of subintervals used in the trapezoid rule must be large enough to guarantee that the approximation is within `abs_tolerance` of the true value. You may use the function `scipy.integrate.trapz`.

### Simpson's Rule

1. Find $f''''(x)$ for $f(x) = e^{-x^2}$.
2. Plot $f''''(x)$ for $x \in [0,5]$. Determine a bound $M$ such that $|f''''(x)| \leq M$ for $x \geq 0$.
3. Write a function called `erf` which takes input parameters `u` and `abs_tolerance` (with default value 0.0001) such that both are positive numbers. The function uses Simpson's rule to compute and return an approximation of the integral
$$
\int_0^u e^{-x^2} dx
$$
The number $N$ of subintervals used in Simpson's rule must be large enough to guarantee that the approximation is within `abs_tolerance` of the true value. You may use the function `scipy.integrate.simps`.

## Differential Equations

### Lorenz Equations

The Lorenz equations are the system of nonlinear differential equations

\begin{align}
\frac{dx}{dt} &= \sigma(y - x) \\\
\frac{dy}{dt} &= x(\rho - z) - y \\\
\frac{dz}{dt} &= xy - \beta z
\end{align}

where $\sigma$, $\rho$ and $\beta$ are positive numbers. Write a function called `lorenz` which takes input parameters `sigma`, `rho`, `beta`, `u0`, `t0`, `tf`, `N` and `plot_vars` (with default value `[0,1]`). The function computes and plots a numerical approximation of the corresponding solution of the Lorenz equations using the function `scipy.integrate.odeint`. The input parameters are:

* `sigma`, `rho` and `beta` define the parameters $\sigma$, $\rho$ and $\beta$
* `u0` is a list of numbers of length 3 defining the initial conditions $[x(t_0),y(t_0),z(t_0)]$
* `t0` is the start of the interval of integration $[t_0,t_f]$
* `tf` is the end of the interval of integration $[t_0,t_f]$
* `N` is an integer specifying the number of evenly spaced points from $t_0$ to $t_f$ (inclusively) over which to compute the solution of the system
* `plot_vars` is a list of length 2 specifying which 2 components to plot where $x=0$, $y=1$ and $z=2$. For example, if `plot_vars` is $[0,1]$ then plot the solution $x$ versus $y$. If `plot_vars` is $[1,2]$ then plot the solution $y$ versus $z$. Note $x$ versus $y$ means $x$ is the horizontal axis and $y$ is the vertical. Default value is `[0,1]` which plots $x$ versus $y$.

The function `lorenz` returns a 2D NumPy array with 4 columns:

* column at index 0 is the array of $N$ evenly spaced $t$ values from $t_0$ to $t_f$ (inclusively)
* column at index 1 is the array of $x$ values of the solution
* column at index 2 is the array of $y$ values of the solution
* column at index 3 is the array of $z$ values of the solution

### Damped Oscillator

Write a function called `damping` which takes input parameters `m`, `b`, `k`, `F`, `u0`, `t0`, `tf` and `N`. The function uses `scipy.integrate.odeint` to compute a numerical approximation of the corresponding solution of the nonlinear damping equation:

$$
m y'' + b |y'| y' + ky = F(t)
$$

The input parameters are:

* `m`, `b` and `k` are positive numbers in the nonlinear damping equation
* `F` is a function of one variable $F(t)$ in the nonlinear damping equation
* `u0` is a list of numbers of length 2 defining the initial conditions $[y(t_0),y'(t_0)]$
* `t0` is the start of the interval of integration $[t_0,t_f]$
* `tf` is the end of the interval of integration $[t_0,t_f]$
* `N` is an integer specifying the number of evenly spaced points from $t_0$ to $t_f$ (inclusively) over which to compute the solution

The function `damping` plots the approximation of the solution $y(t)$ and returns a 2D Numpy array with 2 columns:

* column at index 0 is the array of $N$ evenly spaced $t$ values from $t_0$ to $t_f$ (inclusively)
* column at index 1 is the array of $y$ values of the solution              

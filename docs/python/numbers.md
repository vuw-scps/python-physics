# Numbers

The main [numeric types](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex) in Python are integers, floating point numbers and complex numbers. The syntax for arithmetic operators are: addition `+`, subtraction `-`, multiplication `*`, division `/` and exponentiation `**`.

## Integers

Add integers:


```python
8 + 12
```




    20



Subtract integers:


```python
2019 - 21
```




    1998



Multiply integers:


```python
45 * 11
```




    495



Divide integers (and notice that division of integers *always* returns a float):


```python
100 / 4
```




    25.0



Compute powers of integers:


```python
2**10
```




    1024



Use the built-in function `type()` to verify the type of a Python object:


```python
type(42)
```




    int



## Floating Point Numbers

A floating point number (or float) is a real number written in decimal form. Python stores floats and integers in different ways and if we combine integers and floats using arithmetic operations the result is always a float.

Approximate $\sqrt{2} \,$:


```python
2**0.5
```




    1.4142135623730951



Approximate $2 \pi$:


```python
2 * 3.14159
```




    6.28318



Use scientific notation to create $0.00001$:


```python
1e-5
```




    1e-05



Again, use the `type()` function to verify the type of a number:


```python
type(42)
```




    int




```python
type(42.0)
```




    float



## Complex Numbers

Use the built-in function `complex()` to create a complex number in Python or use the letter `j` for $j = \sqrt{-1}$. The built-in function `complex()` takes 2 parameters defining the real and imaginary part of the complex number.

Create the complex number $1 + j$:


```python
complex(1,1)
```




    (1+1j)



Add complex numbers:


```python
(1 + 2j) + (2 - 3j)
```




    (3-1j)



Multiply complex numbers:


```python
(2 - 1j) * (5 + 2j)
```




    (12-1j)



Use the `type()` function to verify the type of a number:


```python
type(2 - 7j)
```




    complex



<!--

### Complex Methods

The complex datatype has a few methods. For example, we can access the real and imaginary parts of a complex number:

```python
z.real
```

```nohighlight
1.0
```

```python
z.imag
```

```nohighlight
1.0
```

```python
print(z)
```

```nohighlight
(1+1j)
```

The conjugate of a complex number $z = a + b\sqrt{-1}$ is $\overline{z} = a - b\sqrt{-1}$.

```python
z.conjugate()
```

```nohighlight
(1-1j)
```

The modulus of a complex number $z = a + b\sqrt{-1}$ is $|z| = \sqrt{a^2 + b^2}$ which is computed by the builtin function `abs` (which is the absolute value when applied to integers and floats).

```python
print(z)
abs(z)
```

```nohighlight
(1+1j)
1.4142135623730951
```

```python
(1**2 + 1**2)**(0.5)
```

```nohighlight
1.4142135623730951
```
-->

## Arithmetic Operators

The syntax for arithmetic operators in Python are:

| Operator | Description  |
| :---: | :---: |
| `+` | addition |
| `-` | subtraction |
| `*` | multiplication |
| `/` | division |
| `**` | exponentiation |
| `%` | remainder (or modulo) |
| `//` | integer division |

Notice that division of integers always returns a float:


```python
4 / 3
```




    1.3333333333333333



Even if the mathematical result is an integer:


```python
4 / 2
```




    2.0



Use parentheses to group combinations of arithmetic operations:


```python
5 * (4 + 3) - 2
```




    33



An integer power of an integer is again an integer:


```python
2**4
```




    16



An exponent involving a float is a float:


```python
9**0.5
```




    3.0



The remainder operator computes the remainder of division of integers:


```python
11 % 4
```




    3



Integer division is:


```python
11 // 4
```




    2



## Examples

### Taylor Approximation

The [Taylor series](https://en.wikipedia.org/wiki/Taylor_series) of the exponential function $e^x$ is given by

$$
e^x = \sum_{k=0}^{\infty} \frac{x^k}{k!}
$$

Compute the Taylor polynomial of degree 5 evaluated at $x = 1$ to find an approximation of $e$

$$
e \approx \frac{1}{0!} + \frac{1}{1!} + \frac{1}{2!} + \frac{1}{3!} + \frac{1}{4!} + \frac{1}{5!}
$$


```python
1 + 1 + 1/2 + 1/(3*2) + 1/(4*3*2) + 1/(5*4*3*2)
```




    2.7166666666666663



### Ramanujan's $\pi$ Formula

[Srinivasa Ramanujan](https://en.wikipedia.org/wiki/Srinivasa_Ramanujan#Mathematical_achievements) discovered the following beautiful (and very rapidly converging) series representation of $\pi$

$$
\frac{1}{\pi} = \frac{2 \sqrt{2}}{99^2} \sum_{k = 0}^{\infty} \frac{(4k)!}{k!^4} \frac{1103 + 26390k}{396^{4k}}
$$

Let's find an approximation of $\pi$ by computing the *reciprocal* of the sum of the first 3 terms of the series:

$$
\pi \approx \frac{99^2}{2 \sqrt{2}} \frac{1}{\left( 1103 + 4! \frac{1103 + 26390}{396^{4}} + \frac{8!}{2^4} \frac{1103 + 26390(2)}{396^{8}} \right)}
$$


```python
99**2 / (2 * 2**0.5) / (1103 + 4*3*2 * (26390 + 1103) / 396**4
                       + 8*7*6*5*4*3*2 / 2**4 * (26390*2 + 1103) / 396**8)
```




    3.141592653589793



These are exactly the first 16 digits of [$\pi$](https://en.wikipedia.org/wiki/Pi).

## Exercises

1. The [Taylor series](https://en.wikipedia.org/wiki/Taylor_series) of $\cos x$ is given by

    $$
    \cos x = \sum_{k=0}^{\infty} (-1)^k \frac{x^{2k}}{(2k)!}
    $$

    Compute the Taylor polynomial of degree 6 evaluated at $x=2$:

    $$
    \cos(2) \approx 1 - \frac{2^2}{2!} + \frac{2^4}{4!} - \frac{2^6}{6!}
    $$

2. The [Riemann zeta function](https://en.wikipedia.org/wiki/Riemann_zeta_function) is the infinite series

    $$
    \zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s}
    $$

    and is intimately related to prime numbers by the [Euler product formula](https://en.wikipedia.org/wiki/Riemann_zeta_function#Euler_product_formula)

    $$
    \sum_{n=1}^{\infty} \frac{1}{n^s} = \prod_p \left( \frac{1}{1 - p^{-s}} \right)
    $$

    where the product is over all primes $p = 2,3,5,7,11,13,\dots$

    Compute the 5th partial sum for $s=2$

    $$
    1 + \frac{1}{2^2} + \frac{1}{3^2} + \frac{1}{4^2} + \frac{1}{5^2}
    $$

    Compute the 5th partial product for $s=2$

    $$
    \left( \frac{1}{1 - 2^{-2}} \right) \left( \frac{1}{1 - 3^{-2}} \right) \left( \frac{1}{1 - 5^{-2}} \right) \left( \frac{1}{1 - 7^{-2}} \right) \left( \frac{1}{1 - 11^{-2}} \right)
    $$

    Given [Euler's special value formula](https://en.wikipedia.org/wiki/Basel_problem)

    $$
    \zeta(2) = \frac{\pi^2}{6}
    $$

    which converges more quickly: the infinite series or product?

3. The [continued fraction](https://en.wikipedia.org/wiki/Continued_fraction#Square_roots) for $\sqrt{2}$ is given by

    $$
    \sqrt{2} = 1 + \frac{1}{2 + \frac{1}{2 + \frac{1}{2 + \frac{1}{2 + \ddots}}}}
    $$

    Compute the following (partial) continued fraction to approximate $\sqrt{2}$

    $$
    \sqrt{2} \approx 1 + \frac{1}{2 + \frac{1}{2 + \frac{1}{2 + \frac{1}{2}}}}
    $$
# Logic

## Boolean Values

The [boolean](https://docs.python.org/3/library/stdtypes.html#truth-value-testing) type has only two values: `True` and `False`. Let's assign a boolean value to a variable and verify the type using the built-in function `type()`:


```python
python_is_fun = True
print(python_is_fun)
```

    True



```python
type(python_is_fun)
```




    bool



Let's assign the value `False` to a variable and again verify the type:


```python
math_is_scary = False
print(math_is_scary)
```

    False



```python
type(math_is_scary)
```




    bool



## Comparison Operators

[Comparison operators](https://docs.python.org/3/library/stdtypes.html#comparisons) produce Boolean values as output. For example, if we have variables `x` and `y` with numeric values, we can evaluate the expression `x < y` and the result is a boolean value either `True` or `False`.

| Comparison Operator | Description  |
| :---: | :---: |
| `<` | strictly less than |
| `<=` | less than or equal |
| `>` | strictly greater than |
| `>=` | greater than or equal |
| `==` | equal |
| `!=` | not equal |

For example:


```python
1 == 2
```




    False




```python
1 < 2
```




    True




```python
2 == 2
```




    True




```python
3 != 3.14159
```




    True




```python
20.00000001 >= 20
```




    True



## Boolean Operators

We combine logical expressions using [boolean operators](https://docs.python.org/3/library/stdtypes.html#boolean-operations-and-or-not) `and`, `or` and `not`.

| Boolean Operator | Description |
| :---: | :---: |
| `A and B` | returns `True` if both `A` and `B` are `True` |
| `A or B` | returns `True` if either `A` or `B` is `True`
| `not A` |  returns `True` if `A` is `False`

For example:


```python
(1 < 2) and (3 != 5)
```




    True




```python
(1 < 2) and (3 < 1)
```




    False




```python
(1 < 2) or (3 < 1)
```




    True




```python
not (1000 <= 999)
```




    True



## if statements

An [if statement](https://docs.python.org/3/tutorial/controlflow.html#if-statements) consists of one or more blocks of code such that only one block is executed depending on logical expressions. Let's do an example:


```python
# Determine if roots of polynomial ax^2 + bx + c = 0
# are real, repeated or complex using the
# quadratic formula x = (-b \pm \sqrt{b^2 - 4ac})/2a
a = 10
b = -234
c = 1984
discriminant = b**2 - 4*a*c
if discriminant > 0:
    print("Discriminant =", discriminant)
    print("Roots are real and distinct.")
elif discriminant < 0:
    print("Discriminant =", discriminant)
    print("Roots are complex.")
else:
    print("Discriminant =", discriminant)
    print("Roots are real and repeated.")
```

    Discriminant = -24604
    Roots are complex.


The main points to observe are:

1. Start with the `if` keyword.
2. Write a logical expression (returning `True` or `False`).
3. End line with a colon `:`.
4. Indent block 4 spaces after `if` statement.
5. Include `elif` and `else` statements if needed.
6. Only one of the blocks `if`, `elif` and `else` is executed.
7. The block  following an `else` statement will execute only if all other logical expressions before it are `False`.

## Examples

### Invertible Matrix

Represent a 2 by 2 square matrix as a list of lists. For example, represent the matrix

$$
\begin{bmatrix} 2 & -1 \\\ 5 & 7 \end{bmatrix}
$$

as the list of lists `[[2,-1],[5,7]]`.

Write a function called `invertible` which takes an input parameter `M`, a list of lists representing a 2 by 2 matrix, and returns `True` if the matrix `M` is invertible and `False` if not.


```python
def invertible(M):
    '''Determine if M is invertible.

    Parameters
    ----------
    M : list of lists
        Representation of a 2 by 2 matrix M = [[a,b],[c,d]].

    Returns
    -------
    bool
        True if M is invertible and False if not.

    Examples
    --------
    >>> invertible([[1,2],[3,4]])
    True
    '''
    # A matrix M is invertible if and only if
    # the determinant is not zero where
    # det(M) = ad - bc for M = [[a,b],[c,d]]
    determinant = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    if determinant != 0:
        return True
    else:
        return False
```

Let's test our function:


```python
invertible([[1,2],[3,4]])
```




    True




```python
invertible([[1,1],[3,3]])
```




    False



### Concavity of a Polynomial

Write a function called `concave_up` which takes input parameters `p` and `a` where `p` is a list representing a polynomial $p(x)$ and `a` is a number, and returns `True` if the function $p(x)$ is concave up at $x=a$ (ie. its second derivative is positive at $x=a$, $p''(a) > 0$).

We'll use the second derivative test for polynomials. In particular, if we have a polynomial of degree $n$

$$
p(x) = c_0 + c_1 x + c_2 x^2 + \cdots + c_n x^n
$$

then the second derivative of $p(x)$ at $x=a$ is the sum

$$
p''(a) = 2(1) c_2 + 3(2)c_3 a + 4(3)c_4 a^2 + \cdots + n(n-1)c_n a^{n-2}
$$


```python
def concave_up(p,a):
    '''Determine if the polynomial p(x) is concave up at x=a.

    Parameters
    ----------
    p : list of numbers
        List [a_0,a_1,a_2,...,a_n] represents the polynomial
        p(x) = a_0 + a_1*x + a_2*x**2 + ... + a_n*x**n

    Returns
    -------
    bool
        True if p(x) is concave up at x=a (ie. p''(a) > 0) and False otherwise.

    Examples
    --------
    >>> concave_up([1,0,-2],0)
    False
    >>> concave_up([1,0,2],0)
    True
    '''
    # Degree of the polynomial p(x)
    degree = len(p) - 1

    # p''(a) == 0 if degree of p(x) is less than 2
    if degree < 2:
        return False
    else:
        # Compute the second derivative p''(a)
        DDp_a = sum([k*(k-1)*p[k]*a**(k-2) for k in range(2,degree + 1)])
        if DDp_a > 0:
            return True
        else:
            return False
```

Let's test our function on $p(x) = 1 + x - x^3$ at $x=2$. Since $p''(x) = -6x$ and $p''(2) = -12 < 0$, the polynomial is concave down at $x=2$.


```python
p = [1,1,0,-1]
a = 2
concavity = concave_up(p,a)
print(concavity)
```

    False


## Exercises

1. The [discriminant](https://en.wikipedia.org/wiki/Discriminant#Cubic) of a cubic polynomial $p(x) = ax^3 + bx^2 + cx + d$ is

    $$
    \Delta = b^2c^2 - 4ac^3 - 4b^3d - 27a^2d^2 + 18abcd
    $$

    The discriminant gives us information about the roots of the polynomial $p(x)$:

    * if $\Delta > 0$, then $p(x)$ has 3 distinct real roots
    * if $\Delta < 0$, then $p(x)$ has 2 distinct complex roots and 1 real root
    * if $\Delta = 0$, then $p(x)$ has at least 2 (real or complex) roots which are the same

    Represent a cubic polynomial $p(x) = ax^3 + bx^2 + cx + d$ as a list `[d,c,b,a]` of numbers. (Note the order of the coefficients is increasing degree.) For example, the polynomial $p(x) = x^3 - x + 1$ is `[1,-1,0,1]`.

    Write a function called `cubic_roots` which takes an input parameter `p`, a list of length 4 representing a cubic polynomial, and returns `True` if $p(x)$ has 3 real distinct roots and `False` otherwise.

2. Represent a 2 by 2 square matrix as a list of lists. For example, represent the matrix

    $$
    \begin{bmatrix} 2 & -1 \\\ 5 & 7 \end{bmatrix}
    $$

    as the list of lists `[[2,-1],[5,7]]`. Write a function called `inverse_a` which takes an input parameter `a` and returns a list of lists representing the [inverse of the matrix](https://en.wikipedia.org/wiki/Invertible_matrix)

    $$
    \begin{bmatrix}
    1 & a \\\
    a & -1
    \end{bmatrix}
    $$

3. Write a function called `real_eigenvalues` which takes an input parameter `M`, a list of lists representing a 2 by 2 matrix (as in the previous exercise), and returns `True` if the [eigenvalues](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors) of the matrix `M` are real numebrs and `False` if not.

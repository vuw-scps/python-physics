# Solving Linear Systems


```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
%matplotlib inline
```

## Linear Systems

A [linear system of equations](https://en.wikipedia.org/wiki/System_of_linear_equations) is a collection of linear equations

\begin{align}
a_{0,0}x_0 + a_{0,1}x_2 + \cdots + a_{0,n}x_n & = b_0 \\\
a_{1,0}x_0 + a_{1,1}x_2 + \cdots + a_{1,n}x_n & = b_1 \\\
& \vdots \\\
a_{m,0}x_0 + a_{m,1}x_2 + \cdots + a_{m,n}x_n & = b_m \\\
\end{align}

In matrix notation, a linear system is $A \mathbf{x}= \mathbf{b}$ where

$$
A = \begin{bmatrix}
a_{0,0} & a_{0,1} & \cdots & a_{0,n} \\\
a_{1,0} & a_{1,1} & \cdots & a_{1,n} \\\
\vdots & & & \vdots \\\
a_{m,0} & a_{m,1} & \cdots & a_{m,n} \\\
\end{bmatrix}
 \ \ , \ \
\mathbf{x} = \begin{bmatrix}
x_0 \\\ x_1 \\\ \vdots \\\ x_n
\end{bmatrix}
 \ \ , \ \
\mathbf{b} = \begin{bmatrix}
b_0 \\\ b_1 \\\ \vdots \\\ b_m
\end{bmatrix} 
$$

## Gaussian elimination

The general procedure to solve a linear system of equation is called [Gaussian elimination](https://en.wikipedia.org/wiki/Gaussian_elimination). The idea is to perform elementary row operations to reduce the system to its row echelon form and then solve.

### Elementary Row Operations

[Elementary row operations](https://en.wikipedia.org/wiki/Elementary_matrix#Elementary_row_operations) include:

1. Add $k$ times row $j$ to row $i$.
2. Multiply row $i$ by scalar $k$.
3. Switch rows $i$ and $j$.

Each of the elementary row operations is the result of matrix multiplication by an elementary matrix (on the left).
To add $k$ times row $i$ to row $j$ in a matrix $A$, we multiply $A$ by the matrix $E$ where $E$ is equal to the identity matrix except the $i,j$ entry is $E_{i,j} = k$. For example, if $A$ is 3 by 3 and we want to add 3 times row 2 to row 0 (using 0 indexing) then

$$
E_1 = \begin{bmatrix}
1 & 0 & 3 \\\
0 & 1 & 0 \\\
0 & 0 & 1
\end{bmatrix}
$$

Let's verify the calculation:


```python
A = np.array([[1,1,2],[-1,3,1],[0,5,2]])
print(A)
```

    [[ 1  1  2]
     [-1  3  1]
     [ 0  5  2]]



```python
E1 = np.array([[1,0,3],[0,1,0],[0,0,1]])
print(E1)
```

    [[1 0 3]
     [0 1 0]
     [0 0 1]]



```python
E1 @ A
```




    array([[ 1, 16,  8],
           [-1,  3,  1],
           [ 0,  5,  2]])



To multiply $k$ times row $i$ in a matrix $A$, we multiply $A$ by the matrix $E$ where $E$ is equal to the identity matrix except the $,i,j$ entry is $E_{i,i} = k$. For example, if $A$ is 3 by 3 and we want to multiply row 1 by -2 then

$$
E_2 = \begin{bmatrix}
1 & 0 & 0 \\\
0 & -2 & 0 \\\
0 & 0 & 1
\end{bmatrix}
$$

Let's verify the calculation:


```python
E2 = np.array([[1,0,0],[0,-2,0],[0,0,1]])
print(E2)
```

    [[ 1  0  0]
     [ 0 -2  0]
     [ 0  0  1]]



```python
E2 @ A
```




    array([[ 1,  1,  2],
           [ 2, -6, -2],
           [ 0,  5,  2]])



Finally, to switch row $i$ and row $j$ in a matrix $A$, we multiply $A$ by the matrix $E$ where $E$ is equal to the identity matrix except $E_{i,i} = 0$, $E_{j,j} = 0$, $E_{i,j} = 1$ and $E_{j,i} = 1$. For example, if $A$ is 3 by 3 and we want to switch row 1 and row 2 then

$$
E^3 = \begin{bmatrix}
1 & 0 & 0 \\\
0 & 0 & 1 \\\
0 & 1 & 0
\end{bmatrix}
$$

Let's verify the calculation:


```python
E3 = np.array([[1,0,0],[0,0,1],[0,1,0]])
print(E3)
```

    [[1 0 0]
     [0 0 1]
     [0 1 0]]



```python
E3 @ A
```




    array([[ 1,  1,  2],
           [ 0,  5,  2],
           [-1,  3,  1]])



### Implementation

Let's write function to implement the elementary row operations. First of all, let's write a function called `add_rows` which takes input parameters $A$, $k$, $i$ and $j$ and returns the NumPy array resulting from adding $k$ times row $j$ to row $i$ in the matrix $A$. If $i=j$, then let's say that the function scales row $i$ by $k+1$ since this would be the result of $k$ times row $i$ added to row $i$.


```python
def add_row(A,k,i,j):
    "Add k times row j to row i in matrix A."
    n = A.shape[0]
    E = np.eye(n)
    if i == j:
        E[i,i] = k + 1
    else:
        E[i,j] = k
    return E @ A
```

Let's test our function:


```python
M = np.array([[1,1],[3,2]])
print(M)
```

    [[1 1]
     [3 2]]



```python
add_row(M,2,0,1)
```




    array([[7., 5.],
           [3., 2.]])




```python
add_row(M,3,1,1)
```




    array([[ 1.,  1.],
           [12.,  8.]])



Let's write a function called `scale_row` which takes 3 input parameters $A$, $k$, and $i$ and returns the matrix that results from $k$ times row $i$ in the matrix $A$.


```python
def scale_row(A,k,i):
    "Multiply row i by k in matrix A."
    n = A.shape[0]
    E = np.eye(n)
    E[i,i] = k
    return E @ A
```


```python
M = np.array([[3,1],[-2,7]])
print(M)
```

    [[ 3  1]
     [-2  7]]



```python
scale_row(M,3,1)
```




    array([[ 3.,  1.],
           [-6., 21.]])




```python
A = np.array([[1,1,1],[1,-1,0]])
print(A)
```

    [[ 1  1  1]
     [ 1 -1  0]]



```python
scale_row(A,5,1)
```




    array([[ 1.,  1.,  1.],
           [ 5., -5.,  0.]])



Let's write a function called `switch_rows` which takes 3 input parameters $A$, $i$ and $j$ and returns the matrix that results from switching rows $i$ and $j$ in the matrix $A$.


```python
def switch_rows(A,i,j):
    "Switch rows i and j in matrix A."
    n = A.shape[0]
    E = np.eye(n)
    E[i,i] = 0
    E[j,j] = 0
    E[i,j] = 1
    E[j,i] = 1
    return E @ A
```


```python
A = np.array([[1,1,1],[1,-1,0]])
print(A)
```

    [[ 1  1  1]
     [ 1 -1  0]]



```python
switch_rows(A,0,1)
```




    array([[ 1., -1.,  0.],
           [ 1.,  1.,  1.]])



## Examples

### Find the Inverse

Let's apply our functions to the augmented matrix $[M \ | \ I]$ to find the inverse of the matrix $M$:


```python
M = np.array([[5,4,2],[-1,2,1],[1,1,1]])
print(M)
```

    [[ 5  4  2]
     [-1  2  1]
     [ 1  1  1]]



```python
A = np.hstack([M,np.eye(3)])
print(A)
```

    [[ 5.  4.  2.  1.  0.  0.]
     [-1.  2.  1.  0.  1.  0.]
     [ 1.  1.  1.  0.  0.  1.]]



```python
A1 = switch_rows(A,0,2)
print(A1)
```

    [[ 1.  1.  1.  0.  0.  1.]
     [-1.  2.  1.  0.  1.  0.]
     [ 5.  4.  2.  1.  0.  0.]]



```python
A2 = add_row(A1,1,1,0)
print(A2)
```

    [[1. 1. 1. 0. 0. 1.]
     [0. 3. 2. 0. 1. 1.]
     [5. 4. 2. 1. 0. 0.]]



```python
A3 = add_row(A2,-5,2,0)
print(A3)
```

    [[ 1.  1.  1.  0.  0.  1.]
     [ 0.  3.  2.  0.  1.  1.]
     [ 0. -1. -3.  1.  0. -5.]]



```python
A4 = switch_rows(A3,1,2)
print(A4)
```

    [[ 1.  1.  1.  0.  0.  1.]
     [ 0. -1. -3.  1.  0. -5.]
     [ 0.  3.  2.  0.  1.  1.]]



```python
A5 = scale_row(A4,-1,1)
print(A5)
```

    [[ 1.  1.  1.  0.  0.  1.]
     [ 0.  1.  3. -1.  0.  5.]
     [ 0.  3.  2.  0.  1.  1.]]



```python
A6 = add_row(A5,-3,2,1)
print(A6)
```

    [[  1.   1.   1.   0.   0.   1.]
     [  0.   1.   3.  -1.   0.   5.]
     [  0.   0.  -7.   3.   1. -14.]]



```python
A7 = scale_row(A6,-1/7,2)
print(A7)
```

    [[ 1.          1.          1.          0.          0.          1.        ]
     [ 0.          1.          3.         -1.          0.          5.        ]
     [ 0.          0.          1.         -0.42857143 -0.14285714  2.        ]]



```python
A8 = add_row(A7,-3,1,2)
print(A8)
```

    [[ 1.          1.          1.          0.          0.          1.        ]
     [ 0.          1.          0.          0.28571429  0.42857143 -1.        ]
     [ 0.          0.          1.         -0.42857143 -0.14285714  2.        ]]



```python
A9 = add_row(A8,-1,0,2)
print(A9)
```

    [[ 1.          1.          0.          0.42857143  0.14285714 -1.        ]
     [ 0.          1.          0.          0.28571429  0.42857143 -1.        ]
     [ 0.          0.          1.         -0.42857143 -0.14285714  2.        ]]



```python
A10 = add_row(A9,-1,0,1)
print(A10)
```

    [[ 1.          0.          0.          0.14285714 -0.28571429  0.        ]
     [ 0.          1.          0.          0.28571429  0.42857143 -1.        ]
     [ 0.          0.          1.         -0.42857143 -0.14285714  2.        ]]


Let's verify that we found the inverse $M^{-1}$ correctly:


```python
Minv = A10[:,3:]
print(Minv)
```

    [[ 0.14285714 -0.28571429  0.        ]
     [ 0.28571429  0.42857143 -1.        ]
     [-0.42857143 -0.14285714  2.        ]]



```python
result = Minv @ M
print(result)
```

    [[ 1.00000000e+00  4.44089210e-16  2.22044605e-16]
     [-6.66133815e-16  1.00000000e+00 -2.22044605e-16]
     [ 0.00000000e+00  0.00000000e+00  1.00000000e+00]]


Success! We can see the result more clearly if we round to 15 decimal places:


```python
np.round(result,15)
```




    array([[ 1.e+00,  0.e+00,  0.e+00],
           [-1.e-15,  1.e+00, -0.e+00],
           [ 0.e+00,  0.e+00,  1.e+00]])



### Solve a System

Let's use our functions to perform Gaussian elimination and solve a linear system of equations $A \mathbf{x} = \mathbf{b}$.


```python
A = np.array([[6,15,1],[8,7,12],[2,7,8]])
print(A)
```

    [[ 6 15  1]
     [ 8  7 12]
     [ 2  7  8]]



```python
b = np.array([[2],[14],[10]])
print(b)
```

    [[ 2]
     [14]
     [10]]


Form the augemented matrix $M$:


```python
M = np.hstack([A,b])
print(M)
```

    [[ 6 15  1  2]
     [ 8  7 12 14]
     [ 2  7  8 10]]


Perform row operations:


```python
M1 = scale_row(M,1/6,0)
print(M1)
```

    [[ 1.          2.5         0.16666667  0.33333333]
     [ 8.          7.         12.         14.        ]
     [ 2.          7.          8.         10.        ]]



```python
M2 = add_row(M1,-8,1,0)
print(M2)
```

    [[  1.           2.5          0.16666667   0.33333333]
     [  0.         -13.          10.66666667  11.33333333]
     [  2.           7.           8.          10.        ]]



```python
M3 = add_row(M2,-2,2,0)
print(M3)
```

    [[  1.           2.5          0.16666667   0.33333333]
     [  0.         -13.          10.66666667  11.33333333]
     [  0.           2.           7.66666667   9.33333333]]



```python
M4 = scale_row(M3,-1/13,1)
print(M4)
```

    [[ 1.          2.5         0.16666667  0.33333333]
     [ 0.          1.         -0.82051282 -0.87179487]
     [ 0.          2.          7.66666667  9.33333333]]



```python
M5 = add_row(M4,-2,2,1)
print(M5)
```

    [[ 1.          2.5         0.16666667  0.33333333]
     [ 0.          1.         -0.82051282 -0.87179487]
     [ 0.          0.          9.30769231 11.07692308]]



```python
M6 = scale_row(M5,1/M5[2,2],2)
print(M6)
```

    [[ 1.          2.5         0.16666667  0.33333333]
     [ 0.          1.         -0.82051282 -0.87179487]
     [ 0.          0.          1.          1.19008264]]



```python
M7 = add_row(M6,-M6[1,2],1,2)
print(M7)
```

    [[1.         2.5        0.16666667 0.33333333]
     [0.         1.         0.         0.1046832 ]
     [0.         0.         1.         1.19008264]]



```python
M8 = add_row(M7,-M7[0,2],0,2)
print(M8)
```

    [[1.         2.5        0.         0.13498623]
     [0.         1.         0.         0.1046832 ]
     [0.         0.         1.         1.19008264]]



```python
M9 = add_row(M8,-M8[0,1],0,1)
print(M9)
```

    [[ 1.          0.          0.         -0.12672176]
     [ 0.          1.          0.          0.1046832 ]
     [ 0.          0.          1.          1.19008264]]


Success! The solution of $Ax=b$ is


```python
x = M9[:,3].reshape(3,1)
print(x)
```

    [[-0.12672176]
     [ 0.1046832 ]
     [ 1.19008264]]


Or, we can do it the easy way...


```python
x = la.solve(A,b)
print(x)
```

    [[-0.12672176]
     [ 0.1046832 ]
     [ 1.19008264]]


## `scipy.linalg.solve`

We are mostly interested in linear systems $A \mathbf{x} = \mathbf{b}$ where there is a unique solution $\mathbf{x}$. This is the case when $A$ is a square matrix ($m=n$) and $\mathrm{det}(A) \not= 0$. To solve such a system, we can use the function [`scipy.linalg.solve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve.html).

The function returns a solution of the system of equations $A \mathbf{x} = \mathbf{b}$. For example:


```python
A = np.array([[1,1],[1,-1]])
print(A)
```

    [[ 1  1]
     [ 1 -1]]



```python
b1 = np.array([2,0])
print(b1)
```

    [2 0]


And solve:


```python
x1 = la.solve(A,b1)
print(x1)
```

    [1. 1.]


Note that the output $\mathbf{x}$ is returned as a 1D NumPy array when the vector $\mathbf{b}$ (the right hand side) is entered as a 1D NumPy array. If we input $\mathbf{b}$ as a 2D NumPy array, then the output is a 2D NumPy array. For example:


```python
A = np.array([[1,1],[1,-1]])
b2 = np.array([2,0]).reshape(2,1)
x2 = la.solve(A,b2)
print(x2)
```

    [[1.]
     [1.]]


Finally, if the right hand side $\mathbf{b}$ is a matrix, then the output is a matrix of the same size. It is the solution of $A \mathbf{x} = \mathbf{b}$ when $\mathbf{b}$ is a matrix. For example:


```python
A = np.array([[1,1],[1,-1]])
b3 = np.array([[2,2],[0,1]])
x3 = la.solve(A,b3)
print(x3)
```

    [[1.  1.5]
     [1.  0.5]]


### Simple Example

Let's compute the solution of the system of equations

\begin{align}
2x + y &= 1 \\\
x + y &= 1
\end{align}

Create the matrix of coefficients:


```python
A = np.array([[2,1],[1,1]])
print(A)
```

    [[2 1]
     [1 1]]


And the vector $\mathbf{b}$:


```python
b = np.array([1,-1]).reshape(2,1)
print(b)
```

    [[ 1]
     [-1]]


And solve:


```python
x = la.solve(A,b)
print(x)
```

    [[ 2.]
     [-3.]]


We can verify the solution by computing the inverse of $A$:


```python
Ainv = la.inv(A)
print(Ainv)
```

    [[ 1. -1.]
     [-1.  2.]]


And multiply $A^{-1} \mathbf{b}$ to solve for $\mathbf{x}$:


```python
x = Ainv @ b
print(x)
```

    [[ 2.]
     [-3.]]


We get the same result. Success!

### Inverse or Solve

It's a bad idea to use the inverse $A^{-1}$ to solve $A \mathbf{x} = \mathbf{b}$ if $A$ is large. It's too computationally expensive. Let's create a large random matrix $A$ and vector $\mathbf{b}$ and compute the solution $\mathbf{x}$ in 2 ways:


```python
N = 1000
A = np.random.rand(N,N)
b = np.random.rand(N,1)
```

Check the first entries $A$:


```python
A[:3,:3]
```




    array([[0.35754719, 0.63135432, 0.6572258 ],
           [0.18450506, 0.14639832, 0.23528745],
           [0.27576474, 0.46264005, 0.26589724]])



And for $\mathbf{b}$:


```python
b[:4,:]
```




    array([[0.82726751],
           [0.96946096],
           [0.31351176],
           [0.63757837]])



Now we compare the speed of `scipy.linalg.solve` with `scipy.linalg.inv`:


```python
%%timeit
x = la.solve(A,b)
```

    2.77 s ± 509 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)



```python
%%timeit
x = la.inv(A) @ b
```

    4.46 s ± 2.04 s per loop (mean ± std. dev. of 7 runs, 1 loop each)


Solving with `scipy.linalg.solve` is about twice as fast!

## Exercises

*Under construction*

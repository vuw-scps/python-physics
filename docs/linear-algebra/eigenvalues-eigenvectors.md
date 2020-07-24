# Eigenvalues and Eigenvectors


```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
```

## Definition

Let $A$ be a square matrix. A non-zero vector $\mathbf{v}$ is an [eigenvector](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors) for $A$ with [eigenvalue](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors) $\lambda$ if

$$
A\mathbf{v} = \lambda \mathbf{v}
$$

Rearranging the equation, we see that $\mathbf{v}$ is a solution of the homogeneous system of equations

$$
\left( A - \lambda I \right) \mathbf{v} = \mathbf{0}
$$

where $I$ is the identity matrix of size $n$. Non-trivial solutions exist only if the matrix $A - \lambda I$ is singular which means $\mathrm{det}(A - \lambda I) = 0$. Therefore eigenvalues of $A$ are roots of the [characteristic polynomial](https://en.wikipedia.org/wiki/Characteristic_polynomial)

$$
p(\lambda) = \mathrm{det}(A - \lambda I)
$$

## scipy.linalg.eig

The function [`scipy.linalg.eig`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eig.html) computes eigenvalues and eigenvectors of a square matrix $A$.

Let's consider a simple example with a diagonal matrix:


```python
A = np.array([[1,0],[0,-2]])
print(A)
```

    [[ 1  0]
     [ 0 -2]]


The function `la.eig` returns a tuple `(eigvals,eigvecs)` where `eigvals` is a 1D NumPy array of complex numbers giving the eigenvalues of $A$, and `eigvecs` is a 2D NumPy array with the corresponding eigenvectors in the columns:


```python
results = la.eig(A)
```

The eigenvalues of $A$ are:


```python
print(results[0])
```

    [ 1.+0.j -2.+0.j]


The corresponding eigenvectors are:


```python
print(results[1])
```

    [[1. 0.]
     [0. 1.]]


We can [unpack the tuple](../../python/sequences/#unpacking-a-sequence):


```python
eigvals, eigvecs = la.eig(A)
print(eigvals)
```

    [ 1.+0.j -2.+0.j]



```python
print(eigvecs)
```

    [[1. 0.]
     [0. 1.]]


If we know that the eigenvalues are real numbers (ie. if $A$ is symmetric), then we can use the NumPy array method `.real` to convert the array of eigenvalues to real numbers:


```python
eigvals = eigvals.real
print(eigvals)
```

    [ 1. -2.]


Notice that the position of an eigenvalue in the array `eigvals` correspond to the column in `eigvecs` with its eigenvector:


```python
lambda1 = eigvals[1]
print(lambda1)
```

    -2.0



```python
v1 = eigvecs[:,1].reshape(2,1)
print(v1)
```

    [[0.]
     [1.]]



```python
A @ v1
```




    array([[ 0.],
           [-2.]])




```python
lambda1 * v1
```




    array([[-0.],
           [-2.]])



## Examples

### Symmetric Matrices

The eigenvalues of a [symmetric matrix](https://en.wikipedia.org/wiki/Symmetric_matrix) are always real and the eigenvectors are always orthogonal! Let's verify these facts with some random matrices:


```python
n = 4
P = np.random.randint(0,10,(n,n))
print(P)
```

    [[7 0 6 2]
     [9 5 1 3]
     [0 2 2 5]
     [6 8 8 6]]


Create the symmetric matrix $S = P P^T$:


```python
S = P @ P.T
print(S)
```

    [[ 89  75  22 102]
     [ 75 116  27 120]
     [ 22  27  33  62]
     [102 120  62 200]]


Let's unpack the eigenvalues and eigenvectors of $S$:


```python
evals, evecs = la.eig(S)
print(evals)
```

    [361.75382302+0.j  42.74593101+0.j  26.33718907+0.j   7.16305691+0.j]


The eigenvalues all have zero imaginary part and so they are indeed real numbers:


```python
evals = evals.real
print(evals)
```

    [361.75382302  42.74593101  26.33718907   7.16305691]


The corresponding eigenvectors of $A$ are:


```python
print(evecs)
```

    [[-0.42552429 -0.42476765  0.76464379 -0.23199439]
     [-0.50507589 -0.54267519 -0.64193252 -0.19576676]
     [-0.20612674  0.54869183 -0.05515612 -0.80833585]
     [-0.72203822  0.4733005   0.01415338  0.50442752]]


Let's check that the eigenvectors are orthogonal to each other:


```python
v1 = evecs[:,0] # First column is the first eigenvector
print(v1)
```

    [-0.42552429 -0.50507589 -0.20612674 -0.72203822]



```python
v2 = evecs[:,1] # Second column is the second eigenvector
print(v2)
```

    [-0.42476765 -0.54267519  0.54869183  0.4733005 ]



```python
v1 @ v2
```




    -1.1102230246251565e-16



The dot product of eigenvectors $\mathbf{v}_1$ and $\mathbf{v}_2$ is zero (the number above is <em>very</em> close to zero and is due to rounding errors in the computations) and so they are orthogonal!

### Diagonalization

A square matrix $M$ is [diagonalizable](https://en.wikipedia.org/wiki/Diagonalizable_matrix) if it is similar to a diagonal matrix. In other words, $M$ is diagonalizable if there exists an invertible matrix $P$ such that $D = P^{-1}MP$ is a diagonal matrix.

A beautiful result in linear algebra is that a square matrix $M$ of size $n$ is diagonalizable if and only if $M$ has $n$ independent eigevectors. Furthermore, $M = PDP^{-1}$ where the columns of $P$ are the eigenvectors of $M$ and $D$ has corresponding eigenvalues along the diagonal.

Let's use this to construct a matrix with given eigenvalues $\lambda_1 = 3, \lambda_2 = 1$, and eigenvectors $v_1 = [1,1]^T, v_2 = [1,-1]^T$.


```python
P = np.array([[1,1],[1,-1]])
print(P)
```

    [[ 1  1]
     [ 1 -1]]



```python
D = np.diag((3,1))
print(D)
```

    [[3 0]
     [0 1]]



```python
M = P @ D @ la.inv(P)
print(M)
```

    [[2. 1.]
     [1. 2.]]


Let's verify that the eigenvalues of $M$ are 3 and 1:


```python
evals, evecs = la.eig(M)
print(evals)
```

    [3.+0.j 1.+0.j]


Verify the eigenvectors:


```python
print(evecs)
```

    [[ 0.70710678 -0.70710678]
     [ 0.70710678  0.70710678]]


### Matrix Powers

Let $M$ be a square matrix. Computing powers of $M$ by matrix multiplication

$$
M^k = \underbrace{M M \cdots M}_k
$$

is computationally expensive. Instead, let's use diagonalization to compute $M^k$ more efficiently

$$
M^k = \left( P D P^{-1} \right)^k  = \underbrace{P D P^{-1} P D P^{-1} \cdots P D P^{-1}}_k = P D^k P^{-1}
$$

Let's compute $M^{20}$ both ways and compare execution time.


```python
Pinv = la.inv(P)
```


```python
k = 20
```


```python
%%timeit
result = M.copy()
for _ in range(1,k):
    result = result @ M
```

    42.1 µs ± 11.4 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)


Let's use diagonalization to do the same computation.


```python
%%timeit
P @ D**k @ Pinv
```

    6.42 µs ± 1.36 µs per loop (mean ± std. dev. of 7 runs, 100000 loops each)


Diagonalization computes $M^{k}$ much faster!

## Exercises

*Under construction*

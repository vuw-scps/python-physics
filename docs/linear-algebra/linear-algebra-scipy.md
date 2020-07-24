# Linear Algebra with SciPy

The main Python package for linear algebra is the SciPy subpackage [`scipy.linalg`](https://docs.scipy.org/doc/scipy/reference/linalg.html) which builds on NumPy. Let's import both packages:


```python
import numpy as np
import scipy.linalg as la
```

## NumPy Arrays

Let's begin with a quick review of [NumPy arrays](../../scipy/numpy/). We can think of a 1D NumPy array as a list of numbers. We can think of a 2D NumPy array as a matrix. And we can think of a 3D array as a cube of numbers.
When we select a row or column from a 2D NumPy array, the result is a 1D NumPy array (called a [slice](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#basic-slicing-and-indexing)). This is different from MATLAB where when you select a column from a matrix it's returned as a column vector which is a 2D MATLAB matrix.

It can get a bit confusing and so we need to keep track of the shape, size and dimension of our NumPy arrays.

### Array Attributes

Create a 1D (one-dimensional) NumPy array and verify its dimensions, shape and size.


```python
a = np.array([1,3,-2,1])
print(a)
```

    [ 1  3 -2  1]


Verify the number of dimensions:


```python
a.ndim
```




    1



Verify the shape of the array:


```python
a.shape
```




    (4,)



The shape of an array is returned as a [Python tuple](https://docs.python.org/3/tutorial/datastructures.html#tuples-and-sequences). The output in the cell above is a tuple of length 1. And we verify the size of the array (ie. the total number of entries in the array):


```python
a.size
```




    4



Create a 2D (two-dimensional) NumPy array (ie. matrix):


```python
M = np.array([[1,2],[3,7],[-1,5]])
print(M)
```

    [[ 1  2]
     [ 3  7]
     [-1  5]]


Verify the number of dimensions:


```python
M.ndim
```




    2



Verify the shape of the array:


```python
M.shape
```




    (3, 2)



Finally, verify the total number of entries in the array:


```python
M.size
```




    6



Select a row or column from a 2D NumPy array and we get a 1D array:


```python
col = M[:,1] 
print(col)
```

    [2 7 5]


Verify the number of dimensions of the slice:


```python
col.ndim
```




    1



Verify the shape and size of the slice:


```python
col.shape
```




    (3,)




```python
col.size
```




    3



When we select a row of column from a 2D NumPy array, the result is a 1D NumPy array. However, we may want to select a column as a 2D column vector. This requires us to use the [reshape](https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html) method.

For example, create a 2D column vector from the 1D slice selected from the matrix `M` above:


```python
print(col)
```

    [2 7 5]



```python
column = np.array([2,7,5]).reshape(3,1)
print(column)
```

    [[2]
     [7]
     [5]]


Verify the dimensions, shape and size of the array:


```python
print('Dimensions:', column.ndim)
print('Shape:', column.shape)
print('Size:', column.size)
```

    Dimensions: 2
    Shape: (3, 1)
    Size: 3


The variables `col` and `column` are different types of objects even though they have the "same" data.


```python
print(col)
```

    [2 7 5]



```python
print('Dimensions:',col.ndim)
print('Shape:',col.shape)
print('Size:',col.size)
```

    Dimensions: 1
    Shape: (3,)
    Size: 3


## Matrix Operations and Functions

### Arithmetic Operations

Recall that arithmetic [array operations](../scipy/numpy/#operations-and-functions) `+`, `-`, `/`, `*` and `**` are performed elementwise on NumPy arrays. Let's create a NumPy array and do some computations:


```python
M = np.array([[3,4],[-1,5]])
print(M)
```

    [[ 3  4]
     [-1  5]]



```python
M * M
```




    array([[ 9, 16],
           [ 1, 25]])



### Matrix Multiplication

We use the `@` operator to do matrix multiplication with NumPy arrays:


```python
M @ M
```




    array([[ 5, 32],
           [-8, 21]])



Let's compute $2I + 3A - AB$ for

$$
A = \begin{bmatrix}
1 & 3 \\\
-1 & 7
\end{bmatrix}
\ \ \ \
B = \begin{bmatrix}
5 & 2 \\\
1 & 2
\end{bmatrix}
$$

and $I$ is the identity matrix of size 2:


```python
A = np.array([[1,3],[-1,7]])
print(A)
```

    [[ 1  3]
     [-1  7]]



```python
B = np.array([[5,2],[1,2]])
print(B)
```

    [[5 2]
     [1 2]]



```python
I = np.eye(2)
print(I)
```

    [[1. 0.]
     [0. 1.]]



```python
2*I + 3*A - A@B
```




    array([[-3.,  1.],
           [-5., 11.]])



### Matrix Powers

There's no symbol for matrix powers and so we must import the function `matrix_power` from the subpackage `numpy.linalg`.


```python
from numpy.linalg import matrix_power as mpow
```


```python
M = np.array([[3,4],[-1,5]])
print(M)
```

    [[ 3  4]
     [-1  5]]



```python
mpow(M,2)
```




    array([[ 5, 32],
           [-8, 21]])




```python
mpow(M,5)
```




    array([[-1525,  3236],
           [ -809,    93]])



Compare with the matrix multiplcation operator:


```python
M @ M @ M @ M @ M
```




    array([[-1525,  3236],
           [ -809,    93]])




```python
mpow(M,3)
```




    array([[-17, 180],
           [-45,  73]])




```python
M @ M @ M
```




    array([[-17, 180],
           [-45,  73]])



###  Tranpose

We can take the transpose with `.T` attribute:


```python
print(M)
```

    [[ 3  4]
     [-1  5]]



```python
print(M.T)
```

    [[ 3 -1]
     [ 4  5]]


Notice that $M M^T$ is a symmetric matrix:


```python
M @ M.T
```




    array([[25, 17],
           [17, 26]])



### Inverse

We can find the inverse using the function `scipy.linalg.inv`:


```python
A = np.array([[1,2],[3,4]])
print(A)
```

    [[1 2]
     [3 4]]



```python
la.inv(A)
```




    array([[-2. ,  1. ],
           [ 1.5, -0.5]])



### Trace

We can find the trace of a matrix using the function `numpy.trace`:


```python
np.trace(A)
```




    5



### Norm

*Under construction*

### Determinant

We find the determinant using the function `scipy.linalg.det`:


```python
A = np.array([[1,2],[3,4]])
print(A)
```

    [[1 2]
     [3 4]]



```python
la.det(A)
```




    -2.0



### Dot Product

*Under construction*

## Examples

### Characteristic Polynomials and Cayley-Hamilton Theorem

The characteristic polynomial of a 2 by 2 square matrix $A$ is

$$
p_A(\lambda) = \det(A - \lambda I) = \lambda^2 - \mathrm{tr}(A) \lambda + \mathrm{det}(A)
$$

The [Cayley-Hamilton Theorem](https://en.wikipedia.org/wiki/Cayley%E2%80%93Hamilton_theorem) states that any square matrix satisfies its characteristic polynomial. For a matrix $A$ of size 2, this means that

$$
p_A(A) = A^2 - \mathrm{tr}(A) A + \mathrm{det}(A) I = 0
$$

Let's verify the Cayley-Hamilton Theorem for a few different matrices.


```python
print(A)
```

    [[1 2]
     [3 4]]



```python
trace_A = np.trace(A)
det_A = la.det(A)
I = np.eye(2)
A @ A - trace_A * A + det_A * I
```




    array([[0., 0.],
           [0., 0.]])



Let's do this again for some random matrices:


```python
N = np.random.randint(0,10,[2,2])
print(N)
```

    [[1 9]
     [4 3]]



```python
trace_N = np.trace(N)
det_N = la.det(N)
I = np.eye(2)
N @ N - trace_N * N + det_N * I
```




    array([[0., 0.],
           [0., 0.]])



### Projections

The formula to project a vector $v$ onto a vector $w$ is

$$
\mathrm{proj}_w(v) = \frac{v \cdot w}{w \cdot w} w
$$

Let's write a function called `proj` which computes the projection $v$ onto $w$.


```python
def proj(v,w):
    '''Project vector v onto w.'''
    v = np.array(v)
    w = np.array(w)
    return np.sum(v * w)/np.sum(w * w) * w   # or (v @ w)/(w @ w) * w
```


```python
proj([1,2,3],[1,1,1])
```




    array([2., 2., 2.])



## Exercises

1. Write a function which takes an input parameter $A$, $i$ and $j$ and returns the dot product of the $i$th and $j$th row (indexing starts at 0).
2. Compute the matrix equation $AB + 2B^2 - I$ for matrices $A = \begin{bmatrix} 3 & 4 \\\ -1 & 2 \end{bmatrix}$ and $B = \begin{bmatrix} 5 & 2 \\\ 8 & -3 \end{bmatrix}$.              

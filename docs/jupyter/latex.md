# LaTeX

[LaTeX](https://www.latex-project.org/) is a typesetting language for producing scientific documents. We introduce a *very* small part of the language for writing mathematical notation. Jupyter notebook recognizes LaTeX code written in markdown cells and renders the symbols in the browser using the [MathJax](https://www.mathjax.org/) JavaScript library.

## Mathematics Inline and Display

Enclose LaTeX code in dollar signs `$ ... $` to display math inline. For example, the code `$\int_a^b f(x) = F(b) - F(a)$` renders inline as $ \int_a^b f(x) dx = F(b) - F(a) $.

Enclose LaTeX code in double dollar signs `$$ ... $$` to display expressions in a centered paragraph. For example:

```LaTeX
$$f'(a) = \lim_{x \to a} \frac{f(x) - f(a)}{x-a}$$
```

renders as

$$f'(a) = \lim_{x \to a} \frac{f(x) - f(a)}{x-a}$$

See the [LaTeX WikiBook](https://en.wikibooks.org/wiki/LaTeX) for more information (especially the section on [mathematics](https://en.wikibooks.org/wiki/LaTeX/Mathematics)).

## Common Symbols

Below we give a *partial list* of commonly used mathematical symbols. Most other symbols can be inferred from these examples. See the [LaTeX WikiBook (Mathematics)](https://en.wikibooks.org/wiki/LaTeX/Mathematics) and the [Detexify App](http://detexify.kirelabs.org/classify.html) to find any symbol you can think of!

| Syntax | Output |
| :---: | :---: |
| `$x_n$` | $x_n$ |
| `$x^2$` | $x^2$ |
| `$\infty$` | $\infty$ |
| `$\frac{a}{b}$` | $\frac{a}{b}$ |
| `$\partial$` | $\partial$ |
| `$\alpha$` | $\alpha$ |
| `$\beta$` | $\beta$ |
| `$\gamma$` | $\gamma$ |
| `$\Gamma$` | $\Gamma$ |
| `$\Delta$` | $\Delta$ |
| `$\sin$` | $\sin$ |
| `$\cos$` | $\cos$ |
| `$\tan$` | $\tan$ |
| `$\sum_{n=0}^{\infty}$` | $\sum_{n=0}^{\infty}$ |
| `$\prod_{n=0}^{\infty}$` | $\prod_{n=0}^{\infty}$ |
| `$\int_a^b$` | $\int_a^b$ |
| `$\lim_{x \to a}$` | $\lim_{x \to a}$ |
| `$\mathrm{Hom}$` | $\mathrm{Hom}$ |
| `$\mathbf{v}$` | $\mathbf{v}$ |
| `$\mathbb{Z}$` | $\mathbb{Z}$ |
| `$\mathscr{L}$` | $\mathscr{L}$ |
| `$\mathfrak{g}$` | $\mathfrak{g}$ |
| `$\dots$` | $\dots$ |
| `$\vdots$` | $\vdots$ |
| `$\ddots$` | $\ddots$ |

## Matrices and Brackets

Create a matrix without brackets:

```LaTeX
$$\begin{matrix} a & b \\ c & d \end{matrix}$$
```

$$
\begin{matrix} a & b \\\ c & d \end{matrix}
$$

Create a matrix with round brackets:

```LaTeX
$$\begin{pmatrix} a & b \\ c & d \end{pmatrix}$$
```

$$
\begin{pmatrix} a & b \\\ c & d \end{pmatrix}
$$

Create a matrix with square brackets:

```LaTeX
$$\begin{bmatrix} 1 & 2 & 1 \\ 3 & 0 & 1 \\ 0 & 2 & 4 \end{bmatrix}$$
```

$$
\begin{bmatrix}
1 & 2 & 1 \\\
3 & 0 & 1 \\\
0 & 2 & 4
\end{bmatrix}
$$

Use `\left` and `\right` to enclose an arbitrary expression in brackets:

```LaTeX
$$\left( \frac{p}{q} \right)$$
```

$$\left( \frac{p}{q} \right)$$

## Examples

### Derivative

The [derivative](https://en.wikipedia.org/wiki/Derivative) $f'(a)$ of the function $f(x)$ at the point $x=a$ is the limit

```LaTeX
$$f'(a) = \lim_{x \to a} \frac{f(x) - f(a)}{x - a}$$
```

$$f'(a) = \lim_{x \to a} \frac{f(x) - f(a)}{x - a}$$

### Continuity

A function $f(x)$ is [continuous](https://en.wikipedia.org/wiki/Continuous_function) at a point $x=a$ if

```LaTeX
$$\lim_{x \to a^-} f(x) = f(a) = \lim_{x \to a^+} f(x)$$
```

$$\lim_{x \to a^-} f(x) = f(a) = \lim_{x \to a^+} f(x)$$

### MacLaurin Series

The [MacLaurin series](https://en.wikipedia.org/wiki/Taylor_series) for $e^x$ is

```LaTeX
$$e^x = \sum_{k=0}^{\infty} \frac{x^k}{k!}$$
```

$$e^x = \sum_{k=0}^{\infty} \frac{x^k}{k!}$$

### Jacobian Matrix

The [Jacobian matrix](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) of the function $\mathbf{f}(x_1, \dots, x_n)$ is

```LaTeX
$$
\mathbf{J}
=
\frac{d \mathbf{f}}{d \mathbf{x}}
=
\left[ \frac{\partial \mathbf{f}}{\partial x_1}
\cdots \frac{\partial \mathbf{f}}{\partial x_n} \right]
=
\begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots &
\frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots &
\frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$
```

$$
\mathbf{J} = \frac{d \mathbf{f}}{d \mathbf{x}} =
\left[ \frac{\partial \mathbf{f}}{\partial x_1}
\cdots \frac{\partial \mathbf{f}}{\partial x_n} \right] =
\begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\\
\vdots & \ddots & \vdots \\\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$

## Exercises

1. Write LaTeX code to display the [angle sum identity](https://en.wikipedia.org/wiki/List_of_trigonometric_identities#Angle_sum_and_difference_identities)

    $$\cos(\alpha \pm \beta) = \cos \alpha \cos \beta \mp \sin \alpha \sin \beta$$

2. Write LaTeX code to display the [indefinite integral](https://en.wikipedia.org/wiki/Lists_of_integrals)

    $$\int \frac{1}{1 + x^2} \, dx = \arctan x + C$$

3. Write LaTeX code to display the [Navier-Stokes Equation for Incompressible Flow](https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations)

    $$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} - \nu \nabla^2 \mathbf{u} = - \nabla w + \mathbf{g}$$

4. Write LaTeX code to display [Green's Theorem](https://en.wikipedia.org/wiki/Green%27s_theorem)

    $$\oint_C (L dx + M dy) = \iint_D \left( \frac{\partial M}{\partial x} - \frac{\partial L}{\partial y} \right) dx \, dy$$

5. Write LaTeX code to display the [Prime Number Theorem](https://en.wikipedia.org/wiki/Prime_number_theorem)

    $$\lim_{x \to \infty} \frac{\pi(x)}{ \frac{x}{\log(x)}} = 1$$

6. Write LaTeX code to display the general formula for [Taylor series](https://en.wikipedia.org/wiki/Taylor_series)

    $$\sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!} (x-a)^n$$

7. Write LaTeX code to display [Stokes' Theorem](https://en.wikipedia.org/wiki/Stokes%27_theorem)

    $$\int_{\partial \Omega} \omega = \int_{\Omega} d \omega$$

8. Write LaTeX code to display the adjoint property of the [tensor product](https://en.wikipedia.org/wiki/Tensor_product)

    $$\mathrm{Hom}(U \otimes V,W) \cong \mathrm{Hom}(U, \mathrm{Hom}(V,W))$$

9. Write LaTeX code to display the definition of the [Laplace transform](https://en.wikipedia.org/wiki/Laplace_transform)

    $$\mathscr{L} \{ f(t) \} = F(s) = \int_0^{\infty} f(t) e^{-st} dt$$

10. Write LaTeX code to display the [inverse matrix](https://en.wikipedia.org/wiki/Invertible_matrix) formula

    $$\begin{bmatrix} a & b \\\ c & d \end{bmatrix}^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\\ -c & a \end{bmatrix}$$

11. Write LaTeX code to display the [infinite product formula](https://en.wikipedia.org/wiki/List_of_trigonometric_identities#Infinite_product_formulae)

    $$\sin x = x \prod_{n=1}^{\infty} \left( 1 - \frac{x^2}{\pi^2 n^2} \right)$$

12. Pick your favourite math course and write the notes from your last class in LaTeX.

# Markdown

[Markdown](https://en.wikipedia.org/wiki/Markdown) is a simple text-to-HTML markup language written in plain text. Jupyter notebook recognizes markdown and renders markdown code as HTML. In this section, we present the basic features of markdown.

![Markdown in Jupyter notebook](../img/jupyter/markdown.gif)

See [Markdown](https://daringfireball.net/projects/markdown/) (by John Gruber) and [GitHub Markdown Help](https://help.github.com/articles/basic-writing-and-formatting-syntax/) for more information.

## Text

| Output | Syntax |
| :---: | :---: |
| *emphasis* | `*emphasis*` |
| **strong** | `**strong**` |
| `code` | ``  `code` `` |

## Headings

| Output | Syntax |
| :---: | :---: |
| <h1>Heading 1</h1> | `# Heading 1` |
| <h2>Heading 2</h2> | `## Heading 2` |
| <h3>Heading 3</h3> | `### Heading 3` |
| <h4>Heading 4</h4> | `#### Heading 4` |
| <h5>Heading 5</h5> | `##### Heading 5` |
| <h6>Heading 6</h6> | `###### Heading 6` |

## Lists

Create an ordered list using numbers:

```plaintext
1. Number theory
2. Algebra
3. Partial differential equations
4. Probability
```

1. Number theory
2. Algebra
3. Partial differential equations
4. Probability

Create an unordered list using an asterisk * for each item:

```plaintext
* Number theory
* Algebra
* Partial differential equations
* Probability
```

* Number theory
* Algebra
* Partial differential equations
* Probability

Use indentation to create nested lists:

```plaintext
1. Mathematics
  * Calculus
  * Linear Algebra
  * Probability
2. Physics
  * Classical Mechanics
  * Relativity
  * Thermodynamics
3. Biology
  * Diffusion and Osmosis
  * Homeostasis
  * Immunology
```

1. Mathematics
    * Calculus
    * Linear Algebra
    * Probability
2. Physics
    * Classical Mechanics
    * Relativity
    * Thermodynamics
3. Biology
    * Diffusion and Osmosis
    * Homeostasis
    * Immunology

## Links

Create a link with the syntax `[description](url)`. For example:

```plaintext
[UBC Math](http://www.math.ubc.ca)
```

creates the link [UBC Math](http://www.math.ubc.ca).

## Images

Include an image using the syntax `![description](url)`. For example:

```plaintext
![Jupyter logo](http://jupyter.org/assets/nav_logo.svg)
```

displays the image

![Jupyter logo](http://jupyter.org/assets/nav_logo.svg)

## Tables

Create a table by separating entries by pipe characters |:

```plaintext
| Python Operator | Description  |
| :---: | :---: |
| `+` | addition |
| `-` | subtraction |
| `*` | multiplication |
| `/` | division |
| `**` | power |
```

| Python Operator | Description  |
| :---: | :---: |
| `+` | addition |
| `-` | subtraction |
| `*` | multiplication |
| `/` | division |
| `**` | power |

The syntax `:---:` specifies the alignment (centered in this case) of the columns. See more about [GitHub flavoured markdown](https://help.github.com/articles/organizing-information-with-tables/).

## Exercises

1. Create a numbered list of the top 5 websites you visit most often and include a link for each site.
2. Write a short biography of your favourite mathematician, provide a link to their Wikipedia page and include an image (with a link and description of the source).
3. Create a table of all the courses that you have taken in university. Include the columns: course number, course title, year (that you took the class), and instructor name.

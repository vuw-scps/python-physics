# Jupyter Notebook

From the official webpage [jupyter.org](http://jupyter.org):

> The Jupyter Notebook is a web application that allows you to create and share documents that contain live code, equations, visualizations and explanatory text.

In this section, we present the basic features of Jupyter notebooks. See the *User Interface Tour* in the Help menu in the Jupyter notebook.

![Jupyter user interface tour](../img/jupyter/user-interface-tour.gif)

## Cells

There are two main types of cells: code cells and markdown cells. Hit `SHIFT+ENTER` to execute the contents of a cell.

Markdown cells contain:

* markdown
* HTML
* LaTeX
* plain text
* images
* videos
* *Anything* that a browser can understand

For more information about markdown see [Markdown Basics on GitHub](https://help.github.com/articles/basic-writing-and-formatting-syntax/) and [Markdown Syntax](https://daringfireball.net/projects/markdown/syntax).

Python code is written in code cells. Hit `SHIFT+ENTER` to execute the code. Output is displayed below the code cell:


```python
# Python code to display the first 10 square numbers
for n in range(1,11):
    print(n**2)
```

    1
    4
    9
    16
    25
    36
    49
    64
    81
    100


## Modes

There are two modes: edit mode and command mode. Press `ESC` to enter command mode and `ENTER` for edit mode.

Edit mode is for writing text and code in the cell. Edit mode is indicated by a green border around the cell.

Command mode is for notebook editing commands such as cut cell, paste cell, and insert cell above. Command mode is indicated by a blue border around the cell.

## Keyboard Shortcuts

The toolbar has buttons for common actions however you can increase the speed of your workflow by memorizing the following keyboard shortcuts in command mode:

| Command Mode Action | Shortcut |
| :---: | :---: |
| insert empty cell above | `a` |
| insert empty cell below | `b` |
| copy cell | `c` |
| cut cell | `x` |
| paste cell below | `v` |
| to code cell | `y` |
| to markdown cell | `m` |
| save and checkpoint | `s` |
| execute cell | `SHIFT+ENTER` |
| to edit mode | `ENTER` |
| to command mode | `ESC` |

See *Help* in the toolbar of the Jupyter notebook to see the list of keyboard shortcuts.

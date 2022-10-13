## Regular expressions

https://byuidatascience.github.io/python4ds/strings.html

https://jakevdp.github.io/WhirlwindTourOfPython/14-strings-and-regular-expressions.html


```python
import numpy as np
import pandas as pd
import re
```

First we try to read one file to make sure we're using the correct parameters:


```python
s = 'filenames/Labram_laser514nm_1mW_30sx4_800nmgrating_T-200K_sample-B.txt'
```


```python
pd.Series(["x", "y", "z"]).str.cat()
```




    'xyz'




```python
pd.Series(["x", "y"]).str.cat(sep = '_')
```




    'x_y'



This seems reasonable, but now we'd like to extract parameters from the filename.


```python
x = pd.Series(["Apple", "Banana", "Pear"])
x.str[0:3]
```




    0    App
    1    Ban
    2    Pea
    dtype: object




```python
x.str[-3:]
```




    0    ple
    1    ana
    2    ear
    dtype: object



### matching patterns with regular expressions


```python
import glob
import os

x = pd.Series(["apple", "banana", "pear"])
x.str.replace("an", "zzz")
```




    0       apple
    1    bzzzzzza
    2        pear
    dtype: object




```python
x.str.replace("^a", "zzz", regex=True)
```




    0    zzzpple
    1     banana
    2       pear
    dtype: object



Repetitions

- ?: 0 or 1
- +: 1 or more
- *: 0 or more


```python
s = 'filenames/Labram_laser514nm_1mW_30sx4_800nmgrating_T-200K_sample-B.txt'

# temp = re.compile('[0-9]{3}')
regex = re.compile('[0-9]{3}K')
regex = re.compile('[0-9]+K')
regex = re.compile('([0-9]+)K')

regex.findall(s)



```




    ['200']



_Download this page [as a Jupyter notebook](https://github.com/vuw-scps/python-physics/raw/master/notebooks/phys345/wrangling_00_regex.ipynb) or as a [standalone Python script](https://github.com/vuw-scps/python-physics/raw/master/scripts/phys345/wrangling_00_regex.py)._



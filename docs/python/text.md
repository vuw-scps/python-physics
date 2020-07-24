# Text

*Under construction*

<!--

## Strings

[Strings](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str) are defined using (single or double) quotes:


```python
mathematician = 'Ramanujan'
print(mathematician)
```

    Ramanujan

A [string](https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str) is a sequence of characters enclosed in quotes.

```python
name = 'Patrick'
print(name)
```

```output
Patrick
```

```python
sentence = 'Math 210 is the best class in the world?!!'
print(sentence)
```

```output
Math 210 is the best class in the world?!!
```

```python
quotes = 'Use a double quote " inside single quotes'
print(quotes)
```

```output
Use a double quote " inside single quotes
```

Strings are sequences of characters and so we can access the characters in a string just like the elements of a list.

```python
name[0]
```

```output
'P'
```

```python
name[-1]
```

```output
'k'
```

```python
type(name)
```

```output
str
```

-->


<!--


## 1. Text data

### Creating strings

The text datatype in Python is called [string](https://docs.python.org/3/tutorial/introduction.html#strings) (`str`). We write strings by typing text enclosed in single or double or triple quotes.


```python
course = 'MATH 210 Introduction to Mathematical Computing'
```


```python
print(course)
```

    MATH 210 Introduction to Mathematical Computing



```python
type(course)
```




    str



Or use double quotes:


```python
course = "MATH 210 Introduction to Mathematical Computing"
```


```python
print(course)
```

    MATH 210 Introduction to Mathematical Computing


Generally, we use double quote's if our string contains a single quote.


```python
today = "It's a rainy day."
```


```python
print(today)
```

    It's a rainy day.


Use triple quotes to write a multiline string:


```python
lyrics = '''To the left, to the left
To the left, to the left
To the left, to the left
Everything you own in the box to the left
In the closet that's my stuff, yes
If I bought it please don't touch'''
```


```python
print(lyrics)
```

    To the left, to the left
    To the left, to the left
    To the left, to the left
    Everything you own in the box to the left
    In the closet that's my stuff, yes
    If I bought it please don't touch


### Strings are sequences

A string is a sequence type and so we can use strings in `for` loops and list comprehensions. For example:


```python
word = 'Math'
for letter in word:
    print('Gimme a',letter + '!')
print('What does that spell?!',word + '!')
```

    Gimme a M!
    Gimme a a!
    Gimme a t!
    Gimme a h!
    What does that spell?! Math!


Note that the addition operator acts as concatenation of strings:


```python
'MATH' + '210'
```




    'MATH210'



We can also convert strings to lists of characters:


```python
list('Mathematics')
```




    ['M', 'a', 't', 'h', 'e', 'm', 'a', 't', 'i', 'c', 's']



Use index syntax just like for lists to access characters in a string:


```python
password = 'syzygy'
```


```python
password[2]
```




    'z'



### String methods

There are *many* [string methods](https://docs.python.org/3/library/stdtypes.html#string-methods) available to manipulate strings. Let's try a few methods:


```python
sentence = "The quick brown fox jumped over the lazy dog."
```


```python
uppercase_sentence = sentence.upper()
```


```python
sentence
```




    'The quick brown fox jumped over the lazy dog.'




```python
uppercase_sentence
```




    'THE QUICK BROWN FOX JUMPED OVER THE LAZY DOG.'



A string (like all Python datatypes) is an [*object*](https://docs.python.org/3/tutorial/classes.html): it's a collection of data *and* methods for manipulating the data. We use the dot notation to access the methods of an object.


```python
euler = "Euler's Method"
```


```python
euler.replace('E','3')
```




    "3uler's Method"




```python
euler.replace
```




    <function str.replace>


-->

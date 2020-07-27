# Molecular Speeds lab Python data analysis - multiple gas dataset processing

This Python workshop was originally developed by VUW students William Holmes-Hewett and Campbell Matthews.  Thanks to both of them for their energy and enthusiams in making this resource.

Take a look at the example code you have been given - or start to workthrough from here.  

## Importing modules

Let's begin by importing all the modules we will need at the top of the code and adding in the variables that we know we will need for the analysis.  


```python
import numpy as np
from matplotlib import pyplot as plt
from math import log
import math
```

## Setting up variables and constants required later


```python
V=2.03*10**-3
A=5.55113*10**-8
ef_begin=0.26
```

What we are going to do here is run a for loop over all the code we wrote for all files in a specific folder. This will be the folder we are in, so make sure that's where you are and you have all the data from the workshop folder.   

## New moldues for file organisation

To start with we need to import the other modules we are going to use.

os (or Miscellaneous Operating system interfaces) Lets us look up file paths and folder locations

Glob lets us find all files matching a pattern

These are file organisation tools.  You can look up more about them if you like.


```python
import os
import glob
```

The first thing we want to do is set the location of our folder which is where the script is stored i.e. CWD or current working directory.


```python
folder = os.getcwd()
```

## File formats

Now we know where the files are stored we need to set the pattern (or type of files) for Glob to look for.


```python
fileformat=folder+str('/*.csv')
```

This will make the script only look for csv files in the correct folder.

We now want to create an array of the filenames, these are named after the gasses and mass numbers.


```python
files = glob.glob(fileformat)
```

## Building the arrays

Create an array of masses from filenames to use later, as filenames are  strings we need to set them as intergers.  Note that -ve indicies run from the end.


```python
masses = [int(f[-6:-4]) for f in files]
```

It might not be totally clear what this is doing unless we print the output. 


```python
print(files)
```

    ['C:\\Users\\plankna\\Desktop\\PythonWorkshop\\Intro_documents\\Multiple\\Argon 38.csv', 'C:\\Users\\plankna\\Desktop\\PythonWorkshop\\Intro_documents\\Multiple\\Carbon Dioxide 40.csv', 'C:\\Users\\plankna\\Desktop\\PythonWorkshop\\Intro_documents\\Multiple\\Chlorine 71.csv', 'C:\\Users\\plankna\\Desktop\\PythonWorkshop\\Intro_documents\\Multiple\\Fluorine 38.csv', 'C:\\Users\\plankna\\Desktop\\PythonWorkshop\\Intro_documents\\Multiple\\Krypton 84.csv', 'C:\\Users\\plankna\\Desktop\\PythonWorkshop\\Intro_documents\\Multiple\\Methane 16.csv', 'C:\\Users\\plankna\\Desktop\\PythonWorkshop\\Intro_documents\\Multiple\\Neon 17.csv', 'C:\\Users\\plankna\\Desktop\\PythonWorkshop\\Intro_documents\\Multiple\\Nitrogen 28.csv', 'C:\\Users\\plankna\\Desktop\\PythonWorkshop\\Intro_documents\\Multiple\\Oxygen 32.csv', 'C:\\Users\\plankna\\Desktop\\PythonWorkshop\\Intro_documents\\Multiple\\Sulphur Dioxide 64.csv']


argh!!!  depending on your filing system this might look horrible...

I think it becomes clearer if we print the output of the array.


```python
print(masses)
```

    [38, 40, 71, 38, 84, 16, 17, 28, 32, 64]


Hopefully this helps to see what has gone on.  We have taken the string from the file name and turned them into an integer.  To do this we counted back from the end of the file name...

We now need to find out how many files we have in the folder.  We know how to do that already.


```python
n=len(files)
```

We also need to create an empty array to populate later.


```python
v=np.zeros(n)
```

## Applying the previous code to multiple files

We now want to simply run though the previous code (in the single gas experiment) for all of the files in the folder.


```python
for j in range(n):

    #we need to indent all this code and change the argument in genfromtext 
    #to files[j] 
    
    rawdata=np.genfromtxt(files[j], delimiter=',')
    time=rawdata[:,0]
    pressure=rawdata[:,1]
    baseP=min(rawdata[:,1])
    P_minus_baseP=[i-0.99*baseP for i in pressure]
    res=[abs(i-ef_begin) for i in P_minus_baseP]
    minres=min(res)
    for i in range (len(res)):
        if res[i]==minres:
            Peff_index=i
            break
    LogP=[log(i) for i in P_minus_baseP]
    fit=np.polyfit(time[Peff_index:1400],LogP[Peff_index:1400],1)
    m=fit[0]
    c=fit[1]
    
```

note, we no longer have cats... ;-)

compare this to your previous code.  It is the same, just compressed.

Now I am adding on to this and plotting the data.


```python
for j in range(n):

    #we need to indent all this code and change the argument in genfromtext 
    #to files[j] 
    
    rawdata=np.genfromtxt(files[j], delimiter=',')
    time=rawdata[:,0]
    pressure=rawdata[:,1]
    baseP=min(rawdata[:,1])
    P_minus_baseP=[i-0.99*baseP for i in pressure]
    res=[abs(i-ef_begin) for i in P_minus_baseP]
    minres=min(res)
    for i in range (len(res)):
        if res[i]==minres:
            Peff_index=i
            break
    LogP=[log(i) for i in P_minus_baseP]
    fit=np.polyfit(time[Peff_index:1400],LogP[Peff_index:1400],1)
    m=fit[0]
    c=fit[1]
    
    
    #now we put each of these in one of the spots in the vector v
    
    v[j]=-4*V*m/A


#now lets print v

######################print(v)

#we can then plot v as a function of mass number

plt.scatter(masses,v)

#we can now create a theoritical line to match these to

#first we need a range of mass numbers
    
    
    
```




    <matplotlib.collections.PathCollection at 0x2501a8d68d0>




![png](../nb_img/phys221/Multiplots_38_1.png)


## Analysis


```python
#now we simply complete the theoritical calculation

vtheory=[math.sqrt(8/math.pi*1.38*(10**-23)*300/i/(1.67*10**-27)) for i in range(10,90)]

#and plot the result on our plot

plt.plot(range(10,90),vtheory,'k--')

#we can add some labels

plt.xlabel('Mass Number')
plt.ylabel('Velocity')
```




    Text(0, 0.5, 'Velocity')




![png](../nb_img/phys221/Multiplots_40_1.png)


Putting these last two parts together we find (this will be what comes up in Spyder).


```python
for j in range(n):

    #we need to indent all this code and change the argument in genfromtext 
    #to files[j] 
    
    rawdata=np.genfromtxt(files[j], delimiter=',')
    time=rawdata[:,0]
    pressure=rawdata[:,1]
    baseP=min(rawdata[:,1])
    P_minus_baseP=[i-0.99*baseP for i in pressure]
    res=[abs(i-ef_begin) for i in P_minus_baseP]
    minres=min(res)
    for i in range (len(res)):
        if res[i]==minres:
            Peff_index=i
            break
    LogP=[log(i) for i in P_minus_baseP]
    fit=np.polyfit(time[Peff_index:1400],LogP[Peff_index:1400],1)
    m=fit[0]
    c=fit[1]
    
    
    #now we put each of these in one of the spots in the vector v
    
    v[j]=-4*V*m/A


#now lets print v

######################print(v)

#we can then plot v as a function of mass number

plt.scatter(masses,v)

#we can now create a theoritical line to match these to

#first we need a range of mass numbers



#now we simply complete the theoritical calculation

vtheory=[math.sqrt(8/math.pi*1.38*(10**-23)*300/i/(1.67*10**-27)) for i in range(10,90)]

#and plot the result on our plot

plt.plot(range(10,90),vtheory,'k--')

#we can add some labels

plt.xlabel('Mass Number')
plt.ylabel('Velocity')
```




    Text(0, 0.5, 'Velocity')




![png](../nb_img/phys221/Multiplots_42_1.png)


Compare this to your results from the PHYS223 lab.  What are the advantages of using the Python script?

_Download this page [as a Jupyter notebook](https://github.com/vuw-scps/python-physics/raw/master/notebooks/phys221/Multiplots.ipynb) or as a [standalone Python script](https://github.com/vuw-scps/python-physics/raw/master/scripts/phys221/Multiplots.py)._

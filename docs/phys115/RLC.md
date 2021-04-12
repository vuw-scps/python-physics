## RLC circuit




```python
from matplotlib import pyplot as plt
import numpy as np
```

## Constants




```python
I=1
R=2
L=3
C=4
f=5

w=2*np.pi*f
```

### IMPEDANCE


```python
XL=w*L
XC=1/(w*C)
X=XL-XC

# Magnitude and angle of impedance
Z=np.sqrt((R**2+X**2))
Angle=np.arctan(X/R)
```

### VOLTAGE


```python
VR=I*R
VL=I*XL
VC=I*XC

# Magnitude and angle of total voltage
V=np.sqrt((VR**2+(VL-VC)**2))
Angle=np.arctan((VL-VC)/VR)

print("")
print("voltage, current, and impedance in the circuit")
print("Current (I)"," ","="," ",I,"Amperes")
print("Impedance (Z) in 'i,j' notation"," ","="," ",R,"+",X,"j"," ","Ohms")
print("Voltage (V) in 'i,j' notation"," ","="," ",V,"+",VL-VC,"j"," ","Volts")

print("")
print("voltage, current, and resistance in the resistor")
print("Current (I)"," ","="," ",I,"Amperes")
print("Resistance (R)"," ","="," ",R,"Ohms")
print("Voltage (VR)","="," ",VR,"Volts")
```

    
    voltage, current, and impedance in the circuit
    Current (I)   =   1 Amperes
    Impedance (Z) in 'i,j' notation   =   2 + 94.2398218605392 j   Ohms
    Voltage (V) in 'i,j' notation   =   94.26104192245151 + 94.2398218605392 j   Volts
    
    voltage, current, and resistance in the resistor
    Current (I)   =   1 Amperes
    Resistance (R)   =   2 Ohms
    Voltage (VR) =   2 Volts


_Download this page [as a Jupyter notebook](https://github.com/vuw-scps/python-physics/raw/master/notebooks/phys115/RLC.ipynb) or as a [standalone Python script](https://github.com/vuw-scps/python-physics/raw/master/scripts/phys115/RLC.py)._

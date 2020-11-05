Interactive visualisation of spherical harmonics.

## Required dependencies

We import the `sph_harm` function from `scipy`'s _special functions_ to calculate spherical harmonics of any degree and order, and `plotly` for the interactive 3D display.


```python
import numpy as np
import scipy as sp
from scipy.special import sph_harm

import plotly.graph_objects as go
import plotly.io as pio 
# pio.renderers.default = "browser"
```

## Spherical harmonics

We'll evaluate the scalar harmonic $Y_{lm}$ on a grid of $(\theta, \phi)$ values, and multiply by the corresponding radial function over a grid of $r$ values.


```python
l = 5
m = 3

R = 10.
Np = 36
Nt = 36
Nr=10
rhalf=Nr//2 -1

r = np.linspace(0.99*R, 1.01*R, Nr)
theta = np.linspace(0, np.pi, Nt)
phi = np.linspace(0, 2*np.pi, Np)

d_r = r[1]-r[0] 
d_theta = theta[1]-theta[0] 
d_phi = phi[1]-phi[0]

theta, phi, r = np.meshgrid(theta, phi, r)


Ylm = 1/r**(l+1) * sph_harm(m, l, phi, theta).real
```

## Gradient – magnetic field

From that 3D array representing the scalar potential we can evaluate a numerical gradient in spherical coordinates, which represents the vector magnetic field.


```python


# prefactors
oneoverr = 1/r
oneoverrsintheta = 1/(r*np.sin(theta)+1e-12)

costheta = np.cos(theta)[:,:,rhalf]
sintheta = np.sin(theta)[:,:,rhalf]
cosphi = np.cos(phi)[:,:,rhalf]
sinphi = np.sin(phi)[:,:,rhalf]

xx = R*sintheta*cosphi 
yy = R*sintheta*sinphi 
zz = R*costheta 


def field_gradient(Y):
    deriv = np.gradient(Y)
    ## from partial derivatives to spherical gradient
    dVt = deriv[1]
    dVp = deriv[0] 
    dVr = deriv[2]
    
    Br = -dVr[:,:,rhalf] * 1/d_r
    Bt = -oneoverr[:,:,rhalf] * dVt[:,:,rhalf] * 1/d_theta
    Bp = -oneoverrsintheta[:,:,rhalf] * dVp[:,:,rhalf] * 1/d_phi


    Bx = sintheta*cosphi*Br + costheta*cosphi*Bt - sinphi*Bp
    By = sintheta*sinphi*Br + costheta*sinphi*Bt + cosphi*Bp
    Bz = costheta*Br - sintheta*Bt

    u=np.where(np.isfinite(Bx), Bx, 0)
    v=np.where(np.isfinite(By), By, 0)
    w=np.where(np.isfinite(Bz), Bz, 0)
    B2 = u**2 + v**2 + w**2
    maxB = np.sqrt(np.max(B2))
    return(u/maxB, v/maxB, w/maxB)

u, v, w = field_gradient(Ylm)
```

## Plotting

We use `plotly` to produce an interactive 3D surface plot, and add arrows to represent the vector field at a given radial distance.


```python
Rout = 1.1 
sizeref = 0.8
step = 1

fig = go.Figure()

fig.add_trace(go.Surface(x=xx, y=yy, z=zz, 
                surfacecolor=Ylm[:,:,rhalf], 
                showscale=False,  
                colorscale='RdBu'))

fig.add_trace(go.Cone(
  x=Rout*np.concatenate(xx[::step, ::step]),
  y=Rout*np.concatenate(yy[::step, ::step]),
  z=Rout*np.concatenate(zz[::step, ::step]),
  u=np.concatenate(u[::step, ::step]),
  v=np.concatenate(v[::step, ::step]),
  w=np.concatenate(w[::step, ::step]),
  showlegend=False,
  showscale=False,
  colorscale=[(0, "orange"), (0.5, "orange"), (1, "orange")],
  sizemode="absolute",
  sizeref=sizeref))

fig.update_layout(title_text="Y({0},{1})".format(l,m),showlegend=False)

# set annotations white for clarity
fig.update_layout(scene = dict(
                    xaxis = dict(
                        nticks=0,
                        color='white',
                         backgroundcolor="white",
                         gridcolor="white",
                         showbackground=False,
                         zerolinecolor="white",),
                    yaxis = dict(
                        color='white',
                        backgroundcolor="white",
                        gridcolor="white",
                        showbackground=False,
                        zerolinecolor="white"),
                    zaxis = dict(
                        color='white',
                        backgroundcolor="white",
                        gridcolor="white",
                        showbackground=False,
                        zerolinecolor="white",),),
                    width=1200,
                    margin=dict(
                    r=10, l=100,
                    b=10, t=10)
                  )
# fig.show() # use this in interactive notebook
fig.show(renderer='svg')

```


![svg](../nb_img/phys304/Ylm_interactive_7_0.svg)


_Download this page [as a Jupyter notebook](https://github.com/vuw-scps/python-physics/raw/master/notebooks/phys304/Ylm_interactive.ipynb) or as a [standalone Python script](https://github.com/vuw-scps/python-physics/raw/master/scripts/phys304/Ylm_interactive.py)._

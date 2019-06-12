#!/bin/env python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys
import os.path

# In[134]:
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("data", help="either npy file, or one of ('wave')")
parser.add_argument("Ns", type=int, help="num. samples")
parser.add_argument("Nt", type=int, help="Nt")
parser.add_argument("Nx", type=int, help="Nx")
parser.add_argument("Ny", type=int, help="Ny")
parser.add_argument("--split", type=float, help="fraction to split into test set. split<=0: skip", default=0.2)
options = parser.parse_args()

def f(modes, Nt, Nx, Ny, sharpness=2.3):
    x= np.arange(Nx)/Nx
    y= np.arange(Ny)/Ny
    x, y = np.meshgrid(x, y)
    val=np.array([np.sum([np.sin(phase + kx*x +ky*y + t*omega)*np.exp(-t*decay) 
                                   for phase, kx, ky, omega, decay in modes], axis=0) 
                           for t in np.arange(Nt)/Nt])
    return x, y, (np.tanh(val*sharpness)+1)/2

nsimulation=options.Ns
Nt=options.Nt
Nx=options.Nx
Ny=options.Ny

x, y, a=f([[0, 9.2, 4.2, 33.3, 0.9], [3.1, -1.2, 12.2, 25.3, 0.2]], Nt, Nx, Ny)


# In[138]:

np.min(a), np.max(a)
a.shape


# In[144]:


fig = plt.figure()
#ax = fig.gca(projection='3d')
ax = Axes3D(fig)

# Plot the surface.
surf = ax.plot_surface(x, y, a[99], cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.1, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

#plt.show()


# In[142]:

seq_len=20
input_seq_len=seq_len//2

nsteps= nsimulation*Nt

nclips = nsteps//seq_len

alldat=np.zeros((nsimulation, Nt, Nx, Ny),dtype=np.float32)

dims=np.array([[1, Nx, Ny]], dtype=np.int32)

clips=np.vstack([np.arange(0,nsteps,input_seq_len), np.full((nclips*2), input_seq_len)])
clips=clips.astype(np.int32).T.reshape((-1,2,2)).transpose((1,0,2))

print("dims",dims)
print("clips", clips.shape)

if os.path.isfile(options.data):
    alldat=np.load(options.data)
else:
    for i in range(alldat.shape[0]):
        nmodes=np.random.randint(1,3)
        modes=[]
        for imode in range(1, nmodes+1):
            k=np.random.uniform(8, 12)
            theta=np.random.uniform(8, 2*np.pi)
            phase=np.random.uniform(0, 2*np.pi)
            omega=np.random.uniform(15, 20)
            decay=np.random.uniform(0.4, 0.6)
            modes.append([phase, k*np.cos(theta), k*np.sin(theta), omega, decay])
        alldat[i]=f(modes, Nt, Nx, Ny)[-1]

alldat=alldat.reshape((-1,1, Nx, Ny))
ntot=alldat.shape[0]
ntest=int(ntot*options.split)
print("all input_raw_data", alldat.shape)

if ntest<=0 or ntest>ntot:
    np.savez("data.npz", dims=dims, clips=clips, input_raw_data=alldat)
    print("all %d saved to data.npz"%(ntot))
else:
    np.savez("train.npz", dims=dims, clips=clips[:,:-ntest,:], input_raw_data=alldat[:-ntest])
    np.savez("test.npz", dims=dims, clips=clips[:,0:ntest,:], input_raw_data=alldat[0:ntest])
    print("saved %d to train.npz and %d to test.npz"%(ntot-ntest, ntest))

#!/bin/env python
#from IPython.display import HTML
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

data = np.load(sys.argv[1])
data=data.reshape((-1,)+data.shape[-2:])

fig, ax = plt.subplots()

ax.set_xlim((0, 64))
ax.set_ylim((0, 64))

#plt.title('0')
im = ax.imshow(data[0,:,:], cmap=plt.get_cmap('hot'), vmin=np.min(data), vmax=np.max(data))
fig.colorbar(im)

def init():
    im.set_data(data[0,:,:])
    ax.set_title('0')
    return (im,)

# animation function. This is called sequentially
def animate(i):
    data_slice = data[i,:,:]
    im.set_data(data_slice)
    print('debug i', i)
    ax.set_title(str(i))
    return (im,)

# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(data), interval=20, blit=True)

plt.show()
#HTML(anim.to_html5_video())

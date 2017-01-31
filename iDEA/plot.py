"""Plotting output quantities of iDEA
"""
ffmpeg_path = '/rwgdisks/home/lt934/packages/ffmpeg-3.2/ffmpeg'

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot3d(O, name, pm, space='it'):
    """Plot quantity O(r,r';it) or O(r,r';iw)

    Produces mp4 movie

    parameters
    -----------
    O: array_like
      the quantity to be plotted
    name: str
      name of the quantity
    pm: object
      parameters file
    space: str
        'it': last index is complex time
        'iw': last index is complex frequency

    """
    import matplotlib.animation as animation
    plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path
    import MBPT

    st = MBPT.SpaceTimeGrid(pm)
    grid = st.x_npt
    xmax = st.x_max
    # this is something one always needs to do for imshow
    O = O[::-1]
    #O = O.swapaxes(0,1)   # Why was this here?
    extent = [-xmax, xmax, -xmax, xmax]

    tau_npt = st.tau_npt
    if space == 'it':
        tau_grid = st.tau_grid
        label = "$\\tau = {:+.2f}$ a.u."
    elif space == 'iw':
        tau_grid = st.omega_grid
        label = "$\\omega = {:+.2f}$ a.u."
    else:
        raise ValueError("space must be either 'it' or 'iw'")


    expected_shape = np.array([grid,grid,tau_npt])
    if not (O.shape == expected_shape).all():
        raise IOError("Dimensions {} are not ({}) as expected"\
                .format(O.shape, expected_shape))

    vmax = np.maximum(np.max(np.abs(O.real)),np.max(np.abs(O.imag)))
    vmin = -vmax

    # Set up formatting for the movie files
    writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(11,5))
    plt.subplots_adjust(left=0.10)

    print("Plotting {} frames".format(tau_npt))
    ims = []
    for it in range(tau_npt):
        im_r = ax1.imshow(O.real[:,:,it],norm=plt.Normalize(vmin,vmax),
                extent=extent, cmap=matplotlib.cm.bwr)
        im_i = ax2.imshow(O.imag[:,:,it],norm=plt.Normalize(vmin,vmax),
                extent=extent, cmap=matplotlib.cm.bwr)
        label_i = ax2.text(0.8, 0.9,label.format(tau_grid[it]),
                           horizontalalignment='center', verticalalignment='center',
                           transform = ax2.transAxes)
        ims.append( (im_r, im_i, label_i,) )


    for ax in [ax1, ax2]:
        ax.set_xlim([-xmax,xmax])
        ax.set_ylim([-xmax,xmax])
        ax.set_xlabel('x [$a_0$]')
        ax.set_ylabel('y [$a_0$]')
        # put color bars
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        if ax == ax1:
            ax.set_title("{} (real)".format(name))
            plt.colorbar(im_r, cax=cax)
        else:
            plt.colorbar(im_i, cax=cax)
            ax.set_title("{} (imag)".format(name))


    im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000,
                                       blit=True)
    mfile = "animations/{}.mp4".format(name)
    print("Making movie {}".format(mfile))
    print("This may take some time...")
    im_ani.save(mfile, writer=writer)



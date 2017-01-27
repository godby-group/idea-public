"""Plotting output quantities of iDEA
"""
ffmpeg_path = '/rwgdisks/home/lt934/packages/ffmpeg-3.2/ffmpeg'

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot3d(O, name, pm, space='it', format='png'):
    """Plot quantity O(r,r';it) or O(r,r';iw)

    Produces pngs or mp4 movie

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
    format: str
        Output format
        'mp4': mpeg 4 movie (requires ffmpeg)
        'png': collection of portable network graphics

    """
    import matplotlib.animation as animation
    import MBPT

    st = MBPT.SpaceTimeGrid(pm)
    grid = st.x_npt
    xmax = st.x_max
    # this is something one always needs to do for imshow
    O = O[::-1]
    O = O.swapaxes(0,1)
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

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(11,5))
    plt.subplots_adjust(left=0.10)

    for ax in [ax1, ax2]:
        ax.set_xlim([-xmax,xmax])
        ax.set_ylim([-xmax,xmax])
        ax.set_xlabel('x [$a_0$]')
        ax.set_ylabel('y [$a_0$]')
        divider = make_axes_locatable(ax)
        if ax == ax1:
            cax1 = divider.append_axes("right", size="5%", pad=0.05)
            ax.set_title("{} (real)".format(name))
        else:
            cax2 = divider.append_axes("right", size="5%", pad=0.05)
            ax.set_title("{} (imag)".format(name))


    if format == 'png':
        print("Saving {} frames as pngs to plots/".format(tau_npt))
    else:
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

        if it == 0:
            plt.colorbar(im_r, cax=cax1)
            plt.colorbar(im_i, cax=cax2)

        if format=='png':
            plt.savefig("{}/{}_{:04d}.png".format('plots',name,it), dpi=150)
            label_i.remove()
            im_r.remove()
            im_i.remove()
        else:
            ims.append( (im_r, im_i, label_i,) )



    if format =='mp4':
        im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000,
                                           blit=True)
        mfile = "animations/{}.mp4".format(name)
        print("Making movie {}".format(mfile))
        print("This may take some time...")
        plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path
        writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        #writer = animation.MencoderWriter(fps=15, bitrate=1800)
        im_ani.save(mfile, writer=writer)


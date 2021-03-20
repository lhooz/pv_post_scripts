"""circulation processing functions"""

import csv
import os
import shutil

import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from matplotlib import colors
from scipy.ndimage import zoom
from scipy.ndimage.filters import gaussian_filter


def read_sfield(field_data_file):
    """read field (vorticity or q) data"""
    vor_array = []
    with open(field_data_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                vor_array.append(
                    [-float(row[0]),
                     float(row[1]),
                     float(row[3])])
                line_count += 1

        print(f'Processed {line_count} lines in {field_data_file}')

    vor_array = np.array(vor_array)
    return vor_array


def read_vfield(field_data_file):
    """read field (vorticity or q) data"""
    v_array = []
    with open(field_data_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                v_array.append([
                    -float(row[0]),
                    float(row[1]), -float(row[3]),
                    float(row[4])
                ])
                line_count += 1

        print(f'Processed {line_count} lines in {field_data_file}')

    v_array = np.array(v_array)
    return v_array


def read_wgeo(wgeo_data_file):
    """read wing geometry data"""
    wgeo_array = []
    with open(wgeo_data_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                wgeo_array.append([-float(row[0]), float(row[1])])
                line_count += 1

        print(f'Processed {line_count} lines in {wgeo_data_file}')

    wgeo_array = np.array(wgeo_array)

    # ----- sorting points in clockwise order ---
    x = wgeo_array[:, 0]
    y = wgeo_array[:, 1]
    cx = np.mean(x)
    cy = np.mean(y)
    a = np.arctan2(y - cy, x - cx)
    order = a.ravel().argsort()
    x = x[order]
    y = y[order]
    wgeo_array = np.vstack((x, y))

    wgeo_array = np.transpose(wgeo_array)
    w_centroid = np.array([cx, cy])

    return wgeo_array, w_centroid


def grid_vorz(window, resolution, vor_array):
    """grid interpolation for vorticity data"""
    grid_x, grid_y = np.mgrid[window[0]:window[1]:resolution[0] * 1j,
                              window[2]:window[3]:resolution[1] * 1j]

    grid_vz = scipy.interpolate.griddata(vor_array[:, 0:2],
                                         vor_array[:, 2], (grid_x, grid_y),
                                         method='nearest')

    return grid_x, grid_y, grid_vz


def grid_ufield(window, resolution, ufield_array):
    """grid interpolation for vorticity data"""
    grid_x, grid_y = np.mgrid[window[0]:window[1]:resolution[0] * 1j,
                              window[2]:window[3]:resolution[1] * 1j]

    grid_ux = scipy.interpolate.griddata(ufield_array[:, 0:2],
                                         ufield_array[:, 2], (grid_x, grid_y),
                                         method='nearest')

    grid_uy = scipy.interpolate.griddata(ufield_array[:, 0:2],
                                         ufield_array[:, 3], (grid_x, grid_y),
                                         method='nearest')

    return grid_x, grid_y, grid_ux, grid_uy


def single_plot_field(images, axto_plot, window, grid_x, grid_y, sdata, vdata,
                      wdata, imnorm, levels):
    """plot one single field data"""
    images.append(
        axto_plot.imshow(sdata,
                         cmap='RdBu',
                         norm=imnorm,
                         aspect='equal',
                         extent=window,
                         origin='lower',
                         interpolation='bicubic'))
    axto_plot.contour(sdata,
                      levels,
                      linewidths=0.1,
                      colors='k',
                      extent=window,
                      origin='lower')
    axto_plot.quiver(grid_x,
                     grid_y,
                     vdata[0],
                     vdata[1],
                     scale=250,
                     width=0.002)

    nverts = len(wdata)
    codes = np.ones(nverts, int) * path.Path.LINETO
    codes[0] = path.Path.MOVETO
    codes[-1] = path.Path.CLOSEPOLY
    wgeopatch = path.Path(wdata, codes)
    patch = patches.PathPatch(wgeopatch,
                              linewidth=0.2,
                              facecolor='w',
                              edgecolor='k',
                              alpha=1.0)
    axto_plot.add_patch(patch)

    print('plotted image no = %s\n' % '{0:.0f}'.format(len(images)))

    return axto_plot


def field_plot(windows, ufield_gridx, ufield_gridy, sfield_data, vfield_data,
               wgeo_data, markt, oimage_file, mode):
    """plot field data"""
    plt.rcParams.update({
        # "text.usetex": True,
        'mathtext.fontset': 'stix',
        'font.family': 'STIXGeneral',
        'font.size': 18,
        'figure.figsize': (24, 12),
        'lines.linewidth': 0.5,
        'lines.markersize': 0.1,
        'lines.markerfacecolor': 'white',
        'figure.dpi': 300,
    })
    marksc = [r'$\^t$ = ' + '{0:.2f}'.format(x) for x in markt]
    window = windows
    no_c = len(wgeo_data)
    imnorm = colors.Normalize(vmin=-100, vmax=100)
    levels = np.arange(-100.0, 100.0, 15)
    zoom_order = 5

    images = []
    ax_all = []
    #---plot t=0.75 data---
    gs_kw = dict(
        left=0.125,
        right=0.9,
        top=0.9,
        bottom=0.2,
        wspace=0.1,
        hspace=0.0,
    )

    fig, axr1 = plt.subplots(nrows=1, ncols=no_c, gridspec_kw=gs_kw)
    axr1 = [axr1]
    for grid_x, grid_y, sdatai, vdatai, wdatai, axr1i, marksci in zip(
            ufield_gridx, ufield_gridy, sfield_data, vfield_data, wgeo_data,
            axr1, marksc):

        sdatai = zoom(sdatai, zoom_order)
        sdatai = gaussian_filter(sdatai, sigma=10.0)
        single_plot_field(images, axr1i, window, grid_x, grid_y, sdatai,
                          vdatai, wdatai, imnorm, levels)

        axr1i.set_xticklabels([])
        axr1i.set_yticklabels([])

        markx_loc = axr1i.get_xlim()[0] + 0.5 * (axr1i.get_xlim()[1] -
                                                 axr1i.get_xlim()[0])
        marky_loc = axr1i.get_ylim()[1] + 0.1 * (axr1i.get_ylim()[1] -
                                                 axr1i.get_ylim()[0])
        axr1i.annotate(s=marksci,
                       xy=(markx_loc, marky_loc),
                       ha='center',
                       va='center',
                       annotation_clip=False)

        ax_all.append(axr1i)

    cb = fig.colorbar(images[-1],
                      ax=ax_all,
                      orientation='horizontal',
                      fraction=0.1,
                      shrink=0.2,
                      pad=0.03)
    cb.ax.set_xlabel(r'$\omega$')
    cb.ax.xaxis.set_label_coords(-0.12, 1.5)

    if mode == 'save':
        plt.savefig(oimage_file)
    elif mode == 'show':
        plt.show()

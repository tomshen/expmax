import time
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pylab import savefig

DATA_DIRECTORY = 'data'
DATA_SOURCE = os.path.join(DATA_DIRECTORY, 'source')
RESULTS_DIRECTORY = 'results'

# list of tuples (x,y) to two lists [[x's], [y's]]
def rearrange_data(data):
    return map(list, zip(*data))

def plot_points(data):
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Data plotted on map of the world')
    im = plt.imread('world.png')
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.imshow(im, extent=[-180, 180, -90, 90])
    plt.plot(data[1], data[0], 'bo')
    plt.show()

def plot_data_seeds(data, seeds, title='Data plotted on map of the world'):
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(title)
    im = plt.imread('world.png')
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.imshow(im, extent=[-180, 180, -90, 90])
    plt.plot(data[1], data[0], 'bo')
    plt.plot(seeds[1], seeds[0], 'gs')
    plt.show()

def plot_compare(title, data1, data2):
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(title)
    im = plt.imread('world.png')
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.imshow(im, extent=[-180, 180, -90, 90])
    plt.scatter(data1[1], data1[0], c='b', marker='o')
    plt.scatter(data2[1], data2[0], c='g', marker='s')
    plt.show()

def plot_data_model(title, data, model_means, model_covs, show=True, filepath=''):
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(title)
    im = plt.imread('world.png')
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])
    plt.imshow(im, extent=[-180, 180, -90, 90])
    plt.scatter(data[1], data[0], c='b', marker='o')
    centers = rearrange_data(model_means)
    for i in xrange(len(model_means)):
        points = list(np.random.multivariate_normal(mean=model_means[i], cov=model_covs[i], size=10000))
        mean = model_means[i]
        mean.reverse()
        cov = [[y, x] for [x, y] in model_covs[i]]
        cov.reverse()
        plot_cov_ellipse(cov, mean, nstd=2, alpha=0.5, color='green')
    plt.scatter(centers[1], centers[0], c='y', marker='s')
    if filepath:
        savefig(filepath, bbox_inches=0)
        plt.clf()
    if show:
        plt.show()

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

def export_array_file(filename, data):
    np.savetxt(filename, data)
    return filename

def import_array_file(filename):
    return np.genfromtxt(filename)

class Timer:    
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

def timed(fn):
    def wrapped(*args, **kwargs):
        res = None
        with Timer() as t:
            res = fn(*args, **kwargs)
        print fn.__name__ + ' took %.03f seconds.' % t.interval
        return res
    return wrapped
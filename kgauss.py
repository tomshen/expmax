"""
A program to generate points from k gaussian distributions.
Author: Tom Shen
Date: 2013/01/29

Note:
Could also use numpy.random.multivariate_normal(mean, cov)
"""

from random import gauss, random
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import normaltest
from util import importFile, exportFile

"""
My implementation of a Gaussian distribution generator, based on the 
Box-Muller transform.

returns an array of n random values
"""
def ngauss(mu, sigma, n=1):
    data = np.empty(n)
    i = 0
    while i < n:
        u = random()
        v = random()
        data[i] = mu + sigma * np.sqrt(-2 * np.log(u)) * np.cos(2 * pi * v)
        i += 1
        if i < n:
            data[i] = mu + sigma * np.sqrt(-2 * np.log(u)) * np.sin(2 * pi * v)
            i += 1
    return data

"""
k distributions
n points of data each
dim dimensional points
lower <= mu < upper
scale * lower <= sigma < scale * upper

returns a list of point tuples
"""
def kgauss(k, n, dim=2, lower=-90, upper=90, sigma=3):
    data = np.empty((dim, k * n))
    index = 0
    for i in xrange(k):
        mu = random() * (upper - lower) + lower
        print mu
        for j in xrange(n):
            for d in xrange(dim):
                data[d][index] = ngauss(mu, sigma)
            index += 1
    return data



def scatterPlot(data):
    plt.xlabel('x-coordinates')
    plt.ylabel('y-coordinates')
    plt.title('Toy Data for 2 gaussian distributions')
    plt.plot(data[0], data[1], 'ro')
    plt.show()

def pdf(data, bins, name):
    plt.xlabel(name)
    plt.ylabel('probability')
    plt.title('Probabilty Density Function')
    plt.hist(data, bins, normed=1)
    plt.show()

def cdf(data, name):
    plt.xlabel(name)
    plt.ylabel('cumulative probability')
    plt.title('Cumulative Probability Function')
    sorted_data = np.sort(data)
    plot_data = np.empty((2, sorted_data.size))
    num_entries = 0
    for i in xrange(sorted_data.size):
        num_entries += 1
        plot_data[0][i] = sorted_data[i]
        print plot_data[0][i]
        plot_data[1][i] = num_entries
    for i in xrange(plot_data[1].size):
        plot_data[1][i] /= num_entries
    plt.plot(plot_data[0], plot_data[1])
    plt.show()

def main():
    filename = 'temp.txt'

    kg = kgauss(2, 100, lower=-90, upper=90)
    exportFile(filename, kg)
    kg = importFile(filename)

    #print normaltest(kg[0])
    scatterPlot(kg)
    # pdf(kg[0], 40, 'x-coordinates')
    # cdf(kg[0], 'x-coordinates')
    
if __name__ == "__main__":
    main()
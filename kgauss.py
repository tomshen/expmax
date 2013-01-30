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

#plt.hist(ngauss(100, 10, 100000), bins = 50)
#plt.show()

"""
k distributions
n points of data each
dim dimensional points
lower <= mu < upper
scale * lower <= sigma < scale * upper

returns a list of point tuples
"""
def kgauss(k, n, dim=2, lower=0, upper=100, scale=0.1):
    data = np.empty((dim, k * n))
    index = 0
    for i in xrange(k):
        mu = random() * (upper - lower) + lower
        sigma = random() * scale * (upper - lower) + scale * lower
        for j in xrange(n):
            for d in xrange(dim):
                data[d][index] = ngauss(mu, sigma)
            index += 1
    return data

def exportFile(filename, data):
    np.savetxt(filename, data)
    return filename

def importFile(filename):
    return np.genfromtxt(filename)

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
    plt.title('Cumulative Distribution Function')
    # need to fill in code here
    plt.show()

def main():
    filename = 'temp.txt'

    kg = kgauss(2, 100, lower=-100, upper=100)
    exportFile(filename, kg)
    kg = importFile(filename)

    print normaltest(kg[0])
    scatterPlot(kg)
    # pdf(kg[0], 40, 'x-coordinates')
    cdf(kg[0], 100, 'x-coordinates')
    
if __name__ == "__main__":
    main()
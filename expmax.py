"""
An implementation of the exp-max algorithm.
Author: Tom Shen
Date: 2013/02/01
"""
from math import fabs
from random import randint
from scipy.stats import norm
import numpy as np
from kgauss import kgauss
from util import Timer, importFile, exportFile

def multi_expectation_maximization(data, dim=2, k=2, sigma=3):
    if dim == 1:
        return expectation_maximization(data, k, sigma)
    h = []
    for d in xrange(dim):
        h.append(expectation_maximization(data[d], k, sigma))
    return h

def expectation_maximization(data, k, sigma):
    h = initial_hypothesis(data, k)
    h_old = None
    while compare_hypothesis(h, h_old):
        expected_values = calculate_expectation(k, data, h, sigma)
        h_old = h
        h = calculate_hypothesis(k, data, expected_values)
    return h

def initial_hypothesis(data, k):
    h = []
    minval = np.min(data)
    maxval = np.max(data)
    interval = (maxval - minval) / k
    h.append(minval)
    for i in xrange(1, k-1):
        h.append(minval + i * interval)
    h.append(maxval)
    return h
    # return [randint(-90, 0), randint(0, 90)]

def compare_hypothesis(h, h_old, threshold=0.01):
    if not h_old:
        return True
    diff = 0
    for i in xrange(len(h)):
        diff += h[i] - h_old[i]
    return fabs(diff) > 0.01

def expected_value_point(point, mu, h, sigma):
    exp_num = prob_point_gauss(point, mu, sigma)
    exp_denom = 0
    for mu_i in h:
        exp_denom += prob_point_gauss(point, mu_i, sigma)
    return (exp_num / exp_denom)

def prob_point_gauss(point, mu, sigma):
    gauss_dist = norm(mu, sigma)
    return gauss_dist.pdf(point)

def calculate_expectation(k, data, h, sigma):
    exp_val = np.empty((k, data.size))
    for i in xrange(data.size):
        for j in xrange(k):
            exp_val[j][i] = expected_value_point(data[i], h[j], h, sigma)
    return exp_val

def calculate_hypothesis(k, data, expected_values):
    h = []
    for j in xrange(k):
        mu_num = 0
        mu_denom = 0
        for i in xrange(data.size):
            mu_num += expected_values[j][i] * data[i]
            mu_denom += expected_values[j][i]
        h.append(mu_num / mu_denom)
    return h

def main():
    #data = kgauss(2, 100, dim=1, lower=-90, upper=90, sigma=3)
    #exportFile('temp.txt', data)
    data = importFile('temp.txt')
    with Timer() as t:
        for i in xrange(10):
            print multi_expectation_maximization(data, dim=1, k=2, sigma=3)
    print('Request took %.03f sec.' % t.interval)

if __name__ == "__main__":
    main()
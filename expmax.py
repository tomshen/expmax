"""
An implementation of the exp-max algorithm.
Author: Tom Shen
Date: 2013/02/01
"""
from math import fabs
from scipy.stats import norm
import numpy as np
from kgauss import importFile

def multi_expectation_maximization(data, dim=2, k=2):
    h = []
    for d in xrange(dim):
        h.append(expectation_maximization(data[d], k, sigma=5))
    return h

def expectation_maximization(data, k=2, sigma=5):
    num_points = data.size
    h = [0 for i in xrange(k)]
    h_old = None
    while fabs(compare_hypothesis(h, h_old)) > 0.01:
        h_new = [0 for j in xrange(k)]
        expected_values = np.empty((k, num_points))
        for i in xrange(num_points):
            for j in xrange(k):
                point = data[i]
                expected_values[j][i] = expected_value_point(point, h[j], h, sigma)
        for j in xrange(k):
            mu_num = 0
            mu_denom = 0
            for i in xrange(num_points):
                mu_num += expected_values[j][i] * data[i]
                mu_denom += expected_values[j][i]
            h_new[j] = mu_num / mu_denom
        h_old = h
        h = h_new
    return tuple(h)

def compare_hypothesis(h, h_old):
    if not h_old:
        return 100
    diff = 0
    for i in xrange(len(h)):
        diff += h[i] - h_old[i]
    return diff

def expected_value_point(point, mu, h, sigma):
    exp_num = prob_point_gauss(point, mu, sigma)
    exp_denom = 0
    for mu_i in h:
        exp_denom += prob_point_gauss(point, mu_i, sigma)
    return (exp_num / exp_denom)

def prob_point_gauss(point, mu, sigma):
    gauss_dist = norm(mu, sigma)
    return gauss_dist.pdf(point)

data = importFile('temp.txt')
print multi_expectation_maximization(data)
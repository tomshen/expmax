"""
An implementation of the exp-max algorithm.
Author: Tom Shen
Date: 2013/02/01
"""
from math import fabs
from scipy.stats import norm
import numpy as np
from kgauss import importFile

def expectation_maximization(data, k=2, dim=2, num_points=200):
    h = tuple([[0 for i in xrange(k)]  for i in xrange(dim)])
    sigma = 5
    h_old = tuple([tuple([90 for i in xrange(k)]) for i in xrange(dim)])
    while fabs(compare_hypothesis(h, h_old)) > 0.01:
        h_old = h
        h_new = [[0 for i in xrange(k)] for i in xrange(dim)]
        expected_values = np.empty((dim, k, num_points))
        for d in xrange(dim):
            for i in xrange(num_points):
                for j in xrange(k):
                    point = data[d][i]
                    expected_values[d][j][i] = expected_value_point(point, h[d][j], h[d], sigma)
            for j in xrange(k):
                mu_num = 0
                mu_denom = 0
                for l in xrange(num_points):
                    mu_num += expected_values[d][j][l] * data[d][l]
                    mu_denom += expected_values[d][j][l]
                h_new[d][j] = mu_num / mu_denom
                print j, h_new[d][j]
    return tuple([tuple(h_new[d]) for d in xrange(dim)])

def compare_hypothesis(h, h_old):
    assert len(h) == len(h_old)
    diff = 0
    for i in xrange(len(h)):
        for j in xrange(len(h[0])):
            diff += h[i][j] - h_old[i][j]
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
print expectation_maximization(data)
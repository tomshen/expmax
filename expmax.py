"""
An implementation of the exp-max algorithm.
Author: Tom Shen
Date: 2013/02/01
"""
from math import fabs
from random import randint
from scipy.stats import norm
import numpy as np
from kgauss import kgauss, kgauss_with_mus
from util import timed, importFile, exportFile

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
    """
    interval = (maxval - minval) / k
    h.append(minval)
    for i in xrange(1, k-1):
        h.append(minval + i * interval)
    h.append(maxval)
    """
    for i in xrange(k):
        h.append(randint(int(minval), int(maxval)))
    return h

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
    return 0 if exp_denom == 0 else (exp_num / exp_denom)

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
        if mu_denom == 0:
            h.append(0)
            print 'div by 0'
        else:
            h.append(mu_num / mu_denom)
    return h

@timed
def test():
    difflist = []
    k=3
    for j in xrange(10):
        data, mus = kgauss_with_mus(k, 100, dim=1, lower=-100, upper=100, sigma=3)  
        model_mus =  expectation_maximization(data[0], k, 3)
        model_mus.sort()
        mus.sort()
        difflist.append(tuple([fabs((mus[i] - model_mus[i])/mus[i]) for i in xrange(k)]))
        print difflist[j]
    diffs = [0 for i in xrange(k)]
    for i in xrange(k):
        for j in xrange(10):
            diffs[i] += difflist[j][i]
        diffs[i] /= 10
    print diffs

def main():
    #data, mus = kgauss(2, 100, dim=1, lower=-100, upper=100, sigma=3)
    #exportFile('temp.txt', data)
    #data = importFile('temp.txt')
    # test()
    data = []
    for i in kgauss(1, 10, dim=1, lower=-100, upper=100, sigma=3)[0]:
        data.append(i)
    for i in kgauss(1, 1000, dim=1, lower=-100, upper=100, sigma=3)[0]:
        data.append(i)
    em = expectation_maximization(np.array(data), 2, 1)
    while not (em[0] and em[1]):
        em = expectation_maximization(np.array(data), 2, 1)
        print 'Let\'s try again'
    print em


if __name__ == "__main__":
    main()
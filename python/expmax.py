"""
An implementation of the exp-max algorithm.
Author: Tom Shen
Date: 2013/02/01
"""
from math import fabs
from random import randint
from scipy.stats import norm
import numpy as np
from kgauss import kgauss, kgauss_with_mus, kgauss_with_mus_sigmas
from util import timed, importFile, exportFile

def multi_expectation_maximization(data, dim, k, sigmas):
    assert(len(sigmas) == k)
    if dim == 1:
        return expectation_maximization(data, k, sigmas)
    h = []
    for d in xrange(dim):
        h.append(expectation_maximization(data[d], k, sigmas[d]))
    return h

def expectation_maximization(data, k, sigmas):
    mus = initial_hypothesis_mu(data, k)
    mus_old = None
    sigmas_old = None
    while compare_hypothesis(mus, mus_old) or compare_hypothesis(sigmas, sigmas_old):
        exp_val = calculate_expectation(k, data, mus, sigmas)
        mus_old = mus
        sigmas_old = sigmas
        mus, sigmas = calculate_hypothesis(k, data, exp_val)
    return mus, sigmas

def initial_hypothesis_mu(data, k):
    h = []
    minval = np.min(data)
    maxval = np.max(data)
    interval = (maxval - minval) / k
    h.append(minval)
    for i in xrange(1, k-1):
        h.append(minval + i * interval)
    h.append(maxval)
    return h

def compare_hypothesis(curr, old, epsilon=0.01):
    if not old:
        return True
    diff = 0
    for i in xrange(len(curr)):
        diff += curr[i] - old[i]
    return fabs(diff) > 0.01

def expected_value_point(point, mu, mus, sigma, sigmas):
    exp_num = prob_point_gauss(point, mu, sigma)
    exp_denom = 0
    for i in xrange(len(mus)):
        exp_denom += prob_point_gauss(point, mus[i], sigmas[i])
    return 0 if exp_denom == 0 else (exp_num / exp_denom)

def prob_point_gauss(point, mu, sigma):
    gauss_dist = norm(mu, sigma)
    return gauss_dist.pdf(point)

def calculate_expectation(k, data, mus, sigmas):
    exp_val = np.empty((k, data.size))
    for i in xrange(data.size):
        for j in xrange(k):
            exp_val[j][i] = expected_value_point(data[i], mus[j], mus, sigmas[j], sigmas)
    return exp_val

def calculate_hypothesis(k, data, expected_values):
    mus = []
    sigmas = []
    for j in xrange(k):
        mu_num = 0
        denom = 0
        sigma_num = 0
        for i in xrange(data.size):
            mu_num += expected_values[j][i] * data[i]
            denom += expected_values[j][i]
        mus.append(mu_num / denom)
        for i in xrange(data.size):
            sigma_num += expected_values[j][i] * (data[i] - mus[j])**2
        sigmas.append((sigma_num / denom)**0.5)
    return mus, sigmas

# prints means squared error
@timed
def test():
    difflist = []
    k = 3
    trials = 10
    for j in xrange(trials):
        data, mus = kgauss_with_mus(k, 100, dim=1, lower=-100, upper=100, sigma=3)  
        model_mus =  expectation_maximization(data[0], k, 3)
        model_mus.sort()
        mus.sort()
        print model_mus, mus
        difflist.append(tuple([(mus[i] - model_mus[i])**2 for i in xrange(k)]))
        print difflist[j]
    diffs = [0 for i in xrange(k)]
    for i in xrange(k):
        for j in xrange(10):
            diffs[i] += difflist[j][i]
        diffs[i] /= 10

def test_ppg():
    point, mean, sigma = randint(-100, 100), randint(-100, 100), randint(5, 20)
    print 'copyValueFromPython = ' + str(prob_point_gauss(point, mean, sigma)) + ';'
    print 'assertEquals(copyValueFromPython,ExpMax.probPoint(' + str(point) + ', ' + str(mean) + ', ' + str(sigma) +'),1e-4);'

def test_em():
    filename = '..\\Java\\temp2.txt'
    data = kgauss(2, 100, dim=1, lower=-100, upper=100, sigma=randint(5, 15))
    exportFile(filename, data)
    # print data[0]
    params = expectation_maximization(data[0], 2, [1, 1])
    print 'pythonParams = new double[][] {',
    for i in params:
        print '{',
        for j in i:
            print str(j) + ',',
        print '},',
    print '};'

def main():
    test_em()    

if __name__ == "__main__":
    main()
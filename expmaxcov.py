"""
An implementation of the exp-max algorithm, using a covariance matrix
Author: Tom Shen
Date: 2013/02/12
"""
from math import fabs, pow, e, pi, isnan
from random import randint
from scipy.stats import norm, gaussian_kde
import numpy as np
from kgauss import kmvgauss
from util import timed, importFile, exportFile

# [[x1, y1], [x2, y2]] -> [[x1, x2], [y1, y2]]
def reformat_data(data):
    return np.array([[z[i] for z in data] for i in xrange(data.shape[1])])

def multivariate_expectation_maximization(data, k, covs):
    fdata = reformat_data(data)
    means = initial_means(fdata, k)
    covs = covs# initial_covs(fdata, k)
    means_old = None
    covs_old = None
    while compare_2d(means, means_old) or compare_3d(covs, covs_old):
        exp_val = calculate_expectation(k, data, means, covs)
        means_old, covs_old = means, covs
        means, covs = calculate_hypothesis(k, fdata, exp_val)
        print means, covs
    return means, covs

# takes in fdata
def initial_means(data, k):
    dim = data.shape[0]
    means = [np.empty(dim) for i in xrange(k)]
    for d in xrange(dim):
        minval = np.min(data[d])
        maxval = np.max(data[d])
        interval = (maxval - minval) / k
        means[0][d] = minval
        for i in xrange(1, k-1):
            means[i][d] = minval + i * interval
        means[k-1][d] = maxval
        #"""
        for i in xrange(k):
            means[i][d] = randint(int(minval), int(maxval))
        #"""
    return means

# takes in fdata
def initial_covs(data, k):
    dim = data.shape[0]
    covs = [np.empty((dim, dim)) for i in xrange(k)]
    for i in xrange(len(covs)):
        for j in xrange(len(covs[0])):
            for k in xrange(len(covs[0][0])):
                covs[i][j][k] = randint(-10, 10)
    return covs

def compare_2d(curr, old, epsilon=0.01):
    if not old:
        return True
    diff = 0
    for i in xrange(len(curr)):
        for d in xrange(len(curr[0])):
            diff += curr[i][d] - old[i][d]
    return fabs(diff) > 0.01

def compare_3d(curr, old, epsilon=0.01):
    if not old:
        return True
    diff = 0
    for i in xrange(len(curr)):
        for j in xrange(len(curr[0])):
            for k in xrange(len(curr[0][0])):
                diff += curr[i][j][k] - old[i][j][k]
    return fabs(diff) > 0.01

def expected_value_point(point, mean, means, cov, covs):
    exp_num = prob_point_gauss(point, mean, cov)
    exp_denom = 0
    for i in xrange(len(means)):
        exp_denom += prob_point_gauss(point, means[i], covs[i])
    return 0 if exp_denom == 0 else (exp_num / exp_denom)

def prob_point_gauss(point, mean, cov):
    kernel = gaussian_kde(np.random.multivariate_normal(mean, cov))
    prob = reduce(lambda x, y: x + y, kernel.evaluate(point)) / len(point)
    return prob
    """
    dim = len(point)
    cov = np.matrix(cov)
    if dim != len(mean) or (dim, dim) != cov.shape:
        raise NameError('Dimensions don\'t match')
    det = np.linalg.det(cov)
    if det == 0:
        raise NameError('Covariance matrix can\'t be singular')
    norm_const = 1.0 / (pow((2*pi), float(dim)/2) * det**0.5)
    point_mean = np.matrix(point - mean)
    inv = cov.I
    return pow(e, -0.5 * (point_mean * inv * point_mean.T).item()) * norm_const
    """

# takes in data
def calculate_expectation(k, data, means, covs):
    exp_val = np.empty((k, data.shape[0]))
    for i in xrange(data.shape[0]):
        for j in xrange(k):
            evp = expected_value_point(data[i], means[j], means, covs[j], covs)
            exp_val[j][i] = evp
    return exp_val

# takes in fdata
def calculate_hypothesis(k, data, exp_val):
    dim = data.shape[0]
    n = data.shape[1]
    means = []
    covs = []
    for j in xrange(k):
        denom = 0
        mean = []
        cov = np.identity(dim)
        for i in xrange(n):
            denom += exp_val[j][i]
        for d in xrange(dim):
            mean_num = 0
            for i in xrange(n):
                mean_num += exp_val[j][i] * data[d][i]
            mean.append(mean_num / denom)
        means.append(mean)

        cv = [0 for i in xrange(dim)]
        for d in xrange(dim):
            ws = 0
            for i in xrange(n):
                ws += exp_val[j][i]**2
                cv[d] += (data[d][i] - mean[d]) * exp_val[j][i]**0.5
        cv = np.matrix(cv)
        cov = cv.T * cv
        cov = np.array(cov)
        for a in xrange(cov.shape[0]):
            for b in xrange(cov.shape[1]):
                cov[a][b] *= denom / (denom**2 - ws)
        covs.append(np.array(cov))
    return means, covs

@timed
def test():
    cov = [[10, 0], [0, 10]]
    data1, means1 = kmvgauss(1, 100, cov, 2)
    data2, means2 = kmvgauss(1, 100, cov, 2)
    data = np.concatenate((data1, data2))
    fdata1, fdata2 = reformat_data(data1), reformat_data(data2)

    cov1, cov2 = np.cov(fdata1), np.cov(fdata2)
    model_means, model_covs = multivariate_expectation_maximization(data, 2, np.array([cov1, cov2]))

    print 'Actual'
    print means1, means2
    print cov1, cov2
    print 'Model'
    print model_means
    print model_covs

def main():
    test()
    
if __name__ == "__main__":
    main()
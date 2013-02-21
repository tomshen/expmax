"""
An implementation of the exp-max algorithm, using a covariance matrix
Author: Tom Shen
Date: 2013/02/12
"""
from math import fabs, pow, e, pi, isnan, sqrt
from random import randint
from scipy.stats import norm, gaussian_kde
import numpy as np
from kgauss import kmvgauss
from util import timed, importFile, exportFile

# [[x1, y1], [x2, y2]] -> [[x1, x2], [y1, y2]]
def reformat_data(data):
    return np.array([[z[i] for z in data] for i in xrange(data.shape[1])])

def multivariate_expectation_maximization(data, k, covs, means):
    fdata = reformat_data(data)
    means = means # initial_means(fdata, k)
    covs = covs # initial_covs(fdata, k)
    means_old = None
    covs_old = None
    i = 0
    while compare_2d(means, means_old) or compare_3d(covs, covs_old):
        
        print '--- ITERATION', i, '---'
        """
        print 'means', means
        print 'covs', covs[0], covs[1]
        """
        i += 1
        exp_val = calculate_expectation(k, data, means, covs)
        print exp_val
        means_old, covs_old = means, covs
        means, covs = calculate_hypothesis(k, fdata, exp_val)
        for mean in means:
            for m in mean:
                if isnan(m):
                    print 'Error: encountered NaN, exiting'
                    exit()
    return means, covs

# takes in fdata
def initial_means(data, k):
    dim = data.shape[0]
    means = [np.empty(dim) for i in xrange(k)]
    """
    for d in xrange(dim):
        minval = np.min(data[d])
        maxval = np.max(data[d])
        interval = (maxval - minval) / k
        means[0][d] = minval
        for i in xrange(1, k-1):
            means[i][d] = minval + i * interval
        means[k-1][d] = maxval
    """
    for d in xrange(dim):
        minval = np.min(data[d])
        maxval = np.max(data[d])
        for i in xrange(k):
            means[i][d] = randint(int(minval), int(maxval))
    # """
    return means

# takes in fdata
def initial_covs(data, k):
    dim = data.shape[0]
    return [np.identity(dim) for i in xrange(k)]
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
            diff += fabs(curr[i][d] - old[i][d])**2
    return sqrt(diff) > epsilon

def compare_3d(curr, old, epsilon=0.01):
    if old == None:
        return True
    diff = 0
    for i in xrange(len(curr)):
        for j in xrange(len(curr[0])):
            for k in xrange(len(curr[0][0])):
                diff += (curr[i][j][k] - old[i][j][k])**2
    return sqrt(diff) > epsilon

# i is which of the k distributions you are considering
def expected_value_point(point, i, means, covs):
    exp_num = prob_point_gauss(point, means[i], covs[i])
    exp_denom = 0
    for j in xrange(len(means)):
        exp_denom += prob_point_gauss(point, means[j], covs[j])
    if exp_denom < exp_num:
        print 'Error: part greater than sum of parts'
        print means, covs
        print [prob_point_gauss(point, kernel) for j in xrange(len(means))], exp_num
        exit()
    return 0 if exp_denom == 0 else (exp_num / exp_denom)

def prob_point_gauss(point, mean, cov):
    dim = len(point)
    cov = np.matrix(cov)
    if dim != len(mean) or (dim, dim) != cov.shape:
        raise NameError('Dimensions don\'t match')
    det = np.linalg.det(cov)
    if det == 0:
        raise NameError('Covariance matrix can\'t be singular')
    try:
        norm_const = 1.0 / (pow((2*pi), float(dim)/2) * pow(det, 1.0/2))
    except:
        print det, cov
        exit()
    point_mean = np.matrix(point - mean)
    inv = cov.I
    return norm_const * pow(e, -0.5 * (point_mean * inv * point_mean.T).item()) 

# takes in data
def calculate_expectation(k, data, means, covs):
    exp_val = np.empty((k, data.shape[0]))
    for i in xrange(data.shape[0]):
        for j in xrange(k):
            evp = expected_value_point(data[i], j, means, covs)
            exp_val[j][i] = evp
    return exp_val

def calc_cov(data, dim, mean):
    cov = np.empty((dim, dim))
    sum_ev_squared = 0
    for l in xrange(dim):
        for k in xrange(dim):
            for i in xrange(data.shape[1]):
                cov[l][k] += (data[l][i] - mean[l]) * (data[k][i] - mean[k])
            cov[l][k] /= data.shape[1] - 1
    return cov

# takes in fdata
def calculate_hypothesis(k, data, exp_val):
    dim = data.shape[0]
    n = data.shape[1]
    means = []
    covs = []
    for j in xrange(k):
        denom = 0
        mean = []
        for i in xrange(n):
            denom += exp_val[j][i]

        for d in xrange(dim):
            mean_num = 0
            for i in xrange(n):
                mean_num += exp_val[j][i] * data[d][i]
            mean.append(mean_num / denom)
        means.append(mean)
        cov = np.empty((dim, dim))
        for l in xrange(dim):
            for k in xrange(dim):
                for i in xrange(n):
                    cov[l][k] += exp_val[j][i] * (data[l][i] - mean[l]) * (data[k][i] - mean[k])
                cov[l][k] /= denom * (n - 1) / n
        covs.append(cov)
    return means, covs

@timed
def test():
    cov = [[10, 0], [0, 10]]
    data1, means1 = kmvgauss(1, 100, cov, 2)
    data2, means2 = kmvgauss(1, 100, cov, 2)
    data = np.concatenate((data1, data2))
    fdata1, fdata2 = reformat_data(data1), reformat_data(data2)

    cov1, cov2 = np.cov(fdata1), np.cov(fdata2)

    s = 'new double[][] {'
    for m in [means1, means2]:
        s += '{'
        for i in m:
            for j in i:
                s += str(round(j, -1))
                s += ','
        s += '},'
    s +='};'
    print s
    exportFile('..\\Java\\temp.m', reformat_data(data))
    print 'Actual'
    print means1, means2
    print cov1, cov2

    init_means = [[round(i, -1) for i in mean[0]] for mean in [means1, means2]]
    print init_means
    model_means, model_covs = multivariate_expectation_maximization(data, 2, np.array([cov1, cov2]), [init_means[0], init_means[1]])

    print 'Actual'
    print means1, means2
    print cov1, cov2
    print 'Model'
    print model_means
    print model_covs

def test_ppg():
    def list_to_java_array(l):
        s = '{'
        for i in l:
            s += str(i) + ','
        s += '}'
        return s
    point = [randint(-100, 100), randint(-100, 100)]
    mean = [randint(-100, 100), randint(-100, 100)]
    cov = [[randint(5, 15), 0], [0, randint(5, 15)]]
    
    jpoint = 'new double[] ' + list_to_java_array(point)
    jmean = 'new double[] ' + list_to_java_array(mean)
    jcov = 'new Array2DRowRealMatrix(new double[][] ' + list_to_java_array([list_to_java_array(i) for i in cov])
    jcov += ')'
    print 'copyValueFromPython = ' + str(prob_point_gauss(np.array(point), np.array(mean), np.array(cov))) + ';'
    print 'assertEquals(copyValueFromPython,MultiExpMax.probPoint(' + jpoint,
    print ', ' + jmean + ', ' + jcov +'),1e-330);'

def main():
    test()
    
if __name__ == "__main__":
    main()
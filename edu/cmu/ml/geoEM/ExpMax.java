package edu.cmu.ml.geoEM;

import java.util.*;

import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;

public class ExpMax
{
    private double[][] data;
    private int dim;   
    private int minClusters;
    private int maxClusters;
    private int numPoints;
    /** how close parameters from successive iterations need to be for the
     *  algorithm to terminate */
    private double epsilon = 0.001;
    /** the minimum number of points for a cluster be considered one */
    private int minPointsCluster = 2;
    /** the value to initialize the covariance matrices' diagonals to */
    private double covInit = 10.0;
    /** how close means for two clusters need to be to be considered duplicate
     *  clusters */
    private double meanEpsilon = 5.0;
    /** the distributions for each cluster */
    private ArrayList<MultivariateNormalDistribution> dists;
    /** a constant used to test for distribution uniformity */
    private double uniformityConstant = 1.5;
    
    public ArrayList<Double[]> means;
    public ArrayList<RealMatrix> covs;
    
    /**
     * Initializes parameters for a run of the algorithm.
     * @param  data the coordinates with rows for the dimensions of the data,
     *              and columns for the actual data points
     * @param  kmin the minimum number of clusters allowed
     * @param  kmax the maximum number of clusters allowed
     */
    public ExpMax(double[][] data, int kmin, int kmax) {
        this.data = data;
        dim = data.length;
        minClusters = kmin;
        maxClusters = kmax;
        numPoints = data[0].length;
        minPointsCluster = numPoints / maxClusters;
        initializeMeans();
        initializeCovs();
    }
    
    /**
     * Runs the expectation-maximization algorithm for a mixture of gaussians.
     * Each iteration, the probability that each point belongs to each cluster
     * is calculated, which will be used as weights. Then, the parameters 
     * (including number of clusters) is recalculated based on the weights of 
     * each point. This runs iteratively until means or the covariances of the 
     * clusters are all within {@link #epsilon} of each other. Finally, any
     * duplicate clusters (those with means within {@link #meanEpsilon} of each
     * other) are removed, and the model parameters are printed.
     * @see #calculateExpectation() calculateExpectation
     * @see #calculateHypothesis(ArrayList) calculateHypothesis
     * @see #removeDuplicateClusters() removeDuplicateClusters
     * @see <a href="https://en.wikipedia.org/wiki/Expectation-maximization_algorithm">Expectation-maximization algorithm</a>
     * @see "Machine Learning, by Tom Mitchell, pp. 191-3"
     */
    public void calculateParameters() {
        ArrayList<Double[]> oldMeans;
        ArrayList<RealMatrix> oldCovs;
        int iterations = 0;
        long startTime = System.currentTimeMillis();
        do {
            oldMeans = Util.deepcopyArray(means);
            oldCovs = Util.deepcopyMatrix(covs);
            calculateHypothesis(calculateExpectation());
            iterations++;
        } while(arrayListDifferent(means, oldMeans) 
             || matrixListDifferent(covs, oldCovs));
        removeDuplicateClusters();
        System.out.println("EM with " + means.size() 
                           + " clusters complete! Took " 
                           + iterations + " iterations and "
                           + Double.toString((System.currentTimeMillis() 
                                             - startTime) / 1000.0)
                           + " seconds");
        printParameters();
    }
    
    /**
     * Prints the model means and covariances of the clusters.
     */
    public void printParameters() {
        System.out.println("Means:");
        for(Double[] d : means)
            System.out.println(Arrays.toString(Util.doubleValues(d)));
        System.out.println("Covariances:\n" + Util.matricesToString(covs));
    }
    
    /**
     * Initializes {@link #minClusters} means through K-means++
     * @see <a href="http://en.wikipedia.org/wiki/K-means%2B%2B">K-means++</a>
     */
    private void initializeMeans() {
        means = new ArrayList<Double[]>();
        for(int i = 0; i < minClusters; i++)
            means.add(new Double[dim]);
        int currDist = 0;
        double[] weights = new double[numPoints];
        int meanIndex = (int)(Math.random() * numPoints);
        for(int i = 0; i < dim; i++)
            means.get(currDist)[i] = data[i][meanIndex];
        currDist++;
        while(currDist < means.size()) {
            double sumDS = 0;
            for(int i = 0; i < numPoints; i++) {
                double[] currPoint = new double[dim];
                for(int j = 0; j < dim; j++)
                    currPoint[j] = data[j][i];
                weights[i] = Util.distanceSquared(
                        currPoint, means.get(currDist - 1));
                sumDS += weights[i];
            }
            for(int i = 0; i < numPoints; i++)
                weights[i] /= sumDS;

            meanIndex = -1;
            while(meanIndex == -1) {
                int pointIndex = (int)(Math.random() * numPoints);
                if(Math.random() < weights[pointIndex])
                    meanIndex = pointIndex;
            }
            for(int i = 0; i < dim; i++)
                means.get(currDist)[i] = data[i][meanIndex];
            currDist++;
        }
    }

    /**
     * Initializes {@link #minClusters} covariance matrices. Each matrix is a
     * {@link #dim} by dim and diagonal, with {@link #covInit} as its entries.
     */
    private void initializeCovs() {
        covs = new ArrayList<RealMatrix>();
        for(int i = 0; i < minClusters; i++) {
            covs.add(new Array2DRowRealMatrix(new double[dim][dim]));
            for(int r = 0; r < dim; r++)
                for(int c = 0; c < dim; c++)
                    if(r == c) covs.get(i).setEntry(r, c, covInit);
        }
    }
    
    /**
     * Removes any clusters that have means within {@link #meanEpsilon} of 
     * each other.
     */
    private void removeDuplicateClusters() {
        if(means.size() > minClusters) {
            ArrayList<Double[]> newMeans = new ArrayList<Double[]>();
            ArrayList<RealMatrix> newCovs = new ArrayList<RealMatrix>();
            for(Double[] oldMean : means) {
                boolean duplicateMeans = false;
                for(Double[] newMean : newMeans) {
                    if(Util.distance(oldMean, newMean) < meanEpsilon) {
                        System.out.println(Util.distance(oldMean, newMean));
                        duplicateMeans = true;
                        break;
                    }
                }
                if(!duplicateMeans) {
                    int i = means.indexOf(oldMean);
                    newMeans.add(oldMean);
                    newCovs.add(covs.get(i));
                }
                    
            }
            means = newMeans;
            covs = newCovs;
        }
    }

    /**
     * @return false if each array in one list is within {@link #epsilon} of 
     * the corresponding array in the other list, or true otherwise
     */
    private boolean arrayListDifferent(ArrayList<Double[]> curr, 
            ArrayList<Double[]> old) {
        for(int i = 0; i < curr.size(); i++)
            if(Util.distance(curr.get(i), old.get(i)) > epsilon)
                return true;
        return false;
    }
    /**
     * @return false if each matrix in one list is within {@link #epsilon} of 
     * the corresponding matrix in the other list, or true otherwise
     */
    private boolean matrixListDifferent(ArrayList<RealMatrix> curr, 
            ArrayList<RealMatrix> old) {
        for(int i = 0; i < curr.size(); i++)
            if(Util.distance(curr.get(i).getData(), old.get(i).getData()) > epsilon)
                return true;
        return false;
    }
    /**
     * @param  i index of the point in {@link #data}
     * @return returns an array of double primitives for the point from the data
     */
    private double[] getPoint(int i) {
        double[] point = new double[dim];
            for(int d = 0; d < dim; d++)
                point[d] = data[d][i];
        return point;
    }
    /**
     * @param  i index of the point in {@link #data}
     * @return returns an array of double objects for the point from the data
     */
    private Double[] getPointObj(int i) {
        Double[] point = new Double[dim];
            for(int d = 0; d < dim; d++)
                point[d] = data[d][i];
        return point;
    }
    
    /**
     * Creates distributions representing the clusters based on the model
     * {@link #means} and {@link #covs covariances}, stored in {@link #dists}.
     */
    private void createDists() {
        dists = new ArrayList<MultivariateNormalDistribution>();
        for(int i = 0; i < means.size(); i++)
            dists.add(new MultivariateNormalDistribution(
                    Util.doubleValues(means.get(i)), 
                    covs.get(i).getData()));
    }
    
    /**
     * Checks if the distribution is uniform by seeing if the maximum value in
     * the distribution is less than {@link #uniformityConstant} times the
     * minimum value.
     * @param dist the distribution to check for uniformity
     * @return if the distribution is uniform
     */
    private boolean distributionUniform(double[] dist) {
        double min = dist[0];
        double max = 0;
        for(double d : dist) {
            if(d < min)
                min = d;
            else if(d > max)
                max = d;
        }
        return max < uniformityConstant * min;
    }
    
    /**
     * Finds the probability each point belongs to each cluster. First, a list 
     * of the distributions are generated based on the current model means and
     * covariances. Then, for each point, the probability that it belongs to 
     * each cluster is calculated. If a point is equally likely to belong to 
     * each cluster, then a cluster is added, with that point as its mean. 
     * After the probabilities for all points has been calculated, if any 
     * clusters have less than {@link #minPointsCluster} points, they are 
     * removed. The number of clusters will not go above {@link #maxClusters} or
     * below {@link #minClusters}.
     * 
     * @return a list of arrays for each cluster, where the entries correspond 
     * to the probabilities of points belonging to that cluster
     * @see #createDists() createDists
     * @see #expectedValuePoint(double[], int) expectedValuePoint
     * @see #distributionUniform(double[]) distributionUniform
     */
    private ArrayList<Double[]> calculateExpectation() {
        ArrayList<Double[]> expectedValues = new ArrayList<Double[]>();
        createDists();
        for(int i = 0; i < means.size(); i++) {
            expectedValues.add(new Double[numPoints]);
            for(int j = 0; j < numPoints; j++) 
                expectedValues.get(i)[j] = expectedValuePoint(getPoint(j), i);
        }
        int oldSize = expectedValues.size();
        if(expectedValues.size() < maxClusters) {
            for(int i = 0 ; i < numPoints; i++) {
                if(expectedValues.size() >= maxClusters)
                    break;
                double[] dist = new double[expectedValues.size()];
                for(int j = 0; j < expectedValues.size(); j++) {
                    dist[j] = expectedValues.get(j)[i].doubleValue();
                }
                if(distributionUniform(dist)) {
                    System.err.println("ADDING DIST");
                    means.add(getPointObj(i));
                    covs.add(new Array2DRowRealMatrix(new double[dim][dim]));
                    for(int r = 0; r < dim; r++)
                        for(int c = 0; c < dim; c++)
                            if(r == c)
                                covs.get(covs.size() - 1).setEntry(r, c, covInit);
                    dists.add(new MultivariateNormalDistribution(
                            Util.doubleValues(means.get(means.size() - 1)), 
                            covs.get(covs.size() - 1).getData()));
                    expectedValues.add(new Double[numPoints]);
                    for(int c = 0; c < means.size(); c++)
                        for(int k = 0; k < numPoints; k++)
                            expectedValues.get(c)[k] = 
                                expectedValuePoint(getPoint(k), c);
                    for(Double[] da : expectedValues)
                        da[i] = 0.0;
                    expectedValues.get(expectedValues.size() - 1)[i] = 1.0;
                }
            }
        }
        if(expectedValues.size() > oldSize) {
            int currDist = 0;
            while(currDist < expectedValues.size()) {
                Double[] probs = expectedValues.get(currDist);
                int pointsCluster = 0;
                for(int i = 0; i < numPoints; i++) {
                    if(probs[i] > epsilon)
                        pointsCluster++;
                }
                if(pointsCluster < minPointsCluster) {
                    if(expectedValues.size() <= minClusters)
                        break;
                    System.err.println("REMOVING DIST");
                    means.remove(currDist);
                    covs.remove(currDist);
                    dists.remove(currDist);
                    expectedValues.remove(currDist);
                }
                else currDist++;
            }
        }
        return expectedValues;
    }

    /**
     * @param point
     * @param currDist the distribution that the probability is being calculated
     * for
     * @return the probability that the point belongs to this distribution over
     * all the other distributions
     * @see #calculateExpectation() calculateExpectation
     */
    private double expectedValuePoint(double[] point, int currDist) {
        double probCurrDist = probPoint(point, currDist);
        double probAllDist = 0;
        for(int i = 0; i < means.size(); i++)
            probAllDist += probPoint(point, i);
        if(probAllDist == 0)
            return 0;
        return probCurrDist / probAllDist;
    }
    
    /**
     * @param point
     * @param currDist the distribution the probability is being calculated for 
     * @return the probability the point belongs to this distribution
     * @see #dists dists
     * @see #expectedValuePoint(double[], int) expectedValuePoint
     */
    private double probPoint(double[] point, int currDist) {
        return dists.get(currDist).density(point);
    }
    
    /**
     * Calculates the new parameters for the clusters based on the expected
     * values. The means are calculated based on a weighted average of the 
     * points, where each point is weighted according to the probability it
     * belongs to that cluster as opposed to all the other clusters. The 
     * covariances are similarly calculated with the the distances of each point
     * from the mean weighted with the probabilities of each point belonging to
     * that cluster.
     * @param expectedValues a list of arrays for each point, where the entries 
     * correspond to the probability that the point belongs to a particular 
     * cluster
     */
    private void calculateHypothesis(ArrayList<Double[]> expectedValues) {
        for(int i = 0; i < means.size(); i++) {
            double totalExp = 0;
            means.set(i, new Double[dim]);
            for(int j = 0; j < numPoints; j++)
                totalExp += expectedValues.get(i)[j];
            covs.set(i, new Array2DRowRealMatrix(new double[dim][dim]));
            for(int d = 0; d < dim; d++) {
                means.get(i)[d] = 0.0;
                for(int j = 0; j < numPoints; j++) {
                    means.get(i)[d] += expectedValues.get(i)[j] * data[d][j];
                }
                    
                means.get(i)[d] /= totalExp;
            }
            for(int r = 0; r < dim; r++) {
                for(int c = 0; c < dim; c++) {
                    double entry = 0;
                    for(int j = 0; j < numPoints; j++) {
                        entry += (expectedValues.get(i)[j]
                                * (data[r][j] - means.get(i)[r])
                                * (data[c][j] - means.get(i)[c]));
                    }
                    entry = (entry / totalExp) * numPoints / (numPoints - 1);
                    covs.get(i).setEntry(r, c, entry);
                }
            }
        }
    }
}
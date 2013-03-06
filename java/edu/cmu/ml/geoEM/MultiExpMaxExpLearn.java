package edu.cmu.ml.geoEM;

import java.util.ArrayList;
import java.util.Arrays;

import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;

public class MultiExpMaxExpLearn
{
    public double[][] data; // {{x-coordinates}, {y-coordinates}}
    public int dim;
    public int minDist;
    public int maxDist;
    public int numPoints;
    public ArrayList<Double[]> means;
    public ArrayList<RealMatrix> covs;
    protected static double epsilon = 0.001;
    protected static int minPointsCluster = 2; // minimum # points in a cluster
    
    private ArrayList<MultivariateNormalDistribution> dists;

    // k is upper bound
    public MultiExpMaxExpLearn(double[][] data, int kmin, int kmax) {
        this.data = data;
        dim = data.length;
        minDist = kmin;
        maxDist = kmax;
        numPoints = data[0].length;
        minPointsCluster = numPoints / maxDist;
        initializeMeans();
        initializeCovs();
    }
    
    private double calculateDistanceSquared(double[] p1, double[] p2) {
        assert(p1.length == p2.length);
        double sumSquares = 0;
        for(int i = 0; i < p1.length; i++)
            sumSquares += Math.pow((p1[i] - p2[i]), 2.0);
        return sumSquares;
    }
    
    private double calculateDistanceSquared(double[] p1, Double[] p2) {
    	assert(p1.length == p2.length);
        double sumSquares = 0;
        for(int i = 0; i < p1.length; i++)
            sumSquares += Math.pow((p1[i] - p2[i].doubleValue()), 2.0);
        return sumSquares;
    }
    
    private double calculateDistanceSquared(Double[] p1, double[] p2) {
    	assert(p1.length == p2.length);
        double sumSquares = 0;
        for(int i = 0; i < p1.length; i++)
            sumSquares += Math.pow((p1[i].doubleValue() - p2[i]), 2.0);
        return sumSquares;
    }
    
    private double calculateDistanceSquared(Double[] p1, Double[] p2) {
    	assert(p1.length == p2.length);
        double sumSquares = 0;
        for(int i = 0; i < p1.length; i++)
            sumSquares += Math.pow((p1[i].doubleValue() - p2[i].doubleValue()), 2.0);
        return sumSquares;
    }
    
    protected void initializeMeans() {
        means = new ArrayList<Double[]>();
        for(int i = 0; i < minDist; i++)
        	means.add(new Double[dim]);
        int currDist = 0;
        double[] weights = new double[numPoints];
        int meanIndex = (int)(Math.random() * numPoints);
        for(int i = 0; i < dim; i++)
            means.get(currDist)[i] = data[i][meanIndex];
        currDist++;
        while(currDist < means.size()) {
        	/* DOESN'T WORK
        	double sumDS = 0;
            for(int i = 0; i < numPoints; i++) {
                double[] currPoint = new double[dim];
                for(int j = 0; j < dim; j++)
                    currPoint[j] = data[j][i];
                weights[i] = calculateDistanceSquared(
                        currPoint, means.get(currDist - 1));
                sumDS += weights[i];
            }
            for(int i = 0; i < numPoints; i++)
                weights[i] /= sumDS;
            double r = Math.random();
            double partialSum = 0;
            for(meanIndex = 0; meanIndex < numPoints; meanIndex++) {
            	partialSum += weights[meanIndex++];
            	if(partialSum < r)
            		break;
            }
        	weights[meanIndex] = 0;
        	for(int i = 0; i < dim; i++)
                means.get(currDist)[i] = data[i][meanIndex];
            currDist++;
            */
        	double sumDS = 0;
            for(int i = 0; i < numPoints; i++) {
                double[] currPoint = new double[dim];
                for(int j = 0; j < dim; j++)
                    currPoint[j] = data[j][i];
                weights[i] = calculateDistanceSquared(
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
    
    protected void initializeCovs() {
    	covs = new ArrayList<RealMatrix>();
        for(int i = 0; i < minDist; i++) {
            covs.add(new Array2DRowRealMatrix(new double[dim][dim]));
            for(int r = 0; r < dim; r++)
                for(int c = 0; c < dim; c++)
                    if(r == c) covs.get(i).setEntry(r, c, 10.0);
        }
    }

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
        } while(compareArrays(means, oldMeans) || compareMatrices(covs, oldCovs));
        System.out.println("EM with " + means.size() 
        				   + " clusters complete! Took " 
        				   + iterations + " iterations and "
        				   + Double.toString((System.currentTimeMillis() 
        						   			 - startTime) / 1000.0)
        				   + " seconds");
        printParameters();
    }
    
    public void printParameters() {
    	System.out.println("Means:");
    	for(Double[] d : means)
    		System.out.println(Arrays.toString(Util.doubleObjectToPrimitiveArray(d)));
        System.out.println("Covariances:\n" + Util.matricesToString(covs));
    }

    protected static boolean compare(double[][] curr, double[][] old) {
        if(old == null)
            return true;
        return calcDiff(curr, old) > epsilon;
    }

    protected static boolean compare(RealMatrix[] curr, RealMatrix[] old) {
        if(old == null)
            return true;
        for(int i = 0; i < curr.length; i++)
            if (calcDiff(curr[i].getData(), old[i].getData()) > epsilon)
                return true;
        return false;
    }
    
    protected static boolean compareArrays(ArrayList<Double[]> curr, ArrayList<Double[]> old) {
        if(old == null)
            return true;
        for(int i = 0; i < curr.size(); i++)
            if (calcDiff(curr.get(i), old.get(i)) > epsilon)
                return true;
        return false;
    }

    protected static boolean compareMatrices(ArrayList<RealMatrix> curr, 
    		ArrayList<RealMatrix> old) {
        if(old == null)
            return true;
        for(int i = 0; i < curr.size(); i++)
            if (calcDiff(curr.get(i).getData(), old.get(i).getData()) > epsilon)
                return true;
        return false;
    }

    protected static double calcDiff(double[][] curr, double[][] old) {    
        double diff = 0.0;
        for(int i = 0; i < curr.length; i++)
            diff += FastMath.pow(calcDiff(curr[i], old[i]), 2.0);
        return FastMath.pow(diff, 0.5);
    }

    protected static double calcDiff(double[] curr, double[] old) {
        double diff = 0.0;
        for(int i = 0; i < curr.length; i++)
            diff += FastMath.pow(curr[i] - old[i], 2.0);
        return FastMath.pow(diff, 0.5);
    }
    
    protected static double calcDiff(Double[] curr, double[] old) {
        double diff = 0.0;
        for(int i = 0; i < curr.length; i++)
            diff += FastMath.pow(curr[i].doubleValue() - old[i], 2.0);
        return FastMath.pow(diff, 0.5);
    }
    
    protected static double calcDiff(Double[] curr, Double[] old) {
        double diff = 0.0;
        for(int i = 0; i < curr.length; i++)
            diff += FastMath.pow(curr[i].doubleValue() - old[i].doubleValue(), 2.0);
        return FastMath.pow(diff, 0.5);
    }

    protected double[] getPoint(int i) {
        double[] point = new double[dim];
            for(int d = 0; d < dim; d++)
                point[d] = data[d][i];
        return point;
    }
    
    protected Double[] getPointObj(int i) {
        Double[] point = new Double[dim];
            for(int d = 0; d < dim; d++)
                point[d] = data[d][i];
        return point;
    }
    
    protected void createDists() {
    	dists = new ArrayList<MultivariateNormalDistribution>();
    	for(int i = 0; i < means.size(); i++)
    		dists.add(new MultivariateNormalDistribution(
    				Util.doubleObjectToPrimitiveArray(means.get(i)), 
    				covs.get(i).getData()));
    }
    
    protected static boolean distributionUniform(Double[] dist) {
    	return distributionUniform(Util.doubleObjectToPrimitiveArray(dist));
    }
    
    protected static boolean distributionUniform(double[] dist) {
    	double min = dist[0];
    	double max = 0;
    	for(double d : dist) {
    		if(d < min)
    			min = d;
    		else if(d > max)
    			max = d;
    	}
    	return max < 2 * min;
    }
    
    protected ArrayList<Double[]> calculateExpectation() {
        ArrayList<Double[]> expectedValues = new ArrayList<Double[]>();
        createDists();
        for(int i = 0; i < means.size(); i++) {
        	expectedValues.add(new Double[numPoints]);
        	for(int j = 0; j < numPoints; j++)
                expectedValues.get(i)[j] = expectedValuePoint(getPoint(j), i);
        }
        int oldSize = expectedValues.size();
        if(expectedValues.size() < maxDist) {
	    	for(int i = 0 ; i < numPoints; i++) {
	    		if(expectedValues.size() >= maxDist)
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
	                        	covs.get(covs.size() - 1).setEntry(r, c, 10.0);
	                dists.add(new MultivariateNormalDistribution(
	        				Util.doubleObjectToPrimitiveArray(means.get(means.size() - 1)), 
	        				covs.get(covs.size() - 1).getData()));
	        		expectedValues.add(new Double[numPoints]);
	        		for(int k = 0; k < numPoints; k++)
	                    expectedValues.get(expectedValues.size() - 1)[k] = 
	                    	expectedValuePoint(getPoint(k), expectedValues.size() - 1);
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
	    			if(expectedValues.size() <= minDist)
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

    protected double expectedValuePoint(double[] point, int currDist) {
        double probCurrDist = probPoint(point, currDist);
        double probAllDist = 0;
        for(int i = 0; i < means.size(); i++)
            probAllDist += probPoint(point, i);
        if(probAllDist == 0)
            return 0;
        return probCurrDist / probAllDist;
    }

    public static double probPoint(double[] point, 
                                   double[] means, RealMatrix cov) {
        MultivariateNormalDistribution dist = 
            new MultivariateNormalDistribution(means, cov.getData());
        return dist.density(point);
    }
    
    private double probPoint(double[] point, int currDist) {
    	return dists.get(currDist).density(point);
    }
    
    protected void calculateHypothesis(ArrayList<Double[]> expectedValues) {
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
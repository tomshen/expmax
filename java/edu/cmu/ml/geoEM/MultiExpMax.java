package edu.cmu.ml.geoEM;

import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;

public class MultiExpMax
{
    public double[][] data; // {{x-coordinates}, {y-coordinates}}
    public int dim;
    public int numDist;
    public int numPoints;
    public double[][] means;
    public RealMatrix[] covs;
    protected double epsilon = 0.001;

    public MultiExpMax(double[][] data, int k) {
        this.data = data;
        dim = data.length;
        numDist = k;
        numPoints = data[0].length;
        
        initializeMeans();

        covs = new RealMatrix[numDist];
        for(int i = 0; i < numDist; i++) {
            covs[i] = new Array2DRowRealMatrix(new double[dim][dim]);
            for(int r = 0; r < dim; r++)
                for(int c = 0; c < dim; c++)
                    if(r == c) covs[i].setEntry(r, c, 10.0);
        }
    }
    
    private double calculateDistanceSquared(double[] p1, double[] p2) {
    	assert(p1.length == p2.length);
    	double sumSquares = 0;
    	for(int i = 0; i < p1.length; i++)
    		sumSquares += Math.pow((p1[i] - p2[i]), 2.0);
    	return sumSquares;
    }
    
    protected void initializeMeans() {
    	means = new double[numDist][dim];
		int currDist = 0;
		double[] weights = new double[numPoints];
		int meanIndex = (int)(Math.random() * 200.0);
		for(int i = 0; i < dim; i++)
			means[currDist][i] = data[i][meanIndex];
		currDist++;
		while(currDist < numDist) {
			double sumDS = 0;
			for(int i = 0; i < numPoints; i++) {
				double[] currPoint = new double[dim];
				for(int j = 0; j < dim; j++)
					currPoint[j] = data[j][i];
				weights[i] = calculateDistanceSquared(
						currPoint, means[currDist]);
				sumDS += weights[i];
			}
			for(int i = 0; i < numPoints; i++)
				weights[i] /= sumDS;

			meanIndex = -1;
			while(meanIndex == -1) {
				int pointIndex = (int)(Math.random() * 200.0);
				if(Math.random() < weights[pointIndex])
					meanIndex = pointIndex;	
			}
			for(int i = 0; i < dim; i++)
				means[currDist][i] = data[i][meanIndex];
			currDist++;
		}
	}

	public void calculateParameters() {
        double[][] oldMeans = new double[numDist][dim];
        RealMatrix[] oldCovs = new RealMatrix[numDist];
        int i = 0;
        do {
            System.out.println("iteration " + i++);
            oldMeans = Util.deepcopy(means);
            oldCovs = Util.deepcopy(covs);
            calculateHypothesis(calculateExpectation());
            System.out.println(Util.arrayToString(means));
            System.out.println(Util.matricesToString(covs));
        } while(compare(means, oldMeans) || compare(covs, oldCovs));
        System.out.println("Done!");
        System.out.println("Means:\n" + Util.arrayToString(means));
		System.out.println("Covariances:\n" + Util.matricesToString(covs));
    }

    protected boolean compare(double[][] curr, double[][] old) {
        if(old == null)
            return true;
        return calcDiff(curr, old) > epsilon;
    }

    protected boolean compare(RealMatrix[] curr, RealMatrix[] old) {
        if(old == null)
            return true;
        for(int i = 0; i < curr.length; i++)
            if (calcDiff(curr[i].getData(), old[i].getData()) > epsilon)
                return true;
        return false;
    }

    protected double calcDiff(double[][] curr, double[][] old) {    
        double diff = 0.0;
        for(int i = 0; i < curr.length; i++)
            diff += FastMath.pow(calcDiff(curr[i], old[i]), 2.0);
        return FastMath.pow(diff, 0.5);
    }

    protected double calcDiff(double[] curr, double[] old) {
        double diff = 0.0;
        for(int i = 0; i < curr.length; i++)
            diff += FastMath.pow(curr[i] - old[i], 2.0);
        return FastMath.pow(diff, 0.5);
    }

    protected double[] getPoint(int i) {
        double[] point = new double[dim];
            for(int d = 0; d < dim; d++)
                point[d] = data[d][i];
        return point;
    }
    protected double[][] calculateExpectation() {
        double[][] expectedValues = new double[numDist][numPoints];
        for(int i = 0; i < numDist; i++)
            for(int j = 0; j < numPoints; j++)
                expectedValues[i][j] = expectedValuePoint(getPoint(j), i);
        return expectedValues;
    }

    protected double expectedValuePoint(double[] point, int currDist) {
        double probCurrDist = probPoint(point, means[currDist], covs[currDist]);
        double probAllDist = 0;
        for(int i = 0; i < numDist; i++)
            probAllDist += probPoint(point, means[i], covs[i]);
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
    
    protected void calculateHypothesis(double[][] expectedValues) {
        for(int i = 0; i < numDist; i++) {
            double totalExp = 0;
            means[i] = new double[dim];
            for(int j = 0; j < numPoints; j++)
                totalExp += expectedValues[i][j];
            covs[i] = new Array2DRowRealMatrix(new double[dim][dim]);
            for(int d = 0; d < dim; d++) {
                for(int j = 0; j < numPoints; j++)
                    means[i][d] += expectedValues[i][j] * data[d][j];
                means[i][d] /= totalExp;
            }
            for(int r = 0; r < dim; r++) {
                for(int c = 0; c < dim; c++) {
                    double entry = 0;
                    for(int j = 0; j < numPoints; j++) {
                    	entry += (expectedValues[i][j]
                                * (data[r][j] - means[i][r])
                                * (data[c][j] - means[i][c]));
                    }
                    entry = (entry / totalExp) * numPoints / (numPoints - 1);
                    covs[i].setEntry(r, c, entry);
                }
            }
        }
    }
}
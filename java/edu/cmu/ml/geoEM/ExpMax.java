package edu.cmu.ml.geoEM;
/* 
 * ExpMax.java
 * Description: A implementation of the expectation-maximization algorithm
 * for a mixture of k gaussians, with one-dimensional data
 * Author: Tom Shen
 * Date: 02/18/2013
 */

import java.io.IOException;
import java.util.*;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.stat.descriptive.rank.*;

public class ExpMax
{
    public double[] data;
    public int numDist;
    public int numPoints;
    public double[] means;
    public double[] sigmas;
    protected double epsilon = 0.01;

    
    public ExpMax(double[] data, int k) {
        this.data = data;
        numDist = k;
        numPoints = data.length;
        
        means = new double[numDist];
        double minVal = new Min().evaluate(data, 0, data.length);
        double maxVal = new Max().evaluate(data, 0, data.length);
        double interval = (maxVal - minVal) / (numDist - 1);
        for(int i = 0; i < numDist; i++)
            means[i] = minVal + i * interval;

        sigmas = new double[numDist];
        Arrays.fill(sigmas, 1.0);
    }

    protected double[][] calculateParameters() {
        double[] oldMeans = new double[numDist];
        double[] oldSigmas = new double[numDist];
        do {
            double[][] expectedValues = calculateExpectation();
            System.arraycopy(means, 0, oldMeans, 0, means.length);
            System.arraycopy(sigmas, 0, oldSigmas, 0, sigmas.length);
            calculateHypothesis(expectedValues);
        } while(compare(means, oldMeans) || compare(sigmas, oldSigmas));
        System.out.println("Model means:  " + Arrays.toString(means));
        System.out.println("Model sigmas: " + Arrays.toString(sigmas));
        return new double[][] {means, sigmas};
    }

    protected boolean compare(double[] curr, double[] old) {
        if(old == null)
            return true;
        double diff = 0.0;
        for(int i = 0; i < curr.length; i++)
            diff += FastMath.pow(curr[i] - old[i], 2.0);
        return FastMath.pow(diff, 0.5) > epsilon;
    }

    protected double[][] calculateExpectation() {
        double[][] expectedValues = new double[numDist][numPoints];
        for(int i = 0; i < numDist; i++)
            for(int j = 0; j < numPoints; j++)
                expectedValues[i][j] = expectedValuePoint(data[j], i);
        return expectedValues;
    }

    protected double expectedValuePoint(double point, int currDist) {
        double probCurrDist = probPoint(point, means[currDist], sigmas[currDist]);
        double probAllDist = 0;
        for(int i = 0; i < numDist; i++)
            probAllDist += probPoint(point, means[i], sigmas[i]);
        if(probAllDist == 0)
            return 0;
        return probCurrDist / probAllDist;
    }

    protected static double probPoint(double point, double mean, double sigma) {
        NormalDistribution dist = new NormalDistribution(mean, sigma);
        return dist.density(point);
    }
    
    protected void calculateHypothesis(double[][] expectedValues) {
        for(int i = 0; i < numDist; i++) {
            double totalExp = 0;
            means[i] = 0;
            sigmas[i] = 0;
            for(int j = 0; j < numPoints; j++) {
                means[i] += expectedValues[i][j] * data[j];
                totalExp += expectedValues[i][j];
            }
            means[i] /= totalExp;
            for(int j = 0; j < numPoints; j++) {
                sigmas[i] += expectedValues[i][j]
                             * FastMath.pow((data[j] - means[i]), 2.0);
            }
            sigmas[i] = FastMath.pow(sigmas[i] / totalExp, 0.5);
        }
    }
    public static void main(String args[]) throws IOException {
        int k = 3;
        Util.exportFile("temp.txt", KGauss.kgauss(k, 100, 1, -100, 100, 0.1));
        double[] data = Util.importFile("temp.txt")[0];
        ExpMax em = new ExpMax(data, k);
        em.calculateParameters();
    }
}
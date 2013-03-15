package edu.cmu.ml.geoEM;

import static org.junit.Assert.*;

import java.io.IOException;
import java.util.ArrayList;

import org.junit.Test;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;


public class ExpMaxTest {
    private static double epsilon = 10.0;

    private static boolean compareArrays(ArrayList<Double[]> curr, 
            double[][] old) {
        if(curr.size() != old.length)
            return false;
        double diff = 0;
        for(int i = 0; i < curr.size(); i++)
            for(int j = 0; j < curr.get(i).length; j++)
                diff += curr.get(i)[j] - old[i][j];
        return diff < epsilon;
    }
    
    private void testRun(int numDist, int numPoints) {
        int dim = 2;
        double[][] data = new double[dim][numDist * numPoints];
        int currPoint = 0;
        double means[][] = new double[numDist][dim];
        for(int i = 0; i < numDist; i++) {
            for(int j = 0; j < dim; j++)
                means[i][j] = Math.random() * 200.0 - 100.0;
            double[][] cov = new double[dim][dim];
            for(int r = 0; r < dim; r++)
                for(int c = 0; c < dim; c++)
                    if(r == c) cov[r][c] = 5.0;
            MultivariateNormalDistribution dist =
                    new MultivariateNormalDistribution(
                            means[i], cov);
            for(int j = 0; j < numPoints; j++) {
                double[] point = dist.sample();
                for(int d = 0; d < dim; d++)
                    data[d][currPoint] = point[d];
                currPoint++;
            }
        }
        ExpMax em = new ExpMax(data, numDist, numDist);
        System.out.println("\nEm with k=" + numDist 
                   + " and n=" + numPoints);
        em.calculateParameters();
        System.out.println("Actual means:\n"
                   + Util.arrayToString(means));
        assertTrue(compareArrays(em.means, means));
    }
    
    private void testRunCloseClusters(int numDist, int numPoints) {
        int dim = 2;
        double[][] data = new double[dim][numDist * numPoints];
        int currPoint = 0;
        double means[][] = new double[numDist][dim];
        for(int i = 0; i < numDist; i++) {
            if(i == 0)
                for(int j = 0; j < dim; j++)
                    means[i][j] = Math.random() * 200.0 - 100.0;
            else
                for(int j = 0; j < dim; j++)
                    means[i][j] = means[0][j] + 20.0 * Math.random();
            double[][] cov = new double[dim][dim];
            for(int r = 0; r < dim; r++)
                for(int c = 0; c < dim; c++)
                    if(r == c) cov[r][c] = 5.0;
            MultivariateNormalDistribution dist =
                    new MultivariateNormalDistribution(
                            means[i], cov);
            for(int j = 0; j < numPoints; j++) {
                double[] point = dist.sample();
                for(int d = 0; d < dim; d++)
                    data[d][currPoint] = point[d];
                currPoint++;
            }
        }
        ExpMax em = new ExpMax(data, numDist, numDist);
        System.out.println("\nEm with k=" + numDist 
                           + " and n=" + numPoints);
        em.calculateParameters();
        System.out.println("Actual means:\n"
                   + Util.arrayToString(means));
        assertTrue(compareArrays(em.means, means));
    }
    
    private void testRunSmallCov(int numDist, int numPoints) {
        int dim = 2;
        double[][] data = new double[dim][numDist * numPoints];
        int currPoint = 0;
        double means[][] = new double[numDist][dim];
        for(int i = 0; i < numDist; i++) {
            for(int j = 0; j < dim; j++)
                means[i][j] = Math.random() * 200.0 - 100.0;
            double[][] cov = new double[dim][dim];
            for(int r = 0; r < dim; r++)
                for(int c = 0; c < dim; c++)
                    if(r == c) cov[r][c] = 1.0;
            MultivariateNormalDistribution dist =
                    new MultivariateNormalDistribution(
                            means[i], cov);
            for(int j = 0; j < numPoints; j++) {
                double[] point = dist.sample();
                for(int d = 0; d < dim; d++)
                    data[d][currPoint] = point[d];
                currPoint++;
            }
        }
        ExpMax em = new ExpMax(data, numDist, numDist);
        System.out.println("\nEm with k=" + numDist 
                   + " and n=" + numPoints);
        em.calculateParameters();
        System.out.println("Actual means:\n"
                   + Util.arrayToString(means));
        assertTrue(compareArrays(em.means, means));
    }
    
    @Test
    public void testDifferentNumDist() {
        testRun(2, 100);
        testRun(3, 100);
        testRun(4, 100);
        testRun(5, 100);
        testRun(10, 100);
    }
    
    @Test
    public void testDifferentNumPoints() {
        testRun(2, 1000);
        testRun(3, 1000);
        testRun(2, 50);
        testRun(2, 10);
        testRun(3, 10);
    }
    
    @Test
    public void testCloseClusters() {
        testRunCloseClusters(2, 1000);
        testRunCloseClusters(3, 1000);
        testRunCloseClusters(2, 50);
        testRunCloseClusters(2, 10);
        testRunCloseClusters(3, 10);
    }
    
    @Test
    public void testSmallCov() {
        testRunSmallCov(2, 1000);
        testRunSmallCov(3, 1000);
        testRunSmallCov(2, 50);
        testRunSmallCov(2, 10);
        testRunSmallCov(3, 10);
    }
    
    @Test
    public void testToronto() throws IOException {
        double[][] data = Util.importFile("data\\toronto_data.txt");
        ExpMax em = new ExpMax(data, 2, 10);
        em.calculateParameters();
        
    }
}

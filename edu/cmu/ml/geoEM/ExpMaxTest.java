package edu.cmu.ml.geoEM;

import java.io.*;
import java.util.*;

import static org.junit.Assert.*;
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
        ExpMax em = new ExpMax(data, numDist / 2, 2 * numDist);
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
        ExpMax em = new ExpMax(data, numDist / 2, 2 * numDist);
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
        ExpMax em = new ExpMax(data, numDist / 2, 2 * numDist);
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
    
    public void testData(String locationType, String locationName, int i, int j) throws IOException {
        String filepath = File.separator + locationType + File.separator 
                + locationName;
        double[][] data = Util.importFile(
                Util.getFilepath("data", locationType, locationName, ".data"));
        ExpMax em = new ExpMax(data, i, j, locationName, locationType);
        em.calculateParameters();
        Util.writeFile(Util.getFilepath("results", locationType, locationName, ".comp"),
                em.compareToSeed());
        em.exportParameters(
                Util.getFilepath("results", locationType, locationName, ".results"));
    }
    
    public void testDataNoMin(String locationType, String locationName) throws IOException {
        String filepath = File.separator + locationType + File.separator 
                + locationName;
        double[][] data = Util.importFile(
                Util.getFilepath("data", locationType, locationName, ".data"));
        ExpMax em = new ExpMax(data, 2, 50, locationName, locationType);
        em.calculateParameters();
        Util.writeFile(Util.getResultsFilepath("nomin", locationType, locationName, ".comp"),
                em.compareToSeed());
        em.exportParameters(
                Util.getResultsFilepath("nomin", locationType, locationName, ".results"));
    }
    
    public void testDataFixed(String locationType, String locationName, int i) throws IOException {
        String filepath = File.separator + locationType + File.separator 
                + locationName;
        double[][] data = Util.importFile(
                Util.getFilepath("data", locationType, locationName, ".data"));
        ExpMax em = new ExpMax(data, i, i, locationName, locationType);
        em.calculateParameters();
        Util.writeFile(Util.getResultsFilepath("fixed", locationType, locationName, ".comp"),
                em.compareToSeed());
        em.exportParameters(
                Util.getResultsFilepath("fixed", locationType, locationName, ".results"));
    }

    @Test
    public void testNoMin() throws IOException {
        String[] cities = new String[] {
                "Yorkshire_(disambiguation)", 
                "Bermuda_(disambiguation)",
                "Newcastle", 
                "Aberdeen_(disambiguation)", 
                "Camden", 
                "Erie_(disambiguation)"};
        String[] counties = new String[] {
                "Lake_County",
                "Marion_County",
                "Montgomery_County",
                "Monroe_County",
                "Carroll_County",
                "Grant_County"
        };
        for(String c : cities) {
            System.out.println(c);
            testDataNoMin("city", c);
        }
        
        for(String c : counties) {
            System.out.println(c);
            testDataNoMin("county", c);
        }
    }
    
    @Test
    public void testFixed() throws IOException {
        testDataFixed("city", "Aberdeen_(disambiguation)", 10);
        testDataFixed("city", "Bermuda_(disambiguation)", 6);
        testDataFixed("city", "Camden", 21);
        testDataFixed("city", "Erie_(disambiguation)", 12);
        testDataFixed("city", "Newcastle", 15);
        testDataFixed("city", "Yorkshire_(disambiguation)", 6);
        testDataFixed("county", "Carroll_County", 6);
        testDataFixed("county", "Grant_County", 5);
        testDataFixed("county", "Lake_County", 13);
        testDataFixed("county", "Marion_County", 12);
        testDataFixed("county", "Monroe_County", 10);
        testDataFixed("county", "Montgomery_County", 9);
    }
}

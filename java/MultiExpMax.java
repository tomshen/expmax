import java.util.*;
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
    private double epsilon = 0.01;

    public MultiExpMax(double[][] data, int k) {
        this.data = data;
        dim = data.length;
        numDist = k;
        numPoints = data.length;
        
        means = new double[numDist][dim];
        /* TODO: initialize means */

        covs = new RealMatrix[numDist];
        for(int i = 0; i < numDist; i++) {
            covs[i] = new Array2DRowRealMatrix(new double[dim][dim]);
            for(int r = 0; r < dim; r++)
                for(int c = 0; c < dim; c++)
                    if(r == c) covs[i].setEntry(r, c, 1.0);
        }
    }

    public void calculateParameters() {
        double[][] oldMeans = new double[numDist][dim];
        RealMatrix[] oldCovs = new RealMatrix[numDist];
        do {
            oldMeans = Util.deepcopy(means);
            oldCovs = Util.deepcopy(covs);
            calculateHypothesis(calculateExpectation());
        } while(compare(means, oldMeans) || compare(covs, oldCovs));
        System.out.println("Done! Means: " + Arrays.toString(means));
    }

    private boolean compare(double[][] curr, double[][] old) {
        if(old == null)
            return true;
        return calcDiff(curr, old) > epsilon;
    }

    private boolean compare(RealMatrix[] curr, RealMatrix[] old) {
        if(old == null)
            return true;
        for(int i = 0; i < curr.length; i++)
            if (calcDiff(curr[i].getData(), old[i].getData()) > epsilon)
                return true;
        return false;
    }

    private double calcDiff(double[][] curr, double[][] old) {    
        double diff = 0.0;
        for(int i = 0; i < curr.length; i++)
            diff += FastMath.pow(calcDiff(curr[i], old[i]), 2.0);
        return FastMath.pow(diff, 0.5);
    }

    private double calcDiff(double[] curr, double[] old) {
        double diff = 0.0;
        for(int i = 0; i < curr.length; i++)
            diff += FastMath.pow(curr[i] - old[i], 2.0);
        return FastMath.pow(diff, 0.5);
    }

    private double[] getPoint(int i) {
        double[] point = new double[dim];
            for(int d = 0; d < dim; d++)
                point[d] = data[d][i];
        return point;
    }
    private double[][] calculateExpectation() {
        double[][] expectedValues = new double[numDist][numPoints];
        for(int i = 0; i < numDist; i++)
            for(int j = 0; j < numPoints; j++)
                expectedValues[i][j] = expectedValuePoint(getPoint(j), i);
        return expectedValues;
    }

    private double expectedValuePoint(double[] point, int currDist) {
        double probCurrDist = probPoint(point, means[currDist], covs[currDist]);
        double probAllDist = 0;
        for(int i = 0; i < numDist; i++)
            probAllDist += probPoint(point, means[i], covs[i]);
        if(probAllDist == 0)
            return 0;
        return probCurrDist / probAllDist;
    }

    private static double probPoint(double[] point, double[] means, RealMatrix cov) {
        MultivariateNormalDistribution dist = new MultivariateNormalDistribution(means, cov.getData());
        return dist.density(point);
    }
    
    private void calculateHypothesis(double[][] expectedValues) {
        for(int i = 0; i < numDist; i++) {
            double totalExp = 0;
            means[i] = new double[dim];
            covs[i] = new Array2DRowRealMatrix(new double[dim][dim]);
            for(int d = 0; d < dim; d++) {
                for(int j = 0; j < numPoints; j++) {
                    means[i][d] += expectedValues[i][j] * data[d][j];
                    totalExp += expectedValues[i][j];
                }
            }
            for(int d = 0; d < dim; d++)
                means[i][d] /= totalExp;
            for(int r = 0; r < dim; r++) {
                for(int c = 0; c < dim; c++) {
                    for(int j = 0; j < numPoints; j++) {
                        covs[i].setEntry(r, c, expectedValues[i][j]
                                         * (data[r][j] - means[i][r])
                                         * (data[c][j] - means[i][c]));
                    }
                    covs[i].setEntry(r, c, covs[i].getEntry(r, c) 
                    			     / (totalExp * (numPoints - 1) / numPoints));
                }
            }            
        }
    } 
}
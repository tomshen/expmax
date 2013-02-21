package edu.cmu.ml.geoEM;
import java.util.*;
import java.io.*;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;

public class KGauss
{
	public static double[][] multiVariateKGauss(int k, int dim, int n) {
		double[][] data = new double[dim][n * k];
		int count = 0;
		double[][] means = new double[k][];
		for(int i = 0; i < k; i++) {
			double[] mean = new double[dim];
			for(int j = 0; j < dim; j++)
				mean[j] = 100 - 200 * Math.random();
			double[][] cov = new double[dim][dim];
            for(int r = 0; r < dim; r++)
                for(int c = 0; c < dim; c++)
                    if(r == c) cov[r][c] = 10.0;

			MultivariateNormalDistribution dist = 
		            new MultivariateNormalDistribution(mean, cov);
			
			for(int l = 0; l < n; l++) {
				double[] point = dist.sample();
				for(int d = 0; d < dim; d++)
					data[d][count] = point[d];
				count++;
			}
			means[i] = mean;
		}
		System.out.println(Util.arrayToString(means));
		return data;
	}
	
    public static double[] ngauss(double mu, double sigma, int n) {
        double[] data = new double[n];
        int i = 0;
        while(i < n) {
            double u = Math.random();
            double v = Math.random();
            data[i] = mu + sigma * Math.sqrt(-2 * Math.log(u)) 
                      * Math.cos(2 * Math.PI * v);
            i++;
            if(i < n) {
                data[i] = mu + sigma * Math.sqrt(-2 * Math.log(u)) 
                      * Math.sin(2 * Math.PI * v);
                i++;
            }
        }
        return data;
    }
    public static double ngauss(double mu, double sigma) {
        return ngauss(mu, sigma, 1)[0];
    }

    public static double[][] kgauss(int k, int n, int dim, double lower,
                                           double upper, double scale) {
        double[][] data = new double[dim][n*k];
        int index = 0;
        for(int i = 0; i < k; i++) {
            double mu = Math.random() * (upper - lower) + lower;
            double sigma = Math.random() * scale * (upper - lower) 
                           + scale * lower;
            for(int j = 0; j < n; j++) {
                for(int d = 0; d < dim; d++) {
                    data[d][index] = ngauss(mu, sigma);
                }
                index++;
            }
        }
        return data;
    }

    public static void main(String args[]) throws IOException {
        int k = 2;
        int n = 1000;
        int dim = 1;
        int lower = -100;
        int upper = 100;
        double scale = 0.1;
        if(args.length > 1)
            k = Integer.parseInt(args[0]);
        if(args.length > 2)
            n = Integer.parseInt(args[1]);
        if(args.length > 3)
            dim = Integer.parseInt(args[2]);
        if(args.length > 4)
            lower = Integer.parseInt(args[3]);
        if(args.length > 5)
            upper = Integer.parseInt(args[4]);
        if(args.length > 6)
            scale = Double.parseDouble(args[5]);
        double[][] kg = kgauss(k, n, dim, lower, upper, scale);
        Util.exportFile("temp.txt", kg);
        double[][] arr = Util.importFile("temp.txt");
        System.out.println(Arrays.toString(arr[0]));
    }
}
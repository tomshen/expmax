import java.util.*;
import java.io.*;
import org.apache.commons.math3.distribution.NormalDistribution;

class ExpMax
{
	public static void main(String args[]) {
		NormalDistribution dist = new NormalDistribution(100.0, 10.0);
		System.out.println(dist.density(100.0));
	}
    // dist = NormalDistribution(double mean, double sd);
    // dist.density(double x);
    // mdist = MultivariateNormalDistribution(double[] means, double[][] covariances)
    // mdist.density(double[] vals);
}
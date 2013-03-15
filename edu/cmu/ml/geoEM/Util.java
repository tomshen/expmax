package edu.cmu.ml.geoEM;
import java.io.*;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.Charset;
import java.util.*;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.util.FastMath;

public abstract class Util {
    
    public static String arrayToString(double[][] arr) {
        String s = "";
        for(double[] da : arr) {
            for(double d : da)
                s += Double.toString(d) + " ";
            if(da != arr[arr.length - 1])
                s += "\n";
        }
        return s;
    }
    
    public static double[] doubleValues(Double[] arr) {
        double[] primArr = new double[arr.length];
        for(int i = 0; i < arr.length; i++)
            primArr[i] = arr[i].doubleValue();
        return primArr;
    }
    
    public static String matricesToString(RealMatrix[] arr) {
        String s = "";
        for(RealMatrix M : arr)
            s += "Matrix:\n" + arrayToString(M.getData()) + "\n";
        return s;
    }
    
    public static String matricesToString(ArrayList<RealMatrix> arr) {
        String s = "";
        for(RealMatrix M : arr)
            s += "Matrix:\n" + arrayToString(M.getData()) + "\n";
        return s;
    }

    public static double[][] deepcopy(double[][] arr) {
        double[][] arrCopy = new double[arr.length][arr[0].length];
        for(int i = 0; i < arr.length; i++)
            System.arraycopy(arr[i], 0, arrCopy[i], 0, arr[0].length);
        return arrCopy;
    }

    public static RealMatrix[] deepcopy(RealMatrix[] arr) {
        RealMatrix[] arrCopy = new RealMatrix[arr.length];
        for(int i = 0; i < arr.length; i++)
            arrCopy[i] = arr[i].copy();
        return arrCopy;
    }
    
    public static ArrayList<Double[]> deepcopyArray(ArrayList<Double[]> arr) {
        ArrayList<Double[]> arrCopy = new ArrayList<Double[]>();
        for(int i = 0; i < arr.size(); i++) {
            arrCopy.add(new Double[arr.get(0).length]);
            System.arraycopy(arr.get(i), 0, arrCopy.get(i), 
                    0, arr.get(0).length);
        }
        return arrCopy;
    }

    public static ArrayList<RealMatrix> deepcopyMatrix(ArrayList<RealMatrix> arr) {
        ArrayList<RealMatrix> arrCopy = new ArrayList<RealMatrix>();
        for(int i = 0; i < arr.size(); i++)
            arrCopy.add(arr.get(i).copy());
        return arrCopy;
    }
    
    public static double[][] stringToArray(String s) {
        String[] s1 = s.split("\n");
        ArrayList<ArrayList<Double>> dl = new ArrayList<ArrayList<Double>>();
        for(int i = 0; i < s1.length; i++) {
            dl.add(new ArrayList<Double>());
            for(String entry: s1[i].split(" ")) {
                dl.get(i).add(Double.parseDouble(entry));
            }
        }
        double[][] arr = new double[dl.size()][dl.get(0).size()];
        for(int r = 0; r < dl.size(); r++) {
            for(int c = 0; c < dl.get(0).size(); c++) {
                arr[r][c] = dl.get(r).get(c);
            }
        }
        return arr;
    }
    
    public static double[][] importFile(String filename) throws IOException {
        FileInputStream stream = new FileInputStream(new File(filename));
        try {
            FileChannel fc = stream.getChannel();
            MappedByteBuffer bb = fc.map(FileChannel.MapMode.READ_ONLY, 0, fc.size());
            return stringToArray(Charset.defaultCharset().decode(bb).toString());
        }
        finally {
            stream.close();
        }
    }
    
    public static String exportFile(String filename, double[][] data) {
        File outFile = new File(filename);
        BufferedWriter writer = null;
        try {
            writer = new BufferedWriter(new FileWriter(outFile));
            writer.write(arrayToString(data));
        }
        catch (IOException e) {
            System.out.println("Could not write to " + filename);
        }
        finally {
            try {
                if (writer != null)
                    writer.close();
            }
            catch (IOException e) {
                System.out.println("Could not close writer for " + filename);
            }
         }
         return filename;
    }
    
    public static double[] toArray(ArrayList<Double> al) {
        double[] arr = new double[al.size()];
        int i = 0;
        for(double d : al)
            arr[i++] = d;
        return arr;
    }

    /**
     * @param  p1 the first array
     * @param  p2 the second array
     * @return the distance between p1 and p2
     */
    public static double distance(double[][] p1, double[][] p2) {
        double diff = 0.0;
        for(int i = 0; i < p1.length; i++)
            diff += Util.distanceSquared(p1[i], p2[i]);
        return FastMath.pow(diff, 0.5);
    }

    /**
     * @param  p1 the first point
     * @param  p2 the second point
     * @return the Euclidean distance between p1 and p2
     */
    public static double distance(double[] p1, double[] p2) {
        double diff = 0.0;
        for(int i = 0; i < p1.length; i++)
            diff += FastMath.pow(p1[i] - p2[i], 2.0);
        return FastMath.pow(diff, 0.5);
    }

    /** @see #distance distance */
    public static double distance(Double[] p1, Double[] p2) {
        return distance(doubleValues(p1), doubleValues(p2));
    }

    /**
     * @param  p1 the first point
     * @param  p2 the second point
     * @return the square of the Euclidean distance between p1 and p2
     */
    public static double distanceSquared(double[] p1, double[] p2) {
        double sumSquares = 0;
        for(int i = 0; i < p1.length; i++)
            sumSquares += Math.pow((p1[i] - p2[i]), 2.0);
        return sumSquares;
    }

    /** @see #distanceSquared distanceSquared */
    public static double distanceSquared(double[] p1, Double[] p2) {
        return distanceSquared(p1, doubleValues(p2));
    }

    /**
     * @param point
     * @param mean the mean for the distribution
     * @param cov the covariance for the distribution
     * @return the probability the point belongs to the distribution defined by
     * the given mean and the given covariance
     */
    public static double probPoint(double[] point, 
                                   double[] mean, RealMatrix cov) {
        MultivariateNormalDistribution dist = 
            new MultivariateNormalDistribution(mean, cov.getData());
        return dist.density(point);
    }
}

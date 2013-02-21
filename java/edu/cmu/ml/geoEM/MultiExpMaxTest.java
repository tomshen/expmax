package edu.cmu.ml.geoEM;

import static org.junit.Assert.*;

import java.io.IOException;

import org.apache.commons.math3.linear.*;

import org.junit.Test;

public class MultiExpMaxTest {

	@Test
	public void testProbPoint() {
		double copyValueFromPython = 6.20250972071e-147;
		assertEquals(copyValueFromPython,MultiExpMax.probPoint(new double[] {-27,-6,} , new double[] {-9,-90,}, new Array2DRowRealMatrix(new double[][] {{14,0,},{0,11,},})),1e-150);
		
		copyValueFromPython = 2.95791812698e-315;
		assertEquals(copyValueFromPython,MultiExpMax.probPoint(new double[] {-81,-25,} , new double[] {-98,81,}, new Array2DRowRealMatrix(new double[][] {{8,0,},{0,8,},})),1e-320);
		
		copyValueFromPython = 7.80714811896e-167;
		assertEquals(copyValueFromPython,MultiExpMax.probPoint(new double[] {49,8,} , new double[] {-21,-41,}, new Array2DRowRealMatrix(new double[][] {{10,0,},{0,9,},})),1e-170);
	}
	
	@Test
	public void testExpMax() throws IOException {
		double[][] data = Util.importFile("temp.m");
        MultiExpMax em = new MultiExpMax(data, 2);
        em.calculateParameters();
         
        System.out.println("Model means:  " + Util.arrayToString(em.means));
        for(RealMatrix m : em.covs)
            System.out.println("Model cov: " 
                + Util.arrayToString(m.getData()));
	}
}

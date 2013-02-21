package edu.cmu.ml.geoEM;

import static org.junit.Assert.*;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Arrays;

import org.junit.Test;

public class ExpMaxTest {
	@Test
	public void testexpectedValuePoint() {
		double copyValueFromPython = 0.00539909665132;
		assertEquals(copyValueFromPython,ExpMax.probPoint(20, 0, 10),1e-4);
		
		// randomly generated test values in Python
		copyValueFromPython = 6.66558583233e-17;
		assertEquals(copyValueFromPython,ExpMax.probPoint(-34, 89, 15),1e-4);
		
		copyValueFromPython = 1.2410076451e-6;
		assertEquals(copyValueFromPython,ExpMax.probPoint(-28, 0, 6),1e-4);
		
		copyValueFromPython = 0.00437031484895;
		assertEquals(copyValueFromPython,ExpMax.probPoint(-61, -75, 6),1e-4);
		
		copyValueFromPython = 0.0419314697437;
		assertEquals(copyValueFromPython,ExpMax.probPoint(-55, -52, 9),1e-4);
	}
	
	@Test
	public void testExpMax() throws IOException {
		int k = 2;
    	// Util.exportFile("temp.txt", KGauss.kgauss(k, 100, 1, -100, 100, 0.1));
    	double[] data = Util.importFile("temp.0")[0];
    	double[][] pythonParams = new double[][] { { 5.99909154243, 56.0399394117, }, { 6.15773090876, 5.34897679826, } };
        ExpMax em = new ExpMax(data, k);
        double[][] params = em.calculateParameters();
        Util.roundArray(pythonParams);
        Util.roundArray(params);
        assertArrayEquals(pythonParams, params);
        
        data = Util.importFile("temp.1")[0];
        pythonParams = new double[][] { { -0.410591419827, 28.5893379139, }, { 6.55528895793, 6.84560112468, }, };
        em = new ExpMax(data, k);
        params = em.calculateParameters();
        Util.roundArray(pythonParams);
        Util.roundArray(params);
        assertArrayEquals(pythonParams, params);
        
        data = Util.importFile("temp.2")[0];
        pythonParams = new double[][] { { -94.4554393446, 11.6315567616, }, { 7.71865613213, 7.79345596918, }, };        em = new ExpMax(data, k);
        params = em.calculateParameters();
        Util.roundArray(pythonParams);
        Util.roundArray(params);
        assertArrayEquals(pythonParams, params);
	}

}

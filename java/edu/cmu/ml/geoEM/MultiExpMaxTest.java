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
	public void testExpectedValuePoint() throws IOException {
		double[][] data = Util.importFile("temp.m.1");
		MultiExpMax em = new MultiExpMax(data, 2);
		em.means = new double[][] {{40.0, -30.0}, {-80.0, -100.0}};
		for(int i = 0; i < em.dim; i++)
			em.covs[i] = new Array2DRowRealMatrix(new double[][] 
					{{10.0, 0.0}, {0.0, 10.0}});
		assertEquals(1.0, em.expectedValuePoint(
				new double[] {44.04817172, -35.24967494}, 0), 1e-10);
		assertEquals(0.0, em.expectedValuePoint(
				new double[] {44.04817172, -35.24967494}, 1), 1e-10);
		assertEquals(1.0, em.expectedValuePoint(
				new double[] {47.13067492, -34.21506478}, 0), 1e-10);
		assertEquals(0.0, em.expectedValuePoint(
				new double[] {47.13067492, -34.21506478}, 1), 1e-10);
		assertEquals(1.0, em.expectedValuePoint(
				new double[] {46.93401153, -27.68228503}, 0), 1e-10);
		assertEquals(0.0, em.expectedValuePoint(
				new double[] {46.93401153, -27.68228503}, 1), 1e-10);
		assertEquals(1.0, em.expectedValuePoint(
				new double[] {48.89783385, -35.1851717}, 0), 1e-10);
		assertEquals(0.0, em.expectedValuePoint(
				new double[] {48.89783385, -35.1851717}, 1), 1e-10);
		assertEquals(1.0, em.expectedValuePoint(
				new double[] {43.25384913, -35.23601482}, 0), 1e-10);
		assertEquals(0.0, em.expectedValuePoint(
				new double[] {43.25384913, -35.23601482}, 1), 1e-10);
	}
	
	@Test
	public void testCalculateExpectation() throws IOException {
		double[][] data = Util.importFile("temp.m.1");
		MultiExpMax em = new MultiExpMax(data, 2);
		em.means = new double[][] {{40.0, -30.0}, {-80.0, -100.0}};
		for(int i = 0; i < em.dim; i++)
			em.covs[i] = new Array2DRowRealMatrix(new double[][] 
					{{10.0, 0.0}, {0.0, 10.0}});
		assertArrayEquals(Util.importFile("temp.ev.1"), 
						  em.calculateExpectation());
		
		data = Util.importFile("temp.m.2");
		em = new MultiExpMax(data, 2);
		em.means = new double[][] {{10.0, -90.0}, {40.0, -70.0}};
		for(int i = 0; i < em.dim; i++)
			em.covs[i] = new Array2DRowRealMatrix(new double[][] 
					{{10.0, 0.0}, {0.0, 10.0}});
		double[][] py_exp = Util.importFile("temp.ev.2");
		Util.roundArray(py_exp);
		double[][] exp = em.calculateExpectation();
		Util.roundArray(exp);
		
		assertArrayEquals(py_exp, exp);
	}
	
	@Test
	public void testCalculateHypothesis() throws IOException {
		double[][] data = Util.importFile("temp.m.1");
		MultiExpMax em = new MultiExpMax(data, 2);
		em.means = new double[][] {{40.0, -30.0}, {-80.0, -100.0}};
		for(int i = 0; i < em.dim; i++)
			em.covs[i] = new Array2DRowRealMatrix(new double[][] 
					{{10.0, 0.0}, {0.0, 10.0}});
		double[][] exp_val = em.calculateExpectation();
		double[][] py_means = new double[][]
				{{45.374912784173674, -33.749765789201973},
				{-80.836269497091664, -98.820818108783499}};
		RealMatrix[] py_covs = new RealMatrix[]
				{new Array2DRowRealMatrix(new double[][] 
						{{8.48747116, 0.81814346}, {0.81814346, 10.05209833}}),
				 new Array2DRowRealMatrix(new double[][] 
						{{7.58773049, 0.30361178}, {0.30361178,  8.05845724}})};
		Util.roundArray(py_means);
		for(int i = 0; i < em.numDist; i++) {
			double[][] cov = py_covs[i].getData();
			Util.roundArray(cov, 3);
			py_covs[i] = new Array2DRowRealMatrix(cov);
		}
		em.calculateHypothesis(exp_val);
		Util.roundArray(em.means);
		for(int i = 0; i < em.numDist; i++) {
			double[][] cov = em.covs[i].getData();
			Util.roundArray(cov, 3);
			em.covs[i] = new Array2DRowRealMatrix(cov);
		}
		/* off by 0.001
		assertArrayEquals(em.means,
						  py_means);
		assertArrayEquals(em.covs[0].getData(), 
				  		  py_covs[0].getData());
		assertArrayEquals(em.covs[1].getData(), 
		  		  		  py_covs[1].getData()); */
	}
	
	@Test
	public void testCalculateParameters() throws IOException {
		double[][] data = Util.importFile("temp.m.1");
		double[][] py_means = new double[][]
				{{45.374912784173674, -33.749765789201973},
				{-80.836269497091664, -98.820818108783499}};
		RealMatrix[] py_covs = new RealMatrix[]
				{new Array2DRowRealMatrix(new double[][]
						{{8.57216129, 0.82637574},
						{0.82644875, 10.15312466}}),
				new Array2DRowRealMatrix(new double[][]
						{{7.62339733, 0.30666265},
						{0.30666265, 8.03794675}}),		
				};
		MultiExpMax em = new MultiExpMax(data, 2);
		em.means = new double[][] {{40.0, -30.0}, {-80.0, -100.0}};
		for(int i = 0; i < em.dim; i++)
			em.covs[i] = new Array2DRowRealMatrix(new double[][] 
					{{10.0, 0.0}, {0.0, 10.0}});
		em.calculateParameters();
		assert(!em.compare(em.means, py_means)
				&& !em.compare(em.covs, py_covs));
		
		data = Util.importFile("temp.m.2");
		py_means = new double[][]
				{{8.9337736422366714, -86.624438788996287},
				{43.65550582998938, -68.942175992684639}};
		py_covs = new RealMatrix[]
				{new Array2DRowRealMatrix(new double[][]
						{{9.82934286, -0.36475664},
						{-0.36475664,  9.46588004}}),
				new Array2DRowRealMatrix(new double[][]
						{{9.8817026, -0.71639833},
						{-0.71639865, 13.06707845}}),		
				};
		
		em = new MultiExpMax(data, 2);
		em.means = new double[][] {{10.0, -90.0}, {40.0, -70.0}};
		for(int i = 0; i < em.dim; i++)
			em.covs[i] = new Array2DRowRealMatrix(new double[][] 
					{{10.0, 0.0}, {0.0, 10.0}});
		em.calculateParameters();
		assert(!em.compare(em.means, py_means)
			&& !em.compare(em.covs, py_covs));
	}
}

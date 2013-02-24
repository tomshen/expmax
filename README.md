# Expectation Maximization for Geographical Data

An implementation of the expectation-maximization algorithm for a mixture of 
gaussians, designed to organize geographical data into clusters.

## To Do
* Mean initialization
    * Pick random points from the data
    * Test to make sure points aren't too close
    * Run several trials
* Heuristic for identification of number of clusters
    * exploratory EM -- introduce new cluster groupings on-the-fly
* Testing on real-world data
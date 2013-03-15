# Expectation Maximization for Geographical Data

An implementation of the expectation-maximization algorithm for a mixture of 
gaussians, designed to decompose a single geographical distribution into clusters.

Relies on the [Apache Commons Mathematics Library](http://commons.apache.org/proper/commons-math/)
for matrices and implementation of distributions.

## Known Issues
* Does not work with an unknown number of means
  * Consider checking for "large" covariance matrices--uncommon in real data,
  common in the model
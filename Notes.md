# Notes

## Libraries
### Python
* [SciPy](http://www.scipy.org/)
* [NumPy](http://www.numpy.org/)
* [matplotlib](http://matplotlib.org/)

### Java
* [Apache Commons Math](http://commons.apache.org/math/)

### Final requirements
* unknown number of points - can handle
* unequal points per cluster - can handle
    * breaks if one distribution has 10 points, the other > 1000
* unknown number of clusters
    * Dr. Cohen will talk to another grad student working on heuristics for similar problem
* unknown/unequal sd/covariance for each cluster - working for 1d
* 2d data - can handle
    * longitude [-90, 90]
    * latitude [-180, 180]
* in Java

### Tests
* check how many points are predicted to be in a distribution, vs how many actually are
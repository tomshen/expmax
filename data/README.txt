toronto.txt - This is a portion of the google-inlinks/dbpedia dataset which matches "toronto". It is tab-delimited, as follows:
  text of the link
  conditional probability that the link goes to the dbpedia page (NB not normalized!)
  dbpedia page
  latitude associated with dbpedia page
  longitude associated with dbpedia page
  
  More information about this dataset is at http://www-nlp.stanford.edu/pubs/crosswikis-data.tar.bz2/READ_ME.txt, but the gist of our use of the dataset is as follows: Given some string, we want to know what locations on the earth's surface are associated with that string. It is difficult to get a master list of all valid locations, so we're using dbpedia as an approximation: If a string links to a dbpedia page that has a lat/long, then we say the dbpedia page is a location, and that the string is associated with that location (where "associated with" is a reflexive relationship). Most strings in the dataset are associated with more than one location as defined by dbpedia, but this is partially because dbpedia distinguishes between e.g. the Toronto Post Office and the Toronto Zoo and the Anglican Diocese of Toronto, all of which are located in the city of Toronto... but as you well know, on top of that, there is more than one city called Toronto, and making *that* distinction is what this project is all about.

  So far you've been dealing with a flat set of randomly-generated points. This file can act as a flat set of points, and that's how you should start, just by taking the last two fields and stripping out the other data. This will uncover any issues with the difference between randomly-generated points and actual data (for example, this dataset has a lot of exact duplication in it. Is your algorithm robust? Is it necessary to unique() the set of points first, or should the points be weighted by their frequency, or does it work just fine without any adjustments at all?).

  After it works with the flat set, what we want to do is take the conditional probability value as a weight for the point in the dataset. This adds an extra layer of weights that you haven't had to handle yet, on top of the expected value weights you're already using in EM. Note that these "conditional probability" values are not normalized -- the original data set was not specific to geolocations, and stripping out the pages that don't have latlong values has caused the original probabilities to no longer sum to 1. It may be easier to debug your code if you normalize the weights first in a separate step.

  Optimally, your code should be robust to both weighted and unweighted datasets, taking advantage of as much code reuse as possible between the two cases.

  There are much, much more data where these came from -- if you need another ambiguity example, I can provide.

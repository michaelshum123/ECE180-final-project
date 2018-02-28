import numpy
import urllib
import scipy.optimize


def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

print "Reading data..."
data = list(parseData("http://jmcauley.ucsd.edu/cse190/data/beer/beer_50000.json"))
print "done"

#for least square regression, feat always has to begin with 1 
def feature(datum):
  feat = [1]
  feat.append(datum['beer/ABV'])
  return feat

X = [feature(d) for d in data]

#in our case, y = bitcoin price
y = [d['review/overall'] for d in data]

theta,residuals,rank,s = numpy.linalg.lstsq(X, y)

#to calculate future prices,y = x*theta, y(1,1) = row(1,N) * col(N,1) 
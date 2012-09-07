
from scipy.stats import *
from numpy import *
import sys

## Open sample and test files
sample_file = open('sample', 'r')
test_file = open('test', 'r')

## Parse sample data
sample = []
for line in sample_file:
  pair = line.split()
  sample.append([float(pair[0]), int(pair[1])-1])
sample_file.close()

## Parse test data
test = []
for line in test_file:
  pair = line.split()
  test.append([float(pair[0]), int(pair[1])-1])
test_file.close()

## Group data by classes
raw_classes = [[], [], []]
for data in sample:
  raw_classes[data[1]].append(data[0])

## Convert to numpy arrays
classes = [array(c) for c in raw_classes]

distributions = [ uniform, expon, norm ]

class Likelihood:
  def __init__(self, dist, data):
    if dist == uniform:
      self.min = data.min()
      self.max = data.max()
      self.likelihood = lambda self,x: dist.pdf(x, self.min, self.max)
    elif dist == expon:
      self.lamb = 1.0/data.mean()
      self.likelihood = lambda self,x: dist.pdf(x, 0, self.lamb)
    elif dist == norm:
      self.mean = data.mean()
      self.std = data.std()
      self.likelihood = lambda self,x: dist.pdf(x, self.mean, self.std)
  def __call__(self, x):
    return self.likelihood(self,x)

class Classifier:
  def __init__(self, dist1, dist2, dist3):
    self.likelihoods = [ Likelihood(dist1, classes[0]),
                         Likelihood(dist2, classes[1]),
                         Likelihood(dist3, classes[2]) ]
  def __call__(self,x):
    argmax = 0
    posteriori = 0
    for i in range(3):
      p = self.likelihoods[i](x)
      if p > posteriori:
        posteriori = p
        argmax = i
    return argmax


for dist1 in distributions:
  for dist2 in distributions:
    for dist3 in distributions:
      classify = Classifier(dist1, dist2, dist3)
      hits = 0
      for x in test:
        if classify(x[0]) == x[1]:
          hits += 1
      print(dist1, dist2, dist3, hits)


##print [c.min() for c in classes]
##print [c.max() for c in classes]
##print [c.mean() for c in classes]
##print [c.var() for c in classes]
##print norm.pdf(0.000161578142981)


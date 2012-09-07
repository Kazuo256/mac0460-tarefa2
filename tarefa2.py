
from scipy.stats import *
from numpy import *
import sys

## Open sample and test files
sample_file = open('sample', 'r')
test_file = open('test', 'r')

## Parse sample data
samples = []
for line in sample_file:
  pair = line.split()
  samples.append([float(pair[0]), int(pair[1])-1])
sample_file.close()

## Parse test data
tests = []
for line in test_file:
  pair = line.split()
  tests.append([float(pair[0]), int(pair[1])-1])
test_file.close()

## Group data by class samples
raw_class_samples = [[], [], []]
for data in samples:
  raw_class_samples[data[1]].append(data[0])

## Class to hold samples' relevant information
class SampleInfo:
  def __init__ (self, data):
    self.min = data.min()
    self.max = data.max()
    self.mean = data.mean()
    self.std = data.std()

## Convert to numpy arrays and extract relevant info
class_samples = [SampleInfo(array(c)) for c in raw_class_samples]

## Base likelihood probability class
class Likelihood:
  def __init__(self, dist, sample):
    self.dist = dist
    self.sample = sample
  def __call__(self, x):
    return self.dist.pdf(x, self.local(), self.scale())

## Uniform distribution likelihood class
class UniformLikelihood(Likelihood):
  def local(self):
    return self.sample.min
  def scale(self):
    return self.sample.max - self.sample.min

## Exponencial distribution likelihood class
class ExponencialLikelihood(Likelihood):
  def local(self):
    return 0
  def scale(self):
    return self.sample.mean ## 1.0/lambda

## Normal distribution likelihood class
class NormalLikelihood(Likelihood):
  def local(self):
    return self.sample.mean
  def scale(self):
    return self.sample.std

likelihood_map = {
  uniform: UniformLikelihood,
  expon: ExponencialLikelihood,
  norm: NormalLikelihood
}

class Classifier:
  def __init__(self, dists):
    self.likelihoods = [
      likelihood_map[dist] (dist, class_samples[i]) for i,dist in enumerate(dists)
    ]
  def __call__(self,x):
    ## reminder: P(class) = 1/3 always
    posteriors = array([ likely(x)*(1.0/3.0) for likely in self.likelihoods ])
    return posteriors.argmax()

distributions = {
  'U': uniform,
  'E': expon,
  'N': norm
}

for k1,dist1 in distributions.iteritems():
  for k2,dist2 in distributions.iteritems():
    for k3,dist3 in distributions.iteritems():
      classify = Classifier([dist1, dist2, dist3])
      hits = 0
      for x in tests:
        if classify(x[0]) == x[1]:
          hits += 1
      print(k1, k2, k3, hits)



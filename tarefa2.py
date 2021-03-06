
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
    self.data = data
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
    mles = dist.fit(sample.data)
    self.__local = mles[0]
    self.__scale = mles[1]
  def __call__(self, x):
    return self.dist.pdf(x, loc=self.local(), scale=self.scale())
  def local(self):
    return self.__local
  def scale(self):
    return self.__scale

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

## Maps probability distributions to their likelihoods
## Change mapping to use ScyPy's estimation instead.
likelihood_map = {
  ## These ones directly calculate the estimators
  uniform:  UniformLikelihood,
  expon:    ExponencialLikelihood,
  norm:     NormalLikelihood
  ## These ones use ScyPy's estimation
  #uniform:  Likelihood,
  #expon:    Likelihood,
  #norm:     Likelihood
}

## Class for a Bayesian Classifier
class Classifier:
  def __init__(self, name, dists):
    self.name = name
    self.likelihoods = [
      likelihood_map[dist] (dist, class_samples[i]) for i,dist in enumerate(dists)
    ]
  def __call__(self,x):
    ## reminder: P(class) = 1/3 always
    posteriors = array([ likely(x)*(1.0/3.0) for likely in self.likelihoods ])
    return posteriors.argmax()
  def __str__(self):
    return self.name

## Available distributions
distributions = {
  'U': uniform,
  'E': expon,
  'N': norm
}

## Evaluates a classifiers' performance with the test set data.
## This is the part where I got lazy.
def performance(classify):
  print 'Classificador %s:' % classify
  print '  Classe Real:  Classe escolhida: Dado observado:'
  hits = 0
  classes = [[0,0],[0,0],[0,0]]
  for x in tests:
    chosen_class = classify(x[0])
    classes[chosen_class][0] += 1
    print '       %-7d          %-9d    %f' % (x[1]+1, chosen_class+1, x[0])
    if chosen_class == x[1]:
      hits += 1
      classes[chosen_class][1] += 1
  for i,count in enumerate(classes):
    print 'Classificados como Classe %d: %d (%d corretos)' % \
          (i+1, count[0], count[1])
  print 'Total de acertos: %d\n' % hits
  return classify, hits

## Stores the results of the performance tests
performance_tests = [
  performance(Classifier(('(%s,%s,%s)'%(k1,k2,k3)),[dist1, dist2, dist3]))
  for k1,dist1 in distributions.iteritems()
  for k2,dist2 in distributions.iteritems()
  for k3,dist3 in distributions.iteritems()
]

## Finds out the best classifiers
best = []
last_max = 0
for test in performance_tests:
  ## test[0] -- the classifier itself
  ## test[1] -- classifier's performance
  if test[1] > last_max:
    best = [test[0]]
    last_max = test[1]
  elif test[1] == last_max:
    best.append(test[0])

## Display best classifiers
print 'Melhores classificadores:\n'
for classifier in best:
  print '\tClassificador %s' % classifier
print ''


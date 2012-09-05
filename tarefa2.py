
from scipy.stats import *
from numpy import *
import sys

sample_file = open('sample', 'r')
sample = []

for line in sample_file:
  pair = line.split()
  sample.append([float(pair[0]), int(pair[1])])

raw_classes = [[], [], []]

for data in sample:
  raw_classes[data[1]-1].append(data[0])

classes = [array(c) for c in raw_classes]

print [c.min() for c in classes]
print [c.max() for c in classes]
print [c.mean() for c in classes]
print [c.var() for c in classes]
print norm.pdf(0.000161578142981)


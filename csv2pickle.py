#!/usr/bin/env python

import pandas
import pickle

import sys, os

filename = sys.argv[1]
outputfile = os.path.basename(filename)
outputfile = os.path.splitext(outputfile)[0] + '.pkl'

csv = pandas.read_csv(filename)
csv.to_pickle(outputfile)
#pickle.dump(csv, open(outputfile, 'w'))

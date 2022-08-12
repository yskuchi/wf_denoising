#!/usr/bin/env python

import pandas
import pickle

import sys, os

filename = sys.argv[1]
outputfile = os.path.basename(filename)
outputfile = os.path.splitext(outputfile)[0] + '.pkl' + '.gz'
# outputfile = os.path.splitext(outputfile)[0] + '.pkl' + '.bz2'

csv = pandas.read_csv(filename, header=None)
csv.to_pickle(outputfile, protocol = 4) # when protocol 5 is supported in Google Colab, remove the protocol specification
#pickle.dump(csv, open(outputfile, 'w'))

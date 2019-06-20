#!/usr/bin/env python

import caffe
import string
import numpy as np
import random
import math

np.set_printoptions(threshold='nan')


MODEL_FILE = '../examples/cifar10/cifar10_quick.prototxt'
WEIGHT_FILE = '../examples/cifar10/cifar10_quick_iter_5000.caffemodel.h5'
net1 = caffe.Net(MODEL_FILE, WEIGHT_FILE, caffe.TEST)

layer = 0
for param_name in net1.params.keys():
	weight = net1.params[param_name][0].data
	bias = net1.params[param_name][1].data
	if len(weight.shape)==4:
		# f = open(str(layer)+'.txt', 'r+')
		# flist=f.readlines()
		f = open(str(layer)+'.txt', 'w+')
		for x in range(weight.shape[0]):
			for y in range(weight.shape[1]):
				for z in range(weight.shape[2]):
					for m in range(weight.shape[3]):
						f.write('1\n')
		# f = open(str(layer)+'.txt', 'w+')
		# flist[5]='11111111111111111111111\n'
		# f.writelines(flist)
		f.close()

	layer = layer + 1
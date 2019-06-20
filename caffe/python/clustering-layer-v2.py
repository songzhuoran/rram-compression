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

class Point(object):
	def __init__(self,values,total_values,id_layer,id_point,tag):
		self.total_values=total_values
		self.values=[]
		for i in range(total_values):
			self.values.append(values[i])

		self.id_layer=id_layer
		self.id_point=id_point
		self.tag=0
	
	def getLayer(self):
		return self.id_layer

	def getID(self):
		return self.id_point
		
	def getTag(self):
		return self.tag
		
	def setTag(self,tag):
		return self.tag=tag

	def getValue(self,index):
		return self.values[index]

	def getTotalValues(self):
		return self.total_values
	
	def setValues(self,values,total_values):
		self.values=[]
		for i in range(total_values):
			self.values.append(values[i])

class Layer(object):
	def __init__(self,id_layer):
		self.id_layer=id_layer
		self.points=[]
	def addPoint(self,point):
		self.points.append(point)


id_layer = 0
total_values = 50
for param_name in net1.params.keys():
	weight = net1.params[param_name][0].data
	bias = net1.params[param_name][1].data
	if len(weight.shape)==4:
		values=[]
		count = 0
		Layer(id_layer)
		for x in range(weight.shape[0]):
			for y in range(weight.shape[1]):
				for z in range(weight.shape[2]):
					for m in range(weight.shape[3]):
						if count % total_values == 0 and count != 0:
							Point(values,total_values,id_layer,id_point)
							Layer(id_layer).addPoint(Point)
							values=[]
						count = count +1
						values.append(net1.params[param_name][0].data[x,y,z,m])
		id_layer = id_layer + 1


# max_iterations = 10000
# min_dist = 100000
# for i in range(total):
	# for j in range(i+1,total):
		# sum=0.0
		# for u in range(total_values):
			# sum += pow(points[i].getValue(u)-points[j].getValue(u),2.0)
		# dist = math.sqrt(sum)
		# if dist < min_dist :
			# min_dist = dist

# threshold = 2.0 * min_dist
threshold=1.0

#cluster layer 1 for cifar10
start_layer=0
start_point_num=len(Layer(start_layer).points)
count=0
for x in range(start_point_num):
	count=0
	for i in range(id_layer-1,-1,-1):
		for j in range(len(Layer(i).points)):
			for u in range(total_values):
				sum += pow(Layer(start_layer).points[x].getValue(u)-Layer(i).points[j].getValue(u),2.0)
			dist = math.sqrt(sum)
			if dist < threshold and count!=3:
				count=count+1
				Layer(start_layer).points[x].setTag(i)
				Layer(i).points[j].setTag(start_layer)
			else if count == 3:
				break
		if count == 3:
			break




layer = 0
for param_name in net1.params.keys():
	weight = net1.params[param_name][0].data
	bias = net1.params[param_name][1].data
	if len(weight.shape)==4:
		f = open(str(layer)+'.txt', 'r+')
		flist=f.readlines()
		for x in range(weight.shape[0]):
			for i in range(kmeans.K):
				for j in range(kmeans.clusters[i].getTotalPoints()):
					if layer == kmeans.clusters[i].getPoint(j).getLayer() and x == kmeans.clusters[i].getPoint(j).getFilter() :
						start = kmeans.clusters[i].getPoint(j).getID() * total_values
						for w in range(kmeans.total_values):
							f = open(str(layer)+'.txt', 'w+')
							flist[(w+start)%weight.shape[3] + ((w+start)/weight.shape[3])%weight.shape[2] * weight.shape[3] + ((w+start)/(weight.shape[2]*weight.shape[3]))%weight.shape[1] * weight.shape[3] * weight.shape[2] + x * weight.shape[3] * weight.shape[2] * weight.shape[1]]='0\n'
							f.writelines(flist)
							net1.params[param_name][0].data[x,((w+start)/(weight.shape[2]*weight.shape[3]))%weight.shape[1],((w+start)/weight.shape[3])%weight.shape[2],(w+start)%weight.shape[3]] = kmeans.clusters[i].getPoint(j).getValue(w)

		f.close()
	layer = layer + 1

net1.save('new.caffemodel')

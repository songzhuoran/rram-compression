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
	def __init__(self,values,total_values,id_layer,id_filter,id_point):
		self.total_values = total_values
		self.values=[]
		for i in range(total_values):
			self.values.append(values[i])

		self.id_cluster=-1
		self.id_layer=id_layer
		self.id_filter=id_filter
		self.id_point=id_point
	
	def getLayer(self):
		return self.id_layer

	def getFilter(self):
		return self.id_filter

	def getID(self):
		return self.id_point

	def setCluster(self,id_cluster):
		self.id_cluster=id_cluster

	def getCluster(self):
		return self.id_cluster

	def getValue(self,index):
		return self.values[index]

	def getTotalValues(self):
		return self.total_values

	def addValue(self,value):
		self.values.append(value)
	
	def setValues(self,values,total_values):
		self.values=[]
		for i in range(total_values):
			self.values.append(values[i])


class Cluster(object):
	def __init__(self,id_cluster):
		self.id_cluster=id_cluster
		self.central_values=[]
		self.points=[]


	def addPoint(self,point):
		self.points.append(point)


	def removePoint(self,id_point):
		total_points = len(self.points)
		for i in range(total_points):
			if self.points[i].getID() == id_point :
				self.points.pop(i)
				break


	def getCentralValue(self,index):
		return self.central_values[index]

	def iniCentralValue(self,values):
		length = len(values)
		for i in range(length):
			self.central_values.append(values[i])

	def setCentralValue(self,index,value):
		self.central_values[index] = value

	def getPoint(self,index):
		return self.points[index]

	def getTotalPoints(self):
		return len(self.points)

	def getID(self):
		return self.id_cluster

class KMeans(object):
	def __init__(self,K,total_points,total_values,max_iterations,id_layer,min_dist):
		self.K = K
		self.total_points=total_points
		self.total_values=total_values
		self.max_iterations=max_iterations
		self.clusters=[]
		self.id_layer = id_layer
		self.min_dist = min_dist


	def run(self,points):
		# if self.K > self.total_points:
		# 	return
		
		for i in range(self.K):
			cluster=Cluster(i)
			self.clusters.append(cluster)
		
		id_cluster = 0
		# remember j and u
		mem_j = 0
		mem_u = 0
		judge = 0
		list_u = []
		for i in range(self.id_layer-1):
			list_u = []
			for j in range(self.total_points):
				if points[j].getLayer() == i:
					sub_min_dist = self.min_dist
					for u in range(self.total_points):
						if points[u].getLayer() == i + 1 :
							sum=0.0
							for w in range(self.total_values):
								sum += pow(points[j].getValue(w)-points[u].getValue(w),2.0)
							sum = math.sqrt(sum)
							if sum <= sub_min_dist :
								sub_min_dist = sum
								mem_j = j
								mem_u = u
								judge = 1
					if judge == 1 :
						if mem_u not in list_u:
							list_u.append(mem_u)
							points[mem_j].setCluster(id_cluster)
							points[mem_u].setCluster(id_cluster)
							# print len(self.clusters)
							# print id_cluster
							# print j
							# print total_points
							self.clusters[id_cluster].addPoint(points[mem_j])
							self.clusters[id_cluster].addPoint(points[mem_u])
							initial_value=[]
							for w in range(self.total_values):
								initial_value.append((points[mem_j].getValue(w) + points[mem_u].getValue(w))/2)
							self.clusters[id_cluster].iniCentralValue(initial_value)
							print('id_cluster')
							print id_cluster
							id_cluster = id_cluster + 1

		self.K = id_cluster
		# for i in range(self.K):
		# 	for j in range(self.total_values):
		# 		total_points_cluster = self.clusters[i].getTotalPoints()
		# 		sum = 0.0
		# 		if total_points_cluster > 0 :
		# 			for p in range(total_points_cluster):
		# 				sum += self.clusters[i].getPoint(p).getValue(j)
		# 			self.clusters[i].setCentralValue(j, sum / total_points_cluster)

		


id_layer = 0
total_points = 0
points=[]
total_values_ori=50
total_values = 50
for param_name in net1.params.keys():
	weight = net1.params[param_name][0].data
	bias = net1.params[param_name][1].data
	if len(weight.shape)==4:
		values=[]
		for x in range(weight.shape[0]):
			id_point=0
			count = 0
			values=[]
			for y in range(weight.shape[1]):
				for z in range(weight.shape[2]):
					for m in range(weight.shape[3]):
						if count % total_values == 0 and count != 0:
							points.append(Point(values,total_values,id_layer,x,id_point))
							total_points = total_points + 1
							id_point = id_point + 1
							values=[]
						count = count +1
						values.append(net1.params[param_name][0].data[x,y,z,m])
		id_layer = id_layer + 1


max_iterations = 10000
min_dist = 100000
# define the number of clusters, we let K to be very large
K = pow(total_points,2)
# K = total_points - 1
for i in range(total_points):
	for j in range(i+1,total_points):
		sum=0.0
		for u in range(total_values):
			sum += pow(points[i].getValue(u)-points[j].getValue(u),2.0)
		dist = math.sqrt(sum)
		if dist < min_dist :
			min_dist = dist

print('min_dist:')
print min_dist
print('total_points')
print total_points

min_dist = 10000000.0 * min_dist
sub_min_dist = min_dist

kmeans=KMeans(K, total_points, total_values, max_iterations, id_layer, min_dist)
kmeans.run(points)

print kmeans.K

for i in range(kmeans.K):
	values = []
	for j in range(kmeans.total_values):
		values.append(kmeans.clusters[i].getCentralValue(j))

	total_points_cluster =  kmeans.clusters[i].getTotalPoints()
	for j in range(total_points_cluster):
		kmeans.clusters[i].getPoint(j).setValues(values,kmeans.total_values)


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

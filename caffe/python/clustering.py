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
	def __init__(self,id_cluster,point):
		self.id_cluster=id_cluster
		total_values=point.getTotalValues()
		self.central_values=[]
		self.points=[]
		for i in range(total_values):
			self.central_values.append(point.getValue(i))
		self.points.append(point) 


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

	def setCentralValue(self,index,value):
		self.central_values[index] = value

	def getPoint(self,index):
		return self.points[index]

	def getTotalPoints(self):
		return len(self.points)

	def getID(self):
		return self.id_cluster

class KMeans(object):
	def __init__(self,K,total_points,total_values,max_iterations):
		self.K=K
		self.total_points=total_points
		self.total_values=total_values
		self.max_iterations=max_iterations
		self.clusters=[]

	def getIDNearestCenter(self,point):
		sum=0.0
		id_cluster_center = 0
		for i in range(self.total_values):
			sum += pow(self.clusters[0].getCentralValue(i)-point.getValue(i),2.0)
		min_dist = math.sqrt(sum)
		for i in range(1,self.K):
			sum = 0.0
			for j in range(self.total_values):
				sum += pow(self.clusters[i].getCentralValue(j)-point.getValue(j),2.0)
			dist = math.sqrt(sum)
			if dist < min_dist :
				min_dist = dist
				id_cluster_center = i
		return id_cluster_center


	def run(self,points):
		f = open('test.txt', 'w')
		if self.K > self.total_points:
			return

		prohibited_indexes=[]
		for i in range(self.K):
			while True:
				index_point = random.randint(0,self.total_points-1)
				if index_point not in prohibited_indexes:
					prohibited_indexes.append(index_point)
					points[index_point].setCluster(i)
					cluster=Cluster(i, points[index_point])
					self.clusters.append(cluster)
					break
		iter = 1

		while True:
			done = True
			for i in range(self.total_points):
				id_old_cluster = points[i].getCluster()
				id_nearest_center = self.getIDNearestCenter(points[i])
				if id_old_cluster != id_nearest_center :
					if id_old_cluster != -1 :
						self.clusters[id_old_cluster].removePoint(points[i].getID())
					points[i].setCluster(id_nearest_center)
					self.clusters[id_nearest_center].addPoint(points[i])
					done = False

			for i in range(self.K):
				for j in range(self.total_values):
					total_points_cluster = self.clusters[i].getTotalPoints()
					sum = 0.0
					if total_points_cluster > 0 :
						for p in range(total_points_cluster):
							sum += self.clusters[i].getPoint(p).getValue(j)
						self.clusters[i].setCentralValue(j, sum / total_points_cluster)

			if done == True or iter >= self.max_iterations :
				print('{} {}'.format('Break in iteration', iter))
				break

			iter=iter+1


		# for i in range(self.K):
		# 	total_points_cluster =  self.clusters[i].getTotalPoints()
		# 	print('{} {}'.format('Cluster', self.clusters[i].getID() + 1))
		# 	for j in range(total_points_cluster):
		# 		print('{} {}'.format('id_layer', self.clusters[i].getPoint(j).getLayer()))
		# 		print('{} {}'.format('id_filter', self.clusters[i].getPoint(j).getFilter()))
		# 		print('{} {}'.format('id_point', self.clusters[i].getPoint(j).getID()))
		# 		for p in range(self.total_values):
		# 			print(self.clusters[i].getPoint(j).getValue(p))

		# 	print('Cluster values: ')
		# 	for j in range(self.total_values):
		# 		print(self.clusters[i].getCentralValue(j))
		t = 0
		fwrite = open('result.txt', 'w')
		for i in range(self.K):
			total_points_cluster =  self.clusters[i].getTotalPoints()
			fwrite.write('{}:{}\n'.format('Cluster', self.clusters[i].getID() + 1))
			fwrite.write('{}:{}\n'.format('total cluster values', total_points_cluster))
			# fwrite.write('Cluster values: ')
			# for j in range(self.total_values):
			# 	fwrite.write('{}'.format(self.clusters[i].getCentralValue(j)))
			for j in range(total_points_cluster):
				fwrite.write('{}:{} '.format('id_layer', self.clusters[i].getPoint(j).getLayer()))
				fwrite.write('{}:{} '.format('id_filter', self.clusters[i].getPoint(j).getFilter()))
				fwrite.write('{}:{}\n'.format('id_channel', self.clusters[i].getPoint(j).getID()))

			if total_points_cluster > 1:
				t = t + 1
		# fwrite.write('total large cluster:')
		# fwrite.write('{}'.format(t))
		fwrite.close()

id_layer = 0
total_points = 0
points=[]
total_values_ori=50
total_values = 1
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
						# elif count % total_values_ori == 0 and count != 0:
						# 	count = count +1
						# 	values.append(id_layer)
						count = count +1
						values.append(net1.params[param_name][0].data[x,y,z,m])
	id_layer = id_layer + 1


K = total_points / 100
max_iterations = 20000

kmeans=KMeans(K, total_points, total_values, max_iterations)
kmeans.run(points)

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
		for x in range(weight.shape[0]):
			for i in range(kmeans.K):
				for j in range(kmeans.clusters[i].getTotalPoints()):
					if layer == kmeans.clusters[i].getPoint(j).getLayer() and x == kmeans.clusters[i].getPoint(j).getFilter() :
						start = kmeans.clusters[i].getPoint(j).getID() * total_values
						for w in range(kmeans.total_values):
							net1.params[param_name][0].data[x,((w+start)/(weight.shape[2]*weight.shape[3]))%weight.shape[1],((w+start)/weight.shape[3])%weight.shape[2],(w+start)%weight.shape[3]] = kmeans.clusters[i].getPoint(j).getValue(w)

	layer = layer + 1

net1.save('new.caffemodel')

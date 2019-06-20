#test file
import random
import math

class Point(object):
	def __init__(self,id_point,values,name=''):
		self.id_point=id_point
		self.total_values=len(values)
		self.values=[]
		for i in range(total_values):
			self.values.append(values[i])

		self.name=name
		self.id_cluster=-1
	
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

	def getName(self):
		return self.name

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


		for i in range(self.K):
			total_points_cluster =  self.clusters[i].getTotalPoints()
			print('{} {}'.format('Cluster', self.clusters[i].getID() + 1))
			for j in range(total_points_cluster):
				print('{} {} {}'.format('Point', self.clusters[i].getPoint(j).getID() + 1, ':'))
				for p in range(self.total_values):
					print(self.clusters[i].getPoint(j).getValue(p))
				point_name = self.clusters[i].getPoint(j).getName()
				if point_name != "":
					print('- ', point_name)

			print('Cluster values: ')
			for j in range(self.total_values):
				print(self.clusters[i].getCentralValue(j))

f = open('clustering.txt', 'r')
data=f.readlines()
total_points,total_values,K,max_iterations,has_name = [int(j) for j in data[0].split()]
points=[]
data=data[1:]
for i in range(len(data)):
	values=[]
	values=[float(j) for j in data[i].split()]
	points.append(Point(i,values,point_name if has_name else ''))

kmeans=KMeans(K, total_points, total_values, max_iterations)
kmeans.run(points)

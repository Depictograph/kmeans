import numpy as np
import scipy.io
import math
import matplotlib.pyplot as plt

class Kmeans:
	def __init__(self, data, k):
		self.data = data
		self.k = k; 
		self.means = []
		self.clusters = []
		self.iterations = 10
	def cluster(self):
		self.initialize()
		for j in range(self.iterations):
			for instance in self.data:
				self.choosecluster(instance)
			for i in range(self.k):
				self.update_mean(i)
			self.clusters = [Cluster() for i in range(self.k)]
		return self.means
	def initialize(self):
		for i in range(self.k):
			self.means.append(np.random.randint(0, high=256, size=784))
			self.clusters.append(Cluster())
	def choosecluster(self, vector):
		distances = []
		for i in range(self.k):
			distances.append(norm_squared(vector, self.means[i]))
		index = np.argmin(distances)
		self.clusters[index].addvector(vector, distances[index])
	def total_cost(self):
		total = 0
		for i in range(self.k):
			total += self.clusters[i].total_dist
		return total 
	def update_mean(self, i):
		if (len(self.clusters[i].vectors) != 0):
			newmean = np.sum(self.clusters[i].vectors, axis=0)
			newmean /= float(len(self.clusters[i].vectors))
			self.means[i] = newmean
		return
def norm_squared(x, u):
	return np.square(np.linalg.norm(x - u, ord=None))
class Cluster:
	def __init__(self):
		self.vectors = []
		self.total_dist = 0
	def addvector(self, vector, cost):
		self.vectors.append(vector)
		self.total_dist += cost


def show_image(kmeans, i):
	image = kmeans.means[i]
	image = image.astype('uint8')
	image = np.reshape(image, [28, 28])
	plt.imshow(image)
	plt.gray()
	plt.show()


train_mat = scipy.io.loadmat('mnist_data/images.mat')
train_data = np.array(train_mat['images'], dtype=np.float32)

train_data = np.reshape(train_data, (784, 60000))
train_data = train_data.transpose()
k = 20

kmeans = Kmeans(train_data, k)
kmeans.cluster()




# train_data -= np.matrix(np.mean(train_data, axis=1)).transpose()
# train_data /= np.matrix(np.std(train_data, axis=1)).transpose()














# if __name__ == '__main__':
# 	main()
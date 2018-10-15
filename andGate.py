#!/usr/bin/python3
import numpy as np
import random

class Dataset():
	def __init__(self, size=10, tt_ratio=0.2, batch_size=1):
		# total number of test+train data
		self.size = size
		# test / train data ratio
		self.tt_ratio = tt_ratio
		# number of test data and labels
		self.testNo = int(size*tt_ratio)
		# denotes current train data => input and label
		self.current_indx = 0
		# current epoch of training
		self.current_epoch = 0
		# number of input data per epoch
		self.batch_size = batch_size
		#
		self.total_batch_per_epoch = size - self.testNo
		self.set_datasets()

	def set_datasets(self):
		self.generate_data()
		self.test_d = self.data[0:self.testNo]
		self.test_l = self.label[0:self.testNo]
		self.train_d = self.data[self.testNo:]
		self.train_l = self.label[self.testNo:]

	@staticmethod
	def getLabel(x1, x2):
		''' AND GATE LOGIC HERE '''
		arr = np.array([0,0])
		if x1 == x2 and x1 == 1:
			arr[1] = 1
			return arr.T
		arr[0] = 1
		return arr.T

	def generate_data(self):
		self.data = np.array(np.random.random([self.size, 2]) >= 0.5, dtype=np.int32)
		self.label= np.array([ self.getLabel(d[0],d[1]) for d in self.data])

	@property
	def get_next(self):
		if self.current_indx+1 >= self.total_batch_per_epoch:
			self.train_d, self.train_l = self.shuffle_data(
								self.train_d, 
								self.train_l
							)
			self.current_indx = 0
			self.current_epoch += 1
		batch_input = self.train_d[self.current_indx:(self.current_indx+self.batch_size)]
		batch_label = self.train_l[self.current_indx:(self.current_indx+self.batch_size)]
		self.current_indx += 1
		return [batch_input, batch_label]

	def shuffle_data(self, data, label):
		combind_data = list(zip(data, label))
		random.shuffle(combind_data)
		_x, _y = list(zip(*combind_data))
		_x = np.array(_x)
		_y = np.array(_y)
		return [_x, _y]

class Network():
	def __init__(self, layers=[2,3,2], size=10):
		'''
			in the argument layers, at index 0, 2 is input size
			so, this is not related to any weights and biases
		'''
		self.biases  = [np.random.randn(y, 1) for y in layers[1:]]
		self.weights = [np.random.randn(y, x)
						for x, y in zip(layers[:-1], layers[1:])]
		self.nabla_b = [np.zeros(b.shape) for b in self.biases]
		self.nabla_w = [np.zeros(w.shape) for w in self.weights]
		self.a = None
		self.z = None
		self.layersNo = len(layers)
		self.layers = layers
		self.dataset = Dataset(size=size, tt_ratio=0.2)
		self.loss = None
		self.losses = []

	@staticmethod
	def sigmoid(z):
		return 1 / (1+np.exp(-z))

	@staticmethod
	def sigmoid_derivative(a):
		return a * (1-a)

	def feedforward(self, a):
		'''
			For backpropagation the input data (i.e. a0, 
			input neurons activations) are needed
			to update weights and biases of the first layer
		'''
		A,Z = [a], [None]		
		for w, b in zip(self.weights, self.biases):
			Z.append(np.dot(w,a) + b)
			a = self.sigmoid(np.dot(w,a) + b)
			A.append(a)
		self.a = A
		self.z = Z

	def calculate_loss(self, label):
		self.loss = np.sum(np.power(np.subtract(self.a[-1], label), 2)) / 2*self.dataset.batch_size
		self.losses.append(self.loss)

	def calculate_deltas(self, y):
		'''
			The calculation of the output layer deltas are
			different than the hidden layer's. So We'll deal 
			with the delta calculations considering that fact.

			delta^L = a^L-1 * der(a^L) * (a^L-y); 			   # for output neurons
			delta^L-1 = a^L-2 * der(a^L-1) * (w^L-1 * delta^L) #for hidden
															   #layer neurons
		'''
		cost_derivative = np.subtract(self.a[-1], y)
		delta = cost_derivative * self.sigmoid_derivative(self.a[-1])
		self.nabla_w[-1] = delta.dot(self.a[-2].T)
		self.nabla_b[-1] = delta
		# nabla for hidden layers
		for i in range(2, self.layersNo):
			delta = np.dot(self.weights[-i+1].T, delta) * self.sigmoid_derivative(self.a[-i])
			self.nabla_b[-i] = delta
			self.nabla_w[-i] = np.dot(delta, self.a[-i-1].T)

	def update_weights_biases(self, learning_rate):
		'''
			new_weight = weight + (-1 * learningRate * nabla_w)
			new_bias = bias + (-1 * learningRate * nabla_b)
		'''
		self.weights = [w-(learning_rate/self.dataset.batch_size)*nw 
						for w, nw in zip(self.weights, self.nabla_w)]
		self.biases = [b-(learning_rate/self.dataset.batch_size)*nb 
						for b, nb in zip(self.biases, self.nabla_b)]

	def backwardprop(self, label, learning_rate=0.01):
		self.calculate_loss(label)
		self.calculate_deltas(label)
		self.update_weights_biases(learning_rate)

	def predict(self, a):
		for w, b in zip(self.weights, self.biases):
			a = self.sigmoid(np.dot(w,a) + b)
		return a

	def train(self, iterationNo=1000):
		for i in range(iterationNo):
			data, label = self.dataset.get_next
			self.feedforward(data.T)
			self.backwardprop(label.T, learning_rate=0.01)
			print("Training %s: Loss = %s"%(
					i, self.loss
				)
			)

	@staticmethod
	def extract_output(a):
		return a.argmax()

	def test(self):
		correct = 0
		index = 1
		for data, label in zip(self.dataset.test_d, self.dataset.test_l):
			data = np.array(np.matrix(data).T)
			label = np.array(np.matrix(label).T)
			prediction = self.extract_output(self.predict(data))
			y = self.extract_output(label)
			if prediction == y:
				correct += 1
			print("Test %s: Expected: %s, Predicted: %s, Accuracy: %s percent"%(
					index, y, prediction, ((correct/index)*100)
				)
			)
			index += 1



n = Network([2,3,2], size=10000)
n.train(iterationNo=50000)
n.test()
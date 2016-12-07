import numpy as np
import struct

def sigmoid(x, d = False):
	return x * (1 - x) if d else 1/(1 + np.exp(-x))

def load_image(fn, size = -1):
	f = open(fn, "rb")
	magic_number = f.read(4)
	if struct.unpack(">i", magic_number) != (0x803,):
		print "Wrong Data!"
		f.close()
		return
	num, = struct.unpack(">i", f.read(4))
	rows, = struct.unpack(">i", f.read(4))
	columns, = struct.unpack(">i", f.read(4))
	num = size if size != -1 else num
	data = np.zeros((num, rows * columns))
	for i in range(0, num):
		for j in range(0, rows * columns):
			data[i][j], = struct.unpack("b", f.read(1))
	means = np.mean(data, axis=0)
	stds = np.std(data, axis=0)
	for i in range(0, num):
		for j in range(0, rows * columns):
			if stds[j] != 0:
				data[i][j] = (data[i][j] - means[j])/stds[j]
			else:
				data[i][j] = 0
	f.close()
	return data

def load_label(fn, one_hot = True, size = -1):
	f = open(fn, "rb")
	magic_number = f.read(4)
	if struct.unpack(">i", magic_number) != (0x801,):
		print "Wrong Data!"
		f.close()
		return
	num, = struct.unpack(">i", f.read(4))
	num = size if size != -1 else num
	if one_hot:
		data = np.zeros((num, 10))
		for i in range(0, num):
			number, = struct.unpack("b", f.read(1))
			data[i][number] = 1
	else:
		data = np.zeros((num, 1))
		for i in range(0, num):
			data[i][0], = struct.unpack("b", f.read(1))		
	f.close()
	return data

class Model:
	def val(self, w1, w2, x, y):
		a1 = np.c_[x, np.ones(x.shape[0])]
		z2 = np.dot(a1, w1)
		a2 = np.c_[sigmoid(z2),np.ones(z2.shape[0])]
		z3 = np.dot(a2, w2)
		a3 = sigmoid(z3)
		right = 0
		wrong = 0
		for i in range(0, a3.shape[0]):
			max = -1
			maxn = -1
			for j in range(0, a3.shape[1]):
				if a3[i][j] > max:
					max = a3[i][j]
					maxn = j
			if maxn == y[i][0]:
				right = right + 1
			else:
				wrong = wrong + 1
		return right * 1.0 / (right + wrong)
	def train(self):
		np.set_printoptions(suppress=True)
		l = 1
		lr = -30
		x = load_image("train/train-images.idx3-ubyte", 2000)
		y = load_label("train/train-labels.idx1-ubyte", True, 2000)
		vx = x[1600:2000,:]
		vy = y[1600:2000,:]
		x = x[0:1600,:]
		y = y[0:1600,:]
		m = x.shape[0]
		for hidden in [50, 100, 200, 300]:
			print "Hidden Layer: ", hidden
			w1 = 2*np.random.random((28*28+1, hidden))-1
			w2 = 2*np.random.random((hidden+1, 10))-1
			for i in range(0, 500):
				# print "Epoch ", i, "\t..."
				# x: [60000, 28*28] -> a1: [60000, 28*28+1]
				z1 = x
				a1 = np.c_[z1, np.ones(x.shape[0])]
				# w1: [28*28+1, 100]
				z2 = np.dot(a1, w1)
				# z2: [60000, 100]
				a2 = np.c_[sigmoid(z2),np.ones(z2.shape[0])]
				# a2: [60000, 100+1]
				z3 = np.dot(a2, w2)
				# w2: [100+1, 10]
				a3 = sigmoid(z3)
				
				# temp = y * np.log(a3) + (1 - y) * np.log(1 - a3);
				# J = - 1 / m * np.sum(temp) + l / 2 / m * ( np.sum(w1[:,0:-1] * w1[:,0:-1]) + np.sum(w2[:,0:-1] * w2[:,0:-1]) )
				
				e3 = a3 - y
				e2 = np.dot(e3, w2.T) * sigmoid(a2, True)
				e1 = np.dot(e2[:,0:-1], w1.T) * sigmoid(a1, True)
				d2 = 1.0/m * np.dot(a2.T, e3)
				d1 = 1.0/m * np.dot(a1.T, e2[:,0:-1])
				w2 = w2 + lr * d2
				w1 = w1 + lr * d1

				
				J = 1.0 / m * np.sum(e3 * e3)
				# print "J: ", J
				
			print "AR: ", self.val(w1, w2, vx, vy)
			self.w1 = w1
			self.w2 = w2

	def test(self):
		#x = load_image("train/train-images.idx3-ubyte", 1000)
		#y = load_label("train/train-labels.idx1-ubyte", False, 1000)
		x = load_image("test/t10k-images.idx3-ubyte", 1000)
		y = load_label("test/t10k-labels.idx1-ubyte", False, 1000)
		a1 = np.c_[x, np.ones(x.shape[0])]
		z2 = np.dot(a1, self.w1)
		a2 = np.c_[sigmoid(z2),np.ones(z2.shape[0])]
		z3 = np.dot(a2, self.w2)
		a3 = sigmoid(z3)
		right = 0
		wrong = 0
		for i in range(0, a3.shape[0]):
			max = -1
			maxn = -1
			for j in range(0, a3.shape[1]):
				if a3[i][j] > max:
					max = a3[i][j]
					maxn = j
			# print "R: ",y[i][0], "W: ", maxn
			if maxn == y[i][0]:
				right = right + 1
			else:
				wrong = wrong + 1
		print right
		print wrong
		print right * 1.0 / (right + wrong)

model = Model();
model.train()
# model.test()

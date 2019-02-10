import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
class NN(gluon.Block):
	def __init__(self, **kwargs):
		super(NN, self).__init__(**kwargs)
		with self.name_scope():
			self.dense0 = gluon.nn.Dense(512)
			self.dense1 = gluon.nn.Dense(128)
			self.dense2 = gluon.nn.Dense(64)
			self.dense3 = gluon.nn.Dense(32)
			self.dense4 = gluon.nn.Dense(16)
			self.dense5 = gluon.nn.Dense(10)

	def forward(self, x):
		x = nd.relu(self.dense0(x))
		x = nd.relu(self.dense1(x))
		x = nd.relu(self.dense2(x))
		x = nd.relu(self.dense3(x))
		x = nd.relu(self.dense4(x))

		x = self.dense5(x)
		return x
	def backward(self):
		pass

class NN2(gluon.Block):
	def __init__(self, **kwargs):
		super(NN2, self).__init__(**kwargs)
		with self.name_scope():
			self.dense0 = gluon.nn.Dense(1024)
			self.dense1 = gluon.nn.Dense(512)
			self.dense2 = gluon.nn.Dense(256)
			self.dense3 = gluon.nn.Dense(10)

	def forward(self, x):
		x = nd.relu(self.dense0(x))
		x = nd.relu(self.dense1(x))
		x = nd.relu(self.dense2(x))
		
		x = self.dense3(x)
		return x
	def backward(self):
		pass

class NNBN(gluon.Block):
	def __init__(self, **kwargs):
		super(NNBN, self).__init__(**kwargs)
		with self.name_scope():
			self.dense0 = gluon.nn.Dense(1024)
			self.dense1 = gluon.nn.Dense(512)
			self.dense2 = gluon.nn.Dense(256)
			self.dense3 = gluon.nn.Dense(10)
			self.bn0 = gluon.nn.BatchNorm(axis=1,scale = False)
			self.bn1 = gluon.nn.BatchNorm(axis=1,scale = False)
			self.bn2 = gluon.nn.BatchNorm(axis=1,scale = True)

	def forward(self, x):
		x = nd.relu(self.dense0(x))
		x = self.bn0(x);
		x = nd.relu(self.dense1(x))
		x = self.bn1(x);
		x = nd.relu(self.dense2(x))
		x = self.bn2(x);
		x = self.dense3(x)
		return x
	def backward(self):
		pass

class NNdrop(gluon.Block):
	def __init__(self,drop, **kwargs):
		super(NNdrop, self).__init__(**kwargs)
		with self.name_scope():
			self.dense0 = gluon.nn.Dense(1024)
			self.dense1 = gluon.nn.Dense(512)
			self.dense2 = gluon.nn.Dense(256)
			self.dense3 = gluon.nn.Dense(10)
			self.drop0 = gluon.nn.Dropout(rate=drop)
			self.drop1 = gluon.nn.Dropout(rate=drop)
			self.drop2 = gluon.nn.Dropout(rate=drop)

	def forward(self, x):
		x = nd.relu(self.dense0(x))
		x = self.drop0(x);
		x = nd.relu(self.dense1(x))
		x = self.drop1(x);
		x = nd.relu(self.dense2(x))
		x = self.drop2(x);
		x = self.dense3(x)
		return x
	def backward(self):
		pass
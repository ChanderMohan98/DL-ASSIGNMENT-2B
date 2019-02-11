import numpy as np
import data_loader
import module
import mxnet as mx
from mxnet import nd, autograd, gluon
import matplotlib.pyplot as plt
from sklearn import model_selection
import sys

'''Implement mini-batch SGD here'''
ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
data_ctx = ctx
model_ctx = ctx

batch_size = 64
data_l = data_loader.DataLoader()


imgs, labels = data_l.load_data(mode = 'train')
train_imgs, val_imgs, train_labels, val_labels = model_selection.train_test_split(imgs, labels, test_size = 0.3, random_state = 1, stratify = labels)
train_set = mx.gluon.data.dataset.ArrayDataset(train_imgs, train_labels)
val_set = mx.gluon.data.dataset.ArrayDataset(val_imgs, val_labels)
train_data = mx.gluon.data.DataLoader(train_set,batch_size, shuffle=True)
val_data = mx.gluon.data.DataLoader(val_set,batch_size, shuffle=True)


imgs, labels = data_l.load_data(mode = 'test')
test_set = mx.gluon.data.dataset.ArrayDataset(imgs, labels)
test_data = mx.gluon.data.DataLoader(test_set,
									 batch_size, shuffle=False)
def part1(wt):
	net = module.NN2()
	net.collect_params().initialize(wt, ctx=model_ctx)
	print('\nNetwork 2\n')
		# net.collect_params().initialize(mx.init.Normal(sigma=.01), ctx=model_ctx)
	softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
	trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .01})

	epochs = 5
	# smoothing_constant = .01
	patience = 5

	train_loss = []
	val_loss = []

	for e in range(epochs):
		cumulative_train_loss = 0
		cumulative_val_loss = 0
		train_batch_count = 0
		val_batch_count = 0
		for i, (data, label) in enumerate(train_data):
			data = data.as_in_context(model_ctx).reshape((-1, 784))
			label = label.as_in_context(model_ctx)
			with autograd.record():
				output = net(data.astype('float32'))
				loss = softmax_cross_entropy(output, label.astype('float32'))
			loss.backward()
			trainer.step(data.shape[0])
			cumulative_train_loss += nd.sum(loss).asscalar()
			train_batch_count += data.shape[0]
		for i, (data, label) in enumerate(val_data):
			data = data.as_in_context(model_ctx).reshape((-1, 784))
			label = label.as_in_context(model_ctx)
			output = net(data.astype('float32'))
			loss = softmax_cross_entropy(output, label.astype('float32'))
			cumulative_val_loss += nd.sum(loss).asscalar()	    	
			val_batch_count += data.shape[0]

		train_loss.append(cumulative_train_loss/train_batch_count)
		val_loss.append(cumulative_val_loss/val_batch_count)

		print("Epoch %s. Train Loss: %s, Validation Loss %s" %
			  (e, cumulative_train_loss/train_batch_count, cumulative_val_loss/val_batch_count))

		if e > 0 and val_loss[e] < np.min([val_loss[ep] for ep in np.arange(0,e).tolist()]):
			print('Validation Loss reduced, saving weights....')
			net.save_parameters('../weights/best_model_'+ str(wt) +'.params')
		if e + 1 > patience and np.sum(np.asarray([val_loss[ep + 1] - val_loss[ep] for ep in np.arange(e - patience,e).tolist()]) > 0) == patience: #Stopping criterion
			break
	net.load_parameters('../weights/best_model_'+ str(wt) +'.params')
	acc = mx.metric.Accuracy()
	for i, (data, label) in enumerate(test_data):
		data = data.as_in_context(model_ctx).reshape((-1, 784))
		label = label.as_in_context(model_ctx)
		output = net(data.astype('float32'))
		predictions = nd.argmax(output, axis=1)
		acc.update(preds=predictions, labels=label)
	
	test_accuracy = acc.get()[1]
	print('Test accuracy is ' + str(100 * test_accuracy) + '%.')
	return train_loss
wt_list = [mx.init.Normal(),mx.init.Xavier(magnitude = 0.5),mx.init.Orthogonal(scale = 0.5)] 
loss_list = list()
for i in wt_list:
	loss_list.append(part1(i))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

length_list = list()
for i in range(3):
	length_list.append(len(loss_list[i]))
length  = np.min(length_list)
color_list = ['r','g','b']
line_tuple = ()
for i in range(len(loss_list)):
	line_tuple += ax.plot(length, loss_list[i][0:length], color = color_list[i], lw=2)

ax.legend(line_tuple, ('Normal', 'Xavier', 'Orthogonal'))
plt.rcParams.update({'font.size': 12})
plt.title('Loss vs epochs for diff. wt. init.')
plt.rcParams.update({'font.size': 10})
plt.xlabel('Epoch')
plt.rcParams.update({'font.size': 10})
plt.ylabel('Loss')
plt.xticks([item for item in (np.arange(length) + 1).tolist() if item % 5 == 0])
plt.savefig('part1.png')
# plt.show()


def part2(net):
	net.collect_params().initialize(mx.init.Normal(), ctx=model_ctx)
	print('\nNetwork 2\n')
		# net.collect_params().initialize(mx.init.Normal(sigma=.01), ctx=model_ctx)
	softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
	trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .01})

	epochs = 5
	# smoothing_constant = .01
	patience = 5

	train_loss = []
	val_loss = []

	for e in range(epochs):
		cumulative_train_loss = 0
		cumulative_val_loss = 0
		train_batch_count = 0
		val_batch_count = 0
		for i, (data, label) in enumerate(train_data):
			data = data.as_in_context(model_ctx).reshape((-1, 784))
			label = label.as_in_context(model_ctx)
			with autograd.record():
				output = net(data.astype('float32'))
				loss = softmax_cross_entropy(output, label.astype('float32'))
			loss.backward()
			trainer.step(data.shape[0])
			cumulative_train_loss += nd.sum(loss).asscalar()
			train_batch_count += data.shape[0]
		for i, (data, label) in enumerate(val_data):
			data = data.as_in_context(model_ctx).reshape((-1, 784))
			label = label.as_in_context(model_ctx)
			output = net(data.astype('float32'))
			loss = softmax_cross_entropy(output, label.astype('float32'))
			cumulative_val_loss += nd.sum(loss).asscalar()	    	
			val_batch_count += data.shape[0]

		train_loss.append(cumulative_train_loss/train_batch_count)
		val_loss.append(cumulative_val_loss/val_batch_count)

		print("Epoch %s. Train Loss: %s, Validation Loss %s" %
			  (e, cumulative_train_loss/train_batch_count, cumulative_val_loss/val_batch_count))

		if e > 0 and val_loss[e] < np.min([val_loss[ep] for ep in np.arange(0,e).tolist()]):
			print('Validation Loss reduced, saving weights....')
			net.save_parameters('../weights/best_model_bn.params')
		if e + 1 > patience and np.sum(np.asarray([val_loss[ep + 1] - val_loss[ep] for ep in np.arange(e - patience,e).tolist()]) > 0) == patience: #Stopping criterion
			break
	net.load_parameters('../weights/best_model_bn.params')
	acc = mx.metric.Accuracy()
	for i, (data, label) in enumerate(test_data):
		data = data.as_in_context(model_ctx).reshape((-1, 784))
		label = label.as_in_context(model_ctx)
		output = net(data.astype('float32'))
		predictions = nd.argmax(output, axis=1)
		acc.update(preds=predictions, labels=label)
	
	test_accuracy = acc.get()[1]
	print('Test accuracy is ' + str(100 * test_accuracy) + '%.')
	return train_loss

wt_list = [module.NN2(),module.NNBN()] 
loss_list = list()
for i in wt_list:
	loss_list.append(part2(i))
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

length_list = list()
for i in range(2):
	length_list.append(len(loss_list[i]))
length  = np.min(length_list)
color_list = ['r','g']
line_tuple = ()
for i in range(len(loss_list)):
	line_tuple += ax.plot(length, loss_list[i][0:length], color = color_list[i], lw=2)

ax.legend(line_tuple, ('WithoutBatchNorm', 'WithBatchNorm'))
plt.rcParams.update({'font.size': 12})
plt.title('Loss vs epochs for diff. wt. init.')
plt.rcParams.update({'font.size': 10})
plt.xlabel('Epoch')
plt.rcParams.update({'font.size': 10})
plt.ylabel('Loss')
plt.xticks([item for item in (np.arange(length) + 1).tolist() if item % 5 == 0])
plt.savefig('part2.png')
# plt.show()

def part3(drop):
	net = module.NNdrop(drop = drop)
	net.collect_params().initialize(mx.init.Normal(), ctx=model_ctx)
	print('\nNetwork 2\n')
		# net.collect_params().initialize(mx.init.Normal(sigma=.01), ctx=model_ctx)
	softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
	trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .01})

	epochs = 5
	# smoothing_constant = .01
	patience = 5

	train_loss = []
	val_loss = []

	for e in range(epochs):
		cumulative_train_loss = 0
		cumulative_val_loss = 0
		train_batch_count = 0
		val_batch_count = 0
		for i, (data, label) in enumerate(train_data):
			data = data.as_in_context(model_ctx).reshape((-1, 784))
			label = label.as_in_context(model_ctx)
			with autograd.record():
				output = net(data.astype('float32'))
				loss = softmax_cross_entropy(output, label.astype('float32'))
			loss.backward()
			trainer.step(data.shape[0])
			cumulative_train_loss += nd.sum(loss).asscalar()
			train_batch_count += data.shape[0]
		for i, (data, label) in enumerate(val_data):
			data = data.as_in_context(model_ctx).reshape((-1, 784))
			label = label.as_in_context(model_ctx)
			output = net(data.astype('float32'))
			loss = softmax_cross_entropy(output, label.astype('float32'))
			cumulative_val_loss += nd.sum(loss).asscalar()	    	
			val_batch_count += data.shape[0]

		train_loss.append(cumulative_train_loss/train_batch_count)
		val_loss.append(cumulative_val_loss/val_batch_count)

		print("Epoch %s. Train Loss: %s, Validation Loss %s" %
			  (e, cumulative_train_loss/train_batch_count, cumulative_val_loss/val_batch_count))

		if e > 0 and val_loss[e] < np.min([val_loss[ep] for ep in np.arange(0,e).tolist()]):
			print('Validation Loss reduced, saving weights....')
			net.save_parameters('../weights/best_model'+str(drop)+'.params')
		if e + 1 > patience and np.sum(np.asarray([val_loss[ep + 1] - val_loss[ep] for ep in np.arange(e - patience,e).tolist()]) > 0) == patience: #Stopping criterion
			break
	net.load_parameters('../weights/best_model'+str(drop)+'.params')
	acc = mx.metric.Accuracy()
	for i, (data, label) in enumerate(test_data):
		data = data.as_in_context(model_ctx).reshape((-1, 784))
		label = label.as_in_context(model_ctx)
		output = net(data.astype('float32'))
		predictions = nd.argmax(output, axis=1)
		acc.update(preds=predictions, labels=label)
	
	test_accuracy = acc.get()[1]
	print('Test accuracy is ' + str(100 * test_accuracy) + '%.')
	return train_loss

wt_list = [0,0.1,0.4,0.6] 
loss_list = list()
for i in wt_list:
	loss_list.append(part3(i))
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

length_list = list()
for i in range(4):
	length_list.append(len(loss_list[i]))
length  = np.min(length_list)
color_list = ['r','g','b','y']
line_tuple = ()
for i in range(len(loss_list)):
	line_tuple += ax.plot(length, loss_list[i][0:length], color = color_list[i], lw=2)
ax.legend(line_tuple, ('dropout0', 'dropout0.1','dropout0.4','dropout0.6'))
plt.rcParams.update({'font.size': 12})
plt.title('Loss vs epochs for diff. dropouts ')
plt.rcParams.update({'font.size': 10})
plt.xlabel('Epoch')
plt.rcParams.update({'font.size': 10})
plt.ylabel('Loss')
plt.xticks([item for item in (np.arange(length) + 1).tolist() if item % 5 == 0])
plt.savefig('part3.png')
# plt.show()

def part4(opt):
	net = module.NN2()
	net.collect_params().initialize(mx.init.Normal(), ctx=model_ctx)
	print('\nNetwork 2\n')
		# net.collect_params().initiali(mx.init.Normal(sigma=.01), ctx=model_ctx)
	softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
	trainer = gluon.Trainer(net.collect_params(), opt, {'learning_rate': .01})

	epochs = 5
	# smoothing_constant = .01
	patience = 5

	train_loss = []
	val_loss = []

	for e in range(epochs):
		cumulative_train_loss = 0
		cumulative_val_loss = 0
		train_batch_count = 0
		val_batch_count = 0
		for i, (data, label) in enumerate(train_data):
			data = data.as_in_context(model_ctx).reshape((-1, 784))
			label = label.as_in_context(model_ctx)
			with autograd.record():
				output = net(data.astype('float32'))
				loss = softmax_cross_entropy(output, label.astype('float32'))
			loss.backward()
			trainer.step(data.shape[0])
			cumulative_train_loss += nd.sum(loss).asscalar()
			train_batch_count += data.shape[0]
		for i, (data, label) in enumerate(val_data):
			data = data.as_in_context(model_ctx).reshape((-1, 784))
			label = label.as_in_context(model_ctx)
			output = net(data.astype('float32'))
			loss = softmax_cross_entropy(output, label.astype('float32'))
			cumulative_val_loss += nd.sum(loss).asscalar()	    	
			val_batch_count += data.shape[0]

		train_loss.append(cumulative_train_loss/train_batch_count)
		val_loss.append(cumulative_val_loss/val_batch_count)

		print("Epoch %s. Train Loss: %s, Validation Loss %s" %
			  (e, cumulative_train_loss/train_batch_count, cumulative_val_loss/val_batch_count))

		if e > 0 and val_loss[e] < np.min([val_loss[ep] for ep in np.arange(0,e).tolist()]):
			print('Validation Loss reduced, saving weights....')
			net.save_parameters('../weights/best_model_'+ str(opt) +'.params')
		if e + 1 > patience and np.sum(np.asarray([val_loss[ep + 1] - val_loss[ep] for ep in np.arange(e - patience,e).tolist()]) > 0) == patience: #Stopping criterion
			break
	net.load_parameters('../weights/best_model_'+ str(opt) +'.params')
	acc = mx.metric.Accuracy()
	for i, (data, label) in enumerate(test_data):
		data = data.as_in_context(model_ctx).reshape((-1, 784))
		label = label.as_in_context(model_ctx)
		output = net(data.astype('float32'))
		predictions = nd.argmax(output, axis=1)
		acc.update(preds=predictions, labels=label)
	
	test_accuracy = acc.get()[1]
	print('Test accuracy is ' + str(100 * test_accuracy) + '%.')
	return train_loss


wt_list = ['sgd','nag','adadelta','adagrad','rmsprop','adam'] 
loss_list = list()
for i in wt_list:
	loss_list.append(part1(i))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

length_list = list()
for i in range(6):
	length_list.append(len(loss_list[i]))
length  = np.min(length_list)
color_list = ['r','g','b','y','black','brown']
line_tuple = ()
for i in range(len(loss_list)):
	line_tuple += ax.plot(length, loss_list[i][0:length], color = color_list[i], lw=2)

ax.legend(line_tuple, ('sgd','nag','adadelta','adagrad','rmsprop','adam'))
plt.rcParams.update({'font.size': 12})
plt.title('Loss vs epochs for diff. opt')
plt.rcParams.update({'font.size': 10})
plt.xlabel('Epoch')
plt.rcParams.update({'font.size': 10})
plt.ylabel('Loss')
plt.xticks([item for item in (np.arange(length) + 1).tolist() if item % 5 == 0])
plt.savefig('part4.png')


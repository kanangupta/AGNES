## Plots the metrics obtained by running main.py
## The checkpoint to be loaded needs to be specified 

import torch
import matplotlib.pyplot as plt
import numpy as np

data = {}
runs = 5 #number of times this experiment was repeated
train_size = 60000 #size of training set
batch = 50 #batch size
epochs = 20
epoch_step = train_size/batch
total_steps = epoch_step*epochs
title="MNIST, LeNet5 - "

names = [
	'NAG, η=.001',
	'ADAM',
	'SGD, m=.99',
	'SGD, m=.9',
	'SGD',
	'AGNES, η=.1',
	'AGNES, η=.01',
#	'AGNES, η=.005',
	]

colors = {
	'Nesterov, η=.001':'tab:orange',
	'ADAM':'tab:red',
	'SGD, m=.99':'tab:brown',
	'SGD, m=.9':'tab:pink',
	'SGD':'tab:gray',
	'AGNES, η=.1':'tab:green',
	'AGNES, η=.01':'tab:blue',
#	'AGNES, η=.005':'xkcd:yellow green',
	}

metrics = ['Test Accuracy', 'Test Loss', 'Training Loss']

decay = 0.999

for name in names:
	data[name] = {'Test Loss':[], 'Training Loss':[], 'Test Accuracy':[]}
	for i in range(runs):
		with open(name+'/'+str(i)+'/checkpoint_20.pth', 'rb') as file:
			temp = torch.load(file, map_location=torch.device('cpu'))
			data[name]['Test Loss'].append(temp['test_losses'])
			data[name]['Test Accuracy'].append(temp['test_accuracies'])
			running_averages = [temp['train_losses'][0]]
			for num in temp['train_losses']:
				running_averages.append(decay*running_averages[-1] + (1-decay)*num)
			data[name]['Training Loss'].append(running_averages)


for name in names:
	data[name]['Test Accuracy'] = 100*np.array(data[name]['Test Accuracy'])
	for metric in metrics[1:]:
		data[name][metric] = np.array(data[name][metric])

metric=metrics[0]
plt.figure()
for name in names:
	mean = np.mean(data[name][metric], axis = 0)
	std = np.std(data[name][metric] , axis = 0)

	plt.plot(np.arange(0,total_steps+1,epoch_step), mean, label = name, color = colors[name])
	plt.fill_between(np.arange(0,total_steps+1,epoch_step), mean+std, mean-std, alpha = 0.2, color = colors[name])

plt.ylim([95.5,99.5]) #window for test accuracy
plt.title(title+metric)
#plt.legend()
plt.savefig(title+metric)


metric=metrics[1]
plt.figure()
for name in names:
	mean = np.mean(data[name][metric], axis = 0)
	std = np.std(data[name][metric] , axis = 0)

	plt.semilogy(np.arange(0,total_steps+1,epoch_step), mean, label = name, color = colors[name])
	plt.fill_between(np.arange(0,total_steps+1,epoch_step), mean+std, mean-std, alpha = 0.2, color = colors[name])

plt.title(title+metric)
#plt.legend()
plt.savefig(title+metric)

metric=metrics[2]
plt.figure()
for name in names:
	mean = np.mean(data[name][metric], axis = 0)
	std = np.std(data[name][metric] , axis = 0)

	plt.semilogy(mean, label = name, color = colors[name])
	plt.fill_between(mean+std, mean-std, alpha = 0.2, color = colors[name])

plt.title(title+metric)
plt.legend()
plt.savefig(title+metric)
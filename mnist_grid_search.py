import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToTensor
from AGNES import AGNES
import pickle
import matplotlib.pyplot as plt

#downloading the data
train_data = torchvision.datasets.MNIST(root = 'data', train = True, download=True, transform = ToTensor())
test_data = torchvision.datasets.MNIST(root = 'data', train = False, transform = ToTensor())

train_loader = torch.utils.data.DataLoader(train_data, batch_size = 60, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = 10, shuffle = True)


#defining data loader, optimizer, and the loss function
etas = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
alphas = [1e-1, 1e-2, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5]
optim_names = [(i,j) for i in etas for j in alphas]

models = {name: nn.Sequential(
	nn.Conv2d(1,6,5,padding=2),
	nn.Tanh(),
	nn.AvgPool2d(2,2),
	nn.Conv2d(6,16,5),
	nn.Tanh(),
	nn.AvgPool2d(2,2),
	nn.Flatten(),
	nn.Linear(5*5*16,120),
	nn.Tanh(),
	nn.Linear(120,84),
	nn.Tanh(),
	nn.Linear(84,10)
	) for name in optim_names}

optimizers={name: AGNES(models[name].parameters(), lr = name[0], correction = name[1], momentum=0.99) for name in optim_names}

gpu_available= torch.cuda.is_available()
print("Computing on GPU." if gpu_available else "No GPU available.")
device = torch.device("cuda:0" if gpu_available else "cpu")

for name in optim_names:
	models[name] = models[name].to(device)
	models[name].train()

loss_fn = nn.CrossEntropyLoss() #Note: If you use this loss function, do not use a softmax layer at the end of the neural network

no_of_epochs = 6
losses = {name:[] for name in optim_names}
for name in optim_names:
	for epoch in range(no_of_epochs):
		print("Currently on epoch",epoch,"for",name)
		avgloss = 0
		for datum, label in train_loader:
			datum = datum.to(device)
			label = label.to(device)
			optimizers[name].zero_grad()
			output = torch.squeeze(models[name](datum))
			loss=loss_fn(output, label)
			loss.backward()
			optimizers[name].step()
			avgloss += loss.item()
		losses[name].append(avgloss/len(train_loader))

with open('losses_from_grid_search', 'wb') as file:
	pickle.dump(losses, file)

# for name in optim_names:
# 	plt.semilogy(losses[name], label = name)
# plt.legend()
# plt.savefig('grid_search_plot')
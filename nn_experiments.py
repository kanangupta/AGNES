import torch.nn as nn
from trainer import Trainer
import resnet9model
## Before executing this code, don't forget to modify to update trainer.py
## to ensure that it loads and trains on the appropriate dataset.
## The code can be used with any neural network model; just ensure compatibility between the model and the dataset.

num_runs = 1 #number of times the experiment is repeated (for reporting average performance)
for run in range(num_runs):

    opt_names = {
        'AGNES, η=.01':'AGNES(self.net.parameters(), lr={} , momentum={} , correction={}, weight_decay={})'.format(1e-3, 0.99, 0.01, 1e-5),
        'AGNES, η=0.1':'AGNES(self.net.parameters(), lr={} , momentum={} , correction={}, weight_decay={})'.format(1e-3, 0.99, 0.1, 1e-5),
        'NAG, η=.001':'AGNES(self.net.parameters(), lr={} , momentum={} , correction={}, weight_decay={})'.format(1e-3, 0.99, 1e-3, 1e-5),
        'ADAM': 'torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-5)',
        'SGD, m=.99': 'torch.optim.SGD(self.net.parameters(), lr=1e-3, momentum=0.99, weight_decay=1e-5)',
        'SGD, m=.9': 'torch.optim.SGD(self.net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)',
        'SGD': 'torch.optim.SGD(self.net.parameters(), lr=1e-3, momentum=0, weight_decay=1e-5)'
    }

    for key, opt_name in opt_names.items():
        # model = nn.Sequential(                #LeNet-5
        #     nn.Conv2d(1,6,5,padding=2),
        #     nn.Tanh(),
        #     nn.AvgPool2d(2,2),
        #     nn.Conv2d(6,16,5),
        #     nn.Tanh(),
        #     nn.AvgPool2d(2,2),
        #     nn.Flatten(),
        #     nn.Linear(5*5*16,120),
        #     nn.Tanh(),
        #     nn.Linear(120,84),
        #     nn.Tanh(),
        #     nn.Linear(84,10)
        #     )

        # model = resnet9model.Net()

        # model = models.resnet50(pretrained=True)
        # model.fc = nn.Linear(model.fc.in_features, 100, bias=True)


        model = models.densenet201()
        model.classifier = nn.Linear(model.classifier.in_features, 100, bias=True)


        net = Trainer(model = model, opt_name = opt_name)
        net.train(save_dir = 'cifar100-densenet201'+key+'/'+str(run), batch_size=50, num_epochs = 100, schedule_lr_epochs=50, lr_factor=0.1, manual_seed = (num_runs==1))
        #we don't use a manual seed if the experiment is being repeated
        #the learning rates is multiplied by a factor of lr_factor after schedule_lr_epochs epochs

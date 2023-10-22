# Based on Matthias Wright's cifar10-resnet project licensed under the MIT Licence
# https://github.com/matthias-wright/cifar10-resnet

## The current version of the file is set up to load and train on CIFAR-100. 
## The corresponding lines for CIFAR-10 and MNIST have been commented out.
## This code can be used for other datasets by appropriately modifying the code defining
## train_dataset, train_transform, test_dataset, and test_transform.

import torch
import torchvision
from torchvision import datasets, models, transforms
import os
import util
from AGNES import AGNES


class Trainer:

    def __init__(self, model, opt_name):
        # opt_name : str
        #    The optimizer is passed as a string

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_cuda = torch.cuda.is_available()
        self.net = model.cuda() if self.use_cuda else model()
        self.optimizer = None
        self.train_accuracies = []
        self.train_losses =[]
        self.test_accuracies = []
        self.test_losses = []
        self.start_epoch = 0
        exec('self.optimizer ='+opt_name)

    def train(self, save_dir, num_epochs=100, batch_size=50, schedule_lr_epochs=0, lr_factor=1, test_each_epoch=True, verbose=False, manual_seed=False):
        """Trains the network.

        Parameters
        ----------
        save_dir : str
            The directory in which the parameters will be saved
        opt_name : str
            The name of the optimizer, can be one 'AGNES', 'ADAM', 'SGD0.99M', 'SGD', or 'SGD0.9M'
        num_epochs : int
            The number of epochs
        batch_size : int
            The batch size
        learning_rate : float
            The learning rate
        test_each_epoch : boolean
            True: Test the network after every training epoch, False: no testing
        verbose : boolean
            True: Print training progress to console, False: silent mode
        """

        if manual_seed:
            torch.manual_seed(0)

        ### For CIFAR-10 change the last transform to transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ### For CIFAR-100 change the last transform to transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762))
        ### For MNIST, remove all transforms except transforms.ToTensor()
        ### Don't forget to make the same changes to test_transform as well
        train_transform = transforms.Compose([
            util.Cutout(num_cutouts=2, size=8, p=0.8),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])

        ### Only one of the following lines should be uncommented. Don't forget to use the same dataset
        ### for test_dataset as well.
        train_dataset = datasets.CIFAR100('data/cifar', train=True, download=True, transform=train_transform)
        #train_dataset = datasets.MNIST('data/mnist', train=True, download=True, transform=train_transform)
        #train_dataset = datasets.CIFAR10('data/cifar', train=True, download=True, transform=train_transform)

        data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        criterion = torch.nn.CrossEntropyLoss().cuda() if self.use_cuda else torch.nn.CrossEntropyLoss()

        progress_bar = util.ProgressBar()

        if self.start_epoch==0: #computing test loss and accuracy before the training starts
            test_loss, test_accuracy = self.test(batch_size=batch_size)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_accuracy)

        for epoch in range(self.start_epoch + 1, num_epochs + 1):
            print('Epoch {}/{}'.format(epoch, num_epochs))

            if schedule_lr_epochs:
                if epoch%schedule_lr_epochs == 0:
                #update the learning rate every schedule_lr_epochs epochs
                    for g in self.optimizer.param_groups:
                        g['lr'] *= lr_factor #updating the learning rate
                        if 'correction' in g.keys():
                            g['correction'] *= lr_factor #updating the correction step for AGNES

            #epoch_correct = 0
            #epoch_total = 0
            for i, data in enumerate(data_loader, 1):
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.net.forward(images)
                loss = criterion(outputs, labels.squeeze_())
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs.data, dim=1)
                batch_total = labels.size(0)
                batch_correct = (predicted == labels.flatten()).sum().item()

                self.train_accuracies.append(batch_correct/batch_total)
                self.train_losses.append(loss.item())

                #epoch_total += batch_total
                #epoch_correct += batch_correct

                if verbose:
                    # Update progress bar in console
                    info_str = 'Last batch accuracy: {:.4f} - Running epoch accuracy {:.4f}'.\
                                format(batch_correct / batch_total)
                    progress_bar.update(max_value=len(data_loader), current_value=i, info=info_str)

            #self.train_accuracies.append(epoch_correct / epoch_total)
            if verbose:
                progress_bar.new_line()

            if test_each_epoch:
                test_loss, test_accuracy = self.test()
                self.test_losses.append(test_loss)
                self.test_accuracies.append(test_accuracy)
                if verbose:
                    print('Test accuracy: {}'.format(test_accuracy))

            # Save parameters after every 10 epochs
            if epoch%10==0:
                self.save_parameters(epoch, directory=save_dir)

    def test(self, batch_size=250):
        """Tests the network.

        """
        self.net.eval()

        ### For CIFAR-10 change the last transform to transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ### For CIFAR-100 change the last transform to transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762))
        ### For MNIST, remove all transforms except transforms.ToTensor()
        test_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
                                             ])

        ### Only one of the following lines should be uncommented. Ideally, the same dataset that's used for training.
        test_dataset = datasets.CIFAR100('data/cifar', train=False, download=True, transform=test_transform)
        #test_dataset = datasets.MNIST('data/mnist', train=False, download=True, transform=test_transform)
        #test_dataset = datasets.CIFAR10('data/cifar', train=False, download=True, transform=test_transform)

        data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        criterion = torch.nn.CrossEntropyLoss().cuda() if self.use_cuda else torch.nn.CrossEntropyLoss()

        correct = 0
        total = 0
        loss = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader, 0):
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.net(images)

                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels.flatten()).sum().item()

                loss += criterion(outputs, labels.squeeze_()).item() #loss is normalized by default, but the last batch here would be of a different size

        self.net.train()
        return (loss / (i+1), correct / total)

    def save_parameters(self, epoch, directory):
        """Saves the parameters of the network to the specified directory.

        Parameters
        ----------
        epoch : int
            The current epoch
        directory : str
            The directory to which the parameters will be saved
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save({
            # 'opt_name': self.opt_name,
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_accuracies': self.train_accuracies,
            'train_losses': self.train_losses,
            'test_accuracies': self.test_accuracies,
            'test_losses': self.test_losses
        }, os.path.join(directory, 'checkpoint_' + str(epoch) + '.pth'))

    def load_parameters(self, path):
        """Loads the given set of parameters.

        Parameters
        ----------
        path : str
            The file path pointing to the file containing the parameters
        """
        checkpoint = torch.load(path, map_location=self.device)

        # self.opt_name = checkpoint['opt_name']

        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_accuracies = checkpoint['train_accuracies']
        self.train_losses = checkpoint['train_losses']
        self.test_accuracies = checkpoint['test_accuracies']
        self.test_losses = checkpoint['test_losses']
        self.start_epoch = checkpoint['epoch']

